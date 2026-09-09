"""
This module implements the assignment update rule for the SOGA semantics.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from antlr4 import CommonTokenStream, InputStream, ParseTreeWalker

from pydegas.exceptions import SemanticError, SyntaxParseError
from pydegas.mixtures.distribution import Dist, extend_dist
from pydegas.mixtures.gaussian_mix import GaussianMix
from pydegas.parse.asgmt import ASGMTLexer, ASGMTListener, ASGMTParser
from pydegas.parse.values import unpack_gm_list


ASGMT_TREE_CACHE: dict[str, Any] = {}


logger = logging.getLogger(__name__)


class AsgmtRule(ASGMTListener):
    """ANTLR listener that walks the ASGMT parse tree of an assignment RHS."""

    def __init__(
        self,
        variables: list[str],
        data: dict[str, Any],
        parameters: dict[str, torch.Tensor],
    ) -> None:
        # variables, data and parameters
        self.variables = variables
        self.data = data
        self.parameters = parameters

        # parameters of the assignment
        self.target_index: int | None = None  # stores the index of the target variable
        self.is_product: int | None = None  # checks whether a term is a product of two vars

        # additional random variables (cannot use a tensor here because different a.r.v.s can have different numbers of components)
        self.aux_pis: list[torch.Tensor] = []  # stores the weights of auxiliary variables
        self.aux_means: list[torch.Tensor] = []  # stores the means of auxiliary variables
        self.aux_covs: list[torch.Tensor] = []  # stores the cov matrices of auxiliary variables

        # function to be applied
        self.update_func: Callable[[Dist], Dist] | None = None  # stores the function

        # Internal accumulation fields (set during enterAdd)
        self.linear_coefficients: torch.Tensor | None = None
        self.linear_constant: torch.Tensor | None = None
        self.product_indices: list[int] | None = None
        self.product_coefficient = torch.tensor(1.0)

    def _data_index(self, ctx: ASGMTParser.IddContext) -> int:
        if ctx.NUM() is not None:
            return int(ctx.NUM().getText())
        return int(self.data[ctx.IDV(1).getText()][0].item())

    def _variable_name(self, ctx: ASGMTParser.SymvarsContext) -> str:
        if ctx.idd() is None:
            return ctx.getText()
        indexed = ctx.idd()
        return f"{indexed.IDV(0).getText()}[{self._data_index(indexed)}]"

    def _is_variable(self, term: ASGMTParser.TermContext) -> bool:
        if term.gm() is not None:
            return True
        if term.symvars() is None:
            return False
        indexed = term.symvars().idd()
        return indexed is None or indexed.IDV(0).getText() not in self.data

    def _constant_value(self, term: ASGMTParser.TermContext) -> torch.Tensor:
        if term.NUM() is not None:
            return torch.tensor(float(term.NUM().getText()))
        if term.par() is not None:
            return self.parameters[term.par().IDV().getText()]
        indexed = term.symvars().idd()
        return torch.as_tensor(self.data[indexed.IDV(0).getText()][self._data_index(indexed)])

    def collect_gm_term(self, term: ASGMTParser.TermContext) -> None:
        """Append the weights, means, and variances of a ``gm(π, μ, σ)`` term."""
        gm_ctx = term.gm()
        self.aux_pis.append(unpack_gm_list(gm_ctx.list_(0), self.parameters))
        self.aux_means.append(unpack_gm_list(gm_ctx.list_(1), self.parameters))
        self.aux_covs.append(unpack_gm_list(gm_ctx.list_(2), self.parameters).square())

    def enterAssignment(self, ctx: ASGMTParser.AssignmentContext) -> None:
        self.target_index = self.variables.index(self._variable_name(ctx.symvars()))

    def enterAdd(self, ctx: ASGMTParser.AddContext) -> None:
        variable_counts = [sum(self._is_variable(term) for term in summand.term()) for summand in ctx.add_term()]
        self.is_product = len(variable_counts) == 1 and variable_counts[0] == 2
        if not self.is_product and any(count > 1 for count in variable_counts):
            raise SemanticError("Assignments support affine expressions or a single product of two random variables")
        if self.is_product:
            self.product_indices = []
        else:
            self.linear_coefficients = torch.zeros(len(self.variables))
            self.linear_constant = torch.tensor(0.0)

    def enterAdd_term(self, ctx: ASGMTParser.Add_termContext) -> None:
        if self.is_product:
            # Product of two variables (or one variable × one gm term)
            for term in ctx.term():
                if term.sub() is not None:
                    self.product_coefficient = -self.product_coefficient
                if not self._is_variable(term):
                    self.product_coefficient = self.product_coefficient * self._constant_value(term)
                elif term.gm() is not None:
                    self.collect_gm_term(term)
                    assert self.product_indices is not None
                    self.product_indices.append(len(self.variables) + len(self.aux_pis) - 1)
                elif term.symvars() is not None:
                    assert self.product_indices is not None
                    self.product_indices.append(self.variables.index(self._variable_name(term.symvars())))
            self.update_func = partial(mul_func, self)
        else:
            # Linear combination - collect the coefficient for this term
            coefficient = torch.tensor(1.0)
            variable_index: int | None = None

            for term in ctx.term():
                if term.sub() is not None:
                    coefficient = -1 * coefficient
                else:
                    coefficient = 1 * coefficient

                if not self._is_variable(term):
                    coefficient = coefficient * self._constant_value(term)
                elif term.symvars() is not None:
                    variable_index = self.variables.index(self._variable_name(term.symvars()))
                elif term.gm() is not None:
                    self.collect_gm_term(term)
                    assert self.linear_coefficients is not None
                    variable_index = len(self.linear_coefficients) + 1

            assert self.linear_coefficients is not None and self.linear_constant is not None
            if variable_index is not None:
                if variable_index < len(self.linear_coefficients):
                    self.linear_coefficients[variable_index] = self.linear_coefficients[variable_index] + coefficient
                else:
                    self.linear_coefficients = torch.hstack([self.linear_coefficients, coefficient])
            else:
                self.linear_constant = self.linear_constant + coefficient

    def exitAdd(self, ctx: ASGMTParser.AddContext) -> None:
        if not self.is_product:
            assert self.linear_coefficients is not None
            if not torch.all(self.linear_coefficients == 0):
                self.update_func = partial(add_func, self)
            else:
                # Constant assignment (non-differentiable but needed before smoothing)
                self.update_func = partial(const_func, self)


def add_func(rule: AsgmtRule, distribution: Dist) -> Dist:
    """Linear combination update: ``x_i = Σ a_j·x_j + c``."""
    assert rule.target_index is not None and rule.linear_coefficients is not None and rule.linear_constant is not None
    target = rule.target_index
    original_dim = distribution.gm.n_dim()

    # STEP 1: considers all possible combinations of components of the auxiliary variables
    extended_gm = extend_dist(rule, distribution)  # see libSOGAshared

    # STEP 2: computes vectorially the new means and covariance matrices
    new_means = torch.clone(extended_gm.mu)
    new_means[:, target] = torch.matmul(extended_gm.mu, rule.linear_coefficients) + rule.linear_constant

    new_covariances = torch.clone(extended_gm.sigma)
    new_covariances[:, target, :] = new_covariances[:, :, target] = torch.matmul(
        rule.linear_coefficients, extended_gm.sigma
    )
    new_covariances[:, target, target] = torch.matmul(
        torch.matmul(rule.linear_coefficients, extended_gm.sigma),
        rule.linear_coefficients.reshape(-1, 1),
    ).flatten()

    # STEP 3: return new Dist with zeros removed from mixture
    result = Dist(
        distribution.var_list,
        GaussianMix(
            extended_gm.pi,
            new_means[:, :original_dim],
            new_covariances[:, :original_dim, :original_dim],
        ),
    )
    result.gm.delete_zeros()
    return result


def mul_func(rule: AsgmtRule, distribution: Dist) -> Dist:
    """Product update: ``x_i = a · x_j · x_k`` (second-order Gaussian approximation)."""
    assert rule.target_index is not None and rule.product_indices is not None
    target = rule.target_index
    j, k = rule.product_indices
    coefficient = rule.product_coefficient
    original_dim = distribution.gm.n_dim()

    # STEP 1: considers all possible combinations of components of the auxiliary variables
    extended_gm = extend_dist(rule, distribution)  # see libSOGAshared

    # STEP 2: computes mean and covariance matrix for the extended component
    new_means = torch.clone(extended_gm.mu)
    new_means[:, target] = coefficient * (extended_gm.sigma[:, j, k] + extended_gm.mu[:, j] * extended_gm.mu[:, k])

    new_covariances = torch.clone(extended_gm.sigma)
    new_covariances[:, target, :] = new_covariances[:, :, target] = coefficient * (
        extended_gm.mu[:, j].reshape(-1, 1) * extended_gm.sigma[:, k, :]
        + extended_gm.mu[:, k].reshape(-1, 1) * extended_gm.sigma[:, j, :]
    )
    new_covariances[:, target, target] = coefficient.square() * (
        torch.pow(extended_gm.sigma[:, j, k], 2)
        + 2 * extended_gm.sigma[:, j, k] * extended_gm.mu[:, j] * extended_gm.mu[:, k]
        + extended_gm.sigma[:, j, j] * extended_gm.sigma[:, k, k]
        + extended_gm.sigma[:, j, j] * torch.pow(extended_gm.mu[:, k], 2)
        + extended_gm.sigma[:, k, k] * extended_gm.mu[:, j] ** 2
    )

    # STEP 3: return new Dist with zeros removed from mixture
    result = Dist(
        distribution.var_list,
        GaussianMix(
            extended_gm.pi,
            new_means[:, :original_dim],
            new_covariances[:, :original_dim, :original_dim],
        ),
    )
    result.gm.delete_zeros()
    return result


def const_func(rule: AsgmtRule, distribution: Dist) -> Dist:
    """Constant assignment: ``x_i = c`` (sets mean to c, variance to 0)."""
    assert rule.target_index is not None and rule.linear_constant is not None
    target = rule.target_index

    new_means = torch.clone(distribution.gm.mu)
    new_means[:, target] = rule.linear_constant * torch.ones(len(new_means[:, target]))

    new_covariances = torch.clone(distribution.gm.sigma)
    new_covariances[:, target, :] = new_covariances[:, :, target] = torch.zeros(new_covariances[:, :, target].shape)

    return Dist(distribution.var_list, GaussianMix(distribution.gm.pi, new_means, new_covariances))


def parse_assignment(
    variables: list[str],
    expression: str,
    data: dict[str, Any],
    parameters: dict[str, torch.Tensor],
) -> Callable[[Dist], Dist]:
    """Parse *expression* with the ASGMT grammar and return the update function."""
    if expression not in ASGMT_TREE_CACHE:
        lexer = ASGMTLexer(InputStream(expression))
        stream = CommonTokenStream(lexer)
        parser = ASGMTParser(stream)
        tree = parser.assignment()
        if parser.getNumberOfSyntaxErrors() > 0:
            raise SyntaxParseError(f"Invalid assignment syntax: {expression!r}")
        ASGMT_TREE_CACHE[expression] = tree

    rule = AsgmtRule(variables, data, parameters)
    try:
        ParseTreeWalker().walk(rule, ASGMT_TREE_CACHE[expression])
    except ValueError as e:
        raise SemanticError(f"Invalid assignment expression: {expression!r}") from e
    if rule.update_func is None:
        logger.error("AsgmtRule produced no update function for expression=%r", expression)
        raise SemanticError(f"Unsupported or invalid assignment expression: {expression!r}")
    return rule.update_func


def update_rule(
    distribution: Dist,
    expression: str,
    data: dict[str, Any],
    parameters: dict[str, torch.Tensor],
) -> Dist:
    """
    Apply assignment *expression* to *distribution* and return the updated distribution.
    """
    if expression == "skip":
        return distribution
    logger.debug("update_rule  expression=%r  variables=%s", expression, distribution.var_list)
    update_func = parse_assignment(distribution.var_list, expression, data, parameters)
    return update_func(distribution)
