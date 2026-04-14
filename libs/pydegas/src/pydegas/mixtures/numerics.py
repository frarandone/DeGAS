"""Multivariate normal CDF and covariance symmetry helpers."""

from __future__ import annotations

import logging

import botorch.utils.probability.mvnxpb as mvn
import torch

from pydegas.mixtures.constants import TOL_ERR


logger = logging.getLogger(__name__)


def make_sym(sigma: torch.Tensor) -> torch.Tensor:
    """Make a 2D tensor symmetric by averaging with its transpose."""
    sigma = sigma.clone()
    symmetric_sigma = (sigma + sigma.T) / 2
    diff = torch.abs(symmetric_sigma - sigma)
    indices = torch.nonzero(diff > TOL_ERR, as_tuple=False)
    for i, j in indices:
        logger.warning("Substituting %s with %s", sigma[i, j].item(), symmetric_sigma[i, j].item())
    return symmetric_sigma


def mvncdf(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """CDF of multivariate normal at x (upper orthant style bounds), batch over leading dim of x."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    batch_size = x.shape[0]
    dim = x.shape[1]
    bounds = torch.stack(
        [torch.tensor([[-torch.inf, x[i, j] - mean[j]] for j in range(dim)]) for i in range(batch_size)]
    )
    res = torch.zeros(batch_size)
    for i in range(batch_size):
        result = torch.exp(mvn.MVNXPB(covariance_matrix=cov, bounds=bounds[i]).solve())
        if result.isnan():
            cov = make_sym(cov)
            result = torch.exp(mvn.MVNXPB(covariance_matrix=cov, bounds=bounds[i]).solve())
        res[i] = result
    return res
