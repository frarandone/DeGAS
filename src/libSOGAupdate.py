# Contains the functions for computing the resulting distribution when an assignment instruction is encountered (in state nodes).

# SOGA (defined in SOGA.py)
# |- update_rule
#    |- sym_expr
#    |- update_gaussian

from libSOGAshared import *
from ASGMTListener import *
from ASGMTParser import *
from ASGMTLexer import *


# ---------------------------------------------------------------------------
# Core mathematical utilities
# ---------------------------------------------------------------------------

def poly_moment(alpha, nu, C):
    """
    Compute the generalised polynomial moment  E[X^alpha]  for a (possibly
    formal) Gaussian with mean nu and covariance matrix C.

    Uses the recursion derived from Isserlis' theorem:
        m_0 = 1
        m_{alpha + e_i} = nu_i * m_alpha  +  sum_j  C_{ij} * alpha_j * m_{alpha - e_j}

    Parameters
    ----------
    alpha : tuple of int  (length = n_dim)
        Multi-index of the monomial.  Each entry >= 0.
    nu    : 1-D tensor of length n_dim  (may be complex)
        Mean vector.
    C     : 2-D tensor (n_dim x n_dim)  (may be negative-definite or complex)
        Covariance matrix.

    Returns
    -------
    Scalar tensor (possibly complex).
    """
    n = len(alpha)
    memo = {}

    def _m(a):
        a = tuple(a)
        if a in memo:
            return memo[a]
        if all(x == 0 for x in a):
            return nu.new_ones(()) if not nu.is_complex() else nu.new_ones(()).to(torch.complex128)
        # find the first nonzero index to reduce on
        i = next(k for k in range(n) if a[k] > 0)
        a_prev = list(a)
        a_prev[i] -= 1
        result = nu[i] * _m(tuple(a_prev))
        for j in range(n):
            if a_prev[j] > 0:
                a_prev2 = list(a_prev)
                a_prev2[j] -= 1
                result = result + C[i, j] * a_prev[j] * _m(tuple(a_prev2))
        memo[a] = result
        return result

    return _m(tuple(alpha))


def gaussian_poly_moment_batch(alpha, mu, sigma):
    """
    Vectorised polynomial moment  E[X^alpha]  for a batch of Gaussians.

    Parameters
    ----------
    alpha : tuple of int  (length = n_dim)
    mu    : tensor (n_comp, n_dim)
    sigma : tensor (n_comp, n_dim, n_dim)

    Returns
    -------
    tensor (n_comp,)
    """
    n_comp = mu.shape[0]
    results = torch.stack([poly_moment(alpha, mu[k], sigma[k]) for k in range(n_comp)])
    return results.real if results.is_complex() else results


def exp_mono_moment_batch(alpha, beta, mu, sigma):
    """
    E[ X^alpha * prod_i exp(beta_i * X_i) ]  for a batch of Gaussians.

    Uses the identity:
        = MGF(beta) * E_{Z ~ N(mu + sigma @ beta, sigma)}[ Z^alpha ]

    Parameters
    ----------
    alpha : tuple of int
    beta  : tuple of int  (the exponents on exp factors)
    mu    : tensor (n_comp, n_dim)
    sigma : tensor (n_comp, n_dim, n_dim)

    Returns
    -------
    tensor (n_comp,)
    """
    beta_t = torch.tensor(beta, dtype=mu.dtype)                          # (n_dim,)
    mu_shift = mu + torch.einsum('cij,j->ci', sigma, beta_t)             # (n_comp, n_dim)
    log_mgf = torch.einsum('ci,i->c', mu, beta_t) + \
              0.5 * torch.einsum('i,cij,j->c', beta_t, sigma, beta_t)    # (n_comp,)
    mgf = torch.exp(log_mgf)                                             # (n_comp,)
    moments = torch.stack([poly_moment(alpha, mu_shift[k], sigma[k])
                           for k in range(mu.shape[0])])                 # (n_comp,)
    return mgf * moments


def trig_mono_moment_batch(alpha, beta, gamma, mu, sigma):
    """
    E[ X^alpha * prod_i cos(X_i)^beta_i * prod_i sin(X_i)^gamma_i ]
    for a batch of Gaussians, via the characteristic function formula.

    Parameters
    ----------
    alpha, beta, gamma : tuple of int  (length = n_dim)
    mu    : tensor (n_comp, n_dim)
    sigma : tensor (n_comp, n_dim, n_dim)

    Returns
    -------
    tensor (n_comp,)  (real)
    """
    from itertools import product as iproduct
    from math import comb

    n_dim = len(alpha)
    n_comp = mu.shape[0]

    denom_pow = sum(beta[i] + gamma[i] for i in range(n_dim))
    gamma_pow = sum(alpha[i] + gamma[i] for i in range(n_dim))
    # D = 2^{sum(beta+gamma)} * i^{sum(alpha+gamma)}
    D = (2 ** denom_pow) * (1j ** gamma_pow)

    # promote to complex
    mu_c  = mu.to(torch.complex128)
    sig_c = sigma.to(torch.complex128)
    neg_sig_c = -sig_c                                    # C = -Sigma for CF case

    result = torch.zeros(n_comp, dtype=torch.complex128)

    # sum over (b_1,...,b_m) in [0,beta_i] and (c_1,...,c_m) in [0,gamma_i]
    ranges = [range(beta[i] + 1) for i in range(n_dim)] + \
             [range(gamma[i] + 1) for i in range(n_dim)]
    for combo in iproduct(*ranges):
        b = combo[:n_dim]
        c = combo[n_dim:]
        binom_prod = 1
        for i in range(n_dim):
            binom_prod *= comb(beta[i], b[i]) * comb(gamma[i], c[i])
        sign = (-1) ** sum(gamma[i] - c[i] for i in range(n_dim))
        binom_prod *= sign
        # s_i = 2*(b_i + c_i) - beta_i - gamma_i
        s = torch.tensor([2 * (b[i] + c[i]) - beta[i] - gamma[i]
                          for i in range(n_dim)], dtype=torch.float64)
        s_c = s.to(torch.complex128)
        # Phi_X(s) = exp(i*mu^T*s - 0.5*s^T*Sigma*s)  per component
        log_phi = (1j * torch.einsum('ci,i->c', mu_c, s_c)
                   - 0.5 * torch.einsum('i,cij,j->c', s_c, sig_c, s_c))  # (n_comp,)
        phi = torch.exp(log_phi)
        # nu = i*mu - Sigma*s  per component
        nu_c = (1j * mu_c
                - torch.einsum('cij,j->ci', sig_c, s_c))                  # (n_comp, n_dim)
        for k in range(n_comp):
            m_val = poly_moment(alpha, nu_c[k], neg_sig_c[k])
            result[k] += binom_prod * phi[k] * m_val

    return (result / D).real.to(mu.dtype)


# ---------------------------------------------------------------------------
# AsgmtRule listener
# ---------------------------------------------------------------------------

class AsgmtRule(ASGMTListener):

    def __init__(self, var_list, data, params_dict):
        self.var_list = var_list
        self.data = data
        self.params = params_dict
        self.target = None
        self.is_prod = None
        # auxiliary random variables
        self.aux_pis = []
        self.aux_means = []
        self.aux_covs = []
        # function to be applied
        self.func = None

    def unpack_rvs(self, gm_ctx):
        self.aux_pis.append(gm_ctx.list_()[0].unpack(self.params))
        self.aux_means.append(gm_ctx.list_()[1].unpack(self.params))
        self.aux_covs.append(torch.pow(gm_ctx.list_()[2].unpack(self.params), 2))

    def enterAssignment(self, ctx):
        self.target = self.var_list.index(ctx.symvars().getVar(self.data))

    # ------------------------------------------------------------------
    # Linear case (unchanged)
    # ------------------------------------------------------------------

    def enterAdd(self, ctx):
        if len(ctx.add_term()) == 1 and len(ctx.add_term(0).term()) == 2:
            self.is_prod = 1
            for term in ctx.add_term(0).term():
                self.is_prod = self.is_prod * term.is_var(self.data)
        if self.is_prod:
            self.mul_idx = []
        else:
            self.add_coeff = torch.zeros(len(self.var_list))
            self.add_const = torch.tensor(0.)

    def enterAdd_term(self, ctx):
        if self.is_prod:
            for term in ctx.term():
                if not term.gm() is None:
                    self.unpack_rvs(term.gm())
                    self.mul_idx.append(int(len(self.var_list) + len(self.aux_pis) - 1))
                elif not term.symvars() is None:
                    self.mul_idx.append(self.var_list.index(term.symvars().getVar(self.data)))
            self.func = partial(mul_func, self)
        else:
            coeff = torch.tensor(1.)
            var_idx = None
            for term in ctx.term():
                if term.sub() is not None:
                    coeff = -1 * coeff
                else:
                    coeff = 1 * coeff
                if term.is_const(self.data):
                    coeff = coeff * term.getValue(self.data, self.params)
                elif not term.symvars() is None:
                    var_idx = self.var_list.index(term.symvars().getVar(self.data))
                elif not term.gm() is None:
                    self.unpack_rvs(term.gm())
                    var_idx = len(self.add_coeff) + 1
            if not var_idx is None:
                if var_idx < len(self.add_coeff):
                    self.add_coeff[var_idx] = coeff
                else:
                    self.add_coeff = torch.hstack([self.add_coeff, coeff])
            else:
                self.add_const = self.add_const + coeff

    def exitAdd(self, ctx):
        if not self.is_prod:
            if not torch.all(self.add_coeff == 0):
                self.func = partial(add_func, self)
            else:
                self.func = partial(const_func, self)

    # ------------------------------------------------------------------
    # Polynomial case
    # ------------------------------------------------------------------

    def enterPoly_sum(self, ctx):
        self.poly_terms = []    # list of (coeff, monomial_dict)
        # monomial_dict maps var_index -> exponent

    def enterPoly_term(self, ctx):
        coeff = ctx.getCoeff(self.data, self.params)
        mono = ctx.poly_mono()
        if mono is None:
            # constant term: monomial is the empty product (degree 0)
            self.poly_terms.append((coeff, {}))
        else:
            mono_dict = {}
            for fac in mono.poly_factor():
                exp = fac.getExp()
                if fac.is_gm():
                    self.unpack_rvs(fac.gm())
                    idx = len(self.var_list) + len(self.aux_pis) - 1
                else:
                    var_name = fac.getVar(self.data)
                    idx = self.var_list.index(var_name)
                mono_dict[idx] = mono_dict.get(idx, 0) + exp
            self.poly_terms.append((coeff, mono_dict))

    def exitPoly_sum(self, ctx):
        self.func = partial(poly_func, self)

    # ------------------------------------------------------------------
    # Trigonometric polynomial case
    # ------------------------------------------------------------------

    def enterTrig_sum(self, ctx):
        # each term: (coeff, var_exponents, cos_exponents, sin_exponents)
        # all three dicts map var_index -> exponent
        self.trig_terms = []

    def enterTrig_term(self, ctx):
        coeff = ctx.getCoeff(self.data, self.params)
        mono = ctx.trig_mono()
        if mono is None:
            self.trig_terms.append((coeff, {}, {}, {}))
        else:
            var_exp  = {}
            cos_exp  = {}
            sin_exp  = {}
            for fac in mono.trig_factor():
                exp = fac.getExp()
                if fac.is_var_factor():
                    if fac.is_gm():
                        self.unpack_rvs(fac.gm())
                        idx = len(self.var_list) + len(self.aux_pis) - 1
                    else:
                        idx = self.var_list.index(fac.getVar(self.data))
                    var_exp[idx] = var_exp.get(idx, 0) + exp
                elif fac.is_cos():
                    if fac.is_gm():
                        self.unpack_rvs(fac.gm())
                        idx = len(self.var_list) + len(self.aux_pis) - 1
                    else:
                        idx = self.var_list.index(fac.getVar(self.data))
                    cos_exp[idx] = cos_exp.get(idx, 0) + exp
                else:  # sin
                    if fac.is_gm():
                        self.unpack_rvs(fac.gm())
                        idx = len(self.var_list) + len(self.aux_pis) - 1
                    else:
                        idx = self.var_list.index(fac.getVar(self.data))
                    sin_exp[idx] = sin_exp.get(idx, 0) + exp
            self.trig_terms.append((coeff, var_exp, cos_exp, sin_exp))

    def exitTrig_sum(self, ctx):
        self.func = partial(trig_func, self)

    # ------------------------------------------------------------------
    # Exponential polynomial case
    # ------------------------------------------------------------------

    def enterExp_sum(self, ctx):
        self.exp_terms = []

    def enterExp_term(self, ctx):
        coeff = ctx.getCoeff(self.data, self.params)
        mono = ctx.exp_mono()
        if mono is None:
            self.exp_terms.append((coeff, {}, {}))
        else:
            var_exp = {}
            exp_exp = {}
            for fac in mono.exp_factor():
                e = fac.getExp()
                if fac.is_var_factor():
                    if fac.is_gm():
                        self.unpack_rvs(fac.gm())
                        idx = len(self.var_list) + len(self.aux_pis) - 1
                    else:
                        idx = self.var_list.index(fac.getVar(self.data))
                    var_exp[idx] = var_exp.get(idx, 0) + e
                else:  # exp(x)^n
                    if fac.is_gm():
                        self.unpack_rvs(fac.gm())
                        idx = len(self.var_list) + len(self.aux_pis) - 1
                    else:
                        idx = self.var_list.index(fac.getVar(self.data))
                    exp_exp[idx] = exp_exp.get(idx, 0) + e
            self.exp_terms.append((coeff, var_exp, exp_exp))

    def exitExp_sum(self, ctx):
        self.func = partial(exp_func, self)


# ---------------------------------------------------------------------------
# asgmt_parse / update_rule
# ---------------------------------------------------------------------------

def asgmt_parse(var_list, expr, data, params_dict):
    """ Parses expr using ANTLR4. Returns a function """
    lexer = ASGMTLexer(InputStream(expr))
    stream = CommonTokenStream(lexer)
    parser = ASGMTParser(stream)
    tree = parser.assignment()
    asgmt_rule = AsgmtRule(var_list, data, params_dict)
    walker = ParseTreeWalker()
    walker.walk(asgmt_rule, tree)
    return asgmt_rule.func


def update_rule(dist, expr, data, params_dict):
    """ Applies expr to dist. It first parses expr using the function asgmt_parse, implemented as an ANTLR listener. asgmt_parse returns a function rule_func, such that, rule_func(GaussianMix) returns a new GaussianMix object obtained applying expr to the initial distribution. rule_func is applied to each component of dist, and the resulting Gaussian mixtures are stored in a single GaussianMix object."""
    if expr == 'skip':
        return dist
    else:
        rule_func = asgmt_parse(dist.var_list, expr, data, params_dict)
        return rule_func(dist)


# ---------------------------------------------------------------------------
# Semantic update functions
# ---------------------------------------------------------------------------

def _dicts_to_tuples(var_dict, n_dim):
    """Convert a {index: exponent} dict to a full tuple of length n_dim."""
    a = [0] * n_dim
    for idx, exp in var_dict.items():
        a[idx] = exp
    return tuple(a)


def _expect_mono_poly(alpha, extended_gm):
    """E[X^alpha] for a batch of Gaussians (n_comp, n_dim)."""
    return gaussian_poly_moment_batch(alpha, extended_gm.mu, extended_gm.sigma)


def _expect_mono_trig(alpha, beta_t, gamma_t, extended_gm):
    """E[X^alpha * cos^beta * sin^gamma] for a batch of Gaussians."""
    return trig_mono_moment_batch(alpha, beta_t, gamma_t,
                                  extended_gm.mu, extended_gm.sigma)


def _expect_mono_exp(alpha, beta_e, extended_gm):
    """E[X^alpha * exp^beta] for a batch of Gaussians."""
    return exp_mono_moment_batch(alpha, beta_e, extended_gm.mu, extended_gm.sigma)


def _expect_xj_times_mono(j, alpha, extended_gm, expect_fn, **kwargs):
    """E[X_j * mono] = expect_fn with alpha[j] incremented by 1."""
    alpha2 = list(alpha)
    alpha2[j] += 1
    return expect_fn(tuple(alpha2), extended_gm=extended_gm, **kwargs)


def poly_func(self, dist):
    i = self.target
    old_dim = dist.gm.n_dim()
    extended_gm = extend_dist(self, dist)
    n_dim_ext = extended_gm.n_dim()

    terms = self.poly_terms

    # ---- new mean of X_i ----
    new_mu_i = torch.zeros(extended_gm.pi.shape[0])
    for coeff, var_dict in terms:
        alpha = _dicts_to_tuples(var_dict, n_dim_ext)
        new_mu_i = new_mu_i + coeff * _expect_mono_poly(alpha, extended_gm)

    # ---- new covariance row/col for X_i ----
    # Cov(X_i, X_j) = E[X_i * X_j] - E[X_i]*E[X_j]
    # E[X_i * X_j] = sum_k c_k * E[X_j * m_k]
    new_sigma_i = torch.zeros(extended_gm.pi.shape[0], n_dim_ext)
    for j in range(n_dim_ext):
        e_xi_xj = torch.zeros(extended_gm.pi.shape[0])
        for coeff, var_dict in terms:
            alpha = list(_dicts_to_tuples(var_dict, n_dim_ext))
            alpha[j] += 1
            e_xi_xj = e_xi_xj + coeff * _expect_mono_poly(tuple(alpha), extended_gm)
        new_sigma_i[:, j] = e_xi_xj

    # Var(X_i) = E[X_i^2] - E[X_i]^2
    e_xi2 = torch.zeros(extended_gm.pi.shape[0])
    for k1, (c1, d1) in enumerate(terms):
        a1 = _dicts_to_tuples(d1, n_dim_ext)
        for k2, (c2, d2) in enumerate(terms):
            a2 = _dicts_to_tuples(d2, n_dim_ext)
            alpha_prod = tuple(a1[j] + a2[j] for j in range(n_dim_ext))
            e_xi2 = e_xi2 + c1 * c2 * _expect_mono_poly(alpha_prod, extended_gm)

    # assemble
    extended_mu    = torch.clone(extended_gm.mu)
    extended_mu[:, i] = new_mu_i
    extended_sigma = torch.clone(extended_gm.sigma)
    extended_sigma[:, i, :] = new_sigma_i - new_mu_i.unsqueeze(1) * extended_gm.mu
    extended_sigma[:, :, i] = extended_sigma[:, i, :]
    extended_sigma[:, i, i] = e_xi2 - new_mu_i ** 2

    new_dist = Dist(dist.var_list,
                    GaussianMix(extended_gm.pi,
                                extended_mu[:, :old_dim],
                                extended_sigma[:, :old_dim, :old_dim]))
    new_dist.gm.delete_zeros()
    return new_dist


def trig_func(self, dist):
    i = self.target
    old_dim = dist.gm.n_dim()
    extended_gm = extend_dist(self, dist)
    n_dim_ext = extended_gm.n_dim()

    terms = self.trig_terms  # list of (coeff, var_exp, cos_exp, sin_exp)

    def _E_mono(var_dict, cos_dict, sin_dict):
        alpha = _dicts_to_tuples(var_dict, n_dim_ext)
        beta  = _dicts_to_tuples(cos_dict, n_dim_ext)
        gamma = _dicts_to_tuples(sin_dict, n_dim_ext)
        return _expect_mono_trig(alpha, beta, gamma, extended_gm)

    # mean
    new_mu_i = torch.zeros(extended_gm.pi.shape[0])
    for coeff, vd, cd, sd in terms:
        new_mu_i = new_mu_i + coeff * _E_mono(vd, cd, sd)

    # covariance
    new_sigma_i = torch.zeros(extended_gm.pi.shape[0], n_dim_ext)
    for j in range(n_dim_ext):
        e_xi_xj = torch.zeros(extended_gm.pi.shape[0])
        for coeff, vd, cd, sd in terms:
            vd2 = dict(vd)
            vd2[j] = vd2.get(j, 0) + 1
            e_xi_xj = e_xi_xj + coeff * _E_mono(vd2, cd, sd)
        new_sigma_i[:, j] = e_xi_xj

    e_xi2 = torch.zeros(extended_gm.pi.shape[0])
    for (c1, vd1, cd1, sd1) in terms:
        for (c2, vd2, cd2, sd2) in terms:
            vd_prod = {k: vd1.get(k, 0) + vd2.get(k, 0) for k in set(vd1) | set(vd2)}
            cd_prod = {k: cd1.get(k, 0) + cd2.get(k, 0) for k in set(cd1) | set(cd2)}
            sd_prod = {k: sd1.get(k, 0) + sd2.get(k, 0) for k in set(sd1) | set(sd2)}
            e_xi2 = e_xi2 + c1 * c2 * _E_mono(vd_prod, cd_prod, sd_prod)

    extended_mu    = torch.clone(extended_gm.mu)
    extended_mu[:, i] = new_mu_i
    extended_sigma = torch.clone(extended_gm.sigma)
    extended_sigma[:, i, :] = new_sigma_i - new_mu_i.unsqueeze(1) * extended_gm.mu
    extended_sigma[:, :, i] = extended_sigma[:, i, :]
    extended_sigma[:, i, i] = e_xi2 - new_mu_i ** 2

    new_dist = Dist(dist.var_list,
                    GaussianMix(extended_gm.pi,
                                extended_mu[:, :old_dim],
                                extended_sigma[:, :old_dim, :old_dim]))
    new_dist.gm.delete_zeros()
    return new_dist


def exp_func(self, dist):
    i = self.target
    old_dim = dist.gm.n_dim()
    extended_gm = extend_dist(self, dist)
    n_dim_ext = extended_gm.n_dim()

    terms = self.exp_terms  # list of (coeff, var_exp, exp_exp)

    def _E_mono(var_dict, exp_dict):
        alpha = _dicts_to_tuples(var_dict, n_dim_ext)
        beta  = _dicts_to_tuples(exp_dict, n_dim_ext)
        return _expect_mono_exp(alpha, beta, extended_gm)

    # mean
    new_mu_i = torch.zeros(extended_gm.pi.shape[0])
    for coeff, vd, ed in terms:
        new_mu_i = new_mu_i + coeff * _E_mono(vd, ed)

    # covariance
    new_sigma_i = torch.zeros(extended_gm.pi.shape[0], n_dim_ext)
    for j in range(n_dim_ext):
        e_xi_xj = torch.zeros(extended_gm.pi.shape[0])
        for coeff, vd, ed in terms:
            vd2 = dict(vd)
            vd2[j] = vd2.get(j, 0) + 1
            e_xi_xj = e_xi_xj + coeff * _E_mono(vd2, ed)
        new_sigma_i[:, j] = e_xi_xj

    e_xi2 = torch.zeros(extended_gm.pi.shape[0])
    for (c1, vd1, ed1) in terms:
        for (c2, vd2, ed2) in terms:
            vd_prod = {k: vd1.get(k, 0) + vd2.get(k, 0) for k in set(vd1) | set(vd2)}
            ed_prod = {k: ed1.get(k, 0) + ed2.get(k, 0) for k in set(ed1) | set(ed2)}
            e_xi2 = e_xi2 + c1 * c2 * _E_mono(vd_prod, ed_prod)

    extended_mu    = torch.clone(extended_gm.mu)
    extended_mu[:, i] = new_mu_i
    extended_sigma = torch.clone(extended_gm.sigma)
    extended_sigma[:, i, :] = new_sigma_i - new_mu_i.unsqueeze(1) * extended_gm.mu
    extended_sigma[:, :, i] = extended_sigma[:, i, :]
    extended_sigma[:, i, i] = e_xi2 - new_mu_i ** 2

    new_dist = Dist(dist.var_list,
                    GaussianMix(extended_gm.pi,
                                extended_mu[:, :old_dim],
                                extended_sigma[:, :old_dim, :old_dim]))
    new_dist.gm.delete_zeros()
    return new_dist


# ---------------------------------------------------------------------------
# Original linear / product update functions (unchanged)
# ---------------------------------------------------------------------------

def add_func(self, dist):

    i = self.target
    old_dim = dist.gm.n_dim()

    extended_gm = extend_dist(self, dist)

    extended_mu = torch.clone(extended_gm.mu)
    extended_mu[:, i] = torch.matmul(extended_gm.mu, self.add_coeff) + self.add_const
    extended_sigma = torch.clone(extended_gm.sigma)
    extended_sigma[:, i, :] = extended_sigma[:, :, i] = torch.matmul(self.add_coeff, extended_gm.sigma)
    extended_sigma[:, i, i] = torch.matmul(torch.matmul(self.add_coeff, extended_gm.sigma), self.add_coeff.reshape(-1, 1)).flatten()
    new_dist = Dist(dist.var_list, GaussianMix(extended_gm.pi, extended_mu[:, :old_dim], extended_sigma[:, :old_dim, :old_dim]))
    new_dist.gm.delete_zeros()

    return new_dist


def mul_func(self, dist):

    i = self.target
    j, k = self.mul_idx
    old_dim = dist.gm.n_dim()

    extended_gm = extend_dist(self, dist)

    extended_mu = torch.clone(extended_gm.mu)
    extended_mu[:, i] = extended_gm.sigma[:, j, k] + extended_gm.mu[:, j] * extended_gm.mu[:, k]
    extended_sigma = torch.clone(extended_gm.sigma)
    extended_sigma[:, i, :] = extended_sigma[:, :, i] = extended_gm.mu[:, j].reshape(-1, 1) * extended_gm.sigma[:, k, :] + extended_gm.mu[:, k].reshape(-1, 1) * extended_gm.sigma[:, j, :]
    extended_sigma[:, i, i] = (torch.pow(extended_gm.sigma[:, j, k], 2)
                                + 2 * extended_gm.sigma[:, j, k] * extended_gm.mu[:, j] * extended_gm.mu[:, k]
                                + extended_gm.sigma[:, j, j] * extended_gm.sigma[:, k, k]
                                + extended_gm.sigma[:, j, j] * torch.pow(extended_gm.mu[:, k], 2)
                                + extended_gm.sigma[:, k, k] * extended_gm.mu[:, j] ** 2)
    new_dist = Dist(dist.var_list, GaussianMix(extended_gm.pi, extended_mu[:, :old_dim], extended_sigma[:, :old_dim, :old_dim]))
    new_dist.gm.delete_zeros()

    return new_dist


def const_func(self, dist):
    i = self.target
    new_mu = torch.clone(dist.gm.mu)
    new_mu[:, i] = self.add_const * torch.ones(len(new_mu[:, i]))
    new_sigma = torch.clone(dist.gm.sigma)
    new_sigma[:, i, :] = new_sigma[:, :, i] = torch.zeros(new_sigma[:, :, i].shape)
    new_dist = Dist(dist.var_list, GaussianMix(dist.gm.pi, new_mu, new_sigma))
    return new_dist
