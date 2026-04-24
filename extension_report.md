# DeGAS Extension: Polynomial, Trigonometric, and Exponential Assignments

## Overview

This report documents the extension of DeGAS to support three new classes of probabilistic program assignments: **polynomial**, **trigonometric polynomial**, and **exponential polynomial** assignments of the form

```
x = c_0 * m_0 + c_1 * m_1 + ... + c_K * m_K
```

where each monomial `m_k` belongs to one of the three types below. Mixed terms (e.g. `exp` and `cos` in the same assignment) are not allowed and are rejected at the grammar level.

---

## Syntax

### Polynomial assignment

```
x = c_0 * x1^a1 * x2^a2 + c_1 * x3^a3 + ...
```

A monomial is a product of variables raised to non-negative integer powers. Terms with exponent 1 omit the `^1`; terms with exponent 0 omit the variable entirely.

### Trigonometric polynomial assignment

```
x = c_0 * x1^a1 * cos(x2)^b2 * sin(x3)^g3 + ...
```

A trigonometric monomial is a product of polynomial factors `xi^alphai`, cosine factors `cos(xi)^betai`, and sine factors `sin(xi)^gammai`.

### Exponential polynomial assignment

```
x = c_0 * x1^a1 * exp(x2)^b2 + ...
```

An exponential monomial is a product of polynomial factors `xi^alphai` and exponential factors `exp(xi)^betai`.

`gm(...)` random variables may appear as atoms in any of the three monomial types.

---

## Grammar Changes

Both `SOGA.g4` and `ASGMT.g4` were extended with new rules:

- `poly_asgmt` / `poly_sum`: sums of polynomial terms
- `trig_asgmt` / `trig_sum`: sums of trigonometric polynomial terms
- `exp_asgmt` / `exp_sum`: sums of exponential polynomial terms

The original `add` rule (linear assignments) is preserved unchanged and remains the preferred path for linear expressions.

---

## Semantics

### Reduction to monomial expectations

Given `xi = sum_k c_k * m_k`, the update rules for mean and covariance reduce to computing expectations of single monomials:

- **Mean:** `E[Xi] = sum_k c_k * E[m_k]`
- **Covariance row:** `Cov(Xi, Xj) = sum_k c_k * E[Xj * m_k] - E[Xi] * E[Xj]`
- **Variance:** `Var(Xi) = sum_{k1,k2} c_{k1} * c_{k2} * E[m_{k1} * m_{k2}] - E[Xi]^2`

Note that `Xj * m_k`, `m_{k1} * m_{k2}` are themselves monomials of the same type.

### Polynomial monomial: Isserlis' theorem

`E[X^alpha]` for a Gaussian `X ~ N(nu, C)` is computed via the recursion derived from Isserlis' theorem:

```
m_0 = 1
m_{alpha + e_i} = nu_i * m_alpha + sum_j  C_ij * alpha_j * m_{alpha - e_j}
```

This recursion works for any symmetric `C` (including negative-definite and complex entries).

**Implementation:** `poly_moment(alpha, nu, C)` — memoised scalar recursion; `gaussian_poly_moment_batch(alpha, mu, sigma)` — vectorised over mixture components.

### Exponential monomial: MGF identity

`E[X^alpha * prod_i exp(beta_i * Xi)]` is computed via the MGF shift identity:

```
= M_X(beta) * E_{Z ~ N(mu + Sigma @ beta, Sigma)}[Z^alpha]
```

where `M_X(beta) = exp(mu^T beta + 0.5 * beta^T Sigma beta)`. This reduces the expectation to a shifted polynomial moment, which is handled by `poly_moment`.

**Implementation:** `exp_mono_moment_batch(alpha, beta, mu, sigma)`.

### Trigonometric monomial: characteristic function formula

`E[X^alpha * prod cos^beta * prod sin^gamma]` is computed by expanding cosines and sines via Euler's formula and applying derivatives of the characteristic function `Phi_X(t) = exp(i mu^T t - 0.5 t^T Sigma t)`:

```
E[...] = (1 / (2^{sum(beta+gamma)} * i^{sum(alpha+gamma)}))
         * sum_{b in [0,beta]} sum_{c in [0,gamma]}
             prod_i C(beta_i, b_i) * C(gamma_i, c_i)
             * (-1)^{sum_i (gamma_i - c_i)}
             * D^alpha Phi_X(t)|_{t_i = 2(b_i+c_i) - beta_i - gamma_i}
```

The derivative `D^alpha Phi_X(s)` evaluates to `Phi_X(s) * poly_moment(alpha, i*mu - Sigma*s, -Sigma)`, which again reduces to a (complex) polynomial moment. The factor `(-1)^{sum(gamma_i - c_i)}` arises from the binomial expansion of `sin^gamma = ((e^{ix} - e^{-ix}) / 2i)^gamma` and must not be omitted.

**Implementation:** `trig_mono_moment_batch(alpha, beta, gamma, mu, sigma)` using `torch.complex128` intermediate tensors; result is guaranteed real and cast back to the input dtype.

---

## New Functions in `src/libSOGAupdate.py`

| Function | Description |
|---|---|
| `poly_moment(alpha, nu, C)` | Isserlis recursion for `E[X^alpha]`; supports complex `nu`, `C` |
| `gaussian_poly_moment_batch(alpha, mu, sigma)` | Vectorised `poly_moment` over `n_comp` mixture components |
| `exp_mono_moment_batch(alpha, beta, mu, sigma)` | `E[X^alpha * prod exp(beta_i Xi)]` via MGF shift |
| `trig_mono_moment_batch(alpha, beta, gamma, mu, sigma)` | `E[X^alpha * prod cos^beta * prod sin^gamma]` via CF formula |
| `poly_func(self, dist)` | Semantic update for polynomial assignments |
| `trig_func(self, dist)` | Semantic update for trigonometric polynomial assignments |
| `exp_func(self, dist)` | Semantic update for exponential polynomial assignments |

## New Helper Methods in `src/ASGMTParser.py`

Added to `Poly_factorContext`, `Trig_factorContext`, `Exp_factorContext`:
`is_gm()`, `is_var(data)`, `getVar(data)`, `getExp()`, `is_var_factor()`, `is_cos()`, `is_sin()`, `is_exp()`

Added to `Poly_termContext`, `Trig_termContext`, `Exp_termContext`:
`getCoeff(data, params)`

---

## Validation

All three new update types were validated against Monte Carlo estimates (500,000 samples) on a 2D Gaussian `N([1, 2], [[1, 0.5], [0.5, 2]])`:

| Expression | Monte Carlo | DeGAS |
|---|---|---|
| `E[x^2 * y]` (poly) | 4.995 | 5.000 |
| `E[exp(x) * y]` (exp) | 12.26 | 12.28 |
| `E[cos(x)]` (trig) | 0.3096 | 0.3096 |
| `E[x * sin(y)]` (trig) | 0.2595 | 0.2580 |
| `E[cos(x) * sin(y)]` (trig) | 0.1638 | 0.1643 |
