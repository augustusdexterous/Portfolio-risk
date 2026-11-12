"""Covariance estimation and risk-based portfolio weights."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

__all__ = [
    "equal_risk_contribution_weights",
    "portfolio_volatility",
    "risk_contributions",
    "sample_covariance",
    "shrink_covariance",
]

# Relative tolerance: a quantity this small compared with the scale of its inputs is zero.
_REL_ZERO = 1e-12


def _returns_matrix(returns: ArrayLike, min_rows: int = 20) -> NDArray[np.float64]:
    if hasattr(returns, "to_numpy"):
        returns = returns.to_numpy()
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"returns must be a (T, N) matrix, got shape {arr.shape}")
    if arr.shape[0] < min_rows:
        raise ValueError(f"need at least {min_rows} periods, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contain NaN or infinite values")
    if np.max(np.abs(arr)) >= 10.0 or np.mean(np.abs(arr)) > 1.0:
        raise ValueError(
            "returns look like prices or percentages (values of 10 or more, or average magnitude above 1); "
            "returns are decimals (0.01 = 1%)"
        )
    return arr


def _finite_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def sample_covariance(returns: ArrayLike) -> NDArray[np.float64]:
    """Sample covariance of a (T, N) return matrix, ddof=1."""
    r = _returns_matrix(returns)
    return np.atleast_2d(np.cov(r, rowvar=False, ddof=1))


def shrink_covariance(returns: ArrayLike, delta: float = 0.5) -> NDArray[np.float64]:
    """Shrink the sample covariance toward a constant-correlation target.

    ``delta`` in [0, 1] is the shrinkage intensity: 0 keeps the sample covariance and
    1 gives the constant-correlation target. This uses a fixed intensity, not the
    analytically optimal Ledoit-Wolf value, so state it when reporting results.
    """
    if not 0.0 <= delta <= 1.0:
        raise ValueError(f"delta must be in [0, 1], got {delta}")
    r = _returns_matrix(returns)
    s = np.atleast_2d(np.cov(r, rowvar=False, ddof=1))
    sd = np.sqrt(np.diag(s))
    flat = sd <= _REL_ZERO * np.max(np.abs(r), axis=0)
    if np.any(flat):
        raise ValueError(
            f"column {int(np.flatnonzero(flat)[0])} has zero variance; shrinkage needs non-constant returns"
        )
    corr = s / np.outer(sd, sd)
    n = s.shape[0]
    rbar = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
    target = rbar * np.outer(sd, sd)
    np.fill_diagonal(target, np.diag(s))
    return delta * target + (1.0 - delta) * s


def portfolio_volatility(weights: ArrayLike, cov: ArrayLike) -> float:
    """Standard deviation of the portfolio ``sqrt(w' Sigma w)``."""
    w = _finite_array(np.atleast_1d(np.asarray(weights, dtype=np.float64)), "weights")
    c = _finite_array(np.atleast_2d(np.asarray(cov, dtype=np.float64)), "cov")
    if c.shape != (w.size, w.size):
        raise ValueError("cov must be a square matrix matching the number of weights")
    variance = float(w @ c @ w)
    # On a singular matrix, rounding can push an exact zero just below zero.
    tolerance = _REL_ZERO * max(float(np.max(np.abs(c))) * float(w @ w), np.finfo(float).tiny)
    if variance < 0:
        if variance < -tolerance:
            raise ValueError("covariance matrix is not positive semi-definite")
        variance = 0.0
    return float(np.sqrt(variance))


def risk_contributions(weights: ArrayLike, cov: ArrayLike) -> NDArray[np.float64]:
    """Each asset's share of portfolio variance. The shares sum to 1."""
    w = _finite_array(np.atleast_1d(np.asarray(weights, dtype=np.float64)), "weights")
    c = _finite_array(np.atleast_2d(np.asarray(cov, dtype=np.float64)), "cov")
    sigma = portfolio_volatility(w, c)
    if sigma == 0:
        raise ValueError("portfolio volatility is zero; risk shares are undefined")
    marginal = c @ w / sigma
    return w * marginal / sigma


def equal_risk_contribution_weights(cov: ArrayLike) -> NDArray[np.float64]:
    """Long-only weights whose risk contributions are all equal (equal risk contribution).

    Minimizes ``0.5 w' Sigma w - mean(log w)``, whose optimum satisfies
    ``w_i (Sigma w)_i = constant``, then rescales the weights to sum to 1. The solution is
    checked afterwards: if any risk share differs from ``1/N`` by more than 1e-6, the
    function raises instead of returning weights that are not equal-risk.
    """
    c = _finite_array(np.atleast_2d(np.asarray(cov, dtype=np.float64)), "cov")
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] == 0:
        raise ValueError("cov must be a non-empty square matrix")
    n = c.shape[0]
    if not np.allclose(c, c.T, rtol=1e-10, atol=_REL_ZERO * np.max(np.abs(c))):
        raise ValueError("cov must be symmetric")
    if np.any(np.diag(c) <= 0):
        raise ValueError("cov must have strictly positive variances")
    scale = np.sqrt(np.diag(c))
    eigenvalues = np.linalg.eigvalsh(c / np.outer(scale, scale))  # correlation matrix: scale-free
    if eigenvalues.min() <= 1e-9 * eigenvalues.max():
        raise ValueError(
            "cov is singular or nearly singular (smallest eigenvalue is at most 1e-9 of the largest "
            "in correlation terms); equal risk contribution weights are not well defined. "
            "Remove redundant assets or use shrinkage."
        )
    # Equal-risk-contribution weights are unchanged by rescaling the covariance matrix, so
    # solve on the scaled matrix, where the solver's absolute tolerances behave consistently.
    unit = c / np.max(np.diag(c))

    def objective(w):
        return 0.5 * w @ unit @ w - np.mean(np.log(w))

    def gradient(w):
        return unit @ w - 1.0 / (n * w)

    result = optimize.minimize(
        objective,
        np.full(n, 1.0 / n),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(1e-12, None)] * n,
        options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-12},
    )
    w = result.x / result.x.sum()
    shares = risk_contributions(w, c)
    if not np.all(np.isfinite(shares)) or np.max(np.abs(shares - 1.0 / n)) > 1e-6:
        raise RuntimeError(
            "could not compute equal risk contribution weights to tolerance "
            f"(solver: {result.message}). The covariance matrix may be badly scaled or near-singular."
        )
    return w
