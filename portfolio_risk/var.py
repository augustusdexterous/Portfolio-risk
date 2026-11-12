"""Value at Risk and expected shortfall, historical and parametric.

Historical estimates use the empirical distribution. Parametric estimates assume
normally distributed returns. Cornish-Fisher VaR corrects the normal quantile for
sample skewness and excess kurtosis.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from ._validate import as_series, check_alpha

__all__ = [
    "cornish_fisher_var",
    "expected_shortfall",
    "historical_var",
    "parametric_expected_shortfall",
    "parametric_var",
]


def historical_var(returns, alpha: float = 0.05) -> float:
    """Loss exceeded with probability ``alpha``, read from the empirical distribution."""
    r = as_series(returns, min_obs=20)
    a = check_alpha(alpha)
    return float(-np.quantile(r, a, method="linear"))


def expected_shortfall(returns, alpha: float = 0.05) -> float:
    """Average loss in the worst ``alpha`` share of the sample (historical CVaR).

    Uses the empirical estimator of Acerbi and Tasche (2002). With ``k = n * alpha``
    tail observations, the ``floor(k)`` worst are counted fully and the next one is
    weighted by the fractional part, then the total is divided by ``k``. When ``n * alpha``
    is an integer this is the plain average of the worst observations.
    """
    r = as_series(returns, min_obs=20)
    a = check_alpha(alpha)
    worst_first = np.sort(r)
    k = r.size * a
    m = int(np.floor(k))
    frac = k - m
    total = float(worst_first[:m].sum())
    if frac > 0:
        total += frac * float(worst_first[m])
    return float(-total / k)


def parametric_var(returns, alpha: float = 0.05) -> float:
    """Normal-distribution VaR: ``-(mu + z_alpha * sigma)`` with ``sigma`` using ddof=1."""
    r = as_series(returns, min_obs=3)
    a = check_alpha(alpha)
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    return float(-(mu + stats.norm.ppf(a) * sigma))


def parametric_expected_shortfall(returns, alpha: float = 0.05) -> float:
    """Normal-distribution expected shortfall: ``sigma * phi(z_alpha) / alpha - mu``."""
    r = as_series(returns, min_obs=3)
    a = check_alpha(alpha)
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    z = stats.norm.ppf(a)
    return float(sigma * stats.norm.pdf(z) / a - mu)


def cornish_fisher_var(returns, alpha: float = 0.05) -> float:
    """VaR with the normal quantile adjusted for skewness and excess kurtosis.

    The expansion is an approximation and can fail in two ways. It can overstate tail
    losses when excess kurtosis is large: on S&P 500 daily returns (excess kurtosis about
    8) the 1% Cornish-Fisher VaR is well above the historical estimate. And the adjusted
    quantile can stop increasing in ``z`` at a given ``alpha``, which makes the VaR
    meaningless there.

    A UserWarning is emitted when the excess kurtosis exceeds 4 (a heuristic threshold),
    and when the adjusted quantile is not increasing at ``alpha``. Compare with
    ``historical_var`` before relying on it.
    """
    r = as_series(returns, min_obs=30)
    a = check_alpha(alpha)
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma <= 1e-12 * float(np.max(np.abs(r))):
        return float(-mu)  # constant series (up to rounding): no spread, so nothing to correct
    s = float(stats.skew(r, bias=False))
    k = float(stats.kurtosis(r, fisher=True, bias=False))
    z = float(stats.norm.ppf(a))
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    slope = 1 + 2 * z * s / 6 + (3 * z**2 - 3) * k / 24 - (6 * z**2 - 5) * s**2 / 36
    if slope <= 0:
        warnings.warn(
            f"the Cornish-Fisher quantile is not monotone at alpha={a} "
            f"(skew {s:.2f}, excess kurtosis {k:.2f}); the VaR is unreliable. Use historical_var.",
            UserWarning,
            stacklevel=2,
        )
    if k > 4.0:
        warnings.warn(
            f"excess kurtosis {k:.1f} is large; the Cornish-Fisher VaR may overstate tail losses. "
            "Compare with historical_var.",
            UserWarning,
            stacklevel=2,
        )
    return float(-(mu + z_cf * sigma))
