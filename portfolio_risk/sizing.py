"""Position sizing: Kelly fractions and volatility targeting."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ._validate import as_series, check_positive

__all__ = ["kelly_empirical", "kelly_gaussian", "volatility_target_leverage"]


def kelly_gaussian(mu: float, sigma: float, rf: float = 0.0) -> float:
    """Kelly leverage ``(mu - rf) / sigma^2`` for a diffusion with drift ``mu``.

    This is the continuous-time Kelly fraction. For discrete simple returns the exact
    growth-optimal fraction is smaller (for mu=0.001, sigma=0.01 the second-order discrete
    value is about 9.90 against 10.0 here). Use ``kelly_empirical`` for a sample-based
    estimate. Full Kelly is very aggressive when ``mu`` is estimated with error.
    """
    s = check_positive(sigma, "sigma")
    if s < 1e-150:
        raise ValueError("sigma is too small to square without underflow")
    if not (np.isfinite(mu) and np.isfinite(rf)):
        raise ValueError("mu and rf must be finite")
    return float((mu - rf) / s**2)


def kelly_empirical(returns, max_leverage: float = 10.0) -> float:
    """Leverage that maximizes the average log growth of the sample.

    Searches ``f`` in ``[0, max_leverage]`` for the maximum of ``mean(log(1 + f * r))``,
    restricted to values where every ``1 + f * r`` stays positive. Returns 0 when the
    sample has no positive expected log growth. When the sample contains no losses,
    log growth keeps rising with leverage and the result is ``max_leverage``. That is the
    sample's own answer, and it changes sharply once a single loss is observed.
    """
    r = as_series(returns, min_obs=20)
    cap = check_positive(max_leverage, "max_leverage")
    worst = float(r.min())
    if worst >= 0:
        return float(cap)
    upper = min(cap, 0.999999 / -worst)

    def neg_growth(f: float) -> float:
        return -float(np.mean(np.log1p(f * r)))

    result = optimize.minimize_scalar(neg_growth, bounds=(0.0, upper), method="bounded")
    if -result.fun <= 0.0:
        return 0.0
    return float(result.x)


def volatility_target_leverage(asset_vol: float, target_vol: float, max_leverage: float = 1.0) -> float:
    """Leverage that scales an asset to ``target_vol``, capped at ``max_leverage``."""
    sigma = check_positive(asset_vol, "asset_vol")
    target = check_positive(target_vol, "target_vol")
    cap = check_positive(max_leverage, "max_leverage")
    return float(min(target / sigma, cap))
