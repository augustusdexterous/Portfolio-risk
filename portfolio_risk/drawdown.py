"""Drawdowns and drawdown-based ratios."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ._validate import as_series, check_positive

__all__ = ["calmar_ratio", "drawdown_series", "max_drawdown", "max_drawdown_duration", "sortino_ratio"]


def _wealth(returns) -> NDArray[np.float64]:
    r = as_series(returns, min_obs=1)
    if np.any(r <= -1.0):
        raise ValueError("simple returns must be greater than -100%; wealth cannot go negative")
    return np.concatenate([[1.0], np.cumprod(1.0 + r)])


def drawdown_series(returns) -> NDArray[np.float64]:
    """Fraction below the running peak at each point, starting from wealth 1. Values are <= 0."""
    wealth = _wealth(returns)
    peak = np.maximum.accumulate(wealth)
    return wealth / peak - 1.0


def max_drawdown(returns) -> float:
    """Largest peak-to-trough loss as a positive fraction (0.25 = a 25% fall)."""
    return float(-drawdown_series(returns).min())


def max_drawdown_duration(returns) -> int:
    """Longest run of consecutive periods spent below a previous peak."""
    # Tolerance: after an exact return to the prior peak, rounding leaves wealth/peak - 1
    # around -1e-16, which must not count as underwater.
    under = drawdown_series(returns) < -1e-12
    longest = current = 0
    for flag in under:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def sortino_ratio(returns, periods_per_year: float = 252.0, target: float = 0.0) -> float:
    """Annualized excess return over the downside deviation below ``target``."""
    r = as_series(returns, min_obs=2)
    ppy = check_positive(periods_per_year, "periods_per_year")
    downside = np.minimum(r - target, 0.0)
    downside_dev = math.sqrt(float(np.mean(downside**2)))
    if downside_dev == 0.0:
        raise ValueError("no downside observations; the Sortino ratio is undefined")
    return float((r.mean() - target) * ppy / (downside_dev * math.sqrt(ppy)))


def calmar_ratio(returns, periods_per_year: float = 252.0) -> float:
    """Compound annual growth rate divided by the maximum drawdown."""
    r = as_series(returns, min_obs=2)
    ppy = check_positive(periods_per_year, "periods_per_year")
    mdd = max_drawdown(r)
    if mdd == 0.0:
        raise ValueError("no drawdown in the sample; the Calmar ratio is undefined")
    if np.any(r <= -1.0):
        raise ValueError("a period return of -100% or worse makes growth undefined")
    log_growth = float(np.sum(np.log1p(r))) * ppy / r.size
    if log_growth > 700.0:
        raise ValueError("annualized growth overflows; periods_per_year is too large for this sample")
    cagr = float(np.expm1(log_growth))
    return float(cagr / mdd)
