"""VaR backtesting: exceedance counts and the Kupiec proportion-of-failures test."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from ._validate import as_series, check_alpha
from .var import historical_var, parametric_var

__all__ = ["kupiec_pof_test", "rolling_var_exceedances"]


def rolling_var_exceedances(
    returns,
    window: int = 252,
    alpha: float = 0.01,
    method: str = "historical",
) -> NDArray[np.bool_]:
    """For each day after the first ``window``, did the loss exceed the VaR forecast from
    the previous ``window`` days? Returns a boolean array of length ``T - window``.
    """
    r = as_series(returns, min_obs=window + 1)
    a = check_alpha(alpha)
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise TypeError(f"window must be an integer, got {window!r}")
    if window < 20:
        raise ValueError("window must be at least 20 observations")
    estimators = {"historical": historical_var, "parametric": parametric_var}
    if method not in estimators:
        raise ValueError(f"method must be one of {sorted(estimators)}")
    var_fn = estimators[method]
    out = np.empty(r.size - window, dtype=bool)
    for i, t in enumerate(range(window, r.size)):
        forecast = var_fn(r[t - window : t], alpha=a)
        out[i] = -r[t] > forecast
    return out


def kupiec_pof_test(exceedances, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) likelihood-ratio test that the exceedance rate equals ``alpha``.

    Returns ``(statistic, p_value)``. Small p-values reject correct coverage. The test
    has low power on short samples (fewer than a few hundred observations).
    """
    raw = np.asarray(exceedances)
    if raw.ndim != 1:
        raise ValueError("exceedances must be one-dimensional")
    if raw.dtype != bool and not np.all(np.isin(raw, [0, 1])):
        raise ValueError("exceedances must be booleans or 0/1 indicators")
    x = raw.astype(bool)
    a = check_alpha(alpha)
    n = x.size
    if n == 0:
        raise ValueError("no observations to test")
    k = int(x.sum())
    p_hat = k / n

    def log_lik(p: float) -> float:
        ll = 0.0
        if n - k > 0:
            ll += (n - k) * math.log(1.0 - p)
        if k > 0:
            ll += k * math.log(p)
        return ll

    if p_hat in (0.0, 1.0):
        lr = -2.0 * log_lik(a)
    else:
        lr = -2.0 * (log_lik(a) - log_lik(p_hat))
    lr = max(lr, 0.0)
    p_value = float(stats.chi2.sf(lr, df=1))
    return float(lr), p_value
