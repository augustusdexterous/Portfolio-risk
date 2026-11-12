"""One-call risk summary for a single return series."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from ._validate import as_series, check_alpha, check_positive
from .drawdown import calmar_ratio, max_drawdown, max_drawdown_duration, sortino_ratio
from .var import (
    cornish_fisher_var,
    expected_shortfall,
    historical_var,
    parametric_expected_shortfall,
    parametric_var,
)

__all__ = ["RiskReport", "risk_report"]


def _or_nan(fn) -> float:
    try:
        return float(fn())
    except ValueError:
        return float("nan")


def _level(alpha: float) -> str:
    """Confidence level label, e.g. '95%'; falls back to the alpha value when the level rounds to 100%."""
    text = f"{100 * (1 - alpha):.6g}"
    if float(text) >= 100:
        return f"alpha={alpha:g}"
    return f"{text}%"


def _fmt(value: float, pct: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%" if pct else f"{value:.2f}"


@dataclass(frozen=True)
class RiskReport:
    n_obs: int
    alpha: float
    periods_per_year: float
    annual_return: float  # arithmetic mean, annualized
    annual_volatility: float
    sharpe: float  # rf = 0
    sortino: float
    calmar: float
    historical_var: float
    parametric_var: float
    cornish_fisher_var: float
    historical_es: float
    parametric_es: float
    max_drawdown: float
    max_drawdown_duration: int
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        level = _level(self.alpha)
        rows = [
            ("observations", f"{self.n_obs}"),
            ("annual return (mean)", _fmt(self.annual_return, pct=True)),
            ("annual volatility", _fmt(self.annual_volatility, pct=True)),
            ("Sharpe (rf=0)", _fmt(self.sharpe)),
            ("Sortino", _fmt(self.sortino)),
            ("Calmar", _fmt(self.calmar)),
            (f"VaR {level} historical", _fmt(self.historical_var, pct=True)),
            (f"VaR {level} parametric", _fmt(self.parametric_var, pct=True)),
            (f"VaR {level} Cornish-Fisher", _fmt(self.cornish_fisher_var, pct=True)),
            (f"ES  {level} historical", _fmt(self.historical_es, pct=True)),
            (f"ES  {level} parametric", _fmt(self.parametric_es, pct=True)),
            ("max drawdown", _fmt(self.max_drawdown, pct=True)),
            ("longest drawdown (periods)", f"{self.max_drawdown_duration}"),
        ]
        width = max(len(k) for k, _ in rows)
        lines = [f"{k:<{width}} : {v}" for k, v in rows]
        lines += [f"note: {note}" for note in self.notes]
        return "\n".join(lines)


def risk_report(returns, alpha: float = 0.05, periods_per_year: float = 252.0) -> RiskReport:
    """Compute the standard risk measures for one return series.

    Values that are undefined for the sample (for example Sortino with no downside
    observations) are reported as NaN rather than raising. Cornish-Fisher warnings are
    returned in ``notes`` instead of being emitted, so they appear in the summary.
    """
    r = as_series(returns, min_obs=30)
    a = check_alpha(alpha)
    ppy = check_positive(periods_per_year, "periods_per_year")
    mean = float(r.mean())
    sd = float(r.std(ddof=1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cf = cornish_fisher_var(r, a)
    notes = tuple(str(w.message) for w in caught if issubclass(w.category, UserWarning))
    return RiskReport(
        n_obs=int(r.size),
        alpha=a,
        periods_per_year=ppy,
        annual_return=mean * ppy,
        annual_volatility=sd * math.sqrt(ppy),
        sharpe=(mean / sd * math.sqrt(ppy)) if sd > 0 else float("nan"),
        sortino=_or_nan(lambda: sortino_ratio(r, ppy)),
        calmar=_or_nan(lambda: calmar_ratio(r, ppy)),
        historical_var=historical_var(r, a),
        parametric_var=parametric_var(r, a),
        cornish_fisher_var=cf,
        historical_es=expected_shortfall(r, a),
        parametric_es=parametric_expected_shortfall(r, a),
        max_drawdown=max_drawdown(r),
        max_drawdown_duration=max_drawdown_duration(r),
        notes=notes,
    )
