"""Portfolio risk measurement: VaR, expected shortfall, drawdowns, position sizing,
risk contributions and VaR backtesting.

Conventions
-----------
* Returns are simple per-period returns as decimals (0.01 = 1%), oldest first.
* Risk measures that describe losses (VaR, expected shortfall, max drawdown) are
  returned as **positive** numbers: 0.02 means a 2% loss.
* ``alpha`` is the tail probability: ``alpha=0.05`` is the 95% level.
"""

from .backtest import kupiec_pof_test, rolling_var_exceedances
from .drawdown import calmar_ratio, drawdown_series, max_drawdown, max_drawdown_duration, sortino_ratio
from .report import RiskReport, risk_report
from .risk import (
    equal_risk_contribution_weights,
    portfolio_volatility,
    risk_contributions,
    sample_covariance,
    shrink_covariance,
)
from .sizing import kelly_empirical, kelly_gaussian, volatility_target_leverage
from .var import (
    cornish_fisher_var,
    expected_shortfall,
    historical_var,
    parametric_expected_shortfall,
    parametric_var,
)

__version__ = "0.1.0"

__all__ = [
    "RiskReport",
    "__version__",
    "calmar_ratio",
    "cornish_fisher_var",
    "drawdown_series",
    "equal_risk_contribution_weights",
    "expected_shortfall",
    "historical_var",
    "kelly_empirical",
    "kelly_gaussian",
    "kupiec_pof_test",
    "max_drawdown",
    "max_drawdown_duration",
    "parametric_expected_shortfall",
    "parametric_var",
    "portfolio_volatility",
    "risk_contributions",
    "risk_report",
    "rolling_var_exceedances",
    "sample_covariance",
    "shrink_covariance",
    "sortino_ratio",
    "volatility_target_leverage",
]
