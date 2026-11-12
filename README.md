# portfolio-risk

[![CI](https://github.com/augustusdexterous/portfolio-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/augustusdexterous/portfolio-risk/actions/workflows/ci.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

Portfolio risk measures in Python: historical and parametric Value at Risk, expected shortfall, drawdowns, Kelly and volatility-target sizing, risk contributions and equal-risk-contribution weights, and a VaR backtest with the Kupiec test. It comes with a command-line report and a test suite that checks the formulas against independent calculations.

```
$ portfolio-risk examples/demo_assets.csv
== equities
observations               : 504
annual volatility          : 19.12%
VaR 95% historical         : 1.98%
VaR 95% Cornish-Fisher     : 2.12%
ES  95% historical         : 2.80%
max drawdown               : 36.63%
...
note: excess kurtosis 4.1 is large; the Cornish-Fisher VaR may overstate tail losses.
== equal risk contribution weights (covariance shrinkage 0.5)
equities                 weight  18.07%   risk share  33.33%
bonds                    weight  24.06%   risk share  33.33%
gold                     weight  57.87%   risk share  33.33%
```

## Install

```bash
pip install git+https://github.com/augustusdexterous/portfolio-risk.git
```

Requires Python 3.9+, NumPy and SciPy.

## Conventions

- Returns are simple per-period returns as decimals: `0.01` means 1%. Oldest first.
- Loss measures (VaR, expected shortfall, max drawdown) are **positive** numbers: `0.02` means a 2% loss.
- `alpha` is the tail probability: `alpha=0.05` is the 95% level.
- Inputs that look like prices or percentages are rejected with an error (any value of 10 or more, or an average magnitude above 1). Non-finite values are rejected everywhere.
- Simple returns at or below -100% are rejected, because wealth cannot go negative.

## Quick start

```python
import portfolio_risk as pr

rep = pr.risk_report(daily_returns, alpha=0.01, periods_per_year=252)
print(rep.summary())

pr.historical_var(daily_returns, alpha=0.01)
pr.expected_shortfall(daily_returns, alpha=0.01)
pr.max_drawdown(daily_returns)

# VaR backtest: is a 1% VaR forecast from the previous 250 days calibrated?
flags = pr.rolling_var_exceedances(daily_returns, window=250, alpha=0.01, method="historical")
lr, p_value = pr.kupiec_pof_test(flags, alpha=0.01)

# Risk-based weights for several assets (rows are periods, columns are assets)
cov = pr.shrink_covariance(asset_returns, delta=0.5)
weights = pr.equal_risk_contribution_weights(cov)
pr.risk_contributions(weights, cov)  # each share is 1/N
```

## What is included

| Area | Functions |
|---|---|
| Value at Risk | `historical_var`, `parametric_var`, `cornish_fisher_var` |
| Expected shortfall | `expected_shortfall` (historical), `parametric_expected_shortfall` |
| Drawdowns | `drawdown_series`, `max_drawdown`, `max_drawdown_duration`, `calmar_ratio`, `sortino_ratio` |
| Position sizing | `kelly_gaussian`, `kelly_empirical`, `volatility_target_leverage` |
| Risk and weights | `sample_covariance`, `shrink_covariance`, `portfolio_volatility`, `risk_contributions`, `equal_risk_contribution_weights` |
| Backtesting | `rolling_var_exceedances`, `kupiec_pof_test` |
| Summary | `risk_report` returns a `RiskReport` with every standard measure |

## Real data: S&P 500, 1999–2018

`python examples/sp500_var_backtest.py` (needs `pip install arch`, which ships the adjusted-close series). 5,030 daily returns, 250-day rolling window, 1% VaR.

| Method | Exceedances | Rate | Expected | Kupiec LR | p-value |
|---|---|---|---|---|---|
| Historical | 81 | 1.69% | 1.00% | 19.28 | < 0.0001 |
| Parametric | 116 | 2.43% | 1.00% | 70.27 | < 0.0001 |

Both forecasts are too low, and the Kupiec test rejects correct coverage for both. The reason is visible in the data: losses cluster in crises, and a window that has just seen a calm period underestimates the next shock. The parametric forecast is also hurt by fat tails. This is the kind of result the backtest is for, and it is why the library reports coverage tests rather than only the VaR number.

On the same data, the 1% Cornish-Fisher VaR is 5.14%, against 3.31% historical. Excess kurtosis is about 8, and the expansion over-corrects in the far tail. The function warns when excess kurtosis exceeds 4.

## Calibration on simulated normal returns

200 simulated series of 1,500 normal daily returns each, 250-day rolling window (`rolling_var_exceedances`), Kupiec test at 5%:

| Method | alpha | Mean exceedance rate | Expected | Kupiec rejects |
|---|---|---|---|---|
| Historical | 5% | 5.39% (sd 0.31%) | 5.00% | 0 of 200 (0.0%) |
| Parametric | 5% | 5.10% (sd 0.48%) | 5.00% | 1 of 200 (0.5%) |
| Historical | 1% | 1.37% (sd 0.18%) | 1.00% | 32 of 200 (16.0%) |
| Parametric | 1% | 1.06% (sd 0.25%) | 1.00% | 6 of 200 (3.0%) |

Even on data that matches the parametric assumptions, a 250-day historical window over-exceeds its nominal rate: estimating a 1% quantile from 250 observations uses only about 2–3 tail points, and the test rejects the historical 1% forecast in 16% of runs. The parametric 1% forecast is close to nominal. The S&P 500 result above is a different and larger problem: real returns violate the independence assumption behind the backtest.

## Limits

- Parametric measures assume normal returns. Use them as a comparison, not as the answer, for fat-tailed assets.
- Historical measures need enough data in the tail. At 1% with a 250-day window there are only about 2–3 observations past the quantile, so treat 1% estimates from short windows as rough.
- Kupiec's test has low power on short samples.
- Covariance shrinkage uses a fixed intensity, not the analytically optimal Ledoit-Wolf value. Report the intensity you use.
- The library measures risk. It does not model costs, execution, or margin.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
ruff check . && ruff format --check .
```

CI runs the tests on Python 3.9 to 3.13, the command-line report on the demo file, and lint.

## References

- Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4).
- Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2).
- Ledoit, O., & Wolf, M. (2003). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management*, 30(4).
- Maillard, S., Roncalli, T., & Teïletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management*, 36(4).
- Sortino, F. A., & Price, L. N. (1994). Performance measurement in a downside risk framework. *Journal of Investing*, 3(3).
- Jorion, P. (2007). *Value at Risk* (3rd ed.). McGraw-Hill.

## License

MIT. See [LICENSE](LICENSE).
