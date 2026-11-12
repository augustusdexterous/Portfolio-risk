# Launch posts (drafts)

Posting is manual. Check each community's rules on self-promotion before posting, and reply to comments yourself.

## Show HN

**Title:** Show HN: portfolio-risk, VaR, expected shortfall and VaR backtesting in Python

**Text:**
I wrote a small Python library for portfolio risk measures. It has historical, parametric and Cornish-Fisher VaR, expected shortfall, drawdowns, Kelly and volatility-target sizing, covariance shrinkage, equal-risk-contribution weights, and a rolling VaR backtest with the Kupiec test. There is a command-line report too: `portfolio-risk returns.csv`.

I tested it on S&P 500 daily returns from 1999 to 2018. A 1% VaR forecast from the previous 250 days was exceeded 1.7% of the time with historical estimates and 2.4% with parametric ones. The Kupiec test rejects correct coverage for both. The reason is that losses cluster in crises, so the window that precedes a shock is too calm. The README and the example script show the numbers. I think results like that are worth seeing, so I didn't tune the example to look better.

Limits: parametric measures assume normal returns, historical measures are rough at 1% with short windows, and costs and execution are not modelled.

Repo: https://github.com/augustusdexterous/portfolio-risk

## Reddit (r/quant or r/algotrading, check the rules first)

**Title:** A small open-source library for VaR, expected shortfall and VaR backtesting, tested on S&P 500 data

**Text:**
Sharing a Python library I built for portfolio risk: historical, parametric and Cornish-Fisher VaR, expected shortfall (with the Acerbi-Tasche estimator for non-integer tail counts), drawdowns, Kelly and volatility-target sizing, shrinkage covariance, and equal-risk-contribution weights. It also has a rolling VaR backtest with Kupiec's test.

The S&P 500 example (1999–2018) shows why I think backtesting VaR matters: both 1% forecasts were exceeded more often than they should be, and Kupiec rejects them. Feedback on the formulas and on the defaults is very welcome, especially from people who use these measures in practice.

Repo: https://github.com/augustusdexterous/portfolio-risk

## Short post (LinkedIn or X)

I published portfolio-risk, an open-source Python library for VaR, expected shortfall, drawdowns and VaR backtesting. The S&P 500 example shows that a 250-day historical 1% VaR was exceeded 1.7% of the time, and the Kupiec test rejects it. Code, tests and limits are in the README: https://github.com/augustusdexterous/portfolio-risk

---

# Notes for the maintainer

- The README's CI badge points to the workflow on `main`.
- Before posting, check that CI is green on the latest commit.
