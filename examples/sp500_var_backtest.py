"""Backtest 1-day 99% VaR on S&P 500 daily returns, 1999-2018.

Needs the optional package `arch` (ships the series; no network access needed):
    pip install arch
    python examples/sp500_var_backtest.py
"""

import sys

import portfolio_risk as pr


def load_returns():
    try:
        from arch.data import sp500
    except ImportError:
        sys.exit("This example needs the optional package: pip install arch")
    prices = sp500.load()["Adj Close"].to_numpy(dtype=float)
    return prices[1:] / prices[:-1] - 1


if __name__ == "__main__":
    r = load_returns()
    print(f"S&P 500 daily simple returns: {r.size} days, 1999-01-05 to 2018-12-31\n")
    print(pr.risk_report(r, alpha=0.01).summary())
    print()
    window, alpha = 250, 0.01
    header = (
        f"{'method':<12} {'exceedances':>12} {'rate':>8} {'expected':>9} {'Kupiec LR':>10} {'p-value':>9}"
    )
    print(header)
    print("-" * len(header))
    for method in ("historical", "parametric"):
        flags = pr.rolling_var_exceedances(r, window=window, alpha=alpha, method=method)
        lr, p = pr.kupiec_pof_test(flags, alpha)
        print(f"{method:<12} {int(flags.sum()):>12} {flags.mean():>8.4f} {alpha:>9.4f} {lr:>10.2f} {p:>9.4f}")
    print(
        f"\nrolling window: {window} days; exceedance = loss larger than the VaR forecast made from the prior window"
    )
