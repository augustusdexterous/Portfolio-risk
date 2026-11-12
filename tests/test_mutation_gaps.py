"""Targeted tests for behaviour that earlier tests did not pin down.

Each test is aimed at a specific fault that the suite previously let through (found by
injecting faults into the code and checking which tests noticed).
"""

import numpy as np
import pytest
from scipy import stats

import portfolio_risk as pr
from portfolio_risk import risk as risk_module
from portfolio_risk.cli import main


def test_rolling_forecast_uses_only_past_returns_no_look_ahead():
    r = np.zeros(400)
    r[300] = -0.5  # a single loss on day 300
    flags = pr.rolling_var_exceedances(r, window=250, alpha=0.05, method="historical")
    # index 0 of flags is day 250; the loss on day 300 must be flagged, because the forecast
    # for day 300 uses only days 50..299, which are all zero.
    assert flags[300 - 250]
    assert flags.sum() == 1


def test_rolling_forecast_slice_is_exactly_the_prior_window():
    rng = np.random.default_rng(31)
    r = rng.normal(0, 0.01, 300)
    flags = pr.rolling_var_exceedances(r, window=100, alpha=0.05, method="parametric")
    expected = [(-r[t]) > pr.parametric_var(r[t - 100 : t], 0.05) for t in range(100, 300)]
    assert np.array_equal(flags, np.array(expected))


def test_rolling_parametric_method_is_implemented_separately_from_historical():
    rng = np.random.default_rng(4)
    r = -(rng.gamma(2.0, 1.0, 1500) - 2.0) * 0.01  # skewed, so the two methods differ
    hist = pr.rolling_var_exceedances(r, window=250, alpha=0.05, method="historical")
    par = pr.rolling_var_exceedances(r, window=250, alpha=0.05, method="parametric")
    assert not np.array_equal(hist, par)


def test_var_level_test_detects_a_forecast_at_the_wrong_alpha():
    rates = []
    for seed in range(10):
        r = np.random.default_rng(200 + seed).normal(0, 0.01, 1500)
        rates.append(pr.rolling_var_exceedances(r, window=250, alpha=0.05).mean())
    assert float(np.mean(rates)) > 0.045  # a 4% forecast would average about 0.04


def test_kupiec_zero_exceedance_branch_matches_closed_form():
    lr, _ = pr.kupiec_pof_test(np.zeros(500, dtype=bool), 0.01)
    assert lr == pytest.approx(-2 * 500 * np.log(0.99), rel=1e-12)


def test_kupiec_all_exceedance_branch_matches_closed_form():
    lr, _ = pr.kupiec_pof_test(np.ones(50, dtype=bool), 0.1)
    assert lr == pytest.approx(-2 * 50 * np.log(0.1), rel=1e-12)


def test_shrinkage_target_uses_the_average_off_diagonal_sample_correlation():
    rng = np.random.default_rng(5)
    r = rng.multivariate_normal(
        np.zeros(4),
        np.array([[1, 0.5, 0.2, 0.1], [0.5, 1, 0.3, 0.0], [0.2, 0.3, 1, 0.4], [0.1, 0.0, 0.4, 1]]) * 1e-4,
        size=200,
    )
    corr = np.corrcoef(r, rowvar=False)
    rbar = corr[~np.eye(4, dtype=bool)].mean()
    shrunk = pr.shrink_covariance(r, delta=1.0)
    sd = np.sqrt(np.diag(shrunk))
    implied = shrunk / np.outer(sd, sd)
    assert implied[0, 1] == pytest.approx(rbar, rel=1e-10)
    assert implied[2, 3] == pytest.approx(rbar, rel=1e-10)


def test_shrinkage_with_delta_zero_is_the_sample_covariance():
    rng = np.random.default_rng(6)
    r = rng.normal(0, 0.01, (100, 3))
    np.testing.assert_allclose(pr.shrink_covariance(r, 0.0), np.cov(r, rowvar=False), rtol=1e-12)


def test_shrinkage_rejects_price_like_input():
    with pytest.raises(ValueError, match="decimals"):
        pr.shrink_covariance(np.linspace(100, 110, 60).reshape(30, 2))


def test_risk_report_fields_match_their_direct_definitions():
    r = np.random.default_rng(9).normal(0.0004, 0.01, 1000)
    rep = pr.risk_report(r, alpha=0.05, periods_per_year=252)
    assert rep.annual_return == pytest.approx(r.mean() * 252)
    assert rep.annual_volatility == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert rep.sharpe == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert rep.historical_var == pytest.approx(pr.historical_var(r, 0.05))
    assert rep.parametric_var == pytest.approx(pr.parametric_var(r, 0.05))
    assert rep.cornish_fisher_var == pytest.approx(pr.cornish_fisher_var(r, 0.05))
    assert rep.historical_es == pytest.approx(pr.expected_shortfall(r, 0.05))
    assert rep.parametric_es == pytest.approx(pr.parametric_expected_shortfall(r, 0.05))
    assert rep.max_drawdown == pytest.approx(pr.max_drawdown(r))
    assert rep.max_drawdown_duration == pr.max_drawdown_duration(r)


def test_parametric_es_uses_sample_standard_deviation_with_ddof_one():
    r = np.random.default_rng(10).normal(0.0, 0.01, 30)  # small sample, so ddof matters
    mu = r.mean()
    sigma = r.std(ddof=1)
    z = stats.norm.ppf(0.05)
    expected = sigma * stats.norm.pdf(z) / 0.05 - mu
    assert pr.parametric_expected_shortfall(r, 0.05) == pytest.approx(expected, rel=1e-12)


def test_historical_es_integer_tail_boundary():
    r = np.array([-0.05] * 3 + [-0.02] * 3 + [0.0] * 88 + [0.01] * 6)  # k = 5 exactly
    assert pr.expected_shortfall(r, 0.05) == pytest.approx((3 * 0.05 + 2 * 0.02) / 5, rel=1e-12)


def test_drawdown_duration_is_the_longest_run_not_the_total():
    r = np.array([-0.1, 0.0, 0.2, -0.1, -0.1, -0.1, 0.5, 0.1, -0.05, 0.2])
    assert pr.max_drawdown_duration(r) == 3


def test_sortino_uses_the_target_in_the_downside_deviation():
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.0, 0.015, -0.005])
    target = 0.01
    downside = np.minimum(r - target, 0.0)
    expected = (r.mean() - target) * 252 / (np.sqrt(np.mean(downside**2)) * np.sqrt(252))
    assert pr.sortino_ratio(r, 252, target=target) == pytest.approx(expected, rel=1e-12)


def test_equal_risk_contribution_share_check_raises_when_solver_is_wrong(monkeypatch):
    class _Result:
        x = np.array([0.9, 0.05, 0.05])
        message = "forced"

    monkeypatch.setattr(risk_module.optimize, "minimize", lambda *a, **k: _Result())
    cov = np.diag([0.01, 0.04, 0.09])
    with pytest.raises(RuntimeError, match="could not compute"):
        pr.equal_risk_contribution_weights(cov)


def test_equal_risk_contribution_rejects_non_positive_variance():
    with pytest.raises(ValueError, match="strictly positive variances"):
        pr.equal_risk_contribution_weights(np.array([[0.0, 0.0], [0.0, 0.04]]))


def test_risk_returns_matrix_rejects_price_like_input_everywhere():
    with pytest.raises(ValueError, match="decimals"):
        pr.sample_covariance(np.linspace(100, 110, 60).reshape(30, 2))


def test_portfolio_volatility_and_risk_shares_for_a_known_matrix():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.6, 0.4])
    sigma = np.sqrt(w @ cov @ w)
    marginal = cov @ w / sigma
    expected_shares = w * marginal / sigma
    assert pr.portfolio_volatility(w, cov) == pytest.approx(sigma)
    np.testing.assert_allclose(pr.risk_contributions(w, cov), expected_shares)


def test_cli_alpha_changes_the_reported_level(tmp_path, capsys):
    p = tmp_path / "a.csv"
    rng = np.random.default_rng(12)
    rows = [[f"{x:.6f}", f"{y:.6f}"] for x, y in rng.normal(0, 0.01, (120, 2))]
    with open(p, "w") as fh:
        fh.write("a,b\n" + "\n".join(f"{x},{y}" for x, y in rows) + "\n")
    assert main([str(p), "--alpha", "0.01"]) == 0
    assert "VaR 99% historical" in capsys.readouterr().out


def test_cli_periods_per_year_is_used_and_validated(tmp_path, capsys):
    import re

    p = tmp_path / "a.csv"
    rng = np.random.default_rng(13)
    data = rng.normal(0, 0.01, (120, 2))
    with open(p, "w") as fh:
        fh.write("a,b\n" + "\n".join(f"{x:.6f},{y:.6f}" for x, y in data) + "\n")
    assert main([str(p), "--periods-per-year", "12"]) == 0
    out = capsys.readouterr().out
    reported = float(re.search(r"annual volatility\s*: ([-\d.]+)%", out).group(1))
    sd_a = np.std(np.round(data[:, 0], 6), ddof=1)
    assert reported == pytest.approx(sd_a * np.sqrt(12) * 100, abs=0.01)
    assert main([str(p), "--periods-per-year", "0"]) == 2
    assert capsys.readouterr().out == ""


def test_cli_shrinkage_changes_weights_and_weights_are_percentages(tmp_path, capsys):
    p = tmp_path / "a.csv"
    rng = np.random.default_rng(14)
    with open(p, "w") as fh:
        fh.write(
            "a,b,c\n"
            + "\n".join(
                f"{x:.6f},{y:.6f},{z:.6f}" for x, y, z in rng.normal(0, [0.01, 0.02, 0.005], (120, 3))
            )
            + "\n"
        )
    assert main([str(p), "--shrinkage", "0"]) == 0
    block0 = capsys.readouterr().out.split("== equal risk contribution")[1].splitlines()[1:]
    assert main([str(p), "--shrinkage", "1"]) == 0
    block1 = capsys.readouterr().out.split("== equal risk contribution")[1].splitlines()[1:]
    assert block0 != block1  # the weights themselves must change, not only the header
    assert all("%" in line for line in block0 if line.strip())


def test_cli_blank_line_in_the_middle_is_an_error_not_a_silent_drop(tmp_path, capsys):
    p = tmp_path / "gap.csv"
    rng = np.random.default_rng(15)
    lines = ["a,b"] + [f"{x:.6f},{y:.6f}" for x, y in rng.normal(0, 0.01, (60, 2))]
    lines.insert(30, "")
    p.write_text("\n".join(lines) + "\n")
    assert main([str(p)]) == 2
    assert "blank line" in capsys.readouterr().err


def test_cli_blank_lines_at_the_ends_are_allowed(tmp_path):
    from portfolio_risk.cli import load_assets

    p = tmp_path / "ends.csv"
    rng = np.random.default_rng(16)
    lines = ["", "a,b"] + [f"{x:.6f},{y:.6f}" for x, y in rng.normal(0, 0.01, (60, 2))] + ["", ""]
    p.write_text("\n".join(lines) + "\n")
    names, matrix = load_assets(p)
    assert names == ["a", "b"] and matrix.shape == (60, 2)


def test_each_forecast_receives_exactly_the_prior_window(monkeypatch):
    from portfolio_risk import backtest

    rng = np.random.default_rng(17)
    r = rng.normal(0, 0.01, 320)
    window = 100
    received = []
    real = backtest.historical_var

    def recording(sample, alpha=0.05):
        received.append(np.array(sample, copy=True))
        return real(sample, alpha=alpha)

    monkeypatch.setattr(backtest, "historical_var", recording)
    pr.rolling_var_exceedances(r, window=window, alpha=0.05, method="historical")
    assert len(received) == r.size - window
    for i, sample in enumerate(received):
        t = window + i
        np.testing.assert_array_equal(sample, r[t - window : t])
