"""Each test recomputes a quantity by a different route than the library and compares.

These are not snapshots of the library's own output: they use direct formulas, numerical
integration, brute-force search, or a sample large enough for a Monte Carlo comparison.
"""

import math

import numpy as np
import pytest
from scipy import integrate, stats

import portfolio_risk as pr


def test_historical_var_equals_linear_interpolated_quantile():
    r = np.random.default_rng(42).standard_t(5, 400) * 0.01
    q = np.sort(r)
    pos = 0.05 * (len(q) - 1)
    lo = math.floor(pos)
    q05 = q[lo] + (pos - lo) * (q[lo + 1] - q[lo])
    assert pr.historical_var(r, 0.05) == pytest.approx(-q05, rel=1e-12)


def test_parametric_es_matches_numerical_integral_of_normal_tail():
    mu, sd, alpha = 0.0003, 0.012, 0.05
    z = stats.norm.ppf(alpha)
    tail_mean, _ = integrate.quad(lambda x: x * stats.norm.pdf(x, mu, sd), -np.inf, mu + z * sd)
    expected = -tail_mean / alpha
    sample = np.random.default_rng(42).normal(mu, sd, 2_000_000)
    assert pr.parametric_expected_shortfall(sample, alpha) == pytest.approx(expected, rel=1e-3)


def test_cornish_fisher_matches_direct_expansion_for_skewed_sample():
    rng = np.random.default_rng(4)
    r = -(rng.gamma(2.0, 1.0, 20_000) - 2.0) * 0.01
    s = float(stats.skew(r, bias=False))
    k = float(stats.kurtosis(r, fisher=True, bias=False))
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    z = stats.norm.ppf(0.05)
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    assert pr.cornish_fisher_var(r, 0.05) == pytest.approx(-(mu + z_cf * sd), rel=1e-12)


def test_kupiec_statistic_matches_direct_formula():
    n, k, a = 1000, 17, 0.01
    flags = np.zeros(n, dtype=bool)
    flags[:k] = True
    direct = -2 * ((n - k) * math.log(1 - a) + k * math.log(a)) + 2 * (
        (n - k) * math.log(1 - k / n) + k * math.log(k / n)
    )
    lr, p = pr.kupiec_pof_test(flags, a)
    assert lr == pytest.approx(direct, rel=1e-12)
    assert p == pytest.approx(stats.chi2.sf(direct, 1), rel=1e-12)


def test_max_drawdown_matches_brute_force_search():
    r = np.random.default_rng(3).normal(0.0002, 0.01, 800)
    wealth = np.concatenate([[1.0], np.cumprod(1 + r)])
    brute = max((np.max(wealth[: i + 1]) - wealth[i]) / np.max(wealth[: i + 1]) for i in range(wealth.size))
    assert pr.max_drawdown(r) == pytest.approx(brute, rel=1e-12)


def test_equal_risk_contributions_are_equal_on_a_random_correlated_matrix():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(5, 5))
    cov = a @ a.T / 5 + 0.1 * np.eye(5)
    w = pr.equal_risk_contribution_weights(cov)
    shares = pr.risk_contributions(w, cov)
    np.testing.assert_allclose(shares, np.full(5, 0.2), atol=1e-6)
    assert np.all(w > 0)


def test_kelly_empirical_matches_brute_force_grid_maximum():
    r = np.random.default_rng(6).normal(0.001, 0.02, 300)
    feasible_max = 0.999999 / -r.min()  # leverage keeping every 1 + f*r positive
    grid = np.linspace(0.0, feasible_max, 20001)
    growth = np.array([np.mean(np.log1p(f * r)) for f in grid])
    brute = grid[int(np.argmax(growth))]
    assert pr.kelly_empirical(r, max_leverage=1000.0) == pytest.approx(brute, rel=1e-3)


def test_kelly_gaussian_agrees_with_empirical_on_large_normal_sample():
    r = np.random.default_rng(7).normal(0.0005, 0.01, 300_000)
    assert pr.kelly_empirical(r, max_leverage=50) == pytest.approx(0.0005 / 0.01**2, rel=0.05)


def test_monte_carlo_tail_probability_of_parametric_var_on_normal_data():
    rng = np.random.default_rng(8)
    sample = rng.normal(0.0, 0.01, 1_000_000)
    v = pr.parametric_var(sample[:500_000], alpha=0.05)
    assert float(np.mean(sample[500_000:] < -v)) == pytest.approx(0.05, abs=0.002)


def test_equal_risk_contribution_rejects_singular_covariance_clearly():
    v = np.array([[1.0], [-1.0], [0.5]])
    singular = v @ v.T + 1e-10 * np.eye(3)  # rank 1 plus a tiny ridge
    with pytest.raises(ValueError, match="singular"):
        pr.equal_risk_contribution_weights(singular)


def test_equal_risk_contribution_accepts_well_conditioned_extreme_scales():
    cov = np.diag([1e-12, 1e-2, 1.0, 4.0])
    w = pr.equal_risk_contribution_weights(cov)
    shares = pr.risk_contributions(w, cov)
    np.testing.assert_allclose(shares, np.full(4, 0.25), atol=1e-6)
