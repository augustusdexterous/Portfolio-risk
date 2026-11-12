import numpy as np
import pytest
from scipy import stats

import portfolio_risk as pr


def test_historical_var_matches_empirical_quantile():
    r = np.linspace(-0.05, 0.05, 101)  # 101 evenly spaced returns
    assert pr.historical_var(r, alpha=0.05) == pytest.approx(0.045)
    # n=101, k=n*alpha=5.05: the 5 worst are counted fully, the 6th (-0.045) with weight 0.05
    expected = -(sum(-0.05 + 0.001 * i for i in range(5)) + 0.05 * (-0.045)) / 5.05
    assert pr.expected_shortfall(r, alpha=0.05) == pytest.approx(expected, rel=1e-12)


def test_expected_shortfall_fractional_tail_matches_acerbi_tasche_example():
    # n=21, alpha=0.05 gives k=1.05 tail observations: the worst (-0.10) fully and
    # the second worst (-0.02) with weight 0.05, divided by 1.05.
    r = np.array([-0.10, -0.02] + [0.01] * 19)
    assert pr.expected_shortfall(r, alpha=0.05) == pytest.approx((0.10 + 0.05 * 0.02) / 1.05, rel=1e-12)


def test_parametric_var_closed_form():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.02, 5000)
    sigma = r.std(ddof=1)
    mu = r.mean()
    assert pr.parametric_var(r, 0.05) == pytest.approx(-(mu - stats.norm.ppf(0.95) * sigma))
    assert pr.parametric_var(r, 0.01) > pr.parametric_var(r, 0.05)


def test_parametric_es_closed_form_for_normal():
    sigma = 0.01
    r = np.random.default_rng(1).normal(0.0, sigma, 200_000)
    expected = sigma * stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05
    assert pr.parametric_expected_shortfall(r, 0.05) == pytest.approx(expected, rel=0.02)


def test_historical_es_is_at_least_var():
    r = np.random.default_rng(2).standard_t(4, 1000) * 0.01
    for alpha in (0.01, 0.05):
        assert pr.expected_shortfall(r, alpha) >= pr.historical_var(r, alpha)


def test_cornish_fisher_reduces_to_parametric_for_normal_sample():
    r = np.random.default_rng(3).normal(0.0, 0.01, 50_000)
    assert pr.cornish_fisher_var(r, 0.05) == pytest.approx(pr.parametric_var(r, 0.05), rel=0.03)


def test_cornish_fisher_exceeds_parametric_for_negative_skew():
    # negated gamma: long left tail, negative skew, moderate excess kurtosis (no warning)
    rng = np.random.default_rng(4)
    r = -(rng.gamma(2.0, 1.0, 20_000) - 2.0) * 0.01
    assert float(stats.skew(r)) < -0.5
    assert pr.cornish_fisher_var(r, 0.01) > pr.parametric_var(r, 0.01)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_alpha_must_be_inside_unit_interval(bad):
    with pytest.raises(ValueError):
        pr.historical_var(np.zeros(50), alpha=bad)


def test_series_validation():
    with pytest.raises(ValueError):
        pr.historical_var([0.01, 0.02])  # too short
    with pytest.raises(ValueError):
        pr.parametric_var(np.r_[np.zeros(30), np.nan])
    with pytest.raises(ValueError):
        pr.parametric_var(np.zeros((30, 2)))


def test_cornish_fisher_warns_for_large_excess_kurtosis():
    r = (
        np.random.default_rng(12).standard_t(3, 3000) * 0.01
    )  # excess kurtosis is infinite in theory, large in practice
    with pytest.warns(UserWarning, match="excess kurtosis"):
        pr.cornish_fisher_var(r, 0.01)


def test_cornish_fisher_silent_for_moderate_kurtosis():
    import warnings

    r = np.random.default_rng(13).normal(0.0, 0.01, 3000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pr.cornish_fisher_var(r, 0.05)


def test_cornish_fisher_warns_when_the_adjusted_quantile_is_not_monotone():
    r = np.random.default_rng(14).exponential(0.01, 5000)  # skew about 2, excess kurtosis about 6
    with pytest.warns(UserWarning) as record:
        pr.cornish_fisher_var(r, 0.01)
    assert any("not monotone" in str(w.message) for w in record)
