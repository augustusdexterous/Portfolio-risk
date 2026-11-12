"""Regression tests for defects found in the second review. Each uses the failing input from its report."""

import warnings

import numpy as np
import pytest

import portfolio_risk as pr


def test_expected_shortfall_matches_discrete_estimator_for_three_losses_in_100():
    r = np.array([-0.05] * 3 + [0.0] * 90 + [0.01] * 7)  # k = 5 tail observations exactly
    assert pr.expected_shortfall(r, alpha=0.05) == pytest.approx(0.03, rel=1e-12)


def test_shrinkage_rejects_constant_column_with_rounding_noise():
    rng = np.random.default_rng(0)
    r = np.column_stack([np.full(60, 0.01), rng.normal(0, 0.01, 60), rng.normal(0, 0.01, 60)])
    with pytest.raises(ValueError, match="zero variance"):
        pr.shrink_covariance(r, delta=0.5)


def test_cornish_fisher_constant_series_with_rounding_noise_is_not_nan():
    r = np.full(30, 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no moment-precision warnings
        value = pr.cornish_fisher_var(r, 0.05)
    assert value == pytest.approx(-0.01)


def test_portfolio_volatility_accepts_rounding_on_a_singular_covariance():
    cov = [[0.01, 0.05], [0.05, 0.25]]  # rank 1, positive semi-definite
    assert pr.portfolio_volatility([1.5, -0.3], cov) == pytest.approx(0.0, abs=1e-12)
    rc_result = pr.risk_contributions([0.6, 0.4], [[0.04, 0.01], [0.01, 0.09]])
    assert rc_result.sum() == pytest.approx(1.0)


def test_portfolio_volatility_still_rejects_a_genuinely_indefinite_matrix():
    with pytest.raises(ValueError, match="not positive semi-definite"):
        pr.portfolio_volatility([1.0, -1.0], [[1.0, 2.0], [2.0, 1.0]])


def test_portfolio_volatility_handles_scalar_weight_for_one_asset():
    assert pr.risk_contributions(1.0, [[0.04]]) == pytest.approx(np.array([1.0]))


def test_nonfinite_covariance_is_rejected_with_a_clear_message():
    with pytest.raises(ValueError, match="NaN or infinite"):
        pr.portfolio_volatility([0.5, 0.5], [[np.nan, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        pr.risk_contributions([0.5, 0.5], [[np.inf, 0.0], [0.0, 1.0]])


def test_equal_risk_contributions_reject_asymmetric_tiny_matrix():
    with pytest.raises(ValueError, match="symmetric"):
        pr.equal_risk_contribution_weights(np.array([[1e-10, 3e-11], [0.0, 1e-10]]))


def test_equal_risk_contributions_are_scale_invariant():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3))
    base_cov = a @ a.T / 3 + 0.1 * np.eye(3)
    base = pr.equal_risk_contribution_weights(base_cov)
    for scale in (1e-18, 1e24):
        w = pr.equal_risk_contribution_weights(base_cov * scale)
        np.testing.assert_allclose(w, base, rtol=1e-8)


def test_equal_risk_contribution_empty_and_nan_inputs_are_clear():
    with pytest.raises(ValueError):
        pr.equal_risk_contribution_weights(np.zeros((0, 0)))
    with pytest.raises(ValueError, match="NaN or infinite"):
        pr.equal_risk_contribution_weights(np.array([[np.nan, 0.0], [0.0, 1.0]]))


def test_drawdown_rejects_returns_at_or_below_minus_one():
    with pytest.raises(ValueError, match="-100%"):
        pr.max_drawdown([0.1, -1.5, 0.2])
    with pytest.raises(ValueError, match="-100%"):
        pr.drawdown_series([0.1, -1.0])


def test_drawdown_duration_does_not_count_exact_return_to_peak():
    prices = np.array([101, 142, 69, 98, 74, 137, 63, 120, 142], dtype=float)
    returns = prices[1:] / prices[:-1] - 1
    assert pr.max_drawdown_duration(returns) == 6


def test_calmar_overflow_is_a_clear_error_and_nan_in_the_report():
    r = np.tile([0.01, -0.005], 150)  # positive growth with a real drawdown
    with pytest.raises(ValueError, match="overflows"):
        pr.calmar_ratio(r, periods_per_year=1e6)
    rep = pr.risk_report(r, periods_per_year=1e6)
    assert np.isnan(rep.calmar)


def test_kupiec_rejects_nan_and_non_indicator_values():
    with pytest.raises(ValueError):
        pr.kupiec_pof_test(np.array([np.nan, 0, 0, 0]), 0.25)
    with pytest.raises(ValueError):
        pr.kupiec_pof_test(np.array([-0.01, 0.005, 0.0, 0.002]), 0.25)


def test_kupiec_accepts_zero_one_integers():
    lr_bool, p_bool = pr.kupiec_pof_test(np.array([True, False, False, False]), 0.25)
    lr_int, p_int = pr.kupiec_pof_test(np.array([1, 0, 0, 0]), 0.25)
    assert (lr_bool, p_bool) == pytest.approx((lr_int, p_int))


def test_rolling_window_must_be_an_integer():
    with pytest.raises(TypeError, match="integer"):
        pr.rolling_var_exceedances(np.random.default_rng(0).normal(0, 0.01, 400), window=252.0)


def test_kelly_gaussian_rejects_sigma_that_underflows_when_squared():
    with pytest.raises(ValueError, match="too small"):
        pr.kelly_gaussian(0.001, 1e-170)


def test_series_with_prices_is_rejected_everywhere():
    prices = np.linspace(100, 110, 50)
    with pytest.raises(ValueError, match="decimals"):
        pr.historical_var(prices)
    with pytest.raises(ValueError, match="decimals"):
        pr.risk_report(prices)
