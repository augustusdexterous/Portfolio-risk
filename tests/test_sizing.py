import numpy as np
import pytest

import portfolio_risk as pr


def test_kelly_gaussian_formula():
    assert pr.kelly_gaussian(mu=0.001, sigma=0.01) == pytest.approx(10.0)
    assert pr.kelly_gaussian(mu=0.001, sigma=0.01, rf=0.0005) == pytest.approx(5.0)


def test_kelly_empirical_is_close_to_gaussian_formula_on_normal_data():
    r = np.random.default_rng(5).normal(0.0005, 0.01, 200_000)
    assert pr.kelly_empirical(r, max_leverage=50) == pytest.approx(pr.kelly_gaussian(0.0005, 0.01), rel=0.1)


def test_kelly_empirical_is_zero_for_negative_drift():
    r = np.random.default_rng(6).normal(-0.001, 0.01, 5000)
    assert pr.kelly_empirical(r) == 0.0


def test_kelly_empirical_without_losses_returns_the_cap():
    assert pr.kelly_empirical(np.full(50, 0.01), max_leverage=3.0) == pytest.approx(3.0)


def test_volatility_target_caps_leverage():
    assert pr.volatility_target_leverage(0.2, 0.1) == pytest.approx(0.5)
    assert pr.volatility_target_leverage(0.05, 0.1, max_leverage=1.5) == pytest.approx(1.5)


def test_sizing_validation():
    with pytest.raises(ValueError):
        pr.kelly_gaussian(0.001, 0.0)
    with pytest.raises(ValueError):
        pr.volatility_target_leverage(-0.2, 0.1)
