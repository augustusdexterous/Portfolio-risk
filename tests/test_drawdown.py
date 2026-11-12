import numpy as np
import pytest

import portfolio_risk as pr


def test_max_drawdown_on_known_path():
    # wealth 1 -> 0.5 -> 0.75 -> 0.25; worst fall is from 1 to 0.25
    r = np.array([-0.5, 0.5, -2 / 3])
    assert pr.max_drawdown(r) == pytest.approx(0.75)
    np.testing.assert_allclose(pr.drawdown_series(r), [0.0, -0.5, -0.25, -0.75])


def test_no_drawdown_on_monotone_growth():
    r = np.full(10, 0.01)
    assert pr.max_drawdown(r) == 0.0
    assert pr.max_drawdown_duration(r) == 0


def test_drawdown_duration_counts_periods_below_peak():
    r = np.array([0.1, -0.05, -0.01, 0.2, 0.0])  # below the peak for 2 periods after the first fall
    assert pr.max_drawdown_duration(r) == 2


def test_sortino_matches_definition():
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.0])
    downside = np.minimum(r, 0)
    expected = r.mean() * 252 / (np.sqrt(np.mean(downside**2)) * np.sqrt(252))
    assert pr.sortino_ratio(r, 252) == pytest.approx(expected)


def test_sortino_and_calmar_raise_when_undefined():
    with pytest.raises(ValueError):
        pr.sortino_ratio(np.full(10, 0.01))
    with pytest.raises(ValueError):
        pr.calmar_ratio(np.full(10, 0.01))


def test_calmar_uses_cagr_over_max_drawdown():
    r = np.array([0.1, -0.2, 0.15, 0.05])
    growth = np.prod(1 + r)
    cagr = growth ** (252 / r.size) - 1
    assert pr.calmar_ratio(r, 252) == pytest.approx(cagr / pr.max_drawdown(r))


def test_periods_per_year_validation():
    with pytest.raises(ValueError):
        pr.sortino_ratio([0.01, -0.02, 0.03], periods_per_year=0)
