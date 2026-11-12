import numpy as np
import pytest

import portfolio_risk as pr


def test_exceedance_array_length_and_definition():
    r = np.r_[np.zeros(100), -0.5, np.zeros(10)]  # one catastrophic loss after a quiet window
    flags = pr.rolling_var_exceedances(r, window=50, alpha=0.05)
    assert flags.size == r.size - 50
    assert flags.sum() == 1


def test_kupiec_accepts_correct_rate_and_rejects_wrong_rate():
    exact = np.zeros(1000, dtype=bool)
    exact[:10] = True  # 1% of 1000 at alpha=0.01
    lr, p = pr.kupiec_pof_test(exact, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0)

    too_many = np.zeros(1000, dtype=bool)
    too_many[:50] = True  # 5% exceedances against alpha=0.01
    _, p_bad = pr.kupiec_pof_test(too_many, 0.01)
    assert p_bad < 0.001


def test_kupiec_handles_zero_exceedances():
    lr, p = pr.kupiec_pof_test(np.zeros(500, dtype=bool), 0.01)
    assert lr > 0 and 0 < p < 1


def test_backtest_validation():
    with pytest.raises(ValueError):
        pr.rolling_var_exceedances(np.zeros(100), window=10)
    with pytest.raises(ValueError):
        pr.rolling_var_exceedances(np.zeros(100), window=50, method="magic")
    with pytest.raises(ValueError):
        pr.kupiec_pof_test(np.array([], dtype=bool), 0.01)


def test_var_levels_are_close_to_nominal_on_gaussian_data():
    # Average over 10 series: at 5%, a 4% forecast would fail these bands.
    hist, par = [], []
    for seed in range(10):
        r = np.random.default_rng(200 + seed).normal(0, 0.01, 1500)
        hist.append(pr.rolling_var_exceedances(r, window=250, alpha=0.05, method="historical").mean())
        par.append(pr.rolling_var_exceedances(r, window=250, alpha=0.05, method="parametric").mean())
    assert 0.045 <= float(np.mean(hist)) <= 0.060
    assert 0.045 <= float(np.mean(par)) <= 0.056
