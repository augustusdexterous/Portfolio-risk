import numpy as np
import pytest

import portfolio_risk as pr


def _returns(seed=0, n=500, k=4):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(k, k))
    cov = a @ a.T / k + np.diag(np.linspace(0.5, 2.0, k))
    return rng.multivariate_normal(np.zeros(k), cov * 1e-4, size=n)


def test_sample_covariance_matches_numpy():
    r = _returns()
    np.testing.assert_allclose(pr.sample_covariance(r), np.cov(r, rowvar=False))


def test_shrinkage_endpoints():
    r = _returns()
    s = pr.sample_covariance(r)
    np.testing.assert_allclose(pr.shrink_covariance(r, delta=0.0), s)
    shrunk = pr.shrink_covariance(r, delta=1.0)
    # full shrinkage keeps variances and sets all correlations equal
    np.testing.assert_allclose(np.diag(shrunk), np.diag(s))
    corr = shrunk / np.sqrt(np.outer(np.diag(shrunk), np.diag(shrunk)))
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    assert np.allclose(off, off[0])


def test_portfolio_volatility_formula():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.6, 0.4])
    assert pr.portfolio_volatility(w, cov) == pytest.approx(np.sqrt(w @ cov @ w))


def test_risk_contributions_sum_to_one():
    cov = pr.sample_covariance(_returns())
    w = np.full(4, 0.25)
    rc = pr.risk_contributions(w, cov)
    assert rc.sum() == pytest.approx(1.0)
    assert np.all(rc >= -1e-12)


def test_equal_risk_contribution_weights_equalize_risk():
    cov = pr.sample_covariance(_returns(seed=7))
    w = pr.equal_risk_contribution_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w > 0)
    rc = pr.risk_contributions(w, cov)
    np.testing.assert_allclose(rc, np.full(4, 0.25), atol=1e-6)


def test_equal_risk_contribution_diagonal_is_inverse_volatility():
    vols = np.array([0.1, 0.2, 0.3])
    w = pr.equal_risk_contribution_weights(np.diag(vols**2))
    inv = 1 / vols
    np.testing.assert_allclose(w, inv / inv.sum(), rtol=1e-6)


def test_risk_inputs_validation():
    with pytest.raises(ValueError):
        pr.shrink_covariance(_returns(), delta=1.5)
    with pytest.raises(ValueError):
        pr.equal_risk_contribution_weights(np.array([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        pr.portfolio_volatility([0.5, 0.5], np.eye(3))
    with pytest.raises(ValueError):
        pr.sample_covariance(np.zeros((5, 2)))
