"""Tests for portfolio optimizers — verify valid weights for each optimizer."""

import numpy as np
import pandas as pd
import pytest

from kuber.optimizer.base import OptimizationResult
from kuber.optimizer.constraints import PortfolioConstraints
from kuber.optimizer.markowitz import MarkowitzOptimizer
from kuber.optimizer.risk_parity import RiskParityOptimizer
from kuber.optimizer.black_litterman import BlackLittermanOptimizer
from kuber.optimizer.hierarchical import HRPOptimizer
from kuber.optimizer.regime_aware import RegimeAwareOptimizer


@pytest.fixture
def sample_data():
    """Create a small, realistic expected-return / covariance pair."""
    assets = ["A", "B", "C", "D", "E"]
    np.random.seed(42)
    # Annualized expected returns
    mu = pd.Series([0.08, 0.12, 0.06, 0.10, 0.09], index=assets)
    # Random covariance matrix (must be positive semi-definite)
    X = np.random.randn(500, 5) * 0.01
    cov = pd.DataFrame(np.cov(X.T) * 252, index=assets, columns=assets)
    return mu, cov, assets


def _assert_valid_weights(result: OptimizationResult, assets: list[str], tol: float = 1e-4):
    """Assert weights sum to 1, are within bounds, and cover all assets."""
    w = result.weights
    assert isinstance(w, pd.Series)
    assert set(w.index) == set(assets)
    assert abs(w.sum() - 1.0) < tol, f"Weights sum to {w.sum()}, expected 1.0"
    assert (w >= -tol).all(), f"Negative weights found: {w[w < -tol]}"
    assert isinstance(result.expected_return, float)
    assert isinstance(result.expected_volatility, float)
    assert isinstance(result.sharpe_ratio, float)


class TestMarkowitzOptimizer:
    def test_basic(self, sample_data):
        mu, cov, assets = sample_data
        opt = MarkowitzOptimizer(risk_aversion=1.0)
        result = opt.optimize(mu, cov, risk_free_rate=0.04)
        _assert_valid_weights(result, assets)

    def test_with_constraints(self, sample_data):
        mu, cov, assets = sample_data
        constraints = PortfolioConstraints(min_weight=0.05, max_weight=0.40)
        opt = MarkowitzOptimizer()
        result = opt.optimize(mu, cov, constraints=constraints)
        _assert_valid_weights(result, assets)
        assert result.weights.min() >= 0.05 - 1e-4
        assert result.weights.max() <= 0.40 + 1e-4

    def test_with_turnover(self, sample_data):
        mu, cov, assets = sample_data
        current = pd.Series(0.2, index=assets)
        constraints = PortfolioConstraints(max_turnover=0.3)
        opt = MarkowitzOptimizer()
        result = opt.optimize(mu, cov, constraints=constraints, current_weights=current)
        _assert_valid_weights(result, assets)
        turnover = np.abs(result.weights - current).sum()
        assert turnover <= 0.3 + 1e-3


class TestRiskParityOptimizer:
    def test_basic(self, sample_data):
        mu, cov, assets = sample_data
        opt = RiskParityOptimizer()
        result = opt.optimize(mu, cov)
        _assert_valid_weights(result, assets)

    def test_risk_contributions_are_roughly_equal(self, sample_data):
        mu, cov, assets = sample_data
        opt = RiskParityOptimizer()
        result = opt.optimize(mu, cov)
        w = result.weights.values
        Sigma = cov.values
        port_vol = np.sqrt(w @ Sigma @ w)
        marginal = Sigma @ w
        rc = w * marginal / port_vol
        rc_pct = rc / rc.sum()
        # Each should be close to 1/n = 0.2
        assert np.allclose(rc_pct, 0.2, atol=0.05)


class TestBlackLittermanOptimizer:
    def test_basic(self, sample_data):
        mu, cov, assets = sample_data
        opt = BlackLittermanOptimizer(risk_aversion=2.5, tau=0.05)
        result = opt.optimize(mu, cov, risk_free_rate=0.04)
        _assert_valid_weights(result, assets)

    def test_no_signals(self, sample_data):
        _, cov, assets = sample_data
        # Zero expected returns → no views → should still work
        mu_zero = pd.Series(0.0, index=assets)
        opt = BlackLittermanOptimizer()
        result = opt.optimize(mu_zero, cov)
        _assert_valid_weights(result, assets)


class TestHRPOptimizer:
    def test_basic(self, sample_data):
        mu, cov, assets = sample_data
        opt = HRPOptimizer()
        result = opt.optimize(mu, cov)
        _assert_valid_weights(result, assets)

    def test_independent_of_expected_returns(self, sample_data):
        mu, cov, assets = sample_data
        opt = HRPOptimizer()
        r1 = opt.optimize(mu, cov)
        r2 = opt.optimize(mu * 2, cov)
        # Weights should be the same (HRP ignores expected returns)
        pd.testing.assert_series_equal(r1.weights, r2.weights, atol=1e-6)


class TestRegimeAwareOptimizer:
    def test_bull_regime(self, sample_data):
        mu, cov, assets = sample_data
        opt = RegimeAwareOptimizer()
        result = opt.optimize(mu, cov, regime=2)  # 2 = bull
        _assert_valid_weights(result, assets)
        assert result.metadata["regime"] == "bull"

    def test_bear_regime(self, sample_data):
        mu, cov, assets = sample_data
        opt = RegimeAwareOptimizer()
        result = opt.optimize(mu, cov, regime=0)  # 0 = bear
        _assert_valid_weights(result, assets)
        assert result.metadata["regime"] == "bear"

    def test_no_regime(self, sample_data):
        mu, cov, assets = sample_data
        opt = RegimeAwareOptimizer()
        result = opt.optimize(mu, cov, regime=None)
        _assert_valid_weights(result, assets)
        assert result.metadata["regime"] == "sideways"  # fallback
