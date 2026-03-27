"""Integration test: run a small backtest and verify BacktestResult structure."""

import numpy as np
import pandas as pd
import pytest

from kuber.optimizer.markowitz import MarkowitzOptimizer
from kuber.optimizer.risk_parity import RiskParityOptimizer
from kuber.optimizer.constraints import PortfolioConstraints
from kuber.backtest.engine import BacktestEngine
from kuber.backtest.report import BacktestResult


@pytest.fixture
def synthetic_prices():
    """2 years of synthetic price data for 5 tickers."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=504)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]

    # Geometric Brownian motion-ish
    daily_returns = np.random.normal(0.0003, 0.015, (504, 5))
    prices = 100 * np.exp(np.cumsum(daily_returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


class TestBacktestEngine:
    def test_basic_backtest(self, synthetic_prices):
        optimizer = MarkowitzOptimizer(risk_aversion=1.0)
        engine = BacktestEngine(
            optimizer=optimizer,
            rebalance_frequency="monthly",
            lookback_window=63,  # ~3 months
        )

        result = engine.run(synthetic_prices)

        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_returns) > 0
        assert not result.portfolio_weights.empty
        assert len(result.rebalance_dates) > 0
        assert "annualized_return" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics

    def test_risk_parity_backtest(self, synthetic_prices):
        optimizer = RiskParityOptimizer()
        engine = BacktestEngine(
            optimizer=optimizer,
            rebalance_frequency="monthly",
            lookback_window=63,
        )
        result = engine.run(synthetic_prices)
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_returns) > 0

    def test_weights_sum_to_one(self, synthetic_prices):
        optimizer = MarkowitzOptimizer()
        engine = BacktestEngine(
            optimizer=optimizer,
            rebalance_frequency="monthly",
            lookback_window=63,
        )
        result = engine.run(synthetic_prices)

        for _, row in result.portfolio_weights.iterrows():
            non_zero = row[row != 0]
            if len(non_zero) > 0:
                assert abs(non_zero.sum() - 1.0) < 0.01

    def test_trade_log_populated(self, synthetic_prices):
        optimizer = MarkowitzOptimizer()
        engine = BacktestEngine(
            optimizer=optimizer,
            rebalance_frequency="monthly",
            lookback_window=63,
        )
        result = engine.run(synthetic_prices)
        assert not result.trade_log.empty
        assert "date" in result.trade_log.columns
        assert "ticker" in result.trade_log.columns
        assert "old_weight" in result.trade_log.columns
        assert "new_weight" in result.trade_log.columns

    def test_summary_string(self, synthetic_prices):
        optimizer = MarkowitzOptimizer()
        engine = BacktestEngine(
            optimizer=optimizer,
            rebalance_frequency="monthly",
            lookback_window=63,
        )
        result = engine.run(synthetic_prices)
        summary = result.summary()
        assert "KUBER" in summary
        assert "annualized_return" in summary

    def test_to_dict(self, synthetic_prices):
        optimizer = MarkowitzOptimizer()
        engine = BacktestEngine(
            optimizer=optimizer,
            rebalance_frequency="monthly",
            lookback_window=63,
        )
        result = engine.run(synthetic_prices)
        d = result.to_dict()
        assert "metrics" in d
        assert "rebalance_dates" in d
        assert "portfolio_returns" in d

    def test_expanding_window(self, synthetic_prices):
        optimizer = MarkowitzOptimizer()
        engine = BacktestEngine(
            optimizer=optimizer,
            rebalance_frequency="monthly",
            lookback_window=63,
            expanding_window=True,
        )
        result = engine.run(synthetic_prices)
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_returns) > 0

    def test_with_constraints(self, synthetic_prices):
        constraints = PortfolioConstraints(min_weight=0.05, max_weight=0.40, max_turnover=0.5)
        optimizer = MarkowitzOptimizer()
        engine = BacktestEngine(
            optimizer=optimizer,
            constraints=constraints,
            rebalance_frequency="monthly",
            lookback_window=63,
        )
        result = engine.run(synthetic_prices)
        assert isinstance(result, BacktestResult)

    def test_insufficient_data_raises(self):
        """Too few rows for lookback should raise."""
        dates = pd.bdate_range("2020-01-01", periods=20)
        prices = pd.DataFrame(
            np.random.randn(20, 3) * 0.01 + 100,
            index=dates,
            columns=["X", "Y", "Z"],
        )
        engine = BacktestEngine(
            optimizer=MarkowitzOptimizer(),
            lookback_window=252,
        )
        with pytest.raises(ValueError):
            engine.run(prices)
