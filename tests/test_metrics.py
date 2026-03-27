"""Tests for backtest metrics on known return series."""

import numpy as np
import pandas as pd
import pytest

from kuber.backtest.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    win_rate,
    tail_ratio,
    information_ratio,
    compute_all_metrics,
)


@pytest.fixture
def constant_returns():
    """Constant daily return of 0.04% (~10% annualized)."""
    dates = pd.bdate_range("2020-01-01", periods=252)
    return pd.Series(0.0004, index=dates, name="returns")


@pytest.fixture
def zero_returns():
    """Zero returns series."""
    dates = pd.bdate_range("2020-01-01", periods=252)
    return pd.Series(0.0, index=dates, name="returns")


@pytest.fixture
def random_returns():
    """Random returns with known seed."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=504)
    return pd.Series(np.random.normal(0.0003, 0.01, 504), index=dates, name="returns")


class TestAnnualizedReturn:
    def test_constant(self, constant_returns):
        ann = annualized_return(constant_returns)
        # 0.0004 * 252 ≈ 0.1008
        assert abs(ann - 0.1008) < 0.001

    def test_zero(self, zero_returns):
        assert annualized_return(zero_returns) == 0.0


class TestSharpeRatio:
    def test_constant_returns_high_sharpe(self, constant_returns):
        # Constant returns → near-zero vol → very high Sharpe (not infinite)
        s = sharpe_ratio(constant_returns, risk_free_rate=0.0)
        # With a positive mean and near-zero vol, Sharpe is extremely large
        assert s > 1000

    def test_positive_sharpe(self, random_returns):
        s = sharpe_ratio(random_returns, risk_free_rate=0.0)
        # With positive mean and reasonable vol, Sharpe should be positive
        assert s > 0


class TestSortinoRatio:
    def test_all_positive(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        ret = pd.Series(0.001, index=dates)
        s = sortino_ratio(ret)
        assert s == float("inf")

    def test_random(self, random_returns):
        s = sortino_ratio(random_returns)
        assert isinstance(s, float)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        ret = pd.Series(0.001, index=dates)
        assert max_drawdown(ret) == 0.0

    def test_known_drawdown(self):
        # Construct a series that goes up then down 20%
        ret = pd.Series([0.0, 0.10, 0.05, -0.15, -0.10, 0.05])
        mdd = max_drawdown(ret)
        assert mdd > 0
        # Cumulative: 1.0 → 1.10 → 1.155 → 0.98175 → 0.88358 → 0.92775
        # Peak = 1.155, trough ≈ 0.88358, DD ≈ 23.5%
        assert abs(mdd - 0.235) < 0.01


class TestWinRate:
    def test_all_positive(self):
        ret = pd.Series([0.01, 0.02, 0.03])
        assert win_rate(ret) == 1.0

    def test_half(self):
        ret = pd.Series([0.01, -0.01, 0.01, -0.01])
        assert win_rate(ret) == 0.5


class TestCalmarRatio:
    def test_zero_drawdown(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        ret = pd.Series(0.001, index=dates)
        assert calmar_ratio(ret) == 0.0  # DD = 0 → calmar = 0


class TestInformationRatio:
    def test_identical(self, random_returns):
        # Same returns → IR = 0
        ir = information_ratio(random_returns, random_returns)
        assert abs(ir) < 1e-6


class TestComputeAll:
    def test_returns_dict(self, random_returns):
        m = compute_all_metrics(random_returns, risk_free_rate=0.02)
        assert "annualized_return" in m
        assert "sharpe_ratio" in m
        assert "max_drawdown" in m
        assert "sortino_ratio" in m
        assert "calmar_ratio" in m
        assert "win_rate" in m
        assert "tail_ratio" in m

    def test_with_benchmark(self, random_returns):
        np.random.seed(99)
        bench = pd.Series(
            np.random.normal(0.0002, 0.01, len(random_returns)),
            index=random_returns.index,
        )
        m = compute_all_metrics(random_returns, benchmark_returns=bench)
        assert "information_ratio" in m
