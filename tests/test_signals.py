"""Tests for KUBER signal generators.

Uses synthetic data to verify:
- All signals produce output in [-1, 1]
- CompositeSignal aggregates correctly and tracks attribution
- Signals handle NaN gracefully
"""

import numpy as np
import pandas as pd
import pytest

from kuber.signals.momentum import TSMOMSignal, XSMOMSignal, DualMomentumSignal
from kuber.signals.mean_reversion import (
    RSISignal,
    BollingerBandSignal,
    ZScoreReversionSignal,
)
from kuber.signals.volatility import RealizedVolSignal, VolRatioSignal
from kuber.signals.sentiment import SentimentSignal, SentimentMomentumSignal
from kuber.signals.macro import YieldCurveSignal, VIXRegimeSignal, FedStanceSignal
from kuber.signals.composite import CompositeSignal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DAYS = 300
TICKERS = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]


@pytest.fixture()
def prices() -> pd.DataFrame:
    """Synthetic price DataFrame: 300 days, 5 tickers with random walk."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=N_DAYS, freq="B")
    data = np.cumprod(1 + np.random.normal(0.0005, 0.02, (N_DAYS, len(TICKERS))), axis=0)
    data *= 100  # start around 100
    return pd.DataFrame(data, index=dates, columns=TICKERS)


@pytest.fixture()
def macro() -> pd.DataFrame:
    """Synthetic macro DataFrame with VIX, T10Y2Y, FEDFUNDS."""
    np.random.seed(99)
    dates = pd.bdate_range("2023-01-01", periods=N_DAYS, freq="B")
    return pd.DataFrame(
        {
            "VIX": np.random.uniform(10, 35, N_DAYS),
            "T10Y2Y": np.random.normal(0.5, 0.5, N_DAYS),
            "FEDFUNDS": np.cumsum(np.random.normal(0, 0.01, N_DAYS)) + 5.0,
        },
        index=dates,
    )


@pytest.fixture()
def sentiment() -> pd.DataFrame:
    """Synthetic sentiment DataFrame: same shape as prices."""
    np.random.seed(77)
    dates = pd.bdate_range("2023-01-01", periods=N_DAYS, freq="B")
    return pd.DataFrame(
        np.random.normal(0, 0.3, (N_DAYS, len(TICKERS))),
        index=dates,
        columns=TICKERS,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_signal_range(df: pd.DataFrame) -> None:
    """Assert all non-NaN values are in [-1, 1]."""
    valid = df.dropna(how="all")
    if valid.empty:
        return
    assert valid.min().min() >= -1.0 - 1e-9, f"Min below -1: {valid.min().min()}"
    assert valid.max().max() <= 1.0 + 1e-9, f"Max above 1: {valid.max().max()}"


# ---------------------------------------------------------------------------
# Momentum signals
# ---------------------------------------------------------------------------

class TestMomentumSignals:
    def test_tsmom_range(self, prices: pd.DataFrame) -> None:
        sig = TSMOMSignal()
        result = sig.generate(prices)
        assert result.shape[1] == len(TICKERS)
        _assert_signal_range(result)

    def test_xsmom_range(self, prices: pd.DataFrame) -> None:
        sig = XSMOMSignal()
        result = sig.generate(prices)
        _assert_signal_range(result)

    def test_dual_momentum_range(self, prices: pd.DataFrame) -> None:
        sig = DualMomentumSignal()
        result = sig.generate(prices)
        _assert_signal_range(result)

    def test_tsmom_name(self) -> None:
        assert TSMOMSignal().name == "TSMOM"

    def test_xsmom_name(self) -> None:
        assert XSMOMSignal().name == "XSMOM"


# ---------------------------------------------------------------------------
# Mean-reversion signals
# ---------------------------------------------------------------------------

class TestMeanReversionSignals:
    def test_rsi_range(self, prices: pd.DataFrame) -> None:
        sig = RSISignal()
        result = sig.generate(prices)
        _assert_signal_range(result)

    def test_bollinger_range(self, prices: pd.DataFrame) -> None:
        sig = BollingerBandSignal()
        result = sig.generate(prices)
        _assert_signal_range(result)

    def test_zscore_reversion_range(self, prices: pd.DataFrame) -> None:
        sig = ZScoreReversionSignal()
        result = sig.generate(prices)
        _assert_signal_range(result)


# ---------------------------------------------------------------------------
# Volatility signals
# ---------------------------------------------------------------------------

class TestVolatilitySignals:
    def test_realized_vol_range(self, prices: pd.DataFrame) -> None:
        sig = RealizedVolSignal()
        result = sig.generate(prices)
        _assert_signal_range(result)

    def test_vol_ratio_range(self, prices: pd.DataFrame) -> None:
        sig = VolRatioSignal()
        result = sig.generate(prices)
        _assert_signal_range(result)


# ---------------------------------------------------------------------------
# Sentiment signals
# ---------------------------------------------------------------------------

class TestSentimentSignals:
    def test_sentiment_range(
        self, prices: pd.DataFrame, sentiment: pd.DataFrame
    ) -> None:
        sig = SentimentSignal()
        result = sig.generate(prices, sentiment=sentiment)
        _assert_signal_range(result)

    def test_sentiment_no_data(self, prices: pd.DataFrame) -> None:
        sig = SentimentSignal()
        result = sig.generate(prices, sentiment=None)
        assert (result == 0).all().all()

    def test_sentiment_momentum_range(
        self, prices: pd.DataFrame, sentiment: pd.DataFrame
    ) -> None:
        sig = SentimentMomentumSignal()
        result = sig.generate(prices, sentiment=sentiment)
        _assert_signal_range(result)


# ---------------------------------------------------------------------------
# Macro signals
# ---------------------------------------------------------------------------

class TestMacroSignals:
    def test_yield_curve_range(
        self, prices: pd.DataFrame, macro: pd.DataFrame
    ) -> None:
        sig = YieldCurveSignal()
        result = sig.generate(prices, macro=macro)
        _assert_signal_range(result)
        # All tickers should have the same value at each date
        for _, row in result.dropna().iterrows():
            assert row.nunique() <= 1

    def test_vix_regime_range(
        self, prices: pd.DataFrame, macro: pd.DataFrame
    ) -> None:
        sig = VIXRegimeSignal()
        result = sig.generate(prices, macro=macro)
        _assert_signal_range(result)

    def test_fed_stance_range(
        self, prices: pd.DataFrame, macro: pd.DataFrame
    ) -> None:
        sig = FedStanceSignal()
        result = sig.generate(prices, macro=macro)
        _assert_signal_range(result)

    def test_macro_no_data(self, prices: pd.DataFrame) -> None:
        for cls in (YieldCurveSignal, VIXRegimeSignal, FedStanceSignal):
            sig = cls()
            result = sig.generate(prices, macro=None)
            assert (result == 0).all().all()

    def test_macro_broadcast(
        self, prices: pd.DataFrame, macro: pd.DataFrame
    ) -> None:
        """Macro signals should broadcast the same value to all tickers."""
        sig = FedStanceSignal()
        result = sig.generate(prices, macro=macro)
        assert result.columns.tolist() == TICKERS


# ---------------------------------------------------------------------------
# Composite signal
# ---------------------------------------------------------------------------

class TestCompositeSignal:
    def test_composite_range(self, prices: pd.DataFrame) -> None:
        signals = [RSISignal(), BollingerBandSignal()]
        comp = CompositeSignal(signals)
        result = comp.generate(prices)
        _assert_signal_range(result)

    def test_composite_attribution(self, prices: pd.DataFrame) -> None:
        signals = [RSISignal(), BollingerBandSignal()]
        comp = CompositeSignal(signals)
        comp.generate(prices)
        attrib = comp.attribution()
        assert set(attrib.keys()) == {"RSI", "BollingerBand"}
        for name, df in attrib.items():
            assert df.shape == (len(prices), len(TICKERS))

    def test_composite_custom_weights(self, prices: pd.DataFrame) -> None:
        signals = [RSISignal(), BollingerBandSignal()]
        comp = CompositeSignal(signals, weights={"RSI": 0.8, "BollingerBand": 0.2})
        result = comp.generate(prices)
        _assert_signal_range(result)

    def test_composite_describe(self) -> None:
        signals = [RSISignal(), BollingerBandSignal()]
        comp = CompositeSignal(signals)
        desc = comp.describe()
        assert "RSI" in desc
        assert "BollingerBand" in desc


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    def test_signals_with_nan_prices(self, prices: pd.DataFrame) -> None:
        """Insert NaNs and verify signals do not crash."""
        prices_with_nan = prices.copy()
        prices_with_nan.iloc[10:15, 0] = np.nan
        prices_with_nan.iloc[50:55, 2] = np.nan

        signal_classes = [
            TSMOMSignal,
            XSMOMSignal,
            DualMomentumSignal,
            RSISignal,
            BollingerBandSignal,
            ZScoreReversionSignal,
            RealizedVolSignal,
            VolRatioSignal,
        ]
        for cls in signal_classes:
            sig = cls()
            result = sig.generate(prices_with_nan)
            # Should not raise; non-NaN values must be in range
            valid = result.dropna(how="all")
            if not valid.empty:
                _assert_signal_range(valid)
