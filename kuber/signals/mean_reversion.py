"""Mean-reversion trading signals for the KUBER framework.

Provides three mean-reversion variants:
- RSISignal             — Relative Strength Index (inverted)
- BollingerBandSignal   — Bollinger Band z-score (inverted)
- ZScoreReversionSignal — Rolling z-score of price (inverted)
"""

import logging

import numpy as np
import pandas as pd

from kuber.signals.base import Signal

logger = logging.getLogger(__name__)


class RSISignal(Signal):
    """RSI-based mean-reversion signal.

    Computes 14-day RSI, then inverts so that oversold (low RSI)
    maps to positive (bullish) and overbought (high RSI) to negative.
    Output is already in [-1, 1].
    """

    @property
    def name(self) -> str:
        return "RSI"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        period: int = 14,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate RSI mean-reversion signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (DatetimeIndex x tickers).
        period : int
            RSI look-back period (default 14).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1].
        """
        logger.info("Generating RSI signal (period=%d).", period)

        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Avoid division by zero
        avg_loss = avg_loss.replace(0, 1e-10)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Invert: oversold = bullish (+), overbought = bearish (-)
        signal = -(rsi - 50) / 50
        signal = signal.clip(-1, 1)

        logger.info("RSI signal generated: %s rows, %s tickers.", *signal.shape)
        return signal


class BollingerBandSignal(Signal):
    """Bollinger Band mean-reversion signal.

    Computes 20-day MA and 2-std-dev bands, then measures how far
    the price is from the MA relative to the band width. Inverted
    because mean-reversion: price above MA is bearish, below is bullish.
    """

    @property
    def name(self) -> str:
        return "BollingerBand"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        window: int = 20,
        num_std: float = 2.0,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate Bollinger Band mean-reversion signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        window : int
            Moving average window (default 20).
        num_std : float
            Number of standard deviations for bands (default 2.0).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1].
        """
        logger.info("Generating BollingerBand signal (window=%d, std=%.1f).", window, num_std)

        ma = prices.rolling(window=window, min_periods=window).mean()
        std = prices.rolling(window=window, min_periods=window).std()
        std = std.replace(0, 1e-10)

        # Negative because mean-reversion
        signal = -(prices - ma) / (num_std * std)
        signal = signal.clip(-1, 1)

        logger.info("BollingerBand signal generated: %s rows, %s tickers.", *signal.shape)
        return signal


class ZScoreReversionSignal(Signal):
    """Rolling z-score mean-reversion signal.

    Computes the 60-day rolling z-score of price, then inverts:
    below the rolling mean is bullish for a mean-reversion strategy.
    """

    @property
    def name(self) -> str:
        return "ZScoreReversion"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        window: int = 60,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate z-score mean-reversion signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        window : int
            Rolling window for z-score computation (default 60).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1].
        """
        logger.info("Generating ZScoreReversion signal (window=%d).", window)

        rolling_mean = prices.rolling(window=window, min_periods=window).mean()
        rolling_std = prices.rolling(window=window, min_periods=window).std()
        rolling_std = rolling_std.replace(0, 1e-10)

        zscore = (prices - rolling_mean) / rolling_std

        # Invert: below mean = bullish for reversion
        signal = -zscore
        signal = signal.clip(-1, 1)

        logger.info("ZScoreReversion signal generated: %s rows, %s tickers.", *signal.shape)
        return signal
