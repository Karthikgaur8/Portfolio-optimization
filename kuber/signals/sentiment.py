"""Sentiment-based trading signals for the KUBER framework.

Provides:
- SentimentSignal         — smoothed sentiment with cross-sectional z-score
- SentimentMomentumSignal — change in sentiment score
"""

import logging

import numpy as np
import pandas as pd

from kuber.signals.base import Signal

logger = logging.getLogger(__name__)


class SentimentSignal(Signal):
    """Smoothed sentiment signal.

    Takes raw sentiment data (DatetimeIndex x tickers), applies a 5-day
    moving average to smooth noise, then cross-sectionally z-scores
    and normalizes to [-1, 1].
    """

    @property
    def name(self) -> str:
        return "Sentiment"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        smooth_window: int = 5,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate smoothed sentiment signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (used only for index/columns alignment).
        sentiment : pd.DataFrame | None
            Raw sentiment scores (DatetimeIndex x tickers).
        smooth_window : int
            Moving-average smoothing window (default 5).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1].
        """
        logger.info("Generating Sentiment signal (smooth_window=%d).", smooth_window)

        if sentiment is None or sentiment.empty:
            logger.warning("No sentiment data provided; returning zeros.")
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Align to price universe
        sent = sentiment.reindex(index=prices.index, columns=prices.columns)

        # Smooth
        smoothed = sent.rolling(window=smooth_window, min_periods=1).mean()

        # Cross-sectional z-score at each time step
        row_mean = smoothed.mean(axis=1)
        row_std = smoothed.std(axis=1).replace(0, 1)
        z = smoothed.sub(row_mean, axis=0).div(row_std, axis=0)
        signal = (z.clip(-3, 3) / 3).clip(-1, 1)

        logger.info("Sentiment signal generated: %s rows, %s tickers.", *signal.shape)
        return signal


class SentimentMomentumSignal(Signal):
    """Sentiment momentum signal.

    Measures the 5-day change in sentiment score, then normalizes
    to [-1, 1]. Positive change = improving sentiment = bullish.
    """

    @property
    def name(self) -> str:
        return "SentimentMomentum"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        diff_window: int = 5,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate sentiment momentum signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (used for index/columns alignment).
        sentiment : pd.DataFrame | None
            Raw sentiment scores (DatetimeIndex x tickers).
        diff_window : int
            Look-back for sentiment change (default 5).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1].
        """
        logger.info("Generating SentimentMomentum signal (diff_window=%d).", diff_window)

        if sentiment is None or sentiment.empty:
            logger.warning("No sentiment data provided; returning zeros.")
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        sent = sentiment.reindex(index=prices.index, columns=prices.columns)
        delta = sent.diff(diff_window)

        signal = self.normalize(delta, method="zscore").clip(-1, 1)

        logger.info("SentimentMomentum signal generated: %s rows, %s tickers.", *signal.shape)
        return signal
