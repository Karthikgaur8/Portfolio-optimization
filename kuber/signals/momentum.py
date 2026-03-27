"""Momentum-based trading signals for the KUBER framework.

Provides three momentum variants:
- TSMOMSignal   — time-series (absolute) momentum
- XSMOMSignal   — cross-sectional (relative) momentum
- DualMomentumSignal — blended absolute + relative momentum
"""

import logging

import numpy as np
import pandas as pd

from kuber.signals.base import Signal

logger = logging.getLogger(__name__)


class TSMOMSignal(Signal):
    """Time-Series Momentum (12-1).

    Classic Moskowitz / Ooi / Pedersen TSMOM signal:
    trailing 252-day return minus trailing 21-day return,
    then cross-sectionally z-scored at each time step.
    """

    @property
    def name(self) -> str:
        return "TSMOM"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        long_window: int = 252,
        short_window: int = 21,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate time-series momentum signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (DatetimeIndex x tickers).
        long_window : int
            Look-back for the long-term return (default 252 days).
        short_window : int
            Look-back for the short-term return to subtract (default 21 days).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1].
        """
        logger.info("Generating TSMOM signal (long=%d, short=%d).", long_window, short_window)

        ret_long = prices.pct_change(long_window)
        ret_short = prices.pct_change(short_window)
        raw = ret_long - ret_short

        # Cross-sectional z-score at each time step
        row_mean = raw.mean(axis=1)
        row_std = raw.std(axis=1).replace(0, 1)
        z = raw.sub(row_mean, axis=0).div(row_std, axis=0)
        signal = (z.clip(-3, 3) / 3).clip(-1, 1)

        logger.info("TSMOM signal generated: %s rows, %s tickers.", *signal.shape)
        return signal


class XSMOMSignal(Signal):
    """Cross-Sectional Momentum.

    Rank assets by trailing 12-month return at each time step,
    then map ranks to [-1, 1] (top assets +1, bottom -1).
    """

    @property
    def name(self) -> str:
        return "XSMOM"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        lookback: int = 252,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate cross-sectional momentum signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        lookback : int
            Return look-back in trading days (default 252).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1].
        """
        logger.info("Generating XSMOM signal (lookback=%d).", lookback)

        trailing_ret = prices.pct_change(lookback)
        signal = self.normalize(trailing_ret, method="rank")

        logger.info("XSMOM signal generated: %s rows, %s tickers.", *signal.shape)
        return signal


class DualMomentumSignal(Signal):
    """Dual Momentum — blend of absolute and relative momentum.

    Combines:
    - absolute_score: sign(12m return) scaled by magnitude
    - relative_score: cross-sectional rank
    Blend: 0.5 * absolute + 0.5 * relative, then normalized to [-1, 1].
    """

    @property
    def name(self) -> str:
        return "DualMomentum"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        lookback: int = 252,
        abs_weight: float = 0.5,
        rel_weight: float = 0.5,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate dual momentum signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        lookback : int
            Return look-back in trading days (default 252).
        abs_weight : float
            Weight for absolute momentum component (default 0.5).
        rel_weight : float
            Weight for relative momentum component (default 0.5).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1].
        """
        logger.info(
            "Generating DualMomentum signal (lookback=%d, abs=%.2f, rel=%.2f).",
            lookback,
            abs_weight,
            rel_weight,
        )

        trailing_ret = prices.pct_change(lookback)

        # Absolute momentum: sign scaled by magnitude, then z-scored
        absolute_raw = np.sign(trailing_ret) * np.abs(trailing_ret)
        absolute_score = self.normalize(absolute_raw, method="zscore")

        # Relative momentum: cross-sectional rank
        relative_score = self.normalize(trailing_ret, method="rank")

        # Blend
        blended = abs_weight * absolute_score + rel_weight * relative_score
        signal = blended.clip(-1, 1)

        logger.info("DualMomentum signal generated: %s rows, %s tickers.", *signal.shape)
        return signal
