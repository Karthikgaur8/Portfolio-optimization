"""Volatility-based risk signals for the KUBER framework.

These are risk signals (not directional). Higher volatility produces
more negative scores (risk-off).

Provides:
- RealizedVolSignal — rolling 21-day annualized volatility
- VolRatioSignal    — short-term / long-term vol ratio
- GARCHVolSignal    — GARCH(1,1) 1-step-ahead forecast
"""

import logging

import numpy as np
import pandas as pd

from kuber.signals.base import Signal

logger = logging.getLogger(__name__)


class RealizedVolSignal(Signal):
    """Rolling realized volatility signal.

    Computes 21-day annualized standard deviation of daily returns
    and normalizes cross-sectionally. Higher vol = more negative.
    """

    @property
    def name(self) -> str:
        return "RealizedVol"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        window: int = 21,
        annualize: float = np.sqrt(252),
        **kwargs,
    ) -> pd.DataFrame:
        """Generate realized volatility signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        window : int
            Rolling window in trading days (default 21).
        annualize : float
            Annualization factor (default sqrt(252)).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1]. Higher vol is more negative.
        """
        logger.info("Generating RealizedVol signal (window=%d).", window)

        returns = prices.pct_change()
        vol = returns.rolling(window=window, min_periods=window).std() * annualize

        # Cross-sectional normalization — invert so high vol is negative
        signal = -self.normalize(vol, method="zscore")
        signal = signal.clip(-1, 1)

        logger.info("RealizedVol signal generated: %s rows, %s tickers.", *signal.shape)
        return signal


class VolRatioSignal(Signal):
    """Volatility ratio signal.

    Ratio of short-term (21-day) vol to long-term (252-day) vol.
    A ratio > 1 means vol is elevated relative to history. Inverted
    so that elevated vol = negative (risk-off).
    """

    @property
    def name(self) -> str:
        return "VolRatio"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        short_window: int = 21,
        long_window: int = 252,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate volatility ratio signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        short_window : int
            Short-term vol window (default 21).
        long_window : int
            Long-term vol window (default 252).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1].
        """
        logger.info(
            "Generating VolRatio signal (short=%d, long=%d).", short_window, long_window
        )

        returns = prices.pct_change()
        vol_short = returns.rolling(window=short_window, min_periods=short_window).std()
        vol_long = returns.rolling(window=long_window, min_periods=long_window).std()
        vol_long = vol_long.replace(0, 1e-10)

        ratio = vol_short / vol_long

        # Invert and normalize: high ratio = risk-off = negative
        signal = -self.normalize(ratio, method="zscore")
        signal = signal.clip(-1, 1)

        logger.info("VolRatio signal generated: %s rows, %s tickers.", *signal.shape)
        return signal


class GARCHVolSignal(Signal):
    """GARCH(1,1) volatility signal.

    Fits a GARCH(1,1) model per ticker using the ``arch`` package and
    extracts the 1-step-ahead conditional volatility forecast.
    Falls back to realized volatility if fitting fails.
    """

    @property
    def name(self) -> str:
        return "GARCHVol"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        window: int = 252,
        fallback_window: int = 21,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate GARCH volatility signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        window : int
            Minimum observations for GARCH fitting (default 252).
        fallback_window : int
            Realized-vol window used when GARCH fitting fails (default 21).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1]. Higher forecast vol is more negative.
        """
        logger.info("Generating GARCHVol signal (window=%d).", window)

        returns = prices.pct_change().dropna()
        cond_vol = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

        for ticker in returns.columns:
            series = returns[ticker].dropna()
            if len(series) < window:
                logger.warning(
                    "Ticker %s has only %d observations; falling back to realized vol.",
                    ticker,
                    len(series),
                )
                rv = series.rolling(window=fallback_window, min_periods=fallback_window).std() * np.sqrt(252)
                cond_vol[ticker] = rv
                continue

            try:
                from arch import arch_model  # type: ignore[import-untyped]

                # Scale returns to percentage for numerical stability
                scaled = series * 100
                model = arch_model(scaled, vol="Garch", p=1, q=1, dist="normal", mean="Zero")
                result = model.fit(disp="off", show_warning=False)
                # Conditional volatility (in percentage terms), convert back
                cond_vol[ticker] = result.conditional_volatility / 100 * np.sqrt(252)
                logger.info("GARCH fitted successfully for %s.", ticker)
            except Exception as exc:
                logger.warning(
                    "GARCH fitting failed for %s (%s); falling back to realized vol.",
                    ticker,
                    exc,
                )
                rv = series.rolling(window=fallback_window, min_periods=fallback_window).std() * np.sqrt(252)
                cond_vol[ticker] = rv

        # Reindex to original prices index to keep NaN alignment
        cond_vol = cond_vol.reindex(prices.index)

        # Invert and normalize: high vol = negative
        signal = -self.normalize(cond_vol, method="zscore")
        signal = signal.clip(-1, 1)

        logger.info("GARCHVol signal generated: %s rows, %s tickers.", *signal.shape)
        return signal
