"""Portfolio-level macro signals for the KUBER framework.

Macro signals are broadcast to all tickers (same value per row)
because they capture market-wide conditions rather than stock-specific ones.

Provides:
- YieldCurveSignal — 10Y-2Y spread as risk-on / risk-off indicator
- VIXRegimeSignal  — VIX percentile relative to 252-day history
- FedStanceSignal  — direction of Fed Funds rate changes
"""

import logging

import numpy as np
import pandas as pd

from kuber.signals.base import Signal

logger = logging.getLogger(__name__)


def _broadcast_to_tickers(
    series: pd.Series, prices: pd.DataFrame
) -> pd.DataFrame:
    """Broadcast a single-column series to all ticker columns.

    Parameters
    ----------
    series : pd.Series
        Scalar signal per date.
    prices : pd.DataFrame
        Price DataFrame whose columns define the ticker universe.

    Returns
    -------
    pd.DataFrame
        Same value for every ticker at each date.
    """
    df = pd.DataFrame(
        np.tile(series.values[:, None], (1, len(prices.columns))),
        index=series.index,
        columns=prices.columns,
    )
    return df.reindex(prices.index)


class YieldCurveSignal(Signal):
    """Yield curve slope signal.

    Uses the 10Y-2Y Treasury spread (``T10Y2Y`` column in macro data).
    Positive spread = risk-on (+), negative (inverted curve) = risk-off (-).
    Broadcast to all tickers.
    """

    @property
    def name(self) -> str:
        return "YieldCurve"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        column: str = "T10Y2Y",
        **kwargs,
    ) -> pd.DataFrame:
        """Generate yield curve signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (used for index/columns alignment).
        macro : pd.DataFrame | None
            Must contain the ``column`` (default ``T10Y2Y``).
        column : str
            Column name for the yield-curve spread.

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1], broadcast to all tickers.
        """
        logger.info("Generating YieldCurve signal (column=%s).", column)

        if macro is None or column not in macro.columns:
            logger.warning("Macro data missing '%s'; returning zeros.", column)
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        spread = macro[column].reindex(prices.index).ffill()
        normed = self.normalize(spread, method="zscore").clip(-1, 1)

        signal = _broadcast_to_tickers(normed, prices)
        logger.info("YieldCurve signal generated: %s rows, %s tickers.", *signal.shape)
        return signal


class VIXRegimeSignal(Signal):
    """VIX percentile regime signal.

    Computes the percentile rank of the current VIX level relative
    to its trailing 252-day history. High VIX = risk-off (negative),
    low VIX = risk-on (positive). Broadcast to all tickers.
    """

    @property
    def name(self) -> str:
        return "VIXRegime"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        column: str = "VIX",
        lookback: int = 252,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate VIX regime signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        macro : pd.DataFrame | None
            Must contain the ``column`` (default ``VIX``).
        column : str
            Column name for VIX data.
        lookback : int
            Rolling window for percentile calculation (default 252).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1], broadcast to all tickers.
        """
        logger.info("Generating VIXRegime signal (column=%s, lookback=%d).", column, lookback)

        # Try the requested column name, then common alternatives
        vix_col = None
        if macro is not None:
            for candidate in [column, "VIXCLS", "VIX", "vix"]:
                if candidate in macro.columns:
                    vix_col = candidate
                    break

        if macro is None or vix_col is None:
            logger.warning("Macro data missing VIX column; returning zeros.")
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        vix = macro[vix_col].reindex(prices.index).ffill()

        # Rolling percentile rank
        def _rolling_pctile(s: pd.Series, w: int) -> pd.Series:
            out = pd.Series(np.nan, index=s.index)
            arr = s.values
            for i in range(w, len(arr)):
                window = arr[i - w : i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) > 0:
                    out.iloc[i] = (valid < arr[i]).sum() / len(valid)
            return out

        pctile = _rolling_pctile(vix, lookback)

        # High percentile = risk-off (negative), low = risk-on (positive)
        signal_series = (1 - 2 * pctile).clip(-1, 1)

        signal = _broadcast_to_tickers(signal_series, prices)
        logger.info("VIXRegime signal generated: %s rows, %s tickers.", *signal.shape)
        return signal


class FedStanceSignal(Signal):
    """Federal Reserve policy stance signal.

    Measures the direction of Fed Funds rate changes over a trailing
    window. Rising rates = hawkish = negative. Falling = dovish = positive.
    Broadcast to all tickers.
    """

    @property
    def name(self) -> str:
        return "FedStance"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        column: str = "FEDFUNDS",
        lookback: int = 126,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate Fed stance signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        macro : pd.DataFrame | None
            Must contain the ``column`` (default ``FEDFUNDS``).
        column : str
            Column name for the Fed Funds rate.
        lookback : int
            Trailing window in trading days (default 126 ~ 6 months).

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, 1], broadcast to all tickers.
        """
        logger.info("Generating FedStance signal (column=%s, lookback=%d).", column, lookback)

        if macro is None or column not in macro.columns:
            logger.warning("Macro data missing '%s'; returning zeros.", column)
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        rate = macro[column].reindex(prices.index).ffill()

        # Change over trailing window
        rate_change = rate.diff(lookback)

        # Invert: rising = hawkish = negative
        normed = -self.normalize(rate_change, method="zscore").clip(-1, 1)

        signal = _broadcast_to_tickers(normed, prices)
        logger.info("FedStance signal generated: %s rows, %s tickers.", *signal.shape)
        return signal
