"""Base class for all signal generators in the KUBER framework."""

from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Signal(ABC):
    """Base class for all signal generators.

    Every concrete signal must implement ``name`` and ``generate``.
    The ``generate`` method returns a DataFrame with a DatetimeIndex and
    one column per ticker, with values in the [-1, 1] range.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable signal name."""
        ...

    @abstractmethod
    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate signal scores.

        Parameters
        ----------
        prices : pd.DataFrame
            Price DataFrame with DatetimeIndex rows and ticker columns.
        macro : pd.DataFrame | None
            Macro-economic data (e.g. VIX, yield curve).
        sentiment : pd.DataFrame | None
            Sentiment scores per ticker.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex x tickers, values in [-1, 1].
        """
        ...

    def describe(self) -> str:
        """Human-readable description."""
        return f"Signal: {self.name}"

    @staticmethod
    def normalize(
        series: pd.Series | pd.DataFrame, method: str = "zscore"
    ) -> pd.Series | pd.DataFrame:
        """Normalize values to the [-1, 1] range.

        Parameters
        ----------
        series : pd.Series | pd.DataFrame
            Input data to normalize.
        method : str
            One of ``"zscore"``, ``"minmax"``, or ``"rank"``.

        Returns
        -------
        pd.Series | pd.DataFrame
            Normalized data clipped to [-1, 1].
        """
        if method == "zscore":
            mean = series.mean()
            std = series.std()
            if isinstance(std, pd.Series):
                std = std.replace(0, 1)
            else:
                std = std if std != 0 else 1
            z = (series - mean) / std
            return z.clip(-3, 3) / 3  # clip at 3 sigma, scale to [-1, 1]

        elif method == "minmax":
            mn, mx = series.min(), series.max()
            rng = mx - mn
            if isinstance(rng, pd.Series):
                rng = rng.replace(0, 1)
            else:
                rng = rng if rng != 0 else 1
            return 2 * (series - mn) / rng - 1

        elif method == "rank":
            if isinstance(series, pd.DataFrame):
                ranked = series.rank(axis=1, pct=True)
            else:
                ranked = series.rank(pct=True)
            return 2 * ranked - 1

        logger.warning("Unknown normalization method '%s'; returning raw data.", method)
        return series
