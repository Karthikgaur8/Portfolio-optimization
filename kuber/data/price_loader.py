"""
Price data loader for KUBER.

Wraps *yfinance* to download adjusted close prices with local Parquet
caching, missing-data handling, and multi-frequency support.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

from kuber.data.cache import ParquetCache
from kuber.utils.config import get_config

logger = logging.getLogger(__name__)

_FREQ_MAP: dict[str, str] = {
    "daily": "1d",
    "weekly": "1wk",
    "monthly": "1mo",
}

_RESAMPLE_MAP: dict[str, str] = {
    "daily": "D",
    "weekly": "W-FRI",
    "monthly": "MS",
}


class PriceLoader:
    """Download and cache equity / ETF price data via yfinance.

    Args:
        cache: Optional :class:`ParquetCache` instance.  A default one
            is created if omitted.
    """

    def __init__(self, cache: ParquetCache | None = None) -> None:
        self._cfg = get_config()
        self._cache = cache or ParquetCache()
        self._subdir: str = self._cfg.get("data.price.cache_subdir", "prices")
        self._ffill_limit: int = self._cfg.get("data.forward_fill_limit", 2)
        self._max_missing: float = self._cfg.get("data.max_missing_pct", 0.05)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        tickers: list[str],
        start: str | None = None,
        end: str | None = None,
        frequency: str = "daily",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load adjusted-close prices for *tickers*.

        Args:
            tickers: List of ticker symbols.
            start: Start date ``YYYY-MM-DD`` (default from config).
            end: End date ``YYYY-MM-DD`` (default from config).
            frequency: ``"daily"``, ``"weekly"``, or ``"monthly"``.
            use_cache: When ``True`` try the local Parquet cache first.

        Returns:
            DataFrame indexed by date with one column per valid ticker.
        """
        start = start or self._cfg.get("data.default_start", "2020-01-01")
        end = end or self._cfg.get("data.default_end", "2026-03-27")

        cache_key = self._cache_key(tickers, start, end, frequency)

        # Try cache
        if use_cache:
            cached = self._cache.load_parquet(cache_key, self._subdir)
            if cached is not None:
                logger.info("Loaded prices from cache (%s)", cache_key)
                return cached

        # Download
        logger.info(
            "Downloading prices for %d tickers (%s .. %s, %s)",
            len(tickers),
            start,
            end,
            frequency,
        )
        prices = self._download(tickers, start, end, frequency)

        # Quality checks + forward-fill
        prices = self._clean(prices)

        # Resample if needed (yfinance weekly/monthly can be patchy)
        if frequency != "daily" and not prices.empty:
            rule = _RESAMPLE_MAP.get(frequency, "D")
            prices = prices.resample(rule).last().dropna(how="all")

        # Cache
        if not prices.empty:
            self._cache.save_parquet(prices, cache_key, self._subdir)

        return prices

    @staticmethod
    def compute_returns(
        prices: pd.DataFrame,
        method: Literal["log", "simple"] = "log",
    ) -> pd.DataFrame:
        """Compute period returns from a price DataFrame.

        Args:
            prices: DataFrame of prices (output of :meth:`load`).
            method: ``"log"`` for log returns, ``"simple"`` for arithmetic.

        Returns:
            DataFrame of returns (first row is ``NaN``).
        """
        if method == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        return returns

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download(
        self,
        tickers: list[str],
        start: str,
        end: str,
        frequency: str,
    ) -> pd.DataFrame:
        """Download prices from yfinance, handling column naming quirks."""
        interval = _FREQ_MAP.get(frequency, "1d")

        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.error("yfinance download failed: %s", exc)
            return pd.DataFrame()

        if raw.empty:
            logger.warning("yfinance returned empty data")
            return pd.DataFrame()

        # yfinance returns multi-level columns when >1 ticker.
        # We want just the "Close" (auto_adjust=True makes Close = Adj Close).
        if isinstance(raw.columns, pd.MultiIndex):
            # Prefer "Close" level
            if "Close" in raw.columns.get_level_values(0):
                prices = raw["Close"]
            elif "Adj Close" in raw.columns.get_level_values(0):
                prices = raw["Adj Close"]
            else:
                # Fall back to first level
                prices = raw.iloc[:, raw.columns.get_level_values(0) == raw.columns.get_level_values(0).unique()[0]]
                prices.columns = prices.columns.droplevel(0)
        else:
            # Single ticker returns flat columns
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

        # Ensure columns are plain strings
        prices.columns = [str(c) for c in prices.columns]
        return prices

    def _clean(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill small gaps and drop tickers with too much missing."""
        if prices.empty:
            return prices

        total_rows = len(prices)
        drop_cols: list[str] = []

        for col in prices.columns:
            missing_pct = prices[col].isna().sum() / total_rows
            if missing_pct > self._max_missing:
                logger.warning(
                    "Ticker %s has %.1f%% missing data (>%.0f%%) — dropped",
                    col,
                    missing_pct * 100,
                    self._max_missing * 100,
                )
                drop_cols.append(col)
            elif missing_pct > 0:
                logger.debug(
                    "Ticker %s: %.1f%% missing — forward-filling",
                    col,
                    missing_pct * 100,
                )

        if drop_cols:
            prices = prices.drop(columns=drop_cols)

        # Forward fill up to limit, then drop remaining NaN rows at the start
        prices = prices.ffill(limit=self._ffill_limit)
        prices = prices.dropna(how="all")

        return prices

    @staticmethod
    def _cache_key(
        tickers: list[str], start: str, end: str, frequency: str
    ) -> str:
        """Deterministic cache key from request parameters."""
        ticker_str = "_".join(sorted(tickers))
        return f"{ticker_str}_{start}_{end}_{frequency}"
