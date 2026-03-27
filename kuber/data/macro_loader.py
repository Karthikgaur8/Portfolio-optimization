"""
Macro-economic data loader for KUBER.

Pulls indicator series from FRED (Federal Reserve Economic Data) via the
``fredapi`` library, with local Parquet caching.
"""

import logging
from typing import Any

import pandas as pd

from kuber.data.cache import ParquetCache
from kuber.utils.config import get_config

logger = logging.getLogger(__name__)

_DEFAULT_INDICATORS: list[str] = [
    "GS10",      # 10-Year Treasury Yield
    "GS2",       # 2-Year Treasury Yield
    "VIXCLS",    # VIX
    "UNRATE",    # Unemployment Rate
    "CPIAUCSL",  # CPI
    "FEDFUNDS",  # Fed Funds Rate
    "T10Y2Y",    # 10Y minus 2Y spread
]


class MacroLoader:
    """Download and cache macro-economic indicators from FRED.

    The FRED API key is read from the ``FRED_API_KEY`` environment
    variable (loaded via ``.env``).  If missing the loader logs a
    warning and returns an empty DataFrame rather than crashing.

    Args:
        cache: Optional :class:`ParquetCache` instance.
    """

    def __init__(self, cache: ParquetCache | None = None) -> None:
        self._cfg = get_config()
        self._cache = cache or ParquetCache()
        self._subdir: str = self._cfg.get("data.macro.cache_subdir", "macro")
        self._api_key: str | None = self._cfg.get_env("FRED_API_KEY")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        indicators: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load macro indicator series from FRED.

        Args:
            indicators: FRED series IDs.  Defaults to the seven standard
                indicators defined in ``default.yaml``.
            start: Start date ``YYYY-MM-DD``.
            end: End date ``YYYY-MM-DD``.
            use_cache: Try local cache first.

        Returns:
            DataFrame indexed by date with one column per indicator,
            forward-filled to daily frequency.
        """
        indicators = indicators or self._cfg.get(
            "data.macro.default_indicators", _DEFAULT_INDICATORS
        )
        start = start or self._cfg.get("data.default_start", "2020-01-01")
        end = end or self._cfg.get("data.default_end", "2026-03-27")

        cache_key = self._cache_key(indicators, start, end)

        # Try cache
        if use_cache:
            cached = self._cache.load_parquet(cache_key, self._subdir)
            if cached is not None:
                logger.info("Loaded macro data from cache (%s)", cache_key)
                return cached

        # Check API key
        if not self._api_key:
            logger.warning(
                "FRED_API_KEY not set — returning empty DataFrame. "
                "Set it in .env to fetch macro data."
            )
            return pd.DataFrame()

        # Download from FRED
        df = self._download(indicators, start, end)

        # Forward-fill to daily (many FRED series are monthly/weekly)
        if not df.empty:
            idx = pd.date_range(start=start, end=end, freq="D", name="Date")
            df = df.reindex(idx).ffill()
            df.index.name = "Date"

        # Cache
        if not df.empty:
            self._cache.save_parquet(df, cache_key, self._subdir)

        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _download(
        self,
        indicators: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch each indicator from FRED and combine into one DataFrame."""
        try:
            from fredapi import Fred  # noqa: WPS433
        except ImportError:
            logger.error(
                "fredapi is not installed. Run: pip install fredapi"
            )
            return pd.DataFrame()

        fred = Fred(api_key=self._api_key)
        frames: dict[str, pd.Series] = {}

        for series_id in indicators:
            try:
                s = fred.get_series(
                    series_id,
                    observation_start=start,
                    observation_end=end,
                )
                if s is not None and not s.empty:
                    frames[series_id] = s
                    logger.debug(
                        "FRED %s: %d observations", series_id, len(s)
                    )
                else:
                    logger.warning("FRED %s returned no data", series_id)
            except Exception as exc:
                logger.warning(
                    "Failed to fetch FRED %s: %s", series_id, exc
                )

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index.name = "Date"
        logger.info(
            "Downloaded %d FRED indicators (%d rows)",
            len(df.columns),
            len(df),
        )
        return df

    @staticmethod
    def _cache_key(indicators: list[str], start: str, end: str) -> str:
        """Build a deterministic cache key."""
        ind_str = "_".join(sorted(indicators))
        return f"macro_{ind_str}_{start}_{end}"
