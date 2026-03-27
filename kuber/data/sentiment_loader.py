"""
Sentiment data loader for KUBER.

Provides sentiment scores in the range [-1, 1] for a list of tickers.
Three provider backends are supported:

* **synthetic** (default) -- derives sentiment from price returns plus
  Gaussian noise.  Works without any API keys or heavy dependencies.
* **finbert** -- uses the ProsusAI/finbert transformer model (requires
  ``transformers`` and ``torch``).
* **claude** -- placeholder for Anthropic-based sentiment (not yet
  implemented).
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from kuber.data.cache import ParquetCache
from kuber.utils.config import get_config

logger = logging.getLogger(__name__)

ProviderType = Literal["synthetic", "finbert", "claude"]


class SentimentLoader:
    """Generate or fetch sentiment scores for a list of tickers.

    Args:
        provider: Backend to use.  One of ``"synthetic"``, ``"finbert"``,
            or ``"claude"``.  Falls back to ``"synthetic"`` when the
            requested backend is unavailable.
        cache: Optional :class:`ParquetCache` instance.
    """

    def __init__(
        self,
        provider: ProviderType = "synthetic",
        cache: ParquetCache | None = None,
    ) -> None:
        self._cfg = get_config()
        self._cache = cache or ParquetCache()
        self._subdir: str = self._cfg.get(
            "data.sentiment.cache_subdir", "sentiment"
        )
        self._noise_std: float = self._cfg.get(
            "data.sentiment.synthetic_noise_std", 0.3
        )
        self._provider = self._resolve_provider(provider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        tickers: list[str],
        start: str | None = None,
        end: str | None = None,
        prices: pd.DataFrame | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Produce sentiment scores for each ticker.

        Args:
            tickers: Ticker symbols.
            start: Start date ``YYYY-MM-DD``.
            end: End date ``YYYY-MM-DD``.
            prices: Optional price DataFrame (avoids a redundant download
                when synthetic mode is used).
            use_cache: Try local cache first.

        Returns:
            DataFrame indexed by date with values in [-1, 1].
        """
        start = start or self._cfg.get("data.default_start", "2020-01-01")
        end = end or self._cfg.get("data.default_end", "2026-03-27")

        cache_key = self._cache_key(tickers, start, end)

        if use_cache:
            cached = self._cache.load_parquet(cache_key, self._subdir)
            if cached is not None:
                logger.info("Loaded sentiment from cache (%s)", cache_key)
                return cached

        if self._provider == "finbert":
            df = self._score_finbert(tickers, start, end)
        elif self._provider == "claude":
            logger.warning(
                "Claude sentiment provider not yet implemented — "
                "falling back to synthetic"
            )
            df = self._score_synthetic(tickers, start, end, prices)
        else:
            df = self._score_synthetic(tickers, start, end, prices)

        # Clamp to [-1, 1]
        df = df.clip(-1.0, 1.0)

        if not df.empty:
            self._cache.save_parquet(df, cache_key, self._subdir)

        return df

    # ------------------------------------------------------------------
    # Synthetic provider
    # ------------------------------------------------------------------

    def _score_synthetic(
        self,
        tickers: list[str],
        start: str,
        end: str,
        prices: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic sentiment from forward returns + noise.

        Uses a rolling 5-day forward return as the base signal, scales
        it to [-1, 1], and adds Gaussian noise.
        """
        logger.info("Using SYNTHETIC sentiment data")

        if prices is None:
            prices = self._load_prices(tickers, start, end)

        if prices.empty:
            logger.warning("No price data available for synthetic sentiment")
            return pd.DataFrame()

        # Only use tickers present in prices
        available = [t for t in tickers if t in prices.columns]
        if not available:
            return pd.DataFrame()

        prices = prices[available]

        # 5-day forward return as base signal
        fwd_ret = prices.pct_change(periods=5).shift(-5)

        # Scale to roughly [-1, 1] using tanh
        base = np.tanh(fwd_ret * 10)

        # Add Gaussian noise
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0, self._noise_std, size=base.shape)
        sentiment = base + noise

        # Clamp and drop NaN edges
        sentiment = sentiment.clip(-1.0, 1.0).dropna(how="all")

        sentiment.index.name = "Date"
        return sentiment

    # ------------------------------------------------------------------
    # FinBERT provider
    # ------------------------------------------------------------------

    def _score_finbert(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Attempt to score sentiment using FinBERT.

        Falls back to synthetic if torch/transformers are not installed.
        """
        try:
            from transformers import pipeline  # noqa: WPS433
        except ImportError:
            logger.warning(
                "transformers/torch not installed — "
                "falling back to synthetic sentiment"
            )
            return self._score_synthetic(tickers, start, end)

        logger.info("Using FinBERT sentiment provider")

        # FinBERT needs actual news text; for now we generate a simple
        # placeholder that uses the model on generic financial phrases.
        # A production implementation would fetch headlines from a news API.
        try:
            nlp = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None,
            )
        except Exception as exc:
            logger.warning("FinBERT init failed (%s) — using synthetic", exc)
            return self._score_synthetic(tickers, start, end)

        # Generate simple date index
        idx = pd.bdate_range(start=start, end=end, name="Date")
        scores: dict[str, list[float]] = {t: [] for t in tickers}

        # Placeholder: score a generic sentence per ticker per day.
        # Real implementation would plug in a news API here.
        for ticker in tickers:
            text = f"{ticker} stock performance outlook"
            try:
                result = nlp(text)
                # result is [[{label, score}, ...]]
                label_scores = {r["label"]: r["score"] for r in result[0]}
                pos = label_scores.get("positive", 0)
                neg = label_scores.get("negative", 0)
                score_val = pos - neg  # net sentiment
            except Exception:
                score_val = 0.0
            scores[ticker] = [score_val] * len(idx)

        df = pd.DataFrame(scores, index=idx)
        df.index.name = "Date"
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_prices(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Load prices via PriceLoader (lazy import to avoid circularity)."""
        from kuber.data.price_loader import PriceLoader  # noqa: WPS433

        loader = PriceLoader(cache=self._cache)
        return loader.load(tickers, start, end)

    @staticmethod
    def _resolve_provider(requested: str) -> str:
        """Validate provider string, falling back to synthetic."""
        valid = {"synthetic", "finbert", "claude"}
        if requested in valid:
            return requested
        logger.warning(
            "Unknown sentiment provider '%s' — defaulting to synthetic",
            requested,
        )
        return "synthetic"

    @staticmethod
    def _cache_key(tickers: list[str], start: str, end: str) -> str:
        ticker_str = "_".join(sorted(tickers))
        return f"sentiment_{ticker_str}_{start}_{end}"
