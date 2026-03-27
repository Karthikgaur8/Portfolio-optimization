"""
Parquet-based local caching layer for KUBER.

Stores and retrieves pandas DataFrames as Parquet files under the
project-level ``.cache/`` directory (configurable).
"""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Project root: three levels up from this file (kuber/data/cache.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ParquetCache:
    """Simple file-system cache backed by Parquet files.

    Args:
        cache_dir: Root cache directory.  Defaults to ``<project>/.cache``.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = _PROJECT_ROOT / ".cache"
        self._root = Path(cache_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("ParquetCache initialised at %s", self._root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_parquet(
        self, df: pd.DataFrame, key: str, subdir: str = ""
    ) -> Path:
        """Persist *df* as a Parquet file.

        Args:
            df: DataFrame to save.
            key: Logical name used as the filename (sanitised automatically).
            subdir: Optional subdirectory under the cache root
                    (e.g. ``"prices"``).

        Returns:
            The :class:`Path` of the written file.
        """
        path = self._resolve(key, subdir)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow")
        logger.debug("Cached %s  (%d rows)", path.name, len(df))
        return path

    def load_parquet(
        self, key: str, subdir: str = ""
    ) -> pd.DataFrame | None:
        """Load a cached DataFrame, or return ``None`` if not found.

        Args:
            key: Same key that was passed to :meth:`save_parquet`.
            subdir: Subdirectory used when saving.

        Returns:
            The DataFrame, or *None* when no cached file exists.
        """
        path = self._resolve(key, subdir)
        if not path.exists():
            logger.debug("Cache miss: %s", path)
            return None
        logger.debug("Cache hit: %s", path)
        return pd.read_parquet(path, engine="pyarrow")

    def is_cached(self, key: str, subdir: str = "") -> bool:
        """Check whether a cached file exists for *key*.

        Args:
            key: Logical cache key.
            subdir: Subdirectory used when saving.

        Returns:
            ``True`` when a corresponding Parquet file is present.
        """
        return self._resolve(key, subdir).exists()

    def clear(self, subdir: str = "") -> int:
        """Remove all cached files under *subdir* (or entire cache).

        Returns:
            Number of files deleted.
        """
        target = self._root / subdir if subdir else self._root
        count = 0
        if target.is_dir():
            for f in target.glob("*.parquet"):
                f.unlink()
                count += 1
        logger.info("Cleared %d cached files from %s", count, target)
        return count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitise_key(key: str) -> str:
        """Turn an arbitrary key string into a safe filename stem."""
        return re.sub(r"[^\w\-.]", "_", key)

    def _resolve(self, key: str, subdir: str) -> Path:
        """Build the full path for a given key/subdir pair."""
        stem = self._sanitise_key(key)
        if not stem.endswith(".parquet"):
            stem += ".parquet"
        if subdir:
            return self._root / subdir / stem
        return self._root / stem
