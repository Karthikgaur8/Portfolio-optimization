"""
Tests for KUBER data engine modules.

All tests are designed to pass without API keys by mocking external
services where necessary and using small date ranges / few tickers.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for cache tests."""
    return tmp_path / "test_cache"


@pytest.fixture()
def cache(tmp_cache_dir: Path):
    """ParquetCache backed by a temp directory."""
    from kuber.data.cache import ParquetCache
    return ParquetCache(cache_dir=tmp_cache_dir)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small sample DataFrame for cache round-trip tests."""
    dates = pd.bdate_range("2024-01-02", periods=5, name="Date")
    return pd.DataFrame(
        {"SPY": [470.0, 471.0, 472.0, 473.0, 474.0],
         "QQQ": [400.0, 401.0, 402.0, 403.0, 404.0]},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

class TestParquetCache:
    """Tests for the Parquet caching layer."""

    def test_save_and_load_roundtrip(
        self, cache, sample_df: pd.DataFrame
    ) -> None:
        """Save a DF, reload it, and verify equality."""
        cache.save_parquet(sample_df, "test_prices", subdir="prices")
        loaded = cache.load_parquet("test_prices", subdir="prices")
        assert loaded is not None
        pd.testing.assert_frame_equal(
            sample_df, loaded, check_freq=False
        )

    def test_is_cached(self, cache, sample_df: pd.DataFrame) -> None:
        assert not cache.is_cached("missing_key")
        cache.save_parquet(sample_df, "exists", subdir="prices")
        assert cache.is_cached("exists", subdir="prices")

    def test_load_missing_returns_none(self, cache) -> None:
        assert cache.load_parquet("nonexistent") is None

    def test_clear(self, cache, sample_df: pd.DataFrame) -> None:
        cache.save_parquet(sample_df, "a", subdir="sub")
        cache.save_parquet(sample_df, "b", subdir="sub")
        removed = cache.clear(subdir="sub")
        assert removed == 2
        assert not cache.is_cached("a", subdir="sub")


# ---------------------------------------------------------------------------
# Universe tests
# ---------------------------------------------------------------------------

class TestUniverse:
    """Tests for universe loading."""

    def test_list_universes(self) -> None:
        from kuber.data.universe import list_universes
        names = list_universes()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "balanced_etf" in names

    def test_load_universe(self) -> None:
        from kuber.data.universe import load_universe
        uni = load_universe("balanced_etf")
        assert "tickers" in uni
        assert "name" in uni
        assert "description" in uni
        assert isinstance(uni["tickers"], list)
        assert len(uni["tickers"]) > 0

    def test_load_unknown_universe_raises(self) -> None:
        from kuber.data.universe import load_universe
        with pytest.raises(KeyError, match="not found"):
            load_universe("this_does_not_exist")


# ---------------------------------------------------------------------------
# PriceLoader tests
# ---------------------------------------------------------------------------

def _mock_yf_download(*args, **kwargs):
    """Return a small fake price DataFrame mimicking yfinance output."""
    tickers = kwargs.get("tickers") or (args[0] if args else ["SPY"])
    if isinstance(tickers, str):
        tickers = tickers.split()
    dates = pd.bdate_range("2024-01-02", periods=10, name="Date")
    rng = np.random.default_rng(0)
    if len(tickers) == 1:
        return pd.DataFrame(
            {"Close": rng.normal(100, 1, len(dates))},
            index=dates,
        )
    # Multi-ticker: multi-level columns
    cols = pd.MultiIndex.from_product(
        [["Close"], tickers], names=["Price", "Ticker"]
    )
    data = rng.normal(100, 1, (len(dates), len(tickers)))
    return pd.DataFrame(data, index=dates, columns=cols)


class TestPriceLoader:
    """Tests for the yfinance price loader (mocked downloads)."""

    @patch("kuber.data.price_loader.yf.download", side_effect=_mock_yf_download)
    def test_load_returns_dataframe(self, mock_dl, cache) -> None:
        from kuber.data.price_loader import PriceLoader
        loader = PriceLoader(cache=cache)
        df = loader.load(
            ["SPY", "QQQ"], start="2024-01-02", end="2024-01-15",
            use_cache=False,
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df.columns) >= 1

    @patch("kuber.data.price_loader.yf.download", side_effect=_mock_yf_download)
    def test_load_single_ticker(self, mock_dl, cache) -> None:
        from kuber.data.price_loader import PriceLoader
        loader = PriceLoader(cache=cache)
        df = loader.load(
            ["SPY"], start="2024-01-02", end="2024-01-15",
            use_cache=False,
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_compute_returns_log(self, sample_df: pd.DataFrame) -> None:
        from kuber.data.price_loader import PriceLoader
        ret = PriceLoader.compute_returns(sample_df, method="log")
        assert ret.shape == sample_df.shape
        # First row should be NaN
        assert ret.iloc[0].isna().all()
        # Log return for SPY day 2: log(471/470)
        expected = np.log(471.0 / 470.0)
        assert abs(ret["SPY"].iloc[1] - expected) < 1e-10

    def test_compute_returns_simple(self, sample_df: pd.DataFrame) -> None:
        from kuber.data.price_loader import PriceLoader
        ret = PriceLoader.compute_returns(sample_df, method="simple")
        assert ret.shape == sample_df.shape
        expected = (471.0 - 470.0) / 470.0
        assert abs(ret["SPY"].iloc[1] - expected) < 1e-10


# ---------------------------------------------------------------------------
# MacroLoader tests
# ---------------------------------------------------------------------------

class TestMacroLoader:
    """Tests for the FRED macro loader."""

    def test_no_api_key_returns_empty(self, cache) -> None:
        """Without FRED_API_KEY the loader should return an empty DF."""
        from kuber.data.macro_loader import MacroLoader
        with patch.dict("os.environ", {}, clear=False):
            # Ensure no key
            import os
            old = os.environ.pop("FRED_API_KEY", None)
            try:
                loader = MacroLoader(cache=cache)
                loader._api_key = None  # force no key
                df = loader.load(use_cache=False)
                assert isinstance(df, pd.DataFrame)
                assert df.empty
            finally:
                if old is not None:
                    os.environ["FRED_API_KEY"] = old

    def test_load_with_mock_fred(self, cache) -> None:
        """With a mocked fredapi, verify the output shape."""
        from kuber.data.macro_loader import MacroLoader

        dates = pd.bdate_range("2024-01-02", periods=20)
        fake_series = pd.Series(np.linspace(4.0, 4.5, 20), index=dates)

        mock_fred_cls = MagicMock()
        mock_fred_inst = MagicMock()
        mock_fred_inst.get_series.return_value = fake_series
        mock_fred_cls.return_value = mock_fred_inst

        loader = MacroLoader(cache=cache)
        loader._api_key = "FAKE_KEY"

        with patch("kuber.data.macro_loader.Fred", mock_fred_cls, create=True):
            # Patch the import inside _download
            import kuber.data.macro_loader as ml_mod
            original_download = loader._download

            def patched_download(indicators, start, end):
                import kuber.data.macro_loader
                with patch.dict(
                    "sys.modules",
                    {"fredapi": MagicMock(Fred=mock_fred_cls)},
                ):
                    # Re-import inside to use patched module
                    from fredapi import Fred
                    fred = Fred(api_key="FAKE_KEY")
                    frames = {}
                    for sid in indicators:
                        frames[sid] = fake_series
                    df = pd.DataFrame(frames)
                    df.index.name = "Date"
                    return df

            loader._download = patched_download

            df = loader.load(
                indicators=["GS10", "GS2"],
                start="2024-01-02",
                end="2024-01-30",
                use_cache=False,
            )
            assert isinstance(df, pd.DataFrame)
            assert "GS10" in df.columns
            assert "GS2" in df.columns


# ---------------------------------------------------------------------------
# SentimentLoader tests
# ---------------------------------------------------------------------------

class TestSentimentLoader:
    """Tests for the sentiment loader (synthetic mode)."""

    def test_synthetic_returns_bounded_scores(self, cache) -> None:
        from kuber.data.sentiment_loader import SentimentLoader

        # Build a small fake price DataFrame
        dates = pd.bdate_range("2024-01-02", periods=30, name="Date")
        prices = pd.DataFrame(
            {
                "SPY": np.linspace(470, 480, 30),
                "QQQ": np.linspace(400, 410, 30),
            },
            index=dates,
        )

        loader = SentimentLoader(provider="synthetic", cache=cache)
        df = loader.score(
            ["SPY", "QQQ"],
            start="2024-01-02",
            end="2024-02-15",
            prices=prices,
            use_cache=False,
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # All values must be in [-1, 1]
        assert (df.min().min() >= -1.0)
        assert (df.max().max() <= 1.0)

    def test_synthetic_is_default_provider(self) -> None:
        from kuber.data.sentiment_loader import SentimentLoader
        loader = SentimentLoader()
        assert loader._provider == "synthetic"

    def test_unknown_provider_falls_back(self) -> None:
        from kuber.data.sentiment_loader import SentimentLoader
        loader = SentimentLoader(provider="nonexistent")
        assert loader._provider == "synthetic"

    def test_synthetic_with_cached_result(self, cache) -> None:
        """Verify that a second call hits the cache."""
        from kuber.data.sentiment_loader import SentimentLoader

        dates = pd.bdate_range("2024-01-02", periods=20, name="Date")
        prices = pd.DataFrame(
            {"SPY": np.linspace(470, 475, 20)}, index=dates,
        )
        loader = SentimentLoader(provider="synthetic", cache=cache)

        # First call populates cache
        df1 = loader.score(
            ["SPY"], start="2024-01-02", end="2024-01-30",
            prices=prices, use_cache=True,
        )
        # Second call should use cache
        df2 = loader.score(
            ["SPY"], start="2024-01-02", end="2024-01-30",
            use_cache=True,
        )
        assert df2 is not None
        pd.testing.assert_frame_equal(df1, df2, check_freq=False)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Tests for the YAML config loader."""

    def test_get_dot_notation(self) -> None:
        from kuber.utils.config import get_config
        cfg = get_config()
        assert cfg.get("data.cache_dir") == ".cache"
        assert cfg.get("project.name") == "KUBER"

    def test_get_missing_returns_default(self) -> None:
        from kuber.utils.config import get_config
        cfg = get_config()
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_get_env(self) -> None:
        from kuber.utils.config import get_config
        import os
        os.environ["_KUBER_TEST_VAR"] = "hello"
        cfg = get_config()
        assert cfg.get_env("_KUBER_TEST_VAR") == "hello"
        del os.environ["_KUBER_TEST_VAR"]
