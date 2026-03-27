"""Tests for KUBER regime detection modules.

Uses synthetic data to verify:
- HMM detector produces valid labels (0, 1, 2)
- VIX classifier produces correct labels for known VIX values
- Unified RegimeDetector dispatches correctly
"""

import numpy as np
import pandas as pd
import pytest

from kuber.regime.hmm import HMMRegimeDetector
from kuber.regime.vix_classifier import VIXRegimeClassifier
from kuber.regime.detector import RegimeDetector

N_DAYS = 500


@pytest.fixture()
def returns() -> pd.Series:
    """Synthetic daily return series."""
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=N_DAYS, freq="B")
    return pd.Series(np.random.normal(0.0005, 0.015, N_DAYS), index=dates, name="portfolio")


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Synthetic daily return DataFrame (multiple tickers)."""
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=N_DAYS, freq="B")
    tickers = ["AAPL", "GOOG", "MSFT"]
    return pd.DataFrame(
        np.random.normal(0.0005, 0.015, (N_DAYS, len(tickers))),
        index=dates,
        columns=tickers,
    )


@pytest.fixture()
def macro() -> pd.DataFrame:
    """Synthetic macro DataFrame with VIX and T10Y2Y."""
    np.random.seed(99)
    dates = pd.bdate_range("2022-01-01", periods=N_DAYS, freq="B")
    return pd.DataFrame(
        {
            "VIX": np.random.uniform(10, 35, N_DAYS),
            "T10Y2Y": np.random.normal(0.5, 0.5, N_DAYS),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# HMM Regime Detector
# ---------------------------------------------------------------------------

class TestHMMRegimeDetector:
    def test_fit_predict_series(self, returns: pd.Series, macro: pd.DataFrame) -> None:
        """HMM should fit and predict valid regime labels from a Series."""
        det = HMMRegimeDetector(n_regimes=3)
        det.fit(returns, macro)
        labels = det.predict(returns, macro)
        valid = labels.dropna()
        assert not valid.empty
        assert set(valid.unique()).issubset({0, 1, 2})

    def test_fit_predict_dataframe(
        self, returns_df: pd.DataFrame, macro: pd.DataFrame
    ) -> None:
        """HMM should accept a DataFrame of per-ticker returns."""
        det = HMMRegimeDetector(n_regimes=3)
        det.fit(returns_df, macro)
        labels = det.predict(returns_df, macro)
        valid = labels.dropna()
        assert not valid.empty
        assert set(valid.unique()).issubset({0, 1, 2})

    def test_regime_params(self, returns: pd.Series, macro: pd.DataFrame) -> None:
        det = HMMRegimeDetector(n_regimes=3)
        det.fit(returns, macro)
        params = det.get_regime_params()
        assert "bull" in params
        assert "bear" in params
        assert "neutral" in params
        for label, p in params.items():
            assert "mean_return" in p
            assert "volatility" in p

    def test_predict_before_fit(self, returns: pd.Series) -> None:
        """Predict without fit should return NaN series."""
        det = HMMRegimeDetector()
        labels = det.predict(returns)
        assert labels.isna().all()

    def test_few_observations(self) -> None:
        """HMM should handle very short data without crashing."""
        dates = pd.bdate_range("2023-01-01", periods=10, freq="B")
        short_returns = pd.Series(np.random.normal(0, 0.01, 10), index=dates)
        det = HMMRegimeDetector(n_regimes=3)
        det.fit(short_returns)  # may warn but should not crash


# ---------------------------------------------------------------------------
# VIX Regime Classifier
# ---------------------------------------------------------------------------

class TestVIXRegimeClassifier:
    def test_known_values(self) -> None:
        """Verify correct labels for known VIX levels."""
        dates = pd.bdate_range("2023-01-01", periods=5, freq="B")
        macro_df = pd.DataFrame(
            {"VIX": [10.0, 14.9, 15.0, 24.9, 30.0]}, index=dates
        )
        clf = VIXRegimeClassifier(low_threshold=15, high_threshold=25)
        labels = clf.predict(macro=macro_df)

        assert labels.iloc[0] == 2  # bull: 10 < 15
        assert labels.iloc[1] == 2  # bull: 14.9 < 15
        assert labels.iloc[2] == 1  # neutral: 15 <= 15 < 25
        assert labels.iloc[3] == 1  # neutral: 24.9 < 25
        assert labels.iloc[4] == 0  # bear: 30 >= 25

    def test_custom_thresholds(self) -> None:
        dates = pd.bdate_range("2023-01-01", periods=3, freq="B")
        macro_df = pd.DataFrame({"VIX": [12.0, 20.0, 35.0]}, index=dates)
        clf = VIXRegimeClassifier(low_threshold=13, high_threshold=30)
        labels = clf.predict(macro=macro_df)
        assert labels.iloc[0] == 2  # bull
        assert labels.iloc[1] == 1  # neutral
        assert labels.iloc[2] == 0  # bear

    def test_fit_is_noop(self) -> None:
        clf = VIXRegimeClassifier()
        result = clf.fit()
        assert result is clf

    def test_missing_macro(self) -> None:
        clf = VIXRegimeClassifier()
        labels = clf.predict(macro=None)
        assert labels.empty

    def test_get_regime_params(self) -> None:
        clf = VIXRegimeClassifier(low_threshold=15, high_threshold=25)
        params = clf.get_regime_params()
        assert "bull" in params
        assert "neutral" in params
        assert "bear" in params


# ---------------------------------------------------------------------------
# Unified RegimeDetector
# ---------------------------------------------------------------------------

class TestRegimeDetector:
    def test_hmm_dispatch(self, returns: pd.Series, macro: pd.DataFrame) -> None:
        det = RegimeDetector(method="hmm", n_regimes=3)
        det.fit(returns, macro)
        labels = det.predict(returns, macro)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0, 1, 2})

    def test_vix_dispatch(self, macro: pd.DataFrame) -> None:
        det = RegimeDetector(method="vix", low_threshold=15, high_threshold=25)
        det.fit(pd.Series(dtype=float), macro)  # fit is no-op for VIX
        labels = det.predict(macro=macro)
        assert set(labels.unique()).issubset({0, 1, 2})

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            RegimeDetector(method="invalid")

    def test_get_regime_params_hmm(
        self, returns: pd.Series, macro: pd.DataFrame
    ) -> None:
        det = RegimeDetector(method="hmm")
        det.fit(returns, macro)
        params = det.get_regime_params()
        assert isinstance(params, dict)

    def test_get_regime_params_vix(self) -> None:
        det = RegimeDetector(method="vix")
        params = det.get_regime_params()
        assert "bull" in params
