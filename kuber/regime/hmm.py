"""Hidden Markov Model regime detector for the KUBER framework.

Uses ``hmmlearn.GaussianHMM`` to identify market regimes (bull, neutral, bear)
from return and macro features.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """Gaussian HMM-based market regime detector.

    Parameters
    ----------
    n_regimes : int
        Number of hidden states (default 3).
    n_iter : int
        Maximum EM iterations (default 200).
    random_state : int
        Random seed for reproducibility.
    """

    # Regime labels assigned after sorting by mean return
    LABELS = {0: "bear", 1: "neutral", 2: "bull"}

    def __init__(
        self,
        n_regimes: int = 3,
        n_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self._model = None
        self._label_map: dict[int, int] | None = None
        self._regime_params: dict | None = None

    def _build_features(
        self,
        returns: pd.Series | pd.DataFrame,
        macro: pd.DataFrame | None = None,
        vol_window: int = 21,
    ) -> np.ndarray:
        """Build feature matrix for the HMM.

        Features:
        1. Mean daily return across portfolio
        2. Rolling 21-day volatility
        3. VIX level (if available in macro)
        4. Yield curve slope (if available in macro)

        Returns
        -------
        np.ndarray
            2-D array (n_samples, n_features) with no NaN rows.
        """
        if isinstance(returns, pd.DataFrame):
            mean_ret = returns.mean(axis=1)
        else:
            mean_ret = returns

        features = pd.DataFrame(index=mean_ret.index)
        features["return"] = mean_ret
        features["vol"] = mean_ret.rolling(window=vol_window, min_periods=vol_window).std()

        if macro is not None:
            if "VIX" in macro.columns:
                features["vix"] = macro["VIX"].reindex(features.index).ffill()
            if "T10Y2Y" in macro.columns:
                features["yield_curve"] = macro["T10Y2Y"].reindex(features.index).ffill()

        features = features.dropna()
        return features.values, features.index

    def fit(
        self,
        returns: pd.Series | pd.DataFrame,
        macro: pd.DataFrame | None = None,
    ) -> "HMMRegimeDetector":
        """Fit the HMM to historical data.

        Parameters
        ----------
        returns : pd.Series | pd.DataFrame
            Daily returns (Series or DataFrame of per-ticker returns).
        macro : pd.DataFrame | None
            Optional macro data with VIX / T10Y2Y columns.

        Returns
        -------
        HMMRegimeDetector
            self, for chaining.
        """
        logger.info("Fitting HMM regime detector with %d regimes.", self.n_regimes)

        X, idx = self._build_features(returns, macro)
        if len(X) < self.n_regimes * 10:
            logger.warning(
                "Only %d observations available; HMM may not converge.", len(X)
            )

        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]

            # Try "full" covariance first; fall back to "diag" if it fails
            # (e.g. near-singular covariance matrices).
            model = None
            for cov_type in ("full", "diag"):
                try:
                    candidate = GaussianHMM(
                        n_components=self.n_regimes,
                        covariance_type=cov_type,
                        n_iter=self.n_iter,
                        random_state=self.random_state,
                    )
                    candidate.fit(X)
                    model = candidate
                    logger.info("HMM fitted with covariance_type='%s'.", cov_type)
                    break
                except Exception as inner_exc:
                    logger.warning(
                        "HMM fit with covariance_type='%s' failed: %s", cov_type, inner_exc
                    )

            if model is None:
                raise RuntimeError("All covariance types failed.")

            self._model = model
        except Exception as exc:
            logger.error("HMM fitting failed: %s", exc)
            self._model = None
            self._label_map = None
            self._regime_params = None
            return self

        # Determine label mapping: sort regimes by mean of the first feature
        # (portfolio mean return) so that highest = bull, lowest = bear.
        raw_means = model.means_[:, 0]  # mean return per state
        sorted_indices = np.argsort(raw_means)  # ascending: bear, neutral, bull
        self._label_map = {int(sorted_indices[i]): i for i in range(self.n_regimes)}

        # Store regime parameters
        self._regime_params = {}
        for raw_state, mapped_state in self._label_map.items():
            label = self.LABELS.get(mapped_state, f"regime_{mapped_state}")
            self._regime_params[label] = {
                "mean_return": float(model.means_[raw_state, 0]),
                "volatility": float(np.sqrt(model.covars_[raw_state, 0, 0])),
            }

        logger.info("HMM fitting complete. Regime params: %s", self._regime_params)
        return self

    def predict(
        self,
        returns: pd.Series | pd.DataFrame,
        macro: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Predict regime labels.

        Parameters
        ----------
        returns : pd.Series | pd.DataFrame
            Daily returns.
        macro : pd.DataFrame | None
            Optional macro data.

        Returns
        -------
        pd.Series
            Integer regime labels (0=bear, 1=neutral, 2=bull)
            indexed by date.
        """
        if self._model is None:
            logger.warning("Model not fitted; returning NaN series.")
            idx = returns.index if isinstance(returns, pd.Series) else returns.index
            return pd.Series(np.nan, index=idx, name="regime")

        X, idx = self._build_features(returns, macro)
        raw_labels = self._model.predict(X)

        if self._label_map is not None:
            mapped = np.array([self._label_map.get(int(l), l) for l in raw_labels])
        else:
            mapped = raw_labels

        return pd.Series(mapped, index=idx, name="regime")

    def get_regime_params(self) -> dict:
        """Return mean/vol parameters per regime.

        Returns
        -------
        dict
            Keys are regime labels (``"bull"``, ``"neutral"``, ``"bear"``),
            values are dicts with ``"mean_return"`` and ``"volatility"``.
        """
        if self._regime_params is None:
            logger.warning("Model not fitted; no regime params available.")
            return {}
        return self._regime_params
