"""Unified regime detection interface for the KUBER framework.

Delegates to either :class:`HMMRegimeDetector` or :class:`VIXRegimeClassifier`
depending on the chosen method.
"""

import logging

import pandas as pd

from kuber.regime.hmm import HMMRegimeDetector
from kuber.regime.vix_classifier import VIXRegimeClassifier

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Unified regime detection interface.

    Parameters
    ----------
    method : str
        Detection method: ``"hmm"`` or ``"vix"`` (default ``"hmm"``).
    n_regimes : int
        Number of regimes for HMM (default 3, ignored for VIX).
    **kwargs
        Forwarded to the underlying detector constructor.
    """

    _METHODS = {"hmm", "vix"}

    def __init__(
        self,
        method: str = "hmm",
        n_regimes: int = 3,
        **kwargs,
    ) -> None:
        method = method.lower()
        if method not in self._METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {self._METHODS}."
            )

        self.method = method

        if method == "hmm":
            self._detector = HMMRegimeDetector(n_regimes=n_regimes, **kwargs)
        else:
            self._detector = VIXRegimeClassifier(**kwargs)

        logger.info("RegimeDetector initialized with method='%s'.", method)

    def fit(
        self,
        returns: pd.Series | pd.DataFrame,
        macro: pd.DataFrame | None = None,
    ) -> "RegimeDetector":
        """Fit the underlying detector.

        Parameters
        ----------
        returns : pd.Series | pd.DataFrame
            Daily returns.
        macro : pd.DataFrame | None
            Optional macro data.

        Returns
        -------
        RegimeDetector
            self, for chaining.
        """
        self._detector.fit(returns, macro)
        return self

    def predict(
        self,
        returns: pd.Series | pd.DataFrame | None = None,
        macro: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Predict regime labels.

        Parameters
        ----------
        returns : pd.Series | pd.DataFrame | None
            Daily returns (required for HMM, optional for VIX).
        macro : pd.DataFrame | None
            Optional macro data.

        Returns
        -------
        pd.Series
            Integer regime labels (0=bear, 1=neutral, 2=bull).
        """
        return self._detector.predict(returns, macro)

    def get_regime_params(self) -> dict:
        """Return regime parameters from the underlying detector.

        Returns
        -------
        dict
            Regime parameters (mean/vol for HMM, thresholds for VIX).
        """
        return self._detector.get_regime_params()
