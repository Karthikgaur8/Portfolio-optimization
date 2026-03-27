"""Rule-based VIX regime classifier for the KUBER framework.

Simple threshold-based classification:
- VIX < low_threshold  -> bull  (2)
- VIX in [low, high)   -> neutral (1)
- VIX >= high_threshold -> bear (0)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VIXRegimeClassifier:
    """Rule-based regime classifier using VIX thresholds.

    Parameters
    ----------
    low_threshold : float
        Below this VIX level the regime is ``bull`` (default 15).
    high_threshold : float
        At or above this VIX level the regime is ``bear`` (default 25).
    vix_column : str
        Column name for VIX in the macro DataFrame (default ``"VIX"``).
    """

    # Numeric labels consistent with HMMRegimeDetector
    LABEL_MAP = {"bull": 2, "neutral": 1, "bear": 0}

    def __init__(
        self,
        low_threshold: float = 15.0,
        high_threshold: float = 25.0,
        vix_column: str = "VIX",
    ) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.vix_column = vix_column

        logger.info(
            "VIXRegimeClassifier initialized: bull < %.1f, bear >= %.1f",
            low_threshold,
            high_threshold,
        )

    def fit(
        self,
        returns: pd.Series | pd.DataFrame | None = None,
        macro: pd.DataFrame | None = None,
    ) -> "VIXRegimeClassifier":
        """No-op fit (rule-based classifier needs no training).

        Returns
        -------
        VIXRegimeClassifier
            self, for chaining.
        """
        logger.debug("VIXRegimeClassifier.fit() called (no-op).")
        return self

    def predict(
        self,
        returns: pd.Series | pd.DataFrame | None = None,
        macro: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Classify regime based on VIX level.

        Parameters
        ----------
        returns : pd.Series | pd.DataFrame | None
            Ignored; present for API compatibility.
        macro : pd.DataFrame | None
            Must contain the VIX column.

        Returns
        -------
        pd.Series
            Integer regime labels (0=bear, 1=neutral, 2=bull).
        """
        # Try the requested column name, then common alternatives
        vix_col = None
        if macro is not None:
            for candidate in [self.vix_column, "VIXCLS", "VIX", "vix"]:
                if candidate in macro.columns:
                    vix_col = candidate
                    break

        if macro is None or vix_col is None:
            logger.warning(
                "Macro data missing VIX column; returning empty Series."
            )
            return pd.Series(dtype=float, name="regime")

        vix = macro[vix_col].dropna()

        labels = pd.Series(self.LABEL_MAP["neutral"], index=vix.index, name="regime")
        labels[vix < self.low_threshold] = self.LABEL_MAP["bull"]
        labels[vix >= self.high_threshold] = self.LABEL_MAP["bear"]

        logger.info(
            "VIX classification: bull=%d, neutral=%d, bear=%d",
            (labels == 2).sum(),
            (labels == 1).sum(),
            (labels == 0).sum(),
        )
        return labels

    def get_regime_params(self) -> dict:
        """Return threshold configuration.

        Returns
        -------
        dict
            Threshold parameters used for classification.
        """
        return {
            "bull": {"vix_range": f"< {self.low_threshold}"},
            "neutral": {
                "vix_range": f"[{self.low_threshold}, {self.high_threshold})"
            },
            "bear": {"vix_range": f">= {self.high_threshold}"},
        }
