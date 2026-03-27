"""Regime-switching optimizer that delegates to sub-optimizers based on detected regime."""

import logging
from typing import Any

import pandas as pd

from kuber.optimizer.base import Optimizer, OptimizationResult
from kuber.optimizer.constraints import PortfolioConstraints
from kuber.optimizer.black_litterman import BlackLittermanOptimizer
from kuber.optimizer.risk_parity import RiskParityOptimizer
from kuber.optimizer.hierarchical import HRPOptimizer

logger = logging.getLogger(__name__)

# Regime label mapping: HMM outputs 0=bear, 1=neutral, 2=bull
REGIME_LABELS = {0: "bear", 1: "sideways", 2: "bull"}

# Default strategy map: regime_label -> (OptimizerClass, kwargs)
DEFAULT_STRATEGIES: dict[str, tuple[type[Optimizer], dict[str, Any]]] = {
    "bull": (BlackLittermanOptimizer, {"risk_aversion": 1.5}),
    "bear": (RiskParityOptimizer, {}),
    "sideways": (HRPOptimizer, {}),
}


class RegimeAwareOptimizer(Optimizer):
    """Routes optimization to a sub-optimizer based on the current regime.

    Parameters
    ----------
    regime_strategies : dict | None
        Mapping of regime label -> (OptimizerClass, config_dict).
        Defaults to bull->BlackLitterman, bear->RiskParity, sideways->HRP.
    fallback_regime : str
        Regime label to use when the detected regime is unknown.
    """

    def __init__(
        self,
        regime_strategies: dict[str, tuple[type[Optimizer], dict[str, Any]]] | None = None,
        fallback_regime: str = "sideways",
    ) -> None:
        self.regime_strategies = regime_strategies or DEFAULT_STRATEGIES
        self.fallback_regime = fallback_regime
        self._sub_optimizers: dict[str, Optimizer] = {}

        # Pre-instantiate sub-optimizers
        for label, (cls, kwargs) in self.regime_strategies.items():
            self._sub_optimizers[label] = cls(**kwargs)

        logger.info(
            "RegimeAwareOptimizer initialized: %s",
            {k: type(v).__name__ for k, v in self._sub_optimizers.items()},
        )

    def optimize(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
        constraints: PortfolioConstraints | None = None,
        current_weights: pd.Series | None = None,
        regime: str | None = None,
    ) -> OptimizationResult:
        # Resolve regime label
        regime_label = self._resolve_regime(regime)
        optimizer = self._sub_optimizers.get(regime_label)

        if optimizer is None:
            logger.warning("No optimizer for regime '%s'; using fallback '%s'.", regime_label, self.fallback_regime)
            optimizer = self._sub_optimizers[self.fallback_regime]
            regime_label = self.fallback_regime

        logger.info("Regime='%s' -> delegating to %s.", regime_label, type(optimizer).__name__)

        result = optimizer.optimize(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
            constraints=constraints,
            current_weights=current_weights,
            regime=regime_label,
        )
        result.metadata["regime"] = regime_label
        result.metadata["delegated_to"] = type(optimizer).__name__
        return result

    def _resolve_regime(self, regime: str | int | None) -> str:
        """Convert numeric regime code to string label."""
        if regime is None:
            return self.fallback_regime
        if isinstance(regime, (int, float)):
            return REGIME_LABELS.get(int(regime), self.fallback_regime)
        return str(regime)
