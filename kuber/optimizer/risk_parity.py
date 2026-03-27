"""Equal Risk Contribution (Risk Parity) optimizer."""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from kuber.optimizer.base import Optimizer, OptimizationResult
from kuber.optimizer.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class RiskParityOptimizer(Optimizer):
    """Allocate so each asset contributes equal marginal risk to the portfolio."""

    def optimize(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
        constraints: PortfolioConstraints | None = None,
        current_weights: pd.Series | None = None,
        regime: str | None = None,
    ) -> OptimizationResult:
        constraints = constraints or PortfolioConstraints()
        assets = expected_returns.index.tolist()
        n = len(assets)
        Sigma = cov_matrix.loc[assets, assets].values.astype(float)

        # Target: equal risk contribution = 1/n each
        target_rc = np.ones(n) / n

        def _risk_contribution(w: np.ndarray) -> np.ndarray:
            port_vol = np.sqrt(w @ Sigma @ w)
            if port_vol < 1e-12:
                return np.ones(n) / n
            marginal = Sigma @ w
            rc = w * marginal / port_vol
            return rc

        def _objective(w: np.ndarray) -> float:
            rc = _risk_contribution(w)
            rc_norm = rc / rc.sum() if rc.sum() > 0 else rc
            return float(np.sum((rc_norm - target_rc) ** 2))

        x0 = np.ones(n) / n
        bounds = [(max(1e-6, constraints.min_weight), constraints.max_weight)] * n
        cons_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            _objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons_list,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if result.success:
            raw = np.maximum(result.x, 0.0)
            raw = raw / raw.sum()
            weights = pd.Series(raw, index=assets)
        else:
            logger.warning("Risk parity optimization failed: %s. Using equal weight.", result.message)
            weights = pd.Series(1.0 / n, index=assets)

        mu = expected_returns.values.astype(float)
        port_ret = float(weights.values @ mu)
        port_vol = float(np.sqrt(weights.values @ Sigma @ weights.values))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return OptimizationResult(
            weights=weights,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            metadata={"solver_success": result.success, "solver_message": str(result.message)},
        )
