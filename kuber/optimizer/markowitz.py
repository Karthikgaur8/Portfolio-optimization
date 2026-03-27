"""Mean-variance (Markowitz) optimizer using cvxpy."""

import logging

import cvxpy as cp
import numpy as np
import pandas as pd

from kuber.optimizer.base import Optimizer, OptimizationResult
from kuber.optimizer.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class MarkowitzOptimizer(Optimizer):
    """Classic mean-variance optimizer with transaction cost and turnover support.

    Parameters
    ----------
    risk_aversion : float
        Risk-aversion parameter (higher = more conservative).
    long_only : bool
        If True, enforce non-negative weights.
    """

    def __init__(self, risk_aversion: float = 1.0, long_only: bool = True) -> None:
        self.risk_aversion = risk_aversion
        self.long_only = long_only

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
        n = len(expected_returns)
        assets = expected_returns.index.tolist()

        mu = expected_returns.values.astype(float)
        Sigma = cov_matrix.loc[assets, assets].values.astype(float)

        # Decision variable
        w = cp.Variable(n)

        # Objective: maximize expected return - risk_aversion * variance - tc penalty
        ret = mu @ w
        risk = cp.quad_form(w, Sigma)

        tc_penalty = 0.0
        if current_weights is not None:
            w_old = current_weights.reindex(assets, fill_value=0.0).values.astype(float)
            turnover_expr = cp.norm(w - w_old, 1)
            tc_penalty = constraints.transaction_cost_rate * turnover_expr
        else:
            w_old = None
            turnover_expr = None

        objective = cp.Maximize(ret - self.risk_aversion * risk - tc_penalty)

        # Constraints
        cons = [cp.sum(w) == 1]

        # Weight bounds
        min_w = constraints.min_weight if not self.long_only else max(0.0, constraints.min_weight)
        cons.append(w >= min_w)
        cons.append(w <= constraints.max_weight)

        # Turnover constraint
        if turnover_expr is not None and constraints.max_turnover < np.inf:
            cons.append(turnover_expr <= constraints.max_turnover)

        prob = cp.Problem(objective, cons)

        try:
            prob.solve(solver=cp.SCS, warm_start=True, max_iters=10000)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                # Fallback solver
                prob.solve(solver=cp.ECOS, warm_start=True)
        except cp.SolverError:
            prob.solve(solver=cp.ECOS)

        if w.value is None:
            logger.warning("Markowitz optimization failed (status=%s). Returning equal weight.", prob.status)
            weights = pd.Series(1.0 / n, index=assets)
        else:
            raw = np.array(w.value).flatten()
            # Clean up tiny negatives from solver tolerance
            raw = np.maximum(raw, 0.0) if self.long_only else raw
            raw = raw / raw.sum()  # re-normalize
            weights = pd.Series(raw, index=assets)

        port_ret = float(weights.values @ mu)
        port_vol = float(np.sqrt(weights.values @ Sigma @ weights.values))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return OptimizationResult(
            weights=weights,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            metadata={"solver_status": prob.status, "solver": str(prob.solver_stats)},
        )
