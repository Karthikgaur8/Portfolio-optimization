"""Black-Litterman optimizer with signal-derived views."""

import logging

import numpy as np
import pandas as pd

from kuber.optimizer.base import Optimizer, OptimizationResult
from kuber.optimizer.constraints import PortfolioConstraints
from kuber.optimizer.markowitz import MarkowitzOptimizer

logger = logging.getLogger(__name__)


class BlackLittermanOptimizer(Optimizer):
    """Full Black-Litterman implementation.

    Converts composite signal scores into investor "views", combines with
    market-implied equilibrium returns, and feeds the posterior expected
    returns to a mean-variance optimizer.

    Parameters
    ----------
    risk_aversion : float
        Market risk-aversion parameter (delta).  Default 2.5.
    tau : float
        Uncertainty scaling on the prior (typical range 0.01–0.10).
    confidence_scale : float
        Multiplier that maps signal magnitude to view confidence.
        Higher = more confident views.
    market_cap_weights : pd.Series | None
        Market-capitalisation weights used to derive equilibrium returns.
        If None, equal-weight is assumed as a proxy.
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        confidence_scale: float = 1.0,
        market_cap_weights: pd.Series | None = None,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.confidence_scale = confidence_scale
        self.market_cap_weights = market_cap_weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
        constraints: PortfolioConstraints | None = None,
        current_weights: pd.Series | None = None,
        regime: str | None = None,
    ) -> OptimizationResult:
        assets = expected_returns.index.tolist()
        n = len(assets)
        Sigma = cov_matrix.loc[assets, assets].values.astype(float)

        # --- 1. Market-implied equilibrium returns (pi) ---
        if self.market_cap_weights is not None:
            w_mkt = self.market_cap_weights.reindex(assets, fill_value=1.0 / n).values.astype(float)
            w_mkt = w_mkt / w_mkt.sum()
        else:
            w_mkt = np.ones(n) / n

        pi = self.risk_aversion * Sigma @ w_mkt  # equilibrium excess returns

        # --- 2. Convert signal scores to BL views ---
        # Each asset with a non-zero signal becomes an absolute view.
        signal_scores = expected_returns.values.astype(float)
        view_mask = np.abs(signal_scores) > 1e-8
        k = int(view_mask.sum())

        if k == 0:
            # No views — just use equilibrium returns
            posterior_mu = pd.Series(pi, index=assets)
            return self._solve_mv(posterior_mu, cov_matrix, risk_free_rate, constraints, current_weights)

        # P matrix: k x n pick matrix (absolute views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        view_idx = 0
        for i in range(n):
            if view_mask[i]:
                P[view_idx, i] = 1.0
                # View = equilibrium return + signal-scaled adjustment
                # Signal in [-1, 1] → scale to ±(pi magnitude)
                adjustment_scale = np.abs(pi).mean() if np.abs(pi).mean() > 0 else 0.05
                Q[view_idx] = pi[i] + signal_scores[i] * adjustment_scale
                view_idx += 1

        # --- 3. Omega: view uncertainty (diagonal) ---
        # Stronger signal → lower uncertainty
        omega_diag = np.zeros(k)
        view_idx = 0
        for i in range(n):
            if view_mask[i]:
                confidence = min(abs(signal_scores[i]) * self.confidence_scale, 1.0)
                confidence = max(confidence, 0.05)  # floor
                # Omega_ii = (1/confidence - 1) * tau * P_i @ Sigma @ P_i^T
                view_var = self.tau * P[view_idx] @ Sigma @ P[view_idx]
                omega_diag[view_idx] = view_var * (1.0 / confidence - 1.0)
                view_idx += 1

        Omega = np.diag(omega_diag)

        # --- 4. Posterior expected returns ---
        tau_Sigma = self.tau * Sigma
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        Omega_inv = np.linalg.inv(Omega + np.eye(k) * 1e-10)

        # BL formula: E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]
        M = tau_Sigma_inv + P.T @ Omega_inv @ P
        posterior_mu_arr = np.linalg.solve(M, tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)
        posterior_mu = pd.Series(posterior_mu_arr, index=assets)

        return self._solve_mv(posterior_mu, cov_matrix, risk_free_rate, constraints, current_weights)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _solve_mv(
        self,
        mu: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float,
        constraints: PortfolioConstraints | None,
        current_weights: pd.Series | None,
    ) -> OptimizationResult:
        """Delegate to Markowitz optimizer with posterior returns."""
        mv = MarkowitzOptimizer(risk_aversion=self.risk_aversion, long_only=True)
        result = mv.optimize(mu, cov_matrix, risk_free_rate, constraints, current_weights)
        result.metadata["optimizer"] = "black_litterman"
        return result
