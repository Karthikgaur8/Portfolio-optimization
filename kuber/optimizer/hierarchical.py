"""Hierarchical Risk Parity (HRP) optimizer."""

import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from kuber.optimizer.base import Optimizer, OptimizationResult
from kuber.optimizer.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class HRPOptimizer(Optimizer):
    """Hierarchical Risk Parity: cluster assets by correlation, then
    allocate via recursive bisection.  Purely risk-based — does not
    use expected returns."""

    def __init__(self, linkage_method: str = "single") -> None:
        self.linkage_method = linkage_method

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
        corr = self._cov_to_corr(Sigma)

        # 1. Correlation distance matrix
        dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
        np.fill_diagonal(dist, 0.0)
        # Make symmetric
        dist = (dist + dist.T) / 2.0

        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=self.linkage_method)

        # 2. Quasi-diagonalise (reorder by dendrogram leaves)
        sort_ix = list(leaves_list(link).astype(int))

        # 3. Recursive bisection
        raw_weights = self._recursive_bisect(Sigma, sort_ix)

        # Apply weight bounds and re-normalize
        raw_weights = np.clip(raw_weights, constraints.min_weight, constraints.max_weight)
        raw_weights = raw_weights / raw_weights.sum()

        # Map back to asset order
        w = np.zeros(n)
        for idx, leaf in enumerate(sort_ix):
            w[leaf] = raw_weights[idx]

        weights = pd.Series(w, index=assets)

        mu = expected_returns.values.astype(float)
        port_ret = float(weights.values @ mu)
        port_vol = float(np.sqrt(weights.values @ Sigma @ weights.values))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return OptimizationResult(
            weights=weights,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            metadata={"optimizer": "hrp", "linkage_method": self.linkage_method},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        d = np.sqrt(np.diag(cov))
        d[d == 0] = 1.0
        return cov / np.outer(d, d)

    @staticmethod
    def _recursive_bisect(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
        """Allocate weights via recursive bisection on sorted assets."""
        n = len(sort_ix)
        w = np.ones(n)

        # Cluster items are stored as list-of-lists; start with all indices
        clusters = [list(range(n))]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Inverse-variance weight for each half
                left_var = HRPOptimizer._cluster_var(cov, [sort_ix[i] for i in left])
                right_var = HRPOptimizer._cluster_var(cov, [sort_ix[i] for i in right])

                total = left_var + right_var
                if total < 1e-16:
                    alpha = 0.5
                else:
                    alpha = 1.0 - left_var / total  # higher var → less weight

                for i in left:
                    w[i] *= alpha
                for i in right:
                    w[i] *= (1.0 - alpha)

                new_clusters.append(left)
                new_clusters.append(right)

            clusters = new_clusters

        return w

    @staticmethod
    def _cluster_var(cov: np.ndarray, indices: list[int]) -> float:
        """Compute inverse-variance portfolio variance for a cluster."""
        sub_cov = cov[np.ix_(indices, indices)]
        ivp = 1.0 / np.diag(sub_cov)
        ivp = ivp / ivp.sum()
        return float(ivp @ sub_cov @ ivp)
