"""Abstract base class for portfolio optimizers and OptimizationResult dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class OptimizationResult:
    """Container for optimization output."""

    weights: pd.Series  # asset -> weight
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Optimizer(ABC):
    """Abstract base class for all portfolio optimizers."""

    @abstractmethod
    def optimize(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
        constraints: Any = None,
        current_weights: pd.Series | None = None,
        regime: str | None = None,
    ) -> OptimizationResult:
        """Run portfolio optimization.

        Parameters
        ----------
        expected_returns : pd.Series
            Signal-derived expected returns per asset.
        cov_matrix : pd.DataFrame
            Covariance matrix of asset returns.
        risk_free_rate : float
            Risk-free rate (annualized).
        constraints : PortfolioConstraints | None
            Portfolio constraints.
        current_weights : pd.Series | None
            Current portfolio weights (for turnover penalty).
        regime : str | None
            Current market regime label.

        Returns
        -------
        OptimizationResult
        """
        ...
