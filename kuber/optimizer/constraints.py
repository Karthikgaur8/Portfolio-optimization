"""Portfolio constraints dataclass for the KUBER optimizer module."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PortfolioConstraints:
    """Constraints applied during portfolio optimization.

    Attributes
    ----------
    min_weight : float
        Minimum weight per asset (0.0 = no shorting).
    max_weight : float
        Maximum weight per asset.
    max_turnover : float
        Maximum total turnover per rebalance.
    max_sector_weight : float
        Maximum aggregate weight in any single sector.
    transaction_cost_bps : float
        Transaction cost in basis points per trade.
    target_volatility : float | None
        Optional portfolio volatility target (annualized).
    sector_map : dict | None
        Mapping of ticker -> sector for sector constraints.
    """

    min_weight: float = 0.0
    max_weight: float = 0.30
    max_turnover: float = 0.50
    max_sector_weight: float = 0.40
    transaction_cost_bps: float = 10.0
    target_volatility: float | None = None
    sector_map: dict[str, str] | None = None

    @property
    def transaction_cost_rate(self) -> float:
        """Transaction cost as a decimal fraction."""
        return self.transaction_cost_bps / 10_000.0
