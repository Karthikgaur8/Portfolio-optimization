"""BacktestResult dataclass — container for full backtest output."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from kuber.backtest.metrics import compute_all_metrics


@dataclass
class BacktestResult:
    """Container for a completed backtest.

    Attributes
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.
    portfolio_weights : pd.DataFrame
        Weights over time (rows = dates, columns = tickers).
    benchmark_returns : dict[str, pd.Series]
        Benchmark name -> daily return series.
    metrics : dict[str, float]
        All computed performance metrics.
    regime_history : pd.Series
        Detected regimes over time.
    rebalance_dates : list[datetime]
        Dates on which the portfolio was rebalanced.
    signal_history : dict[str, pd.DataFrame]
        Per-signal values over time.
    trade_log : pd.DataFrame
        Columns: date, ticker, old_weight, new_weight, turnover.
    """

    portfolio_returns: pd.Series
    portfolio_weights: pd.DataFrame
    benchmark_returns: dict[str, pd.Series] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    regime_history: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rebalance_dates: list = field(default_factory=list)
    signal_history: dict[str, pd.DataFrame] = field(default_factory=dict)
    trade_log: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    def summary(self) -> str:
        """Human-readable performance summary."""
        lines = ["=" * 60, "KUBER Backtest Summary", "=" * 60]

        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:<25s}: {v:>10.4f}")
            else:
                lines.append(f"  {k:<25s}: {v}")

        lines.append(f"\n  Rebalance count: {len(self.rebalance_dates)}")
        lines.append(f"  Date range: {self.portfolio_returns.index[0].date()} -> {self.portfolio_returns.index[-1].date()}")
        lines.append("=" * 60)

        # Benchmark comparison
        if self.benchmark_returns:
            lines.append("\nBenchmark Comparison:")
            lines.append(f"  {'Strategy':<20s} {'Ann. Return':>12s} {'Ann. Vol':>10s} {'Sharpe':>8s} {'Max DD':>8s}")
            lines.append("  " + "-" * 58)

            # Strategy
            m = self.metrics
            lines.append(
                f"  {'KUBER':<20s} {m.get('annualized_return', 0):>12.4f} "
                f"{m.get('annualized_volatility', 0):>10.4f} "
                f"{m.get('sharpe_ratio', 0):>8.4f} "
                f"{m.get('max_drawdown', 0):>8.4f}"
            )

            for bm_name, bm_ret in self.benchmark_returns.items():
                bm_metrics = compute_all_metrics(bm_ret)
                lines.append(
                    f"  {bm_name:<20s} {bm_metrics['annualized_return']:>12.4f} "
                    f"{bm_metrics['annualized_volatility']:>10.4f} "
                    f"{bm_metrics['sharpe_ratio']:>8.4f} "
                    f"{bm_metrics['max_drawdown']:>8.4f}"
                )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializable version for JSON export / dashboard."""
        result: dict = {
            "metrics": self.metrics,
            "rebalance_dates": [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in self.rebalance_dates],
        }

        if self.portfolio_returns is not None:
            result["portfolio_returns"] = {
                str(d.date()): float(v)
                for d, v in self.portfolio_returns.items()
                if pd.notna(v)
            }

        if self.portfolio_weights is not None and not self.portfolio_weights.empty:
            result["portfolio_weights"] = {
                str(d.date()): row.to_dict()
                for d, row in self.portfolio_weights.iterrows()
            }

        if self.regime_history is not None and len(self.regime_history) > 0:
            result["regime_history"] = {
                str(d.date()) if hasattr(d, "date") else str(d): int(v)
                for d, v in self.regime_history.items()
                if pd.notna(v)
            }

        if self.benchmark_returns:
            bm = {}
            for name, s in self.benchmark_returns.items():
                bm_metrics = compute_all_metrics(s)
                bm[name] = bm_metrics
            result["benchmark_metrics"] = bm

        if self.trade_log is not None and not self.trade_log.empty:
            result["trade_count"] = len(self.trade_log)

        return result
