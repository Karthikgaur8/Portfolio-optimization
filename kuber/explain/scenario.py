"""Scenario / stress-test analysis for portfolio weights."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Result of applying one historical scenario to the portfolio."""

    name: str
    start: str
    end: str
    portfolio_return: float = 0.0
    max_drawdown: float = 0.0
    best_asset: str = ""
    best_asset_return: float = 0.0
    worst_asset: str = ""
    worst_asset_return: float = 0.0
    equal_weight_return: float = 0.0
    sixty_forty_return: float = 0.0


class ScenarioAnalyzer:
    """Apply historical crisis scenarios to a set of portfolio weights.

    Hard-coded scenarios from the PRD.
    """

    SCENARIOS: dict[str, dict[str, str]] = {
        "covid_crash": {
            "name": "COVID-19 Crash",
            "start": "2020-02-19",
            "end": "2020-03-23",
        },
        "rate_shock_2022": {
            "name": "2022 Rate Shock",
            "start": "2022-01-03",
            "end": "2022-10-12",
        },
        "gfc_2008": {
            "name": "Global Financial Crisis",
            "start": "2008-09-01",
            "end": "2009-03-09",
        },
        "dot_com_bust": {
            "name": "Dot-Com Bust",
            "start": "2000-03-10",
            "end": "2002-10-09",
        },
        "vix_spike_2024": {
            "name": "2024 VIX Spike",
            "start": "2024-07-15",
            "end": "2024-08-15",
        },
    }

    def analyze(
        self,
        weights: pd.Series,
        prices: pd.DataFrame,
    ) -> dict[str, ScenarioResult]:
        """Run all scenarios against the given weights.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights (index = ticker).
        prices : pd.DataFrame
            Full price history (DatetimeIndex × tickers).

        Returns
        -------
        dict[str, ScenarioResult]
            Scenario key -> result.
        """
        results: dict[str, ScenarioResult] = {}

        for key, meta in self.SCENARIOS.items():
            result = self._run_one(key, meta, weights, prices)
            if result is not None:
                results[key] = result

        logger.info("Scenario analysis complete: %d/%d scenarios evaluated.",
                     len(results), len(self.SCENARIOS))
        return results

    # ------------------------------------------------------------------
    def _run_one(
        self,
        key: str,
        meta: dict,
        weights: pd.Series,
        prices: pd.DataFrame,
    ) -> ScenarioResult | None:
        start, end = meta["start"], meta["end"]
        name = meta["name"]

        try:
            window = prices.loc[start:end]
        except Exception:
            window = prices[(prices.index >= start) & (prices.index <= end)]

        if window.empty or len(window) < 2:
            logger.warning("Scenario '%s': no price data in range %s – %s.", name, start, end)
            return None

        # Simple returns over the full period
        period_returns = (window.iloc[-1] / window.iloc[0]) - 1.0
        common_tickers = weights.index.intersection(period_returns.index)

        if common_tickers.empty:
            logger.warning("Scenario '%s': no overlapping tickers.", name)
            return None

        w = weights.reindex(common_tickers).fillna(0)
        w = w / w.sum() if w.sum() != 0 else w  # re-normalise
        ret = period_returns.reindex(common_tickers).fillna(0)

        portfolio_return = float((w * ret).sum())

        # Max drawdown within the window
        daily_ret = window[common_tickers].pct_change().dropna()
        port_daily = (daily_ret * w).sum(axis=1)
        cum = (1 + port_daily).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        max_dd = float(-dd.min()) if len(dd) > 0 else 0.0

        # Best / worst assets
        best_ticker = ret.idxmax()
        worst_ticker = ret.idxmin()

        # Equal-weight benchmark
        ew_return = float(ret.mean())

        # 60/40 benchmark (SPY/BND)
        sixty_forty_return = self._sixty_forty(prices, start, end)

        return ScenarioResult(
            name=name,
            start=start,
            end=end,
            portfolio_return=portfolio_return,
            max_drawdown=max_dd,
            best_asset=str(best_ticker),
            best_asset_return=float(ret[best_ticker]),
            worst_asset=str(worst_ticker),
            worst_asset_return=float(ret[worst_ticker]),
            equal_weight_return=ew_return,
            sixty_forty_return=sixty_forty_return,
        )

    @staticmethod
    def _sixty_forty(prices: pd.DataFrame, start: str, end: str) -> float:
        """Compute 60/40 (SPY/BND) return for the period, or NaN."""
        spy = "SPY" if "SPY" in prices.columns else None
        bnd = "BND" if "BND" in prices.columns else None
        if spy is None:
            return float("nan")

        try:
            window = prices.loc[start:end]
        except Exception:
            window = prices[(prices.index >= start) & (prices.index <= end)]

        if window.empty or len(window) < 2:
            return float("nan")

        spy_ret = (window[spy].iloc[-1] / window[spy].iloc[0]) - 1.0 if spy else 0.0
        bnd_ret = (window[bnd].iloc[-1] / window[bnd].iloc[0]) - 1.0 if bnd and bnd in window.columns else 0.0

        return float(0.6 * spy_ret + 0.4 * bnd_ret)

    @staticmethod
    def format_results(results: dict[str, ScenarioResult]) -> str:
        """Return a human-readable table of scenario results."""
        lines = [
            f"{'Scenario':<25s} {'Portfolio':>10s} {'Max DD':>8s} {'EW':>8s} {'60/40':>8s} {'Best':>12s} {'Worst':>12s}",
            "-" * 93,
        ]
        for _key, r in results.items():
            sf = f"{r.sixty_forty_return:>7.1%}" if not np.isnan(r.sixty_forty_return) else "   N/A "
            lines.append(
                f"{r.name:<25s} {r.portfolio_return:>9.1%} {-r.max_drawdown:>7.1%} "
                f"{r.equal_weight_return:>7.1%} {sf} "
                f"{r.best_asset:>5s}({r.best_asset_return:+.0%}) "
                f"{r.worst_asset:>5s}({r.worst_asset_return:+.0%})"
            )
        return "\n".join(lines)
