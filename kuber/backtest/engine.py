"""Walk-forward backtesting engine — the core of KUBER's evaluation framework.

NO LOOK-AHEAD BIAS: at time t, only data available before t is used.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from kuber.optimizer.base import Optimizer, OptimizationResult
from kuber.optimizer.constraints import PortfolioConstraints
from kuber.backtest.metrics import compute_all_metrics
from kuber.backtest.benchmark import compute_all_benchmarks
from kuber.backtest.report import BacktestResult

logger = logging.getLogger(__name__)

_REBAL_FREQ = {
    "daily": "B",       # business-day
    "weekly": "W-FRI",
    "monthly": "MS",    # month-start (first trading day)
    "quarterly": "QS",
}


class BacktestEngine:
    """Walk-forward backtesting engine.

    Parameters
    ----------
    optimizer : Optimizer
        Portfolio optimizer instance.
    signals : list
        List of Signal instances (from kuber.signals).
    regime_detector : object | None
        A fitted RegimeDetector (or None to skip regime detection).
    constraints : PortfolioConstraints | None
        Portfolio constraints.
    rebalance_frequency : str
        One of ``"daily"``, ``"weekly"``, ``"monthly"``, ``"quarterly"``.
    lookback_window : int
        Trading days of data used for estimation at each rebalance.
    expanding_window : bool
        If True, use all data from the start instead of a rolling window.
    risk_free_rate : float
        Annualized risk-free rate.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        signals: list | None = None,
        regime_detector: Any = None,
        constraints: PortfolioConstraints | None = None,
        rebalance_frequency: str = "monthly",
        lookback_window: int = 252,
        expanding_window: bool = False,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.signals = signals or []
        self.regime_detector = regime_detector
        self.constraints = constraints or PortfolioConstraints()
        self.rebalance_frequency = rebalance_frequency
        self.lookback_window = lookback_window
        self.expanding_window = expanding_window
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BacktestResult:
        """Execute the walk-forward backtest.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (DatetimeIndex, columns = tickers).
        macro : pd.DataFrame | None
            Macro data aligned with price dates.
        sentiment : pd.DataFrame | None
            Sentiment scores aligned with price dates.
        start_date, end_date : str | None
            Backtest window (defaults to full price range).

        Returns
        -------
        BacktestResult
        """
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
        if start_date:
            prices = prices.loc[prices.index >= pd.Timestamp(start_date)]
        if end_date:
            prices = prices.loc[prices.index <= pd.Timestamp(end_date)]

        if prices.empty or prices.shape[1] == 0:
            raise ValueError("Price data is empty after date filtering.")

        # Compute simple returns
        daily_returns = prices.pct_change().dropna(how="all")
        trading_dates = daily_returns.index

        # Determine rebalance dates (must fall on or after a trading date)
        rebalance_schedule = self._get_rebalance_dates(trading_dates)
        # Ensure first rebalance has enough lookback
        min_start_idx = self.lookback_window
        rebalance_schedule = [d for d in rebalance_schedule if trading_dates.get_loc(d) >= min_start_idx]

        if not rebalance_schedule:
            raise ValueError(
                f"Not enough data for even one rebalance. Need {self.lookback_window} days lookback, "
                f"have {len(trading_dates)} trading days."
            )

        logger.info(
            "Backtest: %d trading days, %d rebalances, lookback=%d",
            len(trading_dates), len(rebalance_schedule), self.lookback_window,
        )

        # Storage
        portfolio_returns_list: list[tuple[pd.Timestamp, float]] = []
        weights_history: list[tuple[pd.Timestamp, pd.Series]] = []
        regime_history: list[tuple[pd.Timestamp, Any]] = []
        trade_log_rows: list[dict] = []
        signal_history: dict[str, list[tuple[pd.Timestamp, pd.Series]]] = {}
        turnover_list: list[float] = []

        current_weights: pd.Series | None = None

        for i, rebal_date in enumerate(rebalance_schedule):
            rebal_loc = trading_dates.get_loc(rebal_date)

            # --- Lookback window (no look-ahead) ---
            if self.expanding_window:
                window_start = 0
            else:
                window_start = max(0, rebal_loc - self.lookback_window)

            window_end = rebal_loc  # exclusive of rebalance date itself for estimation
            est_prices = prices.iloc[window_start:window_end + 1]
            est_returns = daily_returns.iloc[window_start:window_end]

            if est_returns.empty or len(est_returns) < 20:
                logger.warning("Skipping rebalance %s: insufficient data.", rebal_date)
                continue

            # Filter to assets with complete data in the window
            valid_assets = est_returns.columns[est_returns.notna().sum() >= len(est_returns) * 0.8].tolist()
            if not valid_assets:
                logger.warning("No valid assets for rebalance %s.", rebal_date)
                continue

            est_returns = est_returns[valid_assets]
            est_prices_filtered = est_prices[valid_assets]

            # --- Generate signals ---
            composite_signal = self._generate_signals(est_prices_filtered, macro, sentiment, rebal_date)

            # --- Detect regime ---
            regime_label = self._detect_regime(est_returns, macro, rebal_date)
            regime_history.append((rebal_date, regime_label))

            # --- Estimate covariance & expected returns ---
            cov_matrix = est_returns.cov() * 252  # annualize
            if composite_signal is not None:
                expected_returns = composite_signal
            else:
                expected_returns = est_returns.mean() * 252

            # Align to valid assets
            expected_returns = expected_returns.reindex(valid_assets, fill_value=0.0)
            cov_matrix = cov_matrix.loc[valid_assets, valid_assets]

            # Align current weights to valid assets
            prev_weights = None
            if current_weights is not None:
                prev_weights = current_weights.reindex(valid_assets, fill_value=0.0)

            # --- Optimize ---
            try:
                result = self.optimizer.optimize(
                    expected_returns=expected_returns,
                    cov_matrix=cov_matrix,
                    risk_free_rate=self.risk_free_rate,
                    constraints=self.constraints,
                    current_weights=prev_weights,
                    regime=regime_label,
                )
                new_weights = result.weights.reindex(valid_assets, fill_value=0.0)
            except Exception as e:
                logger.error("Optimization failed at %s: %s. Using equal weight.", rebal_date, e)
                new_weights = pd.Series(1.0 / len(valid_assets), index=valid_assets)

            # --- Record trades ---
            if prev_weights is not None:
                turnover = float(np.abs(new_weights - prev_weights.reindex(new_weights.index, fill_value=0.0)).sum())
            else:
                turnover = float(np.abs(new_weights).sum())  # initial allocation = full turnover
            turnover_list.append(turnover)

            for ticker in new_weights.index:
                old_w = prev_weights[ticker] if prev_weights is not None and ticker in prev_weights.index else 0.0
                new_w = new_weights[ticker]
                if abs(new_w - old_w) > 1e-6:
                    trade_log_rows.append({
                        "date": rebal_date,
                        "ticker": ticker,
                        "old_weight": float(old_w),
                        "new_weight": float(new_w),
                        "turnover": float(abs(new_w - old_w)),
                    })

            weights_history.append((rebal_date, new_weights.copy()))

            # --- Track portfolio between rebalances ---
            next_rebal_loc = (
                trading_dates.get_loc(rebalance_schedule[i + 1])
                if i + 1 < len(rebalance_schedule)
                else len(trading_dates)
            )

            # Expand weights to full asset set (zeros for missing)
            full_weights = new_weights.reindex(prices.columns, fill_value=0.0)

            # Transaction cost on rebalance day
            tc_cost = self.constraints.transaction_cost_rate * turnover

            # Track portfolio with weight drift
            drifting_weights = full_weights.copy()

            for day_loc in range(rebal_loc, next_rebal_loc):
                day = trading_dates[day_loc]
                day_ret_vec = daily_returns.iloc[day_loc].reindex(prices.columns, fill_value=0.0)

                # Portfolio return = weighted sum of asset returns
                port_ret = float((drifting_weights * day_ret_vec).sum())

                # Apply transaction cost on rebalance day only
                if day_loc == rebal_loc:
                    port_ret -= tc_cost

                portfolio_returns_list.append((day, port_ret))

                # Update drifting weights
                new_vals = drifting_weights * (1 + day_ret_vec)
                total = new_vals.sum()
                if total > 0:
                    drifting_weights = new_vals / total
                else:
                    drifting_weights = full_weights.copy()

            current_weights = drifting_weights.copy()

        # --- Compile results ---
        if not portfolio_returns_list:
            raise ValueError("Backtest produced no returns. Check data and parameters.")

        portfolio_returns = pd.Series(
            {d: r for d, r in portfolio_returns_list}, name="portfolio"
        ).sort_index()

        # Weights DataFrame
        if weights_history:
            w_dates, w_series = zip(*weights_history)
            portfolio_weights = pd.DataFrame(list(w_series), index=list(w_dates))
            portfolio_weights = portfolio_weights.reindex(columns=prices.columns, fill_value=0.0)
        else:
            portfolio_weights = pd.DataFrame()

        # Regime history
        if regime_history:
            r_dates, r_labels = zip(*regime_history)
            regime_series = pd.Series(list(r_labels), index=list(r_dates), name="regime")
        else:
            regime_series = pd.Series(dtype=float, name="regime")

        # Trade log
        trade_log = pd.DataFrame(trade_log_rows) if trade_log_rows else pd.DataFrame()

        # Turnover series
        turnover_series = pd.Series(turnover_list, index=[d for d, _ in weights_history[:len(turnover_list)]])

        # Benchmarks
        benchmark_returns = compute_all_benchmarks(prices)

        # Metrics
        metrics = compute_all_metrics(
            portfolio_returns,
            risk_free_rate=self.risk_free_rate,
            benchmark_returns=benchmark_returns.get("spy_buy_hold"),
            turnover_series=turnover_series,
        )

        return BacktestResult(
            portfolio_returns=portfolio_returns,
            portfolio_weights=portfolio_weights,
            benchmark_returns=benchmark_returns,
            metrics=metrics,
            regime_history=regime_series,
            rebalance_dates=list(rebalance_schedule),
            signal_history={},
            trade_log=trade_log,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_rebalance_dates(self, trading_dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """Generate rebalance dates aligned to actual trading dates."""
        freq = _REBAL_FREQ.get(self.rebalance_frequency, "MS")
        schedule = pd.date_range(
            start=trading_dates[0], end=trading_dates[-1], freq=freq
        )

        # Snap each scheduled date to the nearest following trading date
        rebal_dates = []
        for d in schedule:
            mask = trading_dates >= d
            if mask.any():
                rebal_dates.append(trading_dates[mask][0])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for d in rebal_dates:
            if d not in seen:
                seen.add(d)
                unique.append(d)

        return unique

    def _generate_signals(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None,
        sentiment: pd.DataFrame | None,
        as_of: pd.Timestamp,
    ) -> pd.Series | None:
        """Generate composite signal scores using only data up to as_of."""
        if not self.signals:
            return None

        from kuber.signals.composite import CompositeSignal

        composite = CompositeSignal(self.signals)

        # Slice macro/sentiment to avoid look-ahead
        macro_slice = None
        if macro is not None and not macro.empty:
            macro_slice = macro.loc[macro.index <= as_of]

        sentiment_slice = None
        if sentiment is not None and not sentiment.empty:
            sentiment_slice = sentiment.loc[sentiment.index <= as_of]

        try:
            signal_df = composite.generate(prices, macro=macro_slice, sentiment=sentiment_slice)
            # Take the last row as the current signal scores
            if signal_df.empty:
                return None
            return signal_df.iloc[-1]
        except Exception as e:
            logger.warning("Signal generation failed at %s: %s", as_of, e)
            return None

    def _detect_regime(
        self,
        returns: pd.DataFrame,
        macro: pd.DataFrame | None,
        as_of: pd.Timestamp,
    ) -> int | str | None:
        """Detect current regime using only data up to as_of."""
        if self.regime_detector is None:
            return None

        try:
            # Slice macro to avoid look-ahead
            macro_slice = None
            if macro is not None and not macro.empty:
                macro_slice = macro.loc[macro.index <= as_of]

            # Fit on the estimation window
            self.regime_detector.fit(returns, macro_slice)
            regime_series = self.regime_detector.predict(returns, macro_slice)

            if regime_series.empty:
                return None

            # Return the most recent regime
            return regime_series.iloc[-1]
        except Exception as e:
            logger.warning("Regime detection failed at %s: %s", as_of, e)
            return None
