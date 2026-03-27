"""Page 4 — Backtest Results (THE MONEY PAGE)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kuber.optimizer.markowitz import MarkowitzOptimizer
from kuber.optimizer.risk_parity import RiskParityOptimizer
from kuber.optimizer.black_litterman import BlackLittermanOptimizer
from kuber.optimizer.hierarchical import HRPOptimizer
from kuber.optimizer.regime_aware import RegimeAwareOptimizer
from kuber.optimizer.constraints import PortfolioConstraints
from kuber.regime.detector import RegimeDetector
from kuber.signals.momentum import TSMOMSignal
from kuber.signals.mean_reversion import RSISignal
from kuber.signals.volatility import RealizedVolSignal
from kuber.backtest.engine import BacktestEngine
from kuber.backtest.metrics import compute_all_metrics
from dashboard.components.charts import (
    equity_curve_chart,
    drawdown_chart,
    rolling_sharpe_chart,
    monthly_returns_heatmap,
    weights_area_chart,
)

OPTIMIZERS = {
    "markowitz": lambda: MarkowitzOptimizer(risk_aversion=1.0),
    "risk_parity": lambda: RiskParityOptimizer(),
    "black_litterman": lambda: BlackLittermanOptimizer(risk_aversion=2.5),
    "hrp": lambda: HRPOptimizer(),
    "regime_aware": lambda: RegimeAwareOptimizer(),
}


def render() -> None:
    st.header("Backtest Results")

    st.markdown(
        '<div style="background:#f0f2f6; padding:0.8rem 1rem; border-radius:6px; margin-bottom:1rem;">'
        "Walk-forward backtest: train on historical data, test on future data, slide forward — "
        "no look-ahead bias. Compare KUBER strategies against naive benchmarks (equal weight, "
        "60/40, buy-and-hold SPY) with proper transaction cost modeling."
        "</div>",
        unsafe_allow_html=True,
    )

    prices = st.session_state.get("prices")
    if prices is None:
        st.warning("Please load data on the **Universe & Data** page first.")
        return

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        opt_choice = st.selectbox("Optimizer", list(OPTIMIZERS.keys()), index=4, key="p4_opt")
    with col2:
        rebalance = st.selectbox("Rebalance", ["monthly", "weekly", "quarterly"], key="p4_rebal")
    with col3:
        lookback = st.number_input("Lookback (days)", 60, 504, 252, key="p4_lookback")

    if st.button("Run Backtest", type="primary", key="p4_run"):
        _run_backtest(prices, opt_choice, rebalance, lookback)

    # Show cached results
    result = st.session_state.get("backtest_result")
    if result is not None:
        _display_results(result)
    else:
        st.info("Click **Run Backtest** to start. This may take a minute on first run.")


def _run_backtest(prices, opt_choice, rebalance, lookback):
    with st.spinner("Running walk-forward backtest... this may take a moment."):
        start = st.session_state.get("start_date", "2018-01-01")
        end = st.session_state.get("end_date", "2025-01-01")

        signals = [TSMOMSignal(), RSISignal(), RealizedVolSignal()]
        regime_detector = RegimeDetector(method="hmm", n_regimes=3)

        constraints = PortfolioConstraints(
            min_weight=0.0, max_weight=0.30,
            max_turnover=0.50, transaction_cost_bps=10,
        )

        optimizer = OPTIMIZERS[opt_choice]()
        engine = BacktestEngine(
            optimizer=optimizer,
            signals=signals,
            regime_detector=regime_detector,
            constraints=constraints,
            rebalance_frequency=rebalance,
            lookback_window=lookback,
        )

        result = engine.run(prices, start_date=start, end_date=end)
        st.session_state["backtest_result"] = result
        st.session_state["backtest_optimizer"] = opt_choice
        st.success("Backtest complete!")


def _display_results(result):
    """Display the full backtest results dashboard."""
    # Top-line metrics
    st.subheader("Performance Summary")
    m = result.metrics
    cols = st.columns(6)
    metrics_display = [
        ("Ann. Return", m.get("annualized_return", 0), True),
        ("Ann. Vol", m.get("annualized_volatility", 0), True),
        ("Sharpe", m.get("sharpe_ratio", 0), False),
        ("Sortino", m.get("sortino_ratio", 0), False),
        ("Max DD", m.get("max_drawdown", 0), True),
        ("Win Rate", m.get("win_rate", 0), True),
    ]
    for col, (label, val, pct) in zip(cols, metrics_display):
        col.metric(label, f"{val:.1%}" if pct else f"{val:.2f}")

    st.divider()

    # 1. Equity curve
    st.subheader("Equity Curves")
    curves = {"KUBER": result.portfolio_returns}
    for bm_name, bm_ret in result.benchmark_returns.items():
        curves[bm_name] = bm_ret
    fig = equity_curve_chart(curves)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 2. Drawdown
    st.subheader("Drawdown")
    fig = drawdown_chart(result.portfolio_returns, title="Portfolio Drawdown")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 3. Rolling Sharpe
    st.subheader("Rolling 12-Month Sharpe Ratio")
    fig = rolling_sharpe_chart(result.portfolio_returns, window=252)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 4. Monthly returns heatmap
    st.subheader("Monthly Returns Heatmap")
    fig = monthly_returns_heatmap(result.portfolio_returns)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 5. Full metrics table
    st.subheader("Detailed Metrics Comparison")
    _metrics_table(result)

    # 6. Weights over time
    if result.portfolio_weights is not None and not result.portfolio_weights.empty:
        st.divider()
        st.subheader("Weights Over Time")
        fig = weights_area_chart(result.portfolio_weights)
        st.plotly_chart(fig, use_container_width=True)

    # 7. Trade log / turnover
    if result.trade_log is not None and not result.trade_log.empty:
        st.divider()
        st.subheader("Trade Log & Turnover")
        with st.expander("View Trade Log"):
            st.dataframe(result.trade_log.head(200), use_container_width=True)

        if "turnover" in result.trade_log.columns and "date" in result.trade_log.columns:
            turnover_by_date = result.trade_log.groupby("date")["turnover"].sum()
            if not turnover_by_date.empty:
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(
                    x=turnover_by_date.index,
                    y=turnover_by_date.values,
                    marker_color="steelblue",
                ))
                fig.update_layout(
                    title="Portfolio Turnover at Rebalance",
                    xaxis_title="Date",
                    yaxis_title="Turnover",
                    yaxis_tickformat=".0%",
                    template="plotly_white",
                    font=dict(family="Inter, sans-serif"),
                )
                st.plotly_chart(fig, use_container_width=True)


def _metrics_table(result):
    """Build and display a metrics comparison table."""
    all_strats = {"KUBER": result.metrics}
    for bm_name, bm_ret in result.benchmark_returns.items():
        all_strats[bm_name] = compute_all_metrics(bm_ret)

    metric_keys = [
        "annualized_return", "annualized_volatility", "sharpe_ratio",
        "sortino_ratio", "max_drawdown", "calmar_ratio", "win_rate", "tail_ratio",
    ]
    pct_metrics = {"annualized_return", "annualized_volatility", "max_drawdown", "win_rate"}

    rows = []
    for mk in metric_keys:
        row = {"Metric": mk.replace("_", " ").title()}
        for sname, sm in all_strats.items():
            val = sm.get(mk, float("nan"))
            if mk in pct_metrics:
                row[sname] = f"{val:.2%}"
            else:
                row[sname] = f"{val:.3f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
