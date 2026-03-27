"""Page 3 — Portfolio Optimization."""

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
from kuber.signals.momentum import TSMOMSignal
from kuber.signals.mean_reversion import RSISignal
from kuber.signals.volatility import RealizedVolSignal
from kuber.signals.composite import CompositeSignal
from kuber.explain.attribution import WeightAttributor
from dashboard.components.charts import weights_bar_chart, efficient_frontier_chart

OPTIMIZERS = {
    "markowitz": lambda: MarkowitzOptimizer(risk_aversion=1.0),
    "risk_parity": lambda: RiskParityOptimizer(),
    "black_litterman": lambda: BlackLittermanOptimizer(risk_aversion=2.5),
    "hrp": lambda: HRPOptimizer(),
    "regime_aware": lambda: RegimeAwareOptimizer(),
}


def render() -> None:
    st.header("Portfolio Optimization")

    st.markdown(
        '<div style="background:#f0f2f6; padding:0.8rem 1rem; border-radius:6px; margin-bottom:1rem;">'
        "Run an optimizer to compute optimal portfolio weights. Compare multiple strategies: "
        "classical Markowitz, Risk Parity, Black-Litterman with signal-derived views, "
        "Hierarchical Risk Parity, or the Regime-Aware switcher. View weight attribution "
        "to see <b>WHY</b> each asset got its allocation."
        "</div>",
        unsafe_allow_html=True,
    )

    prices = st.session_state.get("prices")
    if prices is None:
        st.warning("Please load data on the **Universe & Data** page first.")
        return

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        opt_choice = st.selectbox("Optimizer", list(OPTIMIZERS.keys()), index=4, key="p3_opt")
    with col2:
        max_weight = st.slider("Max Weight per Asset", 0.05, 1.0, 0.30, 0.05, key="p3_maxw")

    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=max_weight,
        max_turnover=0.50,
        transaction_cost_bps=10,
    )

    if st.button("Run Optimizer", type="primary", key="p3_run"):
        with st.spinner("Optimizing portfolio..."):
            returns = prices.pct_change().dropna()

            # Generate signals
            signals = [TSMOMSignal(), RSISignal(), RealizedVolSignal()]
            composite = CompositeSignal(signals)
            composite_signal = composite.generate(prices)

            # Expected returns from latest composite signal
            expected_returns = composite_signal.iloc[-1] if not composite_signal.empty else pd.Series(0, index=prices.columns)
            expected_returns = expected_returns * 0.10  # +/-10% annual

            # Covariance
            cov_matrix = returns.cov() * 252

            # Regime
            regimes = st.session_state.get("regimes")
            current_regime = None
            if regimes is not None and len(regimes) > 0:
                regime_val = regimes.iloc[-1]
                regime_map = {0: "bear", 1: "sideways", 2: "bull"}
                current_regime = regime_map.get(int(regime_val), None) if not pd.isna(regime_val) else None

            # Run optimizer
            try:
                optimizer = OPTIMIZERS[opt_choice]()
                result = optimizer.optimize(
                    expected_returns=expected_returns,
                    cov_matrix=cov_matrix,
                    risk_free_rate=0.02,
                    constraints=constraints,
                    regime=current_regime,
                )

                st.session_state["opt_result"] = result
                st.session_state["opt_name"] = opt_choice

                st.success("Optimization complete!")

                st.divider()

                # Weights bar chart
                st.subheader("Optimal Weights")
                fig = weights_bar_chart(result.weights, title=f"{opt_choice.replace('_', ' ').title()} Weights")
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Expected Return", f"{result.expected_return:.2%}")
                col_b.metric("Expected Volatility", f"{result.expected_volatility:.2%}")
                col_c.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

                st.divider()

                # Weight attribution
                st.subheader("Weight Attribution")
                attrib_data = composite.attribution()
                signal_values = {}
                for sname, df in attrib_data.items():
                    if not df.empty:
                        signal_values[sname] = df.iloc[-1]

                attributor = WeightAttributor()
                attr_df = attributor.attribute(result.weights, signal_values, composite.weights)
                st.session_state["attribution"] = attr_df

                st.dataframe(
                    attr_df.style.format("{:.2%}").background_gradient(
                        cmap="RdYlGn", axis=None,
                        subset=[c for c in attr_df.columns if c != "final_weight"]
                    ),
                    use_container_width=True,
                )

                # Efficient frontier (Markowitz only)
                if opt_choice == "markowitz":
                    st.divider()
                    st.subheader("Efficient Frontier")
                    _plot_frontier(returns, cov_matrix, expected_returns, constraints,
                                   result.expected_volatility, result.expected_return)

                st.divider()

                # Benchmark comparison
                st.subheader("Comparison to Benchmarks")
                n = len(prices.columns)
                ew = pd.Series(1.0 / n, index=prices.columns)
                comp_data = {
                    "Strategy": [opt_choice, "Equal Weight"],
                    "Exp. Return": [f"{result.expected_return:.2%}", f"{float(expected_returns.mean()):.2%}"],
                    "Exp. Vol": [f"{result.expected_volatility:.2%}", f"{float(np.sqrt(ew @ cov_matrix @ ew)):.2%}"],
                    "Sharpe": [f"{result.sharpe_ratio:.2f}", "---"],
                }
                st.table(pd.DataFrame(comp_data))

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def _plot_frontier(returns, cov_matrix, expected_returns, constraints, port_vol, port_ret):
    """Generate and display an efficient frontier."""
    try:
        vols, rets = [], []
        for ra in np.linspace(0.1, 20, 30):
            opt = MarkowitzOptimizer(risk_aversion=ra)
            try:
                r = opt.optimize(expected_returns, cov_matrix, risk_free_rate=0.02,
                                  constraints=constraints)
                vols.append(r.expected_volatility)
                rets.append(r.expected_return)
            except Exception:
                pass

        if vols:
            fig = efficient_frontier_chart(vols, rets, port_vol, port_ret)
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass
