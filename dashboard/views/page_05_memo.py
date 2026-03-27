"""Page 5 — Investment Memo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kuber.explain.attribution import WeightAttributor
from kuber.explain.scenario import ScenarioAnalyzer
from kuber.explain.memo_generator import MemoGenerator


def render() -> None:
    st.header("Investment Memo")

    st.markdown(
        '<div style="background:#f0f2f6; padding:0.8rem 1rem; border-radius:6px; margin-bottom:1rem;">'
        "Generate an AI-powered investment memo explaining the current allocation in plain English — "
        "covering regime assessment, position rationale, risk factors, and stress test results. "
        "This is the <b>explainable AI</b> capstone."
        "</div>",
        unsafe_allow_html=True,
    )

    result = st.session_state.get("backtest_result")
    prices = st.session_state.get("prices")
    opt_result = st.session_state.get("opt_result")

    if result is None and opt_result is None:
        st.warning(
            "Run a **backtest** (page 4) or **optimization** (page 3) first to generate a memo."
        )
        return

    # Get weights
    if result is not None and result.portfolio_weights is not None and not result.portfolio_weights.empty:
        weights = result.portfolio_weights.iloc[-1]
    elif opt_result is not None:
        weights = opt_result.weights
    else:
        st.error("No portfolio weights available.")
        return

    # Provider selection
    provider = st.radio(
        "Memo Provider",
        ["template", "claude"],
        horizontal=True,
        help="'template' works without API key. 'claude' requires ANTHROPIC_API_KEY.",
        key="p5_provider",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        generate = st.button("Generate Memo", type="primary", key="p5_gen")

    if generate:
        with st.spinner("Generating investment memo..."):
            memo, scenario_results, risk_alert = _generate_memo(
                weights, result, prices, provider
            )
            st.session_state["memo"] = memo
            st.session_state["scenario_results_memo"] = scenario_results
            st.session_state["risk_alert"] = risk_alert

    # Display memo
    memo = st.session_state.get("memo")
    if memo:
        st.divider()
        st.markdown(memo)

        # Download button
        st.download_button(
            "Download Memo (.md)",
            data=memo,
            file_name="kuber_investment_memo.md",
            mime="text/markdown",
        )

    # Risk alert
    risk_alert = st.session_state.get("risk_alert")
    if risk_alert:
        st.divider()
        st.warning(risk_alert)

    # Scenario analysis
    scenario_results = st.session_state.get("scenario_results_memo")
    if scenario_results:
        st.divider()
        st.subheader("Scenario Analysis Details")

        for key, r in scenario_results.items():
            with st.expander(f"{r.name} ({r.start} -> {r.end})"):
                cols = st.columns(4)
                cols[0].metric("Portfolio Return", f"{r.portfolio_return:.1%}")
                cols[1].metric("Max Drawdown", f"{-r.max_drawdown:.1%}")
                cols[2].metric("Best Asset", f"{r.best_asset} ({r.best_asset_return:+.1%})")
                cols[3].metric("Worst Asset", f"{r.worst_asset} ({r.worst_asset_return:+.1%})")

                comp_cols = st.columns(2)
                comp_cols[0].metric("Equal-Weight Return", f"{r.equal_weight_return:.1%}")
                sf = f"{r.sixty_forty_return:.1%}" if not np.isnan(r.sixty_forty_return) else "N/A"
                comp_cols[1].metric("60/40 Return", sf)


def _generate_memo(weights, result, prices, provider):
    """Generate the memo, scenarios, and risk alert."""
    metrics = result.metrics if result else {}

    # Attribution
    attribution_df = st.session_state.get("attribution")
    if attribution_df is None:
        from kuber.signals.momentum import TSMOMSignal
        from kuber.signals.mean_reversion import RSISignal
        from kuber.signals.volatility import RealizedVolSignal
        from kuber.signals.composite import CompositeSignal

        if prices is not None:
            signals = [TSMOMSignal(), RSISignal(), RealizedVolSignal()]
            composite = CompositeSignal(signals)
            composite.generate(prices)
            attrib_data = composite.attribution()
            signal_values = {}
            for sname, df in attrib_data.items():
                if not df.empty:
                    signal_values[sname] = df.iloc[-1]
            attributor = WeightAttributor()
            attribution_df = attributor.attribute(weights, signal_values, composite.weights)

    # Scenario analysis
    scenario_results = {}
    if prices is not None:
        analyzer = ScenarioAnalyzer()
        scenario_results = analyzer.analyze(weights, prices)

    # Regime info
    regimes = st.session_state.get("regimes")
    current_regime = "Unknown"
    regime_method = "N/A"
    if regimes is not None and len(regimes) > 0:
        current_regime = regimes.iloc[-1]
        regime_method = st.session_state.get("regime_method", "HMM").upper()

    # Generate
    generator = MemoGenerator(provider=provider)
    memo = generator.generate(
        weights=weights,
        attribution=attribution_df,
        regime=current_regime,
        metrics=metrics,
        scenario_results=scenario_results,
        regime_method=regime_method,
    )

    risk_alert = generator.generate_risk_alert(weights, current_regime, scenario_results)

    return memo, scenario_results, risk_alert
