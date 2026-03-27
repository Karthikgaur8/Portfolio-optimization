"""Page 2 — Signals & Regime."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kuber.signals.momentum import TSMOMSignal
from kuber.signals.mean_reversion import RSISignal
from kuber.signals.volatility import RealizedVolSignal
from kuber.signals.composite import CompositeSignal
from kuber.regime.detector import RegimeDetector
from dashboard.components.charts import regime_overlay_chart, signal_dashboard


@st.cache_data(ttl=3600, show_spinner="Generating signals...")
def _generate_signals(prices_bytes, tickers):
    """Generate signals (cache-friendly via serialized prices)."""
    prices = pd.read_json(prices_bytes)
    signals = [TSMOMSignal(), RSISignal(), RealizedVolSignal()]
    composite = CompositeSignal(signals)
    composite_result = composite.generate(prices)
    attrib = composite.attribution()
    individual = {}
    for s in signals:
        try:
            individual[s.name] = s.generate(prices).reindex(columns=prices.columns)
        except Exception:
            pass
    return composite_result, individual, attrib


def render() -> None:
    st.header("Signals & Regime")

    st.markdown(
        '<div style="background:#f0f2f6; padding:0.8rem 1rem; border-radius:6px; margin-bottom:1rem;">'
        "KUBER generates multiple trading signals (momentum, mean-reversion, volatility) and detects "
        "the current market regime (bull/bear/sideways). These signals drive the portfolio allocation — "
        "every weight decision traces back to specific signals."
        "</div>",
        unsafe_allow_html=True,
    )

    prices = st.session_state.get("prices")
    if prices is None:
        st.warning("Please load data on the **Universe & Data** page first.")
        return

    # Generate signals
    with st.spinner("Computing signals..."):
        prices_json = prices.to_json()
        tickers = list(prices.columns)
        composite_result, individual_signals, attrib = _generate_signals(prices_json, tuple(tickers))

    st.session_state["signals_individual"] = individual_signals
    st.session_state["signals_composite"] = composite_result
    st.session_state["signal_attribution"] = attrib

    # Signal dashboard (small multiples)
    st.subheader("Individual Signal Time Series")
    if individual_signals:
        fig = signal_dashboard(individual_signals)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No signals generated.")

    st.divider()

    # Composite signal heatmap (assets x time)
    st.subheader("Composite Signal Heatmap")
    if composite_result is not None and not composite_result.empty:
        monthly = composite_result.resample("ME").last()
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Heatmap(
            z=monthly.T.values,
            x=monthly.index.strftime("%Y-%m"),
            y=monthly.columns.tolist(),
            colorscale=[[0, "crimson"], [0.5, "white"], [1, "forestgreen"]],
            zmid=0,
            colorbar_title="Signal",
        ))
        fig.update_layout(
            title="Composite Signal Strength (Assets x Time)",
            template="plotly_white",
            height=max(300, 30 * len(tickers)),
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Regime detection
    st.subheader("Regime Detection")
    regime_method = st.radio("Method", ["vix", "hmm"], horizontal=True, key="p2_regime_method")

    returns = prices.pct_change().dropna()
    with st.spinner("Detecting market regime..."):
        try:
            detector = RegimeDetector(method=regime_method, n_regimes=3)
            if regime_method == "hmm":
                port_returns = returns.mean(axis=1)
                detector.fit(port_returns)
                regimes = detector.predict(port_returns)
            else:
                try:
                    from kuber.data.macro_loader import MacroLoader
                    start = st.session_state.get("start_date", "2018-01-01")
                    end = st.session_state.get("end_date", "2025-01-01")
                    macro = MacroLoader().load(start=start, end=end)
                    if macro.empty:
                        raise ValueError("No macro data")
                    detector.fit(returns.mean(axis=1), macro)
                    regimes = detector.predict(returns.mean(axis=1), macro)
                except Exception:
                    st.info("VIX data unavailable — falling back to HMM.")
                    detector = RegimeDetector(method="hmm", n_regimes=3)
                    port_returns = returns.mean(axis=1)
                    detector.fit(port_returns)
                    regimes = detector.predict(port_returns)

            st.session_state["regimes"] = regimes
            st.session_state["regime_method"] = regime_method

            # Current regime indicator
            if len(regimes) > 0:
                latest = int(regimes.iloc[-1])
                labels = {0: ("Bear / Risk-Off", "red"), 1: ("Sideways", "orange"), 2: ("Bull / Risk-On", "green")}
                label, color = labels.get(latest, (f"Regime {latest}", "gray"))
                st.metric("Current Regime", label)

            # Regime overlay on equal-weighted price
            avg_price = prices.mean(axis=1)
            fig = regime_overlay_chart(avg_price, regimes)
            fig.update_layout(title="Equal-Weighted Price with Regime Overlay")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Regime detection failed: {e}")
