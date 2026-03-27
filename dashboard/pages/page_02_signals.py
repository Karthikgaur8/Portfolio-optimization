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
from kuber.data.price_loader import PriceLoader
from dashboard.components.charts import regime_overlay_chart, signal_dashboard


@st.cache_data(show_spinner="Generating signals...")
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
    st.header("📡 Signals & Regime")

    prices = st.session_state.get("prices")
    if prices is None:
        st.warning("Please load data on the **Universe & Data** page first.")
        return

    # Generate signals
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

    # Composite signal heatmap (assets × time)
    st.subheader("Composite Signal Heatmap")
    if composite_result is not None and not composite_result.empty:
        # Downsample for display
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
            title="Composite Signal Strength (Assets × Time)",
            template="plotly_white",
            height=max(300, 30 * len(tickers)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Regime detection
    st.subheader("Regime Detection")
    regime_method = st.radio("Method", ["vix", "hmm"], horizontal=True, key="p2_regime_method")

    returns = prices.pct_change().dropna()
    try:
        detector = RegimeDetector(method=regime_method, n_regimes=3)
        if regime_method == "hmm":
            port_returns = returns.mean(axis=1)
            detector.fit(port_returns)
            regimes = detector.predict(port_returns)
        else:
            # VIX classifier needs macro data — use simple proxy if unavailable
            try:
                from kuber.data.macro_loader import MacroLoader
                start = st.session_state.get("start_date", "2020-01-01")
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
            labels = {0: ("Bear / Risk-Off", "🔴"), 1: ("Sideways", "🟡"), 2: ("Bull / Risk-On", "🟢")}
            label, icon = labels.get(latest, (f"Regime {latest}", "⚪"))
            st.metric("Current Regime", f"{icon} {label}")

        # Regime overlay on equal-weighted price
        avg_price = prices.mean(axis=1)
        fig = regime_overlay_chart(avg_price, regimes)
        fig.update_layout(title="Equal-Weighted Price with Regime Overlay")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Regime detection failed: {e}")
