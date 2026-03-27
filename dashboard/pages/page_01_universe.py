"""Page 1 — Universe & Data overview."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kuber.data.price_loader import PriceLoader
from kuber.data.universe import list_universes, load_universe
from dashboard.components.charts import correlation_heatmap


@st.cache_data(show_spinner="Loading price data...")
def _load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    loader = PriceLoader()
    return loader.load(list(tickers), start=start, end=end)


def render() -> None:
    st.header("📊 Universe & Data")

    # Controls
    col_a, col_b, col_c = st.columns([2, 1, 1])
    universes = list_universes()
    with col_a:
        uname = st.selectbox("Select Universe", universes,
                              index=universes.index("balanced_etf") if "balanced_etf" in universes else 0,
                              key="p1_universe")
    universe = load_universe(uname)
    tickers = universe["tickers"]

    with col_b:
        start = st.date_input("Start", value=date(2020, 1, 1), key="p1_start")
    with col_c:
        end = st.date_input("End", value=date(2025, 1, 1), key="p1_end")

    st.info(f"**{universe['name']}** — {universe['description']}")
    st.code(", ".join(tickers), language=None)

    # Persist selections
    st.session_state["tickers"] = tickers
    st.session_state["universe_name"] = uname
    st.session_state["start_date"] = start.strftime("%Y-%m-%d")
    st.session_state["end_date"] = end.strftime("%Y-%m-%d")

    # Load data
    prices = _load_prices(tuple(tickers), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if prices.empty:
        st.error("No price data returned. Check tickers / date range.")
        return

    st.session_state["prices"] = prices

    # Normalised price chart
    st.subheader("Normalized Prices (base = 100)")
    norm = prices / prices.iloc[0] * 100
    st.line_chart(norm, use_container_width=True)

    # Data quality
    st.subheader("Data Quality")
    missing_pct = prices.isna().mean() * 100
    quality = pd.DataFrame({
        "Ticker": prices.columns,
        "Start": [prices[c].first_valid_index().strftime("%Y-%m-%d") if prices[c].first_valid_index() is not None else "N/A" for c in prices.columns],
        "End": [prices[c].last_valid_index().strftime("%Y-%m-%d") if prices[c].last_valid_index() is not None else "N/A" for c in prices.columns],
        "Missing %": [f"{missing_pct[c]:.2f}%" for c in prices.columns],
        "Days": [prices[c].notna().sum() for c in prices.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

    # Correlation heatmap
    st.subheader("Correlation Matrix")
    returns = prices.pct_change().dropna()
    corr = returns.corr()
    fig = correlation_heatmap(corr)
    st.plotly_chart(fig, use_container_width=True)
