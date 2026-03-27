"""Page 1 — Universe & Data overview."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kuber.data.price_loader import PriceLoader
from dashboard.components.charts import correlation_heatmap


@st.cache_data(ttl=3600, show_spinner="Loading price data...")
def _load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    loader = PriceLoader()
    return loader.load(list(tickers), start=start, end=end)


def render() -> None:
    st.header("Universe & Data")

    st.markdown(
        '<div style="background:#f0f2f6; padding:0.8rem 1rem; border-radius:6px; margin-bottom:1rem;">'
        "Select your investment universe — the pool of assets KUBER will allocate across. "
        "Different universes suit different strategies: <b>balanced_etf</b> for diversified portfolios, "
        "<b>tech_mega</b> for sector-focused analysis, <b>sp500_sectors</b> for rotation strategies."
        "</div>",
        unsafe_allow_html=True,
    )

    tickers = st.session_state.get("tickers")
    uname = st.session_state.get("universe_name", "balanced_etf")
    start = st.session_state.get("start_date", "2018-01-01")
    end = st.session_state.get("end_date", "2025-01-01")
    desc = st.session_state.get("universe_desc", "")

    if not tickers:
        st.info("Select a universe and click **Load Data** in the sidebar.")
        return

    st.info(f"**{uname}** — {desc}  |  {len(tickers)} assets")
    st.code(", ".join(tickers), language=None)

    # Load data (cached)
    with st.spinner("Fetching price data..."):
        prices = _load_prices(tuple(tickers), start, end)

    if prices.empty:
        st.error("No price data returned. Check tickers / date range.")
        return

    st.session_state["prices"] = prices
    st.session_state["_data_loaded"] = True

    st.divider()

    # Normalised price chart
    st.subheader("Normalized Prices (base = 100)")
    norm = prices / prices.iloc[0] * 100
    st.line_chart(norm, use_container_width=True)

    st.divider()

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

    st.divider()

    # Correlation heatmap
    st.subheader("Correlation Matrix")
    returns = prices.pct_change().dropna()
    corr = returns.corr()
    fig = correlation_heatmap(corr)
    st.plotly_chart(fig, use_container_width=True)
