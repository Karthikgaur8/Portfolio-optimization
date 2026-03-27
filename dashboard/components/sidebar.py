"""Shared sidebar controls for the KUBER dashboard."""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kuber.data.universe import list_universes, load_universe


def render_sidebar() -> dict:
    """Render the global sidebar and return a dict of user selections.

    Returns
    -------
    dict with keys:
        universe_name, tickers, start_date, end_date,
        optimizer, rebalance_frequency
    """
    st.sidebar.title("KUBER")
    st.sidebar.caption("Portfolio Research Engine")
    st.sidebar.markdown("---")

    # Universe selection
    universes = list_universes()
    universe_name = st.sidebar.selectbox(
        "Universe",
        universes,
        index=universes.index("balanced_etf") if "balanced_etf" in universes else 0,
        key="sidebar_universe",
    )
    universe = load_universe(universe_name)
    tickers = universe["tickers"]
    st.sidebar.caption(f"{universe['description']}")
    st.sidebar.code(", ".join(tickers), language=None)

    # Date range
    st.sidebar.markdown("### Date Range")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", value=date(2020, 1, 1), key="sidebar_start")
    end_date = col2.date_input("End", value=date(2025, 1, 1), key="sidebar_end")

    # Optimizer
    optimizer_options = ["markowitz", "risk_parity", "black_litterman", "hrp", "regime_aware"]
    optimizer = st.sidebar.selectbox(
        "Optimizer",
        optimizer_options,
        index=optimizer_options.index("regime_aware"),
        key="sidebar_optimizer",
    )

    # Rebalance frequency
    freq_options = ["daily", "weekly", "monthly", "quarterly"]
    rebalance = st.sidebar.selectbox(
        "Rebalance Frequency",
        freq_options,
        index=freq_options.index("monthly"),
        key="sidebar_rebalance",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit + Plotly")

    return {
        "universe_name": universe_name,
        "tickers": tickers,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "optimizer": optimizer,
        "rebalance_frequency": rebalance,
    }
