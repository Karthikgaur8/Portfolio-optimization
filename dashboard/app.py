"""KUBER Dashboard — main entry point.

Launch with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---- Page configuration (must be first Streamlit call) ----
st.set_page_config(
    page_title="KUBER — Portfolio Research Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Sidebar: Branding at top ----
st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 0.2rem 0;">
        <h1 style="margin:0; font-size:2rem;">KUBER</h1>
        <p style="margin:0; font-size:0.8rem; color:#888;">
        Knowledge-driven, Unified, Backtested,<br>Explainable Research Engine
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()

# ---- Navigation ----
page = st.sidebar.radio(
    "Navigate",
    [
        "Universe & Data",
        "Signals & Regime",
        "Portfolio Optimization",
        "Backtest Results",
        "Investment Memo",
    ],
    label_visibility="collapsed",
)
st.sidebar.divider()

# ---- Shared config controls ----
from kuber.data.universe import list_universes, load_universe

UNIVERSE_INFO = {
    "balanced_etf": ("Diversified ETFs — equities, bonds, gold, intl.", 10),
    "tech_mega": ("US mega-cap tech stocks for sector-focused analysis.", 7),
    "sp500_sectors": ("11 GICS sector ETFs for rotation strategies.", 11),
    "original_legacy": ("Original 2022 script tickers — benchmark reference.", 12),
}

universes = list_universes()
universe_labels = []
for u in universes:
    info = UNIVERSE_INFO.get(u)
    if info:
        universe_labels.append(f"{u}  ({info[1]} assets)")
    else:
        universe_labels.append(u)

selected_label = st.sidebar.selectbox(
    "Universe",
    universe_labels,
    index=universe_labels.index(next(l for l in universe_labels if l.startswith("balanced_etf"))) if any(l.startswith("balanced_etf") for l in universe_labels) else 0,
    key="global_universe",
)
# Extract universe name from label
selected_universe = selected_label.split("  (")[0].strip()

# Show description
info = UNIVERSE_INFO.get(selected_universe)
if info:
    st.sidebar.caption(info[0])

# Load Data button
from datetime import date
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start", value=date(2018, 1, 1), key="global_start")
end_date = col2.date_input("End", value=date(2025, 1, 1), key="global_end")

load_clicked = st.sidebar.button("Load Data", type="primary", key="global_load_data", use_container_width=True)

# Handle data loading
if load_clicked:
    universe = load_universe(selected_universe)
    st.session_state["universe_name"] = selected_universe
    st.session_state["tickers"] = universe["tickers"]
    st.session_state["start_date"] = start_date.strftime("%Y-%m-%d")
    st.session_state["end_date"] = end_date.strftime("%Y-%m-%d")
    st.session_state["universe_desc"] = universe["description"]
    st.session_state["_data_loaded"] = False  # trigger reload on page
elif "universe_name" not in st.session_state:
    # Initialize defaults on first load
    universe = load_universe(selected_universe)
    st.session_state["universe_name"] = selected_universe
    st.session_state["tickers"] = universe["tickers"]
    st.session_state["start_date"] = start_date.strftime("%Y-%m-%d")
    st.session_state["end_date"] = end_date.strftime("%Y-%m-%d")
    st.session_state["universe_desc"] = universe["description"]

st.sidebar.divider()
st.sidebar.caption("Built with KUBER | [GitHub](https://github.com/karthik-parthasarathy/kuber)")

# ---- Render the selected page ----
if page == "Universe & Data":
    from dashboard.views.page_01_universe import render
    render()
elif page == "Signals & Regime":
    from dashboard.views.page_02_signals import render
    render()
elif page == "Portfolio Optimization":
    from dashboard.views.page_03_optimize import render
    render()
elif page == "Backtest Results":
    from dashboard.views.page_04_backtest import render
    render()
elif page == "Investment Memo":
    from dashboard.views.page_05_memo import render
    render()
