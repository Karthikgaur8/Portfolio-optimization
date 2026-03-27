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

# ---- Navigation ----
pages = {
    "Universe & Data": "dashboard.pages.01_universe",
    "Signals & Regime": "dashboard.pages.02_signals",
    "Portfolio Optimization": "dashboard.pages.03_optimize",
    "Backtest Results": "dashboard.pages.04_backtest",
    "Investment Memo": "dashboard.pages.05_memo",
}

st.sidebar.title("KUBER")
st.sidebar.caption("Knowledge-driven, Unified, Backtested, Explainable Research Engine")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
st.sidebar.markdown("---")

# ---- Render the selected page ----
if page == "Universe & Data":
    from dashboard.pages.page_01_universe import render
    render()
elif page == "Signals & Regime":
    from dashboard.pages.page_02_signals import render
    render()
elif page == "Portfolio Optimization":
    from dashboard.pages.page_03_optimize import render
    render()
elif page == "Backtest Results":
    from dashboard.pages.page_04_backtest import render
    render()
elif page == "Investment Memo":
    from dashboard.pages.page_05_memo import render
    render()
