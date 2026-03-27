"""Shared sidebar controls for the KUBER dashboard.

Note: The main sidebar is now rendered directly in app.py.
This module is kept for backward compatibility but is no longer
the primary sidebar renderer.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kuber.data.universe import list_universes, load_universe


def render_sidebar() -> dict:
    """Render the global sidebar and return a dict of user selections."""
    universes = list_universes()
    universe_name = st.sidebar.selectbox(
        "Universe",
        universes,
        index=universes.index("balanced_etf") if "balanced_etf" in universes else 0,
        key="sidebar_universe",
    )
    universe = load_universe(universe_name)
    tickers = universe["tickers"]

    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", value=date(2018, 1, 1), key="sidebar_start")
    end_date = col2.date_input("End", value=date(2025, 1, 1), key="sidebar_end")

    return {
        "universe_name": universe_name,
        "tickers": tickers,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
