"""Reusable Plotly chart functions for the KUBER dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Shared colour palette
COLORS = px.colors.qualitative.Set2


def equity_curve_chart(returns_dict: dict[str, pd.Series]) -> go.Figure:
    """Multiple equity curves on one interactive chart.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Strategy name -> daily returns series.
    """
    fig = go.Figure()
    for i, (name, rets) in enumerate(returns_dict.items()):
        cum = (1 + rets).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines", name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=2 if i == 0 else 1.2),
        ))
    fig.update_layout(
        title="Equity Curves",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def drawdown_chart(returns: pd.Series, title: str = "Drawdown") -> go.Figure:
    """Drawdown chart from a daily returns series."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", mode="lines",
        line=dict(color="crimson", width=1),
        fillcolor="rgba(220, 20, 60, 0.25)",
        name="Drawdown",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def weights_area_chart(weights_df: pd.DataFrame) -> go.Figure:
    """Stacked area chart of portfolio weights over time."""
    fig = go.Figure()
    for i, col in enumerate(weights_df.columns):
        fig.add_trace(go.Scatter(
            x=weights_df.index, y=weights_df[col],
            mode="lines", stackgroup="one",
            name=col,
            line=dict(width=0.5, color=COLORS[i % len(COLORS)]),
        ))
    fig.update_layout(
        title="Portfolio Weights Over Time",
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig.update_layout(
        title="Correlation Matrix",
        template="plotly_white",
        width=600, height=500,
    )
    return fig


def monthly_returns_heatmap(returns: pd.Series) -> go.Figure:
    """Year × month heatmap of monthly returns."""
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = df.pivot_table(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.astype(str).tolist(),
        colorscale=[[0, "crimson"], [0.5, "white"], [1, "forestgreen"]],
        zmid=0,
        text=np.where(np.isnan(pivot.values), "", np.vectorize(lambda v: f"{v:.1%}")(pivot.values)),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig.update_layout(
        title="Monthly Returns Heatmap",
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def rolling_sharpe_chart(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.0,
) -> go.Figure:
    """Rolling annualised Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    rolling_sr = (rolling_mean - risk_free_rate) / rolling_std

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_sr.index, y=rolling_sr.values,
        mode="lines", name=f"Rolling {window}-day Sharpe",
        line=dict(color="steelblue", width=1.5),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.add_hline(y=1, line_dash="dot", line_color="green", annotation_text="Sharpe=1")
    fig.update_layout(
        title=f"Rolling {window}-Day Sharpe Ratio",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def regime_overlay_chart(
    prices: pd.Series | pd.DataFrame,
    regimes: pd.Series,
) -> go.Figure:
    """Price line chart with colored background bands for regimes.

    Parameters
    ----------
    prices : pd.Series or single-column DataFrame
        Price series to plot.
    regimes : pd.Series
        Regime labels (0=bear, 1=neutral, 2=bull).
    """
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] == 1:
            prices = prices.iloc[:, 0]
        else:
            # Use equal-weighted average
            prices = prices.mean(axis=1)

    regime_colors = {0: "rgba(255,0,0,0.12)", 1: "rgba(255,165,0,0.10)", 2: "rgba(0,128,0,0.10)"}
    regime_names = {0: "Bear", 1: "Sideways", 2: "Bull"}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        mode="lines", name="Price",
        line=dict(color="black", width=1.5),
    ))

    # Add background bands
    common = prices.index.intersection(regimes.index)
    if len(common) > 0:
        reg_aligned = regimes.reindex(common).ffill().fillna(1)
        prev_regime = None
        band_start = common[0]
        for i, dt in enumerate(common):
            curr = int(reg_aligned.iloc[i]) if not pd.isna(reg_aligned.iloc[i]) else 1
            if curr != prev_regime and prev_regime is not None:
                fig.add_vrect(
                    x0=band_start, x1=dt,
                    fillcolor=regime_colors.get(prev_regime, "rgba(128,128,128,0.1)"),
                    layer="below", line_width=0,
                )
            if curr != prev_regime:
                band_start = dt
                prev_regime = curr
        # Close last band
        if prev_regime is not None:
            fig.add_vrect(
                x0=band_start, x1=common[-1],
                fillcolor=regime_colors.get(prev_regime, "rgba(128,128,128,0.1)"),
                layer="below", line_width=0,
            )

    fig.update_layout(
        title="Price with Regime Overlay",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def signal_dashboard(signal_dict: dict[str, pd.DataFrame]) -> go.Figure:
    """Small multiples chart for each signal (assets as columns)."""
    n = len(signal_dict)
    if n == 0:
        return go.Figure()

    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                         subplot_titles=list(signal_dict.keys()),
                         vertical_spacing=0.05)

    for i, (name, df) in enumerate(signal_dict.items(), 1):
        # Plot the mean signal across assets
        mean_signal = df.mean(axis=1)
        fig.add_trace(go.Scatter(
            x=mean_signal.index, y=mean_signal.values,
            mode="lines", name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=1.2),
            showlegend=True,
        ), row=i, col=1)
        fig.update_yaxes(range=[-1.1, 1.1], row=i, col=1)

    fig.update_layout(
        height=250 * n,
        template="plotly_white",
        title="Signal Dashboard",
        hovermode="x unified",
    )
    return fig


def weights_bar_chart(weights: pd.Series, title: str = "Portfolio Weights") -> go.Figure:
    """Horizontal bar chart of current weights."""
    sorted_w = weights.sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=sorted_w.values,
        y=sorted_w.index,
        orientation="h",
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(sorted_w))],
        text=[f"{v:.1%}" for v in sorted_w.values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Weight",
        xaxis_tickformat=".0%",
        template="plotly_white",
        height=max(300, 35 * len(sorted_w)),
    )
    return fig


def efficient_frontier_chart(
    frontier_vols: list[float] | None = None,
    frontier_rets: list[float] | None = None,
    portfolio_vol: float | None = None,
    portfolio_ret: float | None = None,
) -> go.Figure:
    """Plot efficient frontier with the current portfolio marked."""
    fig = go.Figure()
    if frontier_vols and frontier_rets:
        fig.add_trace(go.Scatter(
            x=frontier_vols, y=frontier_rets,
            mode="lines", name="Efficient Frontier",
            line=dict(color="steelblue", width=2),
        ))
    if portfolio_vol is not None and portfolio_ret is not None:
        fig.add_trace(go.Scatter(
            x=[portfolio_vol], y=[portfolio_ret],
            mode="markers", name="Current Portfolio",
            marker=dict(size=12, color="red", symbol="star"),
        ))
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        template="plotly_white",
    )
    return fig
