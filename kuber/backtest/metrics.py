"""Performance metrics for backtesting.

Each metric is a standalone function.  ``compute_all_metrics`` collects
them into a single dictionary.
"""

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized arithmetic mean return."""
    return float(returns.mean() * periods_per_year)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of returns."""
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio."""
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (downside deviation)."""
    ann_ret = annualized_return(returns, periods_per_year)
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf") if ann_ret > risk_free_rate else 0.0
    downside_vol = float(downside.std() * np.sqrt(periods_per_year))
    if downside_vol == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / downside_vol)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (returned as a positive number)."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    return float(-drawdowns.min()) if len(drawdowns) > 0 else 0.0


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return / max drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    ann_ret = annualized_return(returns, periods_per_year)
    return float(ann_ret / mdd)


def win_rate(returns: pd.Series) -> float:
    """Fraction of positive-return periods."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def average_turnover(turnover_series: pd.Series) -> float:
    """Mean portfolio turnover across rebalance dates."""
    if turnover_series is None or len(turnover_series) == 0:
        return 0.0
    return float(turnover_series.mean())


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Information ratio: alpha / tracking error vs benchmark."""
    # Align dates
    common = returns.index.intersection(benchmark_returns.index)
    if len(common) == 0:
        return 0.0
    excess = returns.loc[common] - benchmark_returns.loc[common]
    te = excess.std() * np.sqrt(periods_per_year)
    if te == 0:
        return 0.0
    alpha = excess.mean() * periods_per_year
    return float(alpha / te)


def tail_ratio(returns: pd.Series) -> float:
    """abs(95th percentile / 5th percentile) — upside vs downside."""
    p95 = np.percentile(returns.dropna(), 95)
    p5 = np.percentile(returns.dropna(), 5)
    if p5 == 0:
        return 0.0
    return float(abs(p95 / p5))


def compute_all_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    benchmark_returns: pd.Series | None = None,
    turnover_series: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute all performance metrics and return as a dictionary."""
    metrics = {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "tail_ratio": tail_ratio(returns),
    }

    if turnover_series is not None:
        metrics["average_turnover"] = average_turnover(turnover_series)

    if benchmark_returns is not None:
        metrics["information_ratio"] = information_ratio(
            returns, benchmark_returns, periods_per_year
        )

    return metrics
