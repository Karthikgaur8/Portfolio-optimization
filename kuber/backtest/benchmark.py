"""Benchmark strategies for backtest comparison."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def equal_weight_returns(prices: pd.DataFrame) -> pd.Series:
    """1/N equal-weight portfolio, rebalanced daily (buy-and-hold drift approximation).

    For simplicity the equal-weight benchmark holds constant 1/N weights
    (equivalent to daily rebalance with zero cost).
    """
    daily_returns = prices.pct_change().dropna(how="all")
    n = daily_returns.shape[1]
    if n == 0:
        return pd.Series(dtype=float, name="equal_weight")
    portfolio_returns = daily_returns.mean(axis=1)
    portfolio_returns.name = "equal_weight"
    return portfolio_returns


def spy_returns(prices: pd.DataFrame) -> pd.Series:
    """Buy-and-hold SPY benchmark.

    If SPY is in the price DataFrame, use it directly.
    Otherwise, try to download it.
    """
    if "SPY" in prices.columns:
        ret = prices["SPY"].pct_change().dropna()
    else:
        try:
            import yfinance as yf

            spy = yf.download("SPY", start=prices.index[0], end=prices.index[-1], progress=False, auto_adjust=True)
            if isinstance(spy.columns, pd.MultiIndex):
                spy_close = spy["Close"].iloc[:, 0] if spy["Close"].shape[1] > 0 else spy["Close"]
            else:
                spy_close = spy["Close"]
            ret = spy_close.pct_change().dropna()
        except Exception as e:
            logger.warning("Could not load SPY data: %s. Returning empty series.", e)
            return pd.Series(dtype=float, name="spy_buy_hold")
    ret.name = "spy_buy_hold"
    return ret


def sixty_forty_returns(prices: pd.DataFrame) -> pd.Series:
    """60% SPY / 40% BND classic balanced portfolio.

    Falls back to equal weight if SPY or BND not available.
    """
    spy_col = "SPY" if "SPY" in prices.columns else None
    bnd_col = "BND" if "BND" in prices.columns else None

    if spy_col is None or bnd_col is None:
        logger.warning("SPY or BND not in prices; 60/40 benchmark unavailable.")
        return pd.Series(dtype=float, name="sixty_forty")

    daily_returns = prices[[spy_col, bnd_col]].pct_change().dropna()
    portfolio_returns = 0.6 * daily_returns[spy_col] + 0.4 * daily_returns[bnd_col]
    portfolio_returns.name = "sixty_forty"
    return portfolio_returns


def legacy_script_returns(prices: pd.DataFrame) -> pd.Series:
    """Run the legacy 2022 optimizer approach as a benchmark.

    Simplified: equal-weight the original universe tickers that exist in prices.
    """
    legacy_tickers = ["SPY", "BND", "GLD", "QQQ", "VTI", "AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
    available = [t for t in legacy_tickers if t in prices.columns]

    if not available:
        logger.warning("No legacy tickers available; returning empty series.")
        return pd.Series(dtype=float, name="legacy_script")

    daily_returns = prices[available].pct_change().dropna()
    portfolio_returns = daily_returns.mean(axis=1)
    portfolio_returns.name = "legacy_script"
    return portfolio_returns


def compute_all_benchmarks(prices: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute all benchmark return series.

    Returns
    -------
    dict[str, pd.Series]
        benchmark_name -> daily return series.
    """
    benchmarks = {}

    ew = equal_weight_returns(prices)
    if len(ew) > 0:
        benchmarks["equal_weight"] = ew

    spy = spy_returns(prices)
    if len(spy) > 0:
        benchmarks["spy_buy_hold"] = spy

    sf = sixty_forty_returns(prices)
    if len(sf) > 0:
        benchmarks["sixty_forty"] = sf

    leg = legacy_script_returns(prices)
    if len(leg) > 0:
        benchmarks["legacy_script"] = leg

    return benchmarks
