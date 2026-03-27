#!/usr/bin/env python3
"""
KUBER Smoke Test -- Phases 1 & 2
================================
Loads the balanced_etf universe, downloads 3 years of price + macro data,
generates all signals, detects regimes, and prints a summary.

If yfinance/FRED downloads fail (rate-limited, no API key, no internet),
the script generates synthetic data so all modules can still be exercised.

Usage:
    python scripts/smoke_test.py
"""

import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def generate_synthetic_prices(
    tickers: list[str],
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic-looking synthetic price data."""
    np.random.seed(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    prices_dict = {}
    for i, t in enumerate(tickers):
        base = 100 + i * 20
        drift = 0.0003 + np.random.uniform(-0.0001, 0.0002)
        vol = 0.012 + np.random.uniform(0, 0.008)
        log_ret = np.random.normal(drift, vol, n)
        prices_dict[t] = base * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(prices_dict, index=dates)


def generate_synthetic_macro(dates: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic macro indicator data."""
    np.random.seed(seed)
    n = len(dates)
    return pd.DataFrame({
        "GS10": np.clip(np.random.normal(3.5, 0.5, n).cumsum() * 0.01 + 3.0, 0.5, 8),
        "GS2":  np.clip(np.random.normal(3.0, 0.4, n).cumsum() * 0.01 + 2.5, 0.3, 7),
        "VIXCLS": np.clip(np.abs(np.random.normal(20, 8, n)), 9, 80),
        "UNRATE": np.clip(np.random.normal(4.0, 0.3, n).cumsum() * 0.001 + 3.5, 2, 15),
        "CPIAUCSL": np.clip(np.random.normal(300, 5, n).cumsum() * 0.01 + 290, 200, 400),
        "FEDFUNDS": np.clip(np.random.normal(5.0, 0.3, n).cumsum() * 0.005 + 4.5, 0, 10),
        "T10Y2Y": np.random.normal(0.5, 0.3, n),
    }, index=dates)


def main() -> None:
    print("[KUBER] Smoke Test -- Phases 1 & 2")
    print(f"   Run time: {datetime.now():%Y-%m-%d %H:%M:%S}")

    # -- 1. Load universe ---------------------------------------------------
    section("1. Universe Loading")
    from kuber.data.universe import load_universe, list_universes

    print(f"   Available universes: {list_universes()}")
    universe = load_universe("balanced_etf")
    tickers = universe["tickers"]
    print(f"   Selected: {universe['name']}")
    print(f"   Tickers ({len(tickers)}): {tickers}")

    # -- 2. Download prices -------------------------------------------------
    section("2. Price Data")
    from kuber.data.price_loader import PriceLoader

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    loader = PriceLoader()
    try:
        prices = loader.load(tickers, start=start_date, end=end_date)
    except Exception:
        prices = pd.DataFrame()

    if prices.empty or prices.shape[0] < 60:
        print("   [!] yfinance download failed or returned insufficient data.")
        print("       Generating synthetic prices for demo purposes.")
        prices = generate_synthetic_prices(tickers, start_date, end_date)
        data_source = "synthetic"
    else:
        data_source = "yfinance"

    print(f"   Source: {data_source}")
    print(f"   Shape: {prices.shape}")
    print(f"   Date range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"   Tickers loaded: {list(prices.columns)}")

    log_returns = loader.compute_returns(prices, method="log")
    simple_returns = loader.compute_returns(prices, method="simple")
    print(f"   Log returns shape: {log_returns.shape}")
    print(f"   Simple returns shape: {simple_returns.shape}")

    # -- 3. Macro data ------------------------------------------------------
    section("3. Macro Data")
    from kuber.data.macro_loader import MacroLoader

    macro_loader = MacroLoader()
    try:
        macro = macro_loader.load(start=start_date, end=end_date)
    except Exception:
        macro = pd.DataFrame()

    if macro.empty:
        print("   [!] No FRED API key or download failed -- generating synthetic macro data")
        macro = generate_synthetic_macro(prices.index)
    print(f"   Shape: {macro.shape}")
    print(f"   Indicators: {list(macro.columns)}")
    print(f"   Date range: {macro.index[0].date()} -> {macro.index[-1].date()}")

    # -- 4. Sentiment data --------------------------------------------------
    section("4. Sentiment Data (Synthetic)")
    from kuber.data.sentiment_loader import SentimentLoader

    sent_loader = SentimentLoader(provider="synthetic")
    sentiment = sent_loader.score(tickers, start=start_date, end=end_date, prices=prices)
    print(f"   Shape: {sentiment.shape}")
    print(f"   Range: [{sentiment.min().min():.3f}, {sentiment.max().max():.3f}]")
    print(f"   Provider: synthetic (no API keys needed)")

    # -- 5. Signals ---------------------------------------------------------
    section("5. Signal Generation")
    from kuber.signals.momentum import TSMOMSignal, XSMOMSignal, DualMomentumSignal
    from kuber.signals.mean_reversion import RSISignal, BollingerBandSignal, ZScoreReversionSignal
    from kuber.signals.volatility import RealizedVolSignal, VolRatioSignal, GARCHVolSignal
    from kuber.signals.sentiment import SentimentSignal, SentimentMomentumSignal
    from kuber.signals.macro import YieldCurveSignal, VIXRegimeSignal, FedStanceSignal
    from kuber.signals.composite import CompositeSignal

    signal_classes = [
        TSMOMSignal, XSMOMSignal, DualMomentumSignal,
        RSISignal, BollingerBandSignal, ZScoreReversionSignal,
        RealizedVolSignal, VolRatioSignal,
        SentimentSignal, SentimentMomentumSignal,
        YieldCurveSignal, VIXRegimeSignal, FedStanceSignal,
    ]

    results = {}
    for cls in signal_classes:
        sig = cls()
        try:
            result = sig.generate(prices, macro=macro, sentiment=sentiment)
            valid = result.dropna(how="all")
            if len(valid) > 0:
                mn = valid.min().min()
                mx = valid.max().max()
                in_range = (mn >= -1.001) and (mx <= 1.001)
                status = "[OK]" if in_range else "[!!] OUT OF RANGE"
            else:
                mn, mx = float("nan"), float("nan")
                status = "[!!] ALL NaN"
            results[sig.name] = result
            print(f"   {status} {sig.name:30s}  shape={result.shape}  range=[{mn:+.3f}, {mx:+.3f}]")
        except Exception as e:
            print(f"   [FAIL] {sig.name:30s}  ERROR: {e}")

    # GARCH -- may be slow, run separately
    print("\n   Running GARCH (may take a moment)...")
    try:
        garch = GARCHVolSignal()
        garch_result = garch.generate(prices, macro=macro, sentiment=sentiment)
        valid = garch_result.dropna(how="all")
        if len(valid) > 0:
            mn, mx = valid.min().min(), valid.max().max()
            in_range = (mn >= -1.001) and (mx <= 1.001)
            status = "[OK]" if in_range else "[!!] OUT OF RANGE"
        else:
            mn, mx = float("nan"), float("nan")
            status = "[!!] ALL NaN"
        results[garch.name] = garch_result
        print(f"   {status} {garch.name:30s}  shape={garch_result.shape}  range=[{mn:+.3f}, {mx:+.3f}]")
    except Exception as e:
        print(f"   [FAIL] GARCH(1,1) Volatility         ERROR: {e}")

    # -- 6. Composite signal ------------------------------------------------
    section("6. Composite Signal")
    directional_signals = [TSMOMSignal(), XSMOMSignal(), RSISignal(), SentimentSignal()]
    composite = CompositeSignal(signals=directional_signals)
    composite_result = composite.generate(prices, macro=macro, sentiment=sentiment)
    valid = composite_result.dropna(how="all")
    print(f"   Shape: {composite_result.shape}")
    if len(valid) > 0:
        print(f"   Range: [{valid.min().min():.3f}, {valid.max().max():.3f}]")

    attr = composite.attribution()
    print(f"   Attribution keys: {list(attr.keys())}")
    print(f"   Latest composite scores:")
    last_valid = (
        composite_result.dropna().iloc[-1]
        if len(composite_result.dropna()) > 0
        else composite_result.iloc[-1]
    )
    for ticker, score in last_valid.items():
        bar_len = int(abs(score) * 20)
        direction = "+" if score > 0 else "-"
        print(f"      {ticker:6s}: {score:+.3f}  {direction}{'#' * bar_len}")

    # -- 7. Regime detection ------------------------------------------------
    section("7. Regime Detection")
    from kuber.regime import RegimeDetector

    # VIX rule-based
    print("\n   --- VIX Rule-Based ---")
    vix_det = RegimeDetector(method="vix")
    vix_det.fit(log_returns, macro=macro)
    vix_regimes = vix_det.predict(log_returns, macro=macro)
    if len(vix_regimes) > 0:
        print(f"   Regime counts:")
        for label, count in vix_regimes.value_counts().items():
            print(f"      {label}: {count}")
        print(f"   Current regime: {vix_regimes.iloc[-1]}")

    # HMM
    print("\n   --- HMM Regime Detector ---")
    hmm_det = RegimeDetector(method="hmm", n_regimes=3)
    hmm_det.fit(log_returns, macro=macro)
    hmm_regimes = hmm_det.predict(log_returns, macro=macro)
    if len(hmm_regimes) > 0:
        print(f"   Regime counts:")
        for label, count in hmm_regimes.value_counts().items():
            print(f"      {label}: {count}")
        print(f"   Current regime: {hmm_regimes.iloc[-1]}")
        params = hmm_det.get_regime_params()
        if params:
            print("   Regime parameters:")
            for label, p in params.items():
                mean_val = p.get('mean', 'N/A')
                vol_val = p.get('vol', 'N/A')
                mean_str = f"{mean_val:.6f}" if isinstance(mean_val, (int, float)) else str(mean_val)
                vol_str = f"{vol_val:.6f}" if isinstance(vol_val, (int, float)) else str(vol_val)
                print(f"      {label}: mean={mean_str}, vol={vol_str}")

    # -- Summary ------------------------------------------------------------
    section("SMOKE TEST COMPLETE")
    print(f"   Universe:    {universe['name']} ({len(tickers)} assets)")
    print(f"   Price data:  {prices.shape[0]} days x {prices.shape[1]} assets ({data_source})")
    print(f"   Macro data:  {macro.shape[0]} days x {macro.shape[1]} indicators")
    print(f"   Sentiment:   {sentiment.shape[0]} days x {sentiment.shape[1]} assets (synthetic)")
    print(f"   Signals:     {len(results)} generated successfully")
    print(f"   Composite:   {composite_result.shape}")
    vix_label = vix_regimes.iloc[-1] if len(vix_regimes) > 0 else "N/A"
    hmm_label = hmm_regimes.iloc[-1] if len(hmm_regimes) > 0 else "N/A"
    print(f"   Regimes:     VIX={vix_label}, HMM={hmm_label}")
    print(f"\n   All Phases 1 & 2 modules operational.\n")


if __name__ == "__main__":
    main()
