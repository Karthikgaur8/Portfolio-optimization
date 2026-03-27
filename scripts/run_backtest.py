#!/usr/bin/env python3
"""CLI script to run a full KUBER backtest pipeline.

Usage
-----
    python scripts/run_backtest.py \
        --universe balanced_etf \
        --start 2020-01-01 \
        --end 2025-01-01 \
        --optimizer regime_aware \
        --rebalance monthly
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kuber.data.price_loader import PriceLoader
from kuber.data.macro_loader import MacroLoader
from kuber.data.universe import load_universe
from kuber.optimizer.markowitz import MarkowitzOptimizer
from kuber.optimizer.risk_parity import RiskParityOptimizer
from kuber.optimizer.black_litterman import BlackLittermanOptimizer
from kuber.optimizer.hierarchical import HRPOptimizer
from kuber.optimizer.regime_aware import RegimeAwareOptimizer
from kuber.optimizer.constraints import PortfolioConstraints
from kuber.regime.detector import RegimeDetector
from kuber.signals.momentum import TSMOMSignal
from kuber.signals.mean_reversion import RSISignal
from kuber.signals.volatility import RealizedVolSignal
from kuber.backtest.engine import BacktestEngine
from kuber.backtest.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kuber.cli")

OPTIMIZERS = {
    "markowitz": lambda: MarkowitzOptimizer(risk_aversion=1.0),
    "risk_parity": lambda: RiskParityOptimizer(),
    "black_litterman": lambda: BlackLittermanOptimizer(risk_aversion=2.5),
    "hrp": lambda: HRPOptimizer(),
    "regime_aware": lambda: RegimeAwareOptimizer(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KUBER Backtest Runner")
    parser.add_argument("--universe", type=str, default="balanced_etf", help="Universe name from config/sample_universes.yaml")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-01-01", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--optimizer", type=str, default="regime_aware", choices=list(OPTIMIZERS.keys()), help="Optimizer to use")
    parser.add_argument("--rebalance", type=str, default="monthly", choices=["daily", "weekly", "monthly", "quarterly"], help="Rebalance frequency")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback window in trading days")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--no-signals", action="store_true", help="Skip signal generation")
    parser.add_argument("--no-regime", action="store_true", help="Skip regime detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  KUBER Backtest Runner")
    print("=" * 60)
    print(f"  Universe:   {args.universe}")
    print(f"  Period:     {args.start} -> {args.end}")
    print(f"  Optimizer:  {args.optimizer}")
    print(f"  Rebalance:  {args.rebalance}")
    print(f"  Lookback:   {args.lookback} days")
    print("=" * 60)

    # 1. Load universe
    universe = load_universe(args.universe)
    tickers = universe["tickers"]
    print(f"\n[1/5] Universe '{universe['name']}': {tickers}")

    # 2. Load price data
    print("\n[2/5] Loading price data...")
    price_loader = PriceLoader()
    prices = price_loader.load(tickers, start=args.start, end=args.end)
    print(f"  Loaded {prices.shape[0]} days × {prices.shape[1]} tickers")

    if prices.empty:
        print("ERROR: No price data loaded. Exiting.")
        sys.exit(1)

    # 3. Load macro data (best-effort)
    print("\n[3/5] Loading macro data...")
    macro = None
    try:
        macro_loader = MacroLoader()
        macro = macro_loader.load(start=args.start, end=args.end)
        if macro.empty:
            print("  Macro data unavailable (no FRED API key?). Proceeding without.")
            macro = None
        else:
            print(f"  Loaded {macro.shape[0]} days × {macro.shape[1]} indicators")
    except Exception as e:
        print(f"  Macro load failed: {e}. Proceeding without.")

    # 4. Set up signals
    signals = []
    if not args.no_signals:
        try:
            signals = [TSMOMSignal(), RSISignal(), RealizedVolSignal()]
            print(f"\n[4/5] Signals: {[s.name for s in signals]}")
        except Exception as e:
            print(f"  Signal setup failed: {e}. Proceeding without signals.")
            signals = []
    else:
        print("\n[4/5] Signals: SKIPPED")

    # 5. Set up regime detector
    regime_detector = None
    if not args.no_regime:
        try:
            # Use VIX classifier if macro is available, otherwise skip
            if macro is not None and ("VIXCLS" in macro.columns or "VIX" in macro.columns):
                regime_detector = RegimeDetector(method="vix")
                print("  Regime detector: VIX classifier")
            else:
                regime_detector = RegimeDetector(method="hmm", n_regimes=3)
                print("  Regime detector: HMM (3 states)")
        except Exception as e:
            print(f"  Regime detector setup failed: {e}. Proceeding without.")
    else:
        print("  Regime detection: SKIPPED")

    # 6. Build optimizer
    optimizer = OPTIMIZERS[args.optimizer]()
    print(f"\n[5/5] Running backtest with {type(optimizer).__name__}...")

    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.30,
        max_turnover=0.50,
        transaction_cost_bps=10,
    )

    engine = BacktestEngine(
        optimizer=optimizer,
        signals=signals,
        regime_detector=regime_detector,
        constraints=constraints,
        rebalance_frequency=args.rebalance,
        lookback_window=args.lookback,
    )

    result = engine.run(prices, macro=macro, start_date=args.start, end_date=args.end)

    # Print results
    print("\n" + result.summary())

    # Print detailed comparison table
    print("\n" + "=" * 80)
    print("  METRICS COMPARISON TABLE")
    print("=" * 80)

    # Collect all strategies
    all_strategies = {"KUBER": result.metrics}
    for bm_name, bm_ret in result.benchmark_returns.items():
        all_strategies[bm_name] = compute_all_metrics(bm_ret)

    # Print table header
    metric_names = ["annualized_return", "annualized_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "win_rate"]
    header = f"  {'Metric':<25s}"
    for strat in all_strategies:
        header += f" {strat:>15s}"
    print(header)
    print("  " + "-" * (25 + 16 * len(all_strategies)))

    for metric in metric_names:
        row = f"  {metric:<25s}"
        for strat_name, strat_metrics in all_strategies.items():
            val = strat_metrics.get(metric, float("nan"))
            if metric in ("annualized_return", "annualized_volatility", "max_drawdown"):
                row += f" {val:>14.2%}"
            else:
                row += f" {val:>15.4f}"
        print(row)

    print("=" * 80)

    # Save results
    output_path = args.output or f"backtest_{args.universe}_{args.optimizer}_{args.start}_{args.end}.json"
    result_dict = result.to_dict()
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
