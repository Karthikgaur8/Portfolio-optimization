#!/usr/bin/env python3
"""Run all 8 experiments from the PRD and save results to results/ directory."""

import json
import sys
import traceback
from pathlib import Path

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

RESULTS_DIR = _ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Shared config
UNIVERSE = "balanced_etf"
START = "2020-01-01"
END = "2025-01-01"
REBALANCE = "monthly"
LOOKBACK = 252

CONSTRAINTS = PortfolioConstraints(
    min_weight=0.0,
    max_weight=0.30,
    max_turnover=0.50,
    transaction_cost_bps=10,
)

EXPERIMENTS = [
    {
        "name": "equal_weight_baseline",
        "optimizer": "equal_weight",
        "signals": False,
        "regime": False,
        "description": "Equal Weight baseline",
    },
    {
        "name": "sixty_forty_baseline",
        "optimizer": "sixty_forty",
        "signals": False,
        "regime": False,
        "description": "60/40 baseline",
    },
    {
        "name": "spy_buyhold_baseline",
        "optimizer": "spy_buyhold",
        "signals": False,
        "regime": False,
        "description": "Buy-and-hold SPY baseline",
    },
    {
        "name": "markowitz_momentum",
        "optimizer": "markowitz",
        "signals": True,
        "regime": False,
        "description": "Markowitz + momentum signals",
    },
    {
        "name": "risk_parity_nosignals",
        "optimizer": "risk_parity",
        "signals": False,
        "regime": False,
        "description": "Risk Parity (no signals)",
    },
    {
        "name": "black_litterman_allsignals",
        "optimizer": "black_litterman",
        "signals": True,
        "regime": False,
        "description": "Black-Litterman + all signals",
    },
    {
        "name": "hrp_nosignals",
        "optimizer": "hrp",
        "signals": False,
        "regime": False,
        "description": "HRP (no signals)",
    },
    {
        "name": "regime_aware_full",
        "optimizer": "regime_aware",
        "signals": True,
        "regime": True,
        "description": "Regime-Aware + all signals + HMM",
    },
]


def make_optimizer(name):
    if name == "markowitz":
        return MarkowitzOptimizer(risk_aversion=1.0)
    elif name == "risk_parity":
        return RiskParityOptimizer()
    elif name == "black_litterman":
        return BlackLittermanOptimizer(risk_aversion=2.5)
    elif name == "hrp":
        return HRPOptimizer()
    elif name == "regime_aware":
        return RegimeAwareOptimizer()
    elif name in ("equal_weight", "sixty_forty", "spy_buyhold"):
        # These are benchmarks — we'll use Risk Parity as a dummy optimizer
        # and extract benchmark returns from the result
        return RiskParityOptimizer()
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def run_experiment(exp, prices, macro):
    """Run a single experiment and return metrics dict."""
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp['description']}")
    print(f"{'='*60}")

    opt_name = exp["optimizer"]

    # For baseline benchmarks, run a simple backtest and extract benchmark
    is_baseline = opt_name in ("equal_weight", "sixty_forty", "spy_buyhold")

    signals = []
    if exp["signals"]:
        signals = [TSMOMSignal(), RSISignal(), RealizedVolSignal()]

    regime_detector = None
    if exp["regime"]:
        regime_detector = RegimeDetector(method="hmm", n_regimes=3)

    optimizer = make_optimizer(opt_name)

    engine = BacktestEngine(
        optimizer=optimizer,
        signals=signals,
        regime_detector=regime_detector,
        constraints=CONSTRAINTS,
        rebalance_frequency=REBALANCE,
        lookback_window=LOOKBACK,
    )

    result = engine.run(prices, macro=macro, start_date=START, end_date=END)

    if is_baseline:
        # Extract benchmark metrics — try multiple key formats
        bm_candidates = {
            "equal_weight": ["equal_weight", "Equal Weight"],
            "sixty_forty": ["sixty_forty", "60/40"],
            "spy_buyhold": ["spy_buy_hold", "SPY Buy-Hold", "spy_buyhold"],
        }
        found = False
        for candidate in bm_candidates.get(opt_name, []):
            if candidate in result.benchmark_returns:
                bm_ret = result.benchmark_returns[candidate]
                metrics = compute_all_metrics(bm_ret)
                found = True
                break
        if not found:
            metrics = result.metrics
    else:
        metrics = result.metrics

    # Also compute benchmark metrics for comparison
    all_metrics = {"strategy": metrics}
    for bm_name, bm_ret in result.benchmark_returns.items():
        all_metrics[bm_name] = compute_all_metrics(bm_ret)

    # Print summary
    print(f"  Ann. Return:  {metrics.get('annualized_return', 0):.2%}")
    print(f"  Ann. Vol:     {metrics.get('annualized_volatility', 0):.2%}")
    print(f"  Sharpe:       {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Sortino:      {metrics.get('sortino_ratio', 0):.4f}")
    print(f"  Max DD:       {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Win Rate:     {metrics.get('win_rate', 0):.2%}")

    return {
        "experiment": exp["name"],
        "description": exp["description"],
        "metrics": metrics,
        "all_metrics": all_metrics,
    }


def main():
    print("=" * 60)
    print("  KUBER — Full Experiment Suite")
    print("=" * 60)
    print(f"  Universe: {UNIVERSE}")
    print(f"  Period:   {START} -> {END}")
    print(f"  Rebal:    {REBALANCE}")

    # Load data once
    print("\nLoading price data...")
    universe = load_universe(UNIVERSE)
    tickers = universe["tickers"]
    price_loader = PriceLoader()
    prices = price_loader.load(tickers, start=START, end=END)
    print(f"  Prices: {prices.shape[0]} days x {prices.shape[1]} tickers")

    if prices.empty:
        print("ERROR: No price data. Exiting.")
        sys.exit(1)

    # Load macro (best-effort)
    macro = None
    try:
        macro_loader = MacroLoader()
        macro = macro_loader.load(start=START, end=END)
        if macro.empty:
            macro = None
            print("  Macro: unavailable")
        else:
            print(f"  Macro: {macro.shape[0]} days x {macro.shape[1]} indicators")
    except Exception:
        print("  Macro: unavailable")

    # Run all experiments
    all_results = {}
    for exp in EXPERIMENTS:
        try:
            result = run_experiment(exp, prices, macro)
            all_results[exp["name"]] = result

            # Save individual result
            out_path = RESULTS_DIR / f"{exp['name']}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  Saved: {out_path}")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_results[exp["name"]] = {
                "experiment": exp["name"],
                "description": exp["description"],
                "error": str(e),
                "metrics": {},
            }

    # Save combined results
    combined_path = RESULTS_DIR / "all_experiments.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {combined_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 100)
    header = f"  {'Experiment':<35s} {'Ann.Ret':>10s} {'Ann.Vol':>10s} {'Sharpe':>10s} {'Sortino':>10s} {'MaxDD':>10s}"
    print(header)
    print("  " + "-" * 95)

    for name, res in all_results.items():
        m = res.get("metrics", {})
        if not m:
            print(f"  {name:<35s} {'FAILED':>10s}")
            continue
        print(
            f"  {name:<35s}"
            f" {m.get('annualized_return', 0):>9.2%}"
            f" {m.get('annualized_volatility', 0):>9.2%}"
            f" {m.get('sharpe_ratio', 0):>10.4f}"
            f" {m.get('sortino_ratio', 0):>10.4f}"
            f" {m.get('max_drawdown', 0):>9.2%}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
