#!/usr/bin/env python3
"""CLI script to run a backtest and generate an investment memo.

Usage
-----
    python scripts/generate_memo.py --universe balanced_etf --provider template
    python scripts/generate_memo.py --provider claude  # requires ANTHROPIC_API_KEY
"""

import argparse
import logging
import sys
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
from kuber.signals.composite import CompositeSignal
from kuber.backtest.engine import BacktestEngine
from kuber.explain.attribution import WeightAttributor
from kuber.explain.scenario import ScenarioAnalyzer
from kuber.explain.memo_generator import MemoGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kuber.memo")

OPTIMIZERS = {
    "markowitz": lambda: MarkowitzOptimizer(risk_aversion=1.0),
    "risk_parity": lambda: RiskParityOptimizer(),
    "black_litterman": lambda: BlackLittermanOptimizer(risk_aversion=2.5),
    "hrp": lambda: HRPOptimizer(),
    "regime_aware": lambda: RegimeAwareOptimizer(),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KUBER Memo Generator")
    p.add_argument("--universe", default="balanced_etf", help="Universe name")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--optimizer", default="regime_aware", choices=list(OPTIMIZERS.keys()))
    p.add_argument("--provider", default="template", choices=["template", "claude"],
                    help="Memo provider (template=no API key needed)")
    p.add_argument("--output", default=None, help="Output file path (default: stdout)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  KUBER Investment Memo Generator")
    print("=" * 60)

    # 1. Load data
    universe = load_universe(args.universe)
    tickers = universe["tickers"]
    print(f"Universe: {universe['name']} — {tickers}")

    price_loader = PriceLoader()
    prices = price_loader.load(tickers, start=args.start, end=args.end)
    if prices.empty:
        print("ERROR: No price data. Exiting.")
        sys.exit(1)
    print(f"Prices: {prices.shape[0]} days × {prices.shape[1]} tickers")

    macro = None
    try:
        macro_loader = MacroLoader()
        macro = macro_loader.load(start=args.start, end=args.end)
        if macro.empty:
            macro = None
    except Exception:
        pass

    # 2. Run backtest
    signals = [TSMOMSignal(), RSISignal(), RealizedVolSignal()]
    regime_detector = RegimeDetector(method="vix") if macro is not None else None

    optimizer = OPTIMIZERS[args.optimizer]()
    constraints = PortfolioConstraints(min_weight=0.0, max_weight=0.30)

    engine = BacktestEngine(
        optimizer=optimizer,
        signals=signals,
        regime_detector=regime_detector,
        constraints=constraints,
        rebalance_frequency="monthly",
    )

    print("\nRunning backtest...")
    result = engine.run(prices, macro=macro, start_date=args.start, end_date=args.end)
    print("Backtest complete.")

    # 3. Get latest weights
    if result.portfolio_weights is not None and not result.portfolio_weights.empty:
        weights = result.portfolio_weights.iloc[-1]
    else:
        n = len(tickers)
        import pandas as pd
        weights = pd.Series(1.0 / n, index=tickers)

    # 4. Signal attribution
    composite = CompositeSignal(signals)
    composite.generate(prices, macro=macro)
    signal_attrib = composite.attribution()

    # Convert signal attribution (dict of DataFrames) to per-asset Series (latest row)
    signal_values = {}
    for sname, df in signal_attrib.items():
        if not df.empty:
            signal_values[sname] = df.iloc[-1]

    attributor = WeightAttributor()
    attribution_df = attributor.attribute(weights, signal_values, composite.weights)

    # 5. Scenario analysis
    scenario_analyzer = ScenarioAnalyzer()
    scenario_results = scenario_analyzer.analyze(weights, prices)

    # 6. Regime info
    current_regime = "Unknown"
    regime_method = "N/A"
    if result.regime_history is not None and len(result.regime_history) > 0:
        current_regime = result.regime_history.iloc[-1]
        regime_method = "VIX Classifier" if regime_detector and regime_detector.method == "vix" else "HMM"

    # 7. Generate memo
    print(f"\nGenerating memo (provider={args.provider})...")
    generator = MemoGenerator(provider=args.provider)
    memo = generator.generate(
        weights=weights,
        attribution=attribution_df,
        regime=current_regime,
        metrics=result.metrics,
        scenario_results=scenario_results,
        regime_method=regime_method,
    )

    # Output
    if args.output:
        Path(args.output).write_text(memo, encoding="utf-8")
        print(f"\nMemo saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print(memo)
        print("=" * 60)

    # Risk alert
    alert = generator.generate_risk_alert(weights, current_regime, scenario_results)
    if alert:
        print(f"\n⚠ {alert}")


if __name__ == "__main__":
    main()
