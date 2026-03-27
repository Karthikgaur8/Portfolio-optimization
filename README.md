# KUBER — Knowledge-driven, Unified, Backtested, Explainable Research Engine

*Named after Kubera (कुबेर), the Hindu god of wealth and prosperity.*

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?logo=plotly)

> An explainable AI portfolio research copilot that generates constrained optimal allocations with transparent rationale, regime awareness, and walk-forward backtesting.

---

## Highlights

- **Multi-signal pipeline:** Momentum, mean-reversion, volatility, sentiment (LLM-powered), macro factors
- **Regime-aware optimization:** HMM-based market state detection with automatic strategy switching
- **5 optimizer strategies:** Markowitz, Risk Parity, Black-Litterman, HRP, Regime-Aware
- **Proper backtesting:** Walk-forward with transaction costs, turnover constraints, multiple benchmarks
- **Explainable AI:** Factor attribution + LLM-generated investment memos (works without API key via template fallback)
- **Interactive dashboard:** 5-page Streamlit app with equity curves, drawdowns, signals, and memo generation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        KUBER SYSTEM                             │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────────┐   │
│  │   DATA    │──▶│   SIGNAL     │──▶│    PORTFOLIO           │   │
│  │  ENGINE   │   │  GENERATION  │   │    OPTIMIZER           │   │
│  └──────────┘   └──────────────┘   └───────────────────────┘   │
│       │               │                       │                 │
│       │               │                       ▼                 │
│       │               │            ┌───────────────────────┐   │
│       │               │            │    BACKTESTING         │   │
│       │               │            │    ENGINE              │   │
│       │               │            └───────────────────────┘   │
│       │               │                       │                 │
│       ▼               ▼                       ▼                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              EXPLANATION ENGINE (LLM)                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              STREAMLIT DASHBOARD                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow:** Price/macro/sentiment data → Signal generation (momentum, mean-reversion, vol, macro, sentiment) → Regime detection (HMM / VIX) → Portfolio optimization (5 strategies) → Walk-forward backtesting → Explanation engine → Interactive dashboard.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables (optional)

```bash
cp .env.example .env
# Edit .env with your API keys:
#   FRED_API_KEY=...         (macro data from FRED)
#   ANTHROPIC_API_KEY=...    (LLM-powered memos — optional)
```

The system works without any API keys — macro data and LLM memos have graceful fallbacks.

### 3. Run the dashboard

```bash
streamlit run dashboard/app.py
```

### 4. Or run from the command line

```bash
# Full backtest
python scripts/run_backtest.py --universe balanced_etf --optimizer regime_aware

# Generate investment memo
python scripts/generate_memo.py --universe balanced_etf --provider template
```

---

## Results

Walk-forward backtest on **balanced_etf** universe (SPY, QQQ, BND, GLD, VTI, IWM, EFA, EEM, TLT, HYG), monthly rebalance, 2020-01-01 to 2025-01-01, 10 bps transaction costs.

| Strategy                  | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD  |
|---------------------------|-------------|----------|--------|---------|---------|
| Equal Weight              | 9.54%       | 5.33%    | 1.79   | 3.11    | -4.46%  |
| 60/40                     | 17.28%      | 10.95%   | 1.58   | 2.75    | -12.39% |
| SPY Buy-and-Hold          | 29.10%      | 17.94%   | 1.62   | 2.83    | -20.70% |
| KUBER: Markowitz + signals| 9.05%       | 9.44%    | 0.96   | 1.71    | -10.55% |
| KUBER: Risk Parity        | 6.58%       | 3.88%    | 1.69   | 2.83    | -3.90%  |
| KUBER: Black-Litterman    | 6.95%       | 3.95%    | 1.76   | 2.92    | -3.78%  |
| KUBER: HRP                | 4.03%       | 3.31%    | 1.22   | 1.99    | -3.57%  |
| KUBER: Regime-Aware (full)| 4.03%       | 3.31%    | 1.22   | 1.99    | -3.57%  |

### Key Findings

**The risk-managed strategies delivered what they promised: lower volatility and smaller drawdowns.**

1. **SPY buy-and-hold dominated on raw returns** (29.1% annualized) but at the cost of 20.7% max drawdown. This period (2020-2025) was an unusually strong equity bull market post-COVID, so any equity-heavy strategy benefited massively from beta exposure.

2. **Black-Litterman had the best risk-adjusted performance among KUBER strategies** (Sharpe 1.76, Sortino 2.92) with the smallest max drawdown (-3.78%). The signal-derived views added marginal value over pure Risk Parity.

3. **Risk Parity delivered excellent risk control** (3.88% volatility, -3.90% max DD) with a strong Sharpe ratio of 1.69 — demonstrating that equal-risk-contribution allocation works well for a diversified ETF universe.

4. **The Regime-Aware strategy tracked HRP** because without macro data (no FRED API key), the HMM regime detector had covariance estimation issues and frequently fell back to the "sideways" regime, which routes to HRP. With proper VIX data from FRED, regime switching would be more active.

5. **Markowitz with signals had the worst Sharpe** (0.96) among KUBER strategies. Mean-variance optimization is notoriously sensitive to expected return estimates, and the signal-derived returns added noise. This is a well-known result in portfolio theory — estimation error in expected returns often hurts more than it helps.

6. **The real value proposition of these strategies isn't beating SPY in a bull market** — it's surviving bear markets. Risk Parity's 3.90% max drawdown vs SPY's 20.70% means a risk-averse investor can stay invested through volatility without panic selling.

---

## Project Structure

```
├── kuber/                       # Main package
│   ├── data/                    # DATA ENGINE
│   │   ├── price_loader.py      # yfinance wrapper with caching
│   │   ├── macro_loader.py      # FRED API macro indicators
│   │   ├── sentiment_loader.py  # Sentiment scoring (FinBERT / synthetic)
│   │   ├── cache.py             # Parquet caching layer
│   │   └── universe.py          # Ticker universe management
│   │
│   ├── signals/                 # SIGNAL GENERATION
│   │   ├── base.py              # Abstract Signal interface
│   │   ├── momentum.py          # Time-series & cross-sectional momentum
│   │   ├── mean_reversion.py    # RSI, Bollinger, z-score reversion
│   │   ├── volatility.py        # Realized vol, GARCH
│   │   ├── sentiment.py         # LLM sentiment signal
│   │   ├── macro.py             # Yield curve, VIX, Fed stance
│   │   └── composite.py         # Signal aggregation + attribution
│   │
│   ├── regime/                  # REGIME DETECTION
│   │   ├── hmm.py               # Hidden Markov Model (3-state)
│   │   ├── vix_classifier.py    # Rule-based VIX classifier
│   │   └── detector.py          # Unified interface
│   │
│   ├── optimizer/               # PORTFOLIO OPTIMIZER
│   │   ├── base.py              # Abstract Optimizer + OptimizationResult
│   │   ├── markowitz.py         # Mean-variance (cvxpy)
│   │   ├── risk_parity.py       # Equal risk contribution
│   │   ├── black_litterman.py   # Bayesian with signal-derived views
│   │   ├── hierarchical.py      # Hierarchical Risk Parity
│   │   ├── regime_aware.py      # Regime-switching optimizer
│   │   └── constraints.py       # Constraint library
│   │
│   ├── backtest/                # BACKTESTING ENGINE
│   │   ├── engine.py            # Walk-forward loop (no look-ahead)
│   │   ├── metrics.py           # Sharpe, Sortino, Calmar, max DD, etc.
│   │   ├── benchmark.py         # EW, 60/40, SPY benchmarks
│   │   └── report.py            # BacktestResult container
│   │
│   ├── explain/                 # EXPLANATION ENGINE
│   │   ├── attribution.py       # Per-signal weight decomposition
│   │   ├── scenario.py          # Historical crisis stress tests
│   │   ├── templates.py         # LLM prompt templates
│   │   └── memo_generator.py    # Claude API + template fallback
│   │
│   └── utils/                   # Shared utilities
│       └── config.py            # YAML config loader
│
├── dashboard/                   # STREAMLIT DASHBOARD
│   ├── app.py                   # Main entry point
│   ├── components/
│   │   ├── charts.py            # Reusable Plotly chart functions
│   │   └── sidebar.py           # Shared sidebar controls
│   └── views/
│       ├── page_01_universe.py  # Universe & Data overview
│       ├── page_02_signals.py   # Signals & Regime visualization
│       ├── page_03_optimize.py  # Portfolio optimization
│       ├── page_04_backtest.py  # Backtest results (equity curves, drawdowns)
│       └── page_05_memo.py      # AI-generated investment memo
│
├── scripts/
│   ├── run_backtest.py          # CLI: full backtest pipeline
│   ├── run_all_experiments.py   # CLI: run all 8 experiments
│   └── generate_memo.py         # CLI: generate investment memo
│
├── tests/                       # pytest test suite
├── config/                      # YAML configs + universe definitions
├── results/                     # Backtest experiment results (JSON)
├── docs/                        # Sample outputs + demo guide
├── legacy/
│   └── original_optimizer.py    # The 2022 script — preserved as benchmark
│
├── requirements.txt
└── .env.example
```

---

## Evolution from 2022 Script

This project evolved from a single-file Python script written in 2022. The original script (`legacy/original_optimizer.py`) demonstrated basic Markowitz optimization but had significant limitations:

| Aspect | 2022 Script | KUBER |
|--------|------------|-------|
| Structure | 1 file, ~200 lines | Modular package, 40+ files |
| Signals | Linear regression on 1-day lags | 5 signal families (momentum, mean-reversion, vol, sentiment, macro) |
| Optimizer | Single SLSQP run | 5 strategies including regime-aware switching |
| Regime | None | HMM + VIX classifier |
| Backtesting | None | Walk-forward with transaction costs |
| Explainability | Bar chart of weights | Factor attribution + LLM investment memos |
| Config | Hardcoded everything | YAML config + environment variables |
| Dashboard | None | 5-page interactive Streamlit app |

The diff between the legacy script and KUBER tells the story of growth from a learning exercise to a system that mirrors how modern quant teams work.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11+ | Standard for quant |
| Price data | yfinance | Free equity/ETF data |
| Macro data | FRED API (fredapi) | Treasury yields, VIX, CPI |
| ML / Stats | scikit-learn, hmmlearn, arch | Regime detection, GARCH |
| Optimization | scipy, cvxpy | Convex portfolio optimization |
| Visualization | Plotly | Interactive charts |
| Dashboard | Streamlit | Python-native web app |
| LLM | Anthropic Claude (optional) | Investment memo generation |
| Config | PyYAML, python-dotenv | Configuration management |
| Testing | pytest | Test suite |

---

## License

MIT
