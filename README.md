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

> *Run backtests to populate this table with real numbers.*

| Strategy                  | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD  | Turnover |
|---------------------------|-------------|----------|--------|---------|---------|----------|
| Equal Weight              |    —        |   —      |  —     |  —      | —       |   —      |
| 60/40                     |    —        |   —      |  —     |  —      | —       |   —      |
| KUBER: Markowitz          |    —        |   —      |  —     |  —      | —       |   —      |
| KUBER: Risk Parity        |    —        |   —      |  —     |  —      | —       |   —      |
| KUBER: Black-Litterman    |    —        |   —      |  —     |  —      | —       |   —      |
| KUBER: HRP                |    —        |   —      |  —     |  —      | —       |   —      |
| KUBER: Regime-Aware (full)|    —        |   —      |  —     |  —      | —       |   —      |

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
│   └── pages/
│       ├── page_01_universe.py  # Universe & Data overview
│       ├── page_02_signals.py   # Signals & Regime visualization
│       ├── page_03_optimize.py  # Portfolio optimization
│       ├── page_04_backtest.py  # Backtest results (equity curves, drawdowns)
│       └── page_05_memo.py      # AI-generated investment memo
│
├── scripts/
│   ├── run_backtest.py          # CLI: full backtest pipeline
│   └── generate_memo.py         # CLI: generate investment memo
│
├── tests/                       # pytest test suite
├── config/                      # YAML configs + universe definitions
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
