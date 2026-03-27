# KUBER Demo Walkthrough

A step-by-step guide for presenting KUBER in ~7 minutes.

---

## 1. The Setup (30 seconds)

> "Let me show you a project that started as a single-file Markowitz optimizer in 2022 and evolved into a full research platform. KUBER — named after the Hindu god of wealth — is an explainable AI portfolio engine with 5 optimizers, regime detection, walk-forward backtesting, and an LLM-powered explanation layer."

Open the repo. Show the README briefly. Point to the results table.

---

## 2. The Architecture (1 minute)

Show the directory structure:

```
kuber/
  data/          → Price, macro, sentiment loaders with Parquet caching
  signals/       → 14 signal generators across 5 families
  regime/        → HMM + VIX market state detection
  optimizer/     → Markowitz, Risk Parity, Black-Litterman, HRP, Regime-Aware
  backtest/      → Walk-forward engine (no look-ahead bias)
  explain/       → Weight attribution + LLM memo generation

dashboard/       → 5-page Streamlit app
scripts/         → CLI tools for backtesting and memo generation
```

Key points to highlight:
- **Modularity**: Every component follows an abstract base class pattern (Signal ABC, Optimizer ABC). Plugging in a new signal or optimizer is one file.
- **No look-ahead bias**: The backtest engine strictly uses only data available at each rebalance date.
- **Graceful degradation**: No FRED API key? Falls back to HMM. No Anthropic key? Falls back to template memos. No internet? Uses Parquet cache.

---

## 3. The Dashboard Demo (3 minutes)

### Start the dashboard:
```bash
streamlit run dashboard/app.py
```

### Step 3a — Universe & Data
1. Select **balanced_etf** in the sidebar (10 diversified ETFs)
2. Set date range: 2020-01-01 to 2025-01-01
3. Click **Load Data**
4. Show the normalized price chart — point out how different asset classes (bonds, gold, equities) move differently
5. Show the correlation matrix — "Low correlation between BND/GLD and SPY/QQQ is exactly what diversification exploits"

### Step 3b — Signals & Regime
1. Navigate to **Signals & Regime**
2. Show individual signal time series — "These are momentum, RSI, and realized volatility signals, each normalized to [-1, 1]"
3. Show composite signal heatmap — "This is the combined signal strength for each asset over time. Green = bullish, red = bearish"
4. Show regime detection — select HMM — "The model identifies bull, sideways, and bear regimes from returns data"

### Step 3c — Portfolio Optimization
1. Navigate to **Portfolio Optimization**
2. Select **regime_aware** optimizer
3. Click **Run Optimizer**
4. Show the weights bar chart — "These are the optimal allocations given current signals and market regime"
5. Show the weight attribution table — "This is the key explainability feature. For each asset, you can see exactly which signal contributed to its weight"
6. Point out the expected return, volatility, and Sharpe ratio metrics

### Step 3d — Backtest Results
1. Navigate to **Backtest Results**
2. Select **regime_aware**, monthly rebalance
3. Click **Run Backtest**
4. Show equity curves — "KUBER vs equal weight, 60/40, and SPY buy-and-hold"
5. Show drawdown chart — "Notice how the risk-managed strategies have much shallower drawdowns"
6. Show the detailed metrics comparison table
7. Show weights over time — "You can see how the optimizer shifts allocations at each rebalance"

### Step 3e — Investment Memo
1. Navigate to **Investment Memo**
2. Select **template** provider
3. Click **Generate Memo**
4. Show the generated memo — "This is an AI-generated investment memo that explains the current allocation in plain English"
5. Show scenario analysis — "It also stress-tests the portfolio against historical crises like COVID and the 2022 rate shock"

---

## 4. The Code Tour (2 minutes)

### Signal ABC (`kuber/signals/base.py`)
> "Every signal implements this interface. `generate()` takes prices, returns a DataFrame of scores in [-1, 1]. This makes signals composable and testable."

### BacktestEngine walk-forward loop (`kuber/backtest/engine.py`)
> "This is the heart of the system. At each rebalance date, it: (1) slices only historical data, (2) generates signals, (3) detects regime, (4) runs the optimizer, (5) records weights, (6) deducts transaction costs. No future data ever leaks into the past."

### RegimeAwareOptimizer (`kuber/optimizer/regime_aware.py`)
> "This is the most interesting optimizer. It detects whether we're in a bull, bear, or sideways market, then delegates to the appropriate sub-optimizer: Black-Litterman for bull markets (exploit signal views), Risk Parity for bear (risk management), HRP for sideways (diversification without return estimates)."

### WeightAttributor (`kuber/explain/attribution.py`)
> "This decomposes the final portfolio weight into per-signal contributions. If a PM asks 'why are we 15% in gold?', this shows that 8% came from the momentum signal and 7% from the macro signal."

---

## 5. The Results (1 minute)

Show the results table from README.md:

| Strategy                  | Ann. Return | Ann. Vol | Sharpe | Max DD  |
|---------------------------|-------------|----------|--------|---------|
| SPY Buy-and-Hold          | 29.10%      | 17.94%   | 1.62   | -20.70% |
| 60/40                     | 17.28%      | 10.95%   | 1.58   | -12.39% |
| Equal Weight              | 9.54%       | 5.33%    | 1.79   | -4.46%  |
| KUBER: Black-Litterman    | 6.95%       | 3.95%    | 1.76   | -3.78%  |
| KUBER: Risk Parity        | 6.58%       | 3.88%    | 1.69   | -3.90%  |

> "SPY crushed everything on raw returns — but that's 2020-2025, one of the strongest equity bull runs in history. The real question is: would you prefer 29% return with a 21% drawdown, or 7% return with a 3.8% drawdown? For institutional allocators, the answer is almost always the latter. Risk Parity and Black-Litterman deliver exactly what they promise: superior risk-adjusted returns with minimal drawdowns."

---

## 6. Anticipated Questions

### "Why not use a library like Zipline/Backtrader?"
> "Building the backtester IS the point — it demonstrates I understand walk-forward methodology, transaction cost modeling, and the pitfalls of look-ahead bias. Anyone can call `bt.run()`. The value is in knowing what happens inside."

### "Does the regime-aware strategy actually outperform?"
> "In this test period, it tracked HRP because without macro data (no FRED API key), the HMM regime detector had covariance estimation issues and defaulted to the 'sideways' regime. With proper VIX data, regime switching is more active. The architecture is there — it's a data quality issue, not a design issue."

### "How would you improve this?"
> "Three directions: (1) RL-based allocation using PPO/SAC to learn dynamic policies, (2) transformer-based return forecasting to replace the linear signal model, (3) alternative data from SEC filings and earnings transcripts via NLP. The modular architecture means each of these is a drop-in replacement for one component — no rewrite needed."

### "What's the weakest part?"
> "The sentiment signal is synthetic — it generates correlated random scores rather than parsing real news. With a news API (Bloomberg, Refinitiv, or even free NewsAPI), the FinBERT sentiment loader would produce real signal. The architecture already supports it via the SentimentLoader provider pattern — I just need the data feed."

### "How does this compare to what quant teams actually use?"
> "The architecture mirrors how systematic macro funds work: signal generation → portfolio construction → risk management → execution. The scale is obviously different (10 ETFs vs 5000 futures), but the methodology is correct. A production system would add: (1) execution management, (2) real-time data feeds, (3) risk limits and compliance checks, (4) distributed computing for large universes."
