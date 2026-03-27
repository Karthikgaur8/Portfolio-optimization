"""Prompt templates for LLM-generated investment memos."""

# --------------------------------------------------------------------------- #
# Main memo prompt (sent to Claude API)
# --------------------------------------------------------------------------- #
MEMO_PROMPT = """\
You are a senior portfolio analyst writing an investment memo for a research committee.

CURRENT ALLOCATION:
{weights_table}

MARKET REGIME: {regime} (detected via {regime_method})
Regime characteristics: {regime_params}

SIGNAL ATTRIBUTION:
{attribution_table}

BACKTEST METRICS (last 3 years):
{metrics_summary}

STRESS TEST RESULTS:
{scenario_table}

Write a concise (400-600 word) investment memo that:
1. Opens with the current market regime assessment
2. Explains the top 3 overweight and top 3 underweight positions with specific signal rationale
3. Flags the key risk to this portfolio and the hedging mechanism in place
4. Presents the stress test results and what they imply
5. Closes with a confidence assessment and recommended review triggers

Tone: Professional, quantitative, direct. Use specific numbers. Avoid hedging language.
"""

# --------------------------------------------------------------------------- #
# Executive summary (shorter variant)
# --------------------------------------------------------------------------- #
EXECUTIVE_SUMMARY_PROMPT = """\
You are a senior portfolio analyst. Based on the data below, write a 3-4 sentence \
executive summary of the current portfolio positioning.

ALLOCATION: {weights_table}
REGIME: {regime}
KEY METRICS: Sharpe {sharpe:.2f}, Ann. Return {ann_return:.1%}, Max DD {max_dd:.1%}

Be specific with numbers. No filler.
"""

# --------------------------------------------------------------------------- #
# Risk alert template
# --------------------------------------------------------------------------- #
RISK_ALERT_PROMPT = """\
You are a risk analyst. The following portfolio may have concerning exposures.

ALLOCATION: {weights_table}
CONCENTRATION: Top 3 holdings = {top3_weight:.0%} of portfolio
WORST SCENARIO: In {worst_scenario_name}, projected loss = {worst_loss:.1%}
CURRENT REGIME: {regime}

Write a concise risk alert (2-3 sentences) highlighting the key concern and \
one actionable recommendation. Be direct.
"""

# --------------------------------------------------------------------------- #
# Template-based fallback memo (no API key needed)
# --------------------------------------------------------------------------- #
TEMPLATE_MEMO = """\
# KUBER Investment Memo

**Generated:** {date}
**Regime:** {regime} (detected via {regime_method})

---

## 1. Market Regime Assessment

The system currently detects a **{regime}** regime{regime_detail}. \
This classification is based on {regime_method} analysis of recent market data.

## 2. Portfolio Positioning

### Current Allocation

{weights_table}

### Top Overweight Positions

{overweight_section}

### Top Underweight / Zero-Weight Positions

{underweight_section}

## 3. Signal Attribution

{attribution_section}

## 4. Performance Metrics

| Metric | Value |
|--------|-------|
| Annualized Return | {ann_return} |
| Annualized Volatility | {ann_vol} |
| Sharpe Ratio | {sharpe} |
| Sortino Ratio | {sortino} |
| Max Drawdown | {max_dd} |
| Calmar Ratio | {calmar} |
| Win Rate | {win_rate} |

## 5. Stress Test Results

{scenario_section}

## 6. Confidence & Review Triggers

{confidence_section}

---

*This memo was generated automatically by KUBER's template engine. \
For LLM-enhanced memos, set the ANTHROPIC_API_KEY environment variable.*
"""
