"""Investment memo generation — Claude API and template fallback."""

from __future__ import annotations

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from kuber.explain.templates import (
    MEMO_PROMPT,
    EXECUTIVE_SUMMARY_PROMPT,
    RISK_ALERT_PROMPT,
    TEMPLATE_MEMO,
)

logger = logging.getLogger(__name__)

# Regime integer -> human label
_REGIME_LABELS = {0: "Bear / Risk-Off", 1: "Sideways / Transition", 2: "Bull / Risk-On"}


def _regime_label(regime) -> str:
    if isinstance(regime, (int, np.integer)):
        return _REGIME_LABELS.get(int(regime), f"Regime {regime}")
    return str(regime)


def _weights_table(weights: pd.Series) -> str:
    """Format weights as a markdown table."""
    lines = ["| Asset | Weight |", "|-------|--------|"]
    for ticker, w in weights.sort_values(ascending=False).items():
        lines.append(f"| {ticker} | {w:.1%} |")
    return "\n".join(lines)


def _metrics_summary(metrics: dict) -> str:
    """Format metrics dict as a readable block."""
    key_map = {
        "annualized_return": ("Ann. Return", True),
        "annualized_volatility": ("Ann. Vol", True),
        "sharpe_ratio": ("Sharpe", False),
        "sortino_ratio": ("Sortino", False),
        "max_drawdown": ("Max DD", True),
        "calmar_ratio": ("Calmar", False),
        "win_rate": ("Win Rate", True),
    }
    lines = []
    for key, (label, pct) in key_map.items():
        v = metrics.get(key)
        if v is None:
            continue
        if pct:
            lines.append(f"- {label}: {v:.1%}")
        else:
            lines.append(f"- {label}: {v:.2f}")
    return "\n".join(lines)


class MemoGenerator:
    """Generate investment memos from portfolio data.

    Parameters
    ----------
    provider : str
        ``"claude"`` for Anthropic API, ``"template"`` for string-based
        fallback.  Default is ``"template"`` (works without an API key).
    """

    def __init__(self, provider: str = "template") -> None:
        self.provider = provider.lower()
        if self.provider == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set — falling back to template provider.")
                self.provider = "template"

    def generate(
        self,
        weights: pd.Series,
        attribution: pd.DataFrame | None = None,
        regime: str | int = "Unknown",
        metrics: dict | None = None,
        scenario_results: dict | None = None,
        signal_summary: dict | None = None,
        regime_method: str = "HMM",
        regime_params: str = "",
    ) -> str:
        """Generate a full investment memo.

        Parameters
        ----------
        weights : pd.Series
            Final portfolio weights.
        attribution : pd.DataFrame | None
            Output from WeightAttributor.attribute().
        regime : str | int
            Current regime label or integer.
        metrics : dict | None
            Backtest performance metrics.
        scenario_results : dict | None
            Output from ScenarioAnalyzer.analyze().
        signal_summary : dict | None
            Optional per-signal summary info.
        regime_method : str
            Regime detection method name.
        regime_params : str
            Human-readable regime parameters.

        Returns
        -------
        str
            Markdown-formatted memo.
        """
        metrics = metrics or {}
        scenario_results = scenario_results or {}
        regime_label = _regime_label(regime)

        if self.provider == "claude":
            return self._generate_claude(
                weights, attribution, regime_label, metrics,
                scenario_results, signal_summary, regime_method, regime_params,
            )
        return self._generate_template(
            weights, attribution, regime_label, metrics,
            scenario_results, signal_summary, regime_method, regime_params,
        )

    # ------------------------------------------------------------------ #
    # Claude API provider
    # ------------------------------------------------------------------ #
    def _generate_claude(
        self,
        weights, attribution, regime, metrics,
        scenario_results, signal_summary, regime_method, regime_params,
    ) -> str:
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic package not installed — falling back to template.")
            return self._generate_template(
                weights, attribution, regime, metrics,
                scenario_results, signal_summary, regime_method, regime_params,
            )

        prompt = MEMO_PROMPT.format(
            weights_table=_weights_table(weights),
            regime=regime,
            regime_method=regime_method,
            regime_params=regime_params or "N/A",
            attribution_table=self._attribution_text(attribution),
            metrics_summary=_metrics_summary(metrics),
            scenario_table=self._scenario_text(scenario_results),
        )

        try:
            client = anthropic.Anthropic()
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as exc:
            logger.error("Claude API call failed: %s. Falling back to template.", exc)
            return self._generate_template(
                weights, attribution, regime, metrics,
                scenario_results, signal_summary, regime_method, regime_params,
            )

    # ------------------------------------------------------------------ #
    # Template provider (default — no API key needed)
    # ------------------------------------------------------------------ #
    def _generate_template(
        self,
        weights, attribution, regime, metrics,
        scenario_results, signal_summary, regime_method, regime_params,
    ) -> str:
        # Overweight / underweight sections
        sorted_w = weights.sort_values(ascending=False)
        overweight = sorted_w.head(3)
        underweight = sorted_w.tail(3)

        ow_lines = []
        for ticker, w in overweight.items():
            reason = self._signal_reason(ticker, attribution, signal_summary)
            ow_lines.append(f"- **{ticker}** ({w:.1%}): {reason}")

        uw_lines = []
        for ticker, w in underweight.items():
            reason = self._signal_reason(ticker, attribution, signal_summary)
            uw_lines.append(f"- **{ticker}** ({w:.1%}): {reason}")

        # Attribution section
        if attribution is not None and not attribution.empty:
            attr_lines = ["| Asset | " + " | ".join(
                c for c in attribution.columns if c != "final_weight"
            ) + " | Net Weight |", "|" + "---|" * (len(attribution.columns)) ]
            for ticker, row in attribution.iterrows():
                cols = [c for c in attribution.columns if c != "final_weight"]
                vals = " | ".join(f"{row[c]:+.1%}" for c in cols)
                attr_lines.append(f"| {ticker} | {vals} | {row['final_weight']:.1%} |")
            attribution_section = "\n".join(attr_lines)
        else:
            attribution_section = "*No signal attribution available.*"

        # Scenario section
        if scenario_results:
            from kuber.explain.scenario import ScenarioResult
            sc_lines = [
                "| Scenario | Portfolio | Max DD | Equal-Weight | 60/40 |",
                "|----------|----------|--------|-------------|-------|",
            ]
            for _k, r in scenario_results.items():
                sf = f"{r.sixty_forty_return:.1%}" if not np.isnan(r.sixty_forty_return) else "N/A"
                sc_lines.append(
                    f"| {r.name} | {r.portfolio_return:.1%} | "
                    f"{-r.max_drawdown:.1%} | {r.equal_weight_return:.1%} | {sf} |"
                )
            scenario_section = "\n".join(sc_lines)
        else:
            scenario_section = "*No scenario data available.*"

        # Confidence section
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown", 0)
        if sharpe > 1.0 and max_dd < 0.20:
            confidence = "**High confidence.** The portfolio shows a strong risk-adjusted return profile (Sharpe > 1) with contained drawdowns."
        elif sharpe > 0.5:
            confidence = "**Moderate confidence.** Risk-adjusted returns are acceptable but warrant monitoring."
        else:
            confidence = "**Low confidence.** Current metrics suggest the portfolio may benefit from rebalancing or strategy review."
        confidence += "\n\n**Review triggers:** Regime change, Sharpe falling below 0.5, drawdown exceeding 15%, or significant turnover in top holdings."

        # Regime detail
        regime_detail = ""
        if regime_params:
            regime_detail = f" ({regime_params})"

        return TEMPLATE_MEMO.format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            regime=regime,
            regime_method=regime_method,
            regime_detail=regime_detail,
            weights_table=_weights_table(weights),
            overweight_section="\n".join(ow_lines) if ow_lines else "*N/A*",
            underweight_section="\n".join(uw_lines) if uw_lines else "*N/A*",
            attribution_section=attribution_section,
            ann_return=f"{metrics.get('annualized_return', 0):.1%}",
            ann_vol=f"{metrics.get('annualized_volatility', 0):.1%}",
            sharpe=f"{metrics.get('sharpe_ratio', 0):.2f}",
            sortino=f"{metrics.get('sortino_ratio', 0):.2f}",
            max_dd=f"{metrics.get('max_drawdown', 0):.1%}",
            calmar=f"{metrics.get('calmar_ratio', 0):.2f}",
            win_rate=f"{metrics.get('win_rate', 0):.1%}",
            scenario_section=scenario_section,
            confidence_section=confidence,
        )

    # ------------------------------------------------------------------ #
    # Executive summary (short)
    # ------------------------------------------------------------------ #
    def generate_executive_summary(
        self,
        weights: pd.Series,
        regime: str | int,
        metrics: dict,
    ) -> str:
        """Short 3-4 sentence summary."""
        regime_label = _regime_label(regime)
        if self.provider == "claude":
            try:
                import anthropic
                prompt = EXECUTIVE_SUMMARY_PROMPT.format(
                    weights_table=_weights_table(weights),
                    regime=regime_label,
                    sharpe=metrics.get("sharpe_ratio", 0),
                    ann_return=metrics.get("annualized_return", 0),
                    max_dd=metrics.get("max_drawdown", 0),
                )
                client = anthropic.Anthropic()
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text
            except Exception as exc:
                logger.warning("Claude exec summary failed: %s", exc)

        # Template fallback
        top = weights.nlargest(3)
        top_str = ", ".join(f"{t} ({w:.0%})" for t, w in top.items())
        return (
            f"The portfolio is positioned for a **{regime_label}** regime. "
            f"Top holdings: {top_str}. "
            f"Backtest Sharpe of {metrics.get('sharpe_ratio', 0):.2f} with "
            f"{metrics.get('max_drawdown', 0):.1%} max drawdown. "
            f"Annualised return of {metrics.get('annualized_return', 0):.1%}."
        )

    # ------------------------------------------------------------------ #
    # Risk alert
    # ------------------------------------------------------------------ #
    def generate_risk_alert(
        self,
        weights: pd.Series,
        regime: str | int,
        scenario_results: dict | None = None,
    ) -> str | None:
        """Return a risk alert if the portfolio has concerning exposures, else None."""
        regime_label = _regime_label(regime)
        top3_weight = float(weights.nlargest(3).sum())

        worst_loss = 0.0
        worst_name = "N/A"
        if scenario_results:
            for _k, r in scenario_results.items():
                if r.portfolio_return < worst_loss:
                    worst_loss = r.portfolio_return
                    worst_name = r.name

        # Only alert if concentrated or large scenario loss
        if top3_weight < 0.60 and worst_loss > -0.15:
            return None

        if self.provider == "claude":
            try:
                import anthropic
                prompt = RISK_ALERT_PROMPT.format(
                    weights_table=_weights_table(weights),
                    top3_weight=top3_weight,
                    worst_scenario_name=worst_name,
                    worst_loss=worst_loss,
                    regime=regime_label,
                )
                client = anthropic.Anthropic()
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text
            except Exception:
                pass

        # Template fallback
        alert = f"**Risk Alert:** Portfolio concentration is {top3_weight:.0%} in the top 3 holdings."
        if worst_loss < -0.10:
            alert += f" Under the {worst_name} scenario, projected loss is {worst_loss:.1%}."
        alert += " Consider diversifying or adding hedging positions."
        return alert

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _signal_reason(
        ticker: str,
        attribution: pd.DataFrame | None,
        signal_summary: dict | None,
    ) -> str:
        if attribution is not None and ticker in attribution.index:
            row = attribution.loc[ticker]
            signal_cols = [c for c in attribution.columns if c != "final_weight"]
            parts = []
            for col in signal_cols:
                val = row[col]
                if abs(val) > 0.001:
                    sign = "+" if val >= 0 else ""
                    parts.append(f"{sign}{val:.1%} from {col}")
            if parts:
                return ", ".join(parts)
        return "Allocation driven by composite signal blend and risk constraints."

    @staticmethod
    def _attribution_text(attribution: pd.DataFrame | None) -> str:
        if attribution is None or attribution.empty:
            return "No attribution data available."
        lines = []
        signal_cols = [c for c in attribution.columns if c != "final_weight"]
        for ticker, row in attribution.iterrows():
            parts = []
            for col in signal_cols:
                v = row[col]
                if abs(v) > 0.001:
                    sign = "+" if v >= 0 else ""
                    parts.append(f"{sign}{v:.1%} from {col}")
            fw = row["final_weight"]
            lines.append(f"  {ticker}: {', '.join(parts)} → net {fw:.1%}")
        return "\n".join(lines)

    @staticmethod
    def _scenario_text(scenario_results: dict) -> str:
        if not scenario_results:
            return "No scenario data."
        lines = []
        for _k, r in scenario_results.items():
            sf = f", 60/40: {r.sixty_forty_return:.1%}" if not np.isnan(r.sixty_forty_return) else ""
            lines.append(
                f"  {r.name}: portfolio {r.portfolio_return:+.1%}, "
                f"max DD {-r.max_drawdown:.1%}, EW: {r.equal_weight_return:+.1%}{sf}"
            )
        return "\n".join(lines)
