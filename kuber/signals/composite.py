"""Composite signal aggregator for the KUBER framework.

Combines multiple :class:`Signal` instances into a single weighted signal.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from kuber.signals.base import Signal

logger = logging.getLogger(__name__)


class CompositeSignal:
    """Weighted aggregation of multiple signals.

    Parameters
    ----------
    signals : list[Signal]
        Signal instances to combine.
    weights : dict[str, float] | None
        Mapping of signal name -> weight. If ``None``, equal-weight.
    """

    def __init__(
        self,
        signals: list[Signal],
        weights: dict[str, float] | None = None,
    ) -> None:
        self.signals = signals

        if weights is None:
            n = len(signals)
            self.weights = {s.name: 1.0 / n for s in signals}
        else:
            self.weights = weights

        self._individual_results: dict[str, pd.DataFrame] = {}

        logger.info(
            "CompositeSignal initialized with %d signals: %s",
            len(signals),
            ", ".join(f"{s.name}({self.weights.get(s.name, 0):.2f})" for s in signals),
        )

    @property
    def name(self) -> str:
        return "Composite"

    def generate(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate composite signal as a weighted average of sub-signals.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (DatetimeIndex x tickers).
        macro : pd.DataFrame | None
            Macro data forwarded to each sub-signal.
        sentiment : pd.DataFrame | None
            Sentiment data forwarded to each sub-signal.

        Returns
        -------
        pd.DataFrame
            Composite signal values in [-1, 1].
        """
        logger.info("Generating composite signal from %d sub-signals.", len(self.signals))

        self._individual_results.clear()
        composite = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        total_weight = 0.0
        for sig in self.signals:
            w = self.weights.get(sig.name, 0.0)
            if w == 0.0:
                logger.debug("Skipping signal '%s' (weight=0).", sig.name)
                continue

            try:
                result = sig.generate(
                    prices, macro=macro, sentiment=sentiment, **kwargs
                )
                # Ensure alignment
                result = result.reindex(index=prices.index, columns=prices.columns)
                result = result.clip(-1, 1)
                self._individual_results[sig.name] = result
                composite += w * result.fillna(0)
                total_weight += w
            except Exception as exc:
                logger.error("Signal '%s' failed: %s", sig.name, exc)

        if total_weight > 0:
            composite /= total_weight

        composite = composite.clip(-1, 1)

        logger.info("Composite signal generated: %s rows, %s tickers.", *composite.shape)
        return composite

    def attribution(self) -> dict[str, pd.DataFrame]:
        """Return per-signal contribution to the composite.

        Each entry is the signal's output multiplied by its weight,
        divided by total weight (the actual contribution to the final value).

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of signal name -> weighted contribution DataFrame.
        """
        if not self._individual_results:
            logger.warning("No results available; call generate() first.")
            return {}

        total_weight = sum(
            self.weights.get(name, 0.0) for name in self._individual_results
        )
        if total_weight == 0:
            total_weight = 1.0

        attrib: dict[str, pd.DataFrame] = {}
        for name, result in self._individual_results.items():
            w = self.weights.get(name, 0.0)
            attrib[name] = (w * result.fillna(0)) / total_weight

        return attrib

    def describe(self) -> str:
        """Human-readable description."""
        parts = [f"  {s.name}: weight={self.weights.get(s.name, 0):.3f}" for s in self.signals]
        return "CompositeSignal:\n" + "\n".join(parts)
