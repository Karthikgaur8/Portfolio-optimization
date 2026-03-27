"""Weight attribution — decompose portfolio weights into signal contributions."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WeightAttributor:
    """Decompose final portfolio weights into per-signal contributions.

    The attribution is approximate: each signal's contribution to an asset is
    estimated as the signal score for that asset times the blending weight for
    that signal, scaled so the contributions sum to the final portfolio weight.
    """

    def attribute(
        self,
        weights: pd.Series,
        signal_values: dict[str, pd.Series],
        composite_weights: dict[str, float],
    ) -> pd.DataFrame:
        """Compute per-signal weight attribution.

        Parameters
        ----------
        weights : pd.Series
            Final portfolio weights (index = ticker, values = weight).
        signal_values : dict[str, pd.Series]
            Per-signal scores for each asset (signal_name -> Series with same
            index as *weights*).
        composite_weights : dict[str, float]
            Blending weights used by the CompositeSignal (signal_name -> weight).

        Returns
        -------
        pd.DataFrame
            Rows = assets, columns = signal names + ``'final_weight'``.
            Each cell shows the portion of the asset's weight attributable to
            that signal.  Rows sum to ``final_weight``.
        """
        tickers = weights.index.tolist()
        signal_names = list(signal_values.keys())

        if not signal_names:
            logger.warning("No signal values provided; returning weights only.")
            return pd.DataFrame({"final_weight": weights})

        # Build a matrix of weighted signal scores (tickers × signals)
        raw = pd.DataFrame(index=tickers, columns=signal_names, dtype=float)
        for sname, scores in signal_values.items():
            w = composite_weights.get(sname, 0.0)
            for ticker in tickers:
                raw.loc[ticker, sname] = float(scores.get(ticker, 0.0)) * w

        # Normalise row-wise so contributions sum to the final weight
        row_sums = raw.sum(axis=1).replace(0, np.nan)
        for ticker in tickers:
            total = row_sums.get(ticker, np.nan)
            if pd.isna(total):
                # If all signal contributions are zero, distribute weight
                # equally across signals as a fallback.
                raw.loc[ticker] = weights[ticker] / len(signal_names)
            else:
                raw.loc[ticker] = raw.loc[ticker] / total * weights[ticker]

        raw["final_weight"] = weights
        raw = raw.fillna(0.0)

        logger.info(
            "Attribution computed for %d assets across %d signals.",
            len(tickers),
            len(signal_names),
        )
        return raw

    @staticmethod
    def format_attribution(attribution_df: pd.DataFrame) -> str:
        """Return a human-readable attribution table as a string."""
        lines = []
        signal_cols = [c for c in attribution_df.columns if c != "final_weight"]
        header = f"{'Asset':<8s}" + "".join(f"{c:>14s}" for c in signal_cols) + f"{'Net Weight':>12s}"
        lines.append(header)
        lines.append("-" * len(header))

        for ticker, row in attribution_df.iterrows():
            parts = f"{ticker:<8s}"
            for col in signal_cols:
                val = row[col]
                sign = "+" if val >= 0 else ""
                parts += f"{sign}{val:>12.1%}  "
            parts += f"{row['final_weight']:>10.1%}"
            lines.append(parts)

        return "\n".join(lines)
