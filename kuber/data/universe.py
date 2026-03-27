"""
Universe loader for KUBER.

Reads pre-defined ticker universes from ``config/sample_universes.yaml``
and provides helpers to list, load, and validate them.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_UNIVERSE_FILE = _PROJECT_ROOT / "config" / "sample_universes.yaml"


def _load_all_universes() -> dict[str, Any]:
    """Parse the universes YAML file and return the raw dict."""
    if not _UNIVERSE_FILE.exists():
        logger.error("Universe file not found: %s", _UNIVERSE_FILE)
        return {}
    with open(_UNIVERSE_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("universes", {})


def list_universes() -> list[str]:
    """Return the names (keys) of all available universes.

    Returns:
        Sorted list of universe identifiers.
    """
    universes = _load_all_universes()
    names = sorted(universes.keys())
    logger.debug("Available universes: %s", names)
    return names


def load_universe(name: str) -> dict[str, Any]:
    """Load a single universe by key.

    Args:
        name: Key in ``sample_universes.yaml`` (e.g. ``"balanced_etf"``).

    Returns:
        Dict with keys ``name``, ``tickers``, and ``description``.

    Raises:
        KeyError: If the requested universe does not exist.
    """
    universes = _load_all_universes()
    if name not in universes:
        available = list(universes.keys())
        raise KeyError(
            f"Universe '{name}' not found. Available: {available}"
        )
    entry = universes[name]
    result = {
        "name": entry.get("name", name),
        "tickers": list(entry.get("tickers", [])),
        "description": entry.get("description", ""),
    }
    logger.info(
        "Loaded universe '%s' with %d tickers", name, len(result["tickers"])
    )
    return result


def validate_tickers(tickers: list[str]) -> list[str]:
    """Check which tickers are recognised by yfinance.

    Makes a lightweight info call for each ticker. Tickers that raise
    an error or return no price data are dropped.

    Args:
        tickers: List of ticker symbols to validate.

    Returns:
        Subset of *tickers* that appear valid.
    """
    try:
        import yfinance as yf  # noqa: WPS433
    except ImportError:
        logger.warning(
            "yfinance not installed; returning tickers without validation"
        )
        return list(tickers)

    valid: list[str] = []
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            # fast_info raises or returns empty when ticker is bogus
            if hasattr(info, "last_price") and info.last_price is not None:
                valid.append(t)
            else:
                logger.warning("Ticker '%s' has no price data — skipped", t)
        except Exception:
            logger.warning("Ticker '%s' failed validation — skipped", t)
    logger.info("Validated %d / %d tickers", len(valid), len(tickers))
    return valid
