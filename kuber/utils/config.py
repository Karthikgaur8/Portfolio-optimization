"""
KUBER configuration manager.

Loads YAML config files with environment variable overlay via python-dotenv.
Supports dot-notation access and deep merging of override configs.
"""

import os
import logging
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Config:
    """KUBER configuration manager. Loads YAML config with .env overlay."""

    _instance: "Config | None" = None
    _config: dict[str, Any] = {}

    def __init__(self, config_path: str | None = None) -> None:
        """Initialise config from default.yaml, optional override, and .env.

        Args:
            config_path: Path to an override YAML file. Values in this file
                are deep-merged on top of the defaults.
        """
        # Load .env into os.environ (no-op if file missing)
        project_root = Path(__file__).resolve().parent.parent.parent
        env_path = project_root / ".env"
        load_dotenv(dotenv_path=env_path)

        # Base config
        base_path = project_root / "config" / "default.yaml"
        self._config = {}
        if base_path.exists():
            with open(base_path, encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            logger.info("Base config loaded from %s", base_path)
        else:
            logger.warning("Default config not found at %s", base_path)

        # Optional override
        if config_path and Path(config_path).exists():
            with open(config_path, encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}
            self._deep_merge(self._config, override)
            logger.info("Override config merged from %s", config_path)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot notation.

        Example::

            config.get("data.cache_dir")       # -> ".cache"
            config.get("data.price.source")     # -> "yfinance"

        Args:
            key: Dot-separated path into the config dict.
            default: Returned when the key is missing.

        Returns:
            The config value, or *default* if not found.
        """
        keys = key.split(".")
        val: Any = self._config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    def get_env(self, key: str, default: str | None = None) -> str | None:
        """Get an environment variable.

        Args:
            key: Environment variable name (e.g. ``FRED_API_KEY``).
            default: Returned when the variable is not set.
        """
        return os.environ.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        """Return the full config as a plain dict (shallow copy)."""
        return dict(self._config)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge *override* into *base* (mutates base)."""
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                Config._deep_merge(base[k], v)
            else:
                base[k] = v
        return base


def get_config(config_path: str | None = None) -> Config:
    """Get or create a singleton :class:`Config` instance.

    Passing *config_path* forces a reload.
    """
    if Config._instance is None or config_path is not None:
        Config._instance = Config(config_path)
    return Config._instance
