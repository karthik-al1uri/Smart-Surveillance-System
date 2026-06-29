"""Configuration loader for the Smart Surveillance System.

Loads YAML config files and merges them with optional environment-specific overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(env: str | None = None) -> dict[str, Any]:
    """Load and return the merged configuration dictionary.

    Loads ``config/default.yaml`` and optionally merges an environment-specific
    override file (e.g. ``config/development.yaml``) on top of it.

    Args:
        env: Environment name (e.g. ``"development"``, ``"production"``).
            Defaults to the ``APP_ENV`` environment variable, or ``"development"``
            if unset.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If ``config/default.yaml`` does not exist.
    """
    env = env or os.environ.get("APP_ENV", "development")

    default_path = _CONFIG_DIR / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    with default_path.open("r") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    env_path = _CONFIG_DIR / f"{env}.yaml"
    if env_path.exists():
        with env_path.open("r") as f:
            env_config: dict[str, Any] = yaml.safe_load(f) or {}
        config = _deep_merge(config, env_config)

    return config


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT
