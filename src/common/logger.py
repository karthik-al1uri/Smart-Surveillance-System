"""Structured logging setup for the Smart Surveillance System.

Provides a factory function that returns a named logger with consistent
formatting across all components.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_initialized: bool = False


def _initialize_root_logger(level: int) -> None:
    """Configure the root logger once."""
    global _initialized
    if _initialized:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _initialized = True


def get_logger(component: str, level: Optional[str] = None) -> logging.Logger:
    """Return a named logger for the given component.

    Args:
        component: Dot-separated component name, e.g. ``"detection.yolo"``.
        level: Optional log level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``). Defaults to ``"INFO"``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    _initialize_root_logger(log_level)
    return logging.getLogger(f"sss.{component}")
