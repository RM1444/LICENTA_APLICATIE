"""Project-wide configuration loader.

Reads `config.yaml` from the project root and exposes it as a dict-like object.
Modules must import `get_config()` rather than reading the YAML directly so that
paths and thresholds remain centralised.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    config_path = Path(os.environ.get("FVA_CONFIG", PROJECT_ROOT / "config.yaml"))
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str) -> Path:
    """Resolve a path from config. Absolute paths are returned untouched,
    relative paths are resolved against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
