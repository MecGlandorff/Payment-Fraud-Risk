from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config from disk. Returns {} if file is empty."""
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"Config path is not a file: {p}")

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ConfigError(f"YAML root must be a dict, got {type(cfg).__name__}")

    return cfg