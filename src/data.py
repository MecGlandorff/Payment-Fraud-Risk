from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(path: str) -> dict[str, Any]:
    import yaml

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
