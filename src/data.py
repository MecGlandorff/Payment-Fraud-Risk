from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    dataset_path = Path(cfg["paths"]["dataset_csv"])

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop the unwanted columns as defined in config.
    for col in cfg["data"].get("drop_cols", []):
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped column: {col}")

    # Coerce numeric columns
    for col in cfg["features"]["numeric"]:
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_na = df[col].isna().sum()

            new_nans = after_na - before_na
            if new_nans > 0:
                print(f"{col}: {new_nans} values could not be parsed -> set to NaN")

    # Target cleanup
    target_col = cfg["data"]["target_col"]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    target_before = df[target_col].isna().sum()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    target_after = df[target_col].isna().sum()

    if target_before > 0:
        print(f"{target_col}: {target_before} NaNs set to 0")

    print("\nMissing values per column:")
    print(df.isna().sum())

    return df


def split_xy(cfg: dict[str, Any], df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target_col = cfg["data"]["target_col"]
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def train_test_split_df(cfg: dict[str, Any], X: pd.DataFrame, y: pd.Series):
    seed = int(cfg["project"]["seed"])
    test_size = float(cfg["data"]["test_size"])
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)