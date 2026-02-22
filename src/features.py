from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def prepare_X(cfg: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Select configured features and coerce types (numerics->numeric, cats->string)"""
    cat_cols = cfg["features"]["categorical"]
    num_cols = cfg["features"]["numeric"]
    cols = cat_cols + num_cols

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")

    X = df[cols].copy()

    # numeric
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # categorical (keep NaN, cast non-null to str)
    for c in cat_cols:
        X[c] = X[c].where(X[c].isna(), X[c].astype(str))

    # quick audit
    print("\nMissing values (before imputation):")
    print(X.isna().sum())

    return X


def fit_preprocessor(cfg: dict[str, Any], X: pd.DataFrame) -> ColumnTransformer:
    """Fit a preprocessing pipeline: median impute for numeric, one-hot for categorical."""
    cat_cols = cfg["features"]["categorical"]
    num_cols = cfg["features"]["numeric"]
    min_freq = int(cfg["features"].get("one_hot_min_freq", 1))

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_freq,
                    sparse_output=False,
                ),
            ),
        ]
    )

    pre = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pre.fit(X)
    return pre


def get_feature_names(pre: ColumnTransformer) -> list[str]:
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        return []