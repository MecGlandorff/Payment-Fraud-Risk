from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DERIVED_NUMERIC = [
    "amount_to_avg",
    "amount_minus_avg",
    "attempts_pressure",
    "young_account",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple signal engineered features
    Assumes base columns exist, if missing, feature becomes NaN
    """
    out = df.copy()

    # Helper getters that return NaNs if missing
    def col(name: str):
        return out[name] if name in out.columns else pd.Series([np.nan] * len(out), index=out.index)

    amount = pd.to_numeric(col("transaction_amount"), errors="coerce")
    avg_amount = pd.to_numeric(col("avg_transaction_amount"), errors="coerce")
    prev_fail = pd.to_numeric(col("previous_failed_attempts"), errors="coerce")
    logins = pd.to_numeric(col("login_attempts_last_24h"), errors="coerce")
    age = pd.to_numeric(col("account_age_days"), errors="coerce")

    out["amount_to_avg"] = amount / (avg_amount + 1e-6)
    out["amount_minus_avg"] = amount - avg_amount
    out["attempts_pressure"] = logins + prev_fail
    out["young_account"] = 1.0 / (age + 1.0)

    return out


def prepare_X(cfg: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Select configured base features + engineered features and coerce types.
    Prints missing counts before imputation.
    """
    # Add engineered features first 
    df = add_derived_features(df)

    cat_cols = cfg["features"]["categorical"]
    num_cols = cfg["features"]["numeric"]

    # Ensure engineered features are included as numeric
    for f in DERIVED_NUMERIC:
        if f not in num_cols:
            num_cols = num_cols + [f]

    cols = cat_cols + num_cols
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")

    X = df[cols].copy()

    # numeric coercion
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # categorical coercion (keep NaN and non-null to str)
    for c in cat_cols:
        X[c] = X[c].where(X[c].isna(), X[c].astype(str))

    print("\nMissing values (before imputation):")
    print(X.isna().sum())

    return X


def fit_preprocessor(cfg: dict[str, Any], X: pd.DataFrame) -> ColumnTransformer:
    """
    Fit preprocessing:
    - Numeric: median impute
    - Categorical: most_frequent impute + one-hot
    """
    cat_cols = cfg["features"]["categorical"]
    num_cols = list(cfg["features"]["numeric"])

    # include engineered numeric features
    for f in DERIVED_NUMERIC:
        if f not in num_cols:
            num_cols.append(f)

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