"""
features.py — Shared preprocessing pipeline for online shopper intention dataset.
Used by both train.py and the FastAPI inference server.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month",
    "VisitorType",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "Weekend",
]

TARGET = "Revenue"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Month ordering for ordinal encoding (not used here — OHE handles it fine)
MONTH_ORDER = ["Feb", "Mar", "May", "Jun", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return X, y."""
    df = pd.read_csv(path)
    # Normalize Weekend column to string so OHE is consistent
    df["Weekend"] = df["Weekend"].astype(str)
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(int)  # True→1 (purchase), False→0 (no purchase)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer that:
      - Imputes + scales numeric features
      - Imputes + one-hot-encodes categorical features
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def session_dict_to_dataframe(session: dict) -> pd.DataFrame:
    """Convert a single API session dict → single-row DataFrame."""
    row = {col: session.get(col, None) for col in ALL_FEATURES}
    df = pd.DataFrame([row])
    df["Weekend"] = df["Weekend"].astype(str)
    return df
