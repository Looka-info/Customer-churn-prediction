"""Preprocessing pipelines and data splitting with strict data-leakage prevention."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_prediction.config import AppConfig, get_config


def get_feature_lists(
    df: pd.DataFrame, config: Optional[AppConfig] = None
) -> Tuple[List[str], List[str]]:
    """Extract and validate numeric and categorical feature lists based on configuration and data."""
    cfg = config or get_config()

    numeric_features = [col for col in cfg.numeric_features if col in df.columns]
    categorical_features = [col for col in cfg.categorical_features if col in df.columns]

    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer that handles numeric imputation/scaling
    and categorical imputation/one-hot encoding safely.

    Handles unseen categorical levels safely via `handle_unknown='ignore'`.
    """
    # Numerical pipeline: impute missing with median, optionally scale with StandardScaler
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    # Categorical pipeline: impute missing with mode, one-hot encode
    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop="if_binary",
            ),
        ),
    ]
    categorical_pipeline = Pipeline(steps=cat_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into stratified training and held-out test sets.
    The test set must remain completely isolated until final evaluation.
    """
    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )
    return X_train, X_test, y_train, y_test
