"""Data loading, schema validation, and raw data hygiene for Telco Customer Churn."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import numpy as np
import pandas as pd

from churn_prediction.config import AppConfig, get_config

logger = logging.getLogger(__name__)

OFFICIAL_TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def download_dataset_if_missing(
    dest_path: Path,
    url: str = OFFICIAL_TELCO_URL,
) -> Path:
    """Download the official IBM Telco Customer Churn dataset if not present."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        logger.info("Dataset not found at %s. Downloading from %s...", dest_path, url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())
        logger.info("Successfully downloaded raw dataset to %s", dest_path)
    return dest_path


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw Telco dataset from CSV.
    Fails with an informative message if the file is absent.
    """
    config = get_config()
    target_path = file_path or config.raw_data_path

    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {target_path}\n"
            "Please ensure the official IBM Telco Customer Churn dataset is placed at that path.\n"
            "You can run `python train.py` (which downloads it automatically if missing) or fetch it from:\n"
            f"1) Official IBM repo: {OFFICIAL_TELCO_URL}\n"
            "2) Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
        )

    df = pd.read_csv(target_path)
    return df


def validate_raw_data(df: pd.DataFrame, config: Optional[AppConfig] = None) -> Dict[str, Any]:
    """
    Validate dataset dimensions, column names, target presence, and key invariants.
    Raises ValueError if critical validation fails.
    """
    cfg = config or get_config()
    report: Dict[str, Any] = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_columns": [],
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": 0,
        "blank_total_charges_count": 0,
        "target_distribution": {},
        "is_valid": True,
    }

    # Verify column presence
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    report["missing_columns"] = missing_cols
    if missing_cols:
        report["is_valid"] = False
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Check minimum row count
    if len(df) < 100:
        report["is_valid"] = False
        raise ValueError(f"Dataset row count ({len(df)}) is too small for a valid Telco dataset.")

    # Check duplicate customer identifiers
    if cfg.id_column in df.columns:
        dup_ids = int(df[cfg.id_column].duplicated().sum())
        report["duplicate_customer_ids"] = dup_ids
        if dup_ids > 0:
            report["is_valid"] = False
            raise ValueError(f"Found {dup_ids} duplicate customer IDs in {cfg.id_column}.")

    # Check target values
    if cfg.target_column in df.columns:
        unique_targets = set(df[cfg.target_column].dropna().unique())
        valid_targets = {cfg.positive_label, cfg.negative_label, 1, 0, "1", "0", 1.0, 0.0}
        if not unique_targets.issubset(valid_targets):
            report["is_valid"] = False
            raise ValueError(
                f"Unexpected target values in {cfg.target_column}: {unique_targets}. "
                f"Expected subset of {valid_targets}."
            )
        report["target_distribution"] = df[cfg.target_column].value_counts().to_dict()

    # Detect blank string TotalCharges
    if "TotalCharges" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["TotalCharges"]):
            blank_count = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
            report["blank_total_charges_count"] = blank_count

    return report


def clean_raw_data(df: pd.DataFrame, config: Optional[AppConfig] = None) -> pd.DataFrame:
    """
    Clean the raw dataset for exploratory analysis and modeling preparation:
    1. Makes a copy to prevent in-place mutation.
    2. Converts TotalCharges to numeric, safely replacing whitespace strings with NaN.
       For customers with tenure == 0, TotalCharges is logically 0.0 (no billing cycles completed).
       We fill these specific 0-tenure missing charges with 0.0, and retain np.nan for any others.
    3. Encodes target 'Churn' as binary integer (1 for Yes, 0 for No) if present.
    4. Ensures SeniorCitizen is treated as a clean categorical/binary feature.
    """
    cfg = config or get_config()
    clean_df = df.copy()

    # 1. TotalCharges cleaning
    if "TotalCharges" in clean_df.columns:
        # Convert whitespace to NaN then numeric
        clean_df["TotalCharges"] = pd.to_numeric(
            clean_df["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )
        # Logical imputation for tenure == 0: customers who just joined have incurred 0 total charges
        tenure_zero_mask = (clean_df["tenure"] == 0) & (clean_df["TotalCharges"].isna())
        clean_df.loc[tenure_zero_mask, "TotalCharges"] = 0.0

    # 2. Target encoding
    if cfg.target_column in clean_df.columns:
        # Map string labels to binary integers 1 / 0
        if clean_df[cfg.target_column].dtype == object or isinstance(
            clean_df[cfg.target_column].iloc[0], str
        ):
            mapping = {cfg.positive_label: 1, cfg.negative_label: 0}
            clean_df[cfg.target_column] = clean_df[cfg.target_column].map(mapping)

    return clean_df


def prepare_features_and_target(
    df: pd.DataFrame, config: Optional[AppConfig] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates predictor features from the target variable.
    Crucially drops customerID from features to prevent identifier leakage.
    Returns:
        X: DataFrame of predictor features (without customerID, without target)
        y: Series of target labels (binary 0/1)
    """
    cfg = config or get_config()
    cleaned = clean_raw_data(df, cfg)

    if cfg.target_column not in cleaned.columns:
        raise KeyError(f"Target column '{cfg.target_column}' not found in dataframe.")

    y = cleaned[cfg.target_column].astype(int)

    # Exclude ID column and Target column from features
    cols_to_drop = [cfg.target_column]
    if cfg.id_column in cleaned.columns:
        cols_to_drop.append(cfg.id_column)

    X = cleaned.drop(columns=cols_to_drop)
    return X, y
