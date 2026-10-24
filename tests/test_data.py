"""Unit tests for data loading, schema validation, and raw data hygiene."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from churn_prediction.config import get_config
from churn_prediction.data_loader import (
    clean_raw_data,
    load_raw_data,
    prepare_features_and_target,
    validate_raw_data,
    EXPECTED_COLUMNS,
)


@pytest.fixture
def sample_valid_df():
    """Create a minimal valid DataFrame matching Telco schema."""
    data = {
        "customerID": ["001-AAA", "002-BBB", "003-CCC", "004-DDD"],
        "gender": ["Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [1, 34, 0, 72],
        "PhoneService": ["No", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "Yes", "No", "No internet service"],
        "OnlineBackup": ["Yes", "No", "Yes", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No", "No internet service"],
        "TechSupport": ["No", "No", "No", "No internet service"],
        "StreamingTV": ["No", "No", "No", "No internet service"],
        "StreamingMovies": ["No", "No", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        "MonthlyCharges": [29.85, 89.10, 45.20, 25.00],
        "TotalCharges": ["29.85", "3029.40", " ", "1800.00"],
        "Churn": ["No", "Yes", "No", "No"],
    }
    return pd.DataFrame(data)


def test_schema_validation_success(sample_valid_df):
    """Verify that a schema-conforming dataset passes validation."""
    # Add dummy rows to reach minimum count if needed or pass with len check
    large_df = pd.concat([sample_valid_df] * 30, ignore_index=True)
    # Give unique customer IDs
    large_df["customerID"] = [f"ID_{i}" for i in range(len(large_df))]
    report = validate_raw_data(large_df)
    assert report["is_valid"] is True
    assert report["num_rows"] == len(large_df)
    assert report["duplicate_customer_ids"] == 0
    assert report["blank_total_charges_count"] > 0


def test_schema_validation_missing_column(sample_valid_df):
    """Verify validation fails if a required column is missing."""
    invalid_df = sample_valid_df.drop(columns=["MonthlyCharges"])
    large_df = pd.concat([invalid_df] * 30, ignore_index=True)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_data(large_df)


def test_schema_validation_duplicate_ids(sample_valid_df):
    """Verify validation catches duplicate customer IDs."""
    dup_df = pd.concat([sample_valid_df] * 30, ignore_index=True)
    # duplicate IDs exist because of concat
    with pytest.raises(ValueError, match="duplicate customer IDs"):
        validate_raw_data(dup_df)


def test_clean_raw_data_handles_blank_total_charges(sample_valid_df):
    """Verify that blank string TotalCharges is coerced and tenure=0 is set to 0.0."""
    cleaned = clean_raw_data(sample_valid_df)
    assert cleaned["TotalCharges"].dtype in [np.float64, float]
    # Row index 2 had tenure=0 and " "
    assert cleaned.loc[2, "TotalCharges"] == 0.0
    assert cleaned.loc[2, "tenure"] == 0


def test_target_encoding(sample_valid_df):
    """Verify target 'Churn' is correctly mapped to binary 1 (Yes) and 0 (No)."""
    cleaned = clean_raw_data(sample_valid_df)
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
    assert cleaned.loc[0, "Churn"] == 0  # Was "No"
    assert cleaned.loc[1, "Churn"] == 1  # Was "Yes"


def test_prepare_features_and_target_leakage_prevention(sample_valid_df):
    """
    Critical test: Ensure customerID and Churn are completely removed from features X.
    """
    X, y = prepare_features_and_target(sample_valid_df)
    assert "customerID" not in X.columns, "Data Leakage: customerID must not be in model features X!"
    assert "Churn" not in X.columns, "Data Leakage: Churn must not be in model features X!"
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})


def test_load_raw_data_missing_file_raises_error():
    """Verify clear FileNotFoundError when data path is nonexistent."""
    nonexistent = Path("data/raw/nonexistent_file_xyz.csv")
    with pytest.raises(FileNotFoundError, match="Dataset not found at"):
        load_raw_data(nonexistent)
