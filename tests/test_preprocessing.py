"""Unit tests for preprocessing transformers, pipelines, and train/test splitting."""

import numpy as np
import pandas as pd
import pytest

from churn_prediction.preprocessing import (
    build_preprocessor,
    create_train_test_split,
)


@pytest.fixture
def feature_fixture():
    """Small representative DataFrame for preprocessing tests."""
    df = pd.DataFrame(
        {
            "tenure": [12, 24, np.nan, 48, 60],
            "MonthlyCharges": [50.0, 75.0, 90.0, np.nan, 20.0],
            "TotalCharges": [600.0, 1800.0, 2700.0, 4000.0, 1200.0],
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "Two year",
            ],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
            ],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0])
    return df, y


def test_build_preprocessor_handles_missing_and_scaling(feature_fixture):
    """Test that preprocessor fits and transforms data without error even with NaNs."""
    X, _ = feature_fixture
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = ["gender", "Contract", "PaymentMethod"]

    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    transformed = preprocessor.fit_transform(X)

    # Output should be 2D array without NaN
    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any(), "Preprocessed matrix should contain no NaNs!"
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(num_cols) + len(cat_cols)  # One-hot expansion


def test_preprocessor_handles_unseen_categories(feature_fixture):
    """
    Critical test: Preprocessor must not crash when encountering previously unseen
    categories in test/inference records (handle_unknown='ignore').
    """
    X_train, _ = feature_fixture
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = ["gender", "Contract", "PaymentMethod"]

    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(X_train)

    # Create unseen category data
    X_unseen = pd.DataFrame(
        {
            "tenure": [30],
            "MonthlyCharges": [85.0],
            "TotalCharges": [2550.0],
            "gender": ["NonBinary"],  # Unseen category
            "Contract": ["Three Year Flexible"],  # Unseen category
            "PaymentMethod": ["Crypto Transfer"],  # Unseen category
        }
    )

    transformed_unseen = preprocessor.transform(X_unseen)
    assert transformed_unseen.shape[0] == 1
    assert transformed_unseen.shape[1] == preprocessor.transform(X_train).shape[1]
    assert not np.isnan(transformed_unseen).any()


def test_create_train_test_split_stratification_and_seed(feature_fixture):
    """Verify train_test_split stratifies the target and reproduces identically with seed."""
    X, y = feature_fixture
    # Expand fixture to allow splitting
    X_large = pd.concat([X] * 10, ignore_index=True)
    y_large = pd.concat([y] * 10, ignore_index=True)

    X_train1, X_test1, y_train1, y_test1 = create_train_test_split(
        X_large, y_large, test_size=0.20, random_state=42, stratify=True
    )
    X_train2, X_test2, y_train2, y_test2 = create_train_test_split(
        X_large, y_large, test_size=0.20, random_state=42, stratify=True
    )

    # Identical with same seed
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_series_equal(y_train1, y_train2)

    # Proportions approximately preserved
    train_prop = y_train1.mean()
    test_prop = y_test1.mean()
    assert abs(train_prop - test_prop) < 0.10
