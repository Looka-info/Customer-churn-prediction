"""Unit tests for inference, input validation, serialization, and ChurnPredictor."""

from pathlib import Path
import tempfile
import joblib
import numpy as np
import pandas as pd
import pytest

from churn_prediction.config import get_config
from churn_prediction.predict import ChurnPredictor, REQUIRED_FEATURES
from churn_prediction.preprocessing import build_preprocessor
from churn_prediction.train import build_model_pipeline, create_candidate_models


@pytest.fixture
def trained_pipeline_fixture(tmp_path):
    """Create and persist a minimal trained pipeline for testing inference."""
    np.random.seed(42)
    n = 60
    data = {}
    for feat in REQUIRED_FEATURES:
        if feat in ["tenure"]:
            data[feat] = np.random.randint(1, 72, size=n)
        elif feat in ["MonthlyCharges", "TotalCharges"]:
            data[feat] = np.random.uniform(20.0, 100.0, size=n)
        elif feat in ["SeniorCitizen"]:
            data[feat] = np.random.choice([0, 1], size=n)
        else:
            data[feat] = np.random.choice(["Category_A", "Category_B"], size=n)

    X = pd.DataFrame(data)
    y = np.random.choice([0, 1], size=n)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in REQUIRED_FEATURES if c not in num_cols]

    estimator = create_candidate_models()["Logistic Regression"]
    pipeline = build_model_pipeline(estimator, num_cols, cat_cols, scale_numeric=True)
    pipeline.fit(X, y)

    model_file = tmp_path / "test_model.joblib"
    joblib.dump(pipeline, model_file)
    return model_file, X.iloc[0:2].copy()


def test_prediction_output_shape_and_probability_bounds(trained_pipeline_fixture):
    """Verify prediction outputs have correct columns and valid [0, 1] probability range."""
    model_path, sample_data = trained_pipeline_fixture
    predictor = ChurnPredictor(model_path=model_path)

    results = predictor.predict(sample_data)
    assert len(results) == len(sample_data)
    assert "churn_probability" in results.columns
    assert "predicted_churn" in results.columns
    assert "risk_tier" in results.columns

    probs = results["churn_probability"].values
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert set(results["predicted_churn"].unique()).issubset({0, 1})


def test_prediction_handles_dictionary_input(trained_pipeline_fixture):
    """Verify inference accepts a single dictionary payload."""
    model_path, sample_data = trained_pipeline_fixture
    predictor = ChurnPredictor(model_path=model_path)

    single_dict = sample_data.iloc[0].to_dict()
    results = predictor.predict(single_dict)
    assert len(results) == 1
    assert results.iloc[0]["prediction_label"] in ["Likely to Churn", "Likely to Retain"]


def test_prediction_validation_catches_missing_columns(trained_pipeline_fixture):
    """Verify predictor raises ValueError if required features are omitted."""
    model_path, sample_data = trained_pipeline_fixture
    predictor = ChurnPredictor(model_path=model_path)

    bad_data = sample_data.drop(columns=["tenure", "MonthlyCharges"])
    with pytest.raises(ValueError, match="missing required feature columns"):
        predictor.predict(bad_data)


def test_prediction_safely_removes_customer_id_and_churn(trained_pipeline_fixture):
    """Verify input validation drops customerID and Churn if passed in input payload."""
    model_path, sample_data = trained_pipeline_fixture
    predictor = ChurnPredictor(model_path=model_path)

    data_with_extras = sample_data.copy()
    data_with_extras["customerID"] = ["CUST_001", "CUST_002"]
    data_with_extras["Churn"] = ["Yes", "No"]
    data_with_extras["unrelated_col"] = [123, 456]

    # Should succeed without error
    results = predictor.predict(data_with_extras)
    assert len(results) == 2


def test_predictor_missing_model_file_error():
    """Verify clear FileNotFoundError when model path does not exist."""
    fake_path = Path("models/nonexistent_model_12345.joblib")
    with pytest.raises(FileNotFoundError, match="Trained model artifact not found"):
        ChurnPredictor(model_path=fake_path)
