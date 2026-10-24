"""Unit tests for model instantiation, training pipelines, and cross-validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn_prediction.config import get_config
from churn_prediction.train import (
    build_model_pipeline,
    create_candidate_models,
    cross_validate_candidates,
    select_best_model,
)


@pytest.fixture
def synthetic_training_data():
    """Create a balanced mini-dataset for training checks."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "tenure": np.random.randint(1, 72, size=n),
            "MonthlyCharges": np.random.uniform(20.0, 110.0, size=n),
            "TotalCharges": np.random.uniform(20.0, 7000.0, size=n),
            "gender": np.random.choice(["Male", "Female"], size=n),
            "SeniorCitizen": np.random.choice([0, 1], size=n),
            "Partner": np.random.choice(["Yes", "No"], size=n),
            "Dependents": np.random.choice(["Yes", "No"], size=n),
            "PhoneService": np.random.choice(["Yes", "No"], size=n),
            "MultipleLines": np.random.choice(["No phone service", "No", "Yes"], size=n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
            "PaperlessBilling": np.random.choice(["Yes", "No"], size=n),
            "PaymentMethod": np.random.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=n,
            ),
        }
    )
    # Generate realistic correlated churn
    y = ((df["tenure"] < 15) | (df["MonthlyCharges"] > 80)).astype(int)
    return df, y


def test_candidate_models_contain_required_classifiers():
    """Ensure all 3 required models are configured."""
    candidates = create_candidate_models()
    required = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    for r in required:
        assert r in candidates, f"Candidate model '{r}' is missing."


def test_build_model_pipeline_returns_valid_pipeline(synthetic_training_data):
    """Ensure pipeline compiles and can fit/predict probabilities."""
    X, y = synthetic_training_data
    candidates = create_candidate_models()
    lr = candidates["Logistic Regression"]

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pipe = build_model_pipeline(lr, num_cols, cat_cols, scale_numeric=True)
    assert isinstance(pipe, Pipeline)

    pipe.fit(X, y)
    probs = pipe.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_cross_validation_execution(synthetic_training_data):
    """Verify cross-validation runs on training data without error and returns all metrics."""
    X, y = synthetic_training_data
    config = get_config()
    config.cv_folds = 3  # Fast 3 folds for test

    cv_df, oof_preds = cross_validate_candidates(X, y, config)
    assert len(cv_df) == 3
    for col in [
        "Model",
        "CV_ROC_AUC_Mean",
        "CV_PR_AUC_Mean",
        "CV_F1_Mean",
        "CV_Precision_Mean",
        "CV_Recall_Mean",
    ]:
        assert col in cv_df.columns

    best_model, rationale = select_best_model(cv_df)
    assert best_model in ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    assert "reason" in rationale
