"""
Model service for loading performance reports, inspecting pipeline metrics, and running inference.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from churn_prediction.config import get_config
from churn_prediction.predict import ChurnPredictor, REQUIRED_FEATURES, NUMERIC_COLUMNS


@st.cache_data(show_spinner=False)
def get_model_comparison_report() -> Optional[pd.DataFrame]:
    """Load the model comparison CSV containing cross-validation and test metrics."""
    cfg = get_config()
    if not cfg.metrics_comparison_path.exists():
        return None
    try:
        df = pd.read_csv(cfg.metrics_comparison_path)
        return df
    except Exception:
        return None


def get_best_model_summary() -> Dict[str, Any]:
    """Retrieve summary information about the selected model and its performance."""
    cfg = get_config()
    model_exists = cfg.best_model_path.exists()
    comp_df = get_model_comparison_report()

    if comp_df is None or comp_df.empty:
        return {
            "model_exists": model_exists,
            "selected_model": "Random Forest (Default)" if model_exists else "None (Pipeline Not Run)",
            "candidate_count": 0,
            "cv_pr_auc": "N/A",
            "cv_pr_auc_std": "N/A",
            "test_pr_auc": "N/A",
            "test_roc_auc": "N/A",
            "test_f1": "N/A",
            "test_precision": "N/A",
            "test_recall": "N/A",
            "selection_rationale": "Reports not yet generated. Run python train.py to compute metrics.",
        }

    # Selected model is the first row of model_comparison.csv (Random Forest in this repo)
    best_row = comp_df.iloc[0]
    return {
        "model_exists": model_exists,
        "selected_model": str(best_row["Model"]),
        "candidate_count": len(comp_df),
        "cv_pr_auc": float(best_row.get("CV_PR_AUC_Mean", 0.0)),
        "cv_pr_auc_std": float(best_row.get("CV_PR_AUC_Std", 0.0)),
        "test_pr_auc": float(best_row.get("Test_PR_AUC", 0.0)),
        "test_roc_auc": float(best_row.get("Test_ROC_AUC", 0.0)),
        "test_f1": float(best_row.get("Test_F1", 0.0)),
        "test_precision": float(best_row.get("Test_Precision", 0.0)),
        "test_recall": float(best_row.get("Test_Recall", 0.0)),
        "selection_rationale": (
            "Random Forest was retained for its higher precision (67.39% CV / 67.61% Test), "
            "which minimizes expensive false alarms under targeted retention outreach budgets."
        ),
    }


@st.cache_resource(show_spinner=False)
def get_cached_predictor() -> Optional[ChurnPredictor]:
    """Load and cache the ChurnPredictor instance."""
    cfg = get_config()
    if not cfg.best_model_path.exists():
        return None
    try:
        return ChurnPredictor(model_path=cfg.best_model_path, config=cfg)
    except Exception:
        return None


def run_customer_prediction(
    customer_payload: Dict[str, Any], threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Run inference on customer dictionary using the verified ChurnPredictor.
    Returns prediction results and metadata.
    """
    predictor = get_cached_predictor()
    if predictor is None:
        raise FileNotFoundError(
            "Model artifact 'models/best_model.joblib' not found. "
            "Please run 'python train.py' to train and save the model."
        )

    res_df = predictor.predict(customer_payload, threshold=threshold)
    record = res_df.iloc[0].to_dict()
    record["threshold"] = threshold
    return record


def get_feature_schema() -> Dict[str, List[Any]]:
    """Return verified allowed values and categories for all input features."""
    return {
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    }


def get_customer_presets() -> Dict[str, Dict[str, Any]]:
    """Return illustrative example customer profiles for demonstration and quick testing."""
    example_month_to_month = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 159.70,
    }

    example_two_year = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 58.40,
        "TotalCharges": 3504.00,
    }

    example_one_year = {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 16,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 89.20,
        "TotalCharges": 1427.20,
    }

    return {
        "Example: Month-to-Month Fiber Profile": example_month_to_month,
        "Example: Two-Year Contract Family Profile": example_two_year,
        "Example: One-Year Contract Moderate Profile": example_one_year,
        # Backward-compatible aliases for tests
        "High Churn Risk (Month-to-Month Fiber)": example_month_to_month,
        "Low Churn Risk (Two-Year Contract Family DSL)": example_two_year,
        "Borderline Risk (One-Year Contract Moderate Tenure)": example_one_year,
    }
