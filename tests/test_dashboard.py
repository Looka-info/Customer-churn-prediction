"""
Unit and integration tests for dashboard services, artifact loaders, and prediction interfaces.
"""

from pathlib import Path
import sqlite3
import pytest
import pandas as pd

from churn_prediction.config import get_config
from churn_prediction.predict import REQUIRED_FEATURES
from dashboard.data_service import (
    check_artifacts_status,
    get_dataset_summary,
    load_dataset_cached,
)
from dashboard.model_service import (
    get_best_model_summary,
    get_customer_presets,
    get_feature_schema,
    get_model_comparison_report,
    run_customer_prediction,
)
from dashboard.analytics_service import (
    ANALYTICAL_QUERIES,
    execute_readonly_query,
    is_database_available,
)


def test_artifacts_status_check():
    """Verify that artifact status checking inspects all critical pipeline assets."""
    status = check_artifacts_status()
    expected_keys = [
        "Raw Dataset",
        "SQLite Database",
        "Trained Model Pipeline",
        "Model Comparison Report",
        "Diagnostic Figures Directory",
    ]
    for key in expected_keys:
        assert key in status, f"Missing key '{key}' in artifacts status dictionary."
        assert "exists" in status[key]
        assert "size" in status[key]
        assert "modified" in status[key]
        assert "remediation" in status[key]


def test_dataset_summary_computation():
    """Verify that dataset summary statistics compute correctly from the verified data."""
    summary = get_dataset_summary()
    assert summary is not None, "Dataset summary returned None."
    assert summary["total_records"] == 7043
    assert summary["churned_count"] == 1869
    assert summary["retained_count"] == 5174
    assert abs(summary["churn_rate_pct"] - 26.54) < 0.05
    assert "numeric_summary" in summary
    assert "tenure" in summary["numeric_summary"]
    assert "MonthlyCharges" in summary["numeric_summary"]
    assert "TotalCharges" in summary["numeric_summary"]


def test_model_comparison_report_loader():
    """Verify that model comparison report loads and contains required candidate models."""
    df = get_model_comparison_report()
    assert df is not None, "Model comparison report failed to load."
    assert not df.empty
    models = list(df["Model"])
    assert "Random Forest" in models
    assert "Gradient Boosting" in models
    assert "Logistic Regression" in models
    assert "Test_PR_AUC" in df.columns
    assert "CV_PR_AUC_Mean" in df.columns


def test_best_model_summary_extraction():
    """Verify that best model summary correctly identifies Random Forest and extracts metrics."""
    summary = get_best_model_summary()
    assert summary["model_exists"] is True
    assert summary["selected_model"] == "Random Forest"
    assert summary["candidate_count"] == 3
    assert 0.0 < summary["cv_pr_auc"] < 1.0
    assert 0.0 < summary["test_pr_auc"] < 1.0
    assert 0.0 < summary["test_precision"] < 1.0


def test_customer_presets_schema_completeness():
    """Verify that every preset in the customer presets catalogue contains all required features."""
    presets = get_customer_presets()
    assert len(presets) >= 3

    for name, payload in presets.items():
        for feat in REQUIRED_FEATURES:
            assert feat in payload, f"Preset '{name}' is missing required feature: '{feat}'"


def test_customer_prediction_via_service():
    """Verify that run_customer_prediction returns valid probability and risk tier."""
    presets = get_customer_presets()
    high_risk_payload = presets["High Churn Risk (Month-to-Month Fiber)"]

    result = run_customer_prediction(high_risk_payload, threshold=0.50)
    assert "churn_probability" in result
    assert "predicted_churn" in result
    assert "risk_tier" in result
    assert "recommended_action" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["predicted_churn"] in [0, 1]
    assert result["risk_tier"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert len(result["recommended_action"]) > 0


def test_customer_prediction_excludes_id_and_churn():
    """Verify that passing customerID or target Churn does not contaminate prediction."""
    presets = get_customer_presets()
    payload = presets["Low Churn Risk (Two-Year Contract Family DSL)"].copy()
    payload["customerID"] = "TEST-9999"
    payload["Churn"] = 1  # Intentionally conflicting target label

    result = run_customer_prediction(payload, threshold=0.50)
    assert 0.0 <= result["churn_probability"] <= 1.0
    # Customer should still be predicted with low probability despite Churn=1 passed in payload
    assert result["predicted_churn"] == 0


def test_customer_prediction_rejects_missing_features():
    """Verify that omitting required features raises ValueError."""
    invalid_payload = {"gender": "Female", "tenure": 12}
    with pytest.raises(ValueError, match="missing required feature columns"):
        run_customer_prediction(invalid_payload)


def test_database_availability():
    """Verify that SQLite database availability is correctly detected."""
    assert is_database_available() is True


def test_sql_readonly_execution_safety():
    """Verify that analytical queries execute and database is strictly read-only."""
    query = "SELECT COUNT(*) AS cnt FROM telco_churn;"
    df = execute_readonly_query(query)
    assert not df.empty
    assert int(df.iloc[0]["cnt"]) == 7043

    # Attempting a write query should fail due to read-only URI mode
    write_query = "UPDATE telco_churn SET tenure = 999 WHERE rowid = 1;"
    with pytest.raises((sqlite3.OperationalError, pd.errors.DatabaseError), match="readonly"):
        execute_readonly_query(write_query)


def test_all_catalog_queries_execute():
    """Verify that every query in the analytical catalog runs cleanly."""
    for q in ANALYTICAL_QUERIES:
        df = execute_readonly_query(q["sql"])
        assert not df.empty, f"Query '{q['id']}' returned an empty DataFrame."
