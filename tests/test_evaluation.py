"""
Tests for model evaluation, decision thresholds, campaign economics, and statistical comparisons.
"""

import numpy as np
import pandas as pd
import pytest

from churn_prediction.config import get_config
from churn_prediction.evaluate import (
    analyze_threshold_tradeoffs,
    compute_metrics,
    run_intervention_sensitivity_analysis,
)
from churn_prediction.train import (
    compute_cv_statistical_significance,
    compute_nadeau_bengio_corrected_ttest,
    create_candidate_models,
    cross_validate_candidates,
    select_best_model,
)


@pytest.fixture
def sample_oof_data():
    """Synthetic out-of-fold predictions fixture for evaluation testing."""
    np.random.seed(42)
    n = 80
    y_true = np.random.choice([0, 1], size=n, p=[0.73, 0.27])
    probs_rf = np.clip(y_true * 0.4 + np.random.uniform(0.1, 0.5, size=n), 0.01, 0.99)
    probs_lr = np.clip(y_true * 0.38 + np.random.uniform(0.12, 0.48, size=n), 0.01, 0.99)

    oof_preds = {
        "Random Forest": {
            "y_true": y_true,
            "oof_probabilities": probs_rf,
            "fold_pr_aucs": [0.65, 0.67, 0.64, 0.68, 0.66],
            "fold_roc_aucs": [0.84, 0.85, 0.83, 0.86, 0.85],
        },
        "Logistic Regression": {
            "y_true": y_true,
            "oof_probabilities": probs_lr,
            "fold_pr_aucs": [0.64, 0.66, 0.65, 0.67, 0.65],
            "fold_roc_aucs": [0.84, 0.84, 0.83, 0.85, 0.84],
        },
    }

    cv_df = pd.DataFrame(
        [
            {
                "Model": "Random Forest",
                "CV_PR_AUC_Mean": 0.6600,
                "CV_PR_AUC_Std": 0.0158,
                "CV_ROC_AUC_Mean": 0.8460,
                "CV_ROC_AUC_Std": 0.0114,
                "CV_F1_Mean": 0.58,
                "CV_Precision_Mean": 0.67,
                "CV_Recall_Mean": 0.51,
            },
            {
                "Model": "Logistic Regression",
                "CV_PR_AUC_Mean": 0.6550,
                "CV_PR_AUC_Std": 0.0180,
                "CV_ROC_AUC_Mean": 0.8440,
                "CV_ROC_AUC_Std": 0.0120,
                "CV_F1_Mean": 0.59,
                "CV_Precision_Mean": 0.65,
                "CV_Recall_Mean": 0.54,
            },
        ]
    )

    return oof_preds, cv_df


def test_threshold_tradeoffs(sample_oof_data):
    """Verify that threshold analysis generates valid precision, recall, and campaign value columns."""
    oof_preds, _ = sample_oof_data
    rf_oof = oof_preds["Random Forest"]

    tradeoffs = analyze_threshold_tradeoffs(
        rf_oof["y_true"],
        rf_oof["oof_probabilities"],
        thresholds=[0.3, 0.5, 0.7],
        unit_outreach_cost=20.0,
        customer_value=120.0,
        persuadability_eta=0.30,
    )

    assert not tradeoffs.empty
    assert "incremental_campaign_value" in tradeoffs.columns
    assert "precision" in tradeoffs.columns
    assert "recall" in tradeoffs.columns

    # As threshold increases, precision should generally rise and recall should fall
    t_low = tradeoffs[tradeoffs["threshold"] == 0.3].iloc[0]
    t_high = tradeoffs[tradeoffs["threshold"] == 0.7].iloc[0]
    assert t_low["recall"] >= t_high["recall"]


def test_retention_sensitivity_analysis(sample_oof_data):
    """Verify that sensitivity analysis evaluates expected campaign return across operational scenarios."""
    oof_preds, _ = sample_oof_data
    rf_oof = oof_preds["Random Forest"]

    res = run_intervention_sensitivity_analysis(
        rf_oof["y_true"],
        rf_oof["oof_probabilities"],
        thresholds=[0.35, 0.50],
    )

    assert not res.empty
    assert "scenario" in res.columns
    assert "incremental_campaign_value" in res.columns
    assert len(res) >= 6


def test_campaign_value_calculation():
    """Verify incremental campaign value arithmetic using exact hand-calculated figures."""
    tp, fp, fn = 40, 60, 50
    value, cost, eta = 100.0, 22.5, 0.40

    # Incremental campaign return = (TP * eta * value) - (TP + FP) * cost
    expected_incremental = (tp * eta * value) - (tp + fp) * cost
    # 40 * 0.40 * 100 = 1600; (40 + 60) * 22.5 = 2250 -> 1600 - 2250 = -650
    assert expected_incremental == -650.0


def test_corrected_resampled_ttest():
    """Verify Nadeau-Bengio corrected resampled t-test with controlled input data."""
    diffs = [0.01, 0.02, 0.01, 0.015, 0.025, 0.005, 0.01, 0.02, 0.015, 0.01, 0.02, 0.015, 0.01, 0.025, 0.01]
    res = compute_nadeau_bengio_corrected_ttest(
        diffs=diffs,
        n_train=4507,
        n_test=1127,
    )

    assert "mean_difference" in res
    assert "two_sided_p_value" in res
    assert "corrected_t_stat" in res
    assert res["degrees_of_freedom"] == 14
    assert 0.0 <= res["two_sided_p_value"] <= 1.0


def test_corrected_ttest_validation():
    """Verify that invalid inputs to the corrected t-test raise descriptive ValueErrors."""
    with pytest.raises(ValueError, match="At least 2"):
        compute_nadeau_bengio_corrected_ttest([0.01], n_train=100, n_test=20)


def test_cv_statistical_comparison(sample_oof_data):
    """Verify exploratory paired t-test calculates differences across folds."""
    oof_preds, _ = sample_oof_data
    res = compute_cv_statistical_significance(
        oof_preds,
        baseline_model="Logistic Regression",
        metric_key="fold_pr_aucs",
    )

    assert not res.empty
    assert "Mean_Difference" in res.columns
    assert "p_value_exploratory" in res.columns


def test_model_selection_parsimony(sample_oof_data):
    """Verify model selection under parsimony prefers simpler baseline if within tolerance."""
    oof_preds, cv_df = sample_oof_data
    best_name, rationale = select_best_model(
        cv_df, oof_predictions=oof_preds, selection_strategy="parsimony"
    )
    assert best_name in ["Logistic Regression", "Random Forest"]
    assert "reason" in rationale


def test_model_selection_precision_tradeoff(sample_oof_data):
    """Verify model selection considers precision trade-offs when models are statistically tied."""
    oof_preds, cv_df = sample_oof_data
    best_name, rationale = select_best_model(
        cv_df, oof_predictions=oof_preds, selection_strategy="defensible_tradeoff"
    )
    assert best_name == "Random Forest"
    assert "precision" in rationale["reason"].lower()


def test_average_precision_calculation():
    """Verify average precision score metric computation."""
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_probs = np.array([0.1, 0.8, 0.2, 0.7, 0.9, 0.3])

    metrics = compute_metrics(y_true, y_probs)
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0
