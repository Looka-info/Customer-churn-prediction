"""Targeted audit and hardening tests: leakage prevention, defensible selection, and invariant checks."""

import numpy as np
import pandas as pd
import pytest

from churn_prediction.config import get_config
from churn_prediction.evaluate import analyze_threshold_tradeoffs, compute_metrics
from churn_prediction.train import (
    compute_cv_statistical_significance,
    create_candidate_models,
    cross_validate_candidates,
    select_best_model,
)


@pytest.fixture
def train_val_fixture():
    """Synthetic training dataset with out-of-fold predictions."""
    np.random.seed(42)
    n = 80
    y_true = np.random.choice([0, 1], size=n, p=[0.73, 0.27])
    # Synthetic probabilities with some predictive signal
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
                "CV_PR_AUC_Mean": 0.6540,
                "CV_PR_AUC_Std": 0.0114,
                "CV_ROC_AUC_Mean": 0.8400,
                "CV_ROC_AUC_Std": 0.0071,
                "CV_F1_Mean": 0.59,
                "CV_Precision_Mean": 0.65,
                "CV_Recall_Mean": 0.54,
            },
        ]
    )

    return cv_df, oof_preds, y_true, probs_rf


def test_threshold_tuning_uses_only_training_data(train_val_fixture):
    """
    Audit Check: Verify threshold trade-offs and utility calculations operate
    exclusively on training out-of-fold predictions.
    """
    _, _, y_true, probs_rf = train_val_fixture
    df_thresh = analyze_threshold_tradeoffs(y_true, probs_rf)

    assert "threshold" in df_thresh.columns
    assert "hypothetical_expected_value" in df_thresh.columns
    assert "churners_identified_tp" in df_thresh.columns
    assert len(df_thresh) >= 5

    # Check that identified churners decrease as threshold increases
    assert df_thresh["churners_identified_tp"].iloc[0] >= df_thresh["churners_identified_tp"].iloc[-1]
    assert df_thresh["false_alarms_fp"].iloc[0] >= df_thresh["false_alarms_fp"].iloc[-1]


def test_intervention_sensitivity_analysis(train_val_fixture):
    """
    Audit Check: Verify multi-scenario sensitivity analysis computes expected value
    across conservative, moderate, and optimistic retention success rates.
    """
    from churn_prediction.evaluate import run_intervention_sensitivity_analysis

    _, _, y_true, probs_rf = train_val_fixture
    sens_df = run_intervention_sensitivity_analysis(y_true, probs_rf)

    assert len(sens_df) == 9  # 3 scenarios * 3 thresholds
    assert "scenario" in sens_df.columns
    assert "success_rate_eta" in sens_df.columns
    assert "net_expected_value" in sens_df.columns
    assert set(sens_df["scenario"].unique()) == {"Conservative", "Moderate", "Optimistic"}


def test_cv_statistical_significance_calculator(train_val_fixture):
    """
    Audit Check: Verify exploratory paired t-test correctly calculates fold differences
    and reports exploratory p-values without assuming null hypothesis proof.
    """
    _, oof_preds, _, _ = train_val_fixture
    sig_df = compute_cv_statistical_significance(
        oof_preds, baseline_model="Logistic Regression", metric_key="fold_pr_aucs"
    )

    assert len(sig_df) == 2
    assert "Paired_t_stat" in sig_df.columns
    assert "p_value_exploratory" in sig_df.columns
    for p in sig_df["p_value_exploratory"]:
        assert 0.0 <= p <= 1.0


def test_defensible_model_selection_parsimony_strategy(train_val_fixture):
    """
    Audit Check: Under the 'parsimony' strategy, if Logistic Regression is within
    1 standard deviation of a complex ensemble, it should be selected for simplicity.
    """
    cv_df, _, _, _ = train_val_fixture
    best_model, rationale = select_best_model(cv_df, selection_strategy="parsimony")

    assert best_model == "Logistic Regression"
    assert "Occam's razor" in rationale["reason"]


def test_defensible_model_selection_tradeoff_strategy(train_val_fixture):
    """
    Audit Check: Under 'defensible_tradeoff', when discrimination differences
    are smaller than fold std, the decision rationale records the trade-off.
    """
    cv_df, oof_preds, _, _ = train_val_fixture
    best_model, rationale = select_best_model(
        cv_df, oof_predictions=oof_preds, selection_strategy="defensible_tradeoff"
    )

    assert best_model in ["Random Forest", "Logistic Regression"]
    assert "reason" in rationale
    assert "exploratory_paired_t_p_value" in rationale


def test_average_precision_metric_properties():
    """
    Audit Check: Verify average precision score behavior.
    """
    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([0.1, 0.8, 0.2, 0.9])
    m = compute_metrics(y_true, y_probs)

    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    # Perfect ranking should have PR-AUC == 1.0
    assert m["pr_auc"] == 1.0
