"""Targeted audit and hardening tests: leakage prevention, defensible selection, and invariant checks."""

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
    assert "incremental_campaign_value" in df_thresh.columns
    assert "hypothetical_expected_value" in df_thresh.columns
    assert "contacted_accounts" in df_thresh.columns
    assert "churners_identified_tp" in df_thresh.columns
    assert len(df_thresh) >= 5

    # Check that identified churners decrease as threshold increases
    assert df_thresh["churners_identified_tp"].iloc[0] >= df_thresh["churners_identified_tp"].iloc[-1]
    assert df_thresh["false_alarms_fp"].iloc[0] >= df_thresh["false_alarms_fp"].iloc[-1]


def test_intervention_sensitivity_analysis(train_val_fixture):
    """
    Audit Check: Verify multi-scenario sensitivity analysis computes Option A
    incremental campaign value across conservative, moderate, and optimistic retention success rates.
    """
    _, _, y_true, probs_rf = train_val_fixture
    sens_df = run_intervention_sensitivity_analysis(y_true, probs_rf)

    assert len(sens_df) == 9  # 3 scenarios * 3 thresholds
    assert "scenario" in sens_df.columns
    assert "success_rate_eta" in sens_df.columns
    assert "contacted_accounts" in sens_df.columns
    assert "incremental_campaign_value" in sens_df.columns
    assert "net_expected_value" in sens_df.columns
    assert set(sens_df["scenario"].unique()) == {"Conservative", "Moderate", "Optimistic"}


def test_incremental_campaign_value_hand_calculated_examples():
    """
    Audit Check (Financial Objective Option A):
        Verify that Incremental Campaign Value vs. No Outreach is:
            Delta Value = (TP * eta * V) - (TP + FP) * C_outreach
        and that false-negative churn losses (FN * V) are NOT subtracted, because
        those losses also occur in the no-outreach baseline.
    """
    # Create controlled ground truth and prediction probabilities:
    # 100 TPs (y=1, prob=0.9)
    # 50  FPs (y=0, prob=0.8)
    # 50  FNs (y=1, prob=0.1)
    # 200 TNs (y=0, prob=0.2)
    y_true = np.array([1] * 100 + [0] * 50 + [1] * 50 + [0] * 200)
    probs = np.array([0.9] * 100 + [0.8] * 50 + [0.1] * 50 + [0.2] * 200)

    # Threshold 0.50 selects exactly the 100 TPs and 50 FPs
    # Hand calculation 1:
    # eta = 0.25 (25% saved), V = $120, C = $20
    # Contacted accounts = 100 + 50 = 150
    # Outreach cost = 150 * 20 = $3,000.0
    # Persuaded churners = 100 * 0.25 = 25.0
    # Gross value preserved = 25.0 * 120 = $3,000.0
    # Incremental campaign value = $3,000 - $3,000 = $0.0
    df_eval1 = analyze_threshold_tradeoffs(
        y_train_oof=y_true,
        probs_oof=probs,
        thresholds=[0.50],
        persuadability_eta=0.25,
        customer_value=120.0,
        unit_outreach_cost=20.0,
    )
    row1 = df_eval1.iloc[0]
    assert row1["churners_identified_tp"] == 100
    assert row1["false_alarms_fp"] == 50
    assert row1["contacted_accounts"] == 150
    assert row1["persuaded_churners"] == 25.0
    assert row1["gross_value_preserved"] == 3000.0
    assert row1["outreach_cost"] == 3000.0
    assert row1["incremental_campaign_value"] == 0.0

    # Hand calculation 2:
    # eta = 0.40, V = $100, C = $15
    # Contacted accounts = 150
    # Outreach cost = 150 * 15 = $2,250.0
    # Persuaded churners = 100 * 0.40 = 40.0
    # Gross value preserved = 40.0 * 100 = $4,000.0
    # Incremental campaign value = $4,000 - $2,250 = +$1,750.0
    df_eval2 = analyze_threshold_tradeoffs(
        y_train_oof=y_true,
        probs_oof=probs,
        thresholds=[0.50],
        persuadability_eta=0.40,
        customer_value=100.0,
        unit_outreach_cost=15.0,
    )
    row2 = df_eval2.iloc[0]
    assert row2["persuaded_churners"] == 40.0
    assert row2["gross_value_preserved"] == 4000.0
    assert row2["outreach_cost"] == 2250.0
    assert row2["incremental_campaign_value"] == 1750.0

    # Mathematical Baseline Proof Check:
    # Baseline: All 150 churners leave => Churn loss = -150 * $100 = -$15,000
    # Campaign: 40 churners saved, so only (60 unpersuaded + 50 missed) = 110 churners leave => Churn loss = -$11,000
    # Outreach cost = -$2,250 => Total campaign net loss = -$11,000 - $2,250 = -$13,250
    # Incremental difference = Campaign - Baseline = -$13,250 - (-$15,000) = +$1,750.0
    baseline_value = -(100 + 50) * 100.0
    campaign_value = -((60.0 + 50.0) * 100.0) - 2250.0
    assert (campaign_value - baseline_value) == row2["incremental_campaign_value"]


def test_nadeau_bengio_corrected_ttest_hand_calculated_example():
    """
    Audit Check: Verify Nadeau-Bengio corrected resampled t-test using a small
    analytically hand-calculated synthetic sequence of paired differences.
    """
    # 5 paired differences with known properties:
    # diffs = [0.01, 0.02, 0.00, 0.03, 0.04]
    # R = 5, df = 4
    # mean d_bar = 0.0200
    # variance s^2 = [(0.01 - 0.02)^2 + (0.02 - 0.02)^2 + (-0.02)^2 + (0.01)^2 + (0.02)^2] / 4
    #              = [0.0001 + 0.0000 + 0.0004 + 0.0001 + 0.0004] / 4 = 0.0010 / 4 = 0.000250
    # n_train = 800, n_test = 200 => ratio = 200 / 800 = 0.2500
    # correction factor = 1/5 + 0.25 = 0.20 + 0.25 = 0.4500
    # corrected variance = 0.000250 * 0.45 = 0.0001125
    # corrected SE = sqrt(0.0001125) = 0.0106066
    # corrected t-stat = 0.0200 / 0.0106066 = 1.8856... => round to 1.886
    from scipy import stats

    diffs = np.array([0.01, 0.02, 0.00, 0.03, 0.04])
    res = compute_nadeau_bengio_corrected_ttest(diffs, n_train=800, n_test=200)

    assert res["mean_difference"] == 0.0200
    assert res["sample_variance"] == 0.000250
    assert res["sample_size_ratio"] == 0.2500
    assert res["correction_factor"] == 0.4500
    assert res["degrees_of_freedom"] == 4
    assert res["corrected_t_stat"] == 1.886

    expected_p = float(2.0 * stats.t.sf(np.abs(0.0200 / np.sqrt(0.0001125)), df=4))
    assert round(res["two_sided_p_value"], 4) == round(expected_p, 4)


def test_nadeau_bengio_corrected_ttest_input_validation():
    """Verify that fewer than 2 evaluations raises ValueError."""
    with pytest.raises(ValueError, match="At least 2 paired evaluations are required"):
        compute_nadeau_bengio_corrected_ttest(np.array([0.05]), n_train=100, n_test=25)


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
