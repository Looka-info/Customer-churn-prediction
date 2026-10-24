"""Evaluation metrics, threshold analysis, diagnostic curves, and report generation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from churn_prediction.config import AppConfig, get_config

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics for the positive class (Churn=1).

    NOTE ON PR-AUC METHODOLOGY:
        The 'pr_auc' metric uses `sklearn.metrics.average_precision_score`, which calculates
        the weighted mean of precisions at each threshold (AP = sum (R_n - R_{n-1}) * P_n).
        This is preferred over trapezoidal interpolation (`auc(recall, precision)`),
        which can yield optimistic overestimates on non-monotonic precision-recall curves.
    """
    y_preds = (y_probs >= threshold).astype(int)

    precision = float(precision_score(y_true, y_preds, zero_division=0))
    recall = float(recall_score(y_true, y_preds, zero_division=0))
    f1 = float(f1_score(y_true, y_preds, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_probs))
    roc_auc = float(roc_auc_score(y_true, y_probs))

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def analyze_threshold_tradeoffs(
    y_train_oof: np.ndarray,
    probs_oof: np.ndarray,
    thresholds: Optional[List[float]] = None,
    persuadability_eta: float = 0.30,
    customer_value: float = 120.0,
    unit_outreach_cost: float = 20.0,
    cost_fp: Optional[float] = None,
    cost_fn: Optional[float] = None,
    benefit_tp: Optional[float] = None,
) -> pd.DataFrame:
    """
    Evaluate precision-recall-f1 trade-offs and incremental campaign value strictly on
    out-of-fold training predictions. Never evaluates on the final test set to prevent leakage.

    FINANCIAL UTILITY OBJECTIVE (Option A: Incremental Campaign Value vs. No Outreach):
        The financial objective is defined strictly as the incremental value of an outreach
        intervention campaign compared to a baseline of no outreach:
            Delta Value = Gross Value Preserved - Total Outreach Cost
                        = (TP * eta * V) - (TP + FP) * C_outreach

        Where:
            - TP (True Positives): Actual churners correctly identified and flagged for outreach.
            - FP (False Positives): Non-churners mistakenly flagged (wasted outreach expense).
            - Contacted Accounts = TP + FP.
            - eta (Persuadability): Fraction of contacted churners who accept the retention offer.
            - V (Customer Value): Customer lifetime gross margin preserved if retained.
            - C_outreach (Unit Outreach Cost): Cost of outreach/discount per contacted account.

        Baseline Comparison & False-Negative Invariant:
            In the no-outreach baseline, all actual churners (TP + FN) depart, incurring loss -(TP + FN) * V.
            In the campaign, unpersuaded churners TP * (1 - eta) and missed churners FN still churn,
            incurring churn loss -[TP * (1 - eta) + FN] * V.
            The difference is:
                Value_campaign - Value_baseline
                = [(TP * eta * V) - [TP * (1 - eta) + FN] * V - (TP + FP) * C_outreach] - [-(TP + FN) * V]
                = (TP * eta * V) - (TP + FP) * C_outreach
            Therefore, false-negative churn losses (FN * V) cancel out identically and are NOT
            separately subtracted from incremental campaign value, avoiding baseline confounding.

    SCIENTIFIC QUALIFICATION:
        This utility calculation is a hypothetical expected-value scenario based on assumed unit parameters.
        It does NOT represent demonstrated financial return on investment (ROI) or confirmed saves,
        because the observational dataset contains no randomized uplift trial or intervention logs.
    """
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]

    # Support legacy parameter overrides if explicitly supplied
    c = unit_outreach_cost if cost_fp is None else cost_fp
    v = customer_value
    eta = persuadability_eta

    rows = []
    for th in thresholds:
        m = compute_metrics(y_train_oof, probs_oof, threshold=th)
        cm = confusion_matrix(y_train_oof, (probs_oof >= th).astype(int))
        tn, fp, fn, tp = cm.ravel()

        contacted = tp + fp
        persuaded = tp * eta
        gross_preserved = persuaded * v
        outreach_cost = contacted * c
        incremental_value = gross_preserved - outreach_cost

        rows.append(
            {
                "threshold": th,
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "contacted_accounts": int(contacted),
                "churners_identified_tp": int(tp),
                "false_alarms_fp": int(fp),
                "retained_correct_tn": int(tn),
                "missed_churners_fn": int(fn),
                "persuaded_churners": round(float(persuaded), 1),
                "gross_value_preserved": round(float(gross_preserved), 2),
                "outreach_cost": round(float(outreach_cost), 2),
                "incremental_campaign_value": round(float(incremental_value), 2),
                # Backwards-compatibility alias for existing references
                "hypothetical_expected_value": round(float(incremental_value), 2),
            }
        )

    return pd.DataFrame(rows)


def run_intervention_sensitivity_analysis(
    y_train_oof: np.ndarray,
    probs_oof: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Sensitivity analysis over plausible retention intervention scenarios strictly on training OOF predictions.
    Evaluates Option A: Incremental Campaign Value vs. No Outreach across Conservative, Moderate,
    and Optimistic parameters.

    Option A Formula:
        Contacted Accounts = TP + FP
        Persuaded Churners = TP * eta
        Gross Value Preserved = Persuaded Churners * customer_value = (TP * eta) * customer_value
        Outreach Cost = Contacted Accounts * unit_outreach_cost = (TP + FP) * unit_outreach_cost
        Incremental Campaign Value = Gross Value Preserved - Outreach Cost
                                   = (TP * eta * customer_value) - (TP + FP) * unit_outreach_cost

    Baseline Note:
        False-negative churn losses (FN * customer_value) occur in both the no-outreach baseline
        and the campaign, and therefore cancel out when measuring incremental campaign value over baseline.
    """
    if thresholds is None:
        thresholds = [0.25, 0.35, 0.50]

    scenarios = [
        {"name": "Conservative", "eta": 0.15, "unit_cost": 25.0, "customer_value": 100.0},
        {"name": "Moderate", "eta": 0.30, "unit_cost": 20.0, "customer_value": 120.0},
        {"name": "Optimistic", "eta": 0.50, "unit_cost": 15.0, "customer_value": 150.0},
    ]

    rows = []
    for sc in scenarios:
        eta = sc["eta"]
        c = sc["unit_cost"]
        v = sc["customer_value"]
        for th in thresholds:
            cm = confusion_matrix(y_train_oof, (probs_oof >= th).astype(int))
            tn, fp, fn, tp = cm.ravel()

            contacted = tp + fp
            persuaded = tp * eta
            gross_preserved = persuaded * v
            outreach_cost = contacted * c
            incremental_value = gross_preserved - outreach_cost

            rows.append(
                {
                    "scenario": sc["name"],
                    "success_rate_eta": f"{int(eta * 100)}%",
                    "unit_outreach_cost": f"${c:.0f}",
                    "customer_value": f"${v:.0f}",
                    "threshold": th,
                    "contacted_accounts": int(contacted),
                    "churners_identified_tp": int(tp),
                    "false_alarms_fp": int(fp),
                    "persuaded_churners": round(float(persuaded), 1),
                    "gross_value_preserved": round(float(gross_preserved), 2),
                    "outreach_cost": round(float(outreach_cost), 2),
                    "incremental_campaign_value": round(float(incremental_value), 2),
                    # Backwards compatibility alias
                    "net_expected_value": round(float(incremental_value), 2),
                }
            )

    return pd.DataFrame(rows)


def evaluate_on_test_set(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Final evaluation on held-out test set once.
    Calculates metrics, confusion matrix, and full classification report.
    """
    y_test_array = y_test.values
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    metrics = compute_metrics(y_test_array, test_probs, threshold=threshold)
    cm = confusion_matrix(y_test_array, test_preds)
    tn, fp, fn, tp = cm.ravel()

    report_str = classification_report(
        y_test_array,
        test_preds,
        target_names=["Retained (0)", "Churned (1)"],
        digits=4,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "counts": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "classification_report": report_str,
        "probabilities": test_probs,
        "predictions": test_preds,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix - Final Test Set",
) -> None:
    """Generate and save publication-quality confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))

    labels = ["Retained (0)", "Churned (1)"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(title, fontsize=13, pad=12, weight="bold")
    plt.xlabel("Predicted Outcome", fontsize=11)
    plt.ylabel("Actual Outcome", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix to %s", output_path)


def plot_pr_and_roc_curves(
    y_test: np.ndarray,
    y_probs: np.ndarray,
    output_path: Path,
    model_name: str = "Gradient Boosting",
) -> None:
    """Plot dual Precision-Recall and ROC diagnostic curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    baseline_pr = float(np.mean(y_test))

    ax1.plot(rec, prec, color="#1f77b4", lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
    ax1.axhline(y=baseline_pr, color="gray", linestyle="--", label=f"No Skill Baseline ({baseline_pr:.3f})")
    ax1.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax1.set_ylabel("Precision", fontsize=11)
    ax1.set_title("Precision-Recall Curve", fontsize=12, weight="bold")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)

    ax2.plot(fpr, tpr, color="#ff7f0e", lw=2, label=f"{model_name} (ROC-AUC = {roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance (0.500)")
    ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax2.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax2.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, weight="bold")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved PR and ROC curves to %s", output_path)


def generate_model_comparison_artifact(
    cv_summary_df: pd.DataFrame,
    test_results_dict: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> pd.DataFrame:
    """
    Build structured comparison table that clearly distinguishes validation performance
    from final held-out test performance for all candidate models.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, cv_row in cv_summary_df.iterrows():
        model_name = cv_row["Model"]
        test_res = test_results_dict.get(model_name, {}).get("metrics", {})

        row = {
            "Model": model_name,
            # Cross-validation validation scores (Stratified 5-Fold)
            "CV_PR_AUC_Mean": round(cv_row["CV_PR_AUC_Mean"], 4),
            "CV_PR_AUC_Std": round(cv_row["CV_PR_AUC_Std"], 4),
            "CV_ROC_AUC_Mean": round(cv_row["CV_ROC_AUC_Mean"], 4),
            "CV_F1_Mean": round(cv_row["CV_F1_Mean"], 4),
            "CV_Precision_Mean": round(cv_row["CV_Precision_Mean"], 4),
            "CV_Recall_Mean": round(cv_row["CV_Recall_Mean"], 4),
            # Final held-out test set evaluation
            "Test_PR_AUC": round(test_res.get("pr_auc", 0.0), 4),
            "Test_ROC_AUC": round(test_res.get("roc_auc", 0.0), 4),
            "Test_F1": round(test_res.get("f1", 0.0), 4),
            "Test_Precision": round(test_res.get("precision", 0.0), 4),
            "Test_Recall": round(test_res.get("recall", 0.0), 4),
        }
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(output_path, index=False)
    logger.info("Saved model comparison table to %s", output_path)
    return comp_df
