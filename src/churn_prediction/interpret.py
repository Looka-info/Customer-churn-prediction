"""Model interpretation: coefficients, tree feature importances, permutation importance, and error analysis."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from churn_prediction.config import AppConfig, get_config

logger = logging.getLogger(__name__)


def get_feature_names_from_pipeline(pipeline: Pipeline) -> List[str]:
    """Extract transformed feature names from ColumnTransformer inside pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def extract_logistic_regression_weights(
    pipeline: Pipeline, top_n: int = 15
) -> pd.DataFrame:
    """
    Extract coefficients and odds ratios from fitted Logistic Regression pipeline.
    Odds ratio > 1 implies increased churn odds; odds ratio < 1 implies decreased churn odds.
    """
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names_from_pipeline(pipeline)
    coefs = classifier.coef_[0]

    df_coef = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
            "abs_impact": np.abs(coefs),
        }
    ).sort_values(by="abs_impact", ascending=False).reset_index(drop=True)

    return df_coef.head(top_n)


def extract_tree_feature_importances(
    pipeline: Pipeline, top_n: int = 15
) -> pd.DataFrame:
    """Extract impurity-based (MDI) feature importances from a tree-based pipeline."""
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names_from_pipeline(pipeline)

    if not hasattr(classifier, "feature_importances_"):
        raise AttributeError("Classifier does not possess feature_importances_ attribute.")

    importances = classifier.feature_importances_
    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_imp.head(top_n)


def compute_permutation_importance(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Compute model-agnostic permutation feature importance on held-out evaluation data.
    Provides an unbiased alternative to Mean Decrease Impurity (MDI).
    """
    perm_result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
        n_jobs=-1,
    )

    feature_names = list(X_val.columns)
    df_perm = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm_result.importances_mean,
            "importance_std": perm_result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    return df_perm.head(top_n)


def analyze_prediction_errors(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_probs: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, pd.DataFrame]:
    """
    Isolate False Positives (predicted churn, but stayed) and False Negatives (predicted retained, but churned).
    Enables deep inspection of model failure modes.
    """
    y_preds = (y_probs >= threshold).astype(int)
    eval_df = X_test.copy()
    eval_df["actual_churn"] = y_test.values
    eval_df["churn_probability"] = np.round(y_probs, 4)
    eval_df["predicted_churn"] = y_preds

    # False Positives: actual=0, predicted=1
    false_positives = eval_df[(eval_df["actual_churn"] == 0) & (eval_df["predicted_churn"] == 1)]

    # False Negatives: actual=1, predicted=0
    false_negatives = eval_df[(eval_df["actual_churn"] == 1) & (eval_df["predicted_churn"] == 0)]

    return {
        "false_positives": false_positives.sort_values(by="churn_probability", ascending=False),
        "false_negatives": false_negatives.sort_values(by="churn_probability", ascending=True),
    }


def plot_feature_importances(
    df_imp: pd.DataFrame,
    output_path: Path,
    title: str = "Top Predictive Features (Gradient Boosting)",
) -> None:
    """Plot clean horizontal bar chart of top predictive features."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))

    imp_col = "importance" if "importance" in df_imp.columns else "importance_mean"
    sorted_df = df_imp.sort_values(by=imp_col, ascending=True)

    plt.barh(
        sorted_df["feature"],
        sorted_df[imp_col],
        color="#2ca02c",
        edgecolor="#1c6b1c",
        alpha=0.85,
    )
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.title(title, fontsize=12, pad=12, weight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved feature importance chart to %s", output_path)
