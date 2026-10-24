"""Model training, stratified cross-validation, and model selection module."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from churn_prediction.config import AppConfig, get_config
from churn_prediction.preprocessing import build_preprocessor, get_feature_lists

logger = logging.getLogger(__name__)


def create_candidate_models(config: Optional[AppConfig] = None) -> Dict[str, Any]:
    """Instantiate the three required candidate classifiers with configured hyperparameters."""
    cfg = config or get_config()

    lr_params = cfg.logistic_regression_params or {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
    }
    rf_params = cfg.random_forest_params or {
        "n_estimators": 150,
        "max_depth": 8,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    }
    gb_params = cfg.gradient_boosting_params or {
        "n_estimators": 100,
        "learning_rate": 0.08,
        "max_depth": 4,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "subsample": 0.85,
    }

    candidates = {
        "Logistic Regression": LogisticRegression(
            random_state=cfg.random_seed, **lr_params
        ),
        "Random Forest": RandomForestClassifier(
            random_state=cfg.random_seed, **rf_params
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=cfg.random_seed, **gb_params
        ),
    }
    return candidates


def build_model_pipeline(
    estimator: Any,
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True,
) -> Pipeline:
    """Combine the ColumnTransformer preprocessor and the classifier into a leak-free scikit-learn Pipeline."""
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=scale_numeric,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def cross_validate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Optional[AppConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Perform stratified K-fold cross-validation on the training set for all candidates.
    Never touches the held-out test set.
    Collects fold-level metrics and out-of-fold validation probabilities.
    """
    cfg = config or get_config()
    numeric_features, categorical_features = get_feature_lists(X_train, cfg)
    candidates = create_candidate_models(cfg)

    skf = StratifiedKFold(
        n_splits=cfg.cv_folds,
        shuffle=True,
        random_state=cfg.random_seed,
    )

    cv_summary_rows = []
    oof_predictions: Dict[str, Dict[str, Any]] = {}

    for name, estimator in candidates.items():
        logger.info("Evaluating %s with %d-fold cross-validation...", name, cfg.cv_folds)

        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_pr_aucs = []
        fold_roc_aucs = []

        oof_probs = np.zeros(len(y_train))
        y_train_array = y_train.values

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Logistic regression requires scaling; tree models also function correctly with scaled features
            pipeline = build_model_pipeline(
                estimator=estimator,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                scale_numeric=True,
            )

            # Fit ONLY on the training fold
            pipeline.fit(X_fold_train, y_fold_train)

            # Predict on validation fold
            val_probs = pipeline.predict_proba(X_fold_val)[:, 1]
            val_preds = (val_probs >= cfg.default_threshold).astype(int)

            oof_probs[val_idx] = val_probs

            fold_precisions.append(precision_score(y_fold_val, val_preds, zero_division=0))
            fold_recalls.append(recall_score(y_fold_val, val_preds, zero_division=0))
            fold_f1s.append(f1_score(y_fold_val, val_preds, zero_division=0))
            fold_pr_aucs.append(average_precision_score(y_fold_val, val_probs))
            fold_roc_aucs.append(roc_auc_score(y_fold_val, val_probs))

        mean_precision = float(np.mean(fold_precisions))
        std_precision = float(np.std(fold_precisions))
        mean_recall = float(np.mean(fold_recalls))
        std_recall = float(np.std(fold_recalls))
        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        mean_pr_auc = float(np.mean(fold_pr_aucs))
        std_pr_auc = float(np.std(fold_pr_aucs))
        mean_roc_auc = float(np.mean(fold_roc_aucs))
        std_roc_auc = float(np.std(fold_roc_aucs))

        cv_summary_rows.append(
            {
                "Model": name,
                "CV_ROC_AUC_Mean": mean_roc_auc,
                "CV_ROC_AUC_Std": std_roc_auc,
                "CV_PR_AUC_Mean": mean_pr_auc,
                "CV_PR_AUC_Std": std_pr_auc,
                "CV_F1_Mean": mean_f1,
                "CV_F1_Std": std_f1,
                "CV_Precision_Mean": mean_precision,
                "CV_Precision_Std": std_precision,
                "CV_Recall_Mean": mean_recall,
                "CV_Recall_Std": std_recall,
            }
        )

        oof_predictions[name] = {
            "oof_probabilities": oof_probs,
            "y_true": y_train_array,
        }

    cv_df = pd.DataFrame(cv_summary_rows).sort_values(by="CV_PR_AUC_Mean", ascending=False).reset_index(drop=True)
    return cv_df, oof_predictions


def select_best_model(cv_df: pd.DataFrame) -> str:
    """
    Select best model based on validation metrics.
    We prioritize PR-AUC (Average Precision) because customer churn is an imbalanced
    retention problem where identifying positive churn risk reliably is key.
    """
    best_model_name = str(cv_df.iloc[0]["Model"])
    logger.info("Selected best model: %s (CV PR-AUC = %.4f)", best_model_name, cv_df.iloc[0]["CV_PR_AUC_Mean"])
    return best_model_name


def train_and_save_final_pipeline(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_path: Path,
    config: Optional[AppConfig] = None,
) -> Pipeline:
    """
    Fit the selected model pipeline on the full training dataset and persist via joblib.
    """
    cfg = config or get_config()
    numeric_features, categorical_features = get_feature_lists(X_train, cfg)
    candidates = create_candidate_models(cfg)

    if model_name not in candidates:
        raise ValueError(f"Unknown model name '{model_name}'. Available: {list(candidates.keys())}")

    estimator = candidates[model_name]
    pipeline = build_model_pipeline(
        estimator=estimator,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=True,
    )

    logger.info("Fitting final pipeline (%s) on full training set (N=%d)...", model_name, len(X_train))
    pipeline.fit(X_train, y_train)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info("Saved final fitted pipeline to %s", output_path)

    return pipeline
