"""
Customer Churn Prediction — Training & Evaluation Pipeline
Loads the IBM Telco dataset, evaluates candidate classification models,
runs statistical comparisons, evaluates on the held-out test set,
and serializes the final model pipeline.
"""

import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from churn_prediction.config import get_config
from churn_prediction.data_loader import (
    download_dataset_if_missing,
    load_raw_data,
    prepare_features_and_target,
    validate_raw_data,
)
from churn_prediction.database import (
    load_data_to_sqlite,
    verify_sql_against_pandas,
)
from churn_prediction.eda import generate_all_eda_figures
from churn_prediction.evaluate import (
    analyze_threshold_tradeoffs,
    evaluate_on_test_set,
    generate_model_comparison_artifact,
    plot_confusion_matrix,
    plot_pr_and_roc_curves,
    run_intervention_sensitivity_analysis,
)
from churn_prediction.interpret import (
    analyze_prediction_errors,
    compute_permutation_importance,
    extract_logistic_regression_weights,
    extract_tree_feature_importances,
    plot_feature_importances,
)
from churn_prediction.predict import ChurnPredictor
from churn_prediction.preprocessing import create_train_test_split
from churn_prediction.train import (
    build_model_pipeline,
    compute_cv_statistical_significance,
    create_candidate_models,
    cross_validate_candidates,
    run_feature_ablation_study,
    run_repeated_cv_comparison,
    select_best_model,
    train_and_save_final_pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def main():
    config = get_config()
    logger.info("Starting training pipeline...")

    # 1. Ingestion & Validation
    download_dataset_if_missing(config.raw_data_path)
    raw_df = load_raw_data(config.raw_data_path)
    validate_raw_data(raw_df, config)

    # 2. SQLite Database Population & Verification
    load_data_to_sqlite(raw_df, config.database_path, config.sql_schema_path)
    verify_sql_against_pandas(config.database_path, raw_df)

    # 3. Generate Exploratory Figures
    generate_all_eda_figures(raw_df, config.figures_dir)

    # 4. Train / Test Split
    X, y = prepare_features_and_target(raw_df, config)
    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=config.stratify,
    )
    logger.info("Split dataset: %d train, %d test", len(X_train), len(X_test))

    # 5. Cross-Validation (Candidate Models)
    logger.info("Running 5-fold cross-validation across candidate models...")
    cv_summary, oof_preds = cross_validate_candidates(X_train, y_train, config)
    print("\n--- Cross-Validation Results (Training Set) ---")
    print(cv_summary.to_string(index=False))

    # 6. Statistical Comparison (Nadeau-Bengio Corrected Resampled t-Test)
    logger.info("Running repeated cross-validation statistical comparison...")
    repeated_cv_df = run_repeated_cv_comparison(
        X_train, y_train, n_splits=5, n_repeats=3, random_state=config.random_seed, config=config
    )
    print("\n--- Repeated-CV Statistical Comparison (Nadeau-Bengio Correction) ---")
    print(repeated_cv_df.to_string(index=False))

    # 7. Model Selection
    best_model_name, selection_rationale = select_best_model(cv_summary, oof_predictions=oof_preds)
    logger.info("Selected model: %s (%s)", best_model_name, selection_rationale["reason"])

    # 8. Feature Collinearity Ablation (With vs Without TotalCharges)
    ablation_df = run_feature_ablation_study(X_train, y_train, feature_to_drop="TotalCharges", config=config)
    print("\n--- Feature Ablation (With vs Without TotalCharges) ---")
    print(ablation_df.to_string(index=False))

    # 9. Decision Threshold Analysis (Out-of-Fold Predictions)
    best_oof = oof_preds[best_model_name]
    threshold_analysis = analyze_threshold_tradeoffs(best_oof["y_true"], best_oof["oof_probabilities"])
    sensitivity_df = run_intervention_sensitivity_analysis(best_oof["y_true"], best_oof["oof_probabilities"])
    print("\n--- Threshold Trade-offs & Expected Campaign Value ---")
    print(threshold_analysis.to_string(index=False))
    print("\n--- Campaign Sensitivity Analysis ---")
    print(sensitivity_df.to_string(index=False))

    # 10. Held-Out Test Evaluation
    logger.info("Evaluating candidate models on held-out test set...")
    candidates = create_candidate_models(config)
    test_evaluations = {}

    for name, estimator in candidates.items():
        pipe = build_model_pipeline(
            estimator=estimator,
            numeric_features=config.numeric_features,
            categorical_features=config.categorical_features,
            scale_numeric=True,
        )
        pipe.fit(X_train, y_train)
        test_res = evaluate_on_test_set(pipe, X_test, y_test, threshold=config.default_threshold)
        test_evaluations[name] = test_res

    comparison_df = generate_model_comparison_artifact(
        cv_summary, test_evaluations, config.metrics_comparison_path
    )
    print("\n--- Final Model Comparison (CV vs Held-Out Test) ---")
    print(comparison_df.to_string(index=False))

    # 11. Train & Save Final Selected Pipeline
    logger.info("Training final pipeline (%s) and saving artifact...", best_model_name)
    best_pipeline = train_and_save_final_pipeline(
        best_model_name,
        X_train,
        y_train,
        config.best_model_path,
        config,
    )

    # 12. Diagnostic Figures
    best_test_eval = test_evaluations[best_model_name]
    plot_confusion_matrix(
        best_test_eval["confusion_matrix"],
        config.figures_dir / "confusion_matrix.png",
        title=f"Confusion Matrix ({best_model_name} on Test Set)",
    )
    plot_pr_and_roc_curves(
        y_test.values,
        best_test_eval["probabilities"],
        config.figures_dir / "pr_roc_curves.png",
        model_name=best_model_name,
    )

    if hasattr(best_pipeline.named_steps["classifier"], "feature_importances_"):
        imp_df = extract_tree_feature_importances(best_pipeline, top_n=12)
        plot_feature_importances(
            imp_df,
            config.figures_dir / "feature_importance.png",
            title=f"Top 12 Predictive Features ({best_model_name})",
        )
    else:
        coef_df = extract_logistic_regression_weights(best_pipeline, top_n=12)
        coef_df["importance"] = coef_df["abs_impact"]
        plot_feature_importances(
            coef_df,
            config.figures_dir / "feature_importance.png",
            title="Top 12 Feature Coefficients (Logistic Regression)",
        )

    # Permutation importance & error breakdown
    compute_permutation_importance(best_pipeline, X_test, y_test, n_repeats=5, random_state=config.random_seed)
    analyze_prediction_errors(X_test, y_test, best_test_eval["probabilities"], threshold=config.default_threshold)

    # 13. Verify Inference
    predictor = ChurnPredictor(config.best_model_path)
    sample_record = {
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
    pred_res = predictor.predict(sample_record)
    logger.info(
        "Inference test: probability=%.2f%%, class=%s",
        pred_res.iloc[0]["churn_probability"] * 100,
        pred_res.iloc[0]["prediction_label"],
    )

    logger.info("Pipeline completed successfully. Artifacts saved in models/ and reports/.")


if __name__ == "__main__":
    main()
