"""
End-to-End Execution Pipeline for Customer Churn Prediction.

Executes all phases:
1. Data loading and hygiene checks
2. SQLite database generation and SQL verification
3. Exploratory Data Analysis figures generation
4. Leakage-free train/test splitting
5. Stratified 5-Fold cross-validation of 3 candidate models
6. Model selection and threshold analysis
7. Final evaluation on held-out test set
8. Model interpretation (coefficients, tree importances, permutation importance)
9. Serialization of fitted pipeline
10. Inference verification on sample customer record
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from churn_prediction.config import get_config
from churn_prediction.data_loader import (
    clean_raw_data,
    download_dataset_if_missing,
    load_raw_data,
    prepare_features_and_target,
    validate_raw_data,
)
from churn_prediction.database import (
    execute_analysis_queries,
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
    create_candidate_models,
    cross_validate_candidates,
    select_best_model,
    train_and_save_final_pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def run_pipeline() -> None:
    logger.info("================================================================================")
    logger.info("Starting Customer Churn Prediction Pipeline")
    logger.info("================================================================================")

    config = get_config()

    # Step 1: Ensure raw dataset is present and load it
    download_dataset_if_missing(config.raw_data_path)
    logger.info("Loading raw dataset from: %s", config.raw_data_path)
    raw_df = load_raw_data(config.raw_data_path)

    # Step 2: Validate raw data
    validation_report = validate_raw_data(raw_df, config)
    logger.info(
        "Data validation passed: %d rows, %d columns, %d duplicate IDs, %d blank TotalCharges",
        validation_report["num_rows"],
        validation_report["num_columns"],
        validation_report["duplicate_customer_ids"],
        validation_report["blank_total_charges_count"],
    )

    # Step 3: SQLite Database Ingestion & Verification
    logger.info("--- Step 3: SQLite Database & SQL Analysis ---")
    rows_inserted = load_data_to_sqlite(
        raw_df, config.database_path, config.sql_schema_path
    )
    logger.info("Inserted %d rows into SQLite database at %s", rows_inserted, config.database_path)

    # Verify SQL against Pandas
    sql_verification = verify_sql_against_pandas(config.database_path, raw_df)
    logger.info(
        "SQL vs Pandas verification: %s. Overall churn rate: %.2f%% (%d / %d)",
        sql_verification["status"],
        sql_verification["overall_churn_rate_percent"],
        sql_verification["overall_churned"],
        sql_verification["overall_total"],
    )

    # Step 4: Exploratory Data Analysis figures
    logger.info("--- Step 4: Generating Exploratory Visualizations ---")
    generate_all_eda_figures(raw_df, config.figures_dir)

    # Step 5: Data Splitting (Leakage-Safe)
    logger.info("--- Step 5: Leakage-Safe Stratified Train/Test Split ---")
    X, y = prepare_features_and_target(raw_df, config)
    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=config.stratify,
    )
    logger.info(
        "Dataset split: %d training samples (%.1f%% churn), %d test samples (%.1f%% churn)",
        len(X_train),
        y_train.mean() * 100,
        len(X_test),
        y_test.mean() * 100,
    )

    # Step 6: 5-Fold Stratified Cross-Validation of 3 Candidates
    logger.info("--- Step 6: Stratified Cross-Validation on Training Data ---")
    cv_summary, oof_preds = cross_validate_candidates(X_train, y_train, config)
    print("\n" + "=" * 75)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION RESULTS (TRAINING DATA ONLY)")
    print("=" * 75)
    print(cv_summary.to_string(index=False))
    print("=" * 75 + "\n")

    # Statistical Significance Audit (Paired t-test across CV folds vs Baseline)
    from churn_prediction.train import compute_cv_statistical_significance
    sig_pr = compute_cv_statistical_significance(oof_preds, baseline_model="Logistic Regression", metric_key="fold_pr_aucs")
    sig_roc = compute_cv_statistical_significance(oof_preds, baseline_model="Logistic Regression", metric_key="fold_roc_aucs")
    print("STATISTICAL SIGNIFICANCE vs LOGISTIC REGRESSION BASELINE (PR-AUC):")
    print(sig_pr.to_string(index=False))
    print("\nSTATISTICAL SIGNIFICANCE vs LOGISTIC REGRESSION BASELINE (ROC-AUC):")
    print(sig_roc.to_string(index=False))
    print()

    # Step 7: Defensible Model Selection
    logger.info("--- Step 7: Defensible Model Selection & Uncertainty Audit ---")
    best_model_name, selection_audit = select_best_model(cv_summary, oof_predictions=oof_preds)
    logger.info("Selected Model: %s", best_model_name)
    logger.info("Selection Rationale: %s", selection_audit["reason"])

    # Threshold analysis on training out-of-fold predictions
    best_oof = oof_preds[best_model_name]
    threshold_analysis = analyze_threshold_tradeoffs(
        best_oof["y_true"], best_oof["oof_probabilities"]
    )
    print("\n" + "=" * 80)
    print("TRAINING-SIDE OUT-OF-FOLD THRESHOLD & BUSINESS UTILITY TRADE-OFFS")
    print("=" * 80)
    print(threshold_analysis.to_string(index=False))
    print("=" * 80)

    best_thresh_idx = threshold_analysis["net_business_value"].idxmax()
    optimal_thresh = float(threshold_analysis.loc[best_thresh_idx, "threshold"])
    logger.info(
        "Training-side optimal economic threshold: %.2f (Net Value: $%,.2f, Recall: %.1f%%, Precision: %.1f%%)",
        optimal_thresh,
        threshold_analysis.loc[best_thresh_idx, "net_business_value"],
        threshold_analysis.loc[best_thresh_idx, "recall"] * 100,
        threshold_analysis.loc[best_thresh_idx, "precision"] * 100,
    )

    # Step 8: Train all candidate models on full training set and evaluate on held-out test set
    logger.info("--- Step 8: Final Evaluation on Held-Out Test Set ---")
    candidates = create_candidate_models(config)
    test_evaluations: dict = {}

    for name, estimator in candidates.items():
        pipe = build_model_pipeline(
            estimator=estimator,
            numeric_features=config.numeric_features,
            categorical_features=config.categorical_features,
            scale_numeric=True,
        )
        pipe.fit(X_train, y_train)
        test_res = evaluate_on_test_set(
            pipe, X_test, y_test, threshold=config.default_threshold
        )
        test_evaluations[name] = test_res

    # Display comparison table
    comparison_df = generate_model_comparison_artifact(
        cv_summary, test_evaluations, config.metrics_comparison_path
    )
    print("\n" + "=" * 85)
    print("FINAL MODEL COMPARISON (CROSS-VALIDATION VS HELD-OUT TEST PERFORMANCE)")
    print("=" * 85)
    print(comparison_df.to_string(index=False))
    print("=" * 85 + "\n")

    # Step 9: Save Best Model Pipeline
    logger.info("--- Step 9: Persisting Selected Pipeline ---")
    best_pipeline = train_and_save_final_pipeline(
        best_model_name,
        X_train,
        y_train,
        config.best_model_path,
        config,
    )

    # Step 10: Final Plots & Interpretation for Selected Model
    logger.info("--- Step 10: Generating Final Diagnostic and Interpretation Plots ---")
    best_test_eval = test_evaluations[best_model_name]

    # Confusion matrix
    plot_confusion_matrix(
        best_test_eval["confusion_matrix"],
        config.figures_dir / "confusion_matrix.png",
        title=f"Confusion Matrix ({best_model_name} on Test Set)",
    )

    # PR and ROC curves
    plot_pr_and_roc_curves(
        y_test.values,
        best_test_eval["probabilities"],
        config.figures_dir / "pr_roc_curves.png",
        model_name=best_model_name,
    )

    # Feature Importance
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

    # Permutation Importance
    logger.info("Calculating permutation importance on test set...")
    perm_df = compute_permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=5, random_state=config.random_seed
    )
    logger.info("Top Permutation Features:\n%s", perm_df.head(6).to_string(index=False))

    # Error Analysis (False Positives & False Negatives)
    error_analysis = analyze_prediction_errors(
        X_test, y_test, best_test_eval["probabilities"], threshold=config.default_threshold
    )
    logger.info(
        "Error breakdown: %d False Positives (retained but flagged), %d False Negatives (churned but missed)",
        len(error_analysis["false_positives"]),
        len(error_analysis["false_negatives"]),
    )

    # Step 11: End-to-End Prediction Verification
    logger.info("--- Step 11: Real-World Inference Test ---")
    predictor = ChurnPredictor(config.best_model_path)

    sample_customer_high_risk = {
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

    sample_customer_low_risk = {
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
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.20,
        "TotalCharges": 3912.00,
    }

    batch_test = predictor.predict([sample_customer_high_risk, sample_customer_low_risk])
    print("\nSAMPLE CUSTOMER INFERENCE RESULTS:")
    print(batch_test[["churn_probability", "predicted_churn", "risk_tier", "recommended_action"]].to_string())

    logger.info("================================================================================")
    logger.info("Pipeline execution completed successfully!")
    logger.info("Model saved to: %s", config.best_model_path)
    logger.info("Metrics comparison saved to: %s", config.metrics_comparison_path)
    logger.info("Figures saved in: %s", config.figures_dir)
    logger.info("================================================================================")


if __name__ == "__main__":
    run_pipeline()
