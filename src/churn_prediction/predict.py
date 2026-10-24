"""
Reusable inference module for predicting customer churn risk.

SECURITY NOTICE:
    Loading serialized models via `joblib` or `pickle` can execute arbitrary code
    if the artifact file is untrusted or tampered with. Only load artifacts generated
    by this verified pipeline and stored in trusted internal storage.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn_prediction.config import AppConfig, get_config

logger = logging.getLogger(__name__)

# Required predictor features (customerID and Churn are excluded from inference)
REQUIRED_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


class ChurnPredictor:
    """Production-grade churn prediction service wrapper around a fitted scikit-learn Pipeline."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[AppConfig] = None,
    ):
        self.config = config or get_config()
        self.model_path = Path(model_path) if model_path else self.config.best_model_path
        self.pipeline: Optional[Pipeline] = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load fitted pipeline artifact from disk with existence validation."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at: {self.model_path}\n"
                "Please run the training pipeline first: `python scripts/run_pipeline.py`"
            )
        try:
            self.pipeline = joblib.load(self.model_path)
            logger.info("Successfully loaded model pipeline from %s", self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize model pipeline from {self.model_path}: {e}")

    def validate_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input schema, column presence, and data hygiene.
        Safely removes customerID and Churn if present in the payload.
        Converts TotalCharges safely and coerces numeric types.
        """
        df = data.copy()

        # Drop identifier and target if accidentally provided in payload
        for col in [self.config.id_column, self.config.target_column]:
            if col in df.columns:
                logger.debug("Removing column '%s' from prediction payload.", col)
                df = df.drop(columns=[col])

        # Verify all required features exist
        missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
        if missing:
            raise ValueError(f"Input payload is missing required feature columns: {missing}")

        # Coerce numeric columns safely
        for num_col in NUMERIC_COLUMNS:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
            # If TotalCharges is NaN due to blank or 0 tenure, fill with 0.0
            if num_col == "TotalCharges":
                tenure_zero = (df["tenure"] == 0) & (df["TotalCharges"].isna())
                df.loc[tenure_zero, "TotalCharges"] = 0.0

        return df[REQUIRED_FEATURES]

    def predict(
        self,
        input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Generate churn risk predictions for customer records.

        Args:
            input_data: Single dictionary, list of dicts, or pandas DataFrame.
            threshold: Probability threshold above which a customer is flagged as Churn (default from config: 0.50).

        Returns:
            DataFrame containing churn_probability, predicted_churn (0 or 1),
            risk_tier (Low, Medium, High), and recommendation.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not loaded.")

        decision_threshold = threshold if threshold is not None else self.config.default_threshold

        # Format input into DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}. Expected dict, list of dicts, or DataFrame.")

        validated_df = self.validate_input(df)

        # Predict probability of churn (positive class = 1)
        probabilities = self.pipeline.predict_proba(validated_df)[:, 1]
        predictions = (probabilities >= decision_threshold).astype(int)

        # Assign operational risk tiers
        # Low: prob < 0.35, Medium: 0.35 <= prob < 0.65, High: prob >= 0.65
        risk_tiers = []
        recommendations = []
        for p in probabilities:
            if p >= 0.65:
                risk_tiers.append("High Risk")
                recommendations.append("Priority retention outreach, incentive offer, check service quality issues.")
            elif p >= 0.35:
                risk_tiers.append("Medium Risk")
                recommendations.append("Monitor account, target with feature adoption or loyalty campaigns.")
            else:
                risk_tiers.append("Low Risk")
                recommendations.append("Standard engagement; low risk of defection.")

        results = pd.DataFrame(
            {
                "churn_probability": np.round(probabilities, 4),
                "predicted_churn": predictions,
                "prediction_label": np.where(predictions == 1, "Likely to Churn", "Likely to Retain"),
                "risk_tier": risk_tiers,
                "recommended_action": recommendations,
            }
        )
        return results


def predict_single_customer(
    customer_dict: Dict[str, Any],
    model_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience helper to predict a single customer record."""
    predictor = ChurnPredictor(model_path=model_path)
    res_df = predictor.predict(customer_dict, threshold=threshold)
    return res_df.iloc[0].to_dict()
