"""
Data service for loading, caching, and inspecting dataset artifacts.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from churn_prediction.config import get_config
from churn_prediction.data_loader import clean_raw_data, load_raw_data


def check_artifacts_status() -> Dict[str, Dict[str, Any]]:
    """
    Check the presence, size, and modification timestamp of all project artifacts.
    Returns structured status dictionary.
    """
    cfg = get_config()
    
    artifacts = {
        "Raw Dataset": {
            "path": cfg.raw_data_path,
            "description": "Original IBM Telco customer churn CSV",
            "remediation": "python train.py",
            "is_required_for": "Data Explorer, SQL ingestion, Training",
        },
        "SQLite Database": {
            "path": cfg.database_path,
            "description": "Local SQLite database with relational analytics tables",
            "remediation": "python train.py",
            "is_required_for": "SQL Insights dashboard page",
        },
        "Trained Model Pipeline": {
            "path": cfg.best_model_path,
            "description": "Serialized scikit-learn Pipeline (Random Forest)",
            "remediation": "python train.py",
            "is_required_for": "Predict Customer Churn form",
        },
        "Model Comparison Report": {
            "path": cfg.metrics_comparison_path,
            "description": "CSV comparison of CV and Test metrics across models",
            "remediation": "python train.py",
            "is_required_for": "Model Performance dashboard page",
        },
        "Diagnostic Figures Directory": {
            "path": cfg.figures_dir,
            "description": "Directory containing generated 300 DPI evaluation figures",
            "remediation": "python train.py",
            "is_required_for": "Visualizations & Diagnostic Plots",
        },
    }

    results = {}
    for name, item in artifacts.items():
        p: Path = item["path"]
        exists = p.exists()
        size_str = "N/A"
        mod_time = "N/A"

        if exists:
            if p.is_file():
                size_bytes = p.stat().st_size
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                mod_time = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            elif p.is_dir():
                file_count = len(list(p.glob("*.png")))
                size_str = f"{file_count} PNG files"
                mod_time = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        results[name] = {
            "exists": exists,
            "path": str(p),
            "size": size_str,
            "modified": mod_time,
            "description": item["description"],
            "remediation": item["remediation"],
            "is_required_for": item["is_required_for"],
        }

    return results


@st.cache_data(show_spinner=False)
def load_dataset_cached() -> Optional[pd.DataFrame]:
    """Load the raw dataset using cached reader."""
    cfg = get_config()
    if not cfg.raw_data_path.exists():
        return None
    try:
        return load_raw_data(cfg.raw_data_path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_clean_dataset_cached() -> Optional[pd.DataFrame]:
    """Load and clean the raw dataset using the project's standard hygiene rules."""
    raw_df = load_dataset_cached()
    if raw_df is None:
        return None
    return clean_raw_data(raw_df)


def get_dataset_summary() -> Optional[Dict[str, Any]]:
    """Compute high-level summary statistics from the actual dataset."""
    df = get_clean_dataset_cached()
    if df is None:
        return None

    total_records = len(df)
    churn_counts = df["Churn"].value_counts().to_dict()
    churned_count = churn_counts.get(1, 0)
    retained_count = churn_counts.get(0, 0)
    churn_rate_pct = (churned_count / total_records) * 100 if total_records > 0 else 0.0

    # Numeric summaries
    numeric_summary = {}
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            numeric_summary[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }

    # Group-level churn rates
    contract_rates = (
        df.groupby("Contract")["Churn"].agg(total="count", churn_rate=lambda s: s.mean() * 100).reset_index()
    )
    payment_rates = (
        df.groupby("PaymentMethod")["Churn"].agg(total="count", churn_rate=lambda s: s.mean() * 100).reset_index()
    )
    internet_rates = (
        df.groupby("InternetService")["Churn"].agg(total="count", churn_rate=lambda s: s.mean() * 100).reset_index()
    )

    return {
        "total_records": total_records,
        "churned_count": churned_count,
        "retained_count": retained_count,
        "churn_rate_pct": round(churn_rate_pct, 2),
        "column_count": len(df.columns),
        "numeric_summary": numeric_summary,
        "contract_rates": contract_rates,
        "payment_rates": payment_rates,
        "internet_rates": internet_rates,
    }
