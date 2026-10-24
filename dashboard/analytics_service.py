"""
Analytics service for executing read-only SQL queries against the local SQLite database.
"""

from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from churn_prediction.config import get_config


# Verified analytical queries to expose on the dashboard
ANALYTICAL_QUERIES = [
    {
        "id": "overall_summary",
        "title": "Overall Customer and Churn Counts",
        "description": "High-level count of total accounts, retained accounts, churned accounts, and average monthly bill.",
        "sql": """
SELECT
    COUNT(*) AS total_customers,
    SUM(churn_numeric) AS churned_customers,
    COUNT(*) - SUM(churn_numeric) AS retained_customers,
    ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_bill
FROM telco_churn;
        """.strip(),
    },
    {
        "id": "churn_by_contract",
        "title": "Churn Rate by Contract Type",
        "description": "Compares churn prevalence and average monthly bill across Month-to-month, One-year, and Two-year contracts.",
        "sql": """
SELECT
    contract,
    COUNT(*) AS total_customers,
    SUM(churn_numeric) AS churned_customers,
    ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM telco_churn
GROUP BY contract
ORDER BY churn_rate_percent DESC;
        """.strip(),
    },
    {
        "id": "churn_by_payment_method",
        "title": "Churn Rate by Payment Method",
        "description": "Compares churn prevalence across Electronic check, Mailed check, and Automatic payment options.",
        "sql": """
SELECT
    payment_method,
    COUNT(*) AS total_customers,
    SUM(churn_numeric) AS churned_customers,
    ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM telco_churn
GROUP BY payment_method
ORDER BY churn_rate_percent DESC;
        """.strip(),
    },
    {
        "id": "churn_by_tenure_group",
        "title": "Churn Rate by Tenure Cohort",
        "description": "Aggregates customer retention across duration cohorts: 0-12, 13-24, 25-48, and 49-72 months.",
        "sql": """
SELECT
    CASE
        WHEN tenure <= 12 THEN '0-12 Months'
        WHEN tenure <= 24 THEN '13-24 Months'
        WHEN tenure <= 48 THEN '25-48 Months'
        ELSE '49-72 Months'
    END AS tenure_group,
    COUNT(*) AS total_customers,
    SUM(churn_numeric) AS churned_customers,
    ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM telco_churn
GROUP BY tenure_group
ORDER BY MIN(tenure) ASC;
        """.strip(),
    },
    {
        "id": "charges_by_churn_status",
        "title": "Average Charges by Churn Status",
        "description": "Compares average monthly charges and total lifetime charges between retained and churned subscribers.",
        "sql": """
SELECT
    churn,
    COUNT(*) AS customer_count,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(total_charges), 2) AS avg_total_charges
FROM telco_churn
GROUP BY churn;
        """.strip(),
    },
    {
        "id": "contract_and_internet_service",
        "title": "Contract and Internet Service Interaction",
        "description": "Cross-tabulation of contract type and internet service tier sorted by highest churn risk.",
        "sql": """
SELECT
    contract,
    internet_service,
    COUNT(*) AS total_customers,
    SUM(churn_numeric) AS churned_customers,
    ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM telco_churn
GROUP BY contract, internet_service
ORDER BY churn_rate_percent DESC;
        """.strip(),
    },
]


def is_database_available() -> bool:
    """Check if the SQLite database file exists and contains the required table."""
    cfg = get_config()
    db_path = cfg.database_path
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='telco_churn';")
            return cursor.fetchone() is not None
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def execute_readonly_query(sql_query: str) -> pd.DataFrame:
    """
    Execute a read-only SQL query against the local SQLite database.
    Enforces URI read-only connection to guarantee no unintended database mutations.
    """
    cfg = get_config()
    db_path = cfg.database_path

    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {db_path}. "
            "Run 'python train.py' to initialize and populate it."
        )

    # Use read-only URI mode to prevent any modification
    uri_path = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri_path, uri=True) as conn:
        df = pd.read_sql_query(sql_query, conn)
    return df


def get_available_queries() -> List[Dict[str, str]]:
    """Return catalog of verified analytical SQL queries."""
    return ANALYTICAL_QUERIES
