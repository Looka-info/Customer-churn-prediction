"""SQLite database manager and SQL verification for customer churn analytics."""

import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from churn_prediction.config import AppConfig, get_config
from churn_prediction.data_loader import clean_raw_data, load_raw_data

logger = logging.getLogger(__name__)


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """Create or connect to the local SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: Path, schema_path: Path) -> None:
    """Initialize database tables using the schema DDL."""
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with get_db_connection(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()
    logger.info("Database schema initialized at %s", db_path)


def load_data_to_sqlite(
    df: pd.DataFrame,
    db_path: Path,
    schema_path: Path,
) -> int:
    """
    Clean and ingest the raw Telco DataFrame into the SQLite table.
    Returns the number of rows inserted.
    """
    init_database(db_path, schema_path)

    # Prepare DataFrame matching SQLite schema
    cleaned = clean_raw_data(df)
    churn_numeric = (cleaned["Churn"] == 1).astype(int) if cleaned["Churn"].dtype != int else cleaned["Churn"]
    churn_text = cleaned["Churn"].map({1: "Yes", 0: "No"}) if set(cleaned["Churn"].unique()).issubset({0, 1}) else df["Churn"]

    sql_df = pd.DataFrame(
        {
            "customer_id": cleaned["customerID"],
            "gender": cleaned["gender"],
            "senior_citizen": cleaned["SeniorCitizen"].astype(int),
            "partner": cleaned["Partner"],
            "dependents": cleaned["Dependents"],
            "tenure": cleaned["tenure"].astype(int),
            "phone_service": cleaned["PhoneService"],
            "multiple_lines": cleaned["MultipleLines"],
            "internet_service": cleaned["InternetService"],
            "online_security": cleaned["OnlineSecurity"],
            "online_backup": cleaned["OnlineBackup"],
            "device_protection": cleaned["DeviceProtection"],
            "tech_support": cleaned["TechSupport"],
            "streaming_tv": cleaned["StreamingTV"],
            "streaming_movies": cleaned["StreamingMovies"],
            "contract": cleaned["Contract"],
            "paperless_billing": cleaned["PaperlessBilling"],
            "payment_method": cleaned["PaymentMethod"],
            "monthly_charges": cleaned["MonthlyCharges"].astype(float),
            "total_charges": cleaned["TotalCharges"],
            "churn": churn_text,
            "churn_numeric": churn_numeric,
        }
    )

    with get_db_connection(db_path) as conn:
        sql_df.to_sql("telco_churn", conn, if_exists="append", index=False)
        conn.commit()

    logger.info("Ingested %d records into telco_churn table", len(sql_df))
    return len(sql_df)


def run_sql_query(query: str, db_path: Path, params: Optional[tuple] = None) -> pd.DataFrame:
    """Execute an analytical SQL query against the SQLite database and return a DataFrame."""
    with get_db_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def execute_analysis_queries(db_path: Path) -> Dict[str, pd.DataFrame]:
    """Execute standard analytical queries and return results dictionary."""
    results = {}

    queries = {
        "overall_summary": """
            SELECT
                COUNT(*) AS total_customers,
                SUM(churn_numeric) AS churned_customers,
                COUNT(*) - SUM(churn_numeric) AS retained_customers,
                ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent,
                ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
            FROM telco_churn;
        """,
        "churn_by_contract": """
            SELECT
                contract,
                COUNT(*) AS total_customers,
                SUM(churn_numeric) AS churned_customers,
                ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent,
                ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
            FROM telco_churn
            GROUP BY contract
            ORDER BY churn_rate_percent DESC;
        """,
        "churn_by_payment_method": """
            SELECT
                payment_method,
                COUNT(*) AS total_customers,
                SUM(churn_numeric) AS churned_customers,
                ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent
            FROM telco_churn
            GROUP BY payment_method
            ORDER BY churn_rate_percent DESC;
        """,
        "churn_by_tenure_group": """
            SELECT
                CASE
                    WHEN tenure <= 12 THEN '0-12 Months'
                    WHEN tenure <= 24 THEN '13-24 Months'
                    WHEN tenure <= 48 THEN '25-48 Months'
                    ELSE '49-72 Months'
                END AS tenure_group,
                COUNT(*) AS total_customers,
                SUM(churn_numeric) AS churned_customers,
                ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent
            FROM telco_churn
            GROUP BY tenure_group
            ORDER BY churn_rate_percent DESC;
        """,
        "charges_by_churn": """
            SELECT
                churn,
                COUNT(*) AS customer_count,
                ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
                ROUND(AVG(total_charges), 2) AS avg_total_charges
            FROM telco_churn
            GROUP BY churn;
        """,
        "churn_by_contract_and_internet": """
            SELECT
                contract,
                internet_service,
                COUNT(*) AS total_customers,
                SUM(churn_numeric) AS churned_customers,
                ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent
            FROM telco_churn
            GROUP BY contract, internet_service
            ORDER BY churn_rate_percent DESC;
        """,
    }

    for name, q in queries.items():
        results[name] = run_sql_query(q, db_path)

    return results


def verify_sql_against_pandas(db_path: Path, raw_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Explicit verification: Compare SQL aggregated metrics against Pandas calculations.
    Ensures mathematical agreement between the two calculation engines.
    """
    clean_df = clean_raw_data(raw_df)
    clean_df["churn_num"] = clean_df["Churn"].astype(int)

    # 1. Overall churn rate
    sql_overall = run_sql_query(
        "SELECT COUNT(*) AS total, SUM(churn_numeric) AS churned, ROUND(AVG(churn_numeric)*100.0, 2) AS rate FROM telco_churn;",
        db_path,
    ).iloc[0]

    pandas_total = len(clean_df)
    pandas_churned = int(clean_df["churn_num"].sum())
    pandas_rate = round((pandas_churned / pandas_total) * 100.0, 2)

    assert int(sql_overall["total"]) == pandas_total, f"Mismatch total: SQL={sql_overall['total']} vs Pandas={pandas_total}"
    assert int(sql_overall["churned"]) == pandas_churned, f"Mismatch churned: SQL={sql_overall['churned']} vs Pandas={pandas_churned}"
    assert abs(float(sql_overall["rate"]) - pandas_rate) < 0.05, f"Mismatch rate: SQL={sql_overall['rate']} vs Pandas={pandas_rate}"

    # 2. Churn rate by contract
    sql_contract = run_sql_query(
        "SELECT contract, COUNT(*) as cnt, SUM(churn_numeric) as churned, ROUND(AVG(churn_numeric)*100.0, 2) as rate FROM telco_churn GROUP BY contract ORDER BY contract;",
        db_path,
    ).set_index("contract")

    pandas_contract = clean_df.groupby("Contract").agg(
        cnt=("churn_num", "count"),
        churned=("churn_num", "sum"),
        rate=("churn_num", lambda x: round(x.mean() * 100.0, 2)),
    ).sort_index()

    for contract in pandas_contract.index:
        s_row = sql_contract.loc[contract]
        p_row = pandas_contract.loc[contract]
        assert int(s_row["cnt"]) == int(p_row["cnt"]), f"Contract count mismatch for {contract}"
        assert int(s_row["churned"]) == int(p_row["churned"]), f"Contract churned mismatch for {contract}"
        assert abs(float(s_row["rate"]) - float(p_row["rate"])) < 0.05, f"Contract rate mismatch for {contract}"

    # 3. Monthly charges by churn status
    sql_charges = run_sql_query(
        "SELECT churn, ROUND(AVG(monthly_charges), 2) as avg_mc FROM telco_churn GROUP BY churn ORDER BY churn;",
        db_path,
    ).set_index("churn")

    clean_df["churn_str"] = clean_df["churn_num"].map({1: "Yes", 0: "No"})
    pandas_charges = clean_df.groupby("churn_str")["MonthlyCharges"].mean().round(2)

    for status in ["No", "Yes"]:
        s_val = float(sql_charges.loc[status, "avg_mc"])
        p_val = float(pandas_charges.loc[status])
        assert abs(s_val - p_val) < 0.05, f"Monthly charges mismatch for status {status}: SQL={s_val} vs Pandas={p_val}"

    return {
        "status": "PASSED",
        "verified_metrics": [
            "total_customers",
            "churned_customers",
            "overall_churn_rate_percent",
            "contract_type_breakdown",
            "monthly_charges_by_churn_status",
        ],
        "overall_total": pandas_total,
        "overall_churned": pandas_churned,
        "overall_churn_rate_percent": pandas_rate,
    }
