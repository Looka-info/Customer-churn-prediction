"""Exploratory Data Analysis and visualization generation module."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from churn_prediction.config import AppConfig, get_config
from churn_prediction.data_loader import clean_raw_data

logger = logging.getLogger(__name__)


def generate_all_eda_figures(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate all Phase One exploratory visualizations from actual dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_df = clean_raw_data(df)
    clean_df["Churn_Label"] = clean_df["Churn"].map({1: "Churned (Yes)", 0: "Retained (No)"})

    # Set overall aesthetic style
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Target Class Distribution
    plt.figure(figsize=(6, 4.5))
    counts = clean_df["Churn_Label"].value_counts()
    percentages = (counts / len(clean_df)) * 100
    ax = sns.barplot(x=counts.index, y=counts.values, palette=["#2ca02c", "#d62728"])
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.annotate(
            f"{int(height):,} ({percentages.iloc[i]:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, height / 2.0),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            weight="bold",
        )
    plt.title("Target Distribution: Customer Churn vs Retained", fontsize=12, pad=12, weight="bold")
    plt.xlabel("Customer Status", fontsize=11)
    plt.ylabel("Number of Customers", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=300)
    plt.close()

    # 2. Missing Value Summary
    plt.figure(figsize=(8, 4))
    raw_missing = df.replace(" ", np.nan).isna().sum()
    missing_cols = raw_missing[raw_missing > 0]
    if len(missing_cols) == 0:
        # Display zero missing indicator
        plt.text(0.5, 0.5, "No Missing Values Found", ha="center", va="center", fontsize=14)
    else:
        sns.barplot(x=missing_cols.index, y=missing_cols.values, color="#e6550d")
        for i, val in enumerate(missing_cols.values):
            plt.text(i, val + 0.3, f"{val} records", ha="center", weight="bold")
        plt.ylabel("Missing Count", fontsize=11)
    plt.title("Data Hygiene: Missing / Blank Value Audit", fontsize=12, pad=12, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values.png", dpi=300)
    plt.close()

    # 3. Numerical Feature Distributions (tenure, MonthlyCharges, TotalCharges)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    titles = ["Tenure (Months)", "Monthly Charges ($)", "Total Charges ($)"]

    for ax, col, title in zip(axes, num_cols, titles):
        sns.histplot(
            data=clean_df,
            x=col,
            hue="Churn_Label",
            multiple="stack",
            palette={"Retained (No)": "#2ca02c", "Churned (Yes)": "#d62728"},
            bins=30,
            ax=ax,
            alpha=0.8,
        )
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel(title, fontsize=10)
        ax.set_ylabel("Customer Count", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "numerical_distributions.png", dpi=300)
    plt.close()

    # 4. Churn by Contract Type
    plt.figure(figsize=(7, 4.5))
    contract_churn = clean_df.groupby("Contract")["Churn"].mean().reset_index()
    contract_churn["Churn_Rate"] = contract_churn["Churn"] * 100
    contract_churn = contract_churn.sort_values(by="Churn_Rate", ascending=False)

    ax = sns.barplot(data=contract_churn, x="Contract", y="Churn_Rate", palette="Blues_r")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (p.get_x() + p.get_width() / 2.0, height + 1.0),
            ha="center",
            fontsize=11,
            weight="bold",
        )
    plt.title("Churn Rate by Contract Commitment Type", fontsize=12, pad=12, weight="bold")
    plt.xlabel("Contract Type", fontsize=11)
    plt.ylabel("Churn Rate (%)", fontsize=11)
    plt.ylim(0, 60)
    plt.tight_layout()
    plt.savefig(output_dir / "churn_by_contract.png", dpi=300)
    plt.close()

    # 5. Churn by Tenure Group
    plt.figure(figsize=(8, 4.5))
    clean_df["tenure_cohort"] = pd.cut(
        clean_df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    )
    tenure_churn = clean_df.groupby("tenure_cohort", observed=True)["Churn"].mean().reset_index()
    tenure_churn["Churn_Rate"] = tenure_churn["Churn"] * 100

    ax = sns.barplot(data=tenure_churn, x="tenure_cohort", y="Churn_Rate", palette="Purples_r")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (p.get_x() + p.get_width() / 2.0, height + 1.0),
            ha="center",
            fontsize=11,
            weight="bold",
        )
    plt.title("Churn Rate by Customer Tenure Cohort", fontsize=12, pad=12, weight="bold")
    plt.xlabel("Tenure Cohort", fontsize=11)
    plt.ylabel("Churn Rate (%)", fontsize=11)
    plt.ylim(0, 60)
    plt.tight_layout()
    plt.savefig(output_dir / "churn_by_tenure.png", dpi=300)
    plt.close()

    # 6. Churn by Payment Method
    plt.figure(figsize=(9, 4.5))
    pm_churn = clean_df.groupby("PaymentMethod")["Churn"].mean().reset_index()
    pm_churn["Churn_Rate"] = pm_churn["Churn"] * 100
    pm_churn = pm_churn.sort_values(by="Churn_Rate", ascending=False)

    ax = sns.barplot(data=pm_churn, x="PaymentMethod", y="Churn_Rate", palette="Oranges_r")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (p.get_x() + p.get_width() / 2.0, height + 1.0),
            ha="center",
            fontsize=10,
            weight="bold",
        )
    plt.title("Churn Rate by Payment Method", fontsize=12, pad=12, weight="bold")
    plt.xlabel("Payment Method", fontsize=11)
    plt.ylabel("Churn Rate (%)", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, 60)
    plt.tight_layout()
    plt.savefig(output_dir / "churn_by_payment_method.png", dpi=300)
    plt.close()

    # 7. Correlation Heatmap for Numerical Features
    plt.figure(figsize=(6, 5))
    num_corr = clean_df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].corr()
    sns.heatmap(
        num_corr,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".3f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap: Numerical Features & Churn", fontsize=12, pad=12, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
    plt.close()

    logger.info("Generated all 7 exploratory analysis figures in %s", output_dir)
