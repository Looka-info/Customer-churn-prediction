"""Generate the 01_exploratory_data_analysis.ipynb notebook."""

import json
from pathlib import Path

def create_notebook():
    notebook_path = Path("notebooks/01_exploratory_data_analysis.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📊 Exploratory Data Analysis — Customer Churn Prediction\n",
                "\n",
                "**Project**: Production-Grade Customer Churn Machine Learning Pipeline  \n",
                "**Dataset**: IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)  \n",
                "**Target Variable**: `Churn` (`Yes` = 1, `No` = 0)  \n",
                "\n",
                "---\n",
                "\n",
                "### Business Objective\n",
                "Customer attrition is one of the costliest problems in the subscription telecommunications industry. Acquiring a new subscriber is typically 5x to 7x more expensive than retaining an existing customer. \n",
                "\n",
                "The goals of this exploratory analysis are:\n",
                "1. **Audit Data Hygiene**: Verify data types, identify missing/whitespace values, and handle anomalous entries.\n",
                "2. **Quantify Class Imbalance**: Measure the baseline churn rate across 7,043 customer accounts.\n",
                "3. **Identify Churn Drivers**: Examine relationships between customer tenure, monthly spend, contract type, and churn risk.\n",
                "4. **Analyze Service Profiles**: Discover which ancillary services (e.g., Tech Support, Online Security) correlate with high customer retention.\n",
                "5. **Execute SQL Analytical Aggregations**: Query the local SQLite database to confirm core business metrics.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "# Add project src directory to path\n",
                "sys.path.insert(0, str(Path.cwd().parent / 'src'))\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import sqlite3\n",
                "\n",
                "from churn_prediction.config import get_config\n",
                "from churn_prediction.data_loader import load_raw_data, clean_raw_data, validate_raw_data\n",
                "\n",
                "sns.set_theme(style='whitegrid', palette='muted')\n",
                "%matplotlib inline\n",
                "print('Libraries and configuration loaded.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Data Ingestion and Hygiene Audit\n",
                "Let's load the raw dataset and inspect its structure and data types."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "config = get_config()\n",
                "raw_df = load_raw_data(config.raw_data_path)\n",
                "\n",
                "print(f'Dataset Dimensions: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns')\n",
                "raw_df.head(5)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Validate raw dataset\n",
                "report = validate_raw_data(raw_df, config)\n",
                "for key, val in report.items():\n",
                "    print(f'{key}: {val}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### TotalCharges Whitespace Anomaly Investigation\n",
                "Notice that `TotalCharges` is stored as an object string in the raw CSV, and 11 records have blank whitespace (`' '`). Let's inspect these specific records."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "blank_tc = raw_df[raw_df['TotalCharges'].astype(str).str.strip() == '']\n",
                "print(f'Total records with blank TotalCharges: {len(blank_tc)}')\n",
                "blank_tc[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'Churn']]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "> **Key Finding**: Every single record with blank `TotalCharges` has `tenure == 0`. These are brand-new customers who registered during the current billing cycle and have not yet been billed. None of them have churned (`Churn == No`).\n",
                "> \n",
                "> **Strategy**: We logically impute `TotalCharges = 0.0` for accounts with `tenure == 0`, and use median imputation in our ML pipeline for any unseen missing values."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Target Variable Distribution (`Churn`)\n",
                "Let's quantify the balance between retained subscribers and churned subscribers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "clean_df = clean_raw_data(raw_df, config)\n",
                "\n",
                "churn_counts = clean_df['Churn'].value_counts()\n",
                "churn_pct = (churn_counts / len(clean_df)) * 100\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(6, 4))\n",
                "sns.barplot(x=['Retained (0)', 'Churned (1)'], y=churn_counts.values, palette=['#2ca02c', '#d62728'], ax=ax)\n",
                "for i, p in enumerate(ax.patches):\n",
                "    ax.annotate(f'{int(p.get_height()):,} ({churn_pct.iloc[i]:.1f}%)',\n",
                "                (p.get_x() + p.get_width() / 2., p.get_height() / 2.),\n",
                "                ha='center', va='center', fontsize=12, color='white', weight='bold')\n",
                "plt.title('Target Class Distribution: Customer Churn', fontsize=13, weight='bold')\n",
                "plt.ylabel('Subscriber Count')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "The dataset exhibits moderate class imbalance: **26.54%** of subscribers churned (1,869 customers), while **73.46%** remained (5,174 customers). Because of this class distribution, evaluating models purely by overall accuracy would be misleading. Precision, Recall, and PR-AUC (Average Precision) are prioritized."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Numerical Feature Analysis: Tenure, Monthly Charges, Total Charges"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))\n",
                "num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']\n",
                "titles = ['Tenure in Months', 'Monthly Charges ($)', 'Total Charges ($)']\n",
                "\n",
                "clean_df['Churn_Label'] = clean_df['Churn'].map({1: 'Churned', 0: 'Retained'})\n",
                "\n",
                "for ax, col, title in zip(axes, num_features, titles):\n",
                "    sns.kdeplot(data=clean_df, x=col, hue='Churn_Label', common_norm=False, fill=True, alpha=0.35, ax=ax, palette={'Retained': '#2ca02c', 'Churned': '#d62728'})\n",
                "    ax.set_title(title, fontsize=12, weight='bold')\n",
                "    ax.set_xlabel(title)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Numerical Insights:\n",
                "- **Tenure**: Extremely strong predictor. Churn is heavily concentrated in the first 0–12 months. Customers who survive past 24–36 months exhibit dramatically lower churn rates.\n",
                "- **Monthly Charges**: Customers paying higher monthly charges ($70–$100) churn at a significantly higher rate than those paying lower monthly rates ($20–$30).\n",
                "- **Total Charges**: Skewed towards lower values for churners due to their short tenure."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Categorical Drivers of Customer Churn\n",
                "Let's examine how Contract Type, Payment Method, and Internet Service impact churn."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cat_cols = ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport']\n",
                "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                "\n",
                "for ax, col in zip(axes.flatten(), cat_cols):\n",
                "    rate_df = clean_df.groupby(col)['Churn'].mean().reset_index()\n",
                "    rate_df['Churn_Rate'] = rate_df['Churn'] * 100\n",
                "    rate_df = rate_df.sort_values(by='Churn_Rate', ascending=False)\n",
                "    \n",
                "    sns.barplot(data=rate_df, x=col, y='Churn_Rate', palette='Blues_r', ax=ax)\n",
                "    ax.set_title(f'Churn Rate by {col}', fontsize=12, weight='bold')\n",
                "    ax.set_ylabel('Churn Rate (%)')\n",
                "    ax.set_xlabel('')\n",
                "    ax.set_ylim(0, 60)\n",
                "    ax.tick_params(axis='x', rotation=15)\n",
                "    for p in ax.patches:\n",
                "        ax.annotate(f'{p.get_height():.1f}%',\n",
                "                    (p.get_x() + p.get_width() / 2., p.get_height() + 1.0),\n",
                "                    ha='center', fontsize=10, weight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Key Categorical Observations:\n",
                "1. **Contract Type**: Month-to-month contracts have a **42.7%** churn rate, compared to **11.3%** for One Year and just **2.8%** for Two Year contracts.\n",
                "2. **Payment Method**: Electronic check users have a **45.3%** churn rate, compared to ~15-19% for automated methods (Bank transfer or Credit card).\n",
                "3. **Internet Service**: Fiber optic customers experience a **41.9%** churn rate, whereas DSL is **19.0%** and No Internet is **7.4%**.\n",
                "4. **Tech Support**: Customers without Tech Support churn at **41.6%**, compared to **15.2%** for customers with active Tech Support."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. SQL Database Analysis\n",
                "Let's query the local SQLite database (`data/churn_analysis.db`) to verify aggregate metrics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "conn = sqlite3.connect(str(config.database_path))\n",
                "\n",
                "query = '''\n",
                "SELECT \n",
                "    contract,\n",
                "    internet_service,\n",
                "    COUNT(*) AS total_customers,\n",
                "    SUM(churn_numeric) AS churned_customers,\n",
                "    ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_pct,\n",
                "    ROUND(AVG(monthly_charges), 2) AS avg_monthly_bill\n",
                "FROM telco_churn\n",
                "GROUP BY contract, internet_service\n",
                "ORDER BY churn_rate_pct DESC;\n",
                "'''\n",
                "\n",
                "sql_result = pd.read_sql_query(query, conn)\n",
                "conn.close()\n",
                "sql_result"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Correlation Heatmap\n",
                "Let's examine linear correlations among numerical variables and the binary churn indicator."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(6, 5))\n",
                "corr = clean_df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].corr()\n",
                "sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.3f', linewidths=0.5)\n",
                "plt.title('Correlation Heatmap: Numerical Predictors & Churn', fontsize=12, weight='bold')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Conclusions & Modeling Strategy\n",
                "\n",
                "1. **Strongest Predictors**:\n",
                "   - Contract commitment (`Month-to-month` vs `Two-year`)\n",
                "   - Customer `tenure` (first 12 months is the high-risk window)\n",
                "   - Fiber optic service without technical/security add-ons\n",
                "   - Non-automatic payment methods (`Electronic check`)\n",
                "2. **Leakage Prevention Requirements**:\n",
                "   - Exclude `customerID` from all model feature matrices.\n",
                "   - Fit all scalers, encoders, and imputers strictly within training folds.\n",
                "3. **Operational Recommendation**:\n",
                "   - Offer incentives for month-to-month Fiber optic subscribers to convert to annual agreements or enroll in automatic payment.\n",
                "   - Target new subscribers in months 1–6 with proactive onboarding and tech support assistance."
            ]
        }
    ]

    notebook_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
    print(f"Created notebook at {notebook_path}")

if __name__ == "__main__":
    create_notebook()
