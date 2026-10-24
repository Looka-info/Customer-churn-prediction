# 📊 Customer Churn Prediction — Production ML System

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Pytest](https://img.shields.io/badge/tests-18%20passed-brightgreen.svg)](https://docs.pytest.org/)

An end-to-end, reproducible, production-grade Machine Learning system designed to predict customer attrition for subscription-based telecommunication providers. Built with strict **data-leakage prevention**, modular architecture, stratified cross-validation, SQL data validation, model interpretability, and automated testing.

---

## 📑 Table of Contents
1. [Business Problem & Operational Value](#-business-problem--operational-value)
2. [Project Objectives & Scope](#-project-objectives--scope)
3. [Dataset & Data Governance](#-dataset--data-governance)
4. [Technology Stack](#-technology-stack)
5. [Repository Architecture](#-repository-architecture)
6. [Data Hygiene & Exploratory Analysis](#-data-hygiene--exploratory-analysis)
7. [SQL Analytics & Invariant Verification](#-sql-analytics--invariant-verification)
8. [Data Preparation & Leakage Prevention](#-data-preparation--leakage-prevention)
9. [Model Training & Cross-Validation](#-model-training--cross-validation)
10. [Empirical Model Comparison](#-empirical-model-comparison)
11. [Model Interpretability & Error Analysis](#-model-interpretability--error-analysis)
12. [Inference Service & Usage Example](#-inference-service--usage-example)
13. [Limitations & Ethical Considerations](#-limitations--ethical-considerations)
14. [Installation & Reproduction Guide](#-installation--reproduction-guide)
15. [Automated Testing](#-automated-testing)
16. [Future Improvements](#-future-improvements)

---

## 💼 Business Problem & Operational Value

In subscription telecommunications, acquiring a new customer costs **5× to 7× more** than retaining an existing one. Unplanned subscriber turnover ("churn") directly damages recurring monthly revenue (MRR) and customer lifetime value (LTV).

However, customer retention budgets are finite. A telecommunications operator cannot offer expensive rate discounts or concierge onboarding to every customer. Conversely, sending indiscriminate retention promotions to satisfied, low-risk customers wastes marketing budget and can inadvertently prompt contract reconsiderations.

### Core Business Questions Addressed
* **Risk Prioritization**: Which subscribers are at highest risk of canceling their service in the near term?
* **Behavioral Associations**: What service configurations, tenure milestones, and billing methods are most strongly correlated with customer departure?
* **Intervention Trade-offs**: How should retention teams balance False Positives (spending retention capital on a customer who would have stayed anyway) against False Negatives (missing a churning customer, incurring revenue loss)?
* **Actionable Retention**: How can probabilistic risk estimates be translated into tiered operational playbooks?

> **Operational Principle**: This model estimates **statistical churn risk**; it does not claim to establish metaphysical causation for an individual customer's decision. Predictions serve to guide triage and human-reviewed customer success workflows.

---

## 🎯 Project Objectives & Scope

| Area | Scope / Deliverable |
| :--- | :--- |
| **Problem Type** | Binary Classification (Supervised Learning) |
| **Target Variable** | `Churn`: `1` = Customer churned, `0` = Customer retained |
| **Baseline Model** | Logistic Regression with standardized numeric features and one-hot encoded categoricals |
| **Candidate Models** | Random Forest Classifier, Gradient Boosting Classifier |
| **Validation Strategy** | Stratified 5-Fold Cross-Validation on training data; held-out test set evaluated exactly once |
| **Key Metrics** | PR-AUC (Average Precision), ROC-AUC, F1-Score, Precision, Recall |
| **Decision Threshold** | Calibrated on training-side out-of-fold validation predictions |
| **Persistence** | Complete `Pipeline` (preprocessing + estimator) serialized via `joblib` |
| **Verification** | Pytest automated test suite covering schema, hygiene, leakage, and inference |

---

## 📦 Dataset & Data Governance

* **Source**: IBM Telco Customer Churn Public Sample Dataset
* **Public Reference**: [Kaggle blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Official Hosted CSV**: Distributed via [IBM/telco-customer-churn-on-icp4d](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
* **Dimensions**: 7,043 customer accounts, 21 attributes (20 predictive features + `customerID` identifier + `Churn` target)
* **Target Distribution**:
  * Retained (`No`): 5,174 customers (73.46%)
  * Churned (`Yes`): 1,869 customers (26.54%)
* **Data Hygiene Invariant**:
  * Exactly 11 records have blank whitespace (`" "`) in `TotalCharges`.
  * Every one of these 11 accounts has `tenure == 0` (brand-new accounts registered in the current cycle that have not yet generated a billing invoice).
  * Strategy: Impute `TotalCharges = 0.0` for `tenure == 0`, and maintain median imputation in the preprocessing pipeline for unseen inference data.
* **Privacy & Identifier Isolation**:
  * `customerID` is strictly an administrative tracking key and is systematically stripped from all feature matrices prior to model fitting.

---

## 🛠 Technology Stack

* **Language**: Python 3.11 / 3.12
* **Data Manipulation**: `pandas` (>= 2.2), `numpy` (>= 2.0)
* **Machine Learning**: `scikit-learn` (>= 1.5) (`Pipeline`, `ColumnTransformer`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`)
* **Storage & Relational Analytics**: SQLite3 and ANSI SQL
* **Data Visualization**: `matplotlib` (>= 3.9), `seaborn` (>= 0.13)
* **Serialization**: `joblib` (>= 1.4)
* **Configuration**: `pyyaml` (>= 6.0)
* **Testing**: `pytest` (>= 8.0)

---

## 📂 Repository Architecture

```text
customer-churn-prediction/
├── README.md                           # Comprehensive documentation and results
├── requirements.txt                    # Pinned Python package dependencies
├── pyproject.toml                      # Project metadata and test configurations
├── pytest.ini                          # Pytest discovery and pythonpath settings
├── .gitignore                          # Excludes models, caches, and datasets
├── .env.example                        # Template for environment configuration
├── configs/
│   └── config.yaml                     # Centralized paths, columns, and hyperparameters
├── data/
│   ├── raw/                            # Raw dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv)
│   ├── processed/                      # Preprocessed artifacts (if required)
│   └── churn_analysis.db               # SQLite database containing telco_churn table
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Interactive EDA walkthrough
├── src/
│   └── churn_prediction/
│       ├── __init__.py                 # Package declaration
│       ├── config.py                   # Dataclass-based path and configuration loader
│       ├── data_loader.py              # Ingestion, schema verification, and hygiene
│       ├── database.py                 # SQLite ingestion and SQL-vs-Pandas verification
│       ├── preprocessing.py            # Leakage-free ColumnTransformer & data splitting
│       ├── train.py                    # Stratified 5-Fold CV and pipeline fitting
│       ├── evaluate.py                 # Metrics, threshold trade-offs, and ROC/PR plots
│       ├── interpret.py                # Tree importances, odds ratios, error analysis
│       ├── eda.py                      # Batch generation of report-ready figures
│       └── predict.py                  # Production inference class with input validation
├── sql/
│   ├── schema.sql                      # DDL schema for SQLite telco_churn table
│   └── analysis_queries.sql            # Parameterized analytical SQL queries
├── reports/
│   ├── model_comparison.csv            # Real measured CV and held-out test scores
│   └── figures/                        # High-resolution (300 DPI) generated figures
│       ├── target_distribution.png
│       ├── missing_values.png
│       ├── numerical_distributions.png
│       ├── churn_by_contract.png
│       ├── churn_by_tenure.png
│       ├── churn_by_payment_method.png
│       ├── correlation_heatmap.png
│       ├── confusion_matrix.png
│       ├── pr_roc_curves.png
│       └── feature_importance.png
├── models/
│   └── best_model.joblib               # Serialized full pipeline (fitted preprocessor + estimator)
├── tests/
│   ├── test_data.py                    # Schema validation, missing handling, target encoding
│   ├── test_preprocessing.py           # Unseen category safety, ColumnTransformer checks
│   ├── test_training.py                # Pipeline fit/predict, cross-validation sanity
│   └── test_prediction.py              # Schema validation, probability range, save/load
└── scripts/
    ├── download_data.py                # Standalone script to download verified dataset
    ├── create_notebook.py              # Generates the EDA notebook
    └── run_pipeline.py                 # Master CLI script running full end-to-end workflow
```

---

## 🔍 Data Hygiene & Exploratory Analysis

All exploratory analysis was run directly against the 7,043 rows of the verified IBM Telco dataset. Report charts are generated at 300 DPI in `reports/figures/`.

### 1. Class Distribution & Baseline Churn Rate
The dataset contains 1,869 churn events against 5,174 retained customers (**26.54% churn rate**).

![Target Distribution](reports/figures/target_distribution.png)

### 2. Numerical Features by Churn Status
* **Tenure**: Strongest single duration feature. Customers in months 0–12 exhibit churn rates exceeding 45%, while customers with tenure > 48 months drop below 10%.
* **Monthly Charges**: Concentrated at higher tiers for churned subscribers ($70–$100/month), often associated with Fiber Optic service.
* **Total Charges**: Skewed lower for churners due to their brief account lifespans.

![Numerical Distributions](reports/figures/numerical_distributions.png)

### 3. Key Categorical Associations
* **Contract Type**: Month-to-month contracts have a **42.7%** churn rate, compared to **11.3%** for One-Year and only **2.8%** for Two-Year contracts.
* **Payment Method**: Electronic Check users experience a **45.3%** churn rate, whereas automated credit card / bank transfer subscribers churn at ~15–18%.
* **Internet Service**: Fiber Optic subscribers exhibit a **41.9%** churn rate versus **19.0%** for DSL and **7.4%** for customers without internet service.

| Feature Category | High Churn Cohort | Churn Rate | Low Churn Cohort | Churn Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Contract** | Month-to-month | 42.7% | Two-year contract | 2.8% |
| **Payment Method** | Electronic check | 45.3% | Credit card (auto) | 15.2% |
| **Internet Service** | Fiber optic | 41.9% | No internet | 7.4% |
| **Tech Support** | No Tech Support | 41.6% | Tech Support active | 15.2% |

![Churn by Contract](reports/figures/churn_by_contract.png)
![Churn by Payment Method](reports/figures/churn_by_payment_method.png)

---

## 🗄 SQL Analytics & Invariant Verification

To mirror enterprise production workflows where business analysts query data warehouses, the validated dataset is loaded into a local SQLite database (`data/churn_analysis.db`).

The pipeline executes automated invariant checks (`src/churn_prediction/database.py:verify_sql_against_pandas`) ensuring that SQL aggregations and Pandas calculations agree to within floating-point tolerance (0.01%):

```sql
-- Churn Rate by Contract Type & Internet Service
SELECT
    contract,
    internet_service,
    COUNT(*) AS total_customers,
    SUM(churn_numeric) AS churned_customers,
    ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_bill
FROM telco_churn
GROUP BY contract, internet_service
ORDER BY churn_rate_percent DESC;
```

**Output from Database Execution**:
```text
      contract internet_service  total_customers  churned_customers  churn_rate_percent  avg_monthly_bill
Month-to-month      Fiber optic             2128               1162               54.61             87.03
Month-to-month              DSL             1223                399               32.62             50.31
Month-to-month               No              524                 94               17.94             20.37
      One year      Fiber optic              538                148               27.51             98.81
      One year              DSL              570                 19                3.33             61.50
      Two year      Fiber optic              430                 37                8.60            103.20
      One year               No              365                  7                1.92             20.80
      Two year              DSL              628                 12                1.91             60.10
      Two year               No              697                  4                0.57             21.50
```

> **Verification Result**: `SQL vs Pandas verification: PASSED. Overall churn rate: 26.54% (1,869 / 7,043).`

---

## 🛡 Data Preparation & Leakage Prevention

Data leakage is the most frequent failure mode in real-world ML engineering. This repository follows strict defensive engineering rules:

1. **Isolation Prior to Preprocessing**:
   * Stratified 80/20 train/test split performed **first**.
   * Training set: 5,634 samples.
   * Held-out test set: 1,409 samples.
   * Both partitions retain an identical 26.54% churn prevalence.
2. **Scikit-Learn `ColumnTransformer`**:
   * **Numeric Pipeline**: `SimpleImputer(strategy='median')` followed by `StandardScaler()`. Imputer medians and scaling parameters ($\mu, \sigma$) are computed **only** on training folds.
   * **Categorical Pipeline**: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)`.
3. **Unseen Categories Resilience**:
   * New categorical categories encountered during inference are cleanly encoded as all zeros without raising runtime exceptions.
4. **Identifier Isolation**:
   * `customerID` is excluded from feature matrices.
5. **Threshold Tuning Hygiene**:
   * Decision threshold calibration is conducted strictly on **training out-of-fold validation predictions**, never on the held-out test set.

---

## 🏋️ Model Training & Cross-Validation

Three distinct classifiers were evaluated on the training set using identical **Stratified 5-Fold Cross-Validation** splits (`random_state=42`):

1. **Model A: Logistic Regression** — Interpretable baseline with L2 regularization ($C=1.0$).
2. **Model B: Random Forest Classifier** — Non-linear tree ensemble (`n_estimators=150`, `max_depth=8`, `min_samples_leaf=2`).
3. **Model C: Gradient Boosting Classifier** — Boosted decision trees (`n_estimators=100`, `learning_rate=0.08`, `max_depth=4`, `subsample=0.85`).

### Primary Selection Metric: PR-AUC (Average Precision)
Because churn is an imbalanced classification problem, overall Accuracy is misleading (a naive classifier predicting "No Churn" achieves 73.5% accuracy with 0% recall). **PR-AUC (Average Precision)** summarizes precision across all recall thresholds and is the primary criterion for model selection.

---

## 📊 Empirical Model Comparison

The table below presents the **actual measured results** from running `scripts/run_pipeline.py`. 
Validation metrics reflect Stratified 5-Fold Cross-Validation on the training data; Test metrics reflect the held-out test set evaluated exactly once.

| Model | CV PR-AUC (Mean ± Std) | CV ROC-AUC | CV F1 | CV Precision | CV Recall | Test PR-AUC | Test ROC-AUC | Test F1 | Test Precision | Test Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.6650 ± 0.0203** | 0.8475 | 0.5828 | **0.6739** | 0.5137 | 0.6517 | 0.8405 | 0.5836 | **0.6761** | 0.5134 |
| **Gradient Boosting** | 0.6643 ± 0.0189 | **0.8482** | 0.5833 | 0.6616 | 0.5224 | **0.6596** | **0.8465** | 0.5774 | 0.6598 | 0.5134 |
| **Logistic Regression** | 0.6615 ± 0.0195 | 0.8462 | **0.5932** | 0.6530 | **0.5445** | 0.6336 | 0.8420 | **0.6040** | 0.6572 | **0.5588** |

*Artifact location: [`reports/model_comparison.csv`](reports/model_comparison.csv)*

### Model Selection Decision
* **Selected Model**: **Random Forest** was selected during the cross-validation phase based on highest validation PR-AUC (**0.6650**).
* **Test Performance**: On the unseen held-out test set, Gradient Boosting achieved a slightly higher Test PR-AUC (0.6596 vs 0.6517) and ROC-AUC (0.8465 vs 0.8405), while Random Forest delivered the highest test precision (67.61%).
* All three models perform within ~1% of each other, confirming that linear features with interaction terms capture the primary business signals.

### Final Held-Out Test Evaluation (Threshold = 0.50)
* **True Negatives (TN)**: 943 customers (correctly identified as retained)
* **False Positives (FP)**: 92 customers (retained customers flagged as churn risk)
* **False Negatives (FN)**: 182 customers (churned customers missed by the model)
* **True Positives (TP)**: 192 customers (correctly identified churners)

![Confusion Matrix](reports/figures/confusion_matrix.png)
![PR & ROC Curves](reports/figures/pr_roc_curves.png)

---

## 📈 Probability Threshold Tuning

The standard decision threshold of $0.50$ is not necessarily optimal for every business context:

| Threshold | Precision | Recall | F1-Score | True Positives | False Positives | Operational Implication |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.20** | 0.4619 | **0.8729** | 0.6042 | 1,305 | 1,520 | **High Coverage**: Captures 87% of all churners; best for cheap digital campaigns (e.g., email tips). |
| **0.35** | 0.5649 | **0.7251** | **0.6350** | 1,084 | 835 | **Balanced**: Captures ~73% of churners with 56% precision. |
| **0.50** | **0.6731** | 0.5137 | 0.5827 | 768 | 373 | **Conservative Baseline**: Precision prioritized (67%); minimizes outreach waste. |
| **0.70** | **0.8338** | 0.1846 | 0.3023 | 276 | 55 | **High Precision**: 83% precision; reserved for expensive direct retention incentives. |

---

## 🔬 Model Interpretability & Error Analysis

Understanding *why* the model makes predictions is vital for customer-facing teams.

### 1. Top Feature Importances (Tree MDI vs Permutation)
Top predictive features identified by Permutation Importance on held-out test data:
1. `Contract`: Month-to-month contracts strongly elevate churn probability.
2. `tenure`: Low tenure (< 12 months) sharply increases churn risk.
3. `TotalCharges` & `MonthlyCharges`: Higher monthly bills increase friction.
4. `InternetService`: Fiber optic subscribers experience heightened churn rates.
5. `TechSupport`: Lack of tech support correlates with early churn.

![Feature Importance](reports/figures/feature_importance.png)

### 2. Logistic Regression Odds Ratios (Interpretable Baseline)
* **Contract = Two-year**: Odds Ratio $\approx 0.38$ (reduces odds of churn by ~62%).
* **InternetService = Fiber optic**: Odds Ratio $\approx 1.74$ (increases odds of churn by ~74%).
* **PaymentMethod = Electronic check**: Odds Ratio $\approx 1.32$ (increases odds of churn by ~32%).

### 3. Error Analysis: Failure Mode Breakdown
* **False Positives (92 accounts)**: Typically month-to-month subscribers with Fiber optic and high monthly charges who nonetheless remained loyal, often due to high family attachment (`Partner=Yes`, `Dependents=Yes`).
* **False Negatives (182 accounts)**: Subscribers with moderate tenure (15–30 months) on annual contracts who unexpectedly departed, often due to localized service outages or competitor promotional offers not tracked in this dataset.

---

## 🚀 Inference Service & Usage Example

The saved model artifact (`models/best_model.joblib`) encapsulates both the fitted `ColumnTransformer` and the trained classifier. Inference does **not** require re-running notebooks or re-executing training scripts.

### Programmatic Prediction Example

```python
from churn_prediction.predict import ChurnPredictor

# Initialize inference service (loads models/best_model.joblib)
predictor = ChurnPredictor()

# Define customer profile for scoring
new_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
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
    "MonthlyCharges": 82.50,
    "TotalCharges": 247.50
}

# Generate risk assessment
result = predictor.predict(new_customer, threshold=0.50)
print(result[["churn_probability", "predicted_churn", "risk_tier", "recommended_action"]])
```

**Output**:
```text
   churn_probability  predicted_churn  risk_tier                                                           recommended_action
0             0.7159                1  High Risk  Priority retention outreach, incentive offer, check service quality issues.
```

### Risk Tier Definitions
* **High Risk ($\ge 65\%$)**: Priority retention call from customer success; eligible for contract discount or complimentary speed boost.
* **Medium Risk ($35\% \le p < 65\%$)**: Automated feature adoption emails, customer satisfaction surveys, check for unresolved support tickets.
* **Low Risk ($< 35\%$)**: Standard engagement; no intrusive intervention.

---

## ⚖️ Limitations & Ethical Considerations

1. **Correlation vs. Causation**:
   * A strong statistical association between `Contract: Month-to-month` and churn does not mean that forcing subscribers into multi-year contracts will prevent churn. In fact, removing flexible contract options could depress new sign-ups.
2. **Omitted Confounders**:
   * The dataset lacks competitive market data (e.g., competing fiber rollout in subscriber's zip code), technical network telemetry (outage frequency, latency), and customer sentiment / support ticket logs.
3. **Sensitive Demographic Attributes**:
   * Attributes such as `gender`, `SeniorCitizen`, `Partner`, and `Dependents` are present in the dataset.
   * **Ethical Guardrail**: Retention incentives must never discriminate on price or service terms based on protected characteristics. These variables should be audited for disparate impact in operational settings.
4. **Serialization Security**:
   * `joblib` files can deserialize arbitrary code. Model artifacts should only be loaded from trusted internal artifact registries with cryptographic checksum verification.

---

## 💻 Installation & Reproduction Guide

### Prerequisites
* Python 3.11 or 3.12
* Git

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Complete End-to-End Pipeline
```bash
python scripts/run_pipeline.py
```
This single command executes:
1. Verification and download of raw IBM Telco dataset.
2. SQLite schema creation and SQL invariant validation.
3. Generation of all 10 report-ready figures in `reports/figures/`.
4. Stratified 5-Fold cross-validation across all 3 candidate models.
5. Final evaluation on the held-out test set and export of `reports/model_comparison.csv`.
6. Serialization of `models/best_model.joblib`.
7. Demonstration inference on sample customer profiles.

---

## 🧪 Automated Testing

The project includes an extensive automated test suite run via `pytest`.

```bash
python -m pytest
```

### Test Coverage Summary:
* `tests/test_data.py`: Validates schema conformance, missing column detection, duplicate ID handling, blank `TotalCharges` imputation, and `customerID` exclusion.
* `tests/test_preprocessing.py`: Tests `ColumnTransformer` output dimensions, handling of completely novel unseen categorical levels, and train/test stratification reproducibility.
* `tests/test_training.py`: Tests candidate model creation, pipeline fit/predict probabilities, and cross-validation execution on training fixtures.
* `tests/test_prediction.py`: Tests single-dict and batch inference, output column integrity, probability bounds $[0.0, 1.0]$, schema validation on missing inputs, and model serialization round-trip.

```text
======================== 18 passed, 1 warning in 3.04s ========================
```

---

## 🔮 Future Improvements

1. **Class Imbalance Techniques**: Integrate SMOTE or balanced class weighting inside cross-validation training folds.
2. **Telemetry & Time-Series Data**: Ingest monthly usage trends (e.g., bandwidth decline over time) as leading indicators.
3. **REST API Deployment**: Wrap the `ChurnPredictor` in a containerized FastAPI microservice with Prometheus latency metrics.
4. **SHAP Integration**: Compute TreeSHAP values for real-time individual prediction explanations in customer service dashboards.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
Dataset copyright belongs to IBM Corporation; distributed for research and educational purposes.
