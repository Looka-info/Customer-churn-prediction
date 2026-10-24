# 📊 Customer Churn Prediction — Production ML System

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Pytest](https://img.shields.io/badge/tests-23%20passed-brightgreen.svg)](https://docs.pytest.org/)

An audited, reproducible, production-grade Machine Learning system designed to predict customer attrition for subscription telecommunications providers. Built with strict **data-leakage prevention**, modular architecture, cross-validation uncertainty audits, SQL data validation, cost-benefit threshold calibration, model interpretability, and automated testing.

---

## 📑 Table of Contents
1. [Business Problem & Operational Value](#-business-problem--operational-value)
2. [Project Objectives & Scope](#-project-objectives--scope)
3. [Real-World Prediction Assumptions & Limitations](#-real-world-prediction-assumptions--limitations)
4. [Dataset Governance & Licensing](#-dataset-governance--licensing)
5. [Technology Stack](#-technology-stack)
6. [Repository Architecture & Artifact Boundaries](#-repository-architecture--artifact-boundaries)
7. [Data Hygiene & Exploratory Analysis](#-data-hygiene--exploratory-analysis)
8. [SQL Analytics & Invariant Verification](#-sql-analytics--invariant-verification)
9. [Data Preparation & Leakage Prevention](#-data-preparation--leakage-prevention)
10. [Model Training, Uncertainty Audit & Defensible Selection](#-model-training-uncertainty-audit--defensible-selection)
11. [Empirical Model Comparison & Metrics](#-empirical-model-comparison--metrics)
12. [Cost-Benefit Threshold Calibration](#-cost-benefit-threshold-calibration)
13. [Model Interpretability & Error Analysis](#-model-interpretability--error-analysis)
14. [Inference Service & Usage Example](#-inference-service--usage-example)
15. [Responsible AI & Sensitive Attributes Safeguards](#-responsible-ai--sensitive-attributes-safeguards)
16. [Installation & Reproduction Guide](#-installation--reproduction-guide)
17. [Automated Testing](#-automated-testing)
18. [Future Improvements](#-future-improvements)

---

## 💼 Business Problem & Operational Value

In subscription telecommunications, acquiring a new subscriber costs **5× to 7× more** than retaining an existing customer. Unplanned subscriber turnover directly erodes recurring monthly revenue (MRR) and customer lifetime value (LTV).

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
| **Primary Metric** | **PR-AUC (Average Precision)** via `sklearn.metrics.average_precision_score` |
| **Decision Threshold** | Calibrated on training-side out-of-fold validation predictions using an economic utility model |
| **Persistence** | Complete `Pipeline` (preprocessing + estimator) serialized via `joblib` |
| **Verification** | Pytest automated test suite (23 passing tests) covering schema, hygiene, leakage, and inference |

---

## ⏱ Real-World Prediction Assumptions & Limitations

A critical audit requirement for production machine learning systems is articulating when predictions occur, what data is legitimately available, and the structural limitations of the dataset:

### 1. Intended Prediction Point & Horizon
* **Prediction Point ($T$)**: The model is intended to execute at the close of a subscriber's monthly billing cycle (or 30–60 days prior to contract expiration).
* **Prediction Horizon ($T + 30\text{ days}$)**: The target represents whether the subscriber will terminate service during the subsequent 30-day billing cycle.

### 2. Available vs. Proxy Features (`TotalCharges` Analysis)
* In our audit of the 7,043 customer records, `TotalCharges` exhibits a **0.99956 correlation** with the mathematical product of `tenure * MonthlyCharges` (mean absolute difference = $45.02 across up to 72 months of service).
* **Multicollinearity & Suppressor Effects**: In linear models (Logistic Regression), including `tenure`, `MonthlyCharges`, and `TotalCharges` simultaneously causes severe multicollinearity. Specifically, `MonthlyCharges` takes on a negative coefficient (-0.599) in the presence of `TotalCharges` and `tenure`, despite having a positive univariate correlation (+0.193) with churn. When `tenure` and `TotalCharges` are held constant, a higher `MonthlyCharges` mathematically implies shorter tenure or a recent plan change. Tree-based models (Random Forest, Gradient Boosting) are naturally resilient to this collinearity because split points evaluate features independently.

### 3. Cross-Sectional vs. Prospective Generalization
* The IBM Telco dataset is a **retrospective cross-sectional snapshot**. All tenure cohorts (from month 1 to month 72) are observed simultaneously.
* **Limitation**: A random stratified train/test split evaluates *cross-sectional* generalization across customers, but **not prospective temporal generalization** across calendar time. In an enterprise production setting, a model trained on data up to Month $T$ must predict on Month $T+1$ through $T+6$. If competitor campaigns, seasonal patterns, or macroeconomic conditions change, performance may degrade. A production deployment requires rolling-window temporal validation and drift monitoring.

---

## 📦 Dataset Governance & Licensing

* **Dataset Origin**: IBM Telco Customer Churn Public Sample Dataset, originally published for IBM Cloud Pak for Data and Cognos Analytics.
* **Public References**: [Kaggle blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and [IBM Community Samples](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv).
* **License Notice**: 
  * The code, pipelines, and test suites in this repository are licensed under the **MIT License**.
  * The underlying IBM Telco Customer Churn dataset is property of IBM Corporation and is distributed under IBM sample terms for educational, research, and non-commercial development. The repository MIT license does **not** purport to re-license third-party data assets.
* **Dimensions & Balance**:
  * Total Accounts: 7,043.
  * Retained (`No`): 5,174 accounts (73.46%).
  * Churned (`Yes`): 1,869 accounts (26.54%).
* **Data Hygiene Invariant**:
  * Exactly 11 accounts have blank whitespace (`" "`) in `TotalCharges`.
  * Every one of these 11 accounts has `tenure == 0` (brand-new accounts registered in the current cycle that have not yet generated a billing invoice).
  * **Hygiene Rule**: Impute `TotalCharges = 0.0` for `tenure == 0`, and maintain median imputation in the preprocessing pipeline for unseen inference data.

---

## 🛠 Technology Stack

* **Language**: Python 3.11 / 3.12
* **Data Processing**: `pandas` (>= 2.2), `numpy` (>= 2.0)
* **Statistical Auditing**: `scipy.stats` (paired t-tests across cross-validation folds)
* **Machine Learning**: `scikit-learn` (>= 1.5) (`Pipeline`, `ColumnTransformer`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`)
* **Storage & Relational Analytics**: SQLite3 and ANSI SQL
* **Visualization**: `matplotlib` (>= 3.9), `seaborn` (>= 0.13)
* **Serialization**: `joblib` (>= 1.4)
* **Configuration**: `pyyaml` (>= 6.0)
* **Testing**: `pytest` (>= 8.0)

---

## 📂 Repository Architecture & Artifact Boundaries

```text
customer-churn-prediction/
├── README.md                           # Documentation with audited methodology and metrics
├── requirements.txt                    # Pinned Python package dependencies
├── pyproject.toml                      # Project metadata and test configurations
├── pytest.ini                          # Pytest discovery and pythonpath settings
├── LICENSE                             # MIT License (code only)
├── .gitignore                          # Clear boundary between tracked code vs local artifacts
├── .env.example                        # Template for environment configuration
├── configs/
│   └── config.yaml                     # Centralized paths, columns, and hyperparameters
├── data/
│   ├── raw/                            # Ignored in git (downloaded via script)
│   ├── processed/                      # Ignored in git
│   └── churn_analysis.db               # Ignored in git (generated locally from SQL)
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # 22-cell interactive EDA walkthrough
├── src/
│   └── churn_prediction/
│       ├── __init__.py                 # Package declaration
│       ├── config.py                   # Dataclass-based path and configuration loader
│       ├── data_loader.py              # Ingestion, schema verification, and hygiene
│       ├── database.py                 # SQLite ingestion and SQL-vs-Pandas verification
│       ├── preprocessing.py            # Leakage-free ColumnTransformer & data splitting
│       ├── train.py                    # Stratified 5-Fold CV, paired t-tests, defensible selection
│       ├── evaluate.py                 # Metrics, economic utility threshold tuning, curves
│       ├── interpret.py                # Tree importances, odds ratios, error analysis
│       ├── eda.py                      # Batch generation of report-ready figures
│       └── predict.py                  # Production inference class with input validation
├── sql/
│   ├── schema.sql                      # DDL schema for SQLite telco_churn table
│   └── analysis_queries.sql            # Parameterized analytical SQL queries
├── reports/
│   ├── model_comparison.csv            # Tracked analytical evidence (CV & Test metrics)
│   └── figures/                        # Tracked 300 DPI generated diagnostic charts
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
│   └── best_model.joblib               # Ignored in git (generated locally by training pipeline)
├── tests/
│   ├── test_data.py                    # Schema validation, missing handling, target encoding
│   ├── test_preprocessing.py           # Unseen category safety, ColumnTransformer checks
│   ├── test_training.py                # Pipeline fit/predict, cross-validation sanity
│   ├── test_prediction.py              # Schema validation, probability range, save/load
│   └── test_audit.py                   # Audit checks: leakage, statistical tests, threshold utility
└── scripts/
    ├── download_data.py                # Downloads and verifies raw dataset
    ├── create_notebook.py              # Generates the EDA notebook
    └── run_pipeline.py                 # Master CLI script running full end-to-end workflow
```

---

## 🔍 Data Hygiene & Exploratory Analysis

All exploratory analysis was conducted on the full 7,043 records of the verified IBM Telco dataset. Figures are saved at 300 DPI in `reports/figures/`.

### 1. Class Distribution & Baseline Churn Rate
The baseline churn prevalence is **26.54%** (1,869 churners vs. 5,174 retained).

![Target Distribution](reports/figures/target_distribution.png)

### 2. Numerical Features by Churn Status
* **Tenure**: The single most decisive duration feature. Accounts in months 0–12 exhibit churn rates exceeding 45%, while accounts with tenure > 48 months drop below 10%.
* **Monthly Charges**: Concentrated at higher tiers for churned subscribers ($70–$100/month), predominantly driven by Fiber Optic service.
* **Total Charges**: Skewed lower for churners due to premature departure.

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

To mirror enterprise workflows where analysts query data warehouses, the validated dataset is loaded into a local SQLite database (`data/churn_analysis.db`).

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

> **Verification Result**: `SQL vs Pandas verification: PASSED. Overall churn rate: 26.54% (1,869 / 7,043).`

---

## 🛡 Data Preparation & Leakage Prevention

Defensive engineering rules enforce total separation between training data and held-out evaluation:

1. **Isolation Prior to Preprocessing**:
   * Stratified 80/20 train/test split performed **first** (`random_state=42`).
   * Training set: 5,634 samples. Held-out test set: 1,409 samples.
   * Both partitions retain an identical 26.54% churn prevalence.
2. **ColumnTransformer Encapsulation**:
   * **Numeric Pipeline**: `SimpleImputer(strategy='median')` followed by `StandardScaler()`. Imputer medians and scaling parameters ($\mu, \sigma$) are computed **only** on training folds.
   * **Categorical Pipeline**: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)`.
3. **Unseen Categories Resilience**:
   * Previously unseen categorical values at inference time are safely encoded as zeros without throwing runtime errors.
4. **Identifier Isolation**:
   * `customerID` is excluded from all model feature matrices.
5. **Threshold Tuning Hygiene**:
   * Decision threshold calibration is conducted strictly on **training out-of-fold validation predictions**, never on the held-out test set.

---

## 🔬 Model Training, Uncertainty Audit & Defensible Selection

Three distinct classifiers were evaluated on the training set using identical **Stratified 5-Fold Cross-Validation** splits (`random_state=42`):
1. **Model A: Logistic Regression** — Baseline with L2 regularization ($C=1.0$).
2. **Model B: Random Forest Classifier** — Tree ensemble (`n_estimators=150`, `max_depth=8`, `min_samples_leaf=2`).
3. **Model C: Gradient Boosting Classifier** — Boosted trees (`n_estimators=100`, `learning_rate=0.08`, `max_depth=4`, `subsample=0.85`).

### 1. Primary Selection Metric Definition: PR-AUC (Average Precision)
In an imbalanced classification problem (26.5% positive rate), overall Accuracy is uninformative. **PR-AUC (Average Precision)** is selected as the primary ranking metric. 

> **Methodological Note on Metric Calculation**:  
> The PR-AUC reported here uses `sklearn.metrics.average_precision_score`:
> $$\text{AP} = \sum_{n} (R_n - R_{n-1}) P_n$$
> This computes the weighted mean of precisions at each threshold. This is preferred over trapezoidal numerical integration (`sklearn.metrics.auc(recall, precision)`), which is known to produce optimistic overestimates on non-monotonic precision-recall curves.

### 2. Statistical Significance Audit Across CV Folds
Rather than picking a model based on microscopic point differences, we conducted paired t-tests across the 5 cross-validation folds against the Logistic Regression baseline:

```text
STATISTICAL SIGNIFICANCE vs LOGISTIC REGRESSION BASELINE (PR-AUC):
    Candidate_Model      Baseline_Model  Mean_Difference  Paired_t_stat  p_value  Statistically Distinguishable (p < 0.05)
Logistic Regression Logistic Regression           0.0000          0.000    1.000                                     False
      Random Forest Logistic Regression          +0.0035          1.148    0.315                                     False
  Gradient Boosting Logistic Regression          +0.0028          0.875    0.431                                     False

STATISTICAL SIGNIFICANCE vs LOGISTIC REGRESSION BASELINE (ROC-AUC):
    Candidate_Model      Baseline_Model  Mean_Difference  Paired_t_stat  p_value  Statistically Distinguishable (p < 0.05)
Logistic Regression Logistic Regression           0.0000          0.000    1.000                                     False
      Random Forest Logistic Regression          +0.0013          0.440    0.683                                     False
  Gradient Boosting Logistic Regression          +0.0020          0.525    0.627                                     False
```

### 3. Defensible Model Selection Decision
* **Finding**: The differences in discriminative power between Logistic Regression (0.6615), Gradient Boosting (0.6643), and Random Forest (0.6650) are **statistically indistinguishable** ($p > 0.30$, well within the standard deviation of $\pm 0.020$).
* **Decision Rationale**:
  * If the business priority is **Occam's razor / governance**: Logistic Regression is the most defensible choice. It offers closed-form log-odds interpretability, zero ensemble inference latency, and higher default recall (0.5445 vs 0.5137).
  * If the business priority is **minimizing wasted retention budget**: Random Forest is selected because it delivers higher cross-validation precision (**0.6739 vs 0.6530**), reducing false-positive retention outreach in finite-budget customer success campaigns.
  * In this repository, **Random Forest** is retained as the operational pipeline, and the statistical tie with Logistic Regression is transparently documented in the audit trail.

---

## 📊 Empirical Model Comparison & Metrics

All metrics reflect actual execution (`reports/model_comparison.csv`):

| Model | CV PR-AUC (Mean ± Std) | CV ROC-AUC | CV F1 | CV Precision | CV Recall | Test PR-AUC | Test ROC-AUC | Test F1 | Test Precision | Test Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.6650 ± 0.0203** | 0.8475 | 0.5828 | **0.6739** | 0.5137 | 0.6517 | 0.8405 | 0.5836 | **0.6761** | 0.5134 |
| **Gradient Boosting** | 0.6643 ± 0.0189 | **0.8482** | 0.5833 | 0.6616 | 0.5224 | **0.6596** | **0.8465** | 0.5774 | 0.6598 | 0.5134 |
| **Logistic Regression** | 0.6615 ± 0.0195 | 0.8462 | **0.5932** | 0.6530 | **0.5445** | 0.6336 | 0.8420 | **0.6040** | 0.6572 | **0.5588** |

### Test Confusion Matrix Breakdown (Random Forest @ Threshold = 0.50)
* **True Negatives (TN)**: 943 accounts
* **False Positives (FP)**: 92 accounts
* **False Negatives (FN)**: 182 accounts
* **True Positives (TP)**: 192 accounts
* **Precision**: $\frac{192}{192 + 92} = 67.61\%$
* **Recall**: $\frac{192}{192 + 182} = 51.34\%$
* **F1-Score**: $58.36\%$

![Confusion Matrix](reports/figures/confusion_matrix.png)
![PR & ROC Curves](reports/figures/pr_roc_curves.png)

---

## 💰 Cost-Benefit Threshold Calibration

A standard classification threshold of $0.50$ is rarely optimal in business operations. We evaluated threshold trade-offs and net business utility **strictly on training out-of-fold predictions**:

### Business Utility Model Assumptions
* **Cost of False Positive ($C_{FP}$)**: $20 (cost of unnecessary retention offer or discount).
* **Cost of False Negative ($C_{FN}$)**: $100 (lifetime gross margin lost when a churner is missed).
* **Benefit of True Positive ($B_{TP}$)**: $80 (net lifetime margin preserved by successful retention outreach).
* **Net Value**: $(\text{TP} \times \$80) - (\text{FP} \times \$20) - (\text{FN} \times \$100)$

```text
TRAINING-SIDE OUT-OF-FOLD THRESHOLD & BUSINESS UTILITY TRADE-OFFS:
 threshold  precision  recall     f1  true_positives  false_positives  true_negatives  false_negatives  net_business_value
      0.20     0.4619  0.8729 0.6042            1305             1520            2619              190           +$55,000.0
      0.25     0.5028  0.8334 0.6272            1246             1232            2907              249           +$50,140.0
      0.30     0.5335  0.7779 0.6329            1163             1017            3122              332           +$39,500.0
      0.35     0.5649  0.7251 0.6350            1084              835            3304              411           +$28,920.0
      0.40     0.6051  0.6528 0.6281             976              637            3502              519           +$13,440.0
      0.45     0.6370  0.5846 0.6097             874              498            3641              621            -$2,140.0
      0.50     0.6731  0.5137 0.5827             768              373            3766              727           -$18,720.0
      0.60     0.7282  0.3244 0.4489             485              181            3958             1010           -$65,820.0
      0.70     0.8338  0.1846 0.3023             276               55            4084             1219          -$100,920.0
```

### Operational Threshold Comparison on Held-Out Test Set
Evaluating the training-calibrated threshold on the unseen test set demonstrates the real-world operational trade-off:

| Operating Threshold | Precision | Recall | F1-Score | TP | FP | FN | TN | Operational Strategy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.50 (Default)** | **67.61%** | 51.34% | 58.36% | 192 | 92 | 182 | 943 | Conservative: Minimizes wasted retention offers; leaves 182 churners uncontacted. |
| **0.35 (Calibrated)** | 54.79% | **71.93%** | **62.20%** | **269** | 222 | **105** | 813 | High Coverage: Captures 77 additional churners (+20.6% recall) while preserving 55% precision. |

---

## 🔬 Model Interpretability & Error Analysis

Understanding feature influences allows retention teams to craft personalized interventions rather than generic discounts.

### 1. Permutation Importance (Model-Agnostic on Held-Out Data)
Permutation importance measures the drop in PR-AUC when a feature's values are randomly shuffled on held-out test data (unbiased by cardinality):
1. `Contract`: Shuffling contract type degrades PR-AUC by **0.0686 ± 0.0037**.
2. `tenure`: Degradation of **0.0653 ± 0.0070**.
3. `TotalCharges`: Degradation of **0.0327 ± 0.0070**.
4. `InternetService`: Degradation of **0.0317 ± 0.0139**.
5. `TechSupport`: Degradation of **0.0234 ± 0.0051**.

![Feature Importance](reports/figures/feature_importance.png)

### 2. Error Analysis: Failure Mode Breakdown
* **False Positives (92 accounts @ 0.50)**: High-tenure subscribers on month-to-month fiber optic contracts with multiple lines and family accounts (`Partner=Yes`, `Dependents=Yes`). Their high bill triggers risk flags, but high family switching costs keep them retained.
* **False Negatives (182 accounts @ 0.50)**: Subscribers with moderate tenure (12–24 months) on annual contracts who unexpectedly departed. These cases often correspond to unobserved external drivers (e.g. competitor fiber rollouts, localized outages, or billing disputes) absent from this cross-sectional dataset.

---

## 🚀 Inference Service & Usage Example

The saved model pipeline (`models/best_model.joblib`) encapsulates both feature transformations and estimator weights. Inference runs independently without executing training code.

```python
from churn_prediction.predict import ChurnPredictor

# Initialize inference service (loads models/best_model.joblib)
predictor = ChurnPredictor()

# Single customer payload
customer = {
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
    "TotalCharges": 159.70
}

# Predict with calibrated operational threshold (0.35)
result = predictor.predict(customer, threshold=0.35)
print(result[["churn_probability", "predicted_churn", "risk_tier", "recommended_action"]])
```

**Output**:
```text
   churn_probability  predicted_churn  risk_tier                                                           recommended_action
0             0.7159                1  High Risk  Priority retention outreach, incentive offer, check service quality issues.
```

---

## ⚖️ Responsible AI & Sensitive Attributes Safeguards

The dataset contains demographic features: `gender`, `SeniorCitizen` (age proxy), `Partner`, and `Dependents`.

### Safeguards for Operational Deployment:
1. **No Discriminatory Pricing**: Retention discounts, fee waivers, or promotional packages must **never** be conditioned on protected demographic attributes.
2. **Disparate Impact Auditing**: Model acceptance rates across gender and senior citizen cohorts must be monitored to ensure interventions do not systematically disadvantage vulnerable groups.
3. **Correlation $\ne$ Causation**: A statistical association between `Contract: Month-to-month` and churn (42.7%) does not prove that forcing customers into multi-year contracts reduces churn. Eliminating flexible contracts may reduce overall customer acquisition.
4. **Serialization Security**: `joblib` artifacts execute arbitrary bytecode upon deserialization. Only load models from verified internal storage repositories with cryptographic checksum validation.

---

## 💻 Installation & Reproduction Guide

### Prerequisites
* Python 3.11 or 3.12
* Git

```bash
# 1. Clone repository
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate # Linux / macOS

# 3. Install package in editable mode
pip install -r requirements.txt
pip install -e .

# 4. Run automated test suite (23 tests)
python -m pytest

# 5. Run end-to-end pipeline (downloads data, verifies SQL, trains models, saves artifacts)
python scripts/run_pipeline.py
```

---

## 🧪 Automated Testing

The automated test suite verifies all critical invariants across 23 unit and audit tests:

```bash
python -m pytest
```

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1 -- D:\tools\python312\python.exe
testpaths: tests
collected 23 items

tests/test_audit.py::test_threshold_tuning_uses_only_training_data PASSED [  4%]
tests/test_audit.py::test_cv_statistical_significance_calculator PASSED  [  8%]
tests/test_audit.py::test_defensible_model_selection_parsimony_strategy PASSED [ 13%]
tests/test_audit.py::test_defensible_model_selection_tradeoff_strategy PASSED [ 17%]
tests/test_audit.py::test_average_precision_metric_properties PASSED     [ 21%]
tests/test_data.py::test_schema_validation_success PASSED                [ 26%]
tests/test_data.py::test_schema_validation_missing_column PASSED         [ 30%]
tests/test_data.py::test_schema_validation_duplicate_ids PASSED          [ 34%]
tests/test_data.py::test_clean_raw_data_handles_blank_total_charges PASSED [ 39%]
tests/test_data.py::test_target_encoding PASSED                          [ 43%]
tests/test_data.py::test_prepare_features_and_target_leakage_prevention PASSED [ 47%]
tests/test_data.py::test_load_raw_data_missing_file_raises_error PASSED  [ 52%]
tests/test_prediction.py::test_prediction_output_shape_and_probability_bounds PASSED [ 56%]
tests/test_prediction.py::test_prediction_handles_dictionary_input PASSED [ 60%]
tests/test_prediction.py::test_prediction_validation_catches_missing_columns PASSED [ 65%]
tests/test_prediction.py::test_prediction_safely_removes_customer_id_and_churn PASSED [ 69%]
tests/test_prediction.py::test_predictor_missing_model_file_error PASSED [ 73%]
tests/test_preprocessing.py::test_build_preprocessor_handles_missing_and_scaling PASSED [ 78%]
tests/test_preprocessing.py::test_preprocessor_handles_unseen_categories PASSED [ 82%]
tests/test_preprocessing.py::test_create_train_test_split_stratification_and_seed PASSED [ 86%]
tests/test_training.py::test_candidate_models_contain_required_classifiers PASSED [ 91%]
tests/test_training.py::test_build_model_pipeline_returns_valid_pipeline PASSED [ 95%]
tests/test_training.py::test_cross_validation_execution PASSED           [100%]

======================== 23 passed, 1 warning in 3.71s ========================
```

---

## 🔮 Future Improvements

1. **Temporal Validation**: Ingest real longitudinal customer history with explicit snapshot timestamps to test true out-of-time forecasting.
2. **Network Telemetry Integration**: Merge broadband latency, fiber signal degradation, and dropped call frequency to capture operational drivers of attrition.
3. **Causal Uplift Modeling**: Train two-model or T-Learner uplift estimators to identify *persuadable* subscribers (those who only stay if contacted) versus *sure things* or *lost causes*.
4. **FastAPI Microservice**: Expose `ChurnPredictor` via a low-latency asynchronous REST endpoint with OpenTelemetry tracing and Prometheus latency histograms.

---

## 📄 License
This repository code is licensed under the [MIT License](LICENSE).  
The IBM Telco Customer Churn dataset is property of IBM Corporation; used for educational and research benchmarking.
