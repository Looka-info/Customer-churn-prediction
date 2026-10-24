# Customer Churn Prediction

I built this project to understand how machine learning can be used to predict customer churn. I used the IBM Telco Customer Churn dataset, explored the data, trained three classification models, compared their performance, and created a reusable prediction pipeline.

Customer churn is one of the classic tabular classification problems in data science. It has practical challenges that make it an excellent learning project: the classes are imbalanced (roughly 26.5% of customers churned), false positives and false negatives carry very different operational costs, and simple accuracy is misleading. Throughout this project, I focused on building a clean and reproducible workflow, preventing data leakage, evaluating models rigorously using cross-validation and statistical tests, and analyzing the trade-offs involved in operational decision-making.

---

## Table of Contents
1. [Project Overview & Motivation](#project-overview--motivation)
2. [The Dataset & Scope](#the-dataset--scope)
3. [Data Cleaning & Exploratory Analysis](#data-cleaning--exploratory-analysis)
4. [SQL Analysis & Invariant Verification](#sql-analysis--invariant-verification)
5. [Preventing Data Leakage](#preventing-data-leakage)
6. [Feature Collinearity & Controlled Ablation](#feature-collinearity--controlled-ablation)
7. [Why I Evaluated Three Models](#why-i-evaluated-three-models)
8. [Evaluation Metrics & Why Accuracy is Misleading](#evaluation-metrics--why-accuracy-is-misleading)
9. [Measured Results & Model Comparison](#measured-results--model-comparison)
10. [Decision Thresholds & Business Utility](#decision-thresholds--business-utility)
11. [Model Interpretation & Error Analysis](#model-interpretation--error-analysis)
12. [Inference Pipeline & Example Usage](#inference-pipeline--example-usage)
13. [Project Structure](#project-structure)
14. [How to Run the Project](#how-to-run-the-project)
15. [What Did Not Work & What I Would Improve Next](#what-did-not-work--what-i-would-improve-next)

---

## Project Overview & Motivation

In subscription businesses like telecommunications, acquiring a new customer typically costs 5× to 7× more than retaining an existing one. If customer success teams can identify which subscribers are most likely to leave, they can reach out proactively with targeted offers, plan adjustments, or customer service interventions.

However, machine learning models do not magically solve churn. A model simply identifies patterns in historical customer data that correlate with past churn events. Building this project helped me work through several fundamental machine learning engineering questions:

- How do you build an end-to-end pipeline where data preparation, training, evaluation, and inference are clean, modular, and leak-free?
- How do you evaluate an imbalanced classification problem when accuracy is meaningless?
- How do you determine whether a more complex model (like Gradient Boosting or Random Forest) actually outperforms a simple baseline (like Logistic Regression), or if the difference is just random fold noise?
- How do you translate model probability scores into real-world business decisions that account for outreach costs and retention success rates?

---

## The Dataset & Scope

I used the publicly available **IBM Telco Customer Churn dataset** (originally created for IBM Cognos Analytics and IBM Cloud Pak for Data, widely available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and the [IBM Community Samples repository](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)).

* **Total Accounts**: 7,043
* **Features**: 20 customer attributes (account info, subscribed services, demographics, and billing details)
* **Target Variable**: `Churn` (`Yes` / `No`)
* **Class Balance**: 
  * Retained (`No`): 5,174 customers (73.46%)
  * Churned (`Yes`): 1,869 customers (26.54%)

### Important Scope Clarification: Retrospective Classification, Not Prospective Forecasting

A key engineering consideration with this dataset is understanding its temporal structure:
* The IBM Telco dataset contains **zero timestamps**—no account creation dates, no monthly billing event dates, and no churn timestamps.
* This means the project is strictly a **retrospective classification benchmark**. The model learns correlations between static customer profiles and past churn labels.
* It is methodologically incorrect to claim that this model predicts whether a customer will churn "over the next 30 days." There is no forward-looking observation window in this dataset.

To build and evaluate a genuine prospective churn prediction model in a real company, you would need:
1. **Longitudinal event timestamps**: Exact dates for logins, support tickets, billing charges, and cancellations.
2. **Observation window ($[T_0, T_{\text{obs}}]$)**: A historical period (e.g., 90 days) where features like usage trends and support calls are calculated.
3. **Buffer / action window ($[T_{\text{obs}}, T_{\text{target\_start}}]$)**: A 7–14 day operational gap allowing teams to contact the customer before the churn period begins.
4. **Outcome window ($[T_{\text{target\_start}}, T_{\text{target\_end}}]$)**: A forward-looking period (e.g., next 30–60 days) where churn is officially recorded.
5. **Temporal out-of-time splits**: Training on historical time periods and evaluating on future periods to measure how model performance degrades over time.

---

## Data Cleaning & Exploratory Analysis

The raw dataset is relatively clean, but has one notable data hygiene issue:

### Handling Missing Values in `TotalCharges`
* Exactly 11 accounts have blank whitespace strings (`" "`) in the `TotalCharges` column.
* Inspecting these 11 rows reveals that all 11 have `tenure == 0`.
* These are brand-new customers who signed up during the current billing cycle and had not yet received their first bill. None of them had churned.
* **Cleaning rule**: I imputed `TotalCharges = 0.0` for accounts with `tenure == 0`. In the scikit-learn preprocessing pipeline, I also included a median imputer for numerical features to handle any missing values safely during production inference.

### Key Exploratory Observations

Exploring the data revealed several strong patterns associated with customer churn:

1. **Contract Type**:
   * Month-to-month subscribers have a **42.7%** churn rate.
   * One-year contract subscribers churn at **11.3%**.
   * Two-year contract subscribers churn at only **2.8%**.
   Customers committed to longer contracts churn far less frequently.

2. **Payment Method**:
   * Customers paying via Electronic Check experience a **45.3%** churn rate.
   * Customers using automated payment methods (credit card or bank transfer) churn at roughly **15% to 18%**.

3. **Internet Service**:
   * Fiber optic customers have a **41.9%** churn rate.
   * DSL customers churn at **19.0%**.
   * Customers with no internet service churn at **7.4%**.
   Fiber optic customers have higher monthly bills ($70–$100/month) and churn at more than double the rate of DSL customers.

4. **Tenure**:
   * Churn is heavily concentrated in the first 12 months (> 45% churn rate).
   * Customers who stay past 48 months have churn rates below 10%.

| Feature Category | High Churn Cohort | Churn Rate | Low Churn Cohort | Churn Rate |
| :--- | :--- | :---: | :--- | :---: |
| **Contract** | Month-to-month | 42.7% | Two-year contract | 2.8% |
| **Payment Method** | Electronic check | 45.3% | Credit card (auto) | 15.2% |
| **Internet Service** | Fiber optic | 41.9% | No internet | 7.4% |
| **Tech Support** | No Tech Support | 41.6% | Tech Support active | 15.2% |

All exploratory charts are generated programmatically and saved in `reports/figures/`:
* `reports/figures/target_distribution.png`
* `reports/figures/numerical_distributions.png`
* `reports/figures/churn_by_contract.png`
* `reports/figures/churn_by_payment_method.png`
* `reports/figures/correlation_heatmap.png`

---

## SQL Analysis & Invariant Verification

To practice combining relational database workflows with machine learning, I loaded the cleaned dataset into a local SQLite database (`data/churn_analysis.db`) using the schema in `sql/schema.sql`.

I wrote analytical SQL queries in `sql/analysis_queries.sql` to segment customers by contract, internet service, and payment methods. For example:

```sql
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

To ensure data integrity, I added an automated verification function in `src/churn_prediction/database.py` that asserts that SQL aggregations and Pandas calculations agree to within 0.01%.

---

## Preventing Data Leakage

Data leakage occurs when information from outside the training data is inadvertently used to train a model, producing overly optimistic performance estimates that collapse in real usage.

I used several safeguards to prevent data leakage:

1. **Split First**: I split the dataset into an 80% training set (5,634 rows) and a 20% held-out test set (1,409 rows) using stratified sampling (`random_state=42`) **before** running any imputation, scaling, or encoding.
2. **Encapsulated Preprocessing**: I bundled all transformations into a scikit-learn `ColumnTransformer` inside a `Pipeline`. Scalers, imputers, and encoders are fit strictly on training folds and only transform test or validation folds.
3. **Handling Unseen Categories**: The categorical encoder sets `handle_unknown='ignore'`, preventing runtime crashes or leakage if new categories appear during inference.
4. **Identifier Isolation**: The identifier column `customerID` is stripped before preprocessing and never exposed to the models.
5. **Validation-Only Threshold Tuning**: Optimal classification decision thresholds were chosen strictly using out-of-fold predictions from training cross-validation. The held-out test set was evaluated once at the very end.

---

## Feature Collinearity & Controlled Ablation

During exploratory analysis, I noticed a strong collinear relationship between the numeric features:
* `TotalCharges` has an empirical correlation of **0.99956** with `tenure * MonthlyCharges`.
* Correlation of `TotalCharges` with `tenure`: **0.8258**.
* Correlation of `TotalCharges` with `MonthlyCharges`: **0.6511**.

### The Suppressor Effect in Logistic Regression
When fitting Logistic Regression with all three features included, `MonthlyCharges` received a negative standardized coefficient ($\beta = -0.4916 \pm 0.1134$), despite having a positive marginal correlation (+0.193) with churn. 

This is a classic algebraic suppressor effect: once total lifetime charges and tenure are fixed, higher monthly charges mathematically imply that the customer has been with the company for fewer months.

### Tree Models and Correlated Features
Decision tree ensembles are also not immune to collinearity. While correlated features do not cause matrix inversion failures in trees, they compete against each other for split points. This leads to split masking, unstable tree structures across bootstrap resamples, and diluted feature importance scores.

### Controlled Ablation Experiment
To measure whether `TotalCharges` provides unique predictive information, I ran a 5-fold cross-validation ablation experiment on the training set:

| Feature Set | Model | CV PR-AUC (Mean ± Std) | CV ROC-AUC (Mean ± Std) |
| :--- | :--- | :---: | :---: |
| **With TotalCharges** | Logistic Regression | 0.6615 ± 0.0195 | 0.8462 ± 0.0126 |
| **With TotalCharges** | Random Forest | 0.6650 ± 0.0203 | 0.8475 ± 0.0094 |
| **With TotalCharges** | Gradient Boosting | 0.6643 ± 0.0189 | 0.8482 ± 0.0111 |
| **Without TotalCharges** | Logistic Regression | 0.6597 ± 0.0188 | 0.8449 ± 0.0122 |
| **Without TotalCharges** | Random Forest | 0.6566 ± 0.0194 | 0.8443 ± 0.0092 |
| **Without TotalCharges** | Gradient Boosting | 0.6647 ± 0.0180 | 0.8465 ± 0.0099 |

* **Observation**: Removing `TotalCharges` results in a very small change in CV PR-AUC (-0.0084 for Random Forest, +0.0004 for Gradient Boosting, and -0.0018 for Logistic Regression). All changes are smaller than 1 cross-validation standard deviation ($\approx 0.019$).
* This shows that tree and boosting models can effectively reconstruct the predictive signal using `tenure` and `MonthlyCharges` directly.

---

## Why I Evaluated Three Models

I chose three distinct classification algorithms to understand how model complexity affects performance on tabular churn data:

1. **Logistic Regression (Baseline)**:
   * A linear model that provides a fast, interpretable benchmark.
   * Directly outputs log-odds, making it easy to explain individual feature impacts.
   * Serves as an Occam's razor test: if complex models cannot meaningfully beat linear regression, the simpler model should be preferred.

2. **Random Forest**:
   * An ensemble of de-correlated decision trees trained via bagging (bootstrap aggregation).
   * Naturally captures non-linear relationships and high-order interactions (e.g., contract type interacting with tenure) without requiring manual interaction terms.
   * Robust against outliers and less prone to severe overfitting than unconstrained decision trees.

3. **Gradient Boosting**:
   * A sequential ensemble that fits shallow decision trees to the residual errors of preceding trees.
   * Often considered the strongest out-of-the-box family of algorithms for tabular datasets.

---

## Evaluation Metrics & Why Accuracy is Misleading

Because only 26.54% of customers in this dataset churned, a trivial model that predicts "No Churn" for every customer would achieve **73.46% accuracy**, despite identifying zero churners.

To evaluate models appropriately, I used:

* **PR-AUC (Precision-Recall Area Under the Curve / Average Precision)**:
  * Chosen as the **primary model selection metric**.
  * Measures the trade-off between precision (how many flagged customers actually churned) and recall (how many total churners were successfully identified).
  * Evaluates performance specifically on the minority positive class without being inflated by a large number of true negatives.
  * Computed using scikit-learn's `average_precision_score`:
    $$\text{AP} = \sum_{n} (R_n - R_{n-1}) P_n$$
* **ROC-AUC**: Measures ranking quality across all thresholds, useful for comparing overall separation.
* **Precision, Recall, and F1-Score**: Evaluated at specific decision thresholds to understand operational trade-offs.

---

## Measured Results & Model Comparison

All models were evaluated on the training set using 5-fold cross-validation and tested once on the held-out test set:

| Model | CV PR-AUC (Mean ± Std) | CV ROC-AUC | CV F1 | CV Precision | CV Recall | Test PR-AUC | Test ROC-AUC | Test F1 | Test Precision | Test Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.6650 ± 0.0203** | 0.8475 | 0.5828 | **0.6739** | 0.5137 | 0.6517 | 0.8405 | 0.5836 | **0.6761** | 0.5134 |
| **Gradient Boosting** | 0.6643 ± 0.0189 | **0.8482** | 0.5833 | 0.6616 | 0.5224 | **0.6596** | **0.8465** | 0.5774 | 0.6598 | 0.5134 |
| **Logistic Regression** | 0.6615 ± 0.0195 | 0.8462 | **0.5932** | 0.6530 | **0.5445** | 0.6336 | 0.8420 | **0.6040** | 0.6572 | **0.5588** |

### Statistical Comparison: Nadeau–Bengio Corrected Resampled $t$-Test

Standard paired $t$-tests across cross-validation folds violate the independence assumption because the training sets overlap by 80% across folds. This underestimates the variance and inflates Type I errors (falsely claiming statistical significance).

To evaluate whether the candidate models actually outperformed Logistic Regression, I implemented the **Nadeau–Bengio corrected resampled $t$-test** (Nadeau & Bengio 2003; Bouckaert & Frank 2004) across a 3-repeat, 5-fold cross-validation setup ($R = 15$ paired evaluations, $df = 14$):

* **Ratio**: $\frac{n_{\text{test}}}{n_{\text{train}}} = \frac{0.20}{0.80} = 0.2500$
* **Variance Correction Factor**: $\left(\frac{1}{R} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) = \left(\frac{1}{15} + 0.25\right) \approx 0.3167$
* **Corrected Test Statistic**:
  $$t_{\text{corrected}} = \frac{\bar{d}}{\sqrt{s^2 \left(\frac{1}{R} + \frac{n_{\text{test}}}{n_{\text{train}}}\right)}}$$

#### Statistical Test Findings ($R=15, df=14$):
* **Random Forest vs. Logistic Regression**: Mean difference $\bar{d} = +0.0044$, $t = 0.955$, $p = 0.3557$.
* **Gradient Boosting vs. Logistic Regression**: Mean difference $\bar{d} = +0.0049$, $t = 0.815$, $p = 0.4285$.

> **What this means**: A non-significant $p$-value ($p > 0.05$) does not prove that the models are identical, but it shows that the observed mean differences ($+0.0044$ and $+0.0049$) are well within the fold-to-fold resampling variability ($\pm 0.020$). All three models perform similarly on this dataset.

### Why I Selected Random Forest
While Logistic Regression is simpler and achieves slightly higher recall at the default 0.50 threshold, I retained **Random Forest** because it achieved higher cross-validation precision (**67.39% vs 65.30%**). When marketing outreach budgets are constrained, higher precision reduces costly false-alarm contacts.

### Held-Out Test Confusion Matrix (Random Forest @ Threshold 0.50)
* **True Negatives (TN)**: 943 (correctly predicted retained)
* **False Positives (FP)**: 92 (predicted churn, but actually stayed)
* **False Negatives (FN)**: 182 (predicted retained, but actually churned)
* **True Positives (TP)**: 192 (correctly predicted churn)
* **Test Precision**: $67.61\%$
* **Test Recall**: $51.34\%$
* **Test F1**: $58.36\%$

---

## Decision Thresholds & Business Utility

Machine learning models produce continuous probability estimates, not binary decisions. The default 0.50 decision threshold is arbitrary and rarely aligns with business objectives.

### Financial Utility: Incremental Campaign Value vs. No Outreach
To evaluate the financial impact of different decision thresholds, I formulated an expected-value objective: **the incremental net financial value of running a targeted retention campaign compared to doing nothing (no outreach)**:

$$\Delta \text{Value} = (\text{TP} \times \eta \times V) - (\text{TP} + \text{FP}) \times C_{\text{outreach}}$$

Where:
* $V$: Value preserved if a churner is successfully retained (e.g., $120).
* $\eta$: Retention offer conversion rate (e.g., 30% of contacted churners agree to stay).
* $C_{\text{outreach}}$: Unit cost of contacting a customer and providing an incentive (e.g., $20).
* $\text{TP} + \text{FP}$: Total customers contacted.

**Why False Negatives cancel out**:
* In the baseline scenario (no outreach), all true churners ($\text{TP} + \text{FN}$) depart: $\text{Loss}_{\text{base}} = -(\text{TP} + \text{FN}) \times V$.
* In the campaign scenario, uncontacted churners ($\text{FN}$) and unpersuaded churners ($\text{TP} \times (1 - \eta)$) depart: $\text{Loss}_{\text{camp}} = -[\text{TP}(1 - \eta) + \text{FN}] \times V - (\text{TP} + \text{FP}) C_{\text{outreach}}$.
* Subtracting baseline losses from campaign results cancels out the $\text{FN} \times V$ term, showing the true net incremental benefit of the campaign.

### Threshold Trade-offs on Training Out-of-Fold Predictions
Using Moderate assumptions ($V = \$120, \eta = 30\%, C_{\text{outreach}} = \$20$):

```text
 threshold  precision  recall     f1  contacted  churners_tp  false_alarms_fp  incremental_value
      0.20     0.4619  0.8729 0.6042       2825         1305             1520          -$9,520.0
      0.25     0.5028  0.8334 0.6272       2478         1246             1232          -$4,704.0
      0.30     0.5335  0.7779 0.6329       2180         1163             1017          -$1,732.0
      0.35     0.5649  0.7251 0.6350       1919         1084              835            +$644.0
      0.40     0.6051  0.6528 0.6281       1613          976              637          +$2,876.0
      0.45     0.6370  0.5846 0.6097       1372          874              498          +$4,024.0
      0.50     0.6731  0.5137 0.5827       1141          768              373          +$4,828.0
      0.55     0.7065  0.4187 0.5258        886          626              260          +$4,816.0
      0.60     0.7282  0.3244 0.4489        666          485              181          +$4,140.0
      0.70     0.8338  0.1846 0.3023        331          276               55          +$3,316.0
```

### Multi-Scenario Sensitivity Analysis

Because retention parameters vary across businesses, I evaluated how campaign value changes across Conservative, Moderate, and Optimistic scenarios:

```text
    scenario success_rate unit_cost cust_value threshold contacted churners_tp false_alarms incremental_value
Conservative          15%       $25       $100      0.25      2478        1246         1232        -$43,260.0
Conservative          15%       $25       $100      0.35      1919        1084          835        -$31,715.0
Conservative          15%       $25       $100      0.50      1141         768          373        -$17,005.0
    Moderate          30%       $20       $120      0.25      2478        1246         1232         -$4,704.0
    Moderate          30%       $20       $120      0.35      1919        1084          835           +$644.0
    Moderate          30%       $20       $120      0.50      1141         768          373         +$4,828.0
  Optimistic          50%       $15       $150      0.25      2478        1246         1232        +$56,280.0
  Optimistic          50%       $15       $150      0.35      1919        1084          835        +$52,515.0
  Optimistic          50%       $15       $150      0.50      1141         768          373        +$40,485.0
```

> **Takeaway**: When contact costs are non-trivial and conversion rates are realistic ($\le 30\%$), low decision thresholds lose money because outreach costs exceed preserved value. At threshold 0.50, higher precision ($67.31\%$) avoids 859 false-alarm contacts relative to threshold 0.25, maximizing net campaign value ($+\$4,828.0$). If outreach costs are negligible and success rates are high, lowering the threshold to 0.25 captures more churners and maximizes value.

---

## Model Interpretation & Error Analysis

### 1. Permutation Importance (Held-Out Test Set)
I computed permutation feature importance on the held-out test set by shuffling each feature and measuring the degradation in PR-AUC:
1. `Contract`: Shuffling degrades PR-AUC by **0.0686 ± 0.0037**.
2. `tenure`: Degradation of **0.0653 ± 0.0070**.
3. `TotalCharges`: Degradation of **0.0327 ± 0.0070**.
4. `InternetService`: Degradation of **0.0317 ± 0.0139**.
5. `TechSupport`: Degradation of **0.0234 ± 0.0051**.

`Contract` and `tenure` are by far the strongest drivers of churn prediction.

### 2. Error Analysis: Where Does the Model Fail?
To understand the model's limitations, I inspected the false positives and false negatives on the test set:

* **False Positives (92 accounts at 0.50 threshold)**:
  * Customers with month-to-month contracts and fiber optic service who had high bills, triggering risk flags.
  * However, many of these customers had family plans (`Partner=Yes`, `Dependents=Yes`) or multiple services bundled. The high friction of switching an entire household's services likely kept them retained despite the high monthly bill.
* **False Negatives (182 accounts at 0.50 threshold)**:
  * Customers with moderate tenure (12–24 months) on annual contracts who unexpectedly churned.
  * These churn events were likely triggered by external factors not captured in the dataset: localized internet outages, a competitor offering promotional fiber rates in their area, or moving to an uncovered geographic region.

---

## Inference Pipeline & Example Usage

The trained pipeline is serialized to `models/best_model.joblib`. The `ChurnPredictor` class loads this pipeline and provides an interface for predicting churn risk on single customer dictionaries or Pandas DataFrames:

```python
from churn_prediction.predict import ChurnPredictor

predictor = ChurnPredictor()

sample_customer = {
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

result = predictor.predict(sample_customer, threshold=0.50)
print(result[["churn_probability", "predicted_churn", "risk_tier", "recommended_action"]])
```

Output:
```text
   churn_probability  predicted_churn risk_tier                                 recommended_action
0           0.714588                1      High  High churn risk. Schedule proactive retention call.
```

---

## Project Structure

```text
customer-churn-prediction/
├── README.md                           # Project overview, methodology, and results
├── requirements.txt                    # Pinned package dependencies
├── pyproject.toml                      # Package metadata and test configurations
├── pytest.ini                          # Pytest discovery and pythonpath settings
├── LICENSE                             # MIT License
├── train.py                            # End-to-end pipeline (data, CV, evaluation, figures)
├── app.py                              # Interactive Streamlit dashboard application
├── configs/
│   └── config.yaml                     # Centralized paths, columns, and hyperparameters
├── data/
│   ├── raw/                            # Raw CSV (ignored in git, downloaded automatically)
│   ├── processed/                      # Preprocessed splits (ignored in git)
│   └── churn_analysis.db               # SQLite database (generated locally)
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # 7-step EDA learning and experimentation notebook
├── src/
│   └── churn_prediction/
│       ├── __init__.py                 # Package declaration
│       ├── config.py                   # YAML configuration loader
│       ├── data_loader.py              # Ingestion, schema validation, and hygiene
│       ├── database.py                 # SQLite ingestion and SQL-vs-Pandas verification
│       ├── preprocessing.py            # Leakage-free ColumnTransformer & data splitting
│       ├── train.py                    # 5-fold CV, Nadeau-Bengio t-test, ablation, selection
│       ├── evaluate.py                 # Metrics, threshold trade-offs, and sensitivity
│       ├── interpret.py                # Tree importances, odds ratios, error analysis
│       ├── eda.py                      # Figure generation
│       └── predict.py                  # Inference class with input validation
├── sql/
│   ├── schema.sql                      # SQLite DDL table definition
│   └── analysis_queries.sql            # Segmentation SQL queries
├── reports/
│   ├── model_comparison.csv            # Empirical metrics across CV and Test
│   └── figures/                        # Generated diagnostic figures
│       ├── target_distribution.png
│       ├── numerical_distributions.png
│       ├── churn_by_contract.png
│       ├── churn_by_payment_method.png
│       ├── correlation_heatmap.png
│       ├── confusion_matrix.png
│       ├── pr_roc_curves.png
│       └── feature_importance.png
├── models/
│   └── best_model.joblib               # Fitted pipeline artifact (generated locally)
├── dashboard/
│   ├── __init__.py
│   ├── data_service.py                 # Dataset inspection, caching, and hygiene
│   ├── model_service.py                # Performance reports and inference executor
│   ├── analytics_service.py            # Read-only SQLite query engine
│   └── components.py                   # Reusable UI widgets and custom dark theme
└── tests/
    ├── test_data.py                    # Schema validation, missing values, hygiene
    ├── test_preprocessing.py           # Split safety, unseen categories, scaling
    ├── test_training.py                # Pipeline fitting, candidate models, CV sanity
    ├── test_prediction.py              # Predictor validation, probability bounds
    ├── test_evaluation.py              # Statistical t-tests, threshold trade-offs, campaign math
    └── test_dashboard.py               # Dashboard services, read-only safety, presets
```

---

## How to Run the Project

### 1. Clone the repository and set up environment
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate # Linux / macOS

pip install -r requirements.txt
pip install -e .
```

### 2. Run the automated test suite
The test suite contains 38 unit and integration tests verifying schema validation, leakage prevention, statistical test logic, financial utility math, inference behavior, and dashboard services:
```bash
python -m pytest
```

### 3. Run the complete pipeline
To download the dataset (if missing), ingest into SQLite, train all models across 5-fold cross-validation, run the ablation study, execute the Nadeau–Bengio statistical test, evaluate on the test set, and generate all figures:
```bash
python train.py
```

### 4. Launch the interactive Streamlit dashboard
To start the local web application:
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

The dashboard is structured into 5 streamlined pages designed for both quick prediction and data exploration:
- **Overview**: Instant answer to project objectives, key dataset statistics (7,043 accounts, 26.5% churn rate), visual customer balance, and direct action buttons.
- **Predict Churn**: Intuitive 3-section input form (Customer Profile, Services, Billing & Contract) with illustrative example presets, optional collapsed threshold settings, and transparent risk tier estimation.
- **Explore Data**: 4 core visual relationships (churn by contract, payment method, tenure, and monthly charges) with an interactive, filterable data preview.
- **Model Performance**: Approachable model comparison table, selectable metric chart, and held-out test confusion matrix, with advanced statistical test details collapsed in expanders.
- **Reports & SQL**: Combined business SQL questions with safe read-only database execution alongside technical project governance documentation.

### 5. Explore the notebook
The exploratory analysis notebook is organized in a logical 7-step learning sequence:
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## What Did Not Work & What I Would Improve Next

Documenting what didn't work and recognizing the boundaries of the analysis is just as important as reporting metrics:

1. **Static Cross-Sectional Data Cannot Predict Future Churn**:
   * The biggest limitation of this project is the absence of timestamps in the IBM Telco dataset.
   * In a real operational environment, you need time-stamped event streams and out-of-time evaluation to measure how well the model handles seasonal shifts and changing customer behavior.

2. **Complex Models Did Not Substantially Outperform the Baseline**:
   * Gradient Boosting and Random Forest achieved higher CV PR-AUC than Logistic Regression by only $+0.0049$ and $+0.0044$.
   * The Nadeau–Bengio test confirmed that these differences are well within normal resampling variance ($p > 0.35$).
   * For organizations prioritizing transparency, low latency, and regulatory explainability, Logistic Regression remains a strong contender.

3. **Collinearity Distorts Feature Interpretation**:
   * The near-perfect correlation between `TotalCharges` and `tenure * MonthlyCharges` ($\rho = 0.99956$) causes suppressor effects in linear models and split-masking in tree models.
   * Omitting `TotalCharges` hardly degraded performance, showing that the model can rely on tenure and monthly charges directly.

4. **Retention Success Rates Are Unknown Without an A/B Test**:
   * Calculating financial utility requires an assumed customer intervention success rate ($\eta$).
   * Simply identifying high-risk churners does not mean they can be saved. Some customers have already decided to leave (e.g., relocating).
   * **Next Step**: In a real company, I would implement **uplift modeling** (e.g., Causal ML or a two-model approach) to target customers who are *persuadable* (those who only stay if they receive an outreach), rather than wasting budget on lost causes or customers who would have stayed anyway.

5. **Future Technical Improvements**:
   * Conduct systematic hyperparameter tuning using Bayesian optimization (`optuna`).
   * Add probability calibration (`CalibratedClassifierCV`) to ensure predicted probabilities match true observed empirical frequencies.
   * Containerize the inference pipeline with Docker and add a lightweight FastAPI serving layer.

---

## License & Attribution
* The code, pipeline, tests, and documentation in this repository are licensed under the [MIT License](LICENSE).
* The underlying IBM Telco Customer Churn dataset is property of IBM Corporation and is used here for educational and portfolio demonstration.
#   C u s t o m e r - c h u r n - p r e d i c t i o n  
 