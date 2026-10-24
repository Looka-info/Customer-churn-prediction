-- Analysis Query 1: Total Customers Count
-- name: total_customers
SELECT COUNT(*) AS total_customers
FROM telco_churn;

-- Analysis Query 2: Churned and Retained Customer Counts
-- name: churn_counts
SELECT
    churn,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM telco_churn), 2) AS percentage
FROM telco_churn
GROUP BY churn;

-- Analysis Query 3: Overall Churn Rate
-- name: overall_churn_rate
SELECT
    COUNT(*) AS total_customers,
    SUM(churn_numeric) AS churned_customers,
    ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent
FROM telco_churn;

-- Analysis Query 4: Churn Rate by Contract Type
-- name: churn_by_contract
SELECT
    contract,
    COUNT(*) AS total_customers,
    SUM(churn_numeric) AS churned_customers,
    ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM telco_churn
GROUP BY contract
ORDER BY churn_rate_percent DESC;

-- Analysis Query 5: Churn Rate by Payment Method
-- name: churn_by_payment_method
SELECT
    payment_method,
    COUNT(*) AS total_customers,
    SUM(churn_numeric) AS churned_customers,
    ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent
FROM telco_churn
GROUP BY payment_method
ORDER BY churn_rate_percent DESC;

-- Analysis Query 6: Churn Rate by Tenure Group
-- name: churn_by_tenure_group
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
ORDER BY MIN(tenure) ASC;

-- Analysis Query 7: Average Monthly Charges and Total Charges by Churn Status
-- name: charges_by_churn_status
SELECT
    churn,
    COUNT(*) AS customer_count,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(total_charges), 2) AS avg_total_charges
FROM telco_churn
GROUP BY churn;

-- Analysis Query 8: Churn Rates across Contract Type and Internet Service Combinations
-- name: churn_by_contract_and_internet
SELECT
    contract,
    internet_service,
    COUNT(*) AS total_customers,
    SUM(churn_numeric) AS churned_customers,
    ROUND(AVG(churn_numeric) * 100.0, 2) AS churn_rate_percent
FROM telco_churn
GROUP BY contract, internet_service
ORDER BY churn_rate_percent DESC;
