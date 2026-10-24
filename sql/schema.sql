-- Schema for Telco Customer Churn SQLite Database
DROP TABLE IF EXISTS telco_churn;

CREATE TABLE telco_churn (
    customer_id TEXT PRIMARY KEY,
    gender TEXT NOT NULL,
    senior_citizen INTEGER NOT NULL,
    partner TEXT NOT NULL,
    dependents TEXT NOT NULL,
    tenure INTEGER NOT NULL,
    phone_service TEXT NOT NULL,
    multiple_lines TEXT NOT NULL,
    internet_service TEXT NOT NULL,
    online_security TEXT NOT NULL,
    online_backup TEXT NOT NULL,
    device_protection TEXT NOT NULL,
    tech_support TEXT NOT NULL,
    streaming_tv TEXT NOT NULL,
    streaming_movies TEXT NOT NULL,
    contract TEXT NOT NULL,
    paperless_billing TEXT NOT NULL,
    payment_method TEXT NOT NULL,
    monthly_charges REAL NOT NULL,
    total_charges REAL,
    churn TEXT NOT NULL,
    churn_numeric INTEGER NOT NULL
);

CREATE INDEX idx_telco_churn_contract ON telco_churn(contract);
CREATE INDEX idx_telco_churn_internet ON telco_churn(internet_service);
CREATE INDEX idx_telco_churn_payment ON telco_churn(payment_method);
CREATE INDEX idx_telco_churn_status ON telco_churn(churn);
