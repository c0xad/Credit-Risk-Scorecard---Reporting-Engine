-- Credit Risk Scorecard Database Schema

-- Customer Profile Table
CREATE TABLE IF NOT EXISTS customer_profile (
    customer_id VARCHAR(20) PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(10),
    marital_status VARCHAR(20),
    dependents INT,
    education VARCHAR(50),
    employment_status VARCHAR(30) NOT NULL,
    employment_length_years INT,
    annual_income DECIMAL(15, 2) NOT NULL,
    address VARCHAR(200),
    city VARCHAR(50),
    state VARCHAR(20),
    postal_code VARCHAR(20),
    country VARCHAR(30),
    phone VARCHAR(20),
    email VARCHAR(100),
    registration_date DATE NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Credit Bureau Data
CREATE TABLE IF NOT EXISTS credit_bureau_data (
    report_id VARCHAR(36) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    credit_score INT NOT NULL,
    total_accounts INT,
    open_accounts INT,
    closed_accounts INT,
    delinquent_accounts INT,
    total_balance DECIMAL(15, 2),
    total_credit_limit DECIMAL(15, 2),
    credit_utilization_ratio DECIMAL(5, 2),
    length_of_credit_history_months INT,
    hard_inquiries_last_12m INT,
    collections_last_12m INT,
    public_records INT,
    bankruptcy_count INT,
    foreclosure_count INT,
    source VARCHAR(50),
    FOREIGN KEY (customer_id) REFERENCES customer_profile(customer_id),
    INDEX idx_credit_bureau_customer (customer_id, report_date)
);

-- Loan Application
CREATE TABLE IF NOT EXISTS loan_application (
    application_id VARCHAR(36) PRIMARY KEY, 
    customer_id VARCHAR(20) NOT NULL,
    application_date DATE NOT NULL,
    loan_type VARCHAR(30) NOT NULL,
    loan_purpose VARCHAR(50),
    requested_amount DECIMAL(15, 2) NOT NULL,
    term_months INT NOT NULL,
    interest_rate DECIMAL(5, 2),
    status VARCHAR(20) NOT NULL,
    decision_date DATE,
    decision_reason VARCHAR(100),
    underwriter_id VARCHAR(20),
    approval_score INT,
    approved_amount DECIMAL(15, 2),
    FOREIGN KEY (customer_id) REFERENCES customer_profile(customer_id),
    INDEX idx_loan_application_customer (customer_id)
);

-- Loan Account
CREATE TABLE IF NOT EXISTS loan_account (
    loan_id VARCHAR(36) PRIMARY KEY,
    application_id VARCHAR(36) NOT NULL,
    customer_id VARCHAR(20) NOT NULL,
    origination_date DATE NOT NULL,
    loan_amount DECIMAL(15, 2) NOT NULL,
    term_months INT NOT NULL,
    interest_rate DECIMAL(5, 2) NOT NULL,
    monthly_payment DECIMAL(10, 2) NOT NULL,
    remaining_balance DECIMAL(15, 2) NOT NULL,
    status VARCHAR(20) NOT NULL,
    maturity_date DATE NOT NULL,
    last_payment_date DATE,
    next_payment_date DATE,
    FOREIGN KEY (application_id) REFERENCES loan_application(application_id),
    FOREIGN KEY (customer_id) REFERENCES customer_profile(customer_id),
    INDEX idx_loan_account_customer (customer_id)
);

-- Payment History
CREATE TABLE IF NOT EXISTS payment_history (
    payment_id VARCHAR(36) PRIMARY KEY,
    loan_id VARCHAR(36) NOT NULL,
    payment_date DATE NOT NULL,
    payment_amount DECIMAL(10, 2) NOT NULL,
    principal_amount DECIMAL(10, 2) NOT NULL,
    interest_amount DECIMAL(10, 2) NOT NULL,
    fees_amount DECIMAL(10, 2) DEFAULT 0,
    remaining_balance DECIMAL(15, 2) NOT NULL,
    payment_status VARCHAR(20) NOT NULL,
    days_past_due INT DEFAULT 0,
    payment_method VARCHAR(30),
    FOREIGN KEY (loan_id) REFERENCES loan_account(loan_id),
    INDEX idx_payment_history_loan (loan_id, payment_date)
);

-- Default and Delinquency
CREATE TABLE IF NOT EXISTS loan_delinquency (
    delinquency_id VARCHAR(36) PRIMARY KEY,
    loan_id VARCHAR(36) NOT NULL,
    customer_id VARCHAR(20) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    delinquency_status VARCHAR(20) NOT NULL,
    days_past_due INT NOT NULL,
    amount_past_due DECIMAL(10, 2) NOT NULL,
    resolution_type VARCHAR(30),
    resolution_date DATE,
    FOREIGN KEY (loan_id) REFERENCES loan_account(loan_id),
    FOREIGN KEY (customer_id) REFERENCES customer_profile(customer_id),
    INDEX idx_delinquency_customer (customer_id),
    INDEX idx_delinquency_loan (loan_id)
);

-- Risk Models
CREATE TABLE IF NOT EXISTS risk_model (
    model_id VARCHAR(36) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    description TEXT,
    training_date DATE NOT NULL,
    production_date DATE,
    status VARCHAR(20) NOT NULL,
    features_list TEXT,
    performance_metrics TEXT,
    creator VARCHAR(50),
    UNIQUE KEY idx_model_version (model_name, model_version)
);

-- Risk Scores
CREATE TABLE IF NOT EXISTS risk_score (
    score_id VARCHAR(36) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    model_id VARCHAR(36) NOT NULL,
    application_id VARCHAR(36),
    loan_id VARCHAR(36),
    score_date DATE NOT NULL,
    score_value DECIMAL(10, 2) NOT NULL,
    probability_of_default DECIMAL(10, 6),
    risk_band VARCHAR(20),
    score_factors TEXT,
    FOREIGN KEY (customer_id) REFERENCES customer_profile(customer_id),
    FOREIGN KEY (model_id) REFERENCES risk_model(model_id),
    FOREIGN KEY (application_id) REFERENCES loan_application(application_id),
    FOREIGN KEY (loan_id) REFERENCES loan_account(loan_id),
    INDEX idx_risk_score_customer (customer_id),
    INDEX idx_risk_score_model_date (model_id, score_date)
);

-- Data Quality
CREATE TABLE IF NOT EXISTS data_quality (
    quality_id VARCHAR(36) PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    column_name VARCHAR(50) NOT NULL,
    check_date DATE NOT NULL,
    record_count INT NOT NULL,
    null_count INT,
    outlier_count INT,
    min_value VARCHAR(100),
    max_value VARCHAR(100),
    mean_value VARCHAR(100),
    median_value VARCHAR(100),
    quality_score DECIMAL(5, 2),
    issues_detected TEXT,
    INDEX idx_data_quality_table_col (table_name, column_name, check_date)
);

-- Model Monitoring
CREATE TABLE IF NOT EXISTS model_monitoring (
    monitoring_id VARCHAR(36) PRIMARY KEY,
    model_id VARCHAR(36) NOT NULL,
    monitoring_date DATE NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    total_customers INT,
    total_applications INT,
    total_loans INT,
    population_stability_index DECIMAL(10, 6),
    auc_roc DECIMAL(10, 6),
    ks_statistic DECIMAL(10, 6),
    gini_coefficient DECIMAL(10, 6),
    average_score DECIMAL(10, 2),
    median_score DECIMAL(10, 2),
    variable_drift_metrics TEXT,
    alert_status VARCHAR(20),
    alert_message TEXT,
    FOREIGN KEY (model_id) REFERENCES risk_model(model_id),
    INDEX idx_model_monitoring_date (model_id, monitoring_date)
);

-- Scorecard Reports
CREATE TABLE IF NOT EXISTS scorecard_report (
    report_id VARCHAR(36) PRIMARY KEY,
    report_name VARCHAR(100) NOT NULL,
    report_type VARCHAR(50) NOT NULL,
    report_date DATE NOT NULL,
    model_id VARCHAR(36),
    period_start DATE,
    period_end DATE,
    report_content TEXT,
    report_format VARCHAR(20),
    created_by VARCHAR(50),
    notes TEXT,
    FOREIGN KEY (model_id) REFERENCES risk_model(model_id),
    INDEX idx_scorecard_report_date (report_date)
);

-- User queries for underwriters
CREATE TABLE IF NOT EXISTS user_query (
    query_id VARCHAR(36) PRIMARY KEY,
    query_name VARCHAR(100) NOT NULL,
    description TEXT,
    query_sql TEXT NOT NULL,
    parameters TEXT,
    created_by VARCHAR(50),
    created_date DATE NOT NULL,
    last_used_date DATE,
    usage_count INT DEFAULT 0,
    is_shared BOOLEAN DEFAULT FALSE
);

-- Create views for common analysis

-- Current portfolio risk overview
CREATE OR REPLACE VIEW vw_portfolio_risk_overview AS
SELECT 
    rm.model_name, 
    rs.risk_band, 
    COUNT(DISTINCT la.loan_id) AS loan_count,
    SUM(la.remaining_balance) AS total_exposure,
    AVG(rs.probability_of_default) AS avg_pd,
    SUM(la.remaining_balance * rs.probability_of_default) AS expected_loss
FROM loan_account la
JOIN risk_score rs ON la.loan_id = rs.loan_id AND la.customer_id = rs.customer_id
JOIN risk_model rm ON rs.model_id = rm.model_id
WHERE la.status = 'Active' 
AND rs.score_date = (
    SELECT MAX(score_date) 
    FROM risk_score 
    WHERE loan_id = la.loan_id
)
GROUP BY rm.model_name, rs.risk_band
ORDER BY rm.model_name, avg_pd DESC;

-- Customer risk profile
CREATE OR REPLACE VIEW vw_customer_risk_profile AS
SELECT 
    cp.customer_id,
    cp.first_name,
    cp.last_name,
    cp.annual_income,
    cbd.credit_score,
    MAX(rs.score_date) AS last_score_date,
    rs.score_value AS latest_score,
    rs.probability_of_default AS latest_pd,
    rs.risk_band,
    COUNT(DISTINCT la.loan_id) AS active_loans,
    SUM(la.remaining_balance) AS total_debt,
    SUM(la.monthly_payment) AS total_monthly_payment,
    (SUM(la.monthly_payment) / cp.annual_income * 12) * 100 AS debt_to_income_ratio
FROM customer_profile cp
LEFT JOIN credit_bureau_data cbd ON cp.customer_id = cbd.customer_id AND cbd.report_date = (
    SELECT MAX(report_date) FROM credit_bureau_data WHERE customer_id = cp.customer_id
)
LEFT JOIN risk_score rs ON cp.customer_id = rs.customer_id AND rs.score_date = (
    SELECT MAX(score_date) FROM risk_score WHERE customer_id = cp.customer_id
)
LEFT JOIN loan_account la ON cp.customer_id = la.customer_id AND la.status = 'Active'
GROUP BY cp.customer_id, cp.first_name, cp.last_name, cp.annual_income, 
         cbd.credit_score, rs.score_value, rs.probability_of_default, rs.risk_band; 