# Credit Risk Scorecard & Reporting Engine

## Overview

This project provides a comprehensive framework for building, evaluating, deploying, and monitoring credit risk models. It includes modules for database setup, data processing, model training (supporting various algorithms), prediction scoring, model performance monitoring, and automated report generation. The system is designed to work with SQL databases and utilizes Python libraries like Scikit-learn, Pandas, XGBoost, Jinja2, and (optionally) WeasyPrint.

The primary goal is to predict the probability of default (PD) for loan applicants and provide tools for ongoing assessment of model health and portfolio risk.

## Features

*   **Database Management**:
    *   SQL schema definitions (`database/schema.sql`) for customer profiles, loan applications, accounts, credit bureau data, payments, delinquencies, models, scores, and monitoring data.
    *   Database setup script (`database/setup_db.py`) supporting MySQL, PostgreSQL, and **SQLite** (as a fallback or for local development).
    *   Utility functions (`utils/database.py`) for simplified database interaction.
*   **Data Processing**:
    *   Scripts for cleaning, preprocessing, and feature engineering (Conceptual - `data_processing/` directory planned).
    *   Sample data generation for development and testing (`data_processing/generate_sample_data.py` - *Note: Currently may need implementation*).
*   **Machine Learning Models**:
    *   Train credit risk models using various algorithms (`models/train_model.py`). Supported types include:
        *   Logistic Regression (`logistic`)
        *   Gradient Boosting Machines (`gbm`)
        *   XGBoost (`xgboost`)
        *   Random Forest (`rf`)
    *   Hyperparameter tuning using GridSearchCV.
    *   Model and preprocessor persistence using `joblib`.
    *   Feature importance calculation.
    *   **Fallback**: Can train using synthetically generated data if database tables are unavailable.
*   **Prediction Engine**:
    *   Batch or single-instance prediction using trained models (`models/predict.py`).
    *   Calculates Probability of Default (PD) and assigns risk scores/bands.
    *   Saves prediction results to the database or CSV files.
    *   **Fallback**: Can score using synthetically generated customer data if database tables are unavailable.
*   **Model Monitoring**:
    *   Track model performance metrics over time (`monitoring/model_monitoring.py`).
    *   Calculates metrics like AUC-ROC, KS Statistic, Gini Coefficient, and Population Stability Index (PSI).
    *   Detects potential model drift or performance degradation.
    *   Saves monitoring results to the database or JSON files.
    *   Generates performance trend charts.
    *   **Fallback**: Can run using synthetically generated score data if database tables are unavailable.
*   **Automated Reporting**:
    *   Generate comprehensive reports using Jinja2 templates (`reporting/generate_reports.py` and `reporting/templates/`).
    *   Supported report types: Scorecard (more planned).
    *   Outputs reports in **HTML** and optionally **PDF** format (requires WeasyPrint dependencies).
    *   Includes performance summaries, charts (feature importance, stability, segmentation), and risk band distributions.
    *   **Fallback**: Can generate reports using data loaded from fallback files (monitoring JSONs) and synthetic data if database tables are unavailable.
*   **Configuration**:
    *   Uses a `.env` file (`config/.env`) for database credentials and other settings.

## Project Structure

```
├── config/             # Configuration files (e.g., .env)
├── database/           # SQL schema, setup scripts, SQLite storage
│   ├── sqlite/         # Default location for SQLite DB file
│   ├── schema.sql      # Main database schema
│   └── setup_db.py     # Script to create DB and tables
├── data_processing/    # Data cleaning, feature engineering, sample data generation (Conceptual)
├── models/             # Model training, prediction, saved models
│   ├── saved/          # Directory where trained models are stored
│   ├── predict.py      # Script for making predictions
│   └── train_model.py  # Script for training models
├── monitoring/         # Model and data monitoring tools
│   ├── reports/        # Default location for monitoring JSON reports (fallback)
│   └── model_monitoring.py # Script to run model monitoring
├── reporting/          # Report generation scripts, templates, and outputs
│   ├── templates/      # Jinja2 HTML templates for reports
│   ├── charts/         # Temporary storage for generated chart images
│   ├── outputs/        # Default location for generated HTML/PDF reports
│   └── generate_reports.py # Script to generate reports
├── utils/              # Utility functions (e.g., database connection)
│   └── database.py     # Database helper functions
├── requirements.txt    # Python package dependencies
└── README.md           # This file
```

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd Credit-Risk-Scorecard---Reporting-Engine
    ```

2.  **Create and Activate a Virtual Environment**:
    *   Using `venv`:
        ```bash
        python -m venv venv
        # On Windows:
        .\venv\Scripts\activate
        # On macOS/Linux:
        source venv/bin/activate
        ```
    *   Or using `conda`:
        ```bash
        conda create -n creditrisk python=3.10  # Or your preferred Python version
        conda activate creditrisk
        ```

3.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    *   Copy the example environment file (if one exists) or create `config/.env`.
    *   Edit `config/.env` with your database settings. Example for SQLite (default fallback):
        ```dotenv
        # config/.env
        DB_TYPE=sqlite
        DB_NAME=credit_risk
        # DB_PATH=database/sqlite/credit_risk.db # Optional: Specify full path if needed
        ```
    *   Example for MySQL:
        ```dotenv
        # config/.env
        DB_TYPE=mysql
        DB_HOST=localhost
        DB_PORT=3306
        DB_USER=your_db_user
        DB_PASSWORD=your_db_password
        DB_NAME=credit_risk
        ```
    *   Example for PostgreSQL:
        ```dotenv
        # config/.env
        DB_TYPE=postgresql
        DB_HOST=localhost
        DB_PORT=5432
        DB_USER=your_db_user
        DB_PASSWORD=your_db_password
        DB_NAME=credit_risk
        ```

5.  **Set up the Database**:
    *   This script creates the database (if applicable for MySQL/Postgres) and the necessary tables based on `database/schema.sql`.
    *   It uses the settings from your `config/.env` file.
    *   **To use the default SQLite database:**
        ```bash
        python database/setup_db.py --db-type sqlite --create-db
        ```
    *   To set up MySQL or PostgreSQL (ensure the server is running and user/password are correct in `.env`):
        ```bash
        # For MySQL
        python database/setup_db.py --db-type mysql --create-db

        # For PostgreSQL
        python database/setup_db.py --db-type postgresql --create-db
        ```
    *   *Note:* If database setup fails, the other scripts (`train`, `predict`, `monitor`, `report`) have built-in fallbacks using synthetic data and local file storage, allowing basic functionality without a fully configured database.

6.  **Install WeasyPrint Dependencies (Optional, for PDF Reports)**:
    *   Generating PDF reports requires WeasyPrint and its underlying GTK+ dependencies.
    *   **Windows**: Follow the specific WeasyPrint installation guide for Windows, which usually involves installing GTK+ runtime libraries separately (e.g., via MSYS2 or a dedicated installer) and ensuring they are in the system PATH. See: [WeasyPrint Windows Install Docs](https://doc.weasyprint.org/en/stable/install.html#windows)
    *   **macOS**: `brew install pango gdk-pixbuf libffi cairo`
    *   **Linux (Debian/Ubuntu)**: `sudo apt-get update && sudo apt-get install python3-dev python3-pip python3-setuptools python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info`
    *   If dependencies are missing, the report generation script will skip PDF creation and issue a warning.

7.  **Generate Sample Data (Optional, for Development)**:
    *   If you have set up a database and want to populate it with sample data for testing:
        ```bash
        # Ensure database/setup_db.py has run successfully first
        python data_processing/generate_sample_data.py # Requires implementation
        ```
    *   *Note:* This script might need implementation based on the final schema. The fallback mechanisms in other scripts provide an alternative way to test without real database data.

## Workflow & Usage

The typical workflow involves training a model, using it for predictions, monitoring its performance, and generating reports.

1.  **Train a Model**:
    *   Uses data fetched according to the SQL query in the script (or synthetic data if DB fails).
    *   Saves the trained model, preprocessor, and metadata (including metrics) to `models/saved/`.
    *   Attempts to save model metadata to the `risk_model` table (or a fallback JSON file).
    *   Example (training a default logistic regression model):
        ```bash
        python models/train_model.py --model-type logistic --model-name baseline_logistic_v1
        ```
    *   Other options: `--tune-hyperparams`, `--test-size`, specific model types (`gbm`, `xgboost`, `rf`).

2.  **Make Predictions**:
    *   Loads a specified trained model from `models/saved/`.
    *   Fetches customer/loan data to score (or generates synthetic data if DB fails).
    *   Calculates PD and risk scores.
    *   Saves scores to the `risk_score` table (or a fallback CSV file).
    *   Example (scoring in batch mode using the latest version of `baseline_logistic_v1`):
        ```bash
        python models/predict.py --model-name baseline_logistic_v1 --batch --output-csv predictions/batch_scores.csv
        ```
    *   Other options: `--customer-id`, `--application-id`, `--loan-id` for single predictions.

3.  **Monitor Model Performance**:
    *   Calculates performance metrics (AUC, KS, PSI, etc.) over specified periods.
    *   Compares current performance against a reference period.
    *   Fetches score data from `risk_score` (or generates synthetic scores if DB fails).
    *   Saves monitoring results to the `model_monitoring` table (or a fallback JSON file in `monitoring/reports/`).
    *   Optionally generates trend charts.
    *   Example (running monthly monitoring for the specified model, generating plots):
        ```bash
        # Assumes 'baseline_logistic_v1_YYYYMMDD_HHMMSS' is the full model directory name
        python monitoring/model_monitoring.py --model-id baseline_logistic_v1_YYYYMMDD_HHMMSS --period monthly --generate-plots
        ```
    *   Other options: `--lookback`, `--period` (`daily`, `weekly`).

4.  **Generate Reports**:
    *   Fetches model info, monitoring data, score distributions, etc. (using database or file/synthetic fallbacks).
    *   Generates charts summarizing performance and data characteristics.
    *   Renders an HTML report using Jinja2 templates.
    *   Optionally generates a PDF version (if WeasyPrint is correctly installed).
    *   Saves report files to `reporting/outputs/`.
    *   Attempts to save report metadata to the `scorecard_report` table.
    *   Example (generating the default scorecard report):
        ```bash
        # Assumes 'baseline_logistic_v1_YYYYMMDD_HHMMSS' is the full model directory name
        python reporting/generate_reports.py --model-id baseline_logistic_v1_YYYYMMDD_HHMMSS --report-type scorecard --output-format both
        ```
    *   Other options: `--period-start`, `--period-end`.

## Contributing

Contributions are welcome! Please follow standard coding practices and consider adding tests for new features. (Testing framework setup TBD).

## License

(Specify License - e.g., MIT, Apache 2.0, etc.) 