# Credit Risk Scorecard & Reporting Engine

A comprehensive system for credit risk modeling, scoring, and reporting using SQL databases and Python-based machine learning.

## Features

- **Normalized SQL Database**: Customer profiles, credit bureau data, and repayment history
- **ML-based Risk Models**: Logistic regression and gradient boosting models
- **Automated Reporting**: Generate scorecards and risk reports with Jinja2
- **Data Quality Monitoring**: Track data quality metrics and model performance
- **Model Monitoring**: Record PSI/KPI metrics and monitor model drift
- **Parameterized Reports**: Custom views for underwriters and analysts

## Project Structure

```
├── database/           # SQL schema definitions and scripts
├── data_processing/    # Data cleaning and preprocessing modules 
├── models/             # Model training, evaluation, and persistence
├── reporting/          # Report templates and generation
├── monitoring/         # Model and data monitoring tools
├── config/             # Configuration files
├── utils/              # Utility functions
└── tests/              # Unit and integration tests
```

## Setup

1. Create and activate a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Set up the database
   ```
   python database/setup_db.py
   ```

4. Run sample data generation (for development)
   ```
   python data_processing/generate_sample_data.py
   ```

## Usage

1. Data preprocessing
   ```
   python data_processing/preprocess.py
   ```

2. Model training
   ```
   python models/train_model.py
   ```

3. Generate reports
   ```
   python reporting/generate_reports.py
   ```

4. Monitor model performance
   ```
   python monitoring/model_monitoring.py
   ``` 