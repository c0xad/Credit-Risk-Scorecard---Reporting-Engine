@echo off
echo Credit Risk Scorecard Data Preprocessing
echo =======================================

REM Create output directory
mkdir processed_data 2>nul

echo 1. Processing loan application data with rejection as target...
python preprocess_loan_data.py --query default --target rejection_flag --output processed_data/rejection_model

echo 2. Processing loan payment data with delinquency as target...
python preprocess_loan_data.py --query full --target delinquency_flag --output processed_data/delinquency_model

echo 3. Processing sample data from CSV (if available)...
IF EXIST sample_output\applications.csv (
    python preprocess_loan_data.py --source csv --file sample_output\applications.csv --target status --output processed_data/sample_model
) ELSE (
    echo Sample CSV file not found, skipping...
)

echo Processing complete!
echo Results are saved in the processed_data directory.
pause 