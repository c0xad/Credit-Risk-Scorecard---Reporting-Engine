#!/usr/bin/env python3
"""
Score customers using trained credit risk models and save results to the database.
This script can be used for batch scoring or real-time predictions.
"""

import os
import sys
import uuid
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.database import execute_query, execute_statement, get_db_connection, save_df_to_table

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / "config" / ".env")

# Models directory
MODELS_DIR = PROJECT_ROOT / "models" / "saved"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions using a trained credit risk model')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Name or ID of the model to use')
    parser.add_argument('--customer-id', type=str,
                       help='Specific customer ID to score')
    parser.add_argument('--application-id', type=str,
                       help='Specific loan application ID to score')
    parser.add_argument('--loan-id', type=str,
                       help='Specific loan ID to score')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch scoring for all active loans')
    parser.add_argument('--output-csv', type=str,
                       help='Save results to a CSV file')
    return parser.parse_args()


def get_model_path(model_name_or_id: str) -> Path:
    """
    Find the model directory based on name or ID.
    
    Args:
        model_name_or_id (str): Model name or ID
    
    Returns:
        Path: Path to the model directory
    """
    # Check if it's a full path
    if os.path.exists(model_name_or_id):
        return Path(model_name_or_id)
    
    # Check if it's a model ID or name in the database
    try:
        query = """
        SELECT model_id, model_name, model_version
        FROM risk_model
        WHERE model_id = :model_id OR model_name = :model_name
        ORDER BY training_date DESC
        LIMIT 1
        """
        result = execute_query(query, {'model_id': model_name_or_id, 'model_name': model_name_or_id})
        
        if not result.empty:
            model_name = result['model_name'].iloc[0]
            model_version = result['model_version'].iloc[0]
            return MODELS_DIR / f"{model_name}_{model_version}"
    except Exception as e:
        logger.warning(f"Could not find model in database: {e}")
    
    # Check if it's a subdirectory in the models directory
    potential_dirs = list(MODELS_DIR.glob(f"{model_name_or_id}*"))
    if potential_dirs:
        # Get the latest version by sorting
        return sorted(potential_dirs)[-1]
    
    raise FileNotFoundError(f"Could not find model with name or ID: {model_name_or_id}")


def load_model(model_path: Path) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load a trained model and its metadata.
    
    Args:
        model_path (Path): Path to the model directory
    
    Returns:
        Tuple containing:
        - model: Trained model
        - preprocessor: Data preprocessor
        - metadata: Model metadata
    """
    model_file = model_path / 'model.joblib'
    preprocessor_file = model_path / 'preprocessor.joblib'
    metadata_file = model_path / 'metadata.json'
    
    if not model_file.exists() or not preprocessor_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(f"Model files not found in {model_path}")
    
    model = joblib.load(model_file)
    preprocessor = joblib.load(preprocessor_file)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return model, preprocessor, metadata


def fetch_customer_data(customer_id: Optional[str] = None, 
                       application_id: Optional[str] = None,
                       loan_id: Optional[str] = None,
                       batch: bool = False) -> pd.DataFrame:
    """
    Fetch customer data for prediction.
    
    Args:
        customer_id (Optional[str]): Specific customer ID to fetch
        application_id (Optional[str]): Specific loan application ID to fetch
        loan_id (Optional[str]): Specific loan ID to fetch
        batch (bool): Whether to fetch data for all active loans
    
    Returns:
        pd.DataFrame: Customer data for prediction
    """
    # Build the WHERE clause based on input parameters
    where_clause = ""
    params = {}
    
    if customer_id:
        where_clause = "WHERE cp.customer_id = :customer_id"
        params['customer_id'] = customer_id
    elif application_id:
        where_clause = "WHERE la.application_id = :application_id"
        params['application_id'] = application_id
    elif loan_id:
        where_clause = "WHERE lac.loan_id = :loan_id"
        params['loan_id'] = loan_id
    elif batch:
        where_clause = "WHERE lac.status = 'Active'"
    
    # Query to join customer profiles, loan applications, loan accounts, and credit bureau data
    query = f"""
    SELECT 
        cp.customer_id,
        cp.date_of_birth,
        cp.gender,
        cp.marital_status,
        cp.dependents,
        cp.education,
        cp.employment_status,
        cp.employment_length_years,
        cp.annual_income,
        
        cbd.credit_score,
        cbd.total_accounts,
        cbd.open_accounts,
        cbd.delinquent_accounts,
        cbd.credit_utilization_ratio,
        cbd.length_of_credit_history_months,
        cbd.hard_inquiries_last_12m,
        cbd.collections_last_12m,
        
        la.application_id,
        la.loan_type,
        la.loan_purpose,
        la.requested_amount,
        la.term_months,
        la.interest_rate,
        
        lac.loan_id,
        lac.loan_amount,
        lac.remaining_balance,
        lac.monthly_payment
        
    FROM customer_profile cp
    JOIN loan_application la ON cp.customer_id = la.customer_id
    JOIN loan_account lac ON la.application_id = lac.application_id
    LEFT JOIN (
        SELECT customer_id, credit_score, total_accounts, open_accounts, 
               delinquent_accounts, credit_utilization_ratio,
               length_of_credit_history_months, hard_inquiries_last_12m,
               collections_last_12m
        FROM credit_bureau_data cbd
        WHERE report_date = (
            SELECT MAX(report_date) 
            FROM credit_bureau_data cbd2 
            WHERE cbd2.customer_id = cbd.customer_id
        )
    ) cbd ON cp.customer_id = cbd.customer_id
    {where_clause}
    """
    
    try:
        # Execute the query
        df = execute_query(query, params if params else None)
        
        if df.empty:
            logger.warning("No data found for the specified criteria")
            return pd.DataFrame()
        
        # Add age calculation based on date_of_birth
        df['age_years'] = (datetime.now().year - pd.to_datetime(df['date_of_birth']).dt.year)
        
        # Handle missing values for important columns
        for col in ['credit_score', 'annual_income', 'employment_length_years']:
            if col in df.columns and df[col].isnull().sum() > 0:
                logger.warning(f"Found {df[col].isnull().sum()} missing values in {col}")
        
        logger.info(f"Fetched {len(df)} rows of customer data for prediction")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching customer data: {e}")
        raise


def predict(model, preprocessor, metadata: Dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        preprocessor: Data preprocessor
        metadata (Dict[str, Any]): Model metadata
        data (pd.DataFrame): Customer data for prediction
    
    Returns:
        pd.DataFrame: Prediction results
    """
    if data.empty:
        return pd.DataFrame()
    
    try:
        # Prepare input features - ensure only expected columns are used
        feature_names = metadata.get('feature_names', [])
        X = data.drop(['customer_id', 'application_id', 'loan_id', 'date_of_birth'], axis=1, errors='ignore')
        
        # Check if all required features are present
        missing_features = [f for f in feature_names if f not in X.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        # Apply preprocessing
        X_processed = preprocessor.transform(X)
        
        # Make predictions
        probabilities = model.predict_proba(X_processed)[:, 1]
        
        # Get optimal threshold from metadata
        threshold = metadata.get('metrics', {}).get('optimal_threshold', 0.5)
        
        # Create result DataFrame
        results = pd.DataFrame({
            'customer_id': data['customer_id'],
            'application_id': data['application_id'] if 'application_id' in data.columns else None,
            'loan_id': data['loan_id'] if 'loan_id' in data.columns else None,
            'probability_of_default': probabilities,
            'risk_band': pd.cut(
                probabilities, 
                bins=[0, 0.1, 0.2, 0.4, 0.6, 1.0],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        })
        
        # Add score based on probability (higher score = lower risk)
        results['score_value'] = 850 - (probabilities * 550).astype(int)
        
        logger.info(f"Made predictions for {len(results)} customers")
        return results
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise


def save_predictions(predictions: pd.DataFrame, model_id: str) -> None:
    """
    Save prediction results to the database.
    
    Args:
        predictions (pd.DataFrame): Prediction results
        model_id (str): ID of the model used for predictions
    """
    if predictions.empty:
        logger.warning("No predictions to save")
        return
    
    try:
        # Prepare data for insertion
        now = datetime.now().date()
        data = []
        
        for _, row in predictions.iterrows():
            data.append({
                'score_id': str(uuid.uuid4()),
                'customer_id': row['customer_id'],
                'model_id': model_id,
                'application_id': row['application_id'] if pd.notna(row['application_id']) else None,
                'loan_id': row['loan_id'] if pd.notna(row['loan_id']) else None,
                'score_date': now,
                'score_value': float(row['score_value']),
                'probability_of_default': float(row['probability_of_default']),
                'risk_band': str(row['risk_band']),
                'score_factors': None  # Could include feature importance details here
            })
        
        # Save to database
        risk_scores_df = pd.DataFrame(data)
        save_df_to_table(risk_scores_df, 'risk_score', 'append')
        
        logger.info(f"Saved {len(data)} predictions to the database")
    
    except Exception as e:
        logger.error(f"Error saving predictions to database: {e}")
        raise


def main():
    """Main function to make predictions using a trained model."""
    args = parse_arguments()
    
    try:
        # Get model path
        model_path = get_model_path(args.model_name)
        logger.info(f"Using model at: {model_path}")
        
        # Load model
        model, preprocessor, metadata = load_model(model_path)
        
        # Fetch data
        data = fetch_customer_data(
            customer_id=args.customer_id,
            application_id=args.application_id,
            loan_id=args.loan_id,
            batch=args.batch
        )
        
        if data.empty:
            logger.error("No data found for prediction")
            sys.exit(1)
        
        # Make predictions
        predictions = predict(model, preprocessor, metadata, data)
        
        # Save predictions to database
        save_predictions(predictions, metadata['model_id'])
        
        # Save to CSV if requested
        if args.output_csv:
            predictions.to_csv(args.output_csv, index=False)
            logger.info(f"Saved predictions to {args.output_csv}")
        
        logger.info("Prediction process completed successfully")
    
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 