#!/usr/bin/env python3
"""
Train a credit risk prediction model using loan data from the database.
This script builds and trains a model to predict probability of default.
"""

import os
import sys
import uuid
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
import joblib
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.database import execute_query, execute_statement, get_db_connection

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
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a credit risk prediction model')
    parser.add_argument('--model-type', type=str, choices=['logistic', 'gbm', 'xgboost', 'rf'], 
                       default='logistic', help='Type of model to train')
    parser.add_argument('--model-name', type=str, default='default_credit_risk_model',
                       help='Name for the trained model')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--tune-hyperparams', action='store_true',
                       help='Perform hyperparameter tuning')
    return parser.parse_args()


def fetch_training_data() -> pd.DataFrame:
    """
    Fetch and prepare data for training a credit risk model.
    
    Returns:
        pd.DataFrame: Combined dataset with features and target variable
    """
    logger.info("Fetching training data from database...")
    
    # Query to join customer profiles, loan applications, loan accounts, and payment history
    query = """
    SELECT 
        cp.customer_id,
        cp.age_years,
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
        
        CASE WHEN ld.loan_id IS NOT NULL THEN 1 ELSE 0 END AS default_flag
    FROM customer_profile cp
    JOIN loan_application la ON cp.customer_id = la.customer_id
    JOIN loan_account lac ON la.application_id = lac.application_id
    LEFT JOIN (
        SELECT DISTINCT loan_id, customer_id 
        FROM loan_delinquency 
        WHERE days_past_due >= 90
    ) ld ON lac.loan_id = ld.loan_id AND cp.customer_id = ld.customer_id
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
    WHERE la.status = 'Approved'
    """
    
    try:
        # Execute the query
        df = execute_query(query)
        
        # Add age calculation based on date_of_birth
        df['age_years'] = (datetime.now().year - pd.to_datetime(df['date_of_birth']).dt.year)
        
        # Handle missing values for important columns
        for col in ['credit_score', 'annual_income', 'employment_length_years']:
            if col in df.columns and df[col].isnull().sum() > 0:
                logger.warning(f"Found {df[col].isnull().sum()} missing values in {col}")
        
        logger.info(f"Fetched {len(df)} rows of training data")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching training data: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Preprocess the data for model training.
    
    Args:
        df (pd.DataFrame): Raw training data
    
    Returns:
        Tuple containing:
        - X: Feature matrix
        - y: Target vector
        - feature_names: List of feature names
        - preprocessing_info: Dictionary with preprocessing information
    """
    logger.info("Preprocessing training data...")
    
    # Define features and target
    X = df.drop(['default_flag', 'customer_id', 'application_id'], axis=1)
    y = df['default_flag'].values
    
    # Save original feature names
    feature_names = X.columns.tolist()
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Features: {len(feature_names)} total, {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    
    # Define preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding
    preprocessed_feature_names = numeric_features.copy()
    
    # Extract one-hot encoding feature names
    if categorical_features:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = []
        for i, col in enumerate(categorical_features):
            values = ohe.categories_[i]
            for val in values:
                cat_feature_names.append(f"{col}_{val}")
        preprocessed_feature_names.extend(cat_feature_names)
    
    # Store preprocessing information
    preprocessing_info = {
        'preprocessor': preprocessor,
        'original_feature_names': feature_names,
        'processed_feature_names': preprocessed_feature_names,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
    
    logger.info(f"Preprocessing complete. Processed feature matrix shape: {X_processed.shape}")
    
    return X_processed, y, preprocessed_feature_names, preprocessing_info


def train_model(
    X: np.ndarray, 
    y: np.ndarray, 
    model_type: str, 
    feature_names: List[str],
    tune_hyperparams: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a model on the preprocessed data.
    
    Args:
        X (np.ndarray): Preprocessed feature matrix
        y (np.ndarray): Target vector
        model_type (str): Type of model to train ('logistic', 'gbm', 'xgboost', 'rf')
        feature_names (List[str]): List of feature names
        tune_hyperparams (bool): Whether to perform hyperparameter tuning
    
    Returns:
        Tuple containing:
        - model: Trained model
        - metrics: Dictionary with model performance metrics
    """
    logger.info(f"Training {model_type} model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Define model based on type
    if model_type == 'logistic':
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        
        if tune_hyperparams:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            model = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    
    elif model_type == 'gbm':
        model = GradientBoostingClassifier(random_state=42)
        
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 1.0]
            }
            model = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            model = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    if tune_hyperparams:
        logger.info("Performing hyperparameter tuning...")
        model.fit(X_train, y_train)
        logger.info(f"Best parameters: {model.best_params_}")
        model = model.best_estimator_
    else:
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Binary predictions with optimal threshold
    y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
    
    # Feature importance
    feature_importance = {}
    if hasattr(model, 'coef_'):  # Linear models
        importance = np.abs(model.coef_[0])
        for i, name in enumerate(feature_names):
            if i < len(importance):
                feature_importance[name] = float(importance[i])
    
    elif hasattr(model, 'feature_importances_'):  # Tree-based models
        importance = model.feature_importances_
        for i, name in enumerate(feature_names):
            if i < len(importance):
                feature_importance[name] = float(importance[i])
    
    # Compile metrics
    metrics = {
        'auc_roc': float(auc_score),
        'optimal_threshold': float(optimal_threshold),
        'accuracy': float(accuracy),
        'precision': float(precision_score),
        'recall': float(recall_score),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'feature_importance': feature_importance
    }
    
    logger.info(f"Model training complete. AUC-ROC: {auc_score:.4f}, F1 Score: {f1:.4f}")
    
    return model, metrics


def save_model(
    model, 
    preprocessor, 
    metrics: Dict[str, Any], 
    feature_names: List[str],
    model_name: str
) -> str:
    """
    Save the trained model and metadata.
    
    Args:
        model: Trained model
        preprocessor: Data preprocessor
        metrics (Dict[str, Any]): Model performance metrics
        feature_names (List[str]): List of feature names
        model_name (str): Name for the model
    
    Returns:
        str: Path to the saved model
    """
    # Create a unique model ID
    model_id = str(uuid.uuid4())
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model directory
    model_dir = MODELS_DIR / f"{model_name}_{model_version}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata
    metadata = {
        'model_id': model_id,
        'model_name': model_name,
        'model_version': model_version,
        'model_type': model.__class__.__name__,
        'training_date': datetime.now().isoformat(),
        'feature_names': feature_names,
        'metrics': metrics,
        'params': model.get_params()
    }
    
    # Save model
    model_file = model_dir / 'model.joblib'
    joblib.dump(model, model_file)
    
    # Save preprocessor
    preprocessor_file = model_dir / 'preprocessor.joblib'
    joblib.dump(preprocessor, preprocessor_file)
    
    # Save metadata
    metadata_file = model_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved at {model_dir}")
    
    # Save model info to database
    save_model_to_db(metadata)
    
    return str(model_dir)


def save_model_to_db(metadata: Dict[str, Any]) -> None:
    """
    Save model metadata to the database.
    
    Args:
        metadata (Dict[str, Any]): Model metadata
    """
    try:
        # Prepare features and metrics as JSON strings
        features_list = json.dumps(metadata['feature_names'])
        performance_metrics = json.dumps(metadata['metrics'])
        
        # Insert into risk_model table
        insert_query = """
        INSERT INTO risk_model (
            model_id, model_name, model_version, model_type, 
            description, training_date, status, features_list, 
            performance_metrics, creator
        ) VALUES (
            :model_id, :model_name, :model_version, :model_type,
            :description, :training_date, :status, :features_list,
            :performance_metrics, :creator
        )
        """
        
        params = {
            'model_id': metadata['model_id'],
            'model_name': metadata['model_name'],
            'model_version': metadata['model_version'],
            'model_type': metadata['model_type'],
            'description': f"Credit risk model trained on {datetime.now().date()}",
            'training_date': datetime.now().date(),
            'status': 'Trained',
            'features_list': features_list,
            'performance_metrics': performance_metrics,
            'creator': 'system'
        }
        
        execute_statement(insert_query, params)
        logger.info(f"Model metadata saved to database with ID {metadata['model_id']}")
    
    except Exception as e:
        logger.error(f"Error saving model metadata to database: {e}")


def main():
    """Main function to train a credit risk model."""
    args = parse_arguments()
    
    try:
        # Fetch data
        df = fetch_training_data()
        
        # Preprocess data
        X, y, feature_names, preprocessing_info = preprocess_data(df)
        
        # Train model
        model, metrics = train_model(
            X, y, args.model_type, feature_names, args.tune_hyperparams
        )
        
        # Save model
        save_model(
            model,
            preprocessing_info['preprocessor'],
            metrics,
            feature_names,
            args.model_name
        )
        
        logger.info("Model training and saving completed successfully")
    
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 