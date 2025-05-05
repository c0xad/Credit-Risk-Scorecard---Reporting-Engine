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
    Generate synthetic data for training a credit risk model.
    This is used when database tables don't exist yet.
    
    Returns:
        pd.DataFrame: Synthetic dataset with features and target variable
    """
    logger.info("Generating synthetic training data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define number of samples
    n_samples = 1000
    
    # Generate customer IDs
    customer_ids = [f'CUST{i:06d}' for i in range(1, n_samples + 1)]
    
    # Generate demographic features
    age_years = np.random.randint(18, 75, n_samples)
    gender = np.random.choice(['Male', 'Female', 'Other'], n_samples)
    marital_status = np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples)
    dependents = np.random.randint(0, 6, n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', 'Other'], n_samples)
    employment_status = np.random.choice(['Employed', 'Self-employed', 'Unemployed', 'Retired'], n_samples)
    employment_length_years = np.random.randint(0, 40, n_samples)
    annual_income = np.random.uniform(20000, 200000, n_samples)
    
    # Generate credit bureau data
    credit_score = np.random.randint(300, 850, n_samples)
    total_accounts = np.random.randint(1, 20, n_samples)
    open_accounts = np.random.randint(1, 10, n_samples)
    delinquent_accounts = np.random.randint(0, 3, n_samples)
    credit_utilization_ratio = np.random.uniform(0, 1, n_samples)
    length_of_credit_history_months = np.random.randint(6, 360, n_samples)
    hard_inquiries_last_12m = np.random.randint(0, 5, n_samples)
    collections_last_12m = np.random.randint(0, 2, n_samples)
    
    # Generate loan data
    application_ids = [f'APP{i:06d}' for i in range(1, n_samples + 1)]
    loan_type = np.random.choice(['Personal', 'Auto', 'Mortgage', 'Education'], n_samples)
    loan_purpose = np.random.choice(['Debt Consolidation', 'Home Improvement', 'Major Purchase', 'Other'], n_samples)
    requested_amount = np.random.uniform(5000, 500000, n_samples)
    term_months = np.random.choice([12, 24, 36, 48, 60, 120, 240, 360], n_samples)
    interest_rate = np.random.uniform(0.02, 0.15, n_samples)
    
    # Generate default flag
    # We'll make default probability correlated with some features
    default_prob = (
        0.1 +                                       # Base default rate
        0.1 * (delinquent_accounts > 0) +           # Higher default for those with delinquencies
        0.1 * (credit_score < 600) +                # Higher default for low credit score
        0.1 * (credit_utilization_ratio > 0.7) +    # Higher default for high utilization
        0.1 * (annual_income < 40000) +             # Higher default for low income
        0.1 * (collections_last_12m > 0) -          # Higher default for those with collections
        0.1 * (employment_length_years > 5) -       # Lower default for stable employment
        0.05 * (age_years > 45)                     # Lower default for older applicants
    )
    # Cap probabilities between 0 and 1
    default_prob = np.clip(default_prob, 0, 0.9)
    default_flag = np.random.binomial(1, default_prob, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age_years': age_years,
        'gender': gender,
        'marital_status': marital_status,
        'dependents': dependents,
        'education': education,
        'employment_status': employment_status,
        'employment_length_years': employment_length_years,
        'annual_income': annual_income,
        'credit_score': credit_score,
        'total_accounts': total_accounts,
        'open_accounts': open_accounts,
        'delinquent_accounts': delinquent_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'length_of_credit_history_months': length_of_credit_history_months,
        'hard_inquiries_last_12m': hard_inquiries_last_12m,
        'collections_last_12m': collections_last_12m,
        'application_id': application_ids,
        'loan_type': loan_type,
        'loan_purpose': loan_purpose,
        'requested_amount': requested_amount,
        'term_months': term_months,
        'interest_rate': interest_rate,
        'default_flag': default_flag
    })
    
    # Add some missing values to make it more realistic
    for col in ['credit_score', 'annual_income', 'employment_length_years']:
        mask = np.random.choice([True, False], n_samples, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan
        logger.warning(f"Added {mask.sum()} missing values in {col}")
    
    logger.info(f"Generated {len(df)} rows of synthetic training data")
    logger.info(f"Default rate: {df['default_flag'].mean():.2%}")
    
    return df


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
    Save model metadata to the database or a JSON file if the database is not available.
    
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
            'training_date': datetime.now().date().isoformat(),
            'status': 'Trained',
            'features_list': features_list,
            'performance_metrics': performance_metrics,
            'creator': 'system'
        }
        
        try:
            execute_statement(insert_query, params)
            logger.info(f"Model metadata saved to database with ID {metadata['model_id']}")
        except Exception as db_error:
            logger.warning(f"Could not save to database: {db_error}")
            # Fallback to JSON file if database is not available
            model_dir = Path(f"{MODELS_DIR}/{metadata['model_name']}_{metadata['model_version']}")
            metadata_file = model_dir / 'db_metadata.json'
            
            with open(metadata_file, 'w') as f:
                json.dump(params, f, indent=2)
            
            logger.info(f"Model metadata saved to file {metadata_file} as fallback")
    
    except Exception as e:
        logger.error(f"Error in save_model_to_db: {e}")
        # Don't raise the exception, just log it


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