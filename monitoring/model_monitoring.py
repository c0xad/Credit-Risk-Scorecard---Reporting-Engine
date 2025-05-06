#!/usr/bin/env python3
"""
Monitor credit risk model performance over time.
This script calculates and stores key model monitoring metrics.
"""

import os
import sys
import uuid
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
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

# Directory for storing monitoring reports
REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Models directory (needed for fallback)
MODELS_DIR = PROJECT_ROOT / "models" / "saved"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Monitor credit risk model performance')
    parser.add_argument('--model-id', type=str, required=True,
                       help='ID of the model to monitor')
    parser.add_argument('--period', type=str, choices=['daily', 'weekly', 'monthly'], 
                       default='monthly', help='Monitoring period')
    parser.add_argument('--lookback', type=int, default=6,
                       help='Number of periods to look back for trends')
    parser.add_argument('--output-dir', type=str, default=str(REPORTS_DIR),
                       help='Directory to save monitoring reports')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate and save plots')
    return parser.parse_args()


def find_model_path(model_name_or_id: str) -> Path:
    """
    Find the model directory based on name or ID.
    Helper function similar to the one in predict.py.
    
    Args:
        model_name_or_id (str): Model name or ID
    
    Returns:
        Path: Path to the model directory
    """
    # Check if it's a full path
    if os.path.exists(model_name_or_id):
        path_obj = Path(model_name_or_id)
        if path_obj.is_dir():
            return path_obj
    
    # Check if it's a subdirectory in the models directory
    potential_dirs = list(MODELS_DIR.glob(f"{model_name_or_id}*"))
    if potential_dirs:
        # Get the latest version by sorting
        latest_model_dir = sorted(potential_dirs)[-1]
        logger.info(f"Found model directory: {latest_model_dir}")
        return latest_model_dir
    
    # Check if any model directory exists as a fallback
    all_model_dirs = list(MODELS_DIR.glob("*"))
    if all_model_dirs:
        latest_model_dir = sorted(all_model_dirs)[-1]
        logger.warning(f"Model '{model_name_or_id}' not found. Falling back to latest available model: {latest_model_dir}")
        return latest_model_dir
    
    raise FileNotFoundError(f"Could not find any model directory in {MODELS_DIR}")


def calculate_period_dates(period: str, lookback: int) -> Tuple[datetime, datetime, List[Tuple[datetime, datetime]]]:
    """
    Calculate the date ranges for monitoring periods.
    
    Args:
        period (str): Period type ('daily', 'weekly', 'monthly')
        lookback (int): Number of periods to look back
    
    Returns:
        Tuple containing:
        - Current period end date
        - Current period start date
        - List of (start_date, end_date) tuples for all periods (most recent first)
    """
    today = datetime.now().date()
    periods = []
    
    if period == 'daily':
        current_end = today
        current_start = today # Daily period is just today
        for i in range(lookback):
            end_date = today - timedelta(days=i)
            start_date = end_date
            periods.append((start_date, end_date))
    
    elif period == 'weekly':
        # Current week end is today, start is start of the week (e.g., Monday)
        current_end = today
        current_start = today - timedelta(days=today.weekday()) # Monday
        for i in range(lookback):
            end_date = today - timedelta(weeks=i)
            start_date = end_date - timedelta(days=end_date.weekday()) # Monday of that week
            periods.append((start_date, end_date))
    
    elif period == 'monthly':
        # Current month start and end
        current_end = today
        current_start = today.replace(day=1)
        
        current_month_start = current_start
        for i in range(lookback):
            # Start of the month i months ago
            month = current_month_start.month
            year = current_month_start.year
            start_date = current_month_start
            
            # End date is the end of that month, but capped at today if it's the current month
            if i == 0:
                end_date = today
            else:
                 # Last day of the previous month
                if month == 12:
                    end_date = datetime(year, month, 31).date()
                else:
                    end_date = datetime(year, month + 1, 1).date() - timedelta(days=1)
            
            periods.append((start_date, end_date))
            
            # Move to the start of the previous month for the next iteration
            if current_month_start.month == 1:
                current_month_start = current_month_start.replace(year=year-1, month=12)
            else:
                current_month_start = current_month_start.replace(month=month-1)
                
    else:
        raise ValueError(f"Unsupported period type: {period}")
    
    # Ensure current_start and current_end cover the most recent period
    if periods:
        current_start, current_end = periods[0]
    else: # Handle case where lookback might be 0
         current_start = today.replace(day=1) if period == 'monthly' else (today - timedelta(days=today.weekday()) if period == 'weekly' else today)
         current_end = today

    return current_end, current_start, periods


def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get model information from the database or fallback to metadata file.
    
    Args:
        model_id (str): ID of the model to monitor
    
    Returns:
        Dict[str, Any]: Model information
    """
    model_info = None
    
    # Try fetching from database first
    try:
        query = """
        SELECT model_id, model_name, model_version, model_type, training_date, 
               status, features_list, performance_metrics
        FROM risk_model
        WHERE model_id = :model_id
        """
        result = execute_query(query, {'model_id': model_id})
        
        if not result.empty:
            model_info = result.iloc[0].to_dict()
            logger.info(f"Fetched model info for {model_id} from database.")
            # Parse JSON fields
            for field in ['features_list', 'performance_metrics']:
                if model_info[field] and isinstance(model_info[field], str):
                    try:
                        model_info[field] = json.loads(model_info[field])
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse JSON for field {field}")
                        model_info[field] = None # Or handle as appropriate
            return model_info
        else:
             logger.warning(f"Model {model_id} not found in database. Trying filesystem fallback.")

    except Exception as e:
        logger.warning(f"Error getting model info from database: {e}. Trying filesystem fallback.")

    # Filesystem fallback
    try:
        model_path = find_model_path(model_id)
        metadata_file = model_path / 'metadata.json'
        db_metadata_file = model_path / 'db_metadata.json' # Fallback for db info

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded model info for {model_id} from {metadata_file}")
            
            # Try to supplement with db_metadata if available
            db_info = {}
            if db_metadata_file.exists():
                 with open(db_metadata_file, 'r') as f:
                    db_info = json.load(f)
            
            # Combine info, prioritizing specific metadata fields
            model_info = {
                'model_id': metadata.get('model_id', db_info.get('model_id', model_id)),
                'model_name': metadata.get('model_name', db_info.get('model_name', model_path.name.split('_')[0])),
                'model_version': metadata.get('model_version', db_info.get('model_version', model_path.name.split('_')[-1])),
                'model_type': metadata.get('model_type', db_info.get('model_type', 'Unknown')),
                'training_date': metadata.get('training_date', db_info.get('training_date')),
                'status': db_info.get('status', 'Unknown'),
                'features_list': metadata.get('feature_names', db_info.get('features_list', [])),
                'performance_metrics': metadata.get('metrics', db_info.get('performance_metrics', {})),
            }
            return model_info
        else:
            raise FileNotFoundError(f"Metadata file not found in {model_path}")
            
    except FileNotFoundError as e:
         logger.error(f"Filesystem fallback failed: {e}")
         raise ValueError(f"Could not retrieve information for model: {model_id}") from e
    except Exception as e:
        logger.error(f"Error during filesystem fallback: {e}")
        raise ValueError(f"Could not retrieve information for model: {model_id}") from e


def get_scores_for_period(model_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Get model scores for a specific period from DB or generate synthetic data.
    
    Args:
        model_id (str): ID of the model to monitor
        start_date (datetime): Start date of the period
        end_date (datetime): End date of the period
    
    Returns:
        pd.DataFrame: Model scores for the period
    """
    try:
        query = """
        SELECT rs.score_id, rs.customer_id, rs.loan_id, rs.score_date, 
               rs.score_value, rs.probability_of_default, rs.risk_band,
               ld.delinquency_id, CASE WHEN ld.loan_id IS NOT NULL THEN 1 ELSE 0 END AS actual_default
        FROM risk_score rs
        LEFT JOIN loan_account la ON rs.loan_id = la.loan_id
        LEFT JOIN (
            SELECT DISTINCT loan_id, delinquency_id
            FROM loan_delinquency
            WHERE days_past_due >= 90
                  AND start_date BETWEEN :start_date AND :end_date
        ) ld ON rs.loan_id = ld.loan_id
        WHERE rs.model_id = :model_id
              AND rs.score_date BETWEEN :start_date AND :end_date
        """
        
        params = {
            'model_id': model_id,
            'start_date': start_date,
            'end_date': end_date
        }
        
        result = execute_query(query, params)
        
        if not result.empty:
            logger.info(f"Retrieved {len(result)} scores from DB for period {start_date} to {end_date}")
            return result
        else:
            logger.warning(f"No scores found in DB for period {start_date} to {end_date}. Generating synthetic data.")
            return generate_synthetic_scores(model_id, start_date, end_date)
            
    except Exception as e:
        logger.warning(f"Error getting scores from DB: {e}. Generating synthetic data.")
        return generate_synthetic_scores(model_id, start_date, end_date)


def generate_synthetic_scores(model_id: str, start_date: datetime, end_date: datetime, num_samples: int = 500) -> pd.DataFrame:
    """
    Generate synthetic score data for a period.
    
    Args:
        model_id (str): Model ID
        start_date (datetime): Start date of the period
        end_date (datetime): End date of the period
        num_samples (int): Number of synthetic scores to generate
    
    Returns:
        pd.DataFrame: DataFrame with synthetic scores
    """
    np.random.seed(abs(hash(f"{model_id}{start_date}{end_date}")) % (2**32 - 1))
    
    # Generate plausible probabilities of default (e.g., beta distribution)
    # Skewed towards lower probabilities
    a, b = 2, 5 
    probabilities = np.random.beta(a, b, num_samples)
    
    # Generate actual defaults correlated with probabilities
    # Higher probability = higher chance of default
    default_thresholds = np.random.uniform(0, 1, num_samples)
    actual_default = (probabilities > default_thresholds).astype(int)
    
    # Generate score values based on probabilities (higher score = lower risk)
    score_values = 850 - (probabilities * 550).astype(int)
    score_values = np.clip(score_values, 300, 850)
    
    # Generate risk bands based on probabilities
    risk_bands = pd.cut(
        probabilities, 
        bins=[0, 0.1, 0.2, 0.4, 0.6, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )
    
    # Generate other fields
    score_ids = [str(uuid.uuid4()) for _ in range(num_samples)]
    customer_ids = [f'SYNTH_CUST_{i:05d}' for i in range(num_samples)]
    loan_ids = [f'SYNTH_LOAN_{i:05d}' for i in range(num_samples)]
    
    # Generate random dates within the period
    time_delta = (end_date - start_date).days
    random_days = np.random.randint(0, time_delta + 1, num_samples)
    score_dates = [start_date + timedelta(days=int(d)) for d in random_days]
    
    # Create DataFrame
    df = pd.DataFrame({
        'score_id': score_ids,
        'customer_id': customer_ids,
        'loan_id': loan_ids,
        'score_date': score_dates,
        'score_value': score_values,
        'probability_of_default': probabilities,
        'risk_band': risk_bands,
        'delinquency_id': [str(uuid.uuid4()) if d == 1 else None for d in actual_default], # Synthetic ID if defaulted
        'actual_default': actual_default
    })
    
    logger.info(f"Generated {len(df)} synthetic scores for period {start_date} to {end_date}")
    return df


def calculate_population_stability_index(
    reference_scores: pd.DataFrame, 
    current_scores: pd.DataFrame,
    num_bins: int = 10
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate the Population Stability Index (PSI) between two score distributions.
    
    Args:
        reference_scores (pd.DataFrame): Reference period scores
        current_scores (pd.DataFrame): Current period scores
        num_bins (int): Number of bins for score distribution
    
    Returns:
        Tuple containing:
        - PSI value
        - DataFrame with bin details
    """
    # Use probability of default for PSI calculation
    ref_values = reference_scores['probability_of_default'].values
    curr_values = current_scores['probability_of_default'].values
    
    if len(ref_values) == 0 or len(curr_values) == 0:
        logger.warning("Cannot calculate PSI: one or both samples are empty")
        return 0.0, pd.DataFrame()
    
    # Create bins based on reference scores
    bins = np.linspace(0, 1, num_bins + 1)
    
    # Count observations in each bin
    ref_counts, _ = np.histogram(ref_values, bins=bins)
    curr_counts, _ = np.histogram(curr_values, bins=bins)
    
    # Convert to percentages
    ref_pcts = ref_counts / len(ref_values)
    curr_pcts = curr_counts / len(curr_values)
    
    # Replace zeros with small value to avoid division by zero
    ref_pcts = np.where(ref_pcts == 0, 0.0001, ref_pcts)
    curr_pcts = np.where(curr_pcts == 0, 0.0001, curr_pcts)
    
    # Calculate PSI for each bin
    psi_values = (curr_pcts - ref_pcts) * np.log(curr_pcts / ref_pcts)
    
    # Create results dataframe
    bin_results = pd.DataFrame({
        'bin_min': bins[:-1],
        'bin_max': bins[1:],
        'reference_count': ref_counts,
        'reference_pct': ref_pcts,
        'current_count': curr_counts,
        'current_pct': curr_pcts,
        'psi': psi_values
    })
    
    # Calculate total PSI
    psi_total = np.sum(psi_values)
    
    return psi_total, bin_results


def calculate_performance_metrics(scores_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate model performance metrics.
    
    Args:
        scores_df (pd.DataFrame): Scores with actual defaults
    
    Returns:
        Dict[str, Any]: Performance metrics
    """
    if scores_df.empty or 'actual_default' not in scores_df.columns:
        logger.warning("Cannot calculate performance metrics: data is missing or incomplete")
        return {}
    
    # Extract values
    y_true = scores_df['actual_default'].values
    y_prob = scores_df['probability_of_default'].values
    
    # If all values are the same class, some metrics cannot be calculated
    if len(np.unique(y_true)) == 1:
        logger.warning(f"All samples belong to class {y_true[0]}, some metrics cannot be calculated")
        return {
            'total_loans': len(y_true),
            'default_rate': float(y_true.mean()),
            'average_score': float(scores_df['score_value'].mean()),
            'median_score': float(scores_df['score_value'].median()),
            'risk_band_distribution': scores_df['risk_band'].value_counts().to_dict()
        }
    
    # Calculate AUC-ROC
    auc_roc = roc_auc_score(y_true, y_prob)
    
    # Find optimal threshold using precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    
    # Handle case where precision-recall curve has one fewer threshold than precision/recall values
    if len(thresholds) < len(precision):
        thresholds = np.append(thresholds, 1.0)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Binary predictions with optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
    
    # Calculate Kolmogorov-Smirnov statistic
    # Sort by probability of default
    sorted_indices = np.argsort(y_prob)
    sorted_y_true = y_true[sorted_indices]
    cum_pos_rate = np.cumsum(sorted_y_true) / np.sum(sorted_y_true)
    cum_neg_rate = np.cumsum(1 - sorted_y_true) / np.sum(1 - sorted_y_true)
    ks_statistic = np.max(np.abs(cum_pos_rate - cum_neg_rate))
    
    # Calculate Gini coefficient (2*AUC - 1)
    gini_coefficient = 2 * auc_roc - 1
    
    # Risk band distribution
    risk_band_distribution = scores_df['risk_band'].value_counts().to_dict()
    
    # Compile metrics
    metrics = {
        'total_loans': len(y_true),
        'default_rate': float(y_true.mean()),
        'auc_roc': float(auc_roc),
        'ks_statistic': float(ks_statistic),
        'gini_coefficient': float(gini_coefficient),
        'accuracy': float(accuracy),
        'precision': float(precision_score),
        'recall': float(recall_score),
        'f1_score': float(f1),
        'optimal_threshold': float(optimal_threshold),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'average_score': float(scores_df['score_value'].mean()),
        'median_score': float(scores_df['score_value'].median()),
        'risk_band_distribution': risk_band_distribution
    }
    
    return metrics


def calculate_variable_drift(
    reference_data: Optional[pd.DataFrame],
    current_data: Optional[pd.DataFrame],
    feature_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate drift metrics for model features.
    Returns empty dict if data is unavailable.
    
    Args:
        reference_data (Optional[pd.DataFrame]): Reference period data (e.g., features)
        current_data (Optional[pd.DataFrame]): Current period data (e.g., features)
        feature_names (List[str]): List of feature names to check
    
    Returns:
        Dict[str, Dict[str, float]]: Drift metrics by feature
    """
    # If feature data isn't passed (e.g., in synthetic fallback), return empty
    if reference_data is None or current_data is None or reference_data.empty or current_data.empty:
        logger.warning("Cannot calculate variable drift: feature data is missing or empty")
        return {}
    
    # Only use features present in both datasets
    common_features = [f for f in feature_names if f in reference_data.columns and f in current_data.columns]
    
    if not common_features:
        logger.warning("No common features found between reference and current datasets")
        return {}
    
    variable_drift = {}
    
    for feature in common_features:
        ref_values = reference_data[feature].dropna()
        curr_values = current_data[feature].dropna()
        
        if len(ref_values) == 0 or len(curr_values) == 0:
            logger.warning(f"Skipping feature {feature} due to insufficient data")
            continue
        
        # For numeric features, calculate statistical tests
        if np.issubdtype(ref_values.dtype, np.number):
            # Kolmogorov-Smirnov test for distribution equality
            ks_stat, ks_pval = stats.ks_2samp(ref_values, curr_values)
            
            # Mean and standard deviation changes
            ref_mean = ref_values.mean()
            curr_mean = curr_values.mean()
            mean_diff_pct = abs((curr_mean - ref_mean) / (ref_mean + 1e-10)) * 100
            
            ref_std = ref_values.std()
            curr_std = curr_values.std()
            std_diff_pct = abs((curr_std - ref_std) / (ref_std + 1e-10)) * 100
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(ref_values) - 1) * ref_std**2 + 
                                 (len(curr_values) - 1) * curr_std**2) / 
                                (len(ref_values) + len(curr_values) - 2))
            effect_size = abs(curr_mean - ref_mean) / pooled_std if pooled_std > 0 else 0
            
            variable_drift[feature] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'mean_diff_pct': float(mean_diff_pct),
                'std_diff_pct': float(std_diff_pct),
                'effect_size': float(effect_size),
                'ref_mean': float(ref_mean),
                'curr_mean': float(curr_mean),
                'ref_std': float(ref_std),
                'curr_std': float(curr_std)
            }
        
        # For categorical features, calculate chi-square test and distribution changes
        else:
            # Get value counts and align categories
            ref_counts = ref_values.value_counts().to_dict()
            curr_counts = curr_values.value_counts().to_dict()
            
            all_categories = set(ref_counts.keys()) | set(curr_counts.keys())
            
            # Calculate chi-square statistic
            chi2 = 0
            n_ref = len(ref_values)
            n_curr = len(curr_values)
            
            for category in all_categories:
                # Expected vs observed counts
                ref_count = ref_counts.get(category, 0)
                curr_count = curr_counts.get(category, 0)
                
                ref_pct = ref_count / n_ref if n_ref > 0 else 0
                curr_pct = curr_count / n_curr if n_curr > 0 else 0
                
                # Chi-square component
                if ref_pct > 0:
                    chi2 += n_curr * ((curr_pct - ref_pct) ** 2) / ref_pct
            
            variable_drift[feature] = {
                'chi2': float(chi2),
                'n_categories': len(all_categories),
                'distribution_diff': {
                    str(cat): {
                        'ref_pct': float(ref_counts.get(cat, 0) / n_ref if n_ref > 0 else 0),
                        'curr_pct': float(curr_counts.get(cat, 0) / n_curr if n_curr > 0 else 0)
                    }
                    for cat in all_categories
                }
            }
    
    return variable_drift


def generate_plots(
    model_info: Dict[str, Any],
    monitoring_data: Dict[str, Dict[str, Any]],
    periods: List[Tuple[datetime, datetime]],
    output_dir: str
) -> List[str]:
    """
    Generate monitoring plots.
    
    Args:
        model_info (Dict[str, Any]): Model information
        monitoring_data (Dict[str, Dict[str, Any]]): Monitoring data by period
        periods (List[Tuple[datetime, datetime]]): List of periods
        output_dir (str): Directory to save plots
    
    Returns:
        List[str]: List of plot file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_files = []
    
    # Format period labels (e.g., "Jan 2023")
    period_labels = [f"{start.strftime('%b %Y')}" for start, _ in periods]
    
    # Extract metrics over time
    metrics_over_time = {
        'auc_roc': [],
        'ks_statistic': [],
        'gini_coefficient': [],
        'default_rate': [],
        'psi': [],
        'average_score': [],
        'total_loans': []
    }
    
    for period_key in sorted(monitoring_data.keys()):
        period_data = monitoring_data[period_key]
        
        # Add metrics for this period
        for metric in metrics_over_time.keys():
            metrics_over_time[metric].append(period_data.get(metric, 0))
    
    # 1. Performance metrics over time
    plt.figure(figsize=(10, 6))
    for metric in ['auc_roc', 'ks_statistic', 'gini_coefficient']:
        if any(metrics_over_time[metric]):
            plt.plot(period_labels, metrics_over_time[metric], marker='o', label=metric.upper())
    
    plt.title(f"Model Performance Metrics Over Time\nModel: {model_info['model_name']} v{model_info['model_version']}")
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    perf_plot_file = output_dir / f"performance_metrics_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(perf_plot_file)
    plot_files.append(str(perf_plot_file))
    
    # 2. PSI and Default Rate
    plt.figure(figsize=(10, 6))
    
    # PSI on left y-axis
    ax1 = plt.gca()
    ax1.set_xlabel('Period')
    ax1.set_ylabel('PSI', color='blue')
    ax1.plot(period_labels, metrics_over_time['psi'], marker='s', color='blue', label='PSI')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Default rate on right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Default Rate (%)', color='red')
    default_rates_pct = [rate * 100 for rate in metrics_over_time['default_rate']]
    ax2.plot(period_labels, default_rates_pct, marker='^', color='red', label='Default Rate')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title(f"Population Stability and Default Rate\nModel: {model_info['model_name']} v{model_info['model_version']}")
    plt.xticks(rotation=45)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    psi_plot_file = output_dir / f"psi_default_rate_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(psi_plot_file)
    plot_files.append(str(psi_plot_file))
    
    # 3. Score Distribution
    if all('score_distribution' in period_data for period_data in monitoring_data.values()):
        plt.figure(figsize=(12, 6))
        
        for i, period_key in enumerate(sorted(monitoring_data.keys())):
            if 'score_distribution' in monitoring_data[period_key]:
                scores = monitoring_data[period_key]['score_distribution']
                if scores:
                    sns.kdeplot(scores, label=period_labels[i])
        
        plt.title(f"Score Distribution Over Time\nModel: {model_info['model_name']} v{model_info['model_version']}")
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        dist_plot_file = output_dir / f"score_distribution_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(dist_plot_file)
        plot_files.append(str(dist_plot_file))
    
    # Close all plots
    plt.close('all')
    
    return plot_files


def save_monitoring_results(
    model_id: str,
    monitoring_date: datetime,
    period_start: datetime,
    period_end: datetime,
    metrics: Dict[str, Any],
    psi_value: float,
    variable_drift_metrics: Dict[str, Dict[str, float]],
    save_to_db: bool = True # Add flag to control DB saving
) -> None:
    """
    Save monitoring results to the database or a file.
    
    Args:
        model_id (str): ID of the monitored model
        monitoring_date (datetime): Date of monitoring
        period_start (datetime): Start date of the monitoring period
        period_end (datetime): End date of the monitoring period
        metrics (Dict[str, Any]): Performance metrics
        psi_value (float): Population Stability Index
        variable_drift_metrics (Dict[str, Dict[str, float]]): Variable drift metrics
        save_to_db (bool): Attempt to save to database if True
    """
    try:
        # Convert dictionaries to JSON strings for database insertion
        # Ensure complex objects (like DataFrames in psi_bins) are handled
        variable_drift_json = json.dumps(variable_drift_metrics, default=str)
        metrics_json = json.dumps(metrics, default=str)
        
        # Set alert status based on PSI and AUC-ROC
        alert_status = "OK"
        alert_messages = []
        
        if psi_value > 0.25:
            alert_status = "Critical"
            alert_messages.append(f"High PSI value: {psi_value:.4f}")
        elif psi_value > 0.1:
            alert_status = "Warning"
            alert_messages.append(f"Elevated PSI value: {psi_value:.4f}")
        
        if 'auc_roc' in metrics:
            auc_roc = metrics['auc_roc']
            if auc_roc < 0.6:
                alert_status = "Critical"
                alert_messages.append(f"Low AUC-ROC: {auc_roc:.4f}")
            elif auc_roc < 0.7:
                if alert_status != "Critical":
                    alert_status = "Warning"
                alert_messages.append(f"Marginal AUC-ROC: {auc_roc:.4f}")
        
        # Add variable drift alerts
        critical_features = []
        warning_features = []
        
        for feature, drift_metrics in variable_drift_metrics.items():
            if 'ks_pvalue' in drift_metrics and drift_metrics['ks_pvalue'] < 0.01:
                if drift_metrics.get('effect_size', 0) > 0.5:
                    critical_features.append(feature)
                elif drift_metrics.get('effect_size', 0) > 0.2:
                    warning_features.append(feature)
            elif 'chi2' in drift_metrics and drift_metrics['chi2'] > 20: # Example threshold
                critical_features.append(feature)
        
        if critical_features:
            alert_status = "Critical"
            alert_messages.append(f"Critical drift in features: {', '.join(critical_features)}")
        elif warning_features and alert_status != "Critical":
            alert_status = "Warning"
            alert_messages.append(f"Significant drift in features: {', '.join(warning_features)}")
        
        # Prepare data for saving
        monitoring_id = str(uuid.uuid4())
        params = {
            'monitoring_id': monitoring_id,
            'model_id': model_id,
            'monitoring_date': monitoring_date.isoformat(),
            'period_start': period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'total_customers': metrics.get('total_loans', 0), # Using total_loans as proxy
            'total_applications': 0, # Not calculated
            'total_loans': metrics.get('total_loans', 0),
            'population_stability_index': float(psi_value) if not np.isnan(psi_value) else None,
            'auc_roc': metrics.get('auc_roc', None),
            'ks_statistic': metrics.get('ks_statistic', None),
            'gini_coefficient': metrics.get('gini_coefficient', None),
            'average_score': metrics.get('average_score', None),
            'median_score': metrics.get('median_score', None),
            'variable_drift_metrics': variable_drift_json,
            'alert_status': alert_status,
            'alert_message': "; ".join(alert_messages) if alert_messages else None,
            # Add raw metrics for file saving
            'raw_metrics': metrics
        }

        # Attempt to save to database
        if save_to_db:
            try:
                insert_query = """
                INSERT INTO model_monitoring (
                    monitoring_id, model_id, monitoring_date, period_start, period_end,
                    total_customers, total_applications, total_loans,
                    population_stability_index, auc_roc, ks_statistic, gini_coefficient,
                    average_score, median_score, variable_drift_metrics,
                    alert_status, alert_message
                ) VALUES (
                    :monitoring_id, :model_id, :monitoring_date, :period_start, :period_end,
                    :total_customers, :total_applications, :total_loans,
                    :population_stability_index, :auc_roc, :ks_statistic, :gini_coefficient,
                    :average_score, :median_score, :variable_drift_metrics,
                    :alert_status, :alert_message
                )
                """
                # Remove raw_metrics before db insert
                db_params = {k: v for k, v in params.items() if k != 'raw_metrics'}
                execute_statement(insert_query, db_params)
                logger.info(f"Saved monitoring results to database with ID {monitoring_id}")
                return # Exit if DB save is successful
            except Exception as db_error:
                logger.warning(f"Could not save monitoring results to database: {db_error}. Falling back to file.")
        
        # Fallback to saving to JSON file
        output_file = REPORTS_DIR / f"monitoring_report_{model_id}_{period_end.strftime('%Y%m%d')}.json"
        try:
            with open(output_file, 'w') as f:
                # Save the full params dict including raw_metrics
                json.dump(params, f, indent=2, default=str)
            logger.info(f"Saved monitoring results to file: {output_file}")
        except Exception as file_error:
            logger.error(f"Error saving monitoring results to file {output_file}: {file_error}")

    except Exception as e:
        logger.error(f"Error preparing monitoring results: {e}")
        # Do not raise here, just log the error


def main():
    """Main function to monitor model performance."""
    args = parse_arguments()
    
    try:
        # Get model information (handles DB/filesystem fallback)
        try:
            model_info = get_model_info(args.model_id)
            model_id = model_info['model_id'] # Use the definitive ID found
            logger.info(f"Monitoring model: {model_info['model_name']} v{model_info['model_version']} (ID: {model_id})")
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Error retrieving model info: {e}")
            # Try to list available models from filesystem
            try:
                logger.info("Available models in filesystem:")
                for model_dir in sorted(MODELS_DIR.glob("*")):
                     logger.info(f"  - {model_dir.name} (use name or full path)")
            except Exception:
                logger.info("Could not list models in filesystem.")
            sys.exit(1)
            
        # Get feature names for drift calculation
        feature_names = model_info.get('features_list', [])
        if not feature_names:
            logger.warning("Feature list not found in model metadata. Cannot calculate variable drift.")
        
        # Calculate period dates
        current_end, current_start, periods = calculate_period_dates(args.period, args.lookback)
        
        if not periods:
             logger.error("Could not determine monitoring periods.")
             sys.exit(1)
             
        # Get reference period (typically the earliest period)
        reference_start, reference_end = periods[-1]
        logger.info(f"Reference period: {reference_start.isoformat()} to {reference_end.isoformat()}")
        
        # Get reference scores (handles DB/synthetic fallback)
        reference_scores = get_scores_for_period(model_id, reference_start, reference_end)
        
        # NOTE: Variable drift needs feature data, which isn't fetched here.
        # If needed, `get_scores_for_period` would need modification to fetch features.
        reference_feature_data = None # Placeholder - requires feature data fetch
        
        if reference_scores.empty:
            logger.warning("Reference period has no scores (real or synthetic). PSI/Drift calculations might be affected.")
        
        # Process each period
        monitoring_data = {}
        all_period_data = {} # Store data for plotting
        
        for i, (start_date, end_date) in enumerate(reversed(periods)): # Process oldest first
            period_key = f"{start_date.isoformat()}_{end_date.isoformat()}"
            logger.info(f"Processing period {len(periods)-i}/{len(periods)}: {start_date.isoformat()} to {end_date.isoformat()}")
            
            # Get scores for this period (handles DB/synthetic fallback)
            period_scores = get_scores_for_period(model_id, start_date, end_date)
            current_feature_data = None # Placeholder - requires feature data fetch
            
            if period_scores.empty:
                logger.warning(f"No scores found for period {start_date.isoformat()} to {end_date.isoformat()}")
                # Store empty results for this period for plotting consistency
                all_period_data[period_key] = {'psi': 0, 'auc_roc': 0, 'ks_statistic': 0, 'gini_coefficient': 0, 'default_rate': 0, 'average_score': 0, 'total_loans': 0}
                continue
            
            # Calculate performance metrics
            performance_metrics = calculate_performance_metrics(period_scores)
            
            # Calculate PSI against reference period
            psi_value, psi_bins = 0.0, pd.DataFrame()
            if not reference_scores.empty:
                psi_value, psi_bins = calculate_population_stability_index(reference_scores, period_scores)
            else:
                logger.warning("Skipping PSI calculation due to empty reference scores.")
            
            # Calculate variable drift (will be empty if feature data is None)
            variable_drift = calculate_variable_drift(
                reference_feature_data, 
                current_feature_data, 
                feature_names
            )
            
            # Store results for this period
            period_results = {
                **performance_metrics,
                'psi': float(psi_value) if not np.isnan(psi_value) else 0.0,
                'psi_bins': psi_bins.to_dict('records') if not psi_bins.empty else [],
                'variable_drift': variable_drift,
                # Store raw scores distribution for plotting if needed
                'score_distribution': period_scores['score_value'].tolist() 
            }
            monitoring_data[period_key] = period_results
            all_period_data[period_key] = period_results # Add to plot data
            
            # Save monitoring results to the database/file for the *current* actual period only
            is_current_period = (start_date == current_start and end_date == current_end)
            if is_current_period:
                logger.info(f"Saving results for current period: {start_date.isoformat()} to {end_date.isoformat()}")
                save_monitoring_results(
                    model_id,
                    datetime.now().date(),
                    start_date,
                    end_date,
                    performance_metrics,
                    psi_value,
                    variable_drift,
                    save_to_db=True # Attempt DB save for current period
                )
        
        # Check if any data was processed
        if not all_period_data:
            logger.error("No data processed for any monitoring period.")
            sys.exit(1)

        # Generate plots if requested
        if args.generate_plots:
            logger.info("Generating plots...")
            try:
                 # We need consistent periods for plotting, pass original list
                 plot_period_keys = [f"{s.isoformat()}_{e.isoformat()}" for s, e in reversed(periods)]
                 # Ensure data for all plot periods exists, filling gaps with zeros
                 plot_data = {p_key: all_period_data.get(p_key, {'psi': 0, 'auc_roc': 0, 'ks_statistic': 0, 'gini_coefficient': 0, 'default_rate': 0, 'average_score': 0, 'total_loans': 0}) for p_key in plot_period_keys}
                 
                 plot_files = generate_plots(
                    model_info,
                    plot_data, # Use potentially gap-filled data
                    list(reversed(periods)), # Match order of plot_data keys
                    args.output_dir
                )
                 logger.info(f"Generated {len(plot_files)} monitoring plots in {args.output_dir}")
            except Exception as plot_error:
                 logger.error(f"Error generating plots: {plot_error}")
        
        logger.info("Model monitoring completed successfully")
    
    except Exception as e:
        logger.error(f"Error in model monitoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 