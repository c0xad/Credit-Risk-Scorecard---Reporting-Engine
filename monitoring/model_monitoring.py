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
        - List of (start_date, end_date) tuples for all periods
    """
    today = datetime.now().date()
    
    if period == 'daily':
        current_end = today
        current_start = today - timedelta(days=1)
        periods = [(today - timedelta(days=i+1), today - timedelta(days=i)) 
                  for i in range(lookback)]
    
    elif period == 'weekly':
        # Current week end is today, start is 7 days ago
        current_end = today
        current_start = today - timedelta(days=7)
        periods = [(today - timedelta(days=(i+1)*7), today - timedelta(days=i*7)) 
                  for i in range(lookback)]
    
    elif period == 'monthly':
        # Current month
        current_end = today
        # Approximate 30 days for a month
        current_start = today.replace(day=1)
        
        periods = []
        for i in range(lookback):
            # Get end of month
            if i == 0:
                end_date = today
            else:
                # Last day of previous month
                month = (today.month - i) % 12 or 12
                year = today.year - ((today.month - i - 1) // 12)
                if month == 12:
                    end_date = datetime(year, month, 31).date()
                else:
                    end_date = datetime(year, month + 1, 1).date() - timedelta(days=1)
            
            # Get start of month
            month = (today.month - i - 1) % 12 or 12
            year = today.year - ((today.month - i - 1) // 12)
            start_date = datetime(year, month, 1).date()
            
            periods.append((start_date, end_date))
    
    else:
        raise ValueError(f"Unsupported period type: {period}")
    
    return current_end, current_start, periods


def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get model information from the database.
    
    Args:
        model_id (str): ID of the model to monitor
    
    Returns:
        Dict[str, Any]: Model information
    """
    query = """
    SELECT model_id, model_name, model_version, model_type, training_date, 
           status, features_list, performance_metrics
    FROM risk_model
    WHERE model_id = :model_id
    """
    
    try:
        result = execute_query(query, {'model_id': model_id})
        
        if result.empty:
            raise ValueError(f"No model found with ID: {model_id}")
        
        model_info = result.iloc[0].to_dict()
        
        # Parse JSON fields
        for field in ['features_list', 'performance_metrics']:
            if model_info[field] and isinstance(model_info[field], str):
                model_info[field] = json.loads(model_info[field])
        
        return model_info
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise


def get_scores_for_period(model_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Get model scores for a specific period.
    
    Args:
        model_id (str): ID of the model to monitor
        start_date (datetime): Start date of the period
        end_date (datetime): End date of the period
    
    Returns:
        pd.DataFrame: Model scores for the period
    """
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
    
    try:
        params = {
            'model_id': model_id,
            'start_date': start_date,
            'end_date': end_date
        }
        
        result = execute_query(query, params)
        logger.info(f"Retrieved {len(result)} scores for period {start_date} to {end_date}")
        return result
    
    except Exception as e:
        logger.error(f"Error getting scores for period: {e}")
        raise


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
    reference_data: pd.DataFrame, 
    current_data: pd.DataFrame,
    feature_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate drift metrics for model features.
    
    Args:
        reference_data (pd.DataFrame): Reference period data
        current_data (pd.DataFrame): Current period data
        feature_names (List[str]): List of feature names to check
    
    Returns:
        Dict[str, Dict[str, float]]: Drift metrics by feature
    """
    if reference_data.empty or current_data.empty:
        logger.warning("Cannot calculate variable drift: one or both datasets are empty")
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
    variable_drift_metrics: Dict[str, Dict[str, float]]
) -> None:
    """
    Save monitoring results to the database.
    
    Args:
        model_id (str): ID of the monitored model
        monitoring_date (datetime): Date of monitoring
        period_start (datetime): Start date of the monitoring period
        period_end (datetime): End date of the monitoring period
        metrics (Dict[str, Any]): Performance metrics
        psi_value (float): Population Stability Index
        variable_drift_metrics (Dict[str, Dict[str, float]]): Variable drift metrics
    """
    try:
        # Convert dictionaries to JSON strings
        variable_drift_json = json.dumps(variable_drift_metrics)
        
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
            elif 'chi2' in drift_metrics and drift_metrics['chi2'] > 20:
                critical_features.append(feature)
        
        if critical_features:
            alert_status = "Critical"
            alert_messages.append(f"Critical drift in features: {', '.join(critical_features)}")
        elif warning_features and alert_status != "Critical":
            alert_status = "Warning"
            alert_messages.append(f"Significant drift in features: {', '.join(warning_features)}")
        
        # Create monitoring record
        monitoring_id = str(uuid.uuid4())
        
        # Insert into model_monitoring table
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
        
        params = {
            'monitoring_id': monitoring_id,
            'model_id': model_id,
            'monitoring_date': monitoring_date,
            'period_start': period_start,
            'period_end': period_end,
            'total_customers': metrics.get('total_loans', 0),  # Using total_loans as a proxy
            'total_applications': 0,  # Not calculated in this script
            'total_loans': metrics.get('total_loans', 0),
            'population_stability_index': psi_value,
            'auc_roc': metrics.get('auc_roc', 0),
            'ks_statistic': metrics.get('ks_statistic', 0),
            'gini_coefficient': metrics.get('gini_coefficient', 0),
            'average_score': metrics.get('average_score', 0),
            'median_score': metrics.get('median_score', 0),
            'variable_drift_metrics': variable_drift_json,
            'alert_status': alert_status,
            'alert_message': "; ".join(alert_messages) if alert_messages else None
        }
        
        execute_statement(insert_query, params)
        logger.info(f"Saved monitoring results to database with ID {monitoring_id}")
    
    except Exception as e:
        logger.error(f"Error saving monitoring results to database: {e}")
        raise


def main():
    """Main function to monitor model performance."""
    args = parse_arguments()
    
    try:
        # Get model information
        model_info = get_model_info(args.model_id)
        logger.info(f"Monitoring model: {model_info['model_name']} v{model_info['model_version']}")
        
        # Calculate period dates
        current_end, current_start, periods = calculate_period_dates(args.period, args.lookback)
        
        # Get reference period (typically the earliest period)
        reference_start, reference_end = periods[-1]
        logger.info(f"Reference period: {reference_start} to {reference_end}")
        
        # Get reference scores
        reference_scores = get_scores_for_period(args.model_id, reference_start, reference_end)
        
        # Get scores for each period and calculate metrics
        monitoring_data = {}
        
        for i, (start_date, end_date) in enumerate(periods):
            period_key = f"{start_date}_{end_date}"
            logger.info(f"Processing period {i+1}/{len(periods)}: {start_date} to {end_date}")
            
            # Get scores for this period
            period_scores = get_scores_for_period(args.model_id, start_date, end_date)
            
            if period_scores.empty:
                logger.warning(f"No scores found for period {start_date} to {end_date}")
                continue
            
            # Calculate performance metrics
            performance_metrics = calculate_performance_metrics(period_scores)
            
            # Calculate PSI against reference period
            psi_value, psi_bins = calculate_population_stability_index(reference_scores, period_scores)
            
            # Calculate variable drift
            variable_drift = {}
            # This would require additional data about the features
            
            # Store monitoring data for this period
            monitoring_data[period_key] = {
                **performance_metrics,
                'psi': psi_value,
                'psi_bins': psi_bins.to_dict('records') if not psi_bins.empty else [],
                'variable_drift': variable_drift,
                'score_distribution': period_scores['score_value'].tolist()
            }
            
            # Save monitoring results to the database for the current period only
            if i == 0:  # Current period
                save_monitoring_results(
                    args.model_id,
                    datetime.now().date(),
                    start_date,
                    end_date,
                    performance_metrics,
                    psi_value,
                    variable_drift
                )
        
        # Generate plots if requested
        if args.generate_plots:
            plot_files = generate_plots(
                model_info,
                monitoring_data,
                periods,
                args.output_dir
            )
            logger.info(f"Generated {len(plot_files)} monitoring plots")
        
        logger.info("Model monitoring completed successfully")
    
    except Exception as e:
        logger.error(f"Error in model monitoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 