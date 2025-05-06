#!/usr/bin/env python3
"""
Generate credit risk scorecard reports.
This script creates HTML and PDF reports based on model performance data.
"""

import os
import sys
import uuid
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv

# Attempt to import weasyprint, but don't fail if it's not fully functional
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except OSError as e:
    # This likely means GTK dependencies are missing
    logging.warning(f"Could not load weasyprint PDF backend: {e}")
    logging.warning("PDF generation will be disabled. Please install GTK+ runtime libraries (see WeasyPrint documentation) to enable PDF reports.")
    WEASYPRINT_AVAILABLE = False
except ImportError:
    logging.warning("weasyprint library not installed. PDF generation disabled.")
    WEASYPRINT_AVAILABLE = False

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

# Directories
TEMPLATE_DIR = PROJECT_ROOT / "reporting" / "templates"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "reporting" / "outputs"
CHARTS_DIR = PROJECT_ROOT / "reporting" / "charts"

# Models directory (needed for fallback)
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
# Monitoring reports directory (needed for fallback)
MONITORING_REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"

# Create directories if they don't exist
REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate credit risk scorecard reports')
    parser.add_argument('--model-id', type=str, required=True,
                       help='ID of the model to report on')
    parser.add_argument('--report-type', type=str, choices=['scorecard', 'monitoring', 'summary'],
                       default='scorecard', help='Type of report to generate')
    parser.add_argument('--output-format', type=str, choices=['html', 'pdf', 'both'],
                       default='both', help='Output format for the report')
    parser.add_argument('--period-start', type=str,
                       help='Start date for the report period (YYYY-MM-DD)')
    parser.add_argument('--period-end', type=str,
                       help='End date for the report period (YYYY-MM-DD)')
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


def get_model_info(model_id_or_name: str) -> Dict[str, Any]:
    """
    Get model information from the database or fallback to metadata file.
    
    Args:
        model_id_or_name (str): ID or name of the model
    
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
        WHERE model_id = :id_or_name OR model_name = :id_or_name
        ORDER BY training_date DESC
        LIMIT 1
        """
        result = execute_query(query, {'id_or_name': model_id_or_name})
        
        if not result.empty:
            model_info = result.iloc[0].to_dict()
            logger.info(f"Fetched model info for {model_id_or_name} from database.")
            # Parse JSON fields
            for field in ['features_list', 'performance_metrics']:
                if model_info[field] and isinstance(model_info[field], str):
                    try:
                        model_info[field] = json.loads(model_info[field])
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse JSON for field {field}")
                        model_info[field] = None
            return model_info
        else:
             logger.warning(f"Model {model_id_or_name} not found in database. Trying filesystem fallback.")

    except Exception as e:
        logger.warning(f"Error getting model info from database: {e}. Trying filesystem fallback.")

    # Filesystem fallback
    try:
        model_path = find_model_path(model_id_or_name)
        metadata_file = model_path / 'metadata.json'
        db_metadata_file = model_path / 'db_metadata.json' # Fallback for db info

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded model info for {model_id_or_name} from {metadata_file}")
            
            db_info = {}
            if db_metadata_file.exists():
                 with open(db_metadata_file, 'r') as f:
                    db_info = json.load(f)
            
            # Combine info
            model_info = {
                'model_id': metadata.get('model_id', db_info.get('model_id', 'unknown_id')),
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
         raise ValueError(f"Could not retrieve information for model: {model_id_or_name}") from e
    except Exception as e:
        logger.error(f"Error during filesystem fallback: {e}")
        raise ValueError(f"Could not retrieve information for model: {model_id_or_name}") from e


def get_model_monitoring_data(model_id: str, period_start: Optional[str] = None, period_end: Optional[str] = None) -> Dict[str, Any]:
    """
    Get model monitoring data from the database or fallback to JSON report files.
    
    Args:
        model_id (str): ID of the model
        period_start (Optional[str]): Start date for the report period
        period_end (Optional[str]): End date for the report period
    
    Returns:
        Dict[str, Any]: Model monitoring data (latest record, previous, trend)
    """
    monitoring_records = []
    
    # Try fetching from database first
    try:
        # Build WHERE clause based on date parameters
        where_clause = "WHERE model_id = :model_id"
        params = {'model_id': model_id}
        
        if period_start:
            where_clause += " AND monitoring_date >= :period_start"
            params['period_start'] = period_start
        
        if period_end:
            where_clause += " AND monitoring_date <= :period_end"
            params['period_end'] = period_end
        
        query = f"""
        SELECT monitoring_id, model_id, monitoring_date, period_start, period_end,
               total_customers, total_loans, population_stability_index,
               auc_roc, ks_statistic, gini_coefficient, average_score, median_score,
               variable_drift_metrics, alert_status, alert_message
        FROM model_monitoring
        {where_clause}
        ORDER BY monitoring_date DESC
        LIMIT 6
        """
        
        result = execute_query(query, params)
        
        if not result.empty:
            logger.info(f"Fetched {len(result)} monitoring records from database.")
            # Convert to list of dictionaries
            for _, row in result.iterrows():
                record = row.to_dict()
                # Parse JSON fields
                if record.get('variable_drift_metrics') and isinstance(record['variable_drift_metrics'], str):
                    try:
                        record['variable_drift_metrics'] = json.loads(record['variable_drift_metrics'])
                    except json.JSONDecodeError:
                        record['variable_drift_metrics'] = {}
                monitoring_records.append(record)
        else:
            logger.warning(f"No monitoring data found in database for model {model_id}. Trying filesystem fallback.")
            
    except Exception as e:
        logger.warning(f"Error getting monitoring data from database: {e}. Trying filesystem fallback.")

    # Filesystem fallback if DB fails or returns no records
    if not monitoring_records:
        try:
            # Find the latest monitoring report file for this model
            report_files = list(MONITORING_REPORTS_DIR.glob(f"monitoring_report_{model_id}_*.json"))
            if not report_files:
                 logger.warning(f"No monitoring report files found in {MONITORING_REPORTS_DIR} for model {model_id}")
                 return {} # Return empty if no fallback available
                 
            # Sort by date in filename (assuming YYYYMMDD format)
            report_files.sort(key=lambda x: x.name.split('_')[-1].split('.')[0], reverse=True)
            
            # Load the latest file (and potentially previous ones if needed for trend)
            # For simplicity, load only the latest for now
            latest_file = report_files[0]
            logger.info(f"Loading monitoring data from file: {latest_file}")
            with open(latest_file, 'r') as f:
                file_data = json.load(f)
            
            # Reconstruct a record similar to DB output
            # Ensure 'raw_metrics' exists for trend calculation
            latest_record = {
                 **{k: file_data.get(k) for k in [
                    'monitoring_id', 'model_id', 'monitoring_date', 'period_start', 'period_end',
                    'total_customers', 'total_loans', 'population_stability_index',
                    'auc_roc', 'ks_statistic', 'gini_coefficient', 'average_score', 'median_score',
                    'variable_drift_metrics', 'alert_status', 'alert_message'
                 ]}, 
                 **file_data.get('raw_metrics', {}) # Merge raw metrics directly
             }
            monitoring_records.append(latest_record)
            # Note: This simple fallback only loads the latest record, trend calculation might be limited.
            
        except Exception as file_error:
            logger.error(f"Error loading monitoring data from file: {file_error}")
            return {} # Return empty on file error

    # Process the retrieved records (either from DB or file)
    if not monitoring_records:
        logger.warning(f"No monitoring data available for model {model_id}")
        return {}
        
    latest_record = monitoring_records[0]
    
    # Prepare monitoring data structure for the report
    monitoring_data = {
        'records': monitoring_records, # Full list for potential future use
        'latest': latest_record,
        'has_previous': len(monitoring_records) > 1,
        'previous': monitoring_records[1] if len(monitoring_records) > 1 else None,
        'trend': calculate_performance_trend(monitoring_records)
    }
    
    return monitoring_data


def get_score_distribution(model_id: str, period_start: Optional[str] = None, period_end: Optional[str] = None) -> pd.DataFrame:
    """
    Get score distribution data from the database or generate synthetic data.
    
    Args:
        model_id (str): ID of the model
        period_start (Optional[str]): Start date for the report period
        period_end (Optional[str]): End date for the report period
    
    Returns:
        pd.DataFrame: Score distribution data grouped by risk band
    """
    try:
        # Build WHERE clause based on date parameters
        where_clause = "WHERE model_id = :model_id"
        params = {'model_id': model_id}
        
        if period_start:
            where_clause += " AND score_date >= :period_start"
            params['period_start'] = period_start
        
        if period_end:
            where_clause += " AND score_date <= :period_end"
            params['period_end'] = period_end
        
        query = f"""
        SELECT score_value, probability_of_default, risk_band, 
               COUNT(*) as count, AVG(score_value) as avg_score
        FROM risk_score
        {where_clause}
        GROUP BY risk_band
        ORDER BY MIN(probability_of_default) ASC -- Order bands correctly
        """
        
        result = execute_query(query, params)
        
        if not result.empty:
            logger.info(f"Fetched score distribution data from DB for model {model_id}.")
            return result
        else:
            logger.warning(f"No score distribution data found in DB for model {model_id}. Generating synthetic data.")
            return generate_synthetic_score_distribution()
    
    except Exception as e:
        logger.warning(f"Error getting score distribution data from DB: {e}. Generating synthetic data.")
        return generate_synthetic_score_distribution()


def generate_synthetic_score_distribution(num_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic score distribution data.
    
    Args:
        num_samples (int): Number of synthetic scores to base distribution on
    
    Returns:
        pd.DataFrame: Synthetic score distribution data
    """
    np.random.seed(43) # Use a different seed
    
    # Generate plausible probabilities of default (e.g., beta distribution)
    a, b = 2, 5 
    probabilities = np.random.beta(a, b, num_samples)
    score_values = 850 - (probabilities * 550).astype(int)
    score_values = np.clip(score_values, 300, 850)
    
    # Define risk bands
    bins = [0, 0.1, 0.2, 0.4, 0.6, 1.0]
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    risk_bands = pd.cut(probabilities, bins=bins, labels=labels, include_lowest=True)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'probability_of_default': probabilities,
        'score_value': score_values,
        'risk_band': risk_bands
    })
    
    # Group by risk band
    distribution = df.groupby('risk_band').agg(
        count=('score_value', 'count'),
        avg_score=('score_value', 'mean'),
        min_prob=('probability_of_default', 'min') # For sorting
    ).reset_index()
    
    # Sort by the min probability in the band to get the correct order
    distribution = distribution.sort_values('min_prob')
    
    logger.info("Generated synthetic score distribution data.")
    return distribution[['risk_band', 'count', 'avg_score']] # Return relevant columns


def get_segmentation_data(model_id: str, period_start: Optional[str] = None, period_end: Optional[str] = None) -> pd.DataFrame:
    """
    Get segmentation data from the database or generate synthetic data.
    
    Args:
        model_id (str): ID of the model
        period_start (Optional[str]): Start date for the report period
        period_end (Optional[str]): End date for the report period
    
    Returns:
        pd.DataFrame: Segmentation data
    """
    try:
        # Build WHERE clause based on date parameters
        where_clause = "WHERE rs.model_id = :model_id"
        params = {'model_id': model_id}
        
        if period_start:
            where_clause += " AND rs.score_date >= :period_start"
            params['period_start'] = period_start
        
        if period_end:
            where_clause += " AND rs.score_date <= :period_end"
            params['period_end'] = period_end
        
        # Query segments by loan type and risk band
        query = f"""
        SELECT la.loan_type as segment_name, rs.risk_band,
               COUNT(*) as count,
               AVG(rs.score_value) as avg_score,
               SUM(CASE WHEN ld.loan_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 100 as default_rate
        FROM risk_score rs
        JOIN loan_account la ON rs.loan_id = la.loan_id
        LEFT JOIN (
            SELECT DISTINCT loan_id
            FROM loan_delinquency
            WHERE days_past_due >= 90
        ) ld ON rs.loan_id = ld.loan_id
        {where_clause}
        GROUP BY la.loan_type, rs.risk_band
        ORDER BY la.loan_type, rs.risk_band
        """
        
        result = execute_query(query, params)
        
        if not result.empty:
            logger.info(f"Fetched segmentation data from DB for model {model_id}.")
            return result
        else:
            logger.warning(f"No segmentation data found in DB for model {model_id}. Generating synthetic data.")
            return generate_synthetic_segmentation_data()
    
    except Exception as e:
        logger.warning(f"Error getting segmentation data from DB: {e}. Generating synthetic data.")
        return generate_synthetic_segmentation_data()


def generate_synthetic_segmentation_data(num_segments: int = 4, num_bands: int = 5) -> pd.DataFrame:
    """
    Generate synthetic segmentation data.
    
    Args:
        num_segments (int): Number of segments (e.g., loan types)
        num_bands (int): Number of risk bands
    
    Returns:
        pd.DataFrame: Synthetic segmentation data
    """
    np.random.seed(44)
    segments = [f'Segment_{chr(65+i)}' for i in range(num_segments)]
    risk_bands = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    
    data = []
    for seg in segments:
        # Distribute counts across bands (e.g., more in medium/high)
        counts = np.random.multinomial(500, pvals=[0.1, 0.2, 0.3, 0.25, 0.15]) # Example distribution
        for i, band in enumerate(risk_bands):
            count = counts[i]
            if count == 0: continue
            # Simulate score and default rate based on band
            avg_score = 750 - i * 80 + np.random.randint(-20, 20)
            default_rate = (i * 15 + np.random.uniform(0, 10)) # Higher rate for higher risk band
            data.append({
                'segment_name': seg,
                'risk_band': band,
                'count': count,
                'avg_score': np.clip(avg_score, 300, 850),
                'default_rate': np.clip(default_rate, 0, 100)
            })
    
    logger.info("Generated synthetic segmentation data.")
    return pd.DataFrame(data)


def calculate_performance_trend(monitoring_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate performance trend from monitoring records.
    
    Args:
        monitoring_records (List[Dict[str, Any]]): List of monitoring records
    
    Returns:
        Dict[str, Any]: Performance trend data
    """
    if len(monitoring_records) < 2:
        return {}
    
    # Calculate trend for key metrics
    metrics = ['auc_roc', 'ks_statistic', 'gini_coefficient', 'average_score', 'population_stability_index']
    trend = {}
    
    for metric in metrics:
        values = [float(record.get(metric, 0)) for record in monitoring_records if metric in record]
        
        if len(values) >= 2:
            trend[metric] = {
                'current': values[0],
                'previous': values[1],
                'change': values[0] - values[1],
                'change_pct': ((values[0] - values[1]) / values[1] * 100) if values[1] else 0,
                'direction': 'up' if values[0] > values[1] else 'down' if values[0] < values[1] else 'stable',
                'values': values
            }
    
    return trend


def generate_performance_chart(monitoring_data: Dict[str, Any], chart_path: str) -> str:
    """
    Generate performance chart for the report.
    
    Args:
        monitoring_data (Dict[str, Any]): Model monitoring data
        chart_path (str): Path to save the chart
    
    Returns:
        str: Path to the generated chart
    """
    if not monitoring_data or 'records' not in monitoring_data or len(monitoring_data['records']) < 2:
        logger.warning("Insufficient data for performance chart")
        return ""
    
    # Prepare data
    records = monitoring_data['records']
    dates = [record['monitoring_date'] for record in records]
    dates.reverse()  # Show oldest to newest
    
    metrics = {
        'AUC-ROC': [record.get('auc_roc', 0) for record in records],
        'KS Statistic': [record.get('ks_statistic', 0) for record in records],
        'Gini Coefficient': [record.get('gini_coefficient', 0) for record in records]
    }
    
    # Reverse lists to match dates
    for key in metrics:
        metrics[key].reverse()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot each metric
    for metric_name, values in metrics.items():
        plt.plot(dates, values, marker='o', label=metric_name)
    
    plt.title('Model Performance Metrics Over Time')
    plt.xlabel('Monitoring Date')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save chart
    chart_file = os.path.join(chart_path, 'performance_metrics.png')
    plt.savefig(chart_file)
    plt.close()
    
    return chart_file


def generate_feature_importance_chart(model_info: Dict[str, Any], chart_path: str) -> str:
    """
    Generate feature importance chart for the report.
    
    Args:
        model_info (Dict[str, Any]): Model information
        chart_path (str): Path to save the chart
    
    Returns:
        str: Path to the generated chart
    """
    if not model_info or 'performance_metrics' not in model_info or 'feature_importance' not in model_info['performance_metrics']:
        logger.warning("Feature importance data not available")
        return ""
    
    # Get feature importance data
    feature_importance = model_info['performance_metrics']['feature_importance']
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False).head(15)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create horizontal bar chart
    plt.barh(df['Feature'], df['Importance'], color='skyblue')
    
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save chart
    chart_file = os.path.join(chart_path, 'feature_importance.png')
    plt.savefig(chart_file)
    plt.close()
    
    return chart_file


def generate_stability_chart(monitoring_data: Dict[str, Any], chart_path: str) -> str:
    """
    Generate stability chart for the report.
    
    Args:
        monitoring_data (Dict[str, Any]): Model monitoring data
        chart_path (str): Path to save the chart
    
    Returns:
        str: Path to the generated chart
    """
    if not monitoring_data or 'records' not in monitoring_data or len(monitoring_data['records']) < 2:
        logger.warning("Insufficient data for stability chart")
        return ""
    
    # Prepare data
    records = monitoring_data['records']
    dates = [record['monitoring_date'] for record in records]
    dates.reverse()  # Show oldest to newest
    
    psi_values = [float(record.get('population_stability_index', 0)) for record in records]
    psi_values.reverse()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot PSI values
    plt.plot(dates, psi_values, marker='s', color='blue', label='PSI')
    
    # Add reference lines
    plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
    plt.axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
    
    plt.title('Population Stability Index Over Time')
    plt.xlabel('Monitoring Date')
    plt.ylabel('PSI Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save chart
    chart_file = os.path.join(chart_path, 'stability_chart.png')
    plt.savefig(chart_file)
    plt.close()
    
    return chart_file


def generate_segmentation_chart(segmentation_data: pd.DataFrame, chart_path: str) -> str:
    """
    Generate segmentation chart for the report.
    
    Args:
        segmentation_data (pd.DataFrame): Segmentation data
        chart_path (str): Path to save the chart
    
    Returns:
        str: Path to the generated chart
    """
    if segmentation_data.empty:
        logger.warning("No segmentation data available for chart")
        return ""
    
    # Create a pivot table
    pivot = segmentation_data.pivot_table(
        index='segment_name', 
        columns='risk_band', 
        values='count', 
        aggfunc='sum',
        fill_value=0
    )
    
    # Ensure all risk bands are present
    all_risk_bands = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    for band in all_risk_bands:
        if band not in pivot.columns:
            pivot[band] = 0
    
    # Reorder columns
    pivot = pivot[all_risk_bands]
    
    # Calculate percentages
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create stacked bar chart
    ax = pivot_pct.plot(
        kind='barh', 
        stacked=True,
        color=['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
    )
    
    plt.title('Risk Band Distribution by Segment')
    plt.xlabel('Percentage (%)')
    plt.ylabel('Segment')
    plt.legend(title='Risk Band')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Add percentage labels
    for i, (segment, row) in enumerate(pivot_pct.iterrows()):
        cumulative = 0
        for band in all_risk_bands:
            value = row[band]
            if value > 5:  # Only show labels for segments with significant percentage
                plt.text(cumulative + value/2, i, f'{value:.1f}%', va='center', ha='center')
            cumulative += value
    
    plt.tight_layout()
    
    # Save chart
    chart_file = os.path.join(chart_path, 'segmentation_chart.png')
    plt.savefig(chart_file)
    plt.close()
    
    return chart_file


def prepare_scorecard_data(
    model_info: Dict[str, Any],
    monitoring_data: Dict[str, Any],
    score_distribution: pd.DataFrame,
    segmentation_data: pd.DataFrame,
    chart_path: str
) -> Dict[str, Any]:
    """
    Prepare data for the scorecard template.
    
    Args:
        model_info (Dict[str, Any]): Model information
        monitoring_data (Dict[str, Any]): Model monitoring data
        score_distribution (pd.DataFrame): Score distribution data
        segmentation_data (pd.DataFrame): Segmentation data
        chart_path (str): Path to save charts
    
    Returns:
        Dict[str, Any]: Data for the scorecard template
    """
    charts = {}
    try:
        charts['performance'] = generate_performance_chart(monitoring_data, chart_path)
        charts['feature_importance'] = generate_feature_importance_chart(model_info, chart_path)
        charts['stability'] = generate_stability_chart(monitoring_data, chart_path)
        charts['segmentation'] = generate_segmentation_chart(segmentation_data, chart_path)
    except Exception as e:
        logger.error(f"Error generating charts: {e}", exc_info=True)
    
    # Get latest monitoring record
    latest_monitoring = monitoring_data.get('latest', {})
    
    # Get risk band distribution
    risk_bands = {}
    if not score_distribution.empty:
        total_count = score_distribution['count'].sum()
        for _, row in score_distribution.iterrows():
            band = row['risk_band']
            count = row['count']
            pct = (count / total_count) * 100 if total_count > 0 else 0
            risk_bands[band.lower().replace(' ', '_')] = round(pct, 1)
    
    # Ensure all risk bands have values
    for band in ['very_low', 'low', 'medium', 'high', 'very_high']:
        if band not in risk_bands:
            risk_bands[band] = 0
    
    # Get performance metrics
    performance_metrics = []
    
    metrics_to_include = [
        ('AUC-ROC', 'auc_roc'),
        ('KS Statistic', 'ks_statistic'),
        ('Gini Coefficient', 'gini_coefficient'),
        ('Default Rate (%)', 'default_rate', lambda x: round(x * 100, 2)),
        ('Average Score', 'average_score'),
        ('Total Customers', 'total_customers')
    ]
    
    for display_name, metric_key, *formatter in metrics_to_include:
        format_func = formatter[0] if formatter else lambda x: x
        
        current = latest_monitoring.get(metric_key, 0)
        if monitoring_data.get('has_previous', False):
            previous = monitoring_data['previous'].get(metric_key, 0)
            if previous:
                change = current - previous
                change_pct = (change / previous) * 100 if previous else 0
                change_display = f"{change:+.2f} ({change_pct:+.1f}%)"
            else:
                change_display = "N/A"
        else:
            previous = "N/A"
            change_display = "N/A"
        
        performance_metrics.append({
            'name': display_name,
            'value': format_func(current),
            'previous': format_func(previous) if previous != "N/A" else previous,
            'change': change_display
        })
    
    # Get feature importance
    feature_importance = []
    if 'performance_metrics' in model_info and 'feature_importance' in model_info['performance_metrics']:
        importance_data = model_info['performance_metrics']['feature_importance']
        
        # Sort by importance
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
        
        for name, importance in sorted_features[:15]:  # Show top 15
            feature_importance.append({
                'name': name,
                'importance': f"{importance:.4f}",
                'description': get_feature_description(name)
            })
    
    # Get PSI data
    psi_value = latest_monitoring.get('population_stability_index', 0)
    psi_bins = []
    
    # Get segmentation data
    segments = []
    if not segmentation_data.empty:
        for segment_name in segmentation_data['segment_name'].unique():
            segment_rows = segmentation_data[segmentation_data['segment_name'] == segment_name]
            total_count = segment_rows['count'].sum()
            avg_score = segment_rows['avg_score'].mean()
            default_rate = segment_rows['default_rate'].mean()
            
            segments.append({
                'name': segment_name,
                'count': int(total_count),
                'average_score': round(avg_score, 1),
                'default_rate': round(default_rate, 2)
            })
    
    # Prepare template data
    report_title = f"Credit Risk Scorecard: {model_info['model_name']} v{model_info['model_version']}"
    
    executive_summary = (
        f"This report provides an overview of the {model_info['model_name']} "
        f"credit risk model (version {model_info['model_version']}). "
        f"The model was trained on {model_info['training_date']} "
        f"and is currently {model_info['status']}. "
    )
    
    # Add performance summary
    if latest_monitoring:
        executive_summary += (
            f"The model has an AUC-ROC of {latest_monitoring.get('auc_roc', 0):.3f} "
            f"and a Gini coefficient of {latest_monitoring.get('gini_coefficient', 0):.3f}. "
            f"The Population Stability Index (PSI) is {psi_value:.3f}, indicating "
        )
        
        if psi_value < 0.1:
            executive_summary += "population stability."
        elif psi_value < 0.25:
            executive_summary += "moderate population shift worth investigating."
        else:
            executive_summary += "significant population shift requiring attention."
    
    template_data = {
        'report_title': report_title,
        'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'executive_summary': executive_summary,
        'total_customers': int(latest_monitoring.get('total_customers', 0)),
        'average_score': round(latest_monitoring.get('average_score', 0), 1),
        'default_rate': round(latest_monitoring.get('default_rate', 0) * 100, 2),
        'auc_roc': round(latest_monitoring.get('auc_roc', 0), 3),
        'risk_bands': risk_bands,
        'performance_metrics': performance_metrics,
        'feature_importance': feature_importance,
        'psi_value': round(psi_value, 3),
        'psi_bins': psi_bins,
        'segments': segments,
        'current_year': datetime.now().year,
        'performance_chart_url': os.path.relpath(charts['performance'], REPORT_OUTPUT_DIR) if charts.get('performance') else "",
        'feature_importance_chart_url': os.path.relpath(charts['feature_importance'], REPORT_OUTPUT_DIR) if charts.get('feature_importance') else "",
        'stability_chart_url': os.path.relpath(charts['stability'], REPORT_OUTPUT_DIR) if charts.get('stability') else "",
        'segmentation_chart_url': os.path.relpath(charts['segmentation'], REPORT_OUTPUT_DIR) if charts.get('segmentation') else ""
    }
    
    # Add model name/version for filename generation
    template_data['model_name'] = model_info.get('model_name', 'unknown')
    template_data['model_version'] = model_info.get('model_version', 'unknown')
    
    return template_data


def get_feature_description(feature_name: str) -> str:
    """
    Get a description for a feature.
    
    Args:
        feature_name (str): Name of the feature
    
    Returns:
        str: Description of the feature
    """
    # This would ideally come from a database or configuration file
    descriptions = {
        'credit_score': 'Credit bureau score reflecting creditworthiness',
        'annual_income': 'Annual income of the customer',
        'employment_length_years': 'Number of years at current employment',
        'credit_utilization_ratio': 'Ratio of current credit balance to credit limit',
        'debt_to_income_ratio': 'Ratio of monthly debt payments to monthly income',
        'loan_amount': 'Total amount of the loan',
        'term_months': 'Term of the loan in months',
        'interest_rate': 'Interest rate of the loan',
        'age_years': 'Age of the customer in years',
        'delinquent_accounts': 'Number of delinquent accounts',
        'collections_last_12m': 'Collections in the last 12 months',
        'hard_inquiries_last_12m': 'Hard inquiries in the last 12 months',
        'length_of_credit_history_months': 'Length of credit history in months',
        'total_accounts': 'Total number of credit accounts',
        'open_accounts': 'Number of open credit accounts'
    }
    
    # Check if feature name contains any of the keys
    for key, description in descriptions.items():
        if key in feature_name:
            return description
    
    return "Feature used in the model calculation"


def render_template(template_name: str, template_data: Dict[str, Any], output_format: str) -> Dict[str, str]:
    """
    Render a template to HTML and optionally PDF (if weasyprint is available).
    
    Args:
        template_name (str): Name of the template file
        template_data (Dict[str, Any]): Data for the template
        output_format (str): Output format ('html', 'pdf', or 'both')
    
    Returns:
        Dict[str, str]: Paths to the generated files
    """
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(f"{template_name}.html")
    
    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use model name/version if available for better identification
    model_name = template_data.get('model_name', 'unknown_model')
    model_version = template_data.get('model_version', 'unknown_version')
    base_filename = f"{template_name}_{model_name}_{model_version}_{timestamp}"
    
    # Create output paths
    html_path = REPORT_OUTPUT_DIR / f"{base_filename}.html"
    pdf_path = REPORT_OUTPUT_DIR / f"{base_filename}.pdf"
    
    output_files = {}
    
    try:
        # Render HTML
        html_content = template.render(**template_data)
        
        if output_format in ['html', 'both']:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            output_files['html'] = str(html_path)
            logger.info(f"Generated HTML report: {html_path}")
        
        # Convert to PDF if needed and possible
        if output_format in ['pdf', 'both']:
            if WEASYPRINT_AVAILABLE:
                try:
                    weasyprint.HTML(string=html_content, base_url=str(REPORT_OUTPUT_DIR)).write_pdf(pdf_path)
                    output_files['pdf'] = str(pdf_path)
                    logger.info(f"Generated PDF report: {pdf_path}")
                except Exception as pdf_error: # Catch potential errors during PDF generation
                     logger.error(f"Error generating PDF report with WeasyPrint: {pdf_error}")
                     logger.error("Ensure GTK+ dependencies are correctly installed and accessible.")
            else:
                logger.warning("Skipping PDF generation because WeasyPrint backend is not available.")
        
        return output_files
    
    except Exception as e:
        logger.error(f"Error rendering template {template_name}: {e}")
        raise


def save_report_to_db(
    model_id: str,
    report_type: str,
    report_files: Dict[str, str],
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    save_to_db: bool = True # Flag to control DB saving
) -> None:
    """
    Save report information to the database.
    Logs a warning if database saving fails.
    
    Args:
        model_id (str): ID of the model
        report_type (str): Type of report
        report_files (Dict[str, str]): Paths to the generated report files
        period_start (Optional[str]): Start date for the report period
        period_end (Optional[str]): End date for the report period
        save_to_db (bool): Attempt to save to database if True
    """
    if not save_to_db:
        logger.info("Skipping saving report info to database.")
        return
        
    try:
        report_id = str(uuid.uuid4())
        
        # Get the PDF path if available, otherwise HTML
        report_path = report_files.get('pdf', report_files.get('html', ''))
        report_format = 'pdf' if 'pdf' in report_files else 'html'
        
        # Convert report_type to a more descriptive name
        report_names = {
            'scorecard': 'Credit Risk Scorecard',
            'monitoring': 'Model Monitoring Report',
            'summary': 'Risk Summary Report'
        }
        
        report_name = report_names.get(report_type, f"{report_type.capitalize()} Report")
        
        # Insert into scorecard_report table
        insert_query = """
        INSERT INTO scorecard_report (
            report_id, report_name, report_type, report_date,
            model_id, period_start, period_end, report_content,
            report_format, created_by, notes
        ) VALUES (
            :report_id, :report_name, :report_type, :report_date,
            :model_id, :period_start, :period_end, :report_content,
            :report_format, :created_by, :notes
        )
        """
        
        params = {
            'report_id': report_id,
            'report_name': report_name,
            'report_type': report_type,
            'report_date': datetime.now().date().isoformat(), # Use ISO format
            'model_id': model_id,
            'period_start': period_start,
            'period_end': period_end,
            'report_content': report_path, # Store path to the file
            'report_format': report_format,
            'created_by': 'system',
            'notes': f"Auto-generated {report_type} report."
        }
        
        execute_statement(insert_query, params)
        logger.info(f"Saved report information to database with ID {report_id}")
    
    except Exception as e:
        logger.warning(f"Could not save report information to database: {e}")
        # Don't raise, just log the warning


def main():
    """Main function to generate credit risk scorecard reports."""
    args = parse_arguments()
    
    try:
        # Get model information (handles DB/filesystem fallback)
        try:
            model_info = get_model_info(args.model_id)
            # Use the definitive ID found (might differ from input if name was given)
            model_id = model_info.get('model_id', args.model_id)
            logger.info(f"Generating report for model: {model_info.get('model_name', 'unknown')} v{model_info.get('model_version', 'unknown')} (ID: {model_id})")
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
        
        # Fetch data based on report type
        template_data = {}
        save_db_attempt = True # Attempt to save report info to DB by default
        
        if args.report_type == 'scorecard':
            logger.info("Fetching data for Scorecard report...")
            monitoring_data = get_model_monitoring_data(
                model_id,
                period_start=args.period_start,
                period_end=args.period_end
            )
            score_distribution = get_score_distribution(
                model_id,
                period_start=args.period_start,
                period_end=args.period_end
            )
            segmentation_data = get_segmentation_data(
                model_id,
                period_start=args.period_start,
                period_end=args.period_end
            )
            
            # Prepare template data
            template_data = prepare_scorecard_data(
                model_info,
                monitoring_data,
                score_distribution,
                segmentation_data,
                CHARTS_DIR
            )
            template_name = 'scorecard'
            
        # elif args.report_type == 'monitoring':
        #     # Fetch data specific to monitoring report
        #     # template_data = prepare_monitoring_data(...)
        #     template_name = 'monitoring'
        #     logger.error(f"Report type '{args.report_type}' not fully implemented yet")
        #     # sys.exit(1) # Or proceed with limited data
            
        # elif args.report_type == 'summary':
        #     # Fetch data specific to summary report
        #     # template_data = prepare_summary_data(...)
        #     template_name = 'summary'
        #     logger.error(f"Report type '{args.report_type}' not fully implemented yet")
        #     # sys.exit(1)
            
        else:
            logger.error(f"Report type '{args.report_type}' not recognized or implemented.")
            sys.exit(1)

        # Render template
        if template_data:
            logger.info(f"Rendering {args.report_type} report...")
            report_files = render_template(template_name, template_data, args.output_format)
            
            # Save report information to database (with fallback handled inside)
            if report_files:
                save_report_to_db(
                    model_id,
                    args.report_type,
                    report_files,
                    args.period_start,
                    args.period_end,
                    save_to_db=save_db_attempt
                )
                logger.info(f"Report generation completed successfully. Files saved in {REPORT_OUTPUT_DIR}")
            else:
                 logger.error("Report file generation failed.")
                 sys.exit(1)
        else:
             logger.error("Failed to prepare data for the report.")
             sys.exit(1)

    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 