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
import weasyprint
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

# Directories
TEMPLATE_DIR = PROJECT_ROOT / "reporting" / "templates"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "reporting" / "outputs"
CHARTS_DIR = PROJECT_ROOT / "reporting" / "charts"

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


def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get model information from the database.
    
    Args:
        model_id (str): ID of the model
    
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


def get_model_monitoring_data(model_id: str, period_start: Optional[str] = None, period_end: Optional[str] = None) -> Dict[str, Any]:
    """
    Get model monitoring data from the database.
    
    Args:
        model_id (str): ID of the model
        period_start (Optional[str]): Start date for the report period
        period_end (Optional[str]): End date for the report period
    
    Returns:
        Dict[str, Any]: Model monitoring data
    """
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
    
    try:
        result = execute_query(query, params)
        
        if result.empty:
            logger.warning(f"No monitoring data found for model {model_id}")
            return {}
        
        # Convert to list of dictionaries
        monitoring_records = []
        for _, row in result.iterrows():
            record = row.to_dict()
            
            # Parse JSON fields
            if record['variable_drift_metrics'] and isinstance(record['variable_drift_metrics'], str):
                record['variable_drift_metrics'] = json.loads(record['variable_drift_metrics'])
            
            monitoring_records.append(record)
        
        # Get the latest record
        latest_record = monitoring_records[0]
        
        # Prepare monitoring data
        monitoring_data = {
            'records': monitoring_records,
            'latest': latest_record,
            'has_previous': len(monitoring_records) > 1,
            'previous': monitoring_records[1] if len(monitoring_records) > 1 else None,
            'trend': calculate_performance_trend(monitoring_records)
        }
        
        return monitoring_data
    
    except Exception as e:
        logger.error(f"Error getting model monitoring data: {e}")
        raise


def get_score_distribution(model_id: str, period_start: Optional[str] = None, period_end: Optional[str] = None) -> pd.DataFrame:
    """
    Get score distribution data from the database.
    
    Args:
        model_id (str): ID of the model
        period_start (Optional[str]): Start date for the report period
        period_end (Optional[str]): End date for the report period
    
    Returns:
        pd.DataFrame: Score distribution data
    """
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
    ORDER BY probability_of_default ASC
    """
    
    try:
        result = execute_query(query, params)
        
        if result.empty:
            logger.warning(f"No score distribution data found for model {model_id}")
            return pd.DataFrame()
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting score distribution data: {e}")
        raise


def get_segmentation_data(model_id: str, period_start: Optional[str] = None, period_end: Optional[str] = None) -> pd.DataFrame:
    """
    Get segmentation data from the database.
    
    Args:
        model_id (str): ID of the model
        period_start (Optional[str]): Start date for the report period
        period_end (Optional[str]): End date for the report period
    
    Returns:
        pd.DataFrame: Segmentation data
    """
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
    
    try:
        result = execute_query(query, params)
        
        if result.empty:
            logger.warning(f"No segmentation data found for model {model_id}")
            return pd.DataFrame()
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting segmentation data: {e}")
        raise


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
    # Generate charts
    performance_chart = generate_performance_chart(monitoring_data, chart_path)
    feature_importance_chart = generate_feature_importance_chart(model_info, chart_path)
    stability_chart = generate_stability_chart(monitoring_data, chart_path)
    segmentation_chart = generate_segmentation_chart(segmentation_data, chart_path)
    
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
        'performance_chart_url': os.path.relpath(performance_chart, REPORT_OUTPUT_DIR) if performance_chart else "",
        'feature_importance_chart_url': os.path.relpath(feature_importance_chart, REPORT_OUTPUT_DIR) if feature_importance_chart else "",
        'stability_chart_url': os.path.relpath(stability_chart, REPORT_OUTPUT_DIR) if stability_chart else "",
        'segmentation_chart_url': os.path.relpath(segmentation_chart, REPORT_OUTPUT_DIR) if segmentation_chart else ""
    }
    
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
    Render a template to HTML and optionally PDF.
    
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
    base_filename = f"{template_name}_{timestamp}"
    
    # Create output paths
    html_path = os.path.join(REPORT_OUTPUT_DIR, f"{base_filename}.html")
    pdf_path = os.path.join(REPORT_OUTPUT_DIR, f"{base_filename}.pdf")
    
    output_files = {}
    
    try:
        # Render HTML
        html_content = template.render(**template_data)
        
        if output_format in ['html', 'both']:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            output_files['html'] = html_path
            logger.info(f"Generated HTML report: {html_path}")
        
        # Convert to PDF if needed
        if output_format in ['pdf', 'both']:
            weasyprint.HTML(string=html_content, base_url=REPORT_OUTPUT_DIR).write_pdf(pdf_path)
            output_files['pdf'] = pdf_path
            logger.info(f"Generated PDF report: {pdf_path}")
        
        return output_files
    
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        raise


def save_report_to_db(
    model_id: str,
    report_type: str,
    report_files: Dict[str, str],
    period_start: Optional[str] = None,
    period_end: Optional[str] = None
) -> None:
    """
    Save report information to the database.
    
    Args:
        model_id (str): ID of the model
        report_type (str): Type of report
        report_files (Dict[str, str]): Paths to the generated report files
        period_start (Optional[str]): Start date for the report period
        period_end (Optional[str]): End date for the report period
    """
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
            'report_date': datetime.now().date(),
            'model_id': model_id,
            'period_start': period_start,
            'period_end': period_end,
            'report_content': report_path,
            'report_format': report_format,
            'created_by': 'system',
            'notes': f"Auto-generated {report_type} report."
        }
        
        execute_statement(insert_query, params)
        logger.info(f"Saved report information to database with ID {report_id}")
    
    except Exception as e:
        logger.error(f"Error saving report information to database: {e}")
        raise


def main():
    """Main function to generate credit risk scorecard reports."""
    args = parse_arguments()
    
    try:
        # Get model information
        model_info = get_model_info(args.model_id)
        logger.info(f"Generating report for model: {model_info['model_name']} v{model_info['model_version']}")
        
        # Get monitoring data
        monitoring_data = get_model_monitoring_data(
            args.model_id,
            period_start=args.period_start,
            period_end=args.period_end
        )
        
        # Get score distribution
        score_distribution = get_score_distribution(
            args.model_id,
            period_start=args.period_start,
            period_end=args.period_end
        )
        
        # Get segmentation data
        segmentation_data = get_segmentation_data(
            args.model_id,
            period_start=args.period_start,
            period_end=args.period_end
        )
        
        # Prepare template data based on report type
        if args.report_type == 'scorecard':
            template_data = prepare_scorecard_data(
                model_info,
                monitoring_data,
                score_distribution,
                segmentation_data,
                CHARTS_DIR
            )
            
            # Render template
            report_files = render_template('scorecard', template_data, args.output_format)
            
            # Save report information to database
            save_report_to_db(
                args.model_id,
                args.report_type,
                report_files,
                args.period_start,
                args.period_end
            )
            
            logger.info("Report generation completed successfully")
        
        else:
            logger.error(f"Report type '{args.report_type}' not implemented yet")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 