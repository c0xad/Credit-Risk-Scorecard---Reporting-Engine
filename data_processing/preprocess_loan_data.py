#!/usr/bin/env python3
"""
Preprocess Loan Data for Credit Risk Scorecard

This script loads loan application data, applies preprocessing techniques,
and prepares it for credit risk modeling.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.database import get_db_connection
from data_processing.data_preprocessing import CreditRiskPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(source: str = 'db', file_path: str = None, query_type: str = 'default') -> pd.DataFrame:
    """
    Load data from database or CSV file.
    
    Args:
        source: Source of data ('db' for database, 'csv' for CSV file)
        file_path: Path to CSV file (if source is 'csv')
        query_type: Type of query to execute ('default', 'application', 'payment', 'customer')
        
    Returns:
        DataFrame containing loan data
    """
    if source == 'csv':
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"Invalid file path: {file_path}")
        return pd.read_csv(file_path)
    
    # Database queries
    queries = {
        'default': """
            SELECT 
                a.*, 
                c.annual_income, c.employment_status, c.age, c.gender, 
                c.marital_status, c.education, c.housing_status,
                CASE WHEN a.status = 'Rejected' THEN 1 ELSE 0 END as rejection_flag
            FROM 
                loan_application a
            JOIN 
                customer_profile c ON a.customer_id = c.customer_id
        """,
        'application': """
            SELECT * FROM loan_application
        """,
        'payment': """
            SELECT * FROM payment_history
        """,
        'customer': """
            SELECT * FROM customer_profile
        """,
        'credit': """
            SELECT * FROM credit_bureau_data
        """,
        'full': """
            SELECT 
                a.*,
                c.annual_income, c.employment_status, c.age, c.gender, 
                c.marital_status, c.education, c.housing_status,
                c.dependents, c.employment_length_years, c.income_category,
                cb.credit_score, cb.risk_band, cb.delinquent_accounts,
                cb.payment_history_percent, cb.credit_utilization_ratio,
                cb.length_of_credit_history_months, cb.collections_last_12m,
                CASE WHEN a.status = 'Rejected' THEN 1 ELSE 0 END as rejection_flag,
                CASE WHEN p.payment_status IN ('Late 30 Days', 'Late 60 Days', 'Late 90+ Days', 'Missed') 
                     THEN 1 ELSE 0 END as delinquency_flag
            FROM 
                loan_application a
            LEFT JOIN 
                customer_profile c ON a.customer_id = c.customer_id
            LEFT JOIN 
                credit_bureau_data cb ON a.customer_id = cb.customer_id AND 
                cb.report_date <= a.application_date
            LEFT JOIN
                payment_history p ON a.application_id = p.application_id
            WHERE
                a.status = 'Approved'
        """
    }
    
    if query_type not in queries:
        query_type = 'default'
    
    conn = get_db_connection()
    
    # Check if tables exist
    try:
        # Get list of tables from database
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        # Extract tables needed for the query
        if query_type == 'default':
            needed_tables = ['loan_application', 'customer_profile']
        elif query_type == 'full':
            needed_tables = ['loan_application', 'customer_profile', 'credit_bureau_data', 'payment_history']
        else:
            needed_tables = [query_type]
        
        # Check if all needed tables exist
        all_tables_exist = all(table in existing_tables for table in needed_tables)
        
        if all_tables_exist:
            # Tables exist, proceed with database query
            df = pd.read_sql(queries[query_type], conn)
            logger.info(f"Loaded {len(df)} rows from database using '{query_type}' query")
            return df
        else:
            missing_tables = [table for table in needed_tables if table not in existing_tables]
            logger.warning(f"Required tables not found in database: {', '.join(missing_tables)}")
            logger.info("Generating synthetic data instead...")
            return _generate_synthetic_data(query_type)
            
    except Exception as e:
        logger.error(f"Error accessing database: {e}")
        logger.info("Falling back to synthetic data generation...")
        return _generate_synthetic_data(query_type)


def _generate_synthetic_data(data_type: str = 'default') -> pd.DataFrame:
    """
    Generate synthetic data for testing when database tables don't exist.
    
    Args:
        data_type: Type of data to generate ('default', 'application', 'payment', etc.)
        
    Returns:
        DataFrame containing synthetic data
    """
    np.random.seed(42)  # For reproducibility
    n_samples = 100
    
    # Generate customer IDs
    customer_ids = [f'CUST{i:05d}' for i in range(n_samples)]
    
    if data_type == 'customer':
        # Generate customer profile data
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'first_name': [f'First{i}' for i in range(n_samples)],
            'last_name': [f'Last{i}' for i in range(n_samples)],
            'age': np.random.randint(18, 75, n_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
            'education': np.random.choice(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], n_samples),
            'employment_status': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples),
            'employment_length_years': np.random.randint(0, 30, n_samples),
            'annual_income': np.random.normal(60000, 20000, n_samples),
            'income_category': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'housing_status': np.random.choice(['Own', 'Rent', 'Other'], n_samples)
        })
    
    elif data_type == 'credit':
        # Generate credit bureau data
        df = pd.DataFrame({
            'report_id': [f'REP{i:05d}' for i in range(n_samples)],
            'customer_id': customer_ids,
            'report_date': [pd.Timestamp('2023-01-01') - pd.Timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)],
            'credit_score': np.random.randint(300, 850, n_samples),
            'risk_band': np.random.choice(['Very Low', 'Low', 'Medium', 'High', 'Very High'], n_samples),
            'total_accounts': np.random.randint(1, 20, n_samples),
            'open_accounts': np.random.randint(1, 10, n_samples),
            'delinquent_accounts': np.random.randint(0, 3, n_samples),
            'credit_utilization_ratio': np.random.uniform(0, 1, n_samples),
            'payment_history_percent': np.random.uniform(0.5, 1, n_samples),
            'collections_last_12m': np.random.randint(0, 2, n_samples)
        })
    
    elif data_type == 'application':
        # Generate loan application data
        application_ids = [f'APP{i:05d}' for i in range(n_samples)]
        df = pd.DataFrame({
            'application_id': application_ids,
            'customer_id': customer_ids,
            'application_date': [pd.Timestamp('2023-01-01') - pd.Timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)],
            'loan_type': np.random.choice(['Personal', 'Auto', 'Mortgage', 'Credit Card'], n_samples),
            'loan_purpose': np.random.choice(['Debt Consolidation', 'Home Improvement', 'Purchase', 'Education'], n_samples),
            'requested_amount': np.random.normal(15000, 5000, n_samples),
            'term_months': np.random.choice([12, 24, 36, 48, 60], n_samples),
            'interest_rate': np.random.uniform(3, 15, n_samples),
            'status': np.random.choice(['Approved', 'Rejected', 'Pending'], n_samples, p=[0.6, 0.3, 0.1])
        })
        
    elif data_type == 'payment':
        # Generate payment history data
        application_ids = [f'APP{i:05d}' for i in range(n_samples)]
        df = pd.DataFrame({
            'payment_id': [f'PAY{i:05d}' for i in range(n_samples)],
            'application_id': application_ids,
            'customer_id': customer_ids,
            'payment_number': np.random.randint(1, 12, n_samples),
            'scheduled_date': [pd.Timestamp('2023-01-01') - pd.Timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)],
            'scheduled_amount': np.random.normal(500, 100, n_samples),
            'actual_amount': np.random.normal(500, 150, n_samples),
            'payment_status': np.random.choice(['On Time', 'Late 30 Days', 'Late 60 Days', 'Missed'], n_samples, p=[0.8, 0.1, 0.05, 0.05])
        })
        
    elif data_type == 'full':
        # Generate comprehensive data for delinquency modeling
        application_ids = [f'APP{i:05d}' for i in range(n_samples)]
        statuses = np.random.choice(['Approved'], n_samples)  # All approved for full query
        payment_statuses = np.random.choice(['On Time', 'Late 30 Days', 'Late 60 Days', 'Late 90+ Days', 'Missed'], 
                                          n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.05])
        
        df = pd.DataFrame({
            'application_id': application_ids,
            'customer_id': customer_ids,
            'application_date': [pd.Timestamp('2023-01-01') - pd.Timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)],
            'loan_type': np.random.choice(['Personal', 'Auto', 'Mortgage', 'Credit Card'], n_samples),
            'requested_amount': np.random.normal(15000, 5000, n_samples),
            'term_months': np.random.choice([12, 24, 36, 48, 60], n_samples),
            'interest_rate': np.random.uniform(3, 15, n_samples),
            'status': statuses,
            'annual_income': np.random.normal(60000, 20000, n_samples),
            'age': np.random.randint(18, 75, n_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
            'education': np.random.choice(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], n_samples),
            'employment_status': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'credit_utilization_ratio': np.random.uniform(0, 1, n_samples),
            'payment_status': payment_statuses,
            'rejection_flag': np.zeros(n_samples),  # All 0 since all approved
            'delinquency_flag': np.where(
                np.isin(payment_statuses, ['Late 30 Days', 'Late 60 Days', 'Late 90+ Days', 'Missed']), 
                1, 0
            )
        })
        
    else:  # Default
        # Generate combined application and customer data
        statuses = np.random.choice(['Approved', 'Rejected', 'Pending'], n_samples, p=[0.6, 0.3, 0.1])
        
        df = pd.DataFrame({
            'application_id': [f'APP{i:05d}' for i in range(n_samples)],
            'customer_id': customer_ids,
            'application_date': [pd.Timestamp('2023-01-01') - pd.Timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)],
            'loan_type': np.random.choice(['Personal', 'Auto', 'Mortgage', 'Credit Card'], n_samples),
            'requested_amount': np.random.normal(15000, 5000, n_samples),
            'term_months': np.random.choice([12, 24, 36, 48, 60], n_samples),
            'interest_rate': np.random.uniform(3, 15, n_samples),
            'status': statuses,
            'annual_income': np.random.normal(60000, 20000, n_samples),
            'employment_status': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples),
            'age': np.random.randint(18, 75, n_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
            'education': np.random.choice(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], n_samples),
            'housing_status': np.random.choice(['Own', 'Rent', 'Other'], n_samples),
            'rejection_flag': np.where(statuses == 'Rejected', 1, 0)
        })
    
    logger.info(f"Generated synthetic {data_type} data with {len(df)} rows")
    return df

def generate_scorecard_report(preprocessor: CreditRiskPreprocessor, coefficients: dict, 
                             intercept: float, model_performance: dict,
                             output_dir: str = "./output") -> None:
    """
    Generate a scorecard report with visualizations.
    
    Args:
        preprocessor: CreditRiskPreprocessor instance
        coefficients: Dictionary of model coefficients
        intercept: Model intercept
        model_performance: Dictionary containing model performance metrics
        output_dir: Directory to save the report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create scorecard
    features = [f for f in preprocessor.woe_mappings.keys() if f in coefficients]
    scorecard = preprocessor.create_scorecard(None, features, coefficients, intercept)
    
    # Save scorecard to CSV
    scorecard.to_csv(f"{output_dir}/scorecard.csv", index=False)
    
    # Create report with visualizations
    report_path = f"{output_dir}/scorecard_report.html"
    
    # Generate HTML report
    with open(report_path, 'w') as f:
        f.write(f"""
        <html>
        <head>
            <title>Credit Risk Scorecard Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; 
                          border-radius: 5px; background-color: #f8f9fa; min-width: 150px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                .feature-importance {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>Credit Risk Scorecard Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Model Performance</h2>
            <div>
        """)
        
        # Add performance metrics
        for metric, value in model_performance.items():
            f.write(f"""
                <div class="metric">
                    <div class="metric-value">{value:.4f}</div>
                    <div class="metric-label">{metric}</div>
                </div>
            """)
            
        f.write("""
            </div>
            
            <h2>Feature Importance</h2>
            <div class="feature-importance">
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Information Value</th>
                        <th>Coefficient</th>
                        <th>Importance</th>
                    </tr>
        """)
        
        # Add feature importance
        for feature in features:
            iv = preprocessor.iv_values.get(feature, 0)
            coef = abs(coefficients.get(feature, 0))
            importance = iv * coef
            
            f.write(f"""
                <tr>
                    <td>{feature}</td>
                    <td>{iv:.4f}</td>
                    <td>{coef:.4f}</td>
                    <td>{importance:.4f}</td>
                </tr>
            """)
            
        f.write("""
                </table>
            </div>
            
            <h2>Scorecard</h2>
            <table>
                <tr>
                    <th>Characteristic</th>
                    <th>Attribute</th>
                    <th>WoE</th>
                    <th>Coefficient</th>
                    <th>Points</th>
                </tr>
        """)
        
        # Add scorecard rows
        for _, row in scorecard.iterrows():
            f.write(f"""
                <tr>
                    <td>{row.get('Characteristic', '')}</td>
                    <td>{row.get('Attribute', '')}</td>
                    <td>{row.get('WoE', '')}</td>
                    <td>{row.get('Coefficient', '')}</td>
                    <td>{int(round(row.get('Points', 0)))}</td>
                </tr>
            """)
            
        f.write("""
            </table>
            
            <h2>WoE Visualizations</h2>
        """)
        
        # Generate and add WoE visualizations for top features
        top_features = sorted(preprocessor.iv_values.items(), key=lambda x: x[1], reverse=True)[:5]
        for feature, iv in top_features:
            # Generate plot and save as image
            plot_path = f"{output_dir}/{feature}_woe.png"
            preprocessor.plot_woe_analysis(feature, save_path=plot_path)
            
            # Add to report
            f.write(f"""
                <div>
                    <h3>{feature} (IV: {iv:.4f})</h3>
                    <img src="{os.path.basename(plot_path)}" alt="{feature} WoE Analysis" style="max-width: 100%;">
                </div>
            """)
            
        f.write("""
        </body>
        </html>
        """)
    
    logger.info(f"Scorecard report generated at {report_path}")

def main():
    """Main function to preprocess loan data."""
    parser = argparse.ArgumentParser(description='Preprocess loan data for credit risk scoring')
    parser.add_argument('--source', type=str, default='db', choices=['db', 'csv'],
                       help='Source of data (db or csv)')
    parser.add_argument('--file', type=str, default=None,
                       help='Path to CSV file (if source is csv)')
    parser.add_argument('--query', type=str, default='default',
                       choices=['default', 'application', 'payment', 'customer', 'credit', 'full'],
                       help='Type of query to execute if source is db')
    parser.add_argument('--target', type=str, default='rejection_flag',
                       help='Target variable for modeling')
    parser.add_argument('--output', type=str, default='./processed_data',
                       help='Directory to save processed data')
    parser.add_argument('--test_size', type=float, default=0.3,
                       help='Proportion of data to use for testing')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    df = load_data(args.source, args.file, args.query)
    
    # Initialize preprocessor
    preprocessor = CreditRiskPreprocessor(target_variable=args.target)
    
    # Process data
    processed_data = preprocessor.process_data(df, test_size=args.test_size)
    
    # Save processed data
    for name, data in processed_data.items():
        output_path = f"{args.output}/{name}_data.csv"
        data.to_csv(output_path, index=False)
        logger.info(f"Saved {name} data to {output_path}")
    
    # Print Information Value summary
    logger.info("\nFeature Information Values:")
    for feature, iv in sorted(preprocessor.iv_values.items(), key=lambda x: x[1], reverse=True):
        iv_strength = "Useless"
        if iv < 0.02:
            iv_strength = "Useless"
        elif iv < 0.1:
            iv_strength = "Weak"
        elif iv < 0.3:
            iv_strength = "Medium"
        elif iv < 0.5:
            iv_strength = "Strong"
        else:
            iv_strength = "Very Strong"
            
        logger.info(f"{feature}: {iv:.4f} ({iv_strength})")
    
    # Example: Generate mock scorecard for demonstration
    # In a real scenario, this would come from your trained model
    mock_coefficients = {
        feature: -1.0 * iv for feature, iv in preprocessor.iv_values.items()
    }
    mock_intercept = 0.5
    mock_performance = {
        'AUC': 0.82,
        'Gini': 0.64,
        'KS': 0.53,
        'Accuracy': 0.75
    }
    
    # Generate scorecard report
    generate_scorecard_report(
        preprocessor, 
        mock_coefficients, 
        mock_intercept, 
        mock_performance,
        output_dir=args.output
    )
    
    logger.info("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main() 