#!/usr/bin/env python3
"""
Credit Risk Scorecard & Reporting Engine - Main runner script.
This script provides a command-line interface to run different components of the system.
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()


def setup_database(args):
    """Set up the database schema."""
    script_path = PROJECT_ROOT / "database" / "setup_db.py"
    
    cmd = [sys.executable, str(script_path)]
    
    if args.db_type:
        cmd.extend(["--db-type", args.db_type])
    if args.host:
        cmd.extend(["--host", args.host])
    if args.port:
        cmd.extend(["--port", str(args.port)])
    if args.user:
        cmd.extend(["--user", args.user])
    if args.password:
        cmd.extend(["--password", args.password])
    if args.db_name:
        cmd.extend(["--db-name", args.db_name])
    if args.create_db:
        cmd.append("--create-db")
    if args.populate_sample:
        cmd.append("--populate-sample")
    
    logger.info(f"Running database setup with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def generate_sample_data(args):
    """Generate sample data for the database."""
    script_path = PROJECT_ROOT / "data_processing" / "generate_sample_data.py"
    
    cmd = [sys.executable, str(script_path)]
    
    logger.info(f"Generating sample data with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def train_model(args):
    """Train a credit risk model."""
    script_path = PROJECT_ROOT / "models" / "train_model.py"
    
    cmd = [sys.executable, str(script_path)]
    
    if args.model_type:
        cmd.extend(["--model-type", args.model_type])
    if args.model_name:
        cmd.extend(["--model-name", args.model_name])
    if args.test_size:
        cmd.extend(["--test-size", str(args.test_size)])
    if args.tune_hyperparams:
        cmd.append("--tune-hyperparams")
    
    logger.info(f"Training model with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def score_loans(args):
    """Score loans using a trained model."""
    script_path = PROJECT_ROOT / "models" / "predict.py"
    
    cmd = [sys.executable, str(script_path)]
    
    if args.model_name:
        cmd.extend(["--model-name", args.model_name])
    if args.customer_id:
        cmd.extend(["--customer-id", args.customer_id])
    if args.application_id:
        cmd.extend(["--application-id", args.application_id])
    if args.loan_id:
        cmd.extend(["--loan-id", args.loan_id])
    if args.batch:
        cmd.append("--batch")
    if args.output_csv:
        cmd.extend(["--output-csv", args.output_csv])
    
    logger.info(f"Scoring loans with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def monitor_model(args):
    """Monitor model performance."""
    script_path = PROJECT_ROOT / "monitoring" / "model_monitoring.py"
    
    cmd = [sys.executable, str(script_path)]
    
    if args.model_id:
        cmd.extend(["--model-id", args.model_id])
    if args.period:
        cmd.extend(["--period", args.period])
    if args.lookback:
        cmd.extend(["--lookback", str(args.lookback)])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.generate_plots:
        cmd.append("--generate-plots")
    
    logger.info(f"Monitoring model with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def generate_report(args):
    """Generate a credit risk report."""
    script_path = PROJECT_ROOT / "reporting" / "generate_reports.py"
    
    cmd = [sys.executable, str(script_path)]
    
    if args.model_id:
        cmd.extend(["--model-id", args.model_id])
    if args.report_type:
        cmd.extend(["--report-type", args.report_type])
    if args.output_format:
        cmd.extend(["--output-format", args.output_format])
    if args.period_start:
        cmd.extend(["--period-start", args.period_start])
    if args.period_end:
        cmd.extend(["--period-end", args.period_end])
    
    logger.info(f"Generating report with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_all(args):
    """Run a complete end-to-end workflow."""
    # Step 1: Setup database
    logger.info("Step 1: Setting up database...")
    setup_args = argparse.Namespace(
        db_type=args.db_type,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        db_name=args.db_name,
        create_db=True,
        populate_sample=False
    )
    setup_database(setup_args)
    
    # Step 2: Generate sample data
    logger.info("Step 2: Generating sample data...")
    generate_sample_data(argparse.Namespace())
    
    # Step 3: Train model
    logger.info("Step 3: Training model...")
    train_args = argparse.Namespace(
        model_type=args.model_type,
        model_name=args.model_name,
        test_size=0.2,
        tune_hyperparams=args.tune_hyperparams
    )
    train_model(train_args)
    
    # Step 4: Score loans
    logger.info("Step 4: Scoring loans...")
    score_args = argparse.Namespace(
        model_name=args.model_name,
        customer_id=None,
        application_id=None,
        loan_id=None,
        batch=True,
        output_csv=None
    )
    score_loans(score_args)
    
    # Step 5: Monitor model
    logger.info("Step 5: Monitoring model...")
    monitor_args = argparse.Namespace(
        model_id=args.model_name,
        period='monthly',
        lookback=6,
        output_dir=None,
        generate_plots=True
    )
    monitor_model(monitor_args)
    
    # Step 6: Generate report
    logger.info("Step 6: Generating report...")
    today = datetime.now().date()
    month_ago = today - timedelta(days=30)
    report_args = argparse.Namespace(
        model_id=args.model_name,
        report_type='scorecard',
        output_format='both',
        period_start=month_ago.strftime('%Y-%m-%d'),
        period_end=today.strftime('%Y-%m-%d')
    )
    generate_report(report_args)
    
    logger.info("End-to-end workflow completed successfully!")


def main():
    """Main function to parse arguments and run commands."""
    parser = argparse.ArgumentParser(
        description='Credit Risk Scorecard & Reporting Engine'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup Database command
    setup_parser = subparsers.add_parser('setup', help='Set up the database')
    setup_parser.add_argument('--db-type', choices=['mysql', 'postgresql'], 
                            help='Database type (mysql or postgresql)')
    setup_parser.add_argument('--host', help='Database host')
    setup_parser.add_argument('--port', type=int, help='Database port')
    setup_parser.add_argument('--user', help='Database user')
    setup_parser.add_argument('--password', help='Database password')
    setup_parser.add_argument('--db-name', help='Database name')
    setup_parser.add_argument('--create-db', action='store_true', 
                            help='Create database if it does not exist')
    setup_parser.add_argument('--populate-sample', action='store_true',
                            help='Populate with sample data')
    
    # Generate Sample Data command
    sample_parser = subparsers.add_parser('sample', help='Generate sample data')
    
    # Train Model command
    train_parser = subparsers.add_parser('train', help='Train a credit risk model')
    train_parser.add_argument('--model-type', choices=['logistic', 'gbm', 'xgboost', 'rf'], 
                           help='Type of model to train')
    train_parser.add_argument('--model-name', help='Name for the trained model')
    train_parser.add_argument('--test-size', type=float, help='Proportion of data to use for testing')
    train_parser.add_argument('--tune-hyperparams', action='store_true',
                            help='Perform hyperparameter tuning')
    
    # Score Loans command
    score_parser = subparsers.add_parser('score', help='Score loans using a trained model')
    score_parser.add_argument('--model-name', required=True, help='Name or ID of the model to use')
    score_parser.add_argument('--customer-id', help='Specific customer ID to score')
    score_parser.add_argument('--application-id', help='Specific loan application ID to score')
    score_parser.add_argument('--loan-id', help='Specific loan ID to score')
    score_parser.add_argument('--batch', action='store_true', help='Run batch scoring for all active loans')
    score_parser.add_argument('--output-csv', help='Save results to a CSV file')
    
    # Monitor Model command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor model performance')
    monitor_parser.add_argument('--model-id', required=True, help='ID of the model to monitor')
    monitor_parser.add_argument('--period', choices=['daily', 'weekly', 'monthly'], 
                             help='Monitoring period')
    monitor_parser.add_argument('--lookback', type=int, help='Number of periods to look back for trends')
    monitor_parser.add_argument('--output-dir', help='Directory to save monitoring reports')
    monitor_parser.add_argument('--generate-plots', action='store_true', help='Generate and save plots')
    
    # Generate Report command
    report_parser = subparsers.add_parser('report', help='Generate a credit risk report')
    report_parser.add_argument('--model-id', required=True, help='ID of the model to report on')
    report_parser.add_argument('--report-type', choices=['scorecard', 'monitoring', 'summary'],
                            help='Type of report to generate')
    report_parser.add_argument('--output-format', choices=['html', 'pdf', 'both'],
                            help='Output format for the report')
    report_parser.add_argument('--period-start', help='Start date for the report period (YYYY-MM-DD)')
    report_parser.add_argument('--period-end', help='End date for the report period (YYYY-MM-DD)')
    
    # Run All command
    all_parser = subparsers.add_parser('all', help='Run a complete end-to-end workflow')
    all_parser.add_argument('--db-type', choices=['mysql', 'postgresql'], 
                         default='mysql', help='Database type (mysql or postgresql)')
    all_parser.add_argument('--host', default='localhost', help='Database host')
    all_parser.add_argument('--port', type=int, help='Database port')
    all_parser.add_argument('--user', default='root', help='Database user')
    all_parser.add_argument('--password', default='password', help='Database password')
    all_parser.add_argument('--db-name', default='credit_risk', help='Database name')
    all_parser.add_argument('--model-type', choices=['logistic', 'gbm', 'xgboost', 'rf'], 
                         default='logistic', help='Type of model to train')
    all_parser.add_argument('--model-name', default='default_credit_risk_model',
                         help='Name for the trained model')
    all_parser.add_argument('--tune-hyperparams', action='store_true',
                         help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Route to the appropriate command handler
    if args.command == 'setup':
        setup_database(args)
    elif args.command == 'sample':
        generate_sample_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'score':
        score_loans(args)
    elif args.command == 'monitor':
        monitor_model(args)
    elif args.command == 'report':
        generate_report(args)
    elif args.command == 'all':
        run_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 