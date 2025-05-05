#!/usr/bin/env python3
"""
Generate sample data for the Credit Risk Scorecard database.
This script creates realistic sample data for testing and development.
"""

import os
import sys
import uuid
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from faker import Faker
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.database import save_df_to_table, get_db_connection

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

# Initialize Faker
fake = Faker()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
fake.seed_instance(42)

# Constants
NUM_CUSTOMERS = 1000
NUM_LOANS_PER_CUSTOMER_MAX = 3
LOAN_TYPES = ['Personal', 'Auto', 'Mortgage', 'Credit Card', 'Business']
LOAN_PURPOSES = ['Debt Consolidation', 'Home Improvement', 'Education', 'Medical Expenses', 
                'Vacation', 'Wedding', 'Major Purchase', 'Vehicle Purchase', 'Business Expansion']
EMPLOYMENT_STATUSES = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed', 'Retired', 'Student']
MARITAL_STATUSES = ['Single', 'Married', 'Divorced', 'Widowed', 'Separated']
EDUCATION_LEVELS = ['High School', 'Associate Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 
                   'Doctoral Degree', 'Professional Degree', 'Some College']
LOAN_STATUSES = ['Approved', 'Rejected', 'Pending', 'Withdrawn']
PAYMENT_STATUSES = ['On Time', 'Late', 'Missed', 'Partial']
RISK_BANDS = ['Very Low', 'Low', 'Medium', 'High', 'Very High']


def generate_customer_profiles(num_customers: int) -> pd.DataFrame:
    """Generate sample customer profile data."""
    logger.info(f"Generating {num_customers} customer profiles...")
    
    customers = []
    for i in range(num_customers):
        customer_id = f"CUST{i+10000:05d}"
        first_name = fake.first_name()
        last_name = fake.last_name()
        gender = random.choice(['Male', 'Female', 'Other', None])
        
        # Age between 18 and 80
        birth_date = fake.date_of_birth(minimum_age=18, maximum_age=80)
        
        # Registration date between 1 and 5 years ago
        reg_days_ago = random.randint(365, 365 * 5)
        registration_date = datetime.now() - timedelta(days=reg_days_ago)
        
        customers.append({
            'customer_id': customer_id,
            'first_name': first_name,
            'last_name': last_name,
            'date_of_birth': birth_date,
            'gender': gender,
            'marital_status': random.choice(MARITAL_STATUSES),
            'dependents': max(0, int(np.random.normal(1.5, 1.5))),
            'education': random.choice(EDUCATION_LEVELS),
            'employment_status': random.choice(EMPLOYMENT_STATUSES),
            'employment_length_years': max(0, int(np.random.normal(8, 5))),
            'annual_income': max(15000, int(np.random.normal(65000, 30000))),
            'address': fake.street_address(),
            'city': fake.city(),
            'state': fake.state(),
            'postal_code': fake.zipcode(),
            'country': 'USA',
            'phone': fake.phone_number(),
            'email': fake.email(),
            'registration_date': registration_date.date()
        })
    
    return pd.DataFrame(customers)


def generate_credit_bureau_data(customer_df: pd.DataFrame) -> pd.DataFrame:
    """Generate sample credit bureau data for customers."""
    logger.info("Generating credit bureau data...")
    
    bureau_data = []
    for _, customer in customer_df.iterrows():
        # Generate 1-3 credit reports per customer
        num_reports = random.randint(1, 3)
        
        for i in range(num_reports):
            # Report date between 1 day and 2 years ago
            report_days_ago = random.randint(1, 730)
            report_date = datetime.now() - timedelta(days=report_days_ago)
            
            # Credit score between 300 and 850, normally distributed
            credit_score = min(850, max(300, int(np.random.normal(680, 100))))
            
            # Total accounts between 1 and 30
            total_accounts = random.randint(1, 30)
            open_accounts = random.randint(1, total_accounts)
            closed_accounts = total_accounts - open_accounts
            
            # Delinquent accounts (0-20% of total)
            delinquent_accounts = random.randint(0, max(1, int(total_accounts * 0.2)))
            
            # Total balance and credit limit
            total_balance = max(0, np.random.normal(30000, 20000))
            total_credit_limit = max(total_balance, np.random.normal(50000, 30000))
            
            # Credit utilization ratio
            credit_utilization_ratio = min(1.0, max(0.0, total_balance / total_credit_limit if total_credit_limit > 0 else 0))
            
            # Length of credit history in months
            length_of_credit_history_months = random.randint(12, 30 * 12)
            
            bureau_data.append({
                'report_id': str(uuid.uuid4()),
                'customer_id': customer['customer_id'],
                'report_date': report_date.date(),
                'credit_score': credit_score,
                'total_accounts': total_accounts,
                'open_accounts': open_accounts,
                'closed_accounts': closed_accounts,
                'delinquent_accounts': delinquent_accounts,
                'total_balance': round(total_balance, 2),
                'total_credit_limit': round(total_credit_limit, 2),
                'credit_utilization_ratio': round(credit_utilization_ratio, 2),
                'length_of_credit_history_months': length_of_credit_history_months,
                'hard_inquiries_last_12m': random.randint(0, 10),
                'collections_last_12m': random.randint(0, 5),
                'public_records': random.randint(0, 3),
                'bankruptcy_count': random.randint(0, 2),
                'foreclosure_count': random.randint(0, 1),
                'source': random.choice(['Experian', 'Equifax', 'TransUnion'])
            })
    
    return pd.DataFrame(bureau_data)


def generate_loan_applications(customer_df: pd.DataFrame) -> pd.DataFrame:
    """Generate sample loan application data."""
    logger.info("Generating loan application data...")
    
    applications = []
    for _, customer in customer_df.iterrows():
        # Generate 0-5 loan applications per customer
        num_applications = random.randint(0, 5)
        
        for i in range(num_applications):
            # Application date between 1 day and 3 years ago
            app_days_ago = random.randint(1, 1095)
            application_date = datetime.now() - timedelta(days=app_days_ago)
            
            # Requested amount based on loan type
            loan_type = random.choice(LOAN_TYPES)
            if loan_type == 'Mortgage':
                requested_amount = max(100000, np.random.normal(300000, 150000))
            elif loan_type == 'Auto':
                requested_amount = max(5000, np.random.normal(25000, 10000))
            elif loan_type == 'Business':
                requested_amount = max(10000, np.random.normal(100000, 50000))
            else:
                requested_amount = max(1000, np.random.normal(15000, 10000))
            
            # Term in months based on loan type
            if loan_type == 'Mortgage':
                term_months = random.choice([180, 240, 360])
            elif loan_type == 'Auto':
                term_months = random.choice([36, 48, 60, 72])
            else:
                term_months = random.choice([12, 24, 36, 48, 60])
            
            # Application status
            status = random.choice(LOAN_STATUSES)
            
            # Decision date if not pending
            decision_date = None
            if status != 'Pending':
                decision_days_ago = max(1, app_days_ago - random.randint(1, 14))
                decision_date = (datetime.now() - timedelta(days=decision_days_ago)).date()
            
            # Approval details
            approval_score = None
            approved_amount = None
            decision_reason = None
            
            if status == 'Approved':
                approval_score = random.randint(600, 900)
                approved_amount = max(1000, requested_amount * random.uniform(0.7, 1.0))
            elif status == 'Rejected':
                decision_reason = random.choice([
                    'Insufficient credit history',
                    'High debt-to-income ratio',
                    'Low credit score',
                    'Income verification failed',
                    'Employment history insufficient'
                ])
            
            application_id = str(uuid.uuid4())
            
            applications.append({
                'application_id': application_id,
                'customer_id': customer['customer_id'],
                'application_date': application_date.date(),
                'loan_type': loan_type,
                'loan_purpose': random.choice(LOAN_PURPOSES),
                'requested_amount': round(requested_amount, 2),
                'term_months': term_months,
                'interest_rate': round(random.uniform(3.0, 15.0), 2),
                'status': status,
                'decision_date': decision_date,
                'decision_reason': decision_reason,
                'underwriter_id': f"UW{random.randint(1000, 9999)}",
                'approval_score': approval_score,
                'approved_amount': round(approved_amount, 2) if approved_amount else None
            })
    
    return pd.DataFrame(applications)


def create_sample_data():
    """Main function to create and save sample data."""
    try:
        # Generate customer profiles
        customer_df = generate_customer_profiles(NUM_CUSTOMERS)
        save_df_to_table(customer_df, 'customer_profile', 'append')
        logger.info(f"Saved {len(customer_df)} customer profiles")
        
        # Generate credit bureau data
        bureau_df = generate_credit_bureau_data(customer_df)
        save_df_to_table(bureau_df, 'credit_bureau_data', 'append')
        logger.info(f"Saved {len(bureau_df)} credit bureau records")
        
        # Generate loan applications
        application_df = generate_loan_applications(customer_df)
        save_df_to_table(application_df, 'loan_application', 'append')
        logger.info(f"Saved {len(application_df)} loan applications")
        
        # More tables could be generated here
        
        logger.info("Sample data generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    create_sample_data()