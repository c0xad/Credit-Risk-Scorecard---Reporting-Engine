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
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

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

# Initialize Faker with multiple locales
fake = Faker(['en_US', 'en_GB', 'en_CA'])

# Constants
DEFAULT_NUM_CUSTOMERS = 1000
DEFAULT_SEED = 42
DEFAULT_COUNTRIES = ['USA', 'Canada', 'UK']
DEFAULT_START_DATE = datetime.now() - timedelta(days=3*365)  # 3 years ago

# Loan related constants
LOAN_TYPES = ['Personal', 'Auto', 'Mortgage', 'Credit Card', 'Business', 'Student', 'Home Equity', 'Line of Credit']
LOAN_PURPOSES = ['Debt Consolidation', 'Home Improvement', 'Education', 'Medical Expenses', 
                'Vacation', 'Wedding', 'Major Purchase', 'Vehicle Purchase', 'Business Expansion',
                'Emergency Expenses', 'Home Purchase', 'Refinance', 'Investment', 'Tax Payment']
LOAN_STATUSES = ['Approved', 'Rejected', 'Pending', 'Withdrawn', 'Under Review', 'Conditionally Approved']
PAYMENT_STATUSES = ['On Time', 'Late 30 Days', 'Late 60 Days', 'Late 90+ Days', 'Default', 'Charged Off', 'Deferred']
RISK_BANDS = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extreme']

# Personal information constants
EMPLOYMENT_STATUSES = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed', 'Retired', 'Student', 'Contract', 'Seasonal']
MARITAL_STATUSES = ['Single', 'Married', 'Divorced', 'Widowed', 'Separated', 'Domestic Partnership']
EDUCATION_LEVELS = ['High School', 'Associate Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 
                   'Doctoral Degree', 'Professional Degree', 'Some College', 'Trade School', 'GED']
INCOME_CATEGORIES = ['Low', 'Below Average', 'Average', 'Above Average', 'High', 'Very High']
HOUSING_STATUSES = ['Own', 'Rent', 'Live with Parents', 'Military Housing', 'Other']

# Banking and financial constants
BANK_NAMES = ['First National Bank', 'City Credit Union', 'Capital One', 'Wells Fargo', 'Chase', 'Bank of America', 
             'TD Bank', 'PNC Bank', 'US Bank', 'Citi Bank', 'Regions Bank', 'Ally Bank', 'Discover Bank']
ACCOUNT_TYPES = ['Checking', 'Savings', 'Money Market', 'Certificate of Deposit', 'Investment', 'Retirement']
CARD_TYPES = ['Visa', 'Mastercard', 'American Express', 'Discover']
CREDIT_BUREAU_SOURCES = ['Experian', 'Equifax', 'TransUnion', 'Combined']

def generate_customer_profiles(num_customers: int, start_date: datetime = DEFAULT_START_DATE, 
                              countries: List[str] = DEFAULT_COUNTRIES) -> pd.DataFrame:
    """
    Generate sample customer profile data with realistic distributions.
    
    Args:
        num_customers: Number of customer profiles to generate
        start_date: Earliest date customers could have registered
        countries: List of countries to assign customers to
        
    Returns:
        DataFrame containing customer profile data
    """
    logger.info(f"Generating {num_customers} customer profiles...")
    
    customers = []
    for i in range(num_customers):
        customer_id = f"CUST{i+10000:05d}"
        
        # Demographic information with realistic distributions
        country = random.choice(countries)
        if country == 'USA':
            fake.locale = 'en_US'
        elif country == 'UK':
            fake.locale = 'en_GB'
        elif country == 'Canada':
            fake.locale = 'en_CA'
            
        first_name = fake.first_name()
        last_name = fake.last_name()
        gender = random.choice(['Male', 'Female', 'Other', 'Prefer not to say', None])
        
        # Age with realistic distribution: more customers in 25-55 range
        age_weights = [0.1, 0.2, 0.3, 0.25, 0.15]  # weights for age brackets 18-25, 26-35, 36-45, 46-55, 56-75
        age_bracket = random.choices([0, 1, 2, 3, 4], weights=age_weights)[0]
        
        if age_bracket == 0:
            min_age, max_age = 18, 25
        elif age_bracket == 1:
            min_age, max_age = 26, 35
        elif age_bracket == 2:
            min_age, max_age = 36, 45
        elif age_bracket == 3:
            min_age, max_age = 46, 55
        else:
            min_age, max_age = 56, 75
            
        birth_date = fake.date_of_birth(minimum_age=min_age, maximum_age=max_age)
        
        # Calculate actual age
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        
        # Registration date between start_date and now
        days_since_start = (datetime.now() - start_date).days
        reg_days_ago = random.randint(1, days_since_start)
        registration_date = datetime.now() - timedelta(days=reg_days_ago)
        
        # Determine marital status based on age
        if age < 25:
            marital_probs = [0.8, 0.15, 0.01, 0.01, 0.02, 0.01]  # mostly single
        elif age < 35:
            marital_probs = [0.4, 0.4, 0.1, 0.02, 0.05, 0.03]  # mix of single and married
        elif age < 55:
            marital_probs = [0.2, 0.5, 0.15, 0.05, 0.07, 0.03]  # mostly married
        else:
            marital_probs = [0.15, 0.45, 0.15, 0.15, 0.07, 0.03]  # more widowed and divorced
        
        marital_status = random.choices(MARITAL_STATUSES, weights=marital_probs)[0]
        
        # Determine education based on age
        if age < 25:
            edu_weights = [0.3, 0.2, 0.2, 0.05, 0.01, 0.01, 0.2, 0.03]
        elif age < 35:
            edu_weights = [0.15, 0.15, 0.35, 0.15, 0.05, 0.05, 0.05, 0.05]
        else:
            edu_weights = [0.2, 0.1, 0.3, 0.15, 0.1, 0.05, 0.05, 0.05]
            
        education = random.choices(EDUCATION_LEVELS[:8], weights=edu_weights)[0]
        
        # Employment status depends on age and education
        if age < 23 and education in ['High School', 'Some College', 'Associate Degree']:
            emp_weights = [0.3, 0.3, 0.05, 0.05, 0.0, 0.3, 0.0, 0.0]  # more students
        elif age > 60:
            emp_weights = [0.2, 0.1, 0.1, 0.1, 0.45, 0.0, 0.05, 0.0]  # more retired
        else:
            emp_weights = [0.7, 0.1, 0.1, 0.05, 0.01, 0.01, 0.02, 0.01]  # more full-time
            
        employment_status = random.choices(EMPLOYMENT_STATUSES, weights=emp_weights)[0]
        
        # Employment length depends on age and employment status
        if employment_status in ['Student', 'Unemployed']:
            employment_length_years = 0
        elif employment_status == 'Retired':
            employment_length_years = max(0, int(np.random.normal(30, 10)))
        else:
            max_possible_experience = max(0, age - 18)  # can't work before 18
            employment_length_years = min(max_possible_experience, max(0, int(np.random.normal(max_possible_experience/2, max_possible_experience/4))))
        
        # Income depends on employment status, education, and age
        if employment_status in ['Unemployed', 'Student']:
            annual_income = max(0, int(np.random.normal(5000, 3000)))
        elif employment_status == 'Retired':
            annual_income = max(15000, int(np.random.normal(40000, 20000)))
        elif employment_status == 'Part-time':
            annual_income = max(10000, int(np.random.normal(25000, 10000)))
        elif education in ['Master\'s Degree', 'Doctoral Degree', 'Professional Degree']:
            annual_income = max(40000, int(np.random.normal(100000, 40000)))
        else:
            annual_income = max(20000, int(np.random.normal(60000, 25000)))
            
        # Determine income category
        if annual_income < 25000:
            income_category = 'Low'
        elif annual_income < 50000:
            income_category = 'Below Average'
        elif annual_income < 75000:
            income_category = 'Average'
        elif annual_income < 100000:
            income_category = 'Above Average'
        elif annual_income < 150000:
            income_category = 'High'
        else:
            income_category = 'Very High'
        
        # Housing status with realistic distribution
        if age < 25:
            housing_weights = [0.1, 0.3, 0.5, 0.05, 0.05]  # more living with parents
        elif age < 35:
            housing_weights = [0.3, 0.5, 0.1, 0.05, 0.05]  # more renting
        else:
            housing_weights = [0.6, 0.3, 0.03, 0.02, 0.05]  # more owning
            
        housing_status = random.choices(HOUSING_STATUSES, weights=housing_weights)[0]
        
        # Number of dependents based on age and marital status
        if age < 25 or marital_status in ['Single', 'Separated', 'Divorced']:
            dependents = max(0, int(np.random.normal(0.3, 0.5)))
        else:
            dependents = max(0, int(np.random.normal(1.5, 1.2)))
            
        # Banking information
        bank_name = random.choice(BANK_NAMES)
        has_checking_account = random.random() > 0.05  # 95% have checking
        has_savings_account = random.random() > 0.3    # 70% have savings
        has_credit_card = random.random() > 0.2        # 80% have credit card
        
        # Generate location information based on country
        if country == 'USA':
            address = fake.street_address()
            city = fake.city()
            state = fake.state()
            postal_code = fake.zipcode()
        elif country == 'UK':
            address = fake.street_address()
            city = fake.city()
            state = fake.county()
            postal_code = fake.postcode()
        else:  # Canada
            address = fake.street_address()
            city = fake.city()
            state = fake.province()
            postal_code = fake.postcode()
        
        customers.append({
            'customer_id': customer_id,
            'first_name': first_name,
            'last_name': last_name,
            'date_of_birth': birth_date,
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'dependents': dependents,
            'education': education,
            'employment_status': employment_status,
            'employment_length_years': employment_length_years,
            'annual_income': annual_income,
            'income_category': income_category,
            'housing_status': housing_status,
            'address': address,
            'city': city,
            'state': state,
            'postal_code': postal_code,
            'country': country,
            'phone': fake.phone_number(),
            'email': fake.email(),
            'registration_date': registration_date.date(),
            'bank_name': bank_name,
            'has_checking_account': has_checking_account,
            'has_savings_account': has_savings_account,
            'has_credit_card': has_credit_card
        })
    
    return pd.DataFrame(customers)


def generate_credit_bureau_data(customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate sample credit bureau data for customers with realistic correlations.
    
    Args:
        customer_df: DataFrame containing customer profile data
        
    Returns:
        DataFrame containing credit bureau data
    """
    logger.info("Generating credit bureau data...")
    
    bureau_data = []
    for _, customer in customer_df.iterrows():
        # Base credit profile factors on income, age, and employment
        income = customer['annual_income']
        age = customer['age']
        employment_status = customer['employment_status']
        
        # Financial stability factor (0-1) based on customer profile
        # Higher values indicate more financial stability
        stability_factor = 0.0
        
        # Age factor - older customers tend to have more stable credit
        if age < 25:
            stability_factor += 0.1
        elif age < 35:
            stability_factor += 0.2
        elif age < 45:
            stability_factor += 0.3
        elif age < 55:
            stability_factor += 0.4
        else:
            stability_factor += 0.5
            
        # Income factor
        if income < 30000:
            stability_factor += 0.05
        elif income < 60000:
            stability_factor += 0.15
        elif income < 100000:
            stability_factor += 0.25
        else:
            stability_factor += 0.35
            
        # Employment status factor
        if employment_status == 'Full-time':
            stability_factor += 0.15
        elif employment_status in ['Part-time', 'Self-employed', 'Contract']:
            stability_factor += 0.1
        elif employment_status == 'Retired':
            stability_factor += 0.15
        else:
            stability_factor += 0.0
            
        # Normalize factor to 0-1 range
        stability_factor = min(1.0, stability_factor)
        
        # Generate 1-3 credit reports per customer with different timestamps
        num_reports = random.randint(1, 3)
        
        for i in range(num_reports):
            # Report date between 1 day and 2 years ago
            report_days_ago = random.randint(1, 730)
            report_date = datetime.now() - timedelta(days=report_days_ago)
            
            # Credit score correlated with stability_factor (300-850)
            # Higher stability factor means higher likely credit score with less variance
            base_score = 300 + (550 * stability_factor)
            variance = 150 * (1 - stability_factor) + 30  # Less variance for stable customers
            credit_score = min(850, max(300, int(np.random.normal(base_score, variance))))
            
            # Account metrics based on stability and score
            # More stable customers tend to have more accounts but less delinquency
            account_factor = min(1.0, stability_factor + random.uniform(-0.2, 0.2))
            
            # Total accounts between 1 and 30
            total_accounts = min(30, max(1, int(np.random.normal(5 + (20 * account_factor), 5))))
            
            # Open vs closed accounts
            open_proportion = min(1.0, max(0.3, random.uniform(0.5, 0.9) + (stability_factor * 0.1)))
            open_accounts = max(1, int(total_accounts * open_proportion))
            closed_accounts = total_accounts - open_accounts
            
            # Delinquent accounts correlated negatively with stability
            delinquent_factor = max(0.01, 0.3 - (stability_factor * 0.3) + random.uniform(-0.05, 0.1))
            delinquent_accounts = max(0, min(open_accounts, int(total_accounts * delinquent_factor)))
            
            # Payment history (percentage of on-time payments)
            payment_history_factor = min(1.0, stability_factor + random.uniform(-0.1, 0.1))
            payment_history_percent = max(0, min(100, int(100 * payment_history_factor)))
            
            # Total balance and credit limit related to income and stability
            avg_balance_factor = random.uniform(0.3, 1.5) * (1 - (stability_factor * 0.5))
            total_balance = max(0, min(1000000, income * avg_balance_factor))
            
            # Credit limit based on income and credit score
            limit_factor = (credit_score / 850) * random.uniform(1.5, 3.0)
            total_credit_limit = max(total_balance, income * limit_factor)
            
            # Credit utilization ratio
            credit_utilization_ratio = min(1.0, max(0.0, total_balance / total_credit_limit if total_credit_limit > 0 else 0))
            
            # Length of credit history correlated with age
            max_history_months = max(1, (age - 18) * 12)  # Can't have credit before 18
            length_of_credit_history_months = min(max_history_months, random.randint(12, 30 * 12))
            
            # Recent inquiries inversely related to stability
            inquiry_factor = max(0, 0.6 - (stability_factor * 0.5)) + random.uniform(-0.1, 0.1)
            hard_inquiries = max(0, min(15, int(inquiry_factor * 10)))
            
            # Collections and public records also inversely related to stability
            collections_factor = max(0, 0.5 - (stability_factor * 0.5)) + random.uniform(-0.1, 0.1)
            collections_count = max(0, min(10, int(collections_factor * 5)))
            
            public_records_factor = max(0, 0.4 - (stability_factor * 0.4)) + random.uniform(-0.1, 0.1)
            public_records = max(0, min(5, int(public_records_factor * 5)))
            
            # Determine bankruptcy and foreclosure (rare events)
            bankruptcy_probability = max(0, 0.2 - (stability_factor * 0.2))
            bankruptcy_count = 1 if random.random() < bankruptcy_probability else 0
            
            foreclosure_probability = max(0, 0.15 - (stability_factor * 0.15))
            foreclosure_count = 1 if random.random() < foreclosure_probability else 0
            
            # Calculate a risk band based on credit score
            if credit_score >= 750:
                risk_band = 'Very Low'
            elif credit_score >= 700:
                risk_band = 'Low'
            elif credit_score >= 650:
                risk_band = 'Medium'
            elif credit_score >= 600:
                risk_band = 'High'
            elif credit_score >= 550:
                risk_band = 'Very High'
            else:
                risk_band = 'Extreme'
                
            # Select credit bureau source
            source = random.choice(CREDIT_BUREAU_SOURCES)
            
            bureau_data.append({
                'report_id': str(uuid.uuid4()),
                'customer_id': customer['customer_id'],
                'report_date': report_date.date(),
                'credit_score': credit_score,
                'risk_band': risk_band,
                'total_accounts': total_accounts,
                'open_accounts': open_accounts,
                'closed_accounts': closed_accounts,
                'delinquent_accounts': delinquent_accounts,
                'payment_history_percent': payment_history_percent,
                'total_balance': round(total_balance, 2),
                'total_credit_limit': round(total_credit_limit, 2),
                'credit_utilization_ratio': round(credit_utilization_ratio, 2),
                'length_of_credit_history_months': length_of_credit_history_months,
                'hard_inquiries_last_12m': hard_inquiries,
                'collections_last_12m': collections_count,
                'public_records': public_records,
                'bankruptcy_count': bankruptcy_count,
                'foreclosure_count': foreclosure_count,
                'source': source,
                'stability_factor': round(stability_factor, 2)  # Added for transparency
            })
    
    return pd.DataFrame(bureau_data)


def generate_loan_applications(customer_df: pd.DataFrame, bureau_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate sample loan application data with realistic correlations to customer profile and credit data.
    
    Args:
        customer_df: DataFrame containing customer profile data
        bureau_df: DataFrame containing credit bureau data
        
    Returns:
        DataFrame containing loan application data
    """
    logger.info("Generating loan application data...")
    
    # Create a dictionary of the latest credit report for each customer
    latest_credit_reports = {}
    for _, report in bureau_df.iterrows():
        customer_id = report['customer_id']
        report_date = report['report_date']
        
        if customer_id not in latest_credit_reports or report_date > latest_credit_reports[customer_id]['report_date']:
            latest_credit_reports[customer_id] = report.to_dict()
    
    applications = []
    for _, customer in customer_df.iterrows():
        customer_id = customer['customer_id']
        income = customer['annual_income']
        age = customer['age']
        employment_status = customer['employment_status']
        
        # Determine number of applications based on age and income
        # Older customers with higher income tend to have more loan applications
        base_app_count = 0
        if age < 25:
            base_app_count = 1
        elif age < 35:
            base_app_count = 2
        elif age < 50:
            base_app_count = 3
        else:
            base_app_count = 2
            
        if income < 30000:
            income_factor = 0.5
        elif income < 60000:
            income_factor = 1.0
        elif income < 100000:
            income_factor = 1.5
        else:
            income_factor = 2.0
            
        num_applications = max(0, int(np.random.normal(base_app_count * income_factor, 1.5)))
        num_applications = min(7, num_applications)  # Cap at 7 applications
        
        # Get credit information if available
        credit_score = 650  # Default value if no credit report
        if customer_id in latest_credit_reports:
            credit_score = latest_credit_reports[customer_id]['credit_score']
        
        for i in range(num_applications):
            # Application date between 1 day and 3 years ago, more recent applications more likely
            recency_factor = np.random.exponential(0.5)  # Lower values are more recent
            app_days_ago = max(1, min(1095, int(1095 * recency_factor)))
            application_date = datetime.now() - timedelta(days=app_days_ago)
            
            # Determine loan type based on age, income, and a random factor
            loan_type_weights = []
            
            # Mortgage loans more common for older customers with higher income
            mortgage_prob = 0.05
            if age > 30 and income > 50000:
                mortgage_prob = 0.3
            if age > 40 and income > 80000:
                mortgage_prob = 0.5
                
            # Auto loans common for all age groups but increase with income
            auto_prob = 0.1 + (min(0.3, income / 200000))
            
            # Business loans more common for middle-aged customers with higher income
            business_prob = 0.05
            if age > 30 and age < 60 and income > 70000:
                business_prob = 0.2
                
            # Student loans more common for younger customers
            student_prob = 0.0
            if age < 30:
                student_prob = 0.3
            if age < 25:
                student_prob = 0.5
                
            # Credit cards and personal loans common across all segments
            credit_card_prob = 0.2
            personal_prob = 0.2
            
            # Home equity increases with age as equity builds
            home_equity_prob = 0.0
            if age > 40 and customer['housing_status'] == 'Own':
                home_equity_prob = 0.2
                
            # Line of credit more common with higher incomes
            line_of_credit_prob = 0.05
            if income > 75000:
                line_of_credit_prob = 0.15
                
            # Create weight list in same order as LOAN_TYPES
            loan_type_weights = [
                personal_prob,                 # Personal
                auto_prob,                     # Auto
                mortgage_prob,                 # Mortgage
                credit_card_prob,              # Credit Card
                business_prob,                 # Business
                student_prob,                  # Student
                home_equity_prob,              # Home Equity
                line_of_credit_prob            # Line of Credit
            ]
            
            # Normalize weights
            total_weight = sum(loan_type_weights)
            if total_weight > 0:
                loan_type_weights = [w / total_weight for w in loan_type_weights]
            else:
                loan_type_weights = [1/len(LOAN_TYPES)] * len(LOAN_TYPES)
                
            loan_type = random.choices(LOAN_TYPES, weights=loan_type_weights)[0]
            
            # Determine loan purpose based on loan type
            if loan_type == 'Mortgage':
                purpose_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0.3, 0, 0]
            elif loan_type == 'Auto':
                purpose_weights = [0, 0, 0, 0, 0, 0, 0.3, 0.7, 0, 0, 0, 0, 0, 0]
            elif loan_type == 'Student':
                purpose_weights = [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif loan_type == 'Business':
                purpose_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0, 0.3, 0]
            elif loan_type == 'Credit Card':
                purpose_weights = [0.3, 0, 0, 0.1, 0.1, 0, 0.2, 0, 0, 0.3, 0, 0, 0, 0]
            elif loan_type == 'Home Equity':
                purpose_weights = [0.2, 0.5, 0.1, 0.1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0]
            else:  # Personal or Line of Credit
                purpose_weights = [0.3, 0.15, 0.1, 0.1, 0.05, 0.05, 0.15, 0, 0, 0.1, 0, 0, 0, 0]
                
            loan_purpose = random.choices(LOAN_PURPOSES, weights=purpose_weights)[0]
            
            # Requested amount based on loan type, purpose, and income
            income_multiplier = min(5.0, max(0.5, income / 50000))
            
            if loan_type == 'Mortgage':
                base_amount = 250000
                variance = 150000
                requested_amount = max(100000, base_amount * income_multiplier + np.random.normal(0, variance))
            elif loan_type == 'Auto':
                base_amount = 25000
                variance = 10000
                requested_amount = max(5000, base_amount + np.random.normal(0, variance))
            elif loan_type == 'Business':
                base_amount = 75000
                variance = 50000
                requested_amount = max(10000, base_amount * income_multiplier + np.random.normal(0, variance))
            elif loan_type == 'Student':
                base_amount = 20000
                variance = 10000
                requested_amount = max(5000, base_amount + np.random.normal(0, variance))
            elif loan_type == 'Home Equity':
                base_amount = 50000
                variance = 30000
                requested_amount = max(10000, base_amount * income_multiplier + np.random.normal(0, variance))
            elif loan_type == 'Credit Card':
                base_amount = 5000
                variance = 3000
                requested_amount = max(1000, base_amount * income_multiplier + np.random.normal(0, variance))
            else:  # Personal or Line of Credit
                base_amount = 15000
                variance = 10000
                requested_amount = max(1000, base_amount * income_multiplier + np.random.normal(0, variance))
            
            # Term in months based on loan type
            if loan_type == 'Mortgage':
                term_choices = [180, 240, 360]
                term_weights = [0.1, 0.2, 0.7]
                term_months = random.choices(term_choices, weights=term_weights)[0]
            elif loan_type == 'Auto':
                term_choices = [36, 48, 60, 72, 84]
                term_weights = [0.1, 0.2, 0.4, 0.2, 0.1]
                term_months = random.choices(term_choices, weights=term_weights)[0]
            elif loan_type == 'Business':
                term_choices = [12, 24, 36, 48, 60, 84, 120]
                term_weights = [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05]
                term_months = random.choices(term_choices, weights=term_weights)[0]
            elif loan_type == 'Student':
                term_choices = [60, 120, 180, 240]
                term_weights = [0.1, 0.4, 0.3, 0.2]
                term_months = random.choices(term_choices, weights=term_weights)[0]
            elif loan_type == 'Home Equity':
                term_choices = [60, 120, 180, 240]
                term_weights = [0.1, 0.3, 0.4, 0.2]
                term_months = random.choices(term_choices, weights=term_weights)[0]
            else:  # Personal, Credit Card, Line of Credit
                term_choices = [12, 24, 36, 48, 60]
                term_weights = [0.2, 0.3, 0.3, 0.1, 0.1]
                term_months = random.choices(term_choices, weights=term_weights)[0]
            
            # Calculate interest rate based on loan type, credit score, and application date
            # Base rates decrease for newer applications (reflecting current trend)
            recency_adjustment = app_days_ago / 1095 * 0.02  # Up to 2% higher for older applications
            
            # Credit score adjustment (better score = lower rate)
            credit_adjustment = max(0, (700 - credit_score) / 100 * 0.05)  # Up to 5% higher for poor credit
            
            if loan_type == 'Mortgage':
                base_rate = 0.04 + recency_adjustment
            elif loan_type == 'Auto':
                base_rate = 0.05 + recency_adjustment
            elif loan_type == 'Business':
                base_rate = 0.08 + recency_adjustment
            elif loan_type == 'Student':
                base_rate = 0.06 + recency_adjustment
            elif loan_type == 'Credit Card':
                base_rate = 0.18 + recency_adjustment
            elif loan_type == 'Home Equity':
                base_rate = 0.06 + recency_adjustment
            else:  # Personal or Line of Credit
                base_rate = 0.10 + recency_adjustment
                
            interest_rate = base_rate + credit_adjustment + random.uniform(-0.01, 0.01)  # Add some randomness
            interest_rate = max(0.01, min(0.30, interest_rate))  # Cap between 1% and 30%
            
            # Calculate monthly payment (simplified formula)
            monthly_interest = interest_rate / 12
            if monthly_interest > 0:
                monthly_payment = requested_amount * (monthly_interest * (1 + monthly_interest) ** term_months) / ((1 + monthly_interest) ** term_months - 1)
            else:
                monthly_payment = requested_amount / term_months
                
            # Determine application status based on credit score, income, and loan amount
            # Calculate debt-to-income ratio (DTI)
            dti = monthly_payment * 12 / income if income > 0 else 99
            
            # Approval probability
            approval_probability = 0.0
            
            # Credit score factor (0-0.5)
            if credit_score >= 750:
                credit_factor = 0.5
            elif credit_score >= 700:
                credit_factor = 0.4
            elif credit_score >= 650:
                credit_factor = 0.3
            elif credit_score >= 600:
                credit_factor = 0.2
            elif credit_score >= 550:
                credit_factor = 0.1
            else:
                credit_factor = 0.0
                
            # DTI factor (0-0.4)
            if dti <= 0.2:
                dti_factor = 0.4
            elif dti <= 0.3:
                dti_factor = 0.3
            elif dti <= 0.4:
                dti_factor = 0.2
            elif dti <= 0.5:
                dti_factor = 0.1
            else:
                dti_factor = 0.0
                
            # Employment factor (0-0.1)
            if employment_status in ['Full-time', 'Retired']:
                employment_factor = 0.1
            elif employment_status in ['Part-time', 'Self-employed']:
                employment_factor = 0.05
            else:
                employment_factor = 0.0
                
            # Calculate final probability with a random factor
            approval_probability = credit_factor + dti_factor + employment_factor
            approval_probability = min(0.95, max(0.05, approval_probability))
            
            # Determine status based on approval probability
            random_factor = random.random()
            if random_factor < 0.05:  # 5% are withdrawn regardless
                status = 'Withdrawn'
            elif random_factor < 0.10:  # Another 5% are pending
                status = 'Pending'
            elif random_factor < 0.15:  # Another 5% are under review
                status = 'Under Review'
            elif random_factor < approval_probability + 0.15:  # Approved based on probability
                if random_factor < approval_probability + 0.05:
                    status = 'Approved'
                else:
                    status = 'Conditionally Approved'
            else:  # Rejected
                status = 'Rejected'
            
            # Decision date if not pending or under review
            decision_date = None
            if status not in ['Pending', 'Under Review']:
                # Decision typically takes 1-14 days
                decision_days_ago = max(1, app_days_ago - random.randint(1, 14))
                decision_date = (datetime.now() - timedelta(days=decision_days_ago)).date()
            
            # Approval details
            approval_score = None
            approved_amount = None
            decision_reason = None
            
            if status in ['Approved', 'Conditionally Approved']:
                # Approval score (internal risk score, not the same as credit score)
                approval_score = random.randint(600, 900)
                
                # Approved amount (usually slightly less than requested for high-risk customers)
                if credit_score >= 700:
                    amount_factor = random.uniform(0.95, 1.0)
                elif credit_score >= 650:
                    amount_factor = random.uniform(0.85, 0.95)
                else:
                    amount_factor = random.uniform(0.7, 0.85)
                    
                approved_amount = max(1000, requested_amount * amount_factor)
                
                if status == 'Conditionally Approved':
                    decision_reason = random.choice([
                        'Additional income verification required',
                        'Collateral documentation needed',
                        'Employment verification pending',
                        'Lower debt-to-income ratio required',
                        'Additional co-signer needed'
                    ])
            elif status == 'Rejected':
                dti_too_high = dti > 0.4
                low_credit = credit_score < 600
                income_insufficient = requested_amount > income * 0.8
                
                if dti_too_high and low_credit:
                    decision_reason = 'High debt-to-income ratio and low credit score'
                elif dti_too_high:
                    decision_reason = 'Debt-to-income ratio too high'
                elif low_credit:
                    decision_reason = 'Insufficient credit score'
                elif income_insufficient:
                    decision_reason = 'Income insufficient for requested amount'
                else:
                    decision_reason = random.choice([
                        'Insufficient credit history',
                        'Income verification failed',
                        'Employment history insufficient',
                        'High risk based on internal models',
                        'Excessive recent credit inquiries'
                    ])
            
            application_id = str(uuid.uuid4())
            
            applications.append({
                'application_id': application_id,
                'customer_id': customer_id,
                'application_date': application_date.date(),
                'loan_type': loan_type,
                'loan_purpose': loan_purpose,
                'requested_amount': round(requested_amount, 2),
                'term_months': term_months,
                'interest_rate': round(interest_rate * 100, 2),  # Store as percentage
                'monthly_payment': round(monthly_payment, 2),
                'debt_to_income_ratio': round(dti * 100, 2),  # Store as percentage
                'status': status,
                'decision_date': decision_date,
                'decision_reason': decision_reason,
                'underwriter_id': f"UW{random.randint(1000, 9999)}",
                'approval_score': approval_score,
                'approved_amount': round(approved_amount, 2) if approved_amount else None,
                'credit_score_at_application': credit_score
            })
    
    return pd.DataFrame(applications)


def generate_payment_history(application_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate payment history for approved loan applications.
    
    Args:
        application_df: DataFrame containing loan application data
        
    Returns:
        DataFrame containing payment history data
    """
    logger.info("Generating payment history data...")
    
    payment_history = []
    
    # Only generate payment history for approved applications
    approved_applications = application_df[application_df['status'] == 'Approved'].copy()
    
    for _, application in approved_applications.iterrows():
        application_id = application['application_id']
        customer_id = application['customer_id']
        approved_amount = application['approved_amount']
        term_months = application['term_months']
        interest_rate = application['interest_rate'] / 100  # Convert from percentage
        monthly_payment = application['monthly_payment']
        decision_date = application['decision_date']
        
        # Skip if any required field is missing
        if pd.isna(approved_amount) or pd.isna(term_months) or pd.isna(decision_date):
            continue
        
        # Loan start date (usually a few days after approval)
        loan_start_date = datetime.strptime(str(decision_date), '%Y-%m-%d') + timedelta(days=random.randint(3, 10))
        
        # Generate a payment schedule
        # Number of payments made so far
        days_since_start = (datetime.now() - loan_start_date).days
        months_since_start = days_since_start // 30
        num_payments = min(term_months, max(0, months_since_start))
        
        # Calculate amortization schedule
        remaining_balance = approved_amount
        monthly_interest_rate = interest_rate / 12
        
        # Generate customer payment behavior profile
        # Payment behavior score 0-1, higher is better
        payment_behavior = random.uniform(0.5, 1.0)
        
        # Missed payment probability (decreases with better behavior score)
        missed_payment_probability = max(0.01, 0.3 - (payment_behavior * 0.3))
        
        # Late payment probability (decreases with better behavior score)
        late_payment_probability = max(0.05, 0.4 - (payment_behavior * 0.4))
        
        # Partial payment probability
        partial_payment_probability = max(0.05, 0.3 - (payment_behavior * 0.3))
        
        for payment_number in range(1, num_payments + 1):
            # Calculate payment details
            interest_payment = remaining_balance * monthly_interest_rate
            principal_payment = monthly_payment - interest_payment
            
            # Adjust for final payment
            if payment_number == term_months:
                principal_payment = remaining_balance
                monthly_payment = principal_payment + interest_payment
            
            # Calculate scheduled payment date
            scheduled_date = loan_start_date + timedelta(days=30 * payment_number)
            
            # Determine payment status
            if random.random() < missed_payment_probability:
                payment_status = 'Missed'
                actual_payment = 0
                actual_date = None
            elif random.random() < late_payment_probability:
                payment_status = random.choice(['Late 30 Days', 'Late 60 Days', 'Late 90+ Days'])
                
                if payment_status == 'Late 30 Days':
                    days_late = random.randint(5, 30)
                elif payment_status == 'Late 60 Days':
                    days_late = random.randint(31, 60)
                else:
                    days_late = random.randint(61, 120)
                    
                actual_date = scheduled_date + timedelta(days=days_late)
                
                # For late payments, determine if partial or full
                if random.random() < partial_payment_probability:
                    actual_payment = monthly_payment * random.uniform(0.3, 0.9)
                else:
                    actual_payment = monthly_payment
            else:
                payment_status = 'On Time'
                actual_date = scheduled_date + timedelta(days=random.randint(-5, 5))  # Few days before or after
                
                # For on-time payments, determine if partial or full
                if random.random() < partial_payment_probability * 0.5:  # Less likely to be partial if on time
                    actual_payment = monthly_payment * random.uniform(0.5, 0.95)
                else:
                    actual_payment = monthly_payment
            
            # Update remaining balance
            if payment_status != 'Missed':
                actual_principal = min(remaining_balance, actual_payment - min(interest_payment, actual_payment))
                remaining_balance -= actual_principal
            
            # Add payment record
            payment_history.append({
                'payment_id': str(uuid.uuid4()),
                'application_id': application_id,
                'customer_id': customer_id,
                'payment_number': payment_number,
                'scheduled_date': scheduled_date.date(),
                'scheduled_amount': round(monthly_payment, 2),
                'actual_date': actual_date.date() if actual_date else None,
                'actual_amount': round(actual_payment, 2),
                'principal_amount': round(principal_payment, 2),
                'interest_amount': round(interest_payment, 2),
                'remaining_balance': round(remaining_balance, 2),
                'payment_status': payment_status
            })
    
    return pd.DataFrame(payment_history)


def create_sample_data(num_customers: int = DEFAULT_NUM_CUSTOMERS, seed: int = DEFAULT_SEED, 
                      export_only: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Main function to create and save sample data.
    
    Args:
        num_customers: Number of customer profiles to generate
        seed: Random seed for reproducibility
        export_only: If True, only export to CSV without saving to database
        
    Returns:
        Dictionary containing all generated DataFrames
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)
    
    try:
        logger.info(f"Generating sample data with {num_customers} customers (seed: {seed})...")
        
        # Generate customer profiles
        customer_df = generate_customer_profiles(num_customers)
        if not export_only:
            try:
                save_df_to_table(customer_df, 'customer_profile', 'append')
                logger.info(f"Saved {len(customer_df)} customer profiles to database")
            except Exception as e:
                logger.error(f"Error saving customer profiles to database: {e}")
        
        # Generate credit bureau data
        bureau_df = generate_credit_bureau_data(customer_df)
        if not export_only:
            try:
                save_df_to_table(bureau_df, 'credit_bureau_data', 'append')
                logger.info(f"Saved {len(bureau_df)} credit bureau records to database")
            except Exception as e:
                logger.error(f"Error saving credit bureau data to database: {e}")
        
        # Generate loan applications
        application_df = generate_loan_applications(customer_df, bureau_df)
        if not export_only:
            try:
                save_df_to_table(application_df, 'loan_application', 'append')
                logger.info(f"Saved {len(application_df)} loan applications to database")
            except Exception as e:
                logger.error(f"Error saving loan applications to database: {e}")
        
        # Generate payment history
        payment_df = generate_payment_history(application_df)
        if not export_only:
            try:
                save_df_to_table(payment_df, 'payment_history', 'append')
                logger.info(f"Saved {len(payment_df)} payment records to database")
            except Exception as e:
                logger.error(f"Error saving payment history to database: {e}")
        
        logger.info("Sample data generation completed successfully")
        
        # Return all dataframes for potential further processing
        return {
            'customers': customer_df,
            'credit_bureau': bureau_df,
            'applications': application_df,
            'payment_history': payment_df
        }
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Generate sample data for Credit Risk Scorecard database')
    parser.add_argument('--num_customers', type=int, default=DEFAULT_NUM_CUSTOMERS,
                       help=f'Number of customer profiles to generate (default: {DEFAULT_NUM_CUSTOMERS})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help=f'Random seed for reproducibility (default: {DEFAULT_SEED})')
    parser.add_argument('--export_csv', action='store_true',
                       help='Export data to CSV files')
    parser.add_argument('--export_only', action='store_true',
                       help='Export only to CSV files without saving to database')
    parser.add_argument('--csv_dir', type=str, default='./data',
                       help='Directory to store CSV files if --export_csv is used')
    
    args = parser.parse_args()
    
    # Create sample data
    data_dict = create_sample_data(args.num_customers, args.seed, args.export_only)
    
    # Export to CSV if requested
    if args.export_csv or args.export_only:
        csv_dir = Path(args.csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in data_dict.items():
            csv_path = csv_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Exported {len(df)} records to {csv_path}")