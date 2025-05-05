#!/usr/bin/env python3
"""
Data Preprocessing Module for Credit Risk Scorecard

This module contains functions for preprocessing data for credit risk modeling,
including data cleaning, feature engineering, binning, Weight of Evidence (WoE)
calculations, and Information Value (IV) analysis.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CreditRiskPreprocessor:
    """
    Class for preprocessing data for credit risk modeling.
    """
    
    def __init__(self, target_variable: str = 'default_flag', 
                 bad_value: Union[int, str] = 1, good_value: Union[int, str] = 0,
                 random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            target_variable: Name of the target variable column
            bad_value: Value that represents a "bad" outcome in the target variable
            good_value: Value that represents a "good" outcome in the target variable
            random_state: Random seed for reproducibility
        """
        self.target_variable = target_variable
        self.bad_value = bad_value
        self.good_value = good_value
        self.random_state = random_state
        self.binning_rules = {}
        self.woe_mappings = {}
        self.iv_values = {}
        self.numerical_features = []
        self.categorical_features = []
        self.binary_features = []
        self.excluded_features = []
        
    def split_data(self, df: pd.DataFrame, test_size: float = 0.3, 
                  stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data to use for testing
            stratify: Whether to stratify the split based on the target variable
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if stratify and self.target_variable in df.columns:
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=self.random_state,
                stratify=df[self.target_variable]
            )
        else:
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=self.random_state
            )
            
        logger.info(f"Data split into training ({len(train_df)} rows) and testing ({len(test_df)} rows) sets")
        return train_df, test_df
    
    def identify_feature_types(self, df: pd.DataFrame, max_categories: int = 30, 
                              exclude_columns: List[str] = None) -> None:
        """
        Identify numerical, categorical, and binary features in the DataFrame.
        
        Args:
            df: Input DataFrame
            max_categories: Maximum number of unique values for a feature to be considered categorical
            exclude_columns: List of columns to exclude from processing
        """
        exclude_columns = exclude_columns or []
        exclude_columns.append(self.target_variable)
        
        self.excluded_features = exclude_columns
        
        for col in df.columns:
            if col in exclude_columns:
                continue
                
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_vals = df[col].nunique()
                
                if unique_vals <= 2:
                    self.binary_features.append(col)
                elif unique_vals <= max_categories:
                    self.categorical_features.append(col)
                else:
                    self.numerical_features.append(col)
            else:
                # Non-numeric columns are treated as categorical
                self.categorical_features.append(col)
                
        logger.info(f"Identified {len(self.numerical_features)} numerical features, "
                   f"{len(self.categorical_features)} categorical features, and "
                   f"{len(self.binary_features)} binary features")
    
    def clean_data(self, df: pd.DataFrame, handle_missing: bool = True,
                 handle_outliers: bool = True, remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Clean the data by handling missing values, outliers, and duplicates.
        
        Args:
            df: Input DataFrame
            handle_missing: Whether to handle missing values
            handle_outliers: Whether to handle outliers
            remove_duplicates: Whether to remove duplicate rows
            
        Returns:
            Cleaned DataFrame
        """
        result_df = df.copy()
        
        # Remove duplicates
        if remove_duplicates:
            initial_rows = len(result_df)
            result_df = result_df.drop_duplicates()
            logger.info(f"Removed {initial_rows - len(result_df)} duplicate rows")
        
        # Handle missing values
        if handle_missing:
            for col in self.numerical_features:
                if col in result_df.columns and result_df[col].isnull().sum() > 0:
                    median_value = result_df[col].median()
                    result_df[col] = result_df[col].fillna(median_value)
                    logger.info(f"Filled missing values in {col} with median ({median_value})")
                    
            for col in self.categorical_features + self.binary_features:
                if col in result_df.columns and result_df[col].isnull().sum() > 0:
                    mode_value = result_df[col].mode()[0]
                    result_df[col] = result_df[col].fillna(mode_value)
                    logger.info(f"Filled missing values in {col} with mode ({mode_value})")
        
        # Handle outliers using capping (winsorization)
        if handle_outliers:
            for col in self.numerical_features:
                if col in result_df.columns:
                    q1 = result_df[col].quantile(0.01)
                    q3 = result_df[col].quantile(0.99)
                    result_df[col] = result_df[col].clip(lower=q1, upper=q3)
                    logger.info(f"Capped outliers in {col} at 1% and 99% quantiles")
                    
        return result_df
    
    def bin_feature(self, df: pd.DataFrame, feature: str, 
                   bins: Union[int, List[float]] = 10, 
                   strategy: str = 'quantile') -> pd.Series:
        """
        Bin a numerical feature into discrete groups.
        
        Args:
            df: Input DataFrame
            feature: Feature to bin
            bins: Number of bins or list of bin edges
            strategy: Binning strategy ('quantile', 'uniform', or 'kmeans')
            
        Returns:
            Series with binned values
        """
        if feature not in df.columns:
            raise ValueError(f"Feature {feature} not found in DataFrame")
            
        if isinstance(bins, int):
            if strategy == 'quantile':
                bin_edges = [df[feature].quantile(q) for q in np.linspace(0, 1, bins+1)]
                # Make sure bin edges are unique
                bin_edges = sorted(list(set(bin_edges)))
            elif strategy == 'uniform':
                bin_edges = np.linspace(df[feature].min(), df[feature].max(), bins+1)
            else:
                raise ValueError(f"Unsupported binning strategy: {strategy}")
        else:
            bin_edges = bins
            
        # Store binning rules for later use
        self.binning_rules[feature] = bin_edges
        
        # Create labels for bins
        labels = [f"{feature}_{i+1}" for i in range(len(bin_edges)-1)]
        
        # Bin the data
        binned = pd.cut(df[feature], bins=bin_edges, labels=labels, include_lowest=True)
        
        return binned
    
    def calculate_woe_iv(self, df: pd.DataFrame, feature: str, target: str = None) -> Tuple[pd.DataFrame, float]:
        """
        Calculate Weight of Evidence (WoE) and Information Value (IV) for a feature.
        
        Args:
            df: Input DataFrame
            feature: Feature to calculate WoE and IV for
            target: Target variable column name (defaults to self.target_variable)
            
        Returns:
            Tuple of (WoE DataFrame, IV value)
        """
        target = target or self.target_variable
        
        if target not in df.columns:
            raise ValueError(f"Target variable {target} not found in DataFrame")
            
        if feature not in df.columns:
            raise ValueError(f"Feature {feature} not found in DataFrame")
            
        # Create a crosstab of feature vs target
        cross_tab = pd.crosstab(df[feature], df[target])
        
        # Ensure the target column names match expected good/bad values
        if self.good_value not in cross_tab.columns or self.bad_value not in cross_tab.columns:
            if len(cross_tab.columns) == 2:
                cross_tab.columns = [self.good_value, self.bad_value]
            else:
                raise ValueError(f"Target variable {target} does not have exactly two values")
        
        # Calculate statistics
        woe_df = pd.DataFrame({
            'good': cross_tab[self.good_value],
            'bad': cross_tab[self.bad_value]
        })
        
        woe_df['total'] = woe_df['good'] + woe_df['bad']
        woe_df['good_dist'] = woe_df['good'] / woe_df['good'].sum()
        woe_df['bad_dist'] = woe_df['bad'] / woe_df['bad'].sum()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        woe_df['woe'] = np.log((woe_df['good_dist'] + epsilon) / (woe_df['bad_dist'] + epsilon))
        woe_df['iv_component'] = (woe_df['good_dist'] - woe_df['bad_dist']) * woe_df['woe']
        
        # Calculate Information Value
        iv = woe_df['iv_component'].sum()
        
        # Store WoE mapping and IV for this feature
        self.woe_mappings[feature] = dict(zip(woe_df.index, woe_df['woe']))
        self.iv_values[feature] = iv
        
        return woe_df, iv
    
    def apply_woe_transformation(self, df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
        """
        Apply Weight of Evidence transformation to selected features.
        
        Args:
            df: Input DataFrame
            features: List of features to transform (defaults to all features with WoE mappings)
            
        Returns:
            DataFrame with WoE-transformed features
        """
        result_df = df.copy()
        
        features = features or list(self.woe_mappings.keys())
        
        for feature in features:
            if feature in self.woe_mappings:
                # Check if the feature is categorical
                is_categorical = pd.api.types.is_categorical_dtype(df[feature])
                
                # If categorical, convert to string first to avoid category errors
                if is_categorical:
                    temp_feature = df[feature].astype(str)
                else:
                    temp_feature = df[feature]
                
                # Create a new column with WoE values
                new_col_name = f"{feature}_woe"
                result_df[new_col_name] = temp_feature.map(self.woe_mappings[feature])
                
                # Handle values not in the mapping (use mean WoE or 0)
                if result_df[new_col_name].isnull().any():
                    if len(self.woe_mappings[feature]) > 0:
                        default_woe = np.mean(list(self.woe_mappings[feature].values()))
                    else:
                        default_woe = 0
                    result_df[new_col_name] = result_df[new_col_name].fillna(default_woe)
            else:
                logger.warning(f"No WoE mapping found for feature {feature}")
                
        return result_df
    
    def create_scorecard(self, df: pd.DataFrame, features: List[str], 
                        coefficients: Dict[str, float], 
                        intercept: float, 
                        scaling_factor: float = 20,
                        offset: float = 600,
                        pdo: float = 40) -> pd.DataFrame:
        """
        Create a scorecard based on WoE transformation and logistic regression coefficients.
        
        Args:
            df: Input DataFrame with WoE-transformed features
            features: List of WoE-transformed features to include
            coefficients: Dictionary mapping feature names to their coefficients
            intercept: Intercept term from the logistic regression model
            scaling_factor: Scaling factor for the score
            offset: Offset to add to the base score
            pdo: Points to double odds
            
        Returns:
            DataFrame containing the scorecard
        """
        scorecard_df = pd.DataFrame(columns=['Characteristic', 'Attribute', 'WoE', 'Coefficient', 'Points'])
        
        # Factor and offset terms
        factor = scaling_factor / np.log(2) * pdo
        
        # Calculate base points from intercept
        base_points = offset - factor * intercept
        
        # Add a row for the base points
        scorecard_df = pd.concat([scorecard_df, pd.DataFrame({
            'Characteristic': ['Base Points'],
            'Attribute': [''],
            'WoE': [np.nan],
            'Coefficient': [intercept],
            'Points': [base_points]
        })], ignore_index=True)
        
        # Add points for each attribute
        for feature in features:
            if feature in self.woe_mappings and feature in coefficients:
                coef = coefficients[feature]
                
                for attribute, woe in self.woe_mappings[feature].items():
                    points = -factor * coef * woe
                    
                    scorecard_df = pd.concat([scorecard_df, pd.DataFrame({
                        'Characteristic': [feature],
                        'Attribute': [attribute],
                        'WoE': [woe],
                        'Coefficient': [coef],
                        'Points': [points]
                    })], ignore_index=True)
        
        return scorecard_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer new features for credit risk modeling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        result_df = df.copy()
        
        # Example feature engineering for credit risk modeling
        
        # 1. Calculate debt-to-income ratio if relevant columns exist
        if 'monthly_debt_payments' in result_df.columns and 'monthly_income' in result_df.columns:
            result_df['debt_to_income_ratio'] = (
                result_df['monthly_debt_payments'] / result_df['monthly_income'].replace(0, np.nan)
            ).fillna(0)
            
        # 2. Calculate utilization ratio if relevant columns exist
        if 'total_balance' in result_df.columns and 'total_credit_limit' in result_df.columns:
            result_df['utilization_ratio'] = (
                result_df['total_balance'] / result_df['total_credit_limit'].replace(0, np.nan)
            ).fillna(0)
            
        # 3. Create age groups
        if 'age' in result_df.columns:
            result_df['age_group'] = pd.cut(
                result_df['age'],
                bins=[0, 25, 35, 45, 55, 65, float('inf')],
                labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+']
            )
            
        # 4. Calculate time since last delinquency
        if 'days_since_last_delinquency' in result_df.columns:
            result_df['recent_delinquency'] = np.where(
                result_df['days_since_last_delinquency'] <= 90, 1, 0
            )
            
        # 5. Interaction terms between important features
        if 'credit_score' in result_df.columns and 'utilization_ratio' in result_df.columns:
            result_df['score_util_interaction'] = result_df['credit_score'] * (1 - result_df['utilization_ratio'])
            
        # 6. Polynomial features for continuous variables
        for feature in self.numerical_features:
            if feature in result_df.columns:
                result_df[f'{feature}_squared'] = result_df[feature] ** 2
                
        # 7. Calculate payment ratio if payment data is available
        if 'actual_payment' in result_df.columns and 'scheduled_payment' in result_df.columns:
            result_df['payment_ratio'] = (
                result_df['actual_payment'] / result_df['scheduled_payment'].replace(0, np.nan)
            ).fillna(0)
            
        logger.info(f"Engineered {len(result_df.columns) - len(df.columns)} new features")
        return result_df
    
    def plot_woe_analysis(self, feature: str, save_path: Optional[str] = None) -> None:
        """
        Plot the Weight of Evidence analysis for a feature.
        
        Args:
            feature: Feature to plot
            save_path: Path to save the plot to (if provided)
        """
        if feature not in self.woe_mappings or feature not in self.iv_values:
            raise ValueError(f"No WoE analysis found for feature {feature}")
            
        # Create a DataFrame from the WoE mapping
        woe_df = pd.DataFrame({
            'Attribute': list(self.woe_mappings[feature].keys()),
            'WoE': list(self.woe_mappings[feature].values())
        })
        
        # Sort by attribute if numeric, otherwise keep original order
        try:
            woe_df['Attribute'] = woe_df['Attribute'].astype(float)
            woe_df = woe_df.sort_values('Attribute')
        except:
            pass
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Bar plot of WoE values
        ax = sns.barplot(data=woe_df, x='Attribute', y='WoE')
        
        # Add IV value as title
        plt.title(f"Weight of Evidence Analysis for {feature}\nInformation Value: {self.iv_values[feature]:.4f}")
        
        # Rotate x-axis labels if there are many attributes
        if len(woe_df) > 5:
            plt.xticks(rotation=45, ha='right')
            
        # Add zero line
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved WoE plot to {save_path}")
            
        plt.show()
        
    def process_data(self, df: pd.DataFrame, target_variable: str = None, 
                    test_size: float = 0.3) -> Dict[str, pd.DataFrame]:
        """
        Full data preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_variable: Target variable column name (defaults to self.target_variable)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing processed DataFrames (train, test, train_woe, test_woe)
        """
        if target_variable:
            self.target_variable = target_variable
            
        # Identify feature types
        self.identify_feature_types(df)
        
        # Clean the data
        cleaned_df = self.clean_data(df)
        
        # Engineer new features
        engineered_df = self.engineer_features(cleaned_df)
        
        # Split the data
        train_df, test_df = self.split_data(engineered_df, test_size=test_size)
        
        # Bin numerical features
        for feature in self.numerical_features:
            if feature in train_df.columns:
                train_df[f"{feature}_binned"] = self.bin_feature(train_df, feature)
                # Apply same binning to test set
                if feature in self.binning_rules:
                    edges = self.binning_rules[feature]
                    labels = [f"{feature}_{i+1}" for i in range(len(edges)-1)]
                    test_df[f"{feature}_binned"] = pd.cut(
                        test_df[feature], 
                        bins=edges, 
                        labels=labels, 
                        include_lowest=True
                    )
        
        # Calculate WoE and IV for binned features and categorical features
        for feature in [f for f in train_df.columns if '_binned' in f] + self.categorical_features:
            if feature in train_df.columns:
                self.calculate_woe_iv(train_df, feature)
                
        # Apply WoE transformation
        train_woe_df = self.apply_woe_transformation(train_df)
        test_woe_df = self.apply_woe_transformation(test_df)
        
        # Return dictionary of processed DataFrames
        return {
            'train': train_df,
            'test': test_df,
            'train_woe': train_woe_df,
            'test_woe': test_woe_df
        }


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    import sqlite3
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.database import get_db_connection
    
    # Check if required tables exist before trying to query them
    try:
        conn = get_db_connection()
        
        # Check if tables exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('loan_application', 'customer_profile')")
        tables = [row[0] for row in cursor.fetchall()]
        
        if 'loan_application' in tables and 'customer_profile' in tables:
            # Tables exist, proceed with database query
            query = """
            SELECT 
                a.*, 
                c.annual_income, c.employment_status, c.age,
                CASE WHEN a.status = 'Rejected' THEN 1 ELSE 0 END as rejection_flag
            FROM 
                loan_application a
            JOIN 
                customer_profile c ON a.customer_id = c.customer_id
            """
            df = pd.read_sql(query, conn)
            logger.info(f"Loaded {len(df)} rows from database")
        else:
            # Tables don't exist, use sample data instead
            logger.warning("Required database tables not found. Using sample data instead.")
            
            # Create synthetic sample data
            np.random.seed(42)
            n_samples = 100
            
            # Generate sample customer features
            df = pd.DataFrame({
                'customer_id': [f'CUST{i:05d}' for i in range(n_samples)],
                'age': np.random.randint(18, 70, n_samples),
                'annual_income': np.random.normal(60000, 20000, n_samples),
                'employment_status': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples),
                'credit_score': np.random.randint(500, 850, n_samples),
                'loan_amount': np.random.normal(15000, 5000, n_samples),
                'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
                'debt_to_income': np.random.normal(0.3, 0.1, n_samples),
                'has_mortgage': np.random.choice([0, 1], n_samples),
                'rejection_flag': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
            })
            
            logger.info(f"Created synthetic sample data with {len(df)} rows")
    
    except Exception as e:
        # Handle any other database errors
        logger.error(f"Database error: {e}")
        
        # Create synthetic sample data as fallback
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'customer_id': [f'CUST{i:05d}' for i in range(n_samples)],
            'age': np.random.randint(18, 70, n_samples),
            'annual_income': np.random.normal(60000, 20000, n_samples),
            'employment_status': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples),
            'credit_score': np.random.randint(500, 850, n_samples),
            'loan_amount': np.random.normal(15000, 5000, n_samples),
            'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
            'debt_to_income': np.random.normal(0.3, 0.1, n_samples),
            'has_mortgage': np.random.choice([0, 1], n_samples),
            'rejection_flag': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })
        
        logger.info(f"Created synthetic sample data with {len(df)} rows")
    
    # Create preprocessor and process data
    preprocessor = CreditRiskPreprocessor(target_variable='rejection_flag')
    processed_data = preprocessor.process_data(df)
    
    # Print information about the processed data
    for key, value in processed_data.items():
        print(f"{key} shape: {value.shape}")
        
    # Print Information Value for each feature
    print("\nInformation Values:")
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
            
        print(f"{feature}: {iv:.4f} ({iv_strength})")
        
    # Plot WoE analysis for top features
    top_features = sorted(preprocessor.iv_values.items(), key=lambda x: x[1], reverse=True)[:3]
    for feature, _ in top_features:
        preprocessor.plot_woe_analysis(feature) 