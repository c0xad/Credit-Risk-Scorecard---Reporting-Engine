"""
Utility functions for database operations.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, Table, MetaData
from sqlalchemy.engine import Engine

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


def get_db_connection() -> Engine:
    """
    Create a database connection using environment variables.
    
    Returns:
        Engine: SQLAlchemy engine for database connection
    """
    db_type = os.getenv('DB_TYPE', 'mysql')
    
    if db_type == 'sqlite':
        # SQLite connection
        db_path = os.getenv('DB_PATH')
        if not db_path:
            # Use default path if not specified
            sqlite_dir = Path(PROJECT_ROOT) / "database" / "sqlite"
            sqlite_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(sqlite_dir / f"{os.getenv('DB_NAME', 'credit_risk')}.db")
        
        db_url = f"sqlite:///{db_path}"
        logger.info(f"Using SQLite database at: {db_path}")
        
        # Check if SQLite file exists, if not create an empty file
        if not os.path.exists(db_path):
            logger.warning(f"SQLite database file doesn't exist at {db_path}. Creating empty file.")
            Path(db_path).touch()
    elif db_type == 'mysql':
        # MySQL connection
        db_user = os.getenv('DB_USER', 'root')
        db_password = os.getenv('DB_PASSWORD', '')
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '3306')
        db_name = os.getenv('DB_NAME', 'credit_risk')
        
        # URL encode username and password to handle special characters
        import urllib.parse
        encoded_user = urllib.parse.quote_plus(db_user)
        encoded_password = urllib.parse.quote_plus(db_password) if db_password else ""
        
        db_url = f"mysql+mysqlconnector://{encoded_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    elif db_type == 'postgresql':
        # PostgreSQL connection
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', '')
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'credit_risk')
        
        # URL encode username and password to handle special characters
        import urllib.parse
        encoded_user = urllib.parse.quote_plus(db_user)
        encoded_password = urllib.parse.quote_plus(db_password) if db_password else ""
        
        db_url = f"postgresql+psycopg2://{encoded_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    return create_engine(db_url)


def check_database_exists() -> bool:
    """
    Check if the database exists and has the required tables.
    
    Returns:
        bool: True if database exists and has tables, False otherwise
    """
    try:
        # Try to connect to the database
        engine = get_db_connection()
        
        # Check if at least one important table exists
        with engine.connect() as conn:
            # Try to query the customer_profile table
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='customer_profile'"))
            tables = result.fetchall()
            
            return len(tables) > 0
    except Exception as e:
        logger.warning(f"Error checking database: {e}")
        return False


def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Execute a SQL query and return the results as a DataFrame.
    
    Args:
        query (str): SQL query to execute
        params (Dict[str, Any], optional): Parameters for the query
        
    Returns:
        pd.DataFrame: Results of the query or empty DataFrame if error
    """
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            
            # Convert result to DataFrame
            columns = result.keys()
            data = result.fetchall()
            return pd.DataFrame(data, columns=columns)
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        # Return empty DataFrame instead of raising exception
        return pd.DataFrame()


def execute_statement(statement: str, params: Optional[Dict[str, Any]] = None) -> None:
    """
    Execute a SQL statement that doesn't return results (INSERT, UPDATE, DELETE).
    
    Args:
        statement (str): SQL statement to execute
        params (Dict[str, Any], optional): Parameters for the statement
    """
    engine = get_db_connection()
    try:
        with engine.connect() as conn:
            if params:
                conn.execute(text(statement), params)
            else:
                conn.execute(text(statement))
            conn.commit()
    except Exception as e:
        logger.error(f"Error executing statement: {e}")
        raise


def batch_insert(table_name: str, data: List[Dict[str, Any]]) -> None:
    """
    Insert multiple rows into a table.
    
    Args:
        table_name (str): Name of the table
        data (List[Dict[str, Any]]): List of dictionaries, each containing row data
    """
    if not data:
        logger.warning("No data provided for batch insert")
        return
    
    engine = get_db_connection()
    try:
        with engine.connect() as conn:
            insert_stmt = text(f"INSERT INTO {table_name} ({', '.join(data[0].keys())}) "
                          f"VALUES ({', '.join([f':{key}' for key in data[0].keys()])})")
            
            for row in data:
                conn.execute(insert_stmt, row)
            
            conn.commit()
        
        logger.info(f"Successfully inserted {len(data)} rows into {table_name}")
    except Exception as e:
        logger.error(f"Error batch inserting data: {e}")
        raise


def get_table_schema(table_name: str) -> Dict[str, Any]:
    """
    Get the schema information for a table.
    
    Args:
        table_name (str): Name of the table
        
    Returns:
        Dict[str, Any]: Table schema information
    """
    engine = get_db_connection()
    meta = MetaData()
    
    try:
        meta.reflect(bind=engine, only=[table_name])
        table = meta.tables[table_name]
        
        schema_info = {
            'columns': {col.name: {
                'type': str(col.type),
                'nullable': col.nullable,
                'primary_key': col.primary_key,
                'default': col.default,
                'foreign_keys': [f'{fk.column.table.name}.{fk.column.name}' for fk in col.foreign_keys]
            } for col in table.columns},
            'primary_key': [col.name for col in table.primary_key.columns],
            'indexes': [idx.name for idx in table.indexes]
        }
        
        return schema_info
    except Exception as e:
        logger.error(f"Error getting schema for table {table_name}: {e}")
        raise


def get_record_by_id(table_name: str, id_column: str, id_value: Any) -> Optional[Dict[str, Any]]:
    """
    Get a record by its ID.
    
    Args:
        table_name (str): Name of the table
        id_column (str): Name of the ID column
        id_value (Any): Value of the ID
        
    Returns:
        Optional[Dict[str, Any]]: Record as a dictionary or None if not found
    """
    query = f"SELECT * FROM {table_name} WHERE {id_column} = :id_value"
    params = {'id_value': id_value}
    
    result = execute_query(query, params)
    if result.empty:
        return None
    
    return result.iloc[0].to_dict()


def save_df_to_table(df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> None:
    """
    Save a DataFrame to a database table.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        table_name (str): Name of the table
        if_exists (str): How to behave if the table exists ('fail', 'replace', 'append')
    """
    engine = get_db_connection()
    try:
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        logger.info(f"Successfully saved {len(df)} rows to {table_name}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to table: {e}")
        raise 