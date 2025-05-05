#!/usr/bin/env python3
"""
Database setup script for Credit Risk Scorecard & Reporting Engine.
This script creates the necessary database and tables if they don't exist.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
import sqlalchemy
from sqlalchemy import create_engine, text

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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Set up the Credit Risk Database')
    parser.add_argument('--db-type', type=str, choices=['mysql', 'postgresql'], 
                       default=os.getenv('DB_TYPE', 'mysql'),
                       help='Database type (mysql or postgresql)')
    parser.add_argument('--host', type=str, default=os.getenv('DB_HOST', 'localhost'),
                       help='Database host')
    parser.add_argument('--port', type=int, 
                       default=int(os.getenv('DB_PORT', 3306 if os.getenv('DB_TYPE', 'mysql') == 'mysql' else 5432)),
                       help='Database port')
    parser.add_argument('--user', type=str, default=os.getenv('DB_USER', 'root'),
                       help='Database user')
    parser.add_argument('--password', type=str, default=os.getenv('DB_PASSWORD', ''),
                       help='Database password')
    parser.add_argument('--db-name', type=str, default=os.getenv('DB_NAME', 'credit_risk'),
                       help='Database name')
    parser.add_argument('--create-db', action='store_true', 
                       help='Create database if it does not exist')
    parser.add_argument('--populate-sample', action='store_true',
                       help='Populate with sample data')
    return parser.parse_args()

def get_db_url(db_type, user, password, host, port, db_name=None):
    """Create database URL based on connection parameters."""
    if db_type == 'mysql':
        if db_name:
            return f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}"
        else:
            return f"mysql+mysqlconnector://{user}:{password}@{host}:{port}"
    elif db_type == 'postgresql':
        if db_name:
            return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        else:
            return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/postgres"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def create_database(db_type, engine, db_name):
    """Create database if it doesn't exist."""
    try:
        logger.info(f"Creating database {db_name} if it doesn't exist...")
        if db_type == 'mysql':
            engine.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
        elif db_type == 'postgresql':
            # Check if the database exists
            result = engine.execute(text(
                "SELECT 1 FROM pg_database WHERE datname = :db_name"),
                {"db_name": db_name}
            )
            exists = result.scalar() is not None
            if not exists:
                # PostgreSQL needs to commit the transaction before creating a database
                engine.execute(text("COMMIT"))
                engine.execute(text(f"CREATE DATABASE {db_name}"))
        logger.info(f"Database {db_name} is ready.")
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        sys.exit(1)

def create_schema(engine):
    """Create database schema from SQL file."""
    schema_file = PROJECT_ROOT / "database" / "schema.sql"
    try:
        logger.info("Creating database schema...")
        with open(schema_file, 'r') as f:
            sql_script = f.read()
            # Split the script into individual statements and execute each one
            statements = sql_script.split(';')
            for statement in statements:
                if statement.strip():
                    engine.execute(text(statement))
        logger.info("Schema created successfully.")
    except Exception as e:
        logger.error(f"Error creating schema: {e}")
        sys.exit(1)

def main():
    """Main function to setup the database."""
    args = parse_arguments()
    
    # Connect to the server without specifying a database
    if args.create_db:
        logger.info(f"Connecting to {args.db_type} server at {args.host}:{args.port}...")
        engine_url = get_db_url(args.db_type, args.user, args.password, args.host, args.port)
        engine = create_engine(engine_url)
        create_database(args.db_type, engine, args.db_name)
    
    # Connect to the specific database and create schema
    logger.info(f"Connecting to database {args.db_name}...")
    db_url = get_db_url(args.db_type, args.user, args.password, args.host, args.port, args.db_name)
    try:
        engine = create_engine(db_url)
        # Test connection
        engine.connect()
        logger.info("Database connection successful.")
        
        # Create schema
        create_schema(engine)
        
        # Populate with sample data if requested
        if args.populate_sample:
            logger.info("Sample data population requested but not implemented yet.")
            # This would call a function to populate sample data
        
        logger.info("Database setup completed successfully.")
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 