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
import urllib.parse
import time

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
    parser.add_argument('--db-type', type=str, choices=['mysql', 'postgresql', 'sqlite'], 
                       default=os.getenv('DB_TYPE', 'mysql'),
                       help='Database type (mysql, postgresql, or sqlite)')
    parser.add_argument('--host', type=str, default=os.getenv('DB_HOST', 'localhost'),
                       help='Database host (not used for sqlite)')
    parser.add_argument('--port', type=int, 
                       default=int(os.getenv('DB_PORT', 3306 if os.getenv('DB_TYPE', 'mysql') == 'mysql' else 5432)),
                       help='Database port (not used for sqlite)')
    parser.add_argument('--user', type=str, default=os.getenv('DB_USER', 'root'),
                       help='Database user (not used for sqlite)')
    parser.add_argument('--password', type=str, default=os.getenv('DB_PASSWORD', ''),
                       help='Database password (not used for sqlite)')
    parser.add_argument('--db-name', type=str, default=os.getenv('DB_NAME', 'credit_risk'),
                       help='Database name')
    parser.add_argument('--create-db', action='store_true', 
                       help='Create database if it does not exist')
    parser.add_argument('--populate-sample', action='store_true',
                       help='Populate with sample data')
    return parser.parse_args()

def get_db_url(db_type, user, password, host, port, db_name=None):
    """Create database URL based on connection parameters."""
    # URL encode username and password to handle special characters
    encoded_user = urllib.parse.quote_plus(user)
    encoded_password = urllib.parse.quote_plus(password) if password else ""
    
    if db_type == 'mysql':
        if db_name:
            return f"mysql+mysqlconnector://{encoded_user}:{encoded_password}@{host}:{port}/{db_name}"
        else:
            return f"mysql+mysqlconnector://{encoded_user}:{encoded_password}@{host}:{port}"
    elif db_type == 'postgresql':
        if db_name:
            return f"postgresql+psycopg2://{encoded_user}:{encoded_password}@{host}:{port}/{db_name}"
        else:
            return f"postgresql+psycopg2://{encoded_user}:{encoded_password}@{host}:{port}/postgres"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def create_database(db_type, engine, db_name):
    """Create database if it doesn't exist."""
    try:
        logger.info(f"Creating database {db_name} if it doesn't exist...")
        if db_type == 'mysql':
            with engine.connect() as conn:
                try:
                    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
                    conn.commit()
                except Exception as e:
                    logger.error(f"Error executing MySQL create database: {e}")
                    raise
        elif db_type == 'postgresql':
            # Check if the database exists
            with engine.connect() as conn:
                try:
                    result = conn.execute(text(
                        "SELECT 1 FROM pg_database WHERE datname = :db_name"),
                        {"db_name": db_name}
                    )
                    exists = result.scalar() is not None
                    
                    if not exists:
                        # PostgreSQL needs to commit the transaction before creating a database
                        conn.execute(text("COMMIT"))
                        conn.execute(text(f"CREATE DATABASE {db_name}"))
                        conn.commit()
                except Exception as e:
                    logger.error(f"Error executing PostgreSQL query: {e}")
                    raise
        logger.info(f"Database {db_name} is ready.")
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

def create_schema(engine):
    """Create database schema from SQL file."""
    schema_file = PROJECT_ROOT / "database" / "schema.sql"
    try:
        logger.info("Creating database schema...")
        with open(schema_file, 'r', encoding='utf-8', errors='replace') as f:
            sql_script = f.read()
            # Split the script into individual statements and execute each one
            statements = sql_script.split(';')
            with engine.connect() as conn:
                for statement in statements:
                    if statement.strip():
                        try:
                            conn.execute(text(statement))
                        except Exception as stmt_err:
                            logger.error(f"Error executing statement: {stmt_err}")
                            logger.error(f"Statement: {statement}")
                conn.commit()
        logger.info("Schema created successfully.")
    except Exception as e:
        logger.error(f"Error creating schema: {e}")
        sys.exit(1)

def create_database_directly(db_type, host, port, user, password, db_name):
    """Create database directly using the appropriate database connector."""
    try:
        logger.info(f"Creating database {db_name} using direct connection...")
        
        if db_type == 'mysql':
            import mysql.connector
            
            # Connect to MySQL server
            conn = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password
            )
            cursor = conn.cursor()
            
            # Create database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
            conn.commit()
            
            # Close connection
            cursor.close()
            conn.close()
            
        elif db_type == 'postgresql':
            import psycopg2
            
            # Connect to PostgreSQL server
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                dbname="postgres"  # Connect to default database
            )
            conn.autocommit = True  # Set autocommit to True for creating database
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            exists = cursor.fetchone() is not None
            
            # Create database if it doesn't exist
            if not exists:
                cursor.execute(f"CREATE DATABASE {db_name}")
            
            # Close connection
            cursor.close()
            conn.close()
            
        logger.info(f"Database {db_name} created successfully using direct connection.")
        return True
    except Exception as e:
        logger.error(f"Error creating database directly: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def setup_sqlite_database(db_name):
    """Set up a SQLite database as a fallback option."""
    try:
        import sqlite3
        from pathlib import Path
        
        # Create databases directory if it doesn't exist
        db_dir = PROJECT_ROOT / "database" / "sqlite"
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # SQLite database file path
        db_path = db_dir / f"{db_name}.db"
        
        logger.info(f"Setting up SQLite database at {db_path}")
        
        # Connect to SQLite database (creates it if it doesn't exist)
        conn = sqlite3.connect(str(db_path))
        
        # Read schema file and modify for SQLite compatibility
        schema_file = PROJECT_ROOT / "database" / "schema.sql"
        with open(schema_file, 'r', encoding='utf-8', errors='replace') as f:
            sql_script = f.read()
        
        # Replace MySQL/PostgreSQL specific syntax with SQLite compatible syntax
        sql_script = sql_script.replace('AUTO_INCREMENT', 'AUTOINCREMENT')
        sql_script = sql_script.replace('DECIMAL', 'REAL')
        sql_script = sql_script.replace('DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', '')
        
        # Execute schema statements
        cursor = conn.cursor()
        # SQLite doesn't support multiple statements in one execute, so split and execute individually
        statements = [stmt for stmt in sql_script.split(';') if stmt.strip()]
        
        for statement in statements:
            try:
                # Skip MySQL/PostgreSQL specific statements
                if ('CREATE OR REPLACE VIEW' in statement or 
                    'INDEX' in statement or 
                    'FOREIGN KEY' in statement):
                    continue
                cursor.execute(statement)
            except sqlite3.Error as e:
                logger.warning(f"Error executing SQLite statement: {e}")
                logger.warning(f"Statement: {statement[:100]}...")
        
        conn.commit()
        conn.close()
        
        logger.info("SQLite database setup completed successfully")
        return str(db_path)
    
    except Exception as e:
        logger.error(f"Error setting up SQLite database: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    """Main function to setup the database."""
    args = parse_arguments()
    
    try:
        # Handle SQLite explicitly if selected
        if args.db_type == 'sqlite':
            logger.info("Setting up SQLite database...")
            sqlite_path = setup_sqlite_database(args.db_name)
            
            if sqlite_path:
                # Update .env file with SQLite connection info
                env_file = PROJECT_ROOT / "config" / ".env"
                if env_file.exists():
                    try:
                        with open(env_file, 'r', encoding='utf-8', errors='replace') as f:
                            env_content = f.read()
                        
                        # Update DB_TYPE to sqlite
                        env_content = env_content.replace('DB_TYPE=mysql', 'DB_TYPE=sqlite')
                        env_content = env_content.replace('DB_TYPE=postgresql', 'DB_TYPE=sqlite')
                        
                        # Add SQLite path
                        if 'DB_PATH=' not in env_content:
                            env_content += f"\nDB_PATH={sqlite_path}\n"
                        
                        with open(env_file, 'w', encoding='utf-8') as f:
                            f.write(env_content)
                        
                        logger.info(f"Updated .env file with SQLite configuration")
                    except Exception as e:
                        logger.warning(f"Could not update .env file: {e}")
                
                logger.info("SQLite database setup completed successfully")
                return
            else:
                logger.error("Failed to set up SQLite database")
                sys.exit(1)
        
        # Handle MySQL or PostgreSQL
        success = False
        
        # Connect to the server without specifying a database
        if args.create_db:
            logger.info(f"Connecting to {args.db_type} server at {args.host}:{args.port}...")
            
            # Try direct database creation first
            if create_database_directly(args.db_type, args.host, args.port, args.user, args.password, args.db_name):
                logger.info("Database created successfully using direct connection.")
                success = True
            else:
                # Fall back to SQLAlchemy method
                try:
                    logger.info("Falling back to SQLAlchemy for database creation...")
                    engine_url = get_db_url(args.db_type, args.user, args.password, args.host, args.port)
                    logger.info(f"Using connection URL (without password): {engine_url.replace(args.password if args.password else '', '****')}")
                    engine = create_engine(engine_url)
                    create_database(args.db_type, engine, args.db_name)
                    success = True
                except Exception as e:
                    logger.error(f"SQLAlchemy database creation failed: {e}")
            
            if success:
                # Add a small delay to ensure the database is fully created
                import time
                logger.info("Waiting for database to be ready...")
                time.sleep(2)
        
        # Connect to the specific database and create schema
        if success or not args.create_db:
            logger.info(f"Connecting to database {args.db_name}...")
            db_url = get_db_url(args.db_type, args.user, args.password, args.host, args.port, args.db_name)
            logger.info(f"Using database URL (without password): {db_url.replace(args.password if args.password else '', '****')}")
            try:
                engine = create_engine(db_url)
                # Test connection
                with engine.connect() as conn:
                    pass
                logger.info("Database connection successful.")
                
                # Create schema
                create_schema(engine)
                
                # Populate with sample data if requested
                if args.populate_sample:
                    logger.info("Sample data population requested but not implemented yet.")
                
                logger.info("Database setup completed successfully.")
                return
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
        
        # If we get here, both MySQL/PostgreSQL attempts failed
        logger.warning("Falling back to SQLite database...")
        sqlite_path = setup_sqlite_database(args.db_name)
        
        if sqlite_path:
            # Update .env file with SQLite connection info
            env_file = PROJECT_ROOT / "config" / ".env"
            if env_file.exists():
                try:
                    with open(env_file, 'r', encoding='utf-8', errors='replace') as f:
                        env_content = f.read()
                    
                    # Update DB_TYPE to sqlite
                    env_content = env_content.replace('DB_TYPE=mysql', 'DB_TYPE=sqlite')
                    env_content = env_content.replace('DB_TYPE=postgresql', 'DB_TYPE=sqlite')
                    
                    # Add SQLite path
                    if 'DB_PATH=' not in env_content:
                        env_content += f"\nDB_PATH={sqlite_path}\n"
                    
                    with open(env_file, 'w', encoding='utf-8') as f:
                        f.write(env_content)
                    
                    logger.info(f"Updated .env file with SQLite configuration")
                except Exception as e:
                    logger.warning(f"Could not update .env file: {e}")
            
            logger.info("SQLite database setup completed successfully")
        else:
            logger.error("Failed to set up any database")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error in database setup: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 