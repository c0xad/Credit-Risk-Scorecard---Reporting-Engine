#!/usr/bin/env python3
"""
Check database connectivity and verify tables exist.
This script is used to test the database setup.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.database import execute_query, get_db_connection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_database():
    """Check database connectivity and tables."""
    try:
        # Test connection
        engine = get_db_connection()
        logger.info("Database connection successful")
        
        # Get database type from connection
        db_type = str(engine.dialect.name)
        logger.info(f"Connected to {db_type} database")
        
        # List tables
        if db_type == 'sqlite':
            query = "SELECT name FROM sqlite_master WHERE type='table'"
        elif db_type == 'mysql':
            query = "SHOW TABLES"
        elif db_type == 'postgresql':
            query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        else:
            logger.error(f"Unsupported database type: {db_type}")
            return
        
        tables = execute_query(query)
        
        if tables.empty:
            logger.warning("No tables found in database")
        else:
            logger.info(f"Found {len(tables)} tables in database:")
            for table in tables.values.flatten():
                logger.info(f"  - {table}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    check_database() 