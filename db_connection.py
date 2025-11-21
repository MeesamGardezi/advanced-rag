"""
Simplified PostgreSQL connection using psycopg2 (compatible with ETL)
"""

import os
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """PostgreSQL connection manager using psycopg2"""
    
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._pool is None:
            self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            self._pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=20,
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                database=os.getenv("POSTGRES_DB", "construction_rag"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "")
            )
            logger.info("✅ PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, dict_cursor=True):
        """Get a cursor (with automatic commit/rollback)"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor if dict_cursor else None)
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction failed: {e}")
                raise
            finally:
                cursor.close()
    
    def execute(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute query and return results"""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if cursor.description:
                return cursor.fetchall()
            return []
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            logger.info("✅ Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False


# Global instance
_db_instance = None


def get_db():
    """Get database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
    return _db_instance


@contextmanager
def session_scope():
    """
    Provide a transactional scope (compatible with SQLAlchemy-style code)
    """
    db = get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction failed: {e}")
            raise
        finally:
            cursor.close()


def verify_job_exists(job_id: str) -> bool:
    """Check if a job exists"""
    db = get_db()
    with db.get_cursor() as cursor:
        cursor.execute("SELECT 1 FROM jobs WHERE id = %s", (job_id,))
        return cursor.fetchone() is not None


def get_job_data_summary(job_id: str) -> Dict[str, Any]:
    """Get summary of job data"""
    db = get_db()
    with db.get_cursor() as cursor:
        # Get job info
        cursor.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        job = cursor.fetchone()
        
        if not job:
            return {"error": f"Job {job_id} not found"}
        
        # Count estimates
        cursor.execute("SELECT COUNT(*) as count FROM estimates WHERE job_id = %s", (job_id,))
        estimates_count = cursor.fetchone()['count']
        
        # Count flooring estimates
        cursor.execute("SELECT COUNT(*) as count FROM flooring_estimates WHERE job_id = %s", (job_id,))
        flooring_count = cursor.fetchone()['count']
        
        # Count consumed items
        cursor.execute("SELECT COUNT(*) as count FROM consumed_items WHERE job_id = %s", (job_id,))
        consumed_count = cursor.fetchone()['count']
        
        return {
            "job_id": job['id'],
            "job_name": job['name'],
            "company_id": job['company_id'],
            "client_name": job['client_name'],
            "estimates_count": estimates_count,
            "flooring_estimates_count": flooring_count,
            "consumed_items_count": consumed_count
        }


def init_database():
    """Initialize database and create tables"""
    db = get_db()
    
    if not db.test_connection():
        raise ConnectionError("Failed to connect to PostgreSQL")
    
    # Create tables
    create_tables_sql = """
    -- Jobs table
    CREATE TABLE IF NOT EXISTS jobs (
        id VARCHAR(255) PRIMARY KEY,
        company_id VARCHAR(255) NOT NULL,
        name VARCHAR(500) NOT NULL,
        client_name VARCHAR(500),
        site_city VARCHAR(255),
        site_state VARCHAR(100),
        site_location VARCHAR(500),
        project_description TEXT,
        status VARCHAR(50) DEFAULT 'active',
        estimate_type VARCHAR(50) DEFAULT 'general',
        created_date TIMESTAMP,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        schedule_last_updated TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_job_company_id ON jobs(company_id);
    CREATE INDEX IF NOT EXISTS idx_job_name ON jobs(name);
    
    -- Estimates table
    CREATE TABLE IF NOT EXISTS estimates (
        id VARCHAR(255) PRIMARY KEY,
        job_id VARCHAR(255) NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        row_number INTEGER NOT NULL,
        area VARCHAR(500),
        task_scope VARCHAR(500),
        cost_code VARCHAR(255),
        row_type VARCHAR(50) DEFAULT 'estimate',
        description TEXT,
        notes_remarks TEXT,
        units VARCHAR(100),
        qty NUMERIC DEFAULT 0,
        rate NUMERIC DEFAULT 0,
        total NUMERIC NOT NULL DEFAULT 0,
        budgeted_rate NUMERIC DEFAULT 0,
        budgeted_total NUMERIC DEFAULT 0,
        variance NUMERIC DEFAULT 0,
        variance_pct NUMERIC DEFAULT 0,
        materials JSONB,
        has_materials BOOLEAN DEFAULT FALSE,
        material_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_estimate_job_id ON estimates(job_id);
    CREATE INDEX IF NOT EXISTS idx_estimate_cost_code ON estimates(cost_code);
    CREATE INDEX IF NOT EXISTS idx_estimate_area ON estimates(area);
    
    -- Flooring Estimates table
    CREATE TABLE IF NOT EXISTS flooring_estimates (
        id VARCHAR(255) PRIMARY KEY,
        job_id VARCHAR(255) NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        row_number INTEGER NOT NULL,
        floor_type_id VARCHAR(255),
        vendor VARCHAR(500),
        item_material_name VARCHAR(500),
        brand VARCHAR(255),
        unit VARCHAR(100),
        measured_qty NUMERIC DEFAULT 0,
        supplier_qty NUMERIC DEFAULT 0,
        waste_factor NUMERIC DEFAULT 0,
        qty_including_waste NUMERIC DEFAULT 0,
        unit_price NUMERIC DEFAULT 0,
        cost_price NUMERIC DEFAULT 0,
        tax_freight NUMERIC DEFAULT 0,
        total_cost NUMERIC NOT NULL DEFAULT 0,
        sale_price NUMERIC DEFAULT 0,
        profit NUMERIC DEFAULT 0,
        margin_pct NUMERIC DEFAULT 0,
        notes_remarks TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_flooring_job_id ON flooring_estimates(job_id);
    CREATE INDEX IF NOT EXISTS idx_flooring_vendor ON flooring_estimates(vendor);
    
    -- Consumed Items table
    CREATE TABLE IF NOT EXISTS consumed_items (
        id VARCHAR(255) PRIMARY KEY,
        job_id VARCHAR(255) NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        cost_code VARCHAR(255) NOT NULL,
        amount NUMERIC NOT NULL DEFAULT 0,
        category VARCHAR(100),
        last_updated TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_consumed_job_id ON consumed_items(job_id);
    CREATE INDEX IF NOT EXISTS idx_consumed_cost_code ON consumed_items(cost_code);
    CREATE INDEX IF NOT EXISTS idx_consumed_category ON consumed_items(category);
    """
    
    with db.get_cursor() as cursor:
        cursor.execute(create_tables_sql)
    
    logger.info("✅ Database tables created")
    return db


def close_database():
    """Close database connection pool"""
    global _db_instance
    
    if _db_instance and _db_instance._pool:
        try:
            _db_instance._pool.closeall()
            logger.info("✅ Database connection pool closed")
        except Exception as e:
            logger.error(f"❌ Error closing database pool: {e}")
    
    _db_instance = None


def health_check() -> Dict[str, Any]:
    """Perform database health check"""
    try:
        db = get_db()
        
        # Test connection
        with db.get_cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        
        # Get table counts
        counts = {}
        tables = ['jobs', 'estimates', 'flooring_estimates', 'consumed_items']
        
        with db.get_cursor() as cursor:
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    result = cursor.fetchone()
                    counts[table] = result['count'] if result else 0
                except Exception:
                    counts[table] = 0
        
        return {
            'status': 'healthy',
            'connection': 'active',
            'table_counts': counts
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


if __name__ == "__main__":
    print("🧪 Testing database connection...")
    init_database()
    print("✅ Database initialization successful")