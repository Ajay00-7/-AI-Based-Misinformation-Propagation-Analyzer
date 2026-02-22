import os
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages connections to the production data layers.
    - Redis: For high-speed caching and rate-limiting.
    - PostgreSQL: For tenant, user, organization, and billing data.
    """
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.pg_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/misinfo_db")
        
        self.redis_client = None
        self.pg_pool = None
        
    def connect_redis(self):
        """Establish connection to Redis Cache Server"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            # Ping to verify
            self.redis_client.ping()
            logger.info("Successfully connected to Redis Cache.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
            
    def get_redis(self):
        """Dependency provider for FastAPI routes"""
        if not self.redis_client:
            self.connect_redis()
        return self.redis_client

    def connect_postgres(self):
        """Establish connection to PostgreSQL Billing Database"""
        try:
            self.pg_pool = psycopg2.connect(self.pg_url)
            logger.info("Successfully connected to PostgreSQL Database.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
            
    def get_db_cursor(self):
        """Yields a database cursor for query execution"""
        if not self.pg_pool or self.pg_pool.closed:
            self.connect_postgres()
            
        cursor = self.pg_pool.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            self.pg_pool.commit()
        except Exception as e:
            self.pg_pool.rollback()
            raise e
        finally:
            cursor.close()

# Initialize a global instance for injection
db = DatabaseManager()
