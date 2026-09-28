import os
import logging
from typing import Optional
from dotenv import load_dotenv

import psycopg
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# PostgreSQL Connection URL
_raw_url = os.getenv("PG_DATABASE_URL") or os.getenv("DATABASE_URL")
if not _raw_url:
    logger.warning("No PostgreSQL database URL found in environment variables (PG_DATABASE_URL or DATABASE_URL).")
    PG_URL = None
else:
    # psycopg does not support 'schema' in the connection string directly
    # Strip any query parameters after '?' if present
    PG_URL = _raw_url.split("?")[0]


# Global connection pool
_pool: Optional[ConnectionPool] = None

def get_pool() -> ConnectionPool:
    """Get the global connection pool, initializing it if necessary."""
    global _pool
    if _pool is None:
        if not PG_URL:
            raise ValueError("PostgreSQL database URL is not configured.")
        
        # Initialize connection pool with sensible limits
        logger.info("Initializing PostgreSQL connection pool...")
        
        # Enable unnesting and general connection settings
        def configure_connection(conn):
            import pgvector.psycopg
            pgvector.psycopg.register_vector(conn)
            
        _pool = ConnectionPool(
            conninfo=PG_URL,
            min_size=1,
            max_size=10,
            timeout=30.0,
            configure=configure_connection
        )
    return _pool

def init_db():
    """Initialize the database schema for native PostgreSQL RAG."""
    if not PG_URL:
        return
        
    try:
        pool = get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Create the vector table using native DOUBLE PRECISION[]
                cur.execute("""
                CREATE TABLE IF NOT EXISTS vector_embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(document_id, chunk_index)
                );
                """)
                
                # Create index on source_collection (commonly queried metadata)
                # and document_id for faster filtering
                cur.execute("""
                CREATE INDEX IF NOT EXISTS vector_embeddings_metadata_source_idx 
                ON vector_embeddings USING btree ((metadata->>'source_collection'));
                """)
                cur.execute("""
                CREATE INDEX IF NOT EXISTS vector_embeddings_document_id_idx 
                ON vector_embeddings USING btree (document_id);
                """)


                # Create the source_documents table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS source_documents (
                    id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    created_at TIMESTAMPTZ,
                    document_data JSONB NOT NULL
                );
                """)
                
                # Create indexes for source_documents
                cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_documents_collection
                ON source_documents(collection_name);
                """)
                cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_documents_collection_created
                ON source_documents(collection_name, created_at);
                """)
                cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_documents_created
                ON source_documents(created_at);
                """)
                
            conn.commit()
            logger.info("PostgreSQL database initialized with native array schema and source documents table.")
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL database: {e}")
        raise

def check_postgres_connection() -> bool:
    """Health check for PostgreSQL database connection and pgvector extension."""
    if not PG_URL:
        logger.error("Health check failed: PG_DATABASE_URL is not set.")
        return False
        
    try:
        pool = get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # 1. Verify connection
                cur.execute("SELECT 1;")
                result = cur.fetchone()
                if not result or result[0] != 1:
                    logger.error("Health check failed: SELECT 1 returned unexpected result.")
                    return False
                

        logger.info("PostgreSQL health check passed. Connection acquired and native schema available.")
        return True
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        return False

def close_pool():
    """Close the connection pool cleanly."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed.")
