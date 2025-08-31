# api/database.py - Database Connection Management

import logging
import sys
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import text

# Add the parent directory (project root) to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == "api" else current_dir
sys.path.insert(0, str(project_root))

# Now use absolute imports
from api.config import settings
from api.models import Base

logger = logging.getLogger(__name__)


# ============================================================================
# DATABASE ENGINE SETUP
# ============================================================================

def create_database_engine():
    """Create and configure the database engine"""

    # Base engine arguments
    engine_args = {
        "connect_args": settings.database_connect_args
    }

    # Add connection pool settings for non-SQLite databases
    if not settings.is_sqlite:
        engine_args.update({
            "pool_size": settings.db_pool_size,
            "max_overflow": settings.db_max_overflow,
            "pool_timeout": 30,
            "pool_recycle": 3600,  # Recycle connections after 1 hour
        })
    else:
        # SQLite specific settings
        engine_args.update({
            "poolclass": StaticPool,
            "connect_args": {
                **settings.database_connect_args,
                "timeout": 20,  # SQLite connection timeout
            }
        })

    # Create engine
    engine = create_engine(settings.database_url, **engine_args)

    # Add event listeners for better connection management
    if settings.is_sqlite:
        # Enable foreign keys for SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            cursor.close()

    return engine


# Create global engine instance
engine = create_database_engine()

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Keep objects usable after commit
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions

    Usage:
        with get_db_session() as db:
            # Use db session
            pass
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_database():
    """Initialize database tables"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")
        return False


def check_database_connection() -> bool:
    """Check if database connection is working"""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def get_database_info() -> dict:
    """Get database information and statistics"""
    info = {
        "database_url": settings.database_url,
        "database_type": "sqlite" if settings.is_sqlite else "external",
        "is_sqlite": settings.is_sqlite,
        "pool_size": settings.db_pool_size if not settings.is_sqlite else "N/A",
        "max_overflow": settings.db_max_overflow if not settings.is_sqlite else "N/A",
    }

    # Add connection pool info for non-SQLite
    if not settings.is_sqlite:
        try:
            pool = engine.pool
            info.update({
                "pool_checked_in": pool.checkedin(),
                "pool_checked_out": pool.checkedout(),
                "pool_overflow": pool.overflow(),
                "pool_invalid": pool.invalid(),
            })
        except Exception as e:
            logger.warning(f"Could not get pool info: {e}")

    return info


# ============================================================================
# DATABASE HEALTH CHECK
# ============================================================================

def health_check() -> dict:
    """Comprehensive database health check"""
    health_info = {
        "database_connected": False,
        "tables_exist": False,
        "can_query": False,
        "error": None
    }

    try:
        # Test connection
        with engine.connect() as connection:
            health_info["database_connected"] = True

            # Test if tables exist
            inspector = None
            try:
                from sqlalchemy import inspect
                inspector = inspect(engine)
                tables = inspector.get_table_names()
                health_info["tables_exist"] = "chess_predictions" in tables
            except Exception as e:
                logger.warning(f"Could not inspect tables: {e}")

            # Test basic query
            result = connection.execute(text("SELECT 1"))
            if result.fetchone():
                health_info["can_query"] = True

    except Exception as e:
        health_info["error"] = str(e)
        logger.error(f"Database health check failed: {e}")

    return health_info


# ============================================================================
# DATABASE UTILITIES
# ============================================================================

def reset_database():
    """Reset database by dropping and recreating all tables"""
    try:
        logger.warning("🔥 Resetting database - dropping all tables")
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database reset completed")
        return True
    except Exception as e:
        logger.error(f"❌ Database reset failed: {e}")
        return False


def backup_database(backup_path: str = None):
    """Backup database (SQLite only)"""
    if not settings.is_sqlite:
        logger.warning("Database backup only supported for SQLite")
        return False

    try:
        import shutil
        from pathlib import Path

        # Extract database file path from URL
        db_file = settings.database_url.replace("sqlite:///", "")
        db_path = Path(db_file)

        if not db_path.exists():
            logger.error(f"Database file not found: {db_path}")
            return False

        # Create backup path
        if backup_path is None:
            backup_path = f"{db_file}.backup"

        # Copy database file
        shutil.copy2(db_path, backup_path)
        logger.info(f"✅ Database backed up to: {backup_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Database backup failed: {e}")
        return False


def get_database_size() -> dict:
    """Get database size information"""
    size_info = {
        "size_bytes": 0,
        "size_mb": 0,
        "table_count": 0,
        "error": None
    }

    try:
        if settings.is_sqlite:
            # Get SQLite file size
            import os
            from pathlib import Path

            db_file = settings.database_url.replace("sqlite:///", "")
            db_path = Path(db_file)

            if db_path.exists():
                size_bytes = os.path.getsize(db_path)
                size_info["size_bytes"] = size_bytes
                size_info["size_mb"] = round(size_bytes / (1024 * 1024), 2)

        # Count tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        size_info["table_count"] = len(inspector.get_table_names())

    except Exception as e:
        size_info["error"] = str(e)
        logger.error(f"Could not get database size: {e}")

    return size_info


# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize database on import
if __name__ != "__main__":
    init_database()

# Export commonly used items
__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_session",
    "init_database",
    "check_database_connection",
    "health_check",
    "get_database_info"
]