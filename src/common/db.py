"""
Database connection and session management.
Supports PostgreSQL (production) with SQLite fallback (development).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.common.logger import get_logger

logger = get_logger("common.db")

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def create_engine_from_config(config: dict) -> Engine:
    """Create a SQLAlchemy engine from project config.

    Tries the primary PostgreSQL URL first.  Falls back to SQLite if the
    connection fails and ``auto_fallback`` is enabled.

    Args:
        config: Full project config dict; reads ``database`` section.

    Returns:
        Connected :class:`sqlalchemy.engine.Engine`.
    """
    db_cfg = config.get("database", {})
    primary_url: str = db_cfg.get("url", "")
    fallback_url: str = db_cfg.get("fallback_url", "sqlite:///data/sss_dev.db")
    auto_fallback: bool = bool(db_cfg.get("auto_fallback", True))
    echo: bool = bool(db_cfg.get("echo_sql", False))

    pool_kwargs: dict = {}
    if primary_url and not primary_url.startswith("sqlite"):
        pool_kwargs = {
            "pool_size": int(db_cfg.get("pool_size", 5)),
            "max_overflow": int(db_cfg.get("max_overflow", 10)),
            "pool_recycle": int(db_cfg.get("pool_recycle", 3600)),
        }

    if primary_url:
        try:
            engine = _try_connect(primary_url, echo, pool_kwargs)
            logger.info("Connected to primary database: %s", _sanitise_url(primary_url))
            return engine
        except Exception as exc:
            if auto_fallback:
                logger.warning(
                    "Primary DB unavailable (%s) — falling back to SQLite.", exc
                )
            else:
                raise

    engine = _try_connect(fallback_url, echo, {})
    logger.info("Using SQLite fallback: %s", fallback_url)
    return engine


def _try_connect(url: str, echo: bool, pool_kwargs: dict) -> Engine:
    engine = create_engine(url, echo=echo, **pool_kwargs)
    # Eagerly test connectivity
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def _sanitise_url(url: str) -> str:
    """Strip password from URL for safe logging."""
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        if parsed.password:
            netloc = f"{parsed.username}:***@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return url


def init_db(engine: Engine) -> None:
    """Create all tables defined in :mod:`src.common.db_models`.

    Args:
        engine: Connected SQLAlchemy engine.
    """
    from src.common.db_models import Base  # avoid circular import at module level
    Base.metadata.create_all(engine)
    logger.info("Database tables initialised.")


def get_session_factory(engine: Engine) -> sessionmaker:
    """Return a :class:`~sqlalchemy.orm.sessionmaker` bound to *engine*.

    Args:
        engine: Connected SQLAlchemy engine.

    Returns:
        Session factory callable.
    """
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session(factory: sessionmaker) -> Generator[Session, None, None]:
    """Context manager that yields a database session and commits/rolls back.

    Args:
        factory: Session factory from :func:`get_session_factory`.

    Yields:
        Active :class:`~sqlalchemy.orm.Session`.
    """
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
