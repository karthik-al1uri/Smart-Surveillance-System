"""
FastAPI dependency injection for database sessions and repositories.
"""

from __future__ import annotations

from typing import Generator

from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker

from src.api.repositories import (
    AlertRepository,
    CameraRepository,
    EventRepository,
    UserRepository,
)
from src.common.config import load_config
from src.common.model_manager import ModelManager
from src.common.model_registry import ModelRegistry

_session_factory: sessionmaker | None = None


def set_session_factory(factory: sessionmaker) -> None:
    """Configure the global session factory used by all dependencies.

    Args:
        factory: Bound :class:`~sqlalchemy.orm.sessionmaker`.
    """
    global _session_factory
    _session_factory = factory


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and close it afterwards."""
    if _session_factory is None:
        raise RuntimeError("Session factory not initialised. Call set_session_factory() first.")
    session: Session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_event_repo(db: Session = Depends(get_db)) -> EventRepository:
    """Dependency: return an EventRepository for the current request."""
    return EventRepository(db)


def get_camera_repo(db: Session = Depends(get_db)) -> CameraRepository:
    """Dependency: return a CameraRepository for the current request."""
    return CameraRepository(db)


def get_alert_repo(db: Session = Depends(get_db)) -> AlertRepository:
    """Dependency: return an AlertRepository for the current request."""
    return AlertRepository(db)


def get_user_repo(db: Session = Depends(get_db)) -> UserRepository:
    """Dependency: return a UserRepository for the current request."""
    return UserRepository(db)


# Model registry / manager singletons configured from default config
_cfg = load_config()
_registry = ModelRegistry(_cfg.get("model_management", {}).get("registry_path", "models/registry.json"))
_model_manager = ModelManager(_registry)


def get_model_registry() -> ModelRegistry:
    """Dependency: return the global ModelRegistry singleton."""
    return _registry


def get_model_manager() -> ModelManager:
    """Dependency: return the global ModelManager singleton."""
    return _model_manager
