"""
FastAPI application — REST API for the Smart Surveillance System.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.auth import configure_auth
from src.api.dependencies import set_session_factory
from src.api.routes.alert_routes import router as alert_router
from src.api.routes.auth_routes import router as auth_router
from src.api.routes.camera_routes import router as camera_router
from src.api.routes.clip_routes import router as clip_router
from src.api.routes.config_routes import router as config_router, set_runtime_config
from src.api.routes.event_routes import router as event_router
from src.api.routes.user_routes import router as user_router
from src.common.config import load_config
from src.common.db import create_engine_from_config, get_session_factory, init_db
from src.common.logger import get_logger

logger = get_logger("api.main")

_config: dict = {}


def create_app(config: dict | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional config dict.  Loads from ``config/default.yaml``
            if not provided.

    Returns:
        Configured :class:`fastapi.FastAPI` instance.
    """
    cfg = config if config is not None else load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _bootstrap(cfg)
        yield
        logger.info("API shutting down.")

    app = FastAPI(
        title="Smart Surveillance System API",
        version="1.0.0",
        description="REST API for the Smart Surveillance System",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
    app.include_router(camera_router, prefix="/api/v1/cameras", tags=["cameras"])
    app.include_router(event_router, prefix="/api/v1/events", tags=["events"])
    app.include_router(alert_router, prefix="/api/v1/alerts", tags=["alerts"])
    app.include_router(clip_router, prefix="/api/v1/clips", tags=["clips"])
    app.include_router(config_router, prefix="/api/v1", tags=["config"])
    app.include_router(user_router, prefix="/api/v1/users", tags=["users"])

    return app


def _bootstrap(cfg: dict) -> None:
    """Initialise DB, auth config, and default admin user."""
    configure_auth(cfg)
    set_runtime_config(cfg)

    try:
        engine = create_engine_from_config(cfg)
    except Exception as exc:
        logger.error("DB connection failed: %s — using in-memory SQLite.", exc)
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///:memory:", echo=False)

    init_db(engine)
    factory = get_session_factory(engine)
    set_session_factory(factory)

    _ensure_default_admin(factory, cfg)
    logger.info("API bootstrap complete.")


def _ensure_default_admin(factory, cfg: dict) -> None:
    from src.api.repositories import UserRepository
    from src.common.db import get_session

    with get_session(factory) as session:
        repo = UserRepository(session)
        if repo.count() == 0:
            admin_cfg = cfg.get("auth", {}).get("default_admin", {})
            username = admin_cfg.get("username", "admin")
            password = admin_cfg.get("password", "admin")
            role = admin_cfg.get("role", "admin")
            repo.create_user(username=username, password=password, role=role)
            logger.info("Default admin user '%s' created.", username)


app = create_app()
