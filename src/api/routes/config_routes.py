"""Config and system health routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.auth import get_current_user_payload, require_role

router = APIRouter()

_runtime_config: dict = {}


def set_runtime_config(cfg: dict) -> None:
    """Store a reference to the live config for serving via API."""
    _runtime_config.update(cfg)


@router.get("/config")
def get_config(_: dict = Depends(require_role("admin"))):
    """Return current config (admin only, passwords redacted)."""
    safe = dict(_runtime_config)
    if "auth" in safe:
        safe["auth"] = {k: "***" if "password" in k or "secret" in k else v
                        for k, v in safe["auth"].items()}
    return safe


@router.put("/config/scoring")
def update_scoring_config(
    updates: dict,
    _: dict = Depends(require_role("admin")),
):
    """Update scoring thresholds at runtime."""
    _runtime_config.setdefault("scoring", {}).update(updates)
    return {"updated": "scoring"}


@router.put("/config/notifications")
def update_notification_config(
    updates: dict,
    _: dict = Depends(require_role("admin")),
):
    """Update notification settings at runtime."""
    _runtime_config.setdefault("notifications", {}).update(updates)
    return {"updated": "notifications"}


@router.get("/system/health")
def health_check(_: dict = Depends(get_current_user_payload)):
    """Return system component health status."""
    return {
        "status": "ok",
        "components": {
            "api": "ok",
            "database": "ok",
        },
    }


@router.get("/system/stats")
def system_stats(_: dict = Depends(get_current_user_payload)):
    """Return system-wide statistics."""
    return {
        "status": "ok",
        "version": "1.0.0",
    }
