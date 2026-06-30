"""
Model management API routes for Phase 12.

Provides endpoints to register, list, activate, retire, delete, and swap model
versions at runtime, as well as rollback to the previously loaded model.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_current_user_payload, require_role
from src.api.dependencies import get_model_manager, get_model_registry
from src.common.model_manager import ModelManager
from src.common.model_manager_models import ModelStatus, ModelType, ModelVersion
from src.common.model_registry import ModelRegistry

router = APIRouter()


@router.get("/status", tags=["models"])
async def get_model_status(
    manager: ModelManager = Depends(get_model_manager),
    _: dict = Depends(get_current_user_payload),
):
    """Return the runtime status of loaded models across all types."""
    return manager.get_status()


@router.get("/active/{model_type}", tags=["models"])
async def get_active_model(
    model_type: str,
    registry: ModelRegistry = Depends(get_model_registry),
    _: dict = Depends(get_current_user_payload),
):
    """Return the currently active model for a given type."""
    model = registry.get_active(ModelType(model_type))
    if model is None:
        raise HTTPException(status_code=404, detail=f"No active model for type {model_type}")
    return model.to_dict()


@router.get("", tags=["models"])
async def list_models(
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    registry: ModelRegistry = Depends(get_model_registry),
    _: dict = Depends(get_current_user_payload),
):
    """List all registered model versions, optionally filtered by type/status."""
    mt = ModelType(model_type) if model_type else None
    st = ModelStatus(status) if status else None
    models = registry.list_models(model_type=mt, status=st)
    return {"models": [m.to_dict() for m in models], "total": len(models)}


@router.post("", tags=["models"], status_code=201)
async def register_model(
    payload: dict,
    registry: ModelRegistry = Depends(get_model_registry),
    _: dict = Depends(require_role("admin")),
):
    """Register a new model version.  Admin only."""
    required = {"model_id", "model_type", "version", "path"}
    missing = required - set(payload.keys())
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing fields: {', '.join(missing)}")

    try:
        model = ModelVersion(
            model_id=payload["model_id"],
            model_type=ModelType(payload["model_type"]),
            version=payload["version"],
            path=payload["path"],
            description=payload.get("description", ""),
            metrics=payload.get("metrics", {}),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        registry.register(model)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return model.to_dict()


@router.get("/{model_id}", tags=["models"])
async def get_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
    _: dict = Depends(get_current_user_payload),
):
    """Return a single registered model by ID."""
    model = registry.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model.to_dict()


@router.post("/{model_id}/activate", tags=["models"])
async def activate_model(
    model_id: str,
    manager: ModelManager = Depends(get_model_manager),
    registry: ModelRegistry = Depends(get_model_registry),
    _: dict = Depends(require_role("admin")),
):
    """Activate a model version and hot-swap it into the inference pipeline."""
    if registry.get(model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")
    result = manager.swap(model_id)
    return result.to_dict()


@router.post("/{model_id}/retire", tags=["models"])
async def retire_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
    _: dict = Depends(require_role("admin")),
):
    """Retire a model version (admin only)."""
    try:
        model = registry.retire(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return model.to_dict()


@router.delete("/{model_id}", tags=["models"])
async def delete_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
    _: dict = Depends(require_role("admin")),
):
    """Delete a model version from the registry."""
    try:
        registry.delete(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": model_id}


@router.post("/{model_type}/rollback", tags=["models"])
async def rollback_model(
    model_type: str,
    manager: ModelManager = Depends(get_model_manager),
    _: dict = Depends(require_role("admin")),
):
    """Rollback the currently loaded model for a type to the previous version."""
    result = manager.rollback(ModelType(model_type))
    if result is None:
        raise HTTPException(status_code=400, detail="No rollback history available")
    return result.to_dict()
