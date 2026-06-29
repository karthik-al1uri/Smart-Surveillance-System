"""Camera management routes."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import get_current_user_payload, require_role
from src.api.dependencies import get_camera_repo
from src.api.repositories import CameraRepository
from src.api.schemas.camera_schemas import (
    CameraCreate, CameraResponse, CameraUpdate, ZoneUpdate,
)

router = APIRouter()


@router.get("", response_model=List[CameraResponse])
def list_cameras(
    repo: CameraRepository = Depends(get_camera_repo),
    _: dict = Depends(get_current_user_payload),
):
    """List all registered cameras."""
    return repo.list_cameras()


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def create_camera(
    body: CameraCreate,
    repo: CameraRepository = Depends(get_camera_repo),
    _: dict = Depends(require_role("admin")),
):
    """Register a new camera (admin only)."""
    if repo.get_camera(body.id):
        raise HTTPException(status_code=409, detail="Camera ID already exists")
    return repo.create_camera(body.model_dump())


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(
    camera_id: str,
    repo: CameraRepository = Depends(get_camera_repo),
    _: dict = Depends(get_current_user_payload),
):
    """Get a specific camera."""
    cam = repo.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(
    camera_id: str,
    body: CameraUpdate,
    repo: CameraRepository = Depends(get_camera_repo),
    _: dict = Depends(require_role("admin")),
):
    """Update camera config (admin only)."""
    cam = repo.update_camera(camera_id, body.model_dump(exclude_none=True))
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


@router.delete("/{camera_id}")
def delete_camera(
    camera_id: str,
    repo: CameraRepository = Depends(get_camera_repo),
    _: dict = Depends(require_role("admin")),
):
    """Remove a camera (admin only)."""
    if not repo.delete_camera(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"deleted": camera_id}


@router.put("/{camera_id}/zones", response_model=CameraResponse)
def update_zones(
    camera_id: str,
    body: ZoneUpdate,
    repo: CameraRepository = Depends(get_camera_repo),
    _: dict = Depends(get_current_user_payload),
):
    """Update zone polygons for a camera."""
    cam = repo.update_zones(camera_id, [z.model_dump() for z in body.zones])
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


@router.get("/{camera_id}/zones")
def get_zones(
    camera_id: str,
    repo: CameraRepository = Depends(get_camera_repo),
    _: dict = Depends(get_current_user_payload),
):
    """Get zone definitions for a camera."""
    cam = repo.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"zones": cam.zones_config or []}
