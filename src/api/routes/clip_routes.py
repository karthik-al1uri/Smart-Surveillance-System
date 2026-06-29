"""Clip routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.api.auth import get_current_user_payload, require_role
from src.api.dependencies import get_event_repo
from src.api.repositories import EventRepository

router = APIRouter()


@router.get("/storage")
def storage_stats(
    repo: EventRepository = Depends(get_event_repo),
    _: dict = Depends(get_current_user_payload),
):
    """Return clip storage statistics."""
    events, total = repo.list_events(limit=10000)
    clips = [e for e in events if e.clip_path]
    total_size = 0
    for e in clips:
        p = Path(e.clip_path)
        if p.exists():
            total_size += p.stat().st_size
    return {
        "total_clips": len(clips),
        "total_size_bytes": total_size,
        "total_size_gb": round(total_size / (1024 ** 3), 4),
    }


@router.get("")
def list_clips(
    repo: EventRepository = Depends(get_event_repo),
    _: dict = Depends(get_current_user_payload),
):
    """List all events that have associated clips."""
    events, _ = repo.list_events(limit=500)
    return [
        {"event_id": e.id, "camera_id": e.camera_id,
         "clip_path": e.clip_path, "timestamp": str(e.timestamp)}
        for e in events if e.clip_path
    ]


@router.get("/{event_id}")
def get_clip(
    event_id: str,
    repo: EventRepository = Depends(get_event_repo),
    _: dict = Depends(get_current_user_payload),
):
    """Stream/download the clip for a given event."""
    ev = repo.get_event(event_id)
    if not ev or not ev.clip_path:
        raise HTTPException(status_code=404, detail="Clip not found")
    p = Path(ev.clip_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Clip file not found on disk")
    return FileResponse(str(p), media_type="video/mp4", filename=p.name)


@router.delete("/{event_id}")
def delete_clip(
    event_id: str,
    repo: EventRepository = Depends(get_event_repo),
    _: dict = Depends(require_role("admin")),
):
    """Delete a clip file (admin only)."""
    ev = repo.get_event(event_id)
    if not ev or not ev.clip_path:
        raise HTTPException(status_code=404, detail="Clip not found")
    p = Path(ev.clip_path)
    if p.exists():
        p.unlink()
    ev.clip_path = None
    return {"deleted": event_id}
