"""Event routes."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import get_current_user_payload
from src.api.dependencies import get_event_repo
from src.api.repositories import EventRepository
from src.api.schemas.event_schemas import (
    EventListResponse, EventResponse, EventStatsResponse, FeedbackCreate,
)

router = APIRouter()


@router.get("/stats", response_model=EventStatsResponse)
def event_stats(
    camera_id: Optional[str] = None,
    repo: EventRepository = Depends(get_event_repo),
    _: dict = Depends(get_current_user_payload),
):
    """Return aggregated event statistics."""
    return repo.get_event_stats(camera_id=camera_id)


@router.get("", response_model=EventListResponse)
def list_events(
    camera_id: Optional[str] = None,
    category: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    repo: EventRepository = Depends(get_event_repo),
    _: dict = Depends(get_current_user_payload),
):
    """List events with optional filtering and pagination."""
    events, total = repo.list_events(
        camera_id=camera_id, category=category,
        start_time=start_time, end_time=end_time,
        limit=limit, offset=offset,
    )
    return EventListResponse(events=events, total=total, limit=limit, offset=offset)


@router.get("/{event_id}", response_model=EventResponse)
def get_event(
    event_id: str,
    repo: EventRepository = Depends(get_event_repo),
    _: dict = Depends(get_current_user_payload),
):
    """Get a specific event."""
    ev = repo.get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    return ev


@router.post("/{event_id}/feedback")
def submit_feedback(
    event_id: str,
    body: FeedbackCreate,
    repo: EventRepository = Depends(get_event_repo),
    payload: dict = Depends(get_current_user_payload),
):
    """Submit operator feedback for an event."""
    ev = repo.update_feedback(
        event_id,
        is_correct=body.is_correct,
        corrected_label=body.corrected_label,
        notes=body.notes,
        operator=body.operator or payload.get("sub"),
    )
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"updated": event_id}
