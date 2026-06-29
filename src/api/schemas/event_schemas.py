"""Event request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel


class EventResponse(BaseModel):
    id: str
    camera_id: str
    timestamp: datetime
    event_category: str
    event_label: str
    severity_score: float
    dominant_signal: Optional[str]
    track_id: Optional[int]
    zone_name: Optional[str]
    clip_path: Optional[str]
    alert_decision: Optional[str]
    acknowledged: bool
    feedback_correct: Optional[bool]
    feedback_label: Optional[str]
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


class EventListResponse(BaseModel):
    events: List[EventResponse]
    total: int
    limit: int
    offset: int


class FeedbackCreate(BaseModel):
    is_correct: bool
    corrected_label: Optional[str] = None
    notes: Optional[str] = None
    operator: Optional[str] = None


class EventStatsResponse(BaseModel):
    total: int
    by_category: dict
    by_camera: dict
