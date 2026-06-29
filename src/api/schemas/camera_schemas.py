"""Camera request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ZoneSchema(BaseModel):
    zone_id: str
    name: str
    zone_type: str
    polygon: List[List[float]]
    schedule_start: Optional[str] = None
    schedule_end: Optional[str] = None
    rules: Optional[List[dict]] = None


class ZoneUpdate(BaseModel):
    zones: List[ZoneSchema]


class CameraCreate(BaseModel):
    id: str
    name: str
    stream_url: str
    location: Optional[str] = None
    enabled: bool = True
    indoor: bool = True
    resolution_width: Optional[int] = None
    resolution_height: Optional[int] = None
    fps: Optional[int] = None


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    stream_url: Optional[str] = None
    location: Optional[str] = None
    enabled: Optional[bool] = None
    indoor: Optional[bool] = None
    fps: Optional[int] = None


class CameraResponse(BaseModel):
    id: str
    name: str
    stream_url: str
    location: Optional[str]
    enabled: bool
    indoor: bool
    zones_config: Any
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}
