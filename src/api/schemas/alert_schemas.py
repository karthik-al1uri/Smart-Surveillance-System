"""Alert request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel


class AlertResponse(BaseModel):
    id: str
    event_id: str
    priority: str
    title: Optional[str]
    description: Optional[str]
    status: str
    created_at: Optional[datetime]
    delivered_at: Optional[datetime]
    acknowledged_at: Optional[datetime]
    acknowledged_by: Optional[str]
    dismissed_at: Optional[datetime]
    dismiss_reason: Optional[str]

    model_config = {"from_attributes": True}


class AlertListResponse(BaseModel):
    alerts: List[AlertResponse]
    total: int
    limit: int
    offset: int


class AlertAcknowledge(BaseModel):
    operator: Optional[str] = None


class AlertDismiss(BaseModel):
    operator: Optional[str] = None
    reason: str
