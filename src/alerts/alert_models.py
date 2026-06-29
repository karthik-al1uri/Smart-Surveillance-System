"""
Data structures for the alert and notification service.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class NotificationChannel(str, Enum):
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"
    EMAIL = "email"
    SMS = "sms"


class AlertStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"
    FAILED = "failed"


class AlertPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A fully formed alert ready for delivery."""

    alert_id: str
    event_id: str
    camera_id: str
    timestamp: float

    priority: AlertPriority
    title: str
    description: str
    event_category: str
    event_label: str
    severity_score: float

    clip_path: Optional[str] = None
    clip_url: Optional[str] = None
    thumbnail_path: Optional[str] = None

    zone_name: Optional[str] = None
    bbox: Optional[tuple] = None

    status: AlertStatus = AlertStatus.PENDING
    created_at: float = field(default_factory=time.time)
    delivered_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    escalated_at: Optional[float] = None
    dismissed_at: Optional[float] = None
    dismiss_reason: Optional[str] = None

    delivery_attempts: Dict[str, int] = field(default_factory=dict)
    delivery_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON/WebSocket transmission."""
        return {
            "alert_id": self.alert_id,
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "event_category": self.event_category,
            "event_label": self.event_label,
            "severity_score": self.severity_score,
            "clip_url": self.clip_url,
            "zone_name": self.zone_name,
            "status": self.status.value,
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
        }
