"""
SQLAlchemy ORM models for all persistent entities.
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func

try:
    from sqlalchemy import JSON
except ImportError:
    from sqlalchemy.types import JSON  # type: ignore


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class Camera(Base):
    """Represents a registered surveillance camera."""

    __tablename__ = "cameras"

    id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    location = Column(String(200))
    stream_url = Column(String(500), nullable=False)
    enabled = Column(Boolean, default=True)
    indoor = Column(Boolean, default=True)
    resolution_width = Column(Integer)
    resolution_height = Column(Integer)
    fps = Column(Integer)

    zones_config = Column(JSON, default=list)
    detection_config = Column(JSON, default=dict)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    events = relationship("Event", back_populates="camera", lazy="dynamic")


class Event(Base):
    """A detected and scored surveillance event."""

    __tablename__ = "events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)

    event_category = Column(String(50), nullable=False, index=True)
    event_label = Column(String(50), nullable=False)
    severity_score = Column(Float, nullable=False)

    contributing_signals = Column(JSON)
    dominant_signal = Column(String(50))

    track_id = Column(Integer)
    zone_id = Column(String(50))
    zone_name = Column(String(200))
    bbox = Column(JSON)

    clip_path = Column(String(500))
    clip_duration = Column(Float)

    alert_decision = Column(String(20))
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))

    feedback_correct = Column(Boolean)
    feedback_label = Column(String(50))
    feedback_notes = Column(Text)

    created_at = Column(DateTime, server_default=func.now())

    camera = relationship("Camera", back_populates="events")
    alerts = relationship("AlertRecord", back_populates="event", lazy="dynamic")

    __table_args__ = (
        Index("ix_events_camera_timestamp", "camera_id", "timestamp"),
        Index("ix_events_category_timestamp", "event_category", "timestamp"),
    )


class AlertRecord(Base):
    """Persisted record of a delivered alert."""

    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = Column(String(36), ForeignKey("events.id"), nullable=False)

    priority = Column(String(20), nullable=False)
    title = Column(String(500))
    description = Column(Text)

    channels_attempted = Column(JSON)
    channels_succeeded = Column(JSON)

    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime, server_default=func.now())
    delivered_at = Column(DateTime)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))
    dismissed_at = Column(DateTime)
    dismiss_reason = Column(String(200))
    escalated_at = Column(DateTime)

    event = relationship("Event", back_populates="alerts")


class OperatorFeedback(Base):
    """Operator correctness feedback for a detected event."""

    __tablename__ = "operator_feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = Column(String(36), ForeignKey("events.id"), nullable=False)
    operator = Column(String(100), nullable=False)

    is_correct = Column(Boolean, nullable=False)
    corrected_label = Column(String(50))
    notes = Column(Text)

    created_at = Column(DateTime, server_default=func.now())


class SystemLog(Base):
    """Structured system log entry."""

    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    component = Column(String(100), nullable=False, index=True)
    level = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    extra = Column(JSON)
    timestamp = Column(DateTime, server_default=func.now(), index=True)


class User(Base):
    """Operator / admin user account."""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    full_name = Column(String(200))
    role = Column(String(20), default="operator")
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
