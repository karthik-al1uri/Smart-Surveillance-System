"""
Database repository layer.
Handles all CRUD operations with the database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from src.api.auth import hash_password, verify_password
from src.common.db_models import AlertRecord, Camera, Event, OperatorFeedback, User
from src.common.logger import get_logger

logger = get_logger("api.repositories")


# ---------------------------------------------------------------------------
# EventRepository
# ---------------------------------------------------------------------------

class EventRepository:
    """CRUD operations for :class:`~src.common.db_models.Event` records.

    Args:
        session: Active SQLAlchemy session.
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def create_event(self, data: dict) -> Event:
        """Persist a new event record.

        Args:
            data: Dict with event fields (maps directly to ORM columns).

        Returns:
            Persisted :class:`~src.common.db_models.Event`.
        """
        event = Event(
            id=data.get("id", str(uuid.uuid4())),
            camera_id=data["camera_id"],
            timestamp=data["timestamp"],
            event_category=data["event_category"],
            event_label=data["event_label"],
            severity_score=data["severity_score"],
            contributing_signals=data.get("contributing_signals"),
            dominant_signal=data.get("dominant_signal"),
            track_id=data.get("track_id"),
            zone_id=data.get("zone_id"),
            zone_name=data.get("zone_name"),
            bbox=data.get("bbox"),
            clip_path=data.get("clip_path"),
            clip_duration=data.get("clip_duration"),
            alert_decision=data.get("alert_decision"),
        )
        self._db.add(event)
        self._db.flush()
        return event

    def get_event(self, event_id: str) -> Optional[Event]:
        """Return event by ID or ``None``.

        Args:
            event_id: UUID string.
        """
        return self._db.get(Event, event_id)

    def list_events(
        self,
        camera_id: Optional[str] = None,
        category: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Event], int]:
        """Paginated event list with optional filters.

        Args:
            camera_id: Filter by camera.
            category: Filter by event category.
            start_time: Inclusive lower bound on timestamp.
            end_time: Inclusive upper bound on timestamp.
            limit: Page size.
            offset: Page offset.

        Returns:
            Tuple of (events list, total count matching filters).
        """
        q = self._db.query(Event)
        if camera_id:
            q = q.filter(Event.camera_id == camera_id)
        if category:
            q = q.filter(Event.event_category == category)
        if start_time:
            q = q.filter(Event.timestamp >= start_time)
        if end_time:
            q = q.filter(Event.timestamp <= end_time)
        total = q.count()
        events = q.order_by(Event.timestamp.desc()).offset(offset).limit(limit).all()
        return events, total

    def update_feedback(
        self,
        event_id: str,
        is_correct: bool,
        corrected_label: Optional[str] = None,
        notes: Optional[str] = None,
        operator: Optional[str] = None,
    ) -> Optional[Event]:
        """Attach operator feedback to an event.

        Args:
            event_id: Target event.
            is_correct: Whether the prediction was correct.
            corrected_label: Operator's corrected label, if wrong.
            notes: Free-text notes.
            operator: Operator username.

        Returns:
            Updated event or ``None`` if not found.
        """
        event = self.get_event(event_id)
        if not event:
            return None
        event.feedback_correct = is_correct
        event.feedback_label = corrected_label
        event.feedback_notes = notes
        if operator:
            event.acknowledged_by = operator
        feedback = OperatorFeedback(
            event_id=event_id,
            operator=operator or "unknown",
            is_correct=is_correct,
            corrected_label=corrected_label,
            notes=notes,
        )
        self._db.add(feedback)
        self._db.flush()
        return event

    def get_event_stats(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict:
        """Return aggregated event statistics.

        Args:
            camera_id: Limit stats to one camera.
            start_time: Inclusive lower bound.
            end_time: Inclusive upper bound.

        Returns:
            Dict with ``total``, ``by_category``, and ``by_camera`` keys.
        """
        q = self._db.query(Event)
        if camera_id:
            q = q.filter(Event.camera_id == camera_id)
        if start_time:
            q = q.filter(Event.timestamp >= start_time)
        if end_time:
            q = q.filter(Event.timestamp <= end_time)

        events = q.all()
        by_category: Dict[str, int] = {}
        by_camera: Dict[str, int] = {}
        for e in events:
            by_category[e.event_category] = by_category.get(e.event_category, 0) + 1
            by_camera[e.camera_id] = by_camera.get(e.camera_id, 0) + 1

        return {"total": len(events), "by_category": by_category, "by_camera": by_camera}


# ---------------------------------------------------------------------------
# CameraRepository
# ---------------------------------------------------------------------------

class CameraRepository:
    """CRUD operations for :class:`~src.common.db_models.Camera` records.

    Args:
        session: Active SQLAlchemy session.
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def create_camera(self, data: dict) -> Camera:
        """Create and persist a new camera.

        Args:
            data: Dict with camera fields.

        Returns:
            Persisted :class:`~src.common.db_models.Camera`.
        """
        cam = Camera(
            id=data["id"],
            name=data["name"],
            stream_url=data["stream_url"],
            location=data.get("location"),
            enabled=data.get("enabled", True),
            indoor=data.get("indoor", True),
            resolution_width=data.get("resolution_width"),
            resolution_height=data.get("resolution_height"),
            fps=data.get("fps"),
            zones_config=data.get("zones_config", []),
            detection_config=data.get("detection_config", {}),
        )
        self._db.add(cam)
        self._db.flush()
        return cam

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        """Return camera by ID or ``None``."""
        return self._db.get(Camera, camera_id)

    def list_cameras(self) -> List[Camera]:
        """Return all cameras."""
        return self._db.query(Camera).all()

    def update_camera(self, camera_id: str, updates: dict) -> Optional[Camera]:
        """Update camera fields.

        Args:
            camera_id: Camera to update.
            updates: Dict of field → new value pairs.

        Returns:
            Updated camera or ``None`` if not found.
        """
        cam = self.get_camera(camera_id)
        if not cam:
            return None
        for key, val in updates.items():
            if hasattr(cam, key):
                setattr(cam, key, val)
        self._db.flush()
        return cam

    def delete_camera(self, camera_id: str) -> bool:
        """Delete a camera by ID.

        Args:
            camera_id: Camera to delete.

        Returns:
            ``True`` if deleted, ``False`` if not found.
        """
        cam = self.get_camera(camera_id)
        if not cam:
            return False
        self._db.delete(cam)
        self._db.flush()
        return True

    def update_zones(self, camera_id: str, zones: list) -> Optional[Camera]:
        """Replace the zones config for a camera.

        Args:
            camera_id: Target camera.
            zones: New zones list.

        Returns:
            Updated camera or ``None``.
        """
        cam = self.get_camera(camera_id)
        if not cam:
            return None
        cam.zones_config = zones
        self._db.flush()
        return cam


# ---------------------------------------------------------------------------
# AlertRepository
# ---------------------------------------------------------------------------

class AlertRepository:
    """CRUD operations for :class:`~src.common.db_models.AlertRecord`.

    Args:
        session: Active SQLAlchemy session.
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def create_alert(self, data: dict) -> AlertRecord:
        """Persist a new alert record.

        Args:
            data: Dict with alert fields.

        Returns:
            Persisted :class:`~src.common.db_models.AlertRecord`.
        """
        record = AlertRecord(
            id=data.get("id", str(uuid.uuid4())),
            event_id=data["event_id"],
            priority=data["priority"],
            title=data.get("title"),
            description=data.get("description"),
            channels_attempted=data.get("channels_attempted", {}),
            channels_succeeded=data.get("channels_succeeded", {}),
            status=data.get("status", "pending"),
        )
        self._db.add(record)
        self._db.flush()
        return record

    def get_alert(self, alert_id: str) -> Optional[AlertRecord]:
        """Return alert record by ID or ``None``."""
        return self._db.get(AlertRecord, alert_id)

    def list_alerts(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[AlertRecord], int]:
        """Paginated alert list with optional filters.

        Args:
            status: Filter by status string.
            priority: Filter by priority string.
            limit: Page size.
            offset: Page offset.

        Returns:
            Tuple of (alerts list, total count).
        """
        q = self._db.query(AlertRecord)
        if status:
            q = q.filter(AlertRecord.status == status)
        if priority:
            q = q.filter(AlertRecord.priority == priority)
        total = q.count()
        records = q.order_by(AlertRecord.created_at.desc()).offset(offset).limit(limit).all()
        return records, total

    def update_status(
        self,
        alert_id: str,
        new_status: str,
        operator: Optional[str] = None,
    ) -> Optional[AlertRecord]:
        """Update alert lifecycle status.

        Args:
            alert_id: Target alert.
            new_status: New status string.
            operator: Optional operator name for ack/dismiss.

        Returns:
            Updated record or ``None``.
        """
        record = self.get_alert(alert_id)
        if not record:
            return None
        record.status = new_status
        now = datetime.now(timezone.utc)
        if new_status == "acknowledged":
            record.acknowledged_at = now
            record.acknowledged_by = operator
        elif new_status == "dismissed":
            record.dismissed_at = now
            record.acknowledged_by = operator
        elif new_status == "escalated":
            record.escalated_at = now
        self._db.flush()
        return record

    def get_alert_stats(self) -> dict:
        """Return aggregated alert statistics."""
        records = self._db.query(AlertRecord).all()
        by_status: Dict[str, int] = {}
        by_priority: Dict[str, int] = {}
        for r in records:
            by_status[r.status] = by_status.get(r.status, 0) + 1
            by_priority[r.priority] = by_priority.get(r.priority, 0) + 1
        return {"total": len(records), "by_status": by_status, "by_priority": by_priority}


# ---------------------------------------------------------------------------
# UserRepository
# ---------------------------------------------------------------------------

class UserRepository:
    """CRUD operations for :class:`~src.common.db_models.User`.

    Args:
        session: Active SQLAlchemy session.
    """

    def __init__(self, session: Session) -> None:
        self._db = session

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "operator",
        full_name: Optional[str] = None,
    ) -> User:
        """Create and persist a new user.

        Args:
            username: Unique login name.
            password: Plain-text password (will be hashed).
            role: ``"admin"``, ``"operator"``, or ``"viewer"``.
            full_name: Optional display name.

        Returns:
            Persisted :class:`~src.common.db_models.User`.
        """
        user = User(
            username=username,
            hashed_password=hash_password(password),
            role=role,
            full_name=full_name,
        )
        self._db.add(user)
        self._db.flush()
        return user

    def get_user(self, username: str) -> Optional[User]:
        """Return user by username or ``None``."""
        return self._db.query(User).filter(User.username == username).first()

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Return user by ID or ``None``."""
        return self._db.get(User, user_id)

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Verify credentials and return user on success.

        Args:
            username: Login name.
            password: Plain-text candidate password.

        Returns:
            :class:`~src.common.db_models.User` if valid, else ``None``.
        """
        user = self.get_user(username)
        if not user or not user.enabled:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def list_users(self) -> List[User]:
        """Return all users."""
        return self._db.query(User).all()

    def update_user(self, user_id: str, updates: dict) -> Optional[User]:
        """Update user fields.

        Args:
            user_id: Target user ID.
            updates: Dict of field → value pairs. ``"password"`` is hashed automatically.

        Returns:
            Updated user or ``None``.
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        for key, val in updates.items():
            if key == "password":
                user.hashed_password = hash_password(val)
            elif hasattr(user, key):
                setattr(user, key, val)
        self._db.flush()
        return user

    def delete_user(self, user_id: str) -> bool:
        """Delete a user by ID.

        Args:
            user_id: Target user.

        Returns:
            ``True`` if deleted.
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        self._db.delete(user)
        self._db.flush()
        return True

    def count(self) -> int:
        """Return total number of users."""
        return self._db.query(User).count()
