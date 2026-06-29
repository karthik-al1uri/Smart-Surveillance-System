"""
Builds Alert objects from ScoredEvents.
Generates human-readable titles and descriptions.
"""

from __future__ import annotations

import uuid
from typing import Optional

from src.alerts.alert_models import Alert, AlertPriority, AlertStatus
from src.common.logger import get_logger
from src.scoring.scoring_models import ScoredEvent

logger = get_logger("alerts.alert_builder")

_WEAPON_LABELS = {"knife", "gun", "rifle", "pistol", "scissors", "weapon"}


class AlertBuilder:
    """Builds :class:`~src.alerts.alert_models.Alert` objects from
    :class:`~src.scoring.scoring_models.ScoredEvent` objects.

    Args:
        config: Full project config dict.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._scoring_cfg = config.get("scoring", {})
        self._escalation_threshold = float(
            self._scoring_cfg.get("escalation_threshold", 0.85)
        )
        self._camera_names: dict = {}
        for cam in config.get("cameras", []):
            self._camera_names[cam.get("id", "")] = cam.get("name", cam.get("id", ""))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_alert(self, event: ScoredEvent) -> Alert:
        """Build a fully populated :class:`~src.alerts.alert_models.Alert`.

        Args:
            event: A :class:`~src.scoring.scoring_models.ScoredEvent` with
                ``alert_decision`` of ALERT or ESCALATED.

        Returns:
            Populated :class:`~src.alerts.alert_models.Alert` ready for delivery.
        """
        priority = self._determine_priority(event)
        camera_name = self._get_camera_name(event.camera_id)
        title = self._generate_title(event, camera_name)
        description = self._generate_description(event)
        clip_url = f"/api/v1/clips/{event.event_id}"

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            event_id=event.event_id,
            camera_id=event.camera_id,
            timestamp=event.timestamp,
            priority=priority,
            title=title,
            description=description,
            event_category=event.event_category,
            event_label=event.event_label,
            severity_score=event.severity_score,
            clip_url=clip_url,
            clip_path=getattr(event, "clip_path", None),
            zone_name=getattr(event, "zone_name", None),
            bbox=getattr(event, "bbox", None),
            status=AlertStatus.PENDING,
        )
        logger.debug(
            "Built alert %s: priority=%s title=%r",
            alert.alert_id, priority.value, title,
        )
        return alert

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _determine_priority(self, event: ScoredEvent) -> AlertPriority:
        label = event.event_label.lower()
        category = event.event_category.lower()

        if category == "weapon" or any(w in label for w in _WEAPON_LABELS):
            return AlertPriority.CRITICAL
        if event.severity_score >= self._escalation_threshold:
            return AlertPriority.CRITICAL
        if category in ("violent",):
            return AlertPriority.HIGH
        if category in ("urgent",):
            return AlertPriority.HIGH
        if category in ("suspicious",):
            return AlertPriority.MEDIUM
        return AlertPriority.LOW

    def _generate_title(self, event: ScoredEvent, camera_name: str) -> str:
        label = event.event_label.replace("_", " ").title()
        category = event.event_category.lower()
        zone = getattr(event, "zone_name", None)
        zone_suffix = f", Zone {zone}" if zone else ""

        if category == "weapon":
            return f"\u26a0 Weapon detected ({event.event_label}) \u2014 Camera {camera_name}"
        if category == "violent":
            return f"{label} detected \u2014 Camera {camera_name}"
        if category == "urgent":
            return f"Person {label.lower()} detected \u2014 Camera {camera_name}"
        if category == "suspicious":
            return f"{label} alert \u2014 Camera {camera_name}{zone_suffix}"
        return f"{label} \u2014 Camera {camera_name}"

    def _generate_description(self, event: ScoredEvent) -> str:
        confidence_pct = int(event.severity_score * 100)
        label = event.event_label.replace("_", " ")
        category = event.event_category
        zone = getattr(event, "zone_name", None)
        zone_str = f" in {zone}" if zone else ""

        signals = getattr(event, "contributing_signals", [])
        signal_parts = []
        for s in signals:
            sig_name = s.signal_type.value.replace("_", " ")
            signal_parts.append(f"{sig_name} ({s.value:.2f})")
        signals_str = (
            "Contributing factors: " + ", ".join(signal_parts) + "."
            if signal_parts
            else ""
        )

        base = (
            f"{category.title()} activity detected with {confidence_pct}% "
            f"confidence{zone_str}. Event: {label}."
        )
        if signals_str:
            base += f" {signals_str}"
        return base

    def _get_camera_name(self, camera_id: str) -> str:
        """Return display name for a camera, falling back to its ID.

        Args:
            camera_id: Camera identifier.

        Returns:
            Display name string.
        """
        return self._camera_names.get(camera_id, camera_id)
