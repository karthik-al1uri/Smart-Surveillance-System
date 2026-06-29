"""
Anomaly Scoring Engine.

Aggregates signals from action recognition, zone violations, and object
detection into a single severity score and makes alert/suppress/escalate
decisions.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.detection.yolo_detector import Detection
from src.recognition.action_classes import ActionCategory, ActionPrediction
from src.scoring.scoring_models import (
    AlertDecision,
    ScoredEvent,
    ScoringConfig,
    ScoringSignal,
    SignalType,
)
from src.scoring.zone_models import RuleType, ZoneViolation

logger = get_logger("scoring.anomaly_scorer")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZONE_SIGNAL_VALUES: Dict[RuleType, float] = {
    RuleType.NO_ENTRY: 1.0,
    RuleType.INTRUSION: 0.8,
    RuleType.CROWD_LIMIT: 0.7,
    RuleType.LOITERING: None,   # computed from dwell time
    RuleType.ABANDONED_OBJECT: 0.6,
    RuleType.WRONG_DIRECTION: 0.5,
}

_ACTION_CATEGORY_MULTIPLIER: Dict[ActionCategory, float] = {
    ActionCategory.VIOLENT: 1.0,
    ActionCategory.SUSPICIOUS: 0.7,
    ActionCategory.URGENT: 0.9,
    ActionCategory.NORMAL: 0.0,
}


def _build_config(raw: dict) -> ScoringConfig:
    """Parse the ``scoring`` section of a config dict into a :class:`ScoringConfig`."""
    w = raw.get("weights", {})
    return ScoringConfig(
        weight_action=float(w.get("action", 0.35)),
        weight_zone=float(w.get("zone", 0.25)),
        weight_weapon=float(w.get("weapon", 0.30)),
        weight_time_of_day=float(w.get("time_of_day", 0.10)),
        alert_threshold=float(raw.get("alert_threshold", 0.55)),
        escalation_threshold=float(raw.get("escalation_threshold", 0.85)),
        instant_alert_classes=list(raw.get("instant_alert_classes", ["knife", "gun", "rifle", "weapon"])),
        cooldown_seconds=float(raw.get("cooldown_seconds", 30.0)),
        hysteresis_count=int(raw.get("hysteresis_count", 2)),
        high_risk_hours=list(raw.get("high_risk_hours", [22, 23, 0, 1, 2, 3, 4, 5])),
        time_risk_multiplier=float(raw.get("time_risk_multiplier", 0.15)),
        stale_track_cleanup_seconds=float(raw.get("stale_track_cleanup_seconds", 60.0)),
    )


class AnomalyScorer:
    """Aggregates per-frame signals into severity scores and alert decisions.

    Maintains stateful records for hysteresis counters and cooldown timers.
    Caller must invoke :meth:`score_frame` once per analysis cycle.

    Args:
        config: Full project config dict; reads the ``scoring`` section.
    """

    def __init__(self, config: dict) -> None:
        raw = config.get("scoring", {})
        self._global_config: ScoringConfig = _build_config(raw)

        # Per-camera config overrides
        self._camera_configs: Dict[str, ScoringConfig] = {}
        for cam_id, overrides in raw.get("camera_overrides", {}).items():
            merged = dict(raw)
            merged.update(overrides)
            if "weights" in overrides:
                base_weights = dict(raw.get("weights", {}))
                base_weights.update(overrides["weights"])
                merged["weights"] = base_weights
            self._camera_configs[cam_id] = _build_config(merged)

        # Stateful scoring data  key = (camera_id, track_id)
        self._consecutive_high: Dict[Tuple[str, int], int] = {}
        self._last_alert: Dict[Tuple[str, int], float] = {}
        self._last_seen: Dict[Tuple[str, int], float] = {}
        self._event_counter: int = 0

        # Statistics
        self._stats: Dict = {
            "total_events_scored": 0,
            "alerts_fired": 0,
            "alerts_suppressed": 0,
            "alerts_escalated": 0,
            "events_by_category": {},
            "score_sum_by_camera": {},
            "score_count_by_camera": {},
            "false_positive_feedback_count": 0,
        }

        logger.info("AnomalyScorer initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_frame(
        self,
        camera_id: str,
        timestamp: float,
        action_predictions: List[ActionPrediction],
        zone_violations: List[ZoneViolation],
        object_detections: List[Detection],
    ) -> List[ScoredEvent]:
        """Score all signals for one analysis cycle.

        Args:
            camera_id: Source camera identifier.
            timestamp: Current Unix timestamp.
            action_predictions: Action predictions for this frame.
            zone_violations: Zone rule violations for this frame.
            object_detections: Raw object detections (for weapon checks).

        Returns:
            List of :class:`~src.scoring.scoring_models.ScoredEvent` — may be
            empty if no signals are present.
        """
        cfg = self.get_config_for_camera(camera_id)
        self._cleanup_stale_tracks(camera_id, timestamp, cfg.stale_track_cleanup_seconds)

        events: List[ScoredEvent] = []

        # A) Instant weapon alerts
        events.extend(
            self._score_weapon_detections(camera_id, timestamp, object_detections, cfg)
        )

        # B) Per-track person scoring
        track_ids = self._collect_track_ids(action_predictions, zone_violations)
        for track_id in track_ids:
            event = self._score_track(
                camera_id, timestamp, track_id,
                action_predictions, zone_violations, cfg,
            )
            if event:
                events.append(event)

        # C) Non-person zone events (track_id == -1)
        non_person_viols = [v for v in zone_violations if v.track_id == -1]
        if non_person_viols:
            event = self._score_non_person_zone(camera_id, timestamp, non_person_viols, cfg)
            if event:
                events.append(event)

        # Update stats
        for ev in events:
            self._update_stats(camera_id, ev)

        return events

    def get_config_for_camera(self, camera_id: str) -> ScoringConfig:
        """Return camera-specific config, falling back to global default.

        Args:
            camera_id: Camera identifier.
        """
        return self._camera_configs.get(camera_id, self._global_config)

    def update_config(self, camera_id: Optional[str], updates: dict) -> None:
        """Update scoring parameters at runtime.

        Args:
            camera_id: Camera to update, or ``None`` to update global config.
            updates: Mapping of :class:`ScoringConfig` field names → new values.
        """
        target = self._camera_configs.get(camera_id) if camera_id else self._global_config
        if target is None:
            raw = {}
            raw.update(updates)
            self._camera_configs[camera_id] = _build_config(raw)
            return
        for k, v in updates.items():
            if hasattr(target, k):
                setattr(target, k, v)
        logger.debug("Scoring config updated (camera=%s): %s", camera_id, list(updates.keys()))

    def get_stats(self) -> dict:
        """Return cumulative scoring statistics.

        Returns:
            Dict with total counts, per-category breakdown, and per-camera avg scores.
        """
        avg_scores = {}
        for cam, total in self._stats["score_count_by_camera"].items():
            if total > 0:
                avg_scores[cam] = round(self._stats["score_sum_by_camera"][cam] / total, 4)
        return {
            "total_events_scored": self._stats["total_events_scored"],
            "alerts_fired": self._stats["alerts_fired"],
            "alerts_suppressed": self._stats["alerts_suppressed"],
            "alerts_escalated": self._stats["alerts_escalated"],
            "events_by_category": dict(self._stats["events_by_category"]),
            "avg_score_by_camera": avg_scores,
            "false_positive_feedback_count": self._stats["false_positive_feedback_count"],
        }

    def reset_state(self, camera_id: Optional[str] = None) -> None:
        """Clear cooldown timers and hysteresis counters.

        Args:
            camera_id: Reset only this camera's state.  If ``None``, resets all.
        """
        if camera_id is None:
            self._consecutive_high.clear()
            self._last_alert.clear()
            self._last_seen.clear()
        else:
            for d in (self._consecutive_high, self._last_alert, self._last_seen):
                stale = [k for k in d if k[0] == camera_id]
                for k in stale:
                    del d[k]
        logger.debug("Scorer state reset (camera=%s).", camera_id)

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _score_weapon_detections(
        self,
        camera_id: str,
        timestamp: float,
        detections: List[Detection],
        cfg: ScoringConfig,
    ) -> List[ScoredEvent]:
        events = []
        for det in detections:
            if det.class_name.lower() not in [c.lower() for c in cfg.instant_alert_classes]:
                continue
            cooldown_key = (camera_id, -2 - hash(det.class_name) % 10000)
            if self._in_cooldown(cooldown_key, cfg.cooldown_seconds, timestamp):
                continue
            self._last_alert[cooldown_key] = timestamp

            signal = ScoringSignal(
                signal_type=SignalType.WEAPON_DETECTED,
                source="yolo_detector",
                value=det.confidence,
                weight=cfg.weight_weapon,
                weighted_value=det.confidence * cfg.weight_weapon,
                details=f"{det.class_name} detected (conf={det.confidence:.2f})",
                raw_data={"class_name": det.class_name, "confidence": det.confidence, "bbox": det.bbox},
            )
            event = ScoredEvent(
                event_id=self._new_event_id(),
                camera_id=camera_id,
                track_id=None,
                timestamp=timestamp,
                severity_score=1.0,
                contributing_signals=[signal],
                dominant_signal=SignalType.WEAPON_DETECTED,
                event_category="weapon",
                event_label=f"{det.class_name}_detected",
                alert_decision=AlertDecision.ESCALATED,
                bbox=det.bbox,
            )
            events.append(event)
            logger.warning("INSTANT ALERT — weapon: %s (cam=%s)", det.class_name, camera_id)
        return events

    def _score_track(
        self,
        camera_id: str,
        timestamp: float,
        track_id: int,
        action_predictions: List[ActionPrediction],
        zone_violations: List[ZoneViolation],
        cfg: ScoringConfig,
    ) -> Optional[ScoredEvent]:
        self._last_seen[(camera_id, track_id)] = timestamp
        signals: List[ScoringSignal] = []

        # Action signal
        action_pred = next((p for p in action_predictions if p.track_id == track_id), None)
        action_value = 0.0
        action_category = "normal"
        action_label = "normal"
        bbox = None

        if action_pred is not None:
            mult = _ACTION_CATEGORY_MULTIPLIER.get(action_pred.category, 0.0)
            action_value = action_pred.confidence * mult
            action_category = action_pred.category.value
            action_label = action_pred.label.value
            if action_value > 0:
                signals.append(ScoringSignal(
                    signal_type=SignalType.ACTION_CLASSIFICATION,
                    source="action_classifier",
                    value=action_value,
                    weight=cfg.weight_action,
                    weighted_value=action_value * cfg.weight_action,
                    details=f"{action_pred.label.value} detected (conf={action_pred.confidence:.2f})",
                    raw_data={"category": action_pred.category.value, "label": action_pred.label.value,
                              "confidence": action_pred.confidence},
                ))

        # Zone signal
        track_viols = [v for v in zone_violations if v.track_id == track_id]
        zone_value = 0.0
        zone_id = None
        zone_name = None
        best_viol: Optional[ZoneViolation] = None

        for viol in track_viols:
            v = _ZONE_SIGNAL_VALUES.get(viol.rule_type)
            if v is None and viol.rule_type == RuleType.LOITERING:
                if viol.duration_in_zone and cfg.stale_track_cleanup_seconds > 0:
                    v = min(1.0, viol.duration_in_zone / max(1.0, viol.duration_in_zone))
                else:
                    v = 0.5
            if v and v > zone_value:
                zone_value = v
                best_viol = viol

        if best_viol is not None:
            zone_id = best_viol.zone_id
            zone_name = best_viol.zone_name
            signals.append(ScoringSignal(
                signal_type=SignalType.ZONE_VIOLATION,
                source="zone_engine",
                value=zone_value,
                weight=cfg.weight_zone,
                weighted_value=zone_value * cfg.weight_zone,
                details=f"{best_viol.rule_type.value} in '{best_viol.zone_name}'",
                raw_data={"rule_type": best_viol.rule_type.value, "zone_id": best_viol.zone_id,
                          "duration": best_viol.duration_in_zone},
            ))

        # Time-of-day signal
        hour = datetime.fromtimestamp(timestamp).hour
        time_value = cfg.time_risk_multiplier if hour in cfg.high_risk_hours else 0.0
        if time_value > 0:
            signals.append(ScoringSignal(
                signal_type=SignalType.TIME_OF_DAY,
                source="time_context",
                value=time_value,
                weight=cfg.weight_time_of_day,
                weighted_value=time_value * cfg.weight_time_of_day,
                details=f"High-risk hour ({hour:02d}:xx)",
            ))

        # No signals at all
        if not signals:
            return None

        # Compute score
        score = (
            action_value * cfg.weight_action
            + zone_value * cfg.weight_zone
            + time_value * cfg.weight_time_of_day
        )
        score = max(0.0, min(1.0, score))

        # Dominant signal
        dominant = max(signals, key=lambda s: s.weighted_value).signal_type

        # Category / label derivation
        if action_pred and action_pred.category != ActionCategory.NORMAL:
            event_category = action_category
            event_label = action_label
        elif best_viol:
            event_category = "suspicious"
            event_label = best_viol.rule_type.value
        else:
            event_category = "normal"
            event_label = "normal"

        # Hysteresis
        key = (camera_id, track_id)
        if score >= cfg.alert_threshold:
            self._consecutive_high[key] = self._consecutive_high.get(key, 0) + 1
        else:
            self._consecutive_high[key] = 0

        hysteresis_met = self._consecutive_high.get(key, 0) >= cfg.hysteresis_count

        # Decision
        decision: AlertDecision
        suppression_reason: Optional[str] = None

        if score < cfg.alert_threshold:
            decision = AlertDecision.NO_ALERT
            suppression_reason = "below_threshold"
        elif not hysteresis_met:
            decision = AlertDecision.SUPPRESSED
            suppression_reason = "hysteresis"
        elif score >= cfg.escalation_threshold:
            # Escalated events bypass cooldown
            decision = AlertDecision.ESCALATED
            self._last_alert[key] = timestamp
        elif self._in_cooldown(key, cfg.cooldown_seconds, timestamp):
            decision = AlertDecision.SUPPRESSED
            suppression_reason = "cooldown"
        else:
            decision = AlertDecision.ALERT
            self._last_alert[key] = timestamp

        return ScoredEvent(
            event_id=self._new_event_id(),
            camera_id=camera_id,
            track_id=track_id,
            timestamp=timestamp,
            severity_score=score,
            contributing_signals=signals,
            dominant_signal=dominant,
            event_category=event_category,
            event_label=event_label,
            alert_decision=decision,
            suppression_reason=suppression_reason if decision == AlertDecision.SUPPRESSED else None,
            zone_id=zone_id,
            zone_name=zone_name,
            bbox=bbox,
        )

    def _score_non_person_zone(
        self,
        camera_id: str,
        timestamp: float,
        violations: List[ZoneViolation],
        cfg: ScoringConfig,
    ) -> Optional[ScoredEvent]:
        """Score abandoned-object / crowd events (track_id == -1)."""
        best_viol = max(
            violations,
            key=lambda v: _ZONE_SIGNAL_VALUES.get(v.rule_type) or 0.5,
        )
        zone_value = _ZONE_SIGNAL_VALUES.get(best_viol.rule_type) or 0.5
        score = min(1.0, zone_value * cfg.weight_zone)

        signal = ScoringSignal(
            signal_type=SignalType.ZONE_VIOLATION,
            source="zone_engine",
            value=zone_value,
            weight=cfg.weight_zone,
            weighted_value=zone_value * cfg.weight_zone,
            details=f"{best_viol.rule_type.value} in '{best_viol.zone_name}'",
        )

        decision = AlertDecision.NO_ALERT
        if score >= cfg.alert_threshold:
            decision = AlertDecision.ALERT

        return ScoredEvent(
            event_id=self._new_event_id(),
            camera_id=camera_id,
            track_id=None,
            timestamp=timestamp,
            severity_score=score,
            contributing_signals=[signal],
            dominant_signal=SignalType.ZONE_VIOLATION,
            event_category="suspicious",
            event_label=best_viol.rule_type.value,
            alert_decision=decision,
            zone_id=best_viol.zone_id,
            zone_name=best_viol.zone_name,
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _in_cooldown(
        self, key: Tuple[str, int], cooldown_seconds: float, now: float
    ) -> bool:
        last = self._last_alert.get(key)
        return last is not None and (now - last) < cooldown_seconds

    def _cleanup_stale_tracks(
        self, camera_id: str, now: float, stale_after: float
    ) -> None:
        stale = [
            k for k, ts in self._last_seen.items()
            if k[0] == camera_id and (now - ts) > stale_after
        ]
        for k in stale:
            self._consecutive_high.pop(k, None)
            self._last_alert.pop(k, None)
            del self._last_seen[k]

    def _new_event_id(self) -> str:
        self._event_counter += 1
        return f"evt_{self._event_counter:06d}"

    def _collect_track_ids(
        self,
        predictions: List[ActionPrediction],
        violations: List[ZoneViolation],
    ) -> List[int]:
        ids = set()
        for p in predictions:
            ids.add(p.track_id)
        for v in violations:
            if v.track_id >= 0:
                ids.add(v.track_id)
        return sorted(ids)

    def _update_stats(self, camera_id: str, event: ScoredEvent) -> None:
        s = self._stats
        s["total_events_scored"] += 1
        if event.alert_decision == AlertDecision.ALERT:
            s["alerts_fired"] += 1
        elif event.alert_decision == AlertDecision.SUPPRESSED:
            s["alerts_suppressed"] += 1
        elif event.alert_decision == AlertDecision.ESCALATED:
            s["alerts_escalated"] += 1
        cat = event.event_category
        s["events_by_category"][cat] = s["events_by_category"].get(cat, 0) + 1
        s["score_sum_by_camera"][camera_id] = (
            s["score_sum_by_camera"].get(camera_id, 0.0) + event.severity_score
        )
        s["score_count_by_camera"][camera_id] = (
            s["score_count_by_camera"].get(camera_id, 0) + 1
        )
