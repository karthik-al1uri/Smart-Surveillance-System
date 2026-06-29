"""Zone & Rule Engine.

Evaluates spatial and temporal rules against tracked person positions.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

from shapely.geometry import Point, Polygon

from src.common.logger import get_logger
from src.detection.combined_pipeline import TrackedFrameAnalysis
from src.scoring.zone_manager import ZoneManager
from src.scoring.zone_models import Rule, RuleType, Zone, ZoneViolation

logger = get_logger("scoring.zone_engine")


class ZoneEngine:
    """Evaluates zone rules against tracked persons each frame.

    Maintains stateful records for:

    * **Dwell times** — how long each track has been inside each zone.
    * **Intrusion triggers** — which (track, zone) pairs have already fired once.
    * **Cooldown timestamps** — when each rule last generated an alert.
    * **Stationary objects** — for abandoned-object detection.

    Shapely :class:`~shapely.geometry.Polygon` objects are cached so they are
    not reconstructed on every frame.

    Args:
        zone_manager: Initialised :class:`~src.scoring.zone_manager.ZoneManager`.
        config: Optional config dict (reads ``zone_engine`` section).
    """

    def __init__(self, zone_manager: ZoneManager, config: Optional[dict] = None) -> None:
        self._zone_manager = zone_manager
        cfg = (config or {}).get("zone_engine", {})
        self._position_method: str = cfg.get("position_method", "bottom_center")

        self._polygon_cache: Dict[str, Polygon] = {}

        self._dwell_times: Dict[Tuple[int, str], float] = {}
        self._intrusion_triggered: Set[Tuple[int, str]] = set()
        self._last_alert_time: Dict[Tuple[str, str], float] = {}
        self._stationary_objects: Dict[str, dict] = {}

        self._stats: Dict = {
            "total_violations": 0,
            "violations_by_type": {rt.value: 0 for rt in RuleType},
        }

        logger.info("ZoneEngine initialised (position_method=%s).", self._position_method)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, tracked_analysis: TrackedFrameAnalysis) -> List[ZoneViolation]:
        """Evaluate all zone rules for one frame.

        Args:
            tracked_analysis: Output from the detection/tracking pipeline.

        Returns:
            List of :class:`~src.scoring.zone_models.ZoneViolation` objects —
            may be empty.
        """
        camera_id = tracked_analysis.camera_id
        zones = self._zone_manager.get_zones_for_camera(camera_id)
        active_zones = [z for z in zones if self._zone_manager.is_zone_active(z)]

        violations: List[ZoneViolation] = []
        now = time.time()

        active_tracks = [t for t in tracked_analysis.tracks if t.state == "active"]
        active_track_ids = {t.track_id for t in active_tracks}

        self._cleanup_stale_tracks(active_track_ids, active_zones)

        for track in active_tracks:
            pos = self.get_person_position(track.bbox, self._position_method)
            for zone in active_zones:
                if not self.is_point_in_zone(pos, zone):
                    self._handle_zone_exit(track.track_id, zone.zone_id)
                    continue

                rules = self._zone_manager.get_rules_for_zone(zone.zone_id)
                for rule in rules:
                    if not rule.enabled:
                        continue
                    v = self._evaluate_rule(rule, zone, track.track_id, now)
                    if v:
                        violations.append(v)

        self._evaluate_abandoned_objects(tracked_analysis, active_zones, now, violations)

        for v in violations:
            self._stats["total_violations"] += 1
            self._stats["violations_by_type"][v.rule_type.value] += 1

        return violations

    def get_stats(self) -> dict:
        """Return cumulative violation statistics.

        Returns:
            Dict with ``total_violations`` and ``violations_by_type``.
        """
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def is_point_in_zone(self, point: Tuple[float, float], zone: Zone) -> bool:
        """Test whether a 2-D point lies inside a zone polygon.

        Uses a cached :class:`~shapely.geometry.Polygon` for efficiency.

        Args:
            point: ``(x, y)`` position in frame coordinates.
            zone: Zone to test against.

        Returns:
            ``True`` if the point is strictly inside (or on the boundary of)
            the polygon.
        """
        poly = self._get_polygon(zone)
        return bool(poly.contains(Point(point)) or poly.boundary.contains(Point(point)))

    def get_person_position(
        self,
        bbox: Tuple[int, int, int, int],
        method: str = "bottom_center",
    ) -> Tuple[float, float]:
        """Extract a representative 2-D position from a bounding box.

        Args:
            bbox: ``(x1, y1, x2, y2)`` bounding box in pixels.
            method: ``"bottom_center"`` (feet) or ``"centroid"`` (body center).

        Returns:
            ``(x, y)`` position.
        """
        x1, y1, x2, y2 = bbox
        if method == "bottom_center":
            return ((x1 + x2) / 2.0, float(y2))
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_rule(
        self,
        rule: Rule,
        zone: Zone,
        track_id: int,
        now: float,
    ) -> Optional[ZoneViolation]:
        if rule.rule_type == RuleType.NO_ENTRY:
            return self._check_no_entry(rule, zone, track_id, now)
        if rule.rule_type == RuleType.LOITERING:
            return self._check_loitering(rule, zone, track_id, now)
        if rule.rule_type == RuleType.INTRUSION:
            return self._check_intrusion(rule, zone, track_id, now)
        if rule.rule_type == RuleType.CROWD_LIMIT:
            return None
        return None

    def _check_no_entry(
        self, rule: Rule, zone: Zone, track_id: int, now: float
    ) -> Optional[ZoneViolation]:
        cooldown_key = (rule.rule_id, str(track_id))
        if self._in_cooldown(cooldown_key, rule.cooldown_seconds, now):
            return None
        self._last_alert_time[cooldown_key] = now
        return ZoneViolation(
            rule_id=rule.rule_id,
            zone_id=zone.zone_id,
            zone_name=zone.name,
            camera_id=zone.camera_id,
            track_id=track_id,
            rule_type=RuleType.NO_ENTRY,
            timestamp=now,
            details=f"Person (track {track_id}) detected in restricted zone '{zone.name}'.",
            confidence=1.0,
        )

    def _check_loitering(
        self, rule: Rule, zone: Zone, track_id: int, now: float
    ) -> Optional[ZoneViolation]:
        key = (track_id, zone.zone_id)
        if key not in self._dwell_times:
            self._dwell_times[key] = now
            return None
        dwell = now - self._dwell_times[key]
        if dwell < rule.max_duration_seconds:
            return None
        cooldown_key = (rule.rule_id, str(track_id))
        if self._in_cooldown(cooldown_key, rule.cooldown_seconds, now):
            return None
        self._last_alert_time[cooldown_key] = now
        return ZoneViolation(
            rule_id=rule.rule_id,
            zone_id=zone.zone_id,
            zone_name=zone.name,
            camera_id=zone.camera_id,
            track_id=track_id,
            rule_type=RuleType.LOITERING,
            timestamp=now,
            details=(
                f"Person (track {track_id}) loitering in '{zone.name}' "
                f"for {dwell:.0f}s (threshold: {rule.max_duration_seconds:.0f}s)."
            ),
            confidence=1.0,
            duration_in_zone=dwell,
        )

    def _check_intrusion(
        self, rule: Rule, zone: Zone, track_id: int, now: float
    ) -> Optional[ZoneViolation]:
        key = (track_id, zone.zone_id)
        if key in self._intrusion_triggered:
            return None
        self._intrusion_triggered.add(key)
        return ZoneViolation(
            rule_id=rule.rule_id,
            zone_id=zone.zone_id,
            zone_name=zone.name,
            camera_id=zone.camera_id,
            track_id=track_id,
            rule_type=RuleType.INTRUSION,
            timestamp=now,
            details=f"Track {track_id} entered zone '{zone.name}' (intrusion).",
            confidence=1.0,
        )

    def evaluate_crowd_limit(
        self, rule: Rule, zone: Zone, tracks_in_zone: List[int], now: float
    ) -> Optional[ZoneViolation]:
        """Check crowd limit for a zone with a pre-computed track list.

        Exposed separately so tests can call it directly.

        Args:
            rule: The crowd-limit rule.
            zone: Zone being evaluated.
            tracks_in_zone: Track IDs currently inside the zone.
            now: Current Unix timestamp.

        Returns:
            :class:`~src.scoring.zone_models.ZoneViolation` or ``None``.
        """
        count = len(tracks_in_zone)
        if count <= rule.max_persons:
            return None
        cooldown_key = (rule.rule_id, zone.zone_id)
        if self._in_cooldown(cooldown_key, rule.cooldown_seconds, now):
            return None
        self._last_alert_time[cooldown_key] = now
        return ZoneViolation(
            rule_id=rule.rule_id,
            zone_id=zone.zone_id,
            zone_name=zone.name,
            camera_id=zone.camera_id,
            track_id=-1,
            rule_type=RuleType.CROWD_LIMIT,
            timestamp=now,
            details=(
                f"{count} persons in '{zone.name}' (limit: {rule.max_persons})."
            ),
            confidence=1.0,
            persons_in_zone=count,
        )

    def _evaluate_abandoned_objects(
        self,
        analysis: TrackedFrameAnalysis,
        active_zones: List[Zone],
        now: float,
        violations: List[ZoneViolation],
    ) -> None:
        """Check for objects stationary inside zones beyond the dwell threshold."""
        object_classes = {"suitcase", "backpack", "handbag", "bag"}
        for det in analysis.object_detections:
            if det.class_name.lower() not in object_classes:
                continue
            obj_key = f"{det.class_name}_{det.bbox}"
            cx = (det.bbox[0] + det.bbox[2]) / 2.0
            cy = (det.bbox[1] + det.bbox[3]) / 2.0
            pos = (cx, cy)

            if obj_key not in self._stationary_objects:
                self._stationary_objects[obj_key] = {"first_seen": now, "bbox": det.bbox}
                continue

            elapsed = now - self._stationary_objects[obj_key]["first_seen"]

            for zone in active_zones:
                if not self.is_point_in_zone(pos, zone):
                    continue
                rules = self._zone_manager.get_rules_for_zone(zone.zone_id)
                for rule in rules:
                    if rule.rule_type != RuleType.ABANDONED_OBJECT or not rule.enabled:
                        continue
                    if elapsed < rule.max_duration_seconds:
                        continue
                    cooldown_key = (rule.rule_id, obj_key)
                    if self._in_cooldown(cooldown_key, rule.cooldown_seconds, now):
                        continue
                    self._last_alert_time[cooldown_key] = now
                    violations.append(
                        ZoneViolation(
                            rule_id=rule.rule_id,
                            zone_id=zone.zone_id,
                            zone_name=zone.name,
                            camera_id=zone.camera_id,
                            track_id=-1,
                            rule_type=RuleType.ABANDONED_OBJECT,
                            timestamp=now,
                            details=(
                                f"Abandoned {det.class_name} in '{zone.name}' "
                                f"for {elapsed:.0f}s."
                            ),
                            confidence=0.9,
                            duration_in_zone=elapsed,
                        )
                    )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _handle_zone_exit(self, track_id: int, zone_id: str) -> None:
        """Clear dwell and intrusion state when a track leaves a zone."""
        key = (track_id, zone_id)
        self._dwell_times.pop(key, None)
        self._intrusion_triggered.discard(key)

    def _cleanup_stale_tracks(
        self, active_track_ids: Set[int], active_zones: List[Zone]
    ) -> None:
        """Remove state for tracks that no longer exist."""
        stale_keys = [k for k in self._dwell_times if k[0] not in active_track_ids]
        for k in stale_keys:
            del self._dwell_times[k]
        self._intrusion_triggered = {
            k for k in self._intrusion_triggered if k[0] in active_track_ids
        }

    def _in_cooldown(
        self, cooldown_key: Tuple[str, str], cooldown_seconds: float, now: float
    ) -> bool:
        last = self._last_alert_time.get(cooldown_key)
        return last is not None and (now - last) < cooldown_seconds

    def _get_polygon(self, zone: Zone) -> Polygon:
        if zone.zone_id not in self._polygon_cache:
            self._polygon_cache[zone.zone_id] = Polygon(zone.polygon)
        return self._polygon_cache[zone.zone_id]

    def invalidate_polygon_cache(self, zone_id: str) -> None:
        """Force polygon re-creation for an updated zone.

        Args:
            zone_id: Zone whose cached polygon should be dropped.
        """
        self._polygon_cache.pop(zone_id, None)
