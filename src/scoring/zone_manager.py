"""Manages zone definitions: CRUD operations, persistence, and zone lookups."""

from __future__ import annotations

from datetime import datetime, time as dtime
from typing import Dict, List, Optional

import yaml

from src.common.logger import get_logger
from src.scoring.zone_models import Rule, RuleType, Zone, ZoneType

logger = get_logger("scoring.zone_manager")


def _parse_time(value: Optional[str]) -> Optional[dtime]:
    """Parse ``"HH:MM"`` string to :class:`datetime.time`."""
    if value is None:
        return None
    h, m = map(int, str(value).split(":"))
    return dtime(h, m)


class ZoneManager:
    """CRUD manager for :class:`~src.scoring.zone_models.Zone` objects.

    Zones and their rules are loaded from the YAML config and can be modified
    at runtime.  Call :meth:`save_to_config` to persist changes back to disk.

    Args:
        config: Optional pre-loaded config dict.  If *None*, an empty manager
            is created with no zones.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._zones: Dict[str, Zone] = {}
        self._rules: Dict[str, List[Rule]] = {}

        if config:
            self.load_from_config(config)

        logger.info("ZoneManager ready with %d zone(s).", len(self._zones))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_zone(self, zone: Zone) -> None:
        """Add or replace a zone.

        Args:
            zone: :class:`~src.scoring.zone_models.Zone` instance.
        """
        self._zones[zone.zone_id] = zone
        if zone.zone_id not in self._rules:
            self._rules[zone.zone_id] = []
        logger.debug("Zone '%s' added.", zone.zone_id)

    def remove_zone(self, zone_id: str) -> None:
        """Delete a zone and its associated rules.

        Args:
            zone_id: Identifier of the zone to delete.
        """
        self._zones.pop(zone_id, None)
        self._rules.pop(zone_id, None)
        logger.debug("Zone '%s' removed.", zone_id)

    def update_zone(self, zone_id: str, updates: dict) -> None:
        """Apply a dict of attribute updates to an existing zone.

        Args:
            zone_id: Zone to update.
            updates: Mapping of field name → new value.

        Raises:
            KeyError: If ``zone_id`` does not exist.
        """
        zone = self._zones[zone_id]
        for k, v in updates.items():
            if hasattr(zone, k):
                setattr(zone, k, v)
        logger.debug("Zone '%s' updated: %s", zone_id, list(updates.keys()))

    def get_zone(self, zone_id: str) -> Optional[Zone]:
        """Return a zone by ID, or ``None`` if not found.

        Args:
            zone_id: Zone identifier.
        """
        return self._zones.get(zone_id)

    def get_zones_for_camera(self, camera_id: str) -> List[Zone]:
        """Return all zones belonging to a specific camera.

        Args:
            camera_id: Camera identifier.
        """
        return [z for z in self._zones.values() if z.camera_id == camera_id]

    def get_all_zones(self) -> List[Zone]:
        """Return all zones across all cameras."""
        return list(self._zones.values())

    def get_rules_for_zone(self, zone_id: str) -> List[Rule]:
        """Return all rules attached to a zone.

        Args:
            zone_id: Zone identifier.
        """
        return list(self._rules.get(zone_id, []))

    def add_rule(self, rule: Rule) -> None:
        """Attach a rule to its zone.

        Args:
            rule: :class:`~src.scoring.zone_models.Rule` instance.
        """
        if rule.zone_id not in self._rules:
            self._rules[rule.zone_id] = []
        self._rules[rule.zone_id].append(rule)

    # ------------------------------------------------------------------
    # Config I/O
    # ------------------------------------------------------------------

    def load_from_config(self, config: dict) -> None:
        """Load zones and rules from a parsed YAML config dict.

        Expected structure under the ``zones`` key::

            zones:
              - zone_id: "zone_a"
                camera_id: "cam_01"
                name: "Server Room Entrance"
                zone_type: "restricted"
                polygon: [[100, 200], [400, 200], [400, 500], [100, 500]]
                schedule_start: "22:00"
                schedule_end: "06:00"
                active_days: [0, 1, 2, 3, 4]
                rules:
                  - rule_id: "rule_a1"
                    rule_type: "no_entry"
                  - rule_id: "rule_a2"
                    rule_type: "loitering"
                    max_duration_seconds: 120

        Args:
            config: Top-level config dict.
        """
        for zdata in config.get("zones", []):
            zone = Zone(
                zone_id=zdata["zone_id"],
                camera_id=zdata["camera_id"],
                name=zdata["name"],
                zone_type=ZoneType(zdata.get("zone_type", "monitored")),
                polygon=[tuple(p) for p in zdata["polygon"]],
                enabled=zdata.get("enabled", True),
                schedule_start=_parse_time(zdata.get("schedule_start")),
                schedule_end=_parse_time(zdata.get("schedule_end")),
                active_days=zdata.get("active_days", [0, 1, 2, 3, 4, 5, 6]),
            )
            self.add_zone(zone)
            for rdata in zdata.get("rules", []):
                rule = Rule(
                    rule_id=rdata["rule_id"],
                    zone_id=zone.zone_id,
                    rule_type=RuleType(rdata["rule_type"]),
                    enabled=rdata.get("enabled", True),
                    max_duration_seconds=float(rdata.get("max_duration_seconds", 300)),
                    max_persons=int(rdata.get("max_persons", 10)),
                    cooldown_seconds=float(rdata.get("cooldown_seconds", 60)),
                )
                self.add_rule(rule)
        logger.info("Loaded %d zone(s) from config.", len(self._zones))

    def save_to_config(self, path: str) -> None:
        """Persist current zones and rules to a YAML file.

        Args:
            path: File path to write.
        """
        zones_data = []
        for zone in self._zones.values():
            zd: dict = {
                "zone_id": zone.zone_id,
                "camera_id": zone.camera_id,
                "name": zone.name,
                "zone_type": zone.zone_type.value,
                "polygon": [list(p) for p in zone.polygon],
                "enabled": zone.enabled,
                "active_days": zone.active_days,
                "rules": [
                    {
                        "rule_id": r.rule_id,
                        "rule_type": r.rule_type.value,
                        "enabled": r.enabled,
                        "max_duration_seconds": r.max_duration_seconds,
                        "max_persons": r.max_persons,
                        "cooldown_seconds": r.cooldown_seconds,
                    }
                    for r in self._rules.get(zone.zone_id, [])
                ],
            }
            if zone.schedule_start:
                zd["schedule_start"] = zone.schedule_start.strftime("%H:%M")
            if zone.schedule_end:
                zd["schedule_end"] = zone.schedule_end.strftime("%H:%M")
            zones_data.append(zd)

        with open(path, "w") as f:
            yaml.dump({"zones": zones_data}, f, default_flow_style=False, sort_keys=False)
        logger.info("Zones saved to '%s'.", path)

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def is_zone_active(self, zone: Zone, current_time: Optional[datetime] = None) -> bool:
        """Check whether a zone is currently active based on its schedule.

        A zone with no schedule (``schedule_start`` and ``schedule_end`` both
        ``None``) is always active.  Overnight schedules (e.g. 22:00–06:00)
        are handled correctly.

        Args:
            zone: Zone to check.
            current_time: Datetime to evaluate; defaults to ``datetime.now()``.

        Returns:
            ``True`` if the zone is enabled and its schedule is active.
        """
        if not zone.enabled:
            return False

        now = current_time or datetime.now()

        if zone.active_days and now.weekday() not in zone.active_days:
            return False

        if zone.schedule_start is None or zone.schedule_end is None:
            return True

        t = now.time().replace(second=0, microsecond=0)
        start = zone.schedule_start
        end = zone.schedule_end

        if start <= end:
            return start <= t < end
        else:
            return t >= start or t < end
