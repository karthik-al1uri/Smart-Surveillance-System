"""Zone and rule data structures for spatial/temporal event detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dtime
from enum import Enum
from typing import List, Optional, Tuple


class ZoneType(str, Enum):
    """Characterises the security level of a monitored area."""

    RESTRICTED = "restricted"
    MONITORED = "monitored"
    SAFE = "safe"
    PERIMETER = "perimeter"


class RuleType(str, Enum):
    """Defines what behaviour triggers a violation for a zone."""

    NO_ENTRY = "no_entry"
    LOITERING = "loitering"
    INTRUSION = "intrusion"
    ABANDONED_OBJECT = "abandoned_object"
    CROWD_LIMIT = "crowd_limit"
    WRONG_DIRECTION = "wrong_direction"


@dataclass
class Zone:
    """Spatial area associated with one camera.

    Attributes:
        zone_id: Unique identifier (must be unique within a camera).
        camera_id: Which camera's coordinate space the polygon is defined in.
        name: Human-readable label.
        zone_type: Security classification.
        polygon: Ordered list of ``(x, y)`` vertex pairs in frame pixels.
        enabled: Master on/off switch.
        schedule_start: Time of day when the zone becomes active (``None`` → always).
        schedule_end: Time of day when the zone becomes inactive (``None`` → always).
        active_days: ISO weekday numbers that the schedule applies to (0=Mon … 6=Sun).
    """

    zone_id: str
    camera_id: str
    name: str
    zone_type: ZoneType
    polygon: List[Tuple[float, float]]
    enabled: bool = True
    schedule_start: Optional[dtime] = None
    schedule_end: Optional[dtime] = None
    active_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])


@dataclass
class Rule:
    """A behavioural rule attached to a zone.

    Attributes:
        rule_id: Unique identifier.
        zone_id: Zone this rule is attached to.
        rule_type: Which violation type to check.
        enabled: Master on/off switch.
        max_duration_seconds: Dwell/stationary threshold (loitering, abandoned object).
        max_persons: Crowd limit threshold.
        cooldown_seconds: Minimum gap between repeated alerts for the same rule.
    """

    rule_id: str
    zone_id: str
    rule_type: RuleType
    enabled: bool = True
    max_duration_seconds: float = 300.0
    max_persons: int = 10
    cooldown_seconds: float = 60.0


@dataclass
class ZoneViolation:
    """Emitted by the zone engine when a rule is violated.

    Attributes:
        rule_id: Which rule fired.
        zone_id: Zone where the violation occurred.
        zone_name: Human-readable zone name.
        camera_id: Source camera.
        track_id: Person (or object) track that triggered the violation.
        rule_type: Type of violation.
        timestamp: Unix timestamp of the violation.
        details: Human-readable description.
        confidence: 1.0 for deterministic zone checks; lower for edge cases.
        duration_in_zone: Seconds person has been inside the zone (loitering).
        persons_in_zone: Concurrent person count at violation time (crowd limit).
    """

    rule_id: str
    zone_id: str
    zone_name: str
    camera_id: str
    track_id: int
    rule_type: RuleType
    timestamp: float
    details: str
    confidence: float
    duration_in_zone: Optional[float] = None
    persons_in_zone: Optional[int] = None
