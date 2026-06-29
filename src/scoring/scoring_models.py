"""
Data structures for the anomaly scoring engine.
Represents the final scored event that drives alert decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AlertDecision(str, Enum):
    NO_ALERT = "no_alert"
    ALERT = "alert"
    SUPPRESSED = "suppressed"   # Would alert but cooldown/hysteresis blocked it
    ESCALATED = "escalated"     # Higher severity than normal alert


class SignalType(str, Enum):
    ACTION_CLASSIFICATION = "action_classification"
    ZONE_VIOLATION = "zone_violation"
    WEAPON_DETECTED = "weapon_detected"
    DANGEROUS_OBJECT = "dangerous_object"
    TIME_OF_DAY = "time_of_day"
    CROWD_DENSITY = "crowd_density"


@dataclass
class ScoringSignal:
    """A single contributing signal to the final score."""

    signal_type: SignalType
    source: str           # e.g. "action_classifier", "zone_engine", "yolo_detector"
    value: float          # 0.0–1.0 normalised signal strength
    weight: float         # Configured weight for this signal type
    weighted_value: float # value × weight
    details: str          # Human-readable: "Fighting detected (conf=0.82)"
    raw_data: Optional[dict] = None  # Original data for debugging


@dataclass
class ScoredEvent:
    """Final scored event — the output of the anomaly scoring engine."""

    event_id: str
    camera_id: str
    track_id: Optional[int]   # None for non-person events (abandoned object)
    timestamp: float

    # Scoring
    severity_score: float                     # 0.0–1.0 aggregated score
    contributing_signals: List[ScoringSignal] # What contributed to the score
    dominant_signal: SignalType               # Highest weighted contributor

    # Classification (from the dominant signal)
    event_category: str  # "violent", "suspicious", "urgent", "weapon"
    event_label: str     # "fighting", "loitering", "knife_detected"

    # Decision
    alert_decision: AlertDecision
    suppression_reason: Optional[str] = None  # "cooldown", "hysteresis", "below_threshold"

    # Spatial context
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    bbox: Optional[tuple] = None

    # References (filled later by clip capture and alert service)
    clip_path: Optional[str] = None
    alert_id: Optional[str] = None


@dataclass
class ScoringConfig:
    """Configurable scoring parameters — can be per-camera or global."""

    # Signal weights
    weight_action: float = 0.35
    weight_zone: float = 0.25
    weight_weapon: float = 0.30
    weight_time_of_day: float = 0.10

    # Alert thresholds
    alert_threshold: float = 0.55
    escalation_threshold: float = 0.85

    # Instant alert signals (bypass scoring formula)
    instant_alert_classes: List[str] = field(
        default_factory=lambda: ["knife", "gun", "rifle", "weapon"]
    )

    # Cooldown: suppress duplicate alerts for same track+camera
    cooldown_seconds: float = 30.0

    # Hysteresis: require N consecutive high windows before alerting
    hysteresis_count: int = 2

    # Time-of-day risk multiplier
    high_risk_hours: List[int] = field(
        default_factory=lambda: [22, 23, 0, 1, 2, 3, 4, 5]
    )
    time_risk_multiplier: float = 0.15

    # State cleanup
    stale_track_cleanup_seconds: float = 60.0
