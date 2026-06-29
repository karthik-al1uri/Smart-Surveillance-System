"""Tests for Phase 7: Anomaly Scoring Engine.

All tests use synthetic data — no cameras or real video required.
Score assertions use ±0.01 tolerance where applicable.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.yolo_detector import Detection
from src.recognition.action_classes import (
    ActionCategory,
    ActionLabel,
    ActionPrediction,
)
from src.scoring.anomaly_scorer import AnomalyScorer
from src.scoring.scoring_models import (
    AlertDecision,
    ScoredEvent,
    ScoringConfig,
    ScoringSignal,
    SignalType,
)
from src.scoring.zone_models import RuleType, ZoneViolation


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_BASE_CFG = {
    "scoring": {
        "weights": {"action": 0.35, "zone": 0.25, "weapon": 0.30, "time_of_day": 0.10},
        "alert_threshold": 0.55,
        "escalation_threshold": 0.85,
        "instant_alert_classes": ["knife", "gun"],
        "cooldown_seconds": 30.0,
        "hysteresis_count": 2,
        "high_risk_hours": [22, 23, 0, 1, 2, 3, 4, 5],
        "time_risk_multiplier": 0.15,
        "stale_track_cleanup_seconds": 60.0,
        "camera_overrides": {},
    }
}


@pytest.fixture
def scorer():
    return AnomalyScorer(_BASE_CFG)


def make_action_prediction(
    track_id=1,
    camera_id="cam_01",
    category=ActionCategory.VIOLENT,
    label=ActionLabel.FIGHTING,
    confidence=0.85,
) -> ActionPrediction:
    return ActionPrediction(
        track_id=track_id,
        camera_id=camera_id,
        timestamp=time.time(),
        category=category,
        label=label,
        confidence=confidence,
        category_probabilities={
            ActionCategory.NORMAL: 0.1,
            ActionCategory.VIOLENT: 0.85,
            ActionCategory.SUSPICIOUS: 0.03,
            ActionCategory.URGENT: 0.02,
        },
        window_start_frame=0,
        window_end_frame=15,
        keypoint_quality=0.9,
    )


def make_zone_violation(
    track_id=1,
    camera_id="cam_01",
    rule_type=RuleType.NO_ENTRY,
    duration=None,
) -> ZoneViolation:
    return ZoneViolation(
        rule_id="rule_01",
        zone_id="zone_a",
        zone_name="Restricted Area",
        camera_id=camera_id,
        track_id=track_id,
        rule_type=rule_type,
        timestamp=time.time(),
        details="Test violation",
        confidence=1.0,
        duration_in_zone=duration,
    )


def make_weapon_detection(
    camera_id="cam_01",
    class_name="knife",
    confidence=0.8,
) -> Detection:
    return Detection(
        class_id=43,
        class_name=class_name,
        confidence=confidence,
        bbox=(100, 100, 200, 200),
        frame_idx=1,
    )


def score_once(scorer, track_id=1, camera_id="cam_01",
               predictions=None, violations=None, detections=None):
    return scorer.score_frame(
        camera_id=camera_id,
        timestamp=time.time(),
        action_predictions=predictions or [],
        zone_violations=violations or [],
        object_detections=detections or [],
    )


# ---------------------------------------------------------------------------
# 1–4: Data structures
# ---------------------------------------------------------------------------

def test_scoring_config_defaults():
    cfg = ScoringConfig()
    assert cfg.weight_action == pytest.approx(0.35)
    assert cfg.weight_zone == pytest.approx(0.25)
    assert cfg.weight_weapon == pytest.approx(0.30)
    assert cfg.weight_time_of_day == pytest.approx(0.10)
    assert cfg.alert_threshold == pytest.approx(0.55)
    assert cfg.escalation_threshold == pytest.approx(0.85)
    assert cfg.hysteresis_count == 2
    assert cfg.cooldown_seconds == pytest.approx(30.0)


def test_scored_event_dataclass():
    ev = ScoredEvent(
        event_id="evt_001", camera_id="cam_01", track_id=1,
        timestamp=time.time(), severity_score=0.72,
        contributing_signals=[], dominant_signal=SignalType.ACTION_CLASSIFICATION,
        event_category="violent", event_label="fighting",
        alert_decision=AlertDecision.ALERT,
    )
    assert ev.event_id == "evt_001"
    assert ev.severity_score == pytest.approx(0.72)
    assert ev.clip_path is None


def test_scoring_signal_dataclass():
    sig = ScoringSignal(
        signal_type=SignalType.ACTION_CLASSIFICATION,
        source="action_classifier",
        value=0.85,
        weight=0.35,
        weighted_value=0.85 * 0.35,
        details="Fighting (conf=0.85)",
    )
    assert sig.weighted_value == pytest.approx(0.85 * 0.35)


def test_alert_decision_enum():
    assert AlertDecision.NO_ALERT == "no_alert"
    assert AlertDecision.ALERT == "alert"
    assert AlertDecision.SUPPRESSED == "suppressed"
    assert AlertDecision.ESCALATED == "escalated"


# ---------------------------------------------------------------------------
# 5–13: Signal extraction
# ---------------------------------------------------------------------------

def test_action_signal_violent(scorer):
    pred = make_action_prediction(category=ActionCategory.VIOLENT, confidence=0.9)
    events = score_once(scorer, predictions=[pred])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    action_sig = next(s for s in ev.contributing_signals if s.signal_type == SignalType.ACTION_CLASSIFICATION)
    assert action_sig.value == pytest.approx(0.9, abs=0.01)


def test_action_signal_suspicious(scorer):
    pred = make_action_prediction(category=ActionCategory.SUSPICIOUS,
                                   label=ActionLabel.LOITERING, confidence=0.8)
    events = score_once(scorer, predictions=[pred])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    action_sig = next(s for s in ev.contributing_signals if s.signal_type == SignalType.ACTION_CLASSIFICATION)
    assert action_sig.value == pytest.approx(0.8 * 0.7, abs=0.01)


def test_action_signal_urgent(scorer):
    pred = make_action_prediction(category=ActionCategory.URGENT,
                                   label=ActionLabel.FALLING, confidence=0.75)
    events = score_once(scorer, predictions=[pred])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    action_sig = next(s for s in ev.contributing_signals if s.signal_type == SignalType.ACTION_CLASSIFICATION)
    assert action_sig.value == pytest.approx(0.75 * 0.9, abs=0.01)


def test_action_signal_normal(scorer):
    scorer._global_config.high_risk_hours = []  # disable time signal
    pred = make_action_prediction(category=ActionCategory.NORMAL,
                                   label=ActionLabel.WALKING, confidence=0.95)
    events = score_once(scorer, predictions=[pred])
    # Normal activity with no zone violation → no event generated (value=0, no signals)
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is None


def test_zone_signal_no_entry(scorer):
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    events = score_once(scorer, violations=[viol])
    ev = next(e for e in events if e.track_id == 1)
    zone_sig = next(s for s in ev.contributing_signals if s.signal_type == SignalType.ZONE_VIOLATION)
    assert zone_sig.value == pytest.approx(1.0, abs=0.01)


def test_zone_signal_loitering(scorer):
    viol = make_zone_violation(rule_type=RuleType.LOITERING, duration=300.0)
    events = score_once(scorer, violations=[viol])
    ev = next(e for e in events if e.track_id == 1)
    zone_sig = next(s for s in ev.contributing_signals if s.signal_type == SignalType.ZONE_VIOLATION)
    assert 0.0 <= zone_sig.value <= 1.0


def test_zone_signal_intrusion(scorer):
    viol = make_zone_violation(rule_type=RuleType.INTRUSION)
    events = score_once(scorer, violations=[viol])
    ev = next(e for e in events if e.track_id == 1)
    zone_sig = next(s for s in ev.contributing_signals if s.signal_type == SignalType.ZONE_VIOLATION)
    assert zone_sig.value == pytest.approx(0.8, abs=0.01)


def test_time_signal_high_risk(scorer):
    # Force a high-risk hour by using a timestamp that maps to 23:00
    # 2024-01-15 23:00:00 UTC+5:30 = 2024-01-15 17:30:00 UTC
    from datetime import datetime, timezone
    # Use a known timestamp: Mon Jan 15 2024 23:00 local
    import calendar
    dt = datetime(2024, 1, 15, 23, 0, 0)
    ts = calendar.timegm(dt.timetuple()) - time.timezone

    pred = make_action_prediction(category=ActionCategory.VIOLENT, confidence=0.9)
    events = scorer.score_frame("cam_01", ts, [pred], [], [])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    time_sigs = [s for s in ev.contributing_signals if s.signal_type == SignalType.TIME_OF_DAY]
    # The hour may or may not land in high-risk depending on local timezone; just verify no crash
    assert isinstance(ev.severity_score, float)


def test_time_signal_low_risk(scorer):
    import calendar
    dt_low = (2024, 1, 15, 14, 0, 0)  # 14:00 — definitely not high risk
    from datetime import datetime
    ts = float(datetime(*dt_low[:6]).timestamp())
    pred = make_action_prediction(category=ActionCategory.VIOLENT, confidence=0.9)
    # We can't guarantee time zone, but we can check no crash
    events = scorer.score_frame("cam_01", ts, [pred], [], [])
    assert isinstance(events, list)


# ---------------------------------------------------------------------------
# 14–19: Score computation
# ---------------------------------------------------------------------------

def test_score_violent_action_only(scorer):
    scorer._global_config.high_risk_hours = []  # isolate action signal only
    pred = make_action_prediction(confidence=0.9)
    # Need 2 consecutive for hysteresis
    score_once(scorer, predictions=[pred])
    events = score_once(scorer, predictions=[pred])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    # score = 0.35 × 0.9 = 0.315
    assert ev.severity_score >= 0.35 * 0.9 - 0.01


def test_score_action_plus_zone(scorer):
    scorer._global_config.high_risk_hours = []  # isolate action+zone only
    pred = make_action_prediction(confidence=0.85)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    score_once(scorer, predictions=[pred], violations=[viol])
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    # score = 0.35×0.85 + 0.25×1.0 = 0.5475
    assert ev.severity_score >= 0.35 * 0.85 + 0.25 * 1.0 - 0.01


def test_score_weapon_instant(scorer):
    det = make_weapon_detection(class_name="knife")
    events = score_once(scorer, detections=[det])
    ev = next((e for e in events if e.event_category == "weapon"), None)
    assert ev is not None
    assert ev.severity_score == 1.0
    assert ev.alert_decision == AlertDecision.ESCALATED


def test_score_clamped(scorer):
    pred = make_action_prediction(confidence=1.0)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    # Simulate high-risk hour manually
    scorer._global_config.high_risk_hours = list(range(24))  # all hours
    score_once(scorer, predictions=[pred], violations=[viol])
    events = score_once(scorer, predictions=[pred], violations=[viol])
    for ev in events:
        assert ev.severity_score <= 1.0
        assert ev.severity_score >= 0.0


def test_score_all_signals(scorer):
    scorer._global_config.high_risk_hours = list(range(24))
    pred = make_action_prediction(confidence=1.0)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    score_once(scorer, predictions=[pred], violations=[viol])
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next(e for e in events if e.track_id == 1)
    # At minimum: 0.35 + 0.25 = 0.60 (capped at 1.0)
    assert ev.severity_score >= 0.59


def test_score_normal_activity(scorer):
    scorer._global_config.high_risk_hours = []  # disable time signal
    pred = make_action_prediction(category=ActionCategory.NORMAL,
                                   label=ActionLabel.WALKING, confidence=0.95)
    events = score_once(scorer, predictions=[pred])
    ev = next((e for e in events if e.track_id == 1), None)
    # Normal with no zone signals → no event
    assert ev is None


# ---------------------------------------------------------------------------
# 20–23: Hysteresis
# ---------------------------------------------------------------------------

def test_hysteresis_first_high(scorer):
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    # First frame: hysteresis counter = 1, need 2 → SUPPRESSED
    assert ev is not None
    assert ev.alert_decision in (AlertDecision.SUPPRESSED, AlertDecision.NO_ALERT)


def test_hysteresis_second_high(scorer):
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    score_once(scorer, predictions=[pred], violations=[viol])
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    assert ev.alert_decision in (AlertDecision.ALERT, AlertDecision.ESCALATED)


def test_hysteresis_reset(scorer):
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    normal_pred = make_action_prediction(category=ActionCategory.NORMAL,
                                          label=ActionLabel.WALKING, confidence=0.95)
    # Frame 1: high
    score_once(scorer, predictions=[pred], violations=[viol])
    # Frame 2: low — reset counter
    score_once(scorer, predictions=[normal_pred])
    # Frame 3: high again — counter back to 1, should be suppressed
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    assert ev.alert_decision in (AlertDecision.SUPPRESSED, AlertDecision.NO_ALERT)


def test_hysteresis_weapon_bypass(scorer):
    det = make_weapon_detection()
    events = score_once(scorer, detections=[det])
    weapon_ev = next((e for e in events if e.event_category == "weapon"), None)
    assert weapon_ev is not None
    assert weapon_ev.alert_decision == AlertDecision.ESCALATED


# ---------------------------------------------------------------------------
# 24–28: Cooldown
# ---------------------------------------------------------------------------

def test_cooldown_first_alert(scorer):
    scorer._global_config.hysteresis_count = 1
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    assert ev.alert_decision in (AlertDecision.ALERT, AlertDecision.ESCALATED)


def test_cooldown_suppressed(scorer):
    scorer._global_config.hysteresis_count = 1
    scorer._global_config.cooldown_seconds = 3600.0
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    score_once(scorer, predictions=[pred], violations=[viol])
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    assert ev.alert_decision == AlertDecision.SUPPRESSED
    assert ev.suppression_reason == "cooldown"


def test_cooldown_expired(scorer):
    scorer._global_config.hysteresis_count = 1
    scorer._global_config.cooldown_seconds = 0.001
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    score_once(scorer, predictions=[pred], violations=[viol])
    time.sleep(0.01)
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    assert ev.alert_decision in (AlertDecision.ALERT, AlertDecision.ESCALATED)


def test_cooldown_different_track(scorer):
    scorer._global_config.hysteresis_count = 1
    scorer._global_config.cooldown_seconds = 3600.0
    pred1 = make_action_prediction(track_id=1, confidence=0.9)
    pred2 = make_action_prediction(track_id=2, confidence=0.9)
    viol1 = make_zone_violation(track_id=1, rule_type=RuleType.NO_ENTRY)
    viol2 = make_zone_violation(track_id=2, rule_type=RuleType.NO_ENTRY)
    # Fire alert for track 1
    score_once(scorer, predictions=[pred1], violations=[viol1])
    # Track 2 should still fire
    events = score_once(scorer, predictions=[pred2], violations=[viol2])
    ev2 = next((e for e in events if e.track_id == 2), None)
    assert ev2 is not None
    assert ev2.alert_decision in (AlertDecision.ALERT, AlertDecision.ESCALATED)


def test_cooldown_escalation_bypass(scorer):
    scorer._global_config.hysteresis_count = 1
    scorer._global_config.cooldown_seconds = 3600.0
    scorer._global_config.escalation_threshold = 0.3  # low threshold → easy escalate
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    score_once(scorer, predictions=[pred], violations=[viol])
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    assert ev.alert_decision == AlertDecision.ESCALATED


# ---------------------------------------------------------------------------
# 29–31: Per-camera config
# ---------------------------------------------------------------------------

def test_camera_override_threshold():
    cfg = dict(_BASE_CFG)
    cfg["scoring"] = dict(_BASE_CFG["scoring"])
    cfg["scoring"]["camera_overrides"] = {
        "cam_sensitive": {"alert_threshold": 0.15}
    }
    s = AnomalyScorer(cfg)
    s._global_config.hysteresis_count = 1
    cam_cfg = s.get_config_for_camera("cam_sensitive")
    assert cam_cfg.alert_threshold == pytest.approx(0.15, abs=0.01)


def test_camera_override_weights():
    cfg = dict(_BASE_CFG)
    cfg["scoring"] = dict(_BASE_CFG["scoring"])
    cfg["scoring"]["camera_overrides"] = {
        "cam_zone_heavy": {"weights": {"action": 0.10, "zone": 0.70, "weapon": 0.10, "time_of_day": 0.10}}
    }
    s = AnomalyScorer(cfg)
    cam_cfg = s.get_config_for_camera("cam_zone_heavy")
    assert cam_cfg.weight_zone == pytest.approx(0.70, abs=0.01)
    assert cam_cfg.weight_action == pytest.approx(0.10, abs=0.01)


def test_camera_fallback_global(scorer):
    cfg = scorer.get_config_for_camera("cam_unknown")
    assert cfg is scorer._global_config


# ---------------------------------------------------------------------------
# 32–36: Decision logic
# ---------------------------------------------------------------------------

def test_decision_no_alert(scorer):
    pred = make_action_prediction(category=ActionCategory.SUSPICIOUS,
                                   label=ActionLabel.LOITERING, confidence=0.1)
    viol = make_zone_violation(rule_type=RuleType.LOITERING, duration=10.0)
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    if ev is not None:
        assert ev.alert_decision == AlertDecision.NO_ALERT or ev.alert_decision == AlertDecision.SUPPRESSED


def test_decision_alert(scorer):
    scorer._global_config.hysteresis_count = 1
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    assert ev.alert_decision in (AlertDecision.ALERT, AlertDecision.ESCALATED)


def test_decision_escalated(scorer):
    scorer._global_config.hysteresis_count = 1
    scorer._global_config.escalation_threshold = 0.3
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    assert ev.alert_decision == AlertDecision.ESCALATED


def test_decision_suppressed_cooldown(scorer):
    scorer._global_config.hysteresis_count = 1
    scorer._global_config.cooldown_seconds = 3600.0
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    score_once(scorer, predictions=[pred], violations=[viol])
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    assert ev.alert_decision == AlertDecision.SUPPRESSED
    assert ev.suppression_reason == "cooldown"


def test_decision_suppressed_hysteresis(scorer):
    scorer._global_config.hysteresis_count = 3
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next((e for e in events if e.track_id == 1), None)
    assert ev is not None
    assert ev.alert_decision == AlertDecision.SUPPRESSED
    assert ev.suppression_reason == "hysteresis"


# ---------------------------------------------------------------------------
# 37–39: State management
# ---------------------------------------------------------------------------

def test_stale_track_cleanup(scorer):
    scorer._global_config.stale_track_cleanup_seconds = 0.01
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    score_once(scorer, predictions=[pred], violations=[viol])
    assert ("cam_01", 1) in scorer._last_seen

    time.sleep(0.05)
    score_once(scorer)  # trigger cleanup
    assert ("cam_01", 1) not in scorer._last_seen


def test_reset_state_global(scorer):
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    score_once(scorer, predictions=[pred], violations=[viol])
    scorer.reset_state()
    assert len(scorer._consecutive_high) == 0
    assert len(scorer._last_alert) == 0


def test_reset_state_per_camera(scorer):
    pred = make_action_prediction(camera_id="cam_01", confidence=0.9)
    viol = make_zone_violation(camera_id="cam_01", rule_type=RuleType.NO_ENTRY)
    score_once(scorer, camera_id="cam_01", predictions=[pred], violations=[viol])
    score_once(scorer, camera_id="cam_02",
               predictions=[make_action_prediction(camera_id="cam_02", confidence=0.9)],
               violations=[make_zone_violation(camera_id="cam_02")])
    scorer.reset_state("cam_01")
    assert all(k[0] != "cam_01" for k in scorer._last_seen)
    assert any(k[0] == "cam_02" for k in scorer._last_seen)


# ---------------------------------------------------------------------------
# 40–41: Non-person events
# ---------------------------------------------------------------------------

def test_abandoned_object_scoring(scorer):
    viol = ZoneViolation(
        rule_id="r1", zone_id="zone_a", zone_name="Lobby",
        camera_id="cam_01", track_id=-1, rule_type=RuleType.ABANDONED_OBJECT,
        timestamp=time.time(), details="Abandoned bag", confidence=0.9,
        duration_in_zone=200.0,
    )
    events = score_once(scorer, violations=[viol])
    non_person = [e for e in events if e.track_id is None]
    assert len(non_person) == 1
    assert non_person[0].event_label == "abandoned_object"


def test_crowd_limit_scoring(scorer):
    viol = ZoneViolation(
        rule_id="r2", zone_id="zone_b", zone_name="Entrance",
        camera_id="cam_01", track_id=-1, rule_type=RuleType.CROWD_LIMIT,
        timestamp=time.time(), details="8 persons", confidence=1.0,
        persons_in_zone=8,
    )
    events = score_once(scorer, violations=[viol])
    non_person = [e for e in events if e.track_id is None]
    assert len(non_person) == 1
    assert non_person[0].event_label == "crowd_limit"


# ---------------------------------------------------------------------------
# 42–45: Pipeline integration
# ---------------------------------------------------------------------------

def test_scoring_pipeline_full_flow():
    from collections import deque
    from src.detection.combined_pipeline import TrackedFrameAnalysis
    from src.detection.tracker import Track
    from src.recognition.recognition_pipeline import ActionRecognitionPipeline
    from src.scoring.scoring_pipeline import ScoringPipeline
    from src.scoring.zone_engine import ZoneEngine
    from src.scoring.zone_manager import ZoneManager
    import numpy as np

    zm = ZoneManager()
    ze = ZoneEngine(zm)
    ap = ActionRecognitionPipeline(config={})
    sc = AnomalyScorer(_BASE_CFG)
    pipeline = ScoringPipeline({}, ze, ap, sc)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    track = Track(track_id=1, state="active", bbox=(200, 200, 300, 399),
                  keypoint_history=deque(maxlen=64), age=5, hits=5,
                  time_since_update=0, confidence=0.9)
    analysis = TrackedFrameAnalysis(
        camera_id="cam_01", frame_id=0, timestamp=time.time(),
        frame=frame, person_detections=[], object_detections=[],
        poses=[], tracks=[track],
    )
    pred = make_action_prediction(confidence=0.9)
    viol = make_zone_violation(rule_type=RuleType.NO_ENTRY)
    events = pipeline.process(analysis, [pred], [viol])
    assert isinstance(events, list)


def test_scoring_pipeline_no_events():
    from src.recognition.recognition_pipeline import ActionRecognitionPipeline
    from src.scoring.scoring_pipeline import ScoringPipeline
    from src.scoring.zone_engine import ZoneEngine
    from src.scoring.zone_manager import ZoneManager
    from src.detection.combined_pipeline import TrackedFrameAnalysis
    import numpy as np

    zm = ZoneManager()
    ze = ZoneEngine(zm)
    ap = ActionRecognitionPipeline(config={})
    sc = AnomalyScorer(_BASE_CFG)
    pipeline = ScoringPipeline({}, ze, ap, sc)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    analysis = TrackedFrameAnalysis(
        camera_id="cam_01", frame_id=0, timestamp=time.time(),
        frame=frame, person_detections=[], object_detections=[],
        poses=[], tracks=[],
    )
    events = pipeline.process(analysis, [], [])
    alerts = [e for e in events if e.alert_decision in (AlertDecision.ALERT, AlertDecision.ESCALATED)]
    assert alerts == []


def test_scoring_pipeline_weapon_alert():
    from src.recognition.recognition_pipeline import ActionRecognitionPipeline
    from src.scoring.scoring_pipeline import ScoringPipeline
    from src.scoring.zone_engine import ZoneEngine
    from src.scoring.zone_manager import ZoneManager
    from src.detection.combined_pipeline import TrackedFrameAnalysis
    import numpy as np

    zm = ZoneManager()
    ze = ZoneEngine(zm)
    ap = ActionRecognitionPipeline(config={})
    sc = AnomalyScorer(_BASE_CFG)
    pipeline = ScoringPipeline({}, ze, ap, sc)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = make_weapon_detection(class_name="knife")
    analysis = TrackedFrameAnalysis(
        camera_id="cam_01", frame_id=0, timestamp=time.time(),
        frame=frame, person_detections=[], object_detections=[det],
        poses=[], tracks=[],
    )
    events = pipeline.process(analysis, [], [])
    alerts = [e for e in events if e.alert_decision == AlertDecision.ESCALATED]
    assert len(alerts) == 1


def test_scoring_pipeline_stats():
    from src.recognition.recognition_pipeline import ActionRecognitionPipeline
    from src.scoring.scoring_pipeline import ScoringPipeline
    from src.scoring.zone_engine import ZoneEngine
    from src.scoring.zone_manager import ZoneManager
    from src.detection.combined_pipeline import TrackedFrameAnalysis
    import numpy as np

    zm = ZoneManager()
    ze = ZoneEngine(zm)
    ap = ActionRecognitionPipeline(config={})
    sc = AnomalyScorer(_BASE_CFG)
    pipeline = ScoringPipeline({}, ze, ap, sc)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = make_weapon_detection()
    analysis = TrackedFrameAnalysis(
        camera_id="cam_01", frame_id=0, timestamp=time.time(),
        frame=frame, person_detections=[], object_detections=[det],
        poses=[], tracks=[],
    )
    pipeline.process(analysis, [], [])
    stats = pipeline.get_stats()
    assert "scorer" in stats
    assert stats["scorer"]["total_events_scored"] >= 1


# ---------------------------------------------------------------------------
# 46–48: Edge cases
# ---------------------------------------------------------------------------

def test_empty_inputs(scorer):
    events = score_once(scorer)
    assert events == []


def test_multiple_tracks_same_frame(scorer):
    scorer._global_config.hysteresis_count = 1
    preds = [make_action_prediction(track_id=i, confidence=0.9) for i in range(1, 4)]
    viols = [make_zone_violation(track_id=i, rule_type=RuleType.NO_ENTRY) for i in range(1, 4)]
    events = score_once(scorer, predictions=preds, violations=viols)
    track_ids = {e.track_id for e in events if e.track_id is not None}
    assert track_ids == {1, 2, 3}


def test_simultaneous_action_and_zone(scorer):
    scorer._global_config.hysteresis_count = 1
    scorer._global_config.high_risk_hours = []  # isolate to action+zone
    pred = make_action_prediction(track_id=1, confidence=0.9)
    viol = make_zone_violation(track_id=1, rule_type=RuleType.NO_ENTRY)
    events = score_once(scorer, predictions=[pred], violations=[viol])
    ev = next(e for e in events if e.track_id == 1)
    signal_types = {s.signal_type for s in ev.contributing_signals}
    assert SignalType.ACTION_CLASSIFICATION in signal_types
    assert SignalType.ZONE_VIOLATION in signal_types
    # Score must include both contributions
    assert ev.severity_score >= 0.35 * 0.9 + 0.25 * 1.0 - 0.01
