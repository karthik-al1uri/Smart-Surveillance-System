"""Tests for Phase 6: Zone & Rule Engine.

All tests use synthetic data — no cameras or video required.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from datetime import datetime, time as dtime
from pathlib import Path
from typing import List

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.combined_pipeline import TrackedFrameAnalysis
from src.detection.tracker import Track
from src.detection.yolo_detector import Detection
from src.scoring.zone_engine import ZoneEngine
from src.scoring.zone_manager import ZoneManager
from src.scoring.zone_models import Rule, RuleType, Zone, ZoneType, ZoneViolation
from src.common.visualization import draw_violations, draw_zones


# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------

SQUARE_ZONE = Zone(
    zone_id="test_zone",
    camera_id="test_cam",
    name="Test Zone",
    zone_type=ZoneType.RESTRICTED,
    polygon=[(100, 100), (400, 100), (400, 400), (100, 400)],
)

NO_ENTRY_RULE = Rule(
    rule_id="rule_no_entry",
    zone_id="test_zone",
    rule_type=RuleType.NO_ENTRY,
    cooldown_seconds=0.0,
)

LOITERING_RULE = Rule(
    rule_id="rule_loiter",
    zone_id="test_zone",
    rule_type=RuleType.LOITERING,
    max_duration_seconds=300.0,
    cooldown_seconds=0.0,
)

INTRUSION_RULE = Rule(
    rule_id="rule_intrusion",
    zone_id="test_zone",
    rule_type=RuleType.INTRUSION,
    cooldown_seconds=0.0,
)

CROWD_RULE = Rule(
    rule_id="rule_crowd",
    zone_id="test_zone",
    rule_type=RuleType.CROWD_LIMIT,
    max_persons=5,
    cooldown_seconds=0.0,
)


def _make_zone_manager(zone=None, rules=None):
    zm = ZoneManager()
    z = zone or SQUARE_ZONE
    zm.add_zone(z)
    for r in (rules or []):
        zm.add_rule(r)
    return zm


def _make_engine(zone=None, rules=None):
    zm = _make_zone_manager(zone, rules)
    return ZoneEngine(zm)


def _make_track(track_id=1, bbox=(250, 250, 300, 380), state="active"):
    return Track(
        track_id=track_id,
        state=state,
        bbox=bbox,
        keypoint_history=deque(maxlen=64),
        age=5,
        hits=5,
        time_since_update=0,
        confidence=0.9,
    )


def _make_analysis(tracks=None, camera_id="test_cam", frame_id=0, object_detections=None):
    return TrackedFrameAnalysis(
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp=time.time(),
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        person_detections=[],
        object_detections=object_detections or [],
        poses=[],
        tracks=tracks or [],
    )


# ---------------------------------------------------------------------------
# 1–5: Data structure sanity
# ---------------------------------------------------------------------------

def test_zone_dataclass():
    z = SQUARE_ZONE
    assert z.zone_id == "test_zone"
    assert z.zone_type == ZoneType.RESTRICTED
    assert len(z.polygon) == 4
    assert z.enabled is True


def test_rule_dataclass():
    r = NO_ENTRY_RULE
    assert r.rule_id == "rule_no_entry"
    assert r.rule_type == RuleType.NO_ENTRY
    assert r.enabled is True


def test_zone_violation_dataclass():
    vio = ZoneViolation(
        rule_id="r1", zone_id="z1", zone_name="Zone A", camera_id="cam0",
        track_id=3, rule_type=RuleType.NO_ENTRY, timestamp=time.time(),
        details="Test", confidence=1.0,
    )
    assert vio.track_id == 3
    assert vio.rule_type == RuleType.NO_ENTRY


def test_zone_type_enum():
    assert ZoneType.RESTRICTED == "restricted"
    assert ZoneType.MONITORED == "monitored"
    assert ZoneType.SAFE == "safe"
    assert ZoneType.PERIMETER == "perimeter"


def test_rule_type_enum():
    assert RuleType.NO_ENTRY == "no_entry"
    assert RuleType.LOITERING == "loitering"
    assert RuleType.INTRUSION == "intrusion"
    assert RuleType.ABANDONED_OBJECT == "abandoned_object"
    assert RuleType.CROWD_LIMIT == "crowd_limit"


# ---------------------------------------------------------------------------
# 6–10: ZoneManager CRUD
# ---------------------------------------------------------------------------

def test_zone_manager_add():
    zm = ZoneManager()
    zm.add_zone(SQUARE_ZONE)
    assert zm.get_zone("test_zone") is SQUARE_ZONE


def test_zone_manager_remove():
    zm = ZoneManager()
    zm.add_zone(SQUARE_ZONE)
    zm.remove_zone("test_zone")
    assert zm.get_zone("test_zone") is None


def test_zone_manager_update():
    zm = ZoneManager()
    zm.add_zone(SQUARE_ZONE)
    zm.update_zone("test_zone", {"name": "Updated Name"})
    assert zm.get_zone("test_zone").name == "Updated Name"


def test_zone_manager_camera_filter():
    zm = ZoneManager()
    z1 = Zone("z1", "cam_a", "Zone 1", ZoneType.SAFE, [(0,0),(10,0),(10,10),(0,10)])
    z2 = Zone("z2", "cam_b", "Zone 2", ZoneType.SAFE, [(0,0),(10,0),(10,10),(0,10)])
    zm.add_zone(z1)
    zm.add_zone(z2)
    assert len(zm.get_zones_for_camera("cam_a")) == 1
    assert zm.get_zones_for_camera("cam_a")[0].zone_id == "z1"


def test_zone_manager_load_config():
    config = {
        "zones": [
            {
                "zone_id": "cfg_zone",
                "camera_id": "cam01",
                "name": "Config Zone",
                "zone_type": "monitored",
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "rules": [
                    {"rule_id": "r_cfg", "rule_type": "no_entry"}
                ],
            }
        ]
    }
    zm = ZoneManager(config=config)
    assert zm.get_zone("cfg_zone") is not None
    rules = zm.get_rules_for_zone("cfg_zone")
    assert len(rules) == 1
    assert rules[0].rule_type == RuleType.NO_ENTRY


# ---------------------------------------------------------------------------
# 11–14: Schedule logic
# ---------------------------------------------------------------------------

def test_zone_schedule_active():
    z = Zone("z", "c", "N", ZoneType.RESTRICTED, [(0,0),(1,0),(1,1),(0,1)],
             schedule_start=dtime(22, 0), schedule_end=dtime(6, 0))
    zm = ZoneManager()
    zm.add_zone(z)
    assert zm.is_zone_active(z, datetime(2024, 1, 15, 23, 0))


def test_zone_schedule_inactive():
    z = Zone("z", "c", "N", ZoneType.RESTRICTED, [(0,0),(1,0),(1,1),(0,1)],
             schedule_start=dtime(22, 0), schedule_end=dtime(6, 0))
    zm = ZoneManager()
    zm.add_zone(z)
    assert not zm.is_zone_active(z, datetime(2024, 1, 15, 14, 0))


def test_zone_schedule_overnight():
    z = Zone("z", "c", "N", ZoneType.RESTRICTED, [(0,0),(1,0),(1,1),(0,1)],
             schedule_start=dtime(22, 0), schedule_end=dtime(6, 0))
    zm = ZoneManager()
    zm.add_zone(z)
    assert zm.is_zone_active(z, datetime(2024, 1, 15, 2, 30))


def test_zone_schedule_day_filter():
    z = Zone("z", "c", "N", ZoneType.RESTRICTED, [(0,0),(1,0),(1,1),(0,1)],
             schedule_start=dtime(22, 0), schedule_end=dtime(6, 0),
             active_days=[0, 1, 2, 3, 4])
    zm = ZoneManager()
    zm.add_zone(z)
    saturday = datetime(2024, 1, 20, 23, 0)
    assert saturday.weekday() == 5
    assert not zm.is_zone_active(z, saturday)


# ---------------------------------------------------------------------------
# 15–18: Point-in-polygon
# ---------------------------------------------------------------------------

def test_point_in_zone_inside():
    engine = _make_engine()
    assert engine.is_point_in_zone((250, 250), SQUARE_ZONE)


def test_point_in_zone_outside():
    engine = _make_engine()
    assert not engine.is_point_in_zone((50, 50), SQUARE_ZONE)


def test_point_in_zone_on_edge():
    engine = _make_engine()
    result = engine.is_point_in_zone((100, 250), SQUARE_ZONE)
    assert isinstance(result, bool)


def test_point_in_zone_complex_polygon():
    l_shape = Zone(
        "l", "c", "L", ZoneType.MONITORED,
        [(0,0),(200,0),(200,100),(100,100),(100,200),(0,200)],
    )
    zm = ZoneManager()
    zm.add_zone(l_shape)
    engine = ZoneEngine(zm)
    assert engine.is_point_in_zone((50, 50), l_shape)
    assert not engine.is_point_in_zone((150, 150), l_shape)


# ---------------------------------------------------------------------------
# 19–20: Person position extraction
# ---------------------------------------------------------------------------

def test_person_position_bottom_center():
    engine = _make_engine()
    pos = engine.get_person_position((100, 200, 300, 500), "bottom_center")
    assert pos == (200.0, 500.0)


def test_person_position_centroid():
    engine = _make_engine()
    pos = engine.get_person_position((100, 200, 300, 500), "centroid")
    assert pos == (200.0, 350.0)


# ---------------------------------------------------------------------------
# 21–23: NO_ENTRY rule
# ---------------------------------------------------------------------------

def test_no_entry_violation():
    engine = _make_engine(rules=[NO_ENTRY_RULE])
    track = _make_track(bbox=(200, 200, 300, 399))
    analysis = _make_analysis(tracks=[track])
    viols = engine.evaluate(analysis)
    assert len(viols) == 1
    assert viols[0].rule_type == RuleType.NO_ENTRY


def test_no_entry_inactive_schedule():
    z = Zone(
        "test_zone", "test_cam", "Test Zone", ZoneType.RESTRICTED,
        [(100,100),(400,100),(400,400),(100,400)],
        schedule_start=dtime(22, 0), schedule_end=dtime(23, 0),
    )
    zm = _make_zone_manager(z, [NO_ENTRY_RULE])
    engine = ZoneEngine(zm)
    track = _make_track(bbox=(200, 200, 300, 399))
    analysis = _make_analysis(tracks=[track])
    viols = engine.evaluate(analysis)
    assert viols == []


def test_no_entry_cooldown():
    rule = Rule("rule_cd", "test_zone", RuleType.NO_ENTRY, cooldown_seconds=3600.0)
    engine = _make_engine(rules=[rule])
    track = _make_track(bbox=(200, 200, 300, 399))
    analysis = _make_analysis(tracks=[track])
    v1 = engine.evaluate(analysis)
    v2 = engine.evaluate(analysis)
    assert len(v1) == 1
    assert len(v2) == 0


# ---------------------------------------------------------------------------
# 24–26: LOITERING rule
# ---------------------------------------------------------------------------

def test_loitering_under_threshold():
    engine = _make_engine(rules=[LOITERING_RULE])
    track = _make_track(bbox=(200, 200, 300, 399))
    analysis = _make_analysis(tracks=[track])
    engine._dwell_times[(1, "test_zone")] = time.time() - 60
    viols = engine.evaluate(analysis)
    assert all(v.rule_type != RuleType.LOITERING for v in viols)


def test_loitering_over_threshold():
    engine = _make_engine(rules=[LOITERING_RULE])
    track = _make_track(bbox=(200, 200, 300, 399))
    analysis = _make_analysis(tracks=[track])
    engine._dwell_times[(1, "test_zone")] = time.time() - 400
    viols = engine.evaluate(analysis)
    loiter_viols = [v for v in viols if v.rule_type == RuleType.LOITERING]
    assert len(loiter_viols) == 1
    assert loiter_viols[0].duration_in_zone >= 400


def test_loitering_person_leaves():
    engine = _make_engine(rules=[LOITERING_RULE])
    track = _make_track(bbox=(200, 200, 300, 399))
    analysis = _make_analysis(tracks=[track])
    engine._dwell_times[(1, "test_zone")] = time.time() - 100

    outside_track = _make_track(bbox=(10, 10, 50, 50))
    outside_analysis = _make_analysis(tracks=[outside_track])
    engine.evaluate(outside_analysis)
    assert (1, "test_zone") not in engine._dwell_times


# ---------------------------------------------------------------------------
# 27–28: INTRUSION rule
# ---------------------------------------------------------------------------

def test_intrusion_triggers_once():
    engine = _make_engine(rules=[INTRUSION_RULE])
    track = _make_track(bbox=(200, 200, 300, 399))
    analysis = _make_analysis(tracks=[track])
    v1 = engine.evaluate(analysis)
    v2 = engine.evaluate(analysis)
    intrusion_v1 = [v for v in v1 if v.rule_type == RuleType.INTRUSION]
    intrusion_v2 = [v for v in v2 if v.rule_type == RuleType.INTRUSION]
    assert len(intrusion_v1) == 1
    assert len(intrusion_v2) == 0


def test_intrusion_reentry():
    engine = _make_engine(rules=[INTRUSION_RULE])
    track = _make_track(bbox=(200, 200, 300, 399))
    analysis_in = _make_analysis(tracks=[track])
    engine.evaluate(analysis_in)

    outside_track = _make_track(bbox=(10, 10, 50, 50))
    engine.evaluate(_make_analysis(tracks=[outside_track]))

    v2 = engine.evaluate(analysis_in)
    intrusions = [v for v in v2 if v.rule_type == RuleType.INTRUSION]
    assert len(intrusions) == 1


# ---------------------------------------------------------------------------
# 29–30: CROWD_LIMIT rule
# ---------------------------------------------------------------------------

def test_crowd_limit_under():
    engine = _make_engine()
    now = time.time()
    tracks_in = list(range(3))
    vio = engine.evaluate_crowd_limit(CROWD_RULE, SQUARE_ZONE, tracks_in, now)
    assert vio is None


def test_crowd_limit_over():
    engine = _make_engine()
    now = time.time()
    tracks_in = list(range(7))
    vio = engine.evaluate_crowd_limit(CROWD_RULE, SQUARE_ZONE, tracks_in, now)
    assert vio is not None
    assert vio.rule_type == RuleType.CROWD_LIMIT
    assert vio.persons_in_zone == 7


# ---------------------------------------------------------------------------
# 31–32: Multi-zone / multi-track
# ---------------------------------------------------------------------------

def test_multiple_zones_same_camera():
    zm = ZoneManager()
    z1 = Zone("z1", "cam0", "Zone 1", ZoneType.RESTRICTED,
              [(0, 0), (200, 0), (200, 200), (0, 200)])
    z2 = Zone("z2", "cam0", "Zone 2", ZoneType.RESTRICTED,
              [(300, 300), (500, 300), (500, 500), (300, 500)])
    zm.add_zone(z1)
    zm.add_zone(z2)
    r1 = Rule("r1", "z1", RuleType.NO_ENTRY, cooldown_seconds=0)
    r2 = Rule("r2", "z2", RuleType.NO_ENTRY, cooldown_seconds=0)
    zm.add_rule(r1)
    zm.add_rule(r2)
    engine = ZoneEngine(zm)

    track = _make_track(bbox=(50, 50, 150, 190))
    analysis = _make_analysis(tracks=[track], camera_id="cam0")
    viols = engine.evaluate(analysis)
    assert all(v.zone_id == "z1" for v in viols)


def test_multiple_tracks_same_zone():
    engine = _make_engine(rules=[NO_ENTRY_RULE])
    tracks = [_make_track(track_id=i, bbox=(200, 200, 300, 399)) for i in range(1, 4)]
    analysis = _make_analysis(tracks=tracks)
    viols = engine.evaluate(analysis)
    assert len(viols) == 3


# ---------------------------------------------------------------------------
# 33: Full integration
# ---------------------------------------------------------------------------

def test_zone_engine_with_tracked_analysis():
    engine = _make_engine(rules=[NO_ENTRY_RULE])
    track = _make_track(bbox=(200, 200, 300, 399))
    analysis = _make_analysis(tracks=[track])
    viols = engine.evaluate(analysis)
    assert isinstance(viols, list)
    assert all(isinstance(v, ZoneViolation) for v in viols)


# ---------------------------------------------------------------------------
# 34: Visualization
# ---------------------------------------------------------------------------

def test_zone_visualization():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = draw_zones(frame, [SQUARE_ZONE])
    assert result.shape == frame.shape
    assert result.dtype == np.uint8

    vio = ZoneViolation(
        rule_id="r1", zone_id="test_zone", zone_name="Test Zone",
        camera_id="test_cam", track_id=1, rule_type=RuleType.NO_ENTRY,
        timestamp=time.time(), details="Test", confidence=1.0,
    )
    result2 = draw_violations(frame, [vio])
    assert result2.shape == frame.shape


# ---------------------------------------------------------------------------
# 35: State cleanup on track removal
# ---------------------------------------------------------------------------

def test_state_cleanup_on_track_removal():
    engine = _make_engine(rules=[LOITERING_RULE, INTRUSION_RULE])
    track = _make_track(bbox=(200, 200, 300, 399))
    analysis = _make_analysis(tracks=[track])
    engine.evaluate(analysis)

    assert (1, "test_zone") in engine._dwell_times
    assert (1, "test_zone") in engine._intrusion_triggered

    empty_analysis = _make_analysis(tracks=[])
    engine.evaluate(empty_analysis)

    assert (1, "test_zone") not in engine._dwell_times
    assert (1, "test_zone") not in engine._intrusion_triggered
