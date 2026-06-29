"""Tests for Phase 4: Person Tracking (ByteTrack).

All tests use synthetic data — no model inference required.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_config
from src.common.visualization import draw_tracked_frame, draw_tracks
from src.detection.combined_pipeline import FrameAnalysis, TrackedFrameAnalysis
from src.detection.kalman_tracker import KalmanBoxTracker, bbox_to_z, x_to_bbox
from src.detection.pose_structures import COCO_KEYPOINT_NAMES, Keypoint, PoseResult
from src.detection.tracker import ByteTracker, Track, compute_iou, compute_iou_matrix, linear_assignment
from src.detection.yolo_detector import Detection


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

_DEFAULT_KP_OFFSETS = [
    (0, -60), (-8, -50), (8, -50), (-14, -44), (14, -44),
    (-20, -20), (20, -20), (-30, 0), (30, 0), (-35, 20), (35, 20),
    (-14, 20), (14, 20), (-16, 60), (16, 60), (-18, 100), (18, 100),
]


def make_detection(frame_idx: int = 0, x1: int = 100, y1: int = 100,
                   x2: int = 180, y2: int = 350, conf: float = 0.9) -> Detection:
    return Detection(
        class_id=0, class_name="person",
        bbox=(x1, y1, x2, y2),
        confidence=conf,
        frame_idx=frame_idx,
    )


def make_moving_detection(frame_idx: int, start_x: int = 100, velocity: int = 5,
                          conf: float = 0.9) -> Detection:
    x1 = start_x + frame_idx * velocity
    return Detection(
        class_id=0, class_name="person",
        bbox=(x1, 100, x1 + 80, 350),
        confidence=conf,
        frame_idx=frame_idx,
    )


def make_keypoints(x_offset: int = 140, y_offset: int = 200) -> List[Keypoint]:
    return [
        Keypoint(x=x_offset + dx, y=y_offset + dy, confidence=0.9)
        for dx, dy in _DEFAULT_KP_OFFSETS
    ]


def make_pose(frame_idx: int = 0, bbox=(100, 100, 180, 350)) -> PoseResult:
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    return PoseResult(
        camera_id="test_cam",
        frame_id=frame_idx,
        timestamp=time.time(),
        bbox=bbox,
        bbox_confidence=0.9,
        keypoints=make_keypoints(cx, cy),
    )


def make_frame_analysis(
    detections: List[Detection],
    poses: List[PoseResult] = None,
    frame_id: int = 0,
) -> FrameAnalysis:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return FrameAnalysis(
        camera_id="test_cam",
        frame_id=frame_id,
        timestamp=time.time(),
        frame=frame,
        person_detections=detections,
        object_detections=[],
        poses=poses or [],
        inference_time_ms=5.0,
    )


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture
def tracker(config):
    t = ByteTracker(config=config)
    return t


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. Track dataclass
# ---------------------------------------------------------------------------

def test_track_dataclass():
    track = Track(
        track_id=1, state="active", bbox=(10, 20, 100, 300),
        bbox_history=deque(maxlen=64), keypoint_history=deque(maxlen=64),
        pose_history=deque(maxlen=64),
        age=5, hits=3, time_since_update=0, confidence=0.88,
        camera_id="cam0", created_at=time.time(), last_seen_at=time.time(),
    )
    assert track.track_id == 1
    assert track.state == "active"
    assert track.bbox == (10, 20, 100, 300)
    assert track.age == 5
    assert track.hits == 3
    assert track.confidence == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# 2. Kalman filter — predict
# ---------------------------------------------------------------------------

def test_kalman_predict():
    bbox = np.array([100.0, 50.0, 200.0, 300.0])
    kf = KalmanBoxTracker(bbox)
    pred = kf.predict()
    assert pred.shape == (4,)
    assert pred[2] > pred[0]
    assert pred[3] > pred[1]


def test_kalman_predict_returns_float_array():
    kf = KalmanBoxTracker(np.array([0.0, 0.0, 100.0, 200.0]))
    pred = kf.predict()
    assert pred.dtype in (np.float32, np.float64)


# ---------------------------------------------------------------------------
# 3. Kalman filter — update convergence
# ---------------------------------------------------------------------------

def test_kalman_update_converges():
    true_bbox = np.array([100.0, 50.0, 200.0, 300.0])
    kf = KalmanBoxTracker(true_bbox)
    for _ in range(10):
        kf.predict()
        kf.update(true_bbox)
    state = kf.get_state()
    np.testing.assert_allclose(state, true_bbox, atol=5.0)


# ---------------------------------------------------------------------------
# 4-6. IoU — identical, no overlap, partial
# ---------------------------------------------------------------------------

def test_iou_identical_boxes():
    assert compute_iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert compute_iou((0, 0, 50, 50), (100, 100, 200, 200)) == pytest.approx(0.0)


def test_iou_partial_overlap():
    iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
    expected = 2500 / (10000 + 10000 - 2500)
    assert iou == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# 7. IoU matrix
# ---------------------------------------------------------------------------

def test_iou_matrix_shape():
    b1 = np.array([[0, 0, 50, 50], [100, 100, 200, 200], [200, 200, 300, 300]], dtype=float)
    b2 = np.array([[0, 0, 50, 50], [150, 150, 250, 250], [300, 0, 400, 100], [50, 50, 100, 100]], dtype=float)
    mat = compute_iou_matrix(b1, b2)
    assert mat.shape == (3, 4)


def test_iou_matrix_diagonal_self():
    b = np.array([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=float)
    mat = compute_iou_matrix(b, b)
    assert mat[0, 0] == pytest.approx(1.0)
    assert mat[1, 1] == pytest.approx(1.0)
    assert mat[0, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. Linear assignment
# ---------------------------------------------------------------------------

def test_linear_assignment_known_matrix():
    cost = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=float)
    matches, unmatched_r, unmatched_c = linear_assignment(cost)
    match_dict = dict(matches)
    assert match_dict.get(0) == 0
    assert match_dict.get(1) == 1
    assert unmatched_r == []
    assert unmatched_c == []


def test_linear_assignment_empty():
    matches, ur, uc = linear_assignment(np.zeros((0, 0)))
    assert matches == []


def test_linear_assignment_more_detections():
    cost = np.array([[0.2, 0.8, 0.3]], dtype=float)
    matches, ur, uc = linear_assignment(cost)
    assert len(matches) == 1
    assert len(uc) == 2


# ---------------------------------------------------------------------------
# 9. Tracker — single detection → new track
# ---------------------------------------------------------------------------

def test_tracker_single_detection(tracker):
    det = make_detection(frame_idx=0)
    fa = make_frame_analysis([det], frame_id=0)
    tracks = tracker.update(fa)
    assert any(t.state == "active" for t in tracks)


# ---------------------------------------------------------------------------
# 10. Tracker — consistent ID across frames
# ---------------------------------------------------------------------------

def test_tracker_consistent_id(config):
    t = ByteTracker(config=config)
    ids = set()
    for i in range(5):
        det = make_moving_detection(frame_idx=i)
        fa = make_frame_analysis([det], frame_id=i)
        tracks = t.update(fa)
        active = [tr for tr in tracks if tr.state == "active"]
        if active:
            ids.add(active[0].track_id)
    assert len(ids) == 1, f"Expected 1 unique track ID but got {ids}"


# ---------------------------------------------------------------------------
# 11. Tracker — two non-overlapping persons → two tracks
# ---------------------------------------------------------------------------

def test_tracker_two_persons(config):
    t = ByteTracker(config=config)
    det_a = make_detection(x1=50, y1=50, x2=130, y2=300)
    det_b = make_detection(x1=400, y1=50, x2=480, y2=300)
    for i in range(4):
        fa = make_frame_analysis([det_a, det_b], frame_id=i)
        tracks = t.update(fa)
    active_ids = {tr.track_id for tr in tracks if tr.state == "active"}
    assert len(active_ids) == 2


# ---------------------------------------------------------------------------
# 12. Tracker — detection disappears → track goes to lost
# ---------------------------------------------------------------------------

def test_tracker_track_loss(config):
    t = ByteTracker(config=config)
    det = make_detection()
    for i in range(4):
        fa = make_frame_analysis([det], frame_id=i)
        t.update(fa)
    fa_empty = make_frame_analysis([], frame_id=4)
    tracks = t.update(fa_empty)
    assert any(tr.state == "lost" for tr in tracks)


# ---------------------------------------------------------------------------
# 13. Tracker — track removed after max_lost_frames
# ---------------------------------------------------------------------------

def test_tracker_track_removal(config):
    cfg = {**config, "tracker": {**config.get("tracker", {}), "max_lost_frames": 3}}
    t = ByteTracker(config=cfg)
    det = make_detection()
    for i in range(4):
        fa = make_frame_analysis([det], frame_id=i)
        t.update(fa)
    for i in range(5):
        fa = make_frame_analysis([], frame_id=4 + i)
        t.update(fa)
    stats = t.get_stats()
    assert stats["total_removed"] >= 1


# ---------------------------------------------------------------------------
# 14. Tracker — reappearance within max_lost_frames → same track_id
# ---------------------------------------------------------------------------

def test_tracker_reappearance(config):
    cfg = {**config, "tracker": {**config.get("tracker", {}), "max_lost_frames": 10, "min_hits": 1}}
    t = ByteTracker(config=cfg)
    det = make_detection(x1=100, y1=100, x2=180, y2=350)
    for i in range(5):
        fa = make_frame_analysis([det], frame_id=i)
        t.update(fa)
    first_id = t.get_active_tracks()[0].track_id if t.get_active_tracks() else None

    for i in range(3):
        fa = make_frame_analysis([], frame_id=5 + i)
        t.update(fa)

    det2 = make_detection(x1=102, y1=102, x2=182, y2=352)
    for i in range(4):
        fa = make_frame_analysis([det2], frame_id=8 + i)
        t.update(fa)
    active = t.get_active_tracks()
    if first_id is not None and active:
        assert active[0].track_id == first_id


# ---------------------------------------------------------------------------
# 15. Keypoint history — 20 frames → 20 entries
# ---------------------------------------------------------------------------

def test_tracker_keypoint_history(config):
    cfg = {**config, "tracker": {**config.get("tracker", {}), "min_hits": 1}}
    t = ByteTracker(config=cfg)
    for i in range(20):
        det = make_moving_detection(frame_idx=i)
        pose = make_pose(frame_idx=i, bbox=det.bbox)
        fa = make_frame_analysis([det], poses=[pose], frame_id=i)
        t.update(fa)
    active = t.get_active_tracks()
    assert active, "Expected at least one active track"
    assert len(active[0].keypoint_history) == 20


# ---------------------------------------------------------------------------
# 16. get_track_keypoint_sequence — returns (16, 17, 3)
# ---------------------------------------------------------------------------

def test_tracker_keypoint_sequence_extraction(config):
    cfg = {**config, "tracker": {**config.get("tracker", {}), "min_hits": 1}}
    t = ByteTracker(config=cfg)
    for i in range(20):
        det = make_moving_detection(frame_idx=i)
        pose = make_pose(frame_idx=i, bbox=det.bbox)
        fa = make_frame_analysis([det], poses=[pose], frame_id=i)
        t.update(fa)
    active = t.get_active_tracks()
    assert active
    tid = active[0].track_id
    seq = t.get_track_keypoint_sequence(tid, window_size=16)
    assert seq is not None
    assert seq.shape == (16, 17, 3)


# ---------------------------------------------------------------------------
# 17. get_track_keypoint_sequence — insufficient frames → None
# ---------------------------------------------------------------------------

def test_tracker_keypoint_sequence_insufficient(config):
    cfg = {**config, "tracker": {**config.get("tracker", {}), "min_hits": 1}}
    t = ByteTracker(config=cfg)
    for i in range(5):
        det = make_moving_detection(frame_idx=i)
        pose = make_pose(frame_idx=i, bbox=det.bbox)
        fa = make_frame_analysis([det], poses=[pose], frame_id=i)
        t.update(fa)
    active = t.get_active_tracks()
    assert active
    seq = t.get_track_keypoint_sequence(active[0].track_id, window_size=16)
    assert seq is None


# ---------------------------------------------------------------------------
# 18. History capped at maxlen=64
# ---------------------------------------------------------------------------

def test_tracker_history_maxlen(config):
    cfg = {**config, "tracker": {**config.get("tracker", {}), "min_hits": 1, "history_length": 64}}
    t = ByteTracker(config=cfg)
    for i in range(100):
        det = make_moving_detection(frame_idx=i)
        pose = make_pose(frame_idx=i, bbox=det.bbox)
        fa = make_frame_analysis([det], poses=[pose], frame_id=i)
        t.update(fa)
    active = t.get_active_tracks()
    assert active
    assert len(active[0].keypoint_history) <= 64
    assert len(active[0].bbox_history) <= 64


# ---------------------------------------------------------------------------
# 19. Stats tracking
# ---------------------------------------------------------------------------

def test_tracker_stats(config):
    t = ByteTracker(config=config)
    det = make_detection()
    for i in range(3):
        fa = make_frame_analysis([det], frame_id=i)
        t.update(fa)
    stats = t.get_stats()
    assert stats["total_created"] >= 1
    assert "currently_active" in stats
    assert "currently_lost" in stats
    assert "currently_removed" in stats


# ---------------------------------------------------------------------------
# 20. Reset clears all tracks
# ---------------------------------------------------------------------------

def test_tracker_reset(config):
    t = ByteTracker(config=config)
    det = make_detection()
    for i in range(3):
        fa = make_frame_analysis([det], frame_id=i)
        t.update(fa)
    t.reset()
    assert t.get_active_tracks() == []
    assert t.get_stats()["total_created"] == 0


# ---------------------------------------------------------------------------
# 21. Full pipeline: frame buffer → detection+pose → tracking → TrackedFrameAnalysis
# ---------------------------------------------------------------------------

def test_pipeline_with_tracker(config):
    from src.detection.combined_pipeline import CombinedDetectionPipeline
    pipeline = CombinedDetectionPipeline(config=config)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame, camera_id="cam0", frame_id=0)
    assert isinstance(result, TrackedFrameAnalysis)
    assert hasattr(result, "tracks")
    assert isinstance(result.tracks, list)
    assert result.tracking_time_ms >= 0.0


# ---------------------------------------------------------------------------
# 22. Track visualization — draw_tracks produces valid image
# ---------------------------------------------------------------------------

def test_track_visualization(config, blank_frame):
    t = ByteTracker(config=config)
    det = make_detection()
    for i in range(4):
        fa = make_frame_analysis([det], frame_id=i)
        t.update(fa)
    tracks = t.get_active_tracks() + [tr for tr in t._tracks if tr.state == "lost"]
    out = draw_tracks(blank_frame, tracks)
    assert out.shape == blank_frame.shape
    assert out.dtype == np.uint8


def test_draw_tracked_frame_valid_image(config, blank_frame):
    pipeline_cfg = {**config, "tracker": {**config.get("tracker", {}), "min_hits": 1}}
    from src.detection.combined_pipeline import CombinedDetectionPipeline
    pipeline = CombinedDetectionPipeline(config=pipeline_cfg)
    result = pipeline.process_frame(blank_frame)
    out = draw_tracked_frame(blank_frame, result)
    assert out.shape == blank_frame.shape
    assert out.dtype == np.uint8
