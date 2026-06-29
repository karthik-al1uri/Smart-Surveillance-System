"""Tests for Phase 3: Pose Estimation (YOLOv8-Pose).

All model-dependent tests run because weights are present at models/yolov8m-pose.pt.
"""

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_config, get_project_root
from src.common.visualization import draw_detections, draw_frame_analysis, draw_poses
from src.detection.combined_pipeline import CombinedDetectionPipeline, FrameAnalysis
from src.detection.pose_estimator import PoseEstimator
from src.detection.pose_structures import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
    Keypoint,
    PoseResult,
)
from src.detection.yolo_detector import Detection


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

MODEL_EXISTS = (get_project_root() / "models" / "yolov8m-pose.pt").exists()


def _make_keypoints(n: int = 17) -> list:
    return [Keypoint(x=float(i * 10), y=float(i * 5), confidence=0.9) for i in range(n)]


def _make_pose_result(**kwargs) -> PoseResult:
    defaults = dict(
        camera_id="cam0",
        frame_id=0,
        timestamp=time.time(),
        bbox=(100, 50, 300, 480),
        bbox_confidence=0.92,
        keypoints=_make_keypoints(17),
    )
    defaults.update(kwargs)
    return PoseResult(**defaults)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def pose_estimator(config):
    return PoseEstimator(config=config)


@pytest.fixture(scope="module")
def pipeline(config):
    return CombinedDetectionPipeline(config=config)


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame():
    return np.full((480, 640, 3), 255, dtype=np.uint8)


@pytest.fixture
def sample_detection():
    return Detection(
        class_id=0,
        class_name="person",
        bbox=(50, 50, 200, 400),
        confidence=0.88,
        frame_idx=0,
    )


# ---------------------------------------------------------------------------
# 1. Keypoint dataclass
# ---------------------------------------------------------------------------

def test_keypoint_dataclass():
    kp = Keypoint(x=100.5, y=200.3, confidence=0.85)
    assert kp.x == pytest.approx(100.5)
    assert kp.y == pytest.approx(200.3)
    assert kp.confidence == pytest.approx(0.85)


def test_keypoint_is_visible_above_threshold():
    assert Keypoint(x=0, y=0, confidence=0.5).is_visible is True


def test_keypoint_is_not_visible_below_threshold():
    assert Keypoint(x=0, y=0, confidence=0.1).is_visible is False


def test_keypoint_as_array():
    kp = Keypoint(x=10.0, y=20.0, confidence=0.7)
    arr = kp.as_array()
    assert arr == [10.0, 20.0, 0.7]


# ---------------------------------------------------------------------------
# 2. PoseResult dataclass
# ---------------------------------------------------------------------------

def test_pose_result_dataclass():
    pose = _make_pose_result()
    assert pose.camera_id == "cam0"
    assert pose.frame_id == 0
    assert isinstance(pose.bbox, tuple) and len(pose.bbox) == 4
    assert len(pose.keypoints) == 17


def test_pose_result_requires_17_keypoints():
    with pytest.raises(ValueError):
        PoseResult(
            camera_id="cam0",
            frame_id=0,
            timestamp=time.time(),
            bbox=(0, 0, 100, 200),
            bbox_confidence=0.9,
            keypoints=_make_keypoints(16),
        )


def test_pose_result_visible_keypoint_count():
    kps = [Keypoint(x=0, y=0, confidence=0.9)] * 10 + [Keypoint(x=0, y=0, confidence=0.1)] * 7
    pose = _make_pose_result(keypoints=kps)
    assert pose.visible_keypoint_count == 10


def test_pose_result_keypoints_as_array():
    pose = _make_pose_result()
    arr = pose.keypoints_as_array()
    assert len(arr) == 17
    assert len(arr[0]) == 3


# ---------------------------------------------------------------------------
# 3. COCO keypoint names
# ---------------------------------------------------------------------------

def test_coco_keypoint_names_count():
    assert len(COCO_KEYPOINT_NAMES) == 17


def test_coco_keypoint_names_order():
    assert COCO_KEYPOINT_NAMES[0] == "nose"
    assert COCO_KEYPOINT_NAMES[5] == "left_shoulder"
    assert COCO_KEYPOINT_NAMES[11] == "left_hip"
    assert COCO_KEYPOINT_NAMES[16] == "right_ankle"


# ---------------------------------------------------------------------------
# 4. COCO skeleton connections
# ---------------------------------------------------------------------------

def test_coco_skeleton_connections_valid_indices():
    for a, b in COCO_SKELETON:
        assert 0 <= a <= 16, f"Invalid index {a} in skeleton"
        assert 0 <= b <= 16, f"Invalid index {b} in skeleton"


def test_coco_skeleton_is_not_empty():
    assert len(COCO_SKELETON) > 0


# ---------------------------------------------------------------------------
# 5. PoseEstimator init
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_pose_estimator_init(pose_estimator):
    assert pose_estimator is not None


@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_pose_estimator_model_info(pose_estimator):
    info = pose_estimator.get_model_info()
    assert "model_path" in info
    assert "device" in info
    assert "conf_threshold" in info
    assert "single_pass_mode" in info


# ---------------------------------------------------------------------------
# 6. estimate_single — single frame
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_pose_estimator_single_frame_output_structure(pose_estimator, blank_frame):
    out = pose_estimator.estimate_single(blank_frame, camera_id="cam0", frame_id=0)
    assert "poses" in out
    assert "person_detections" in out
    assert "inference_time_ms" in out
    assert isinstance(out["poses"], list)
    assert isinstance(out["person_detections"], list)
    assert out["inference_time_ms"] >= 0.0


# ---------------------------------------------------------------------------
# 7. Empty / blank frame → empty results
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_pose_estimator_empty_frame_returns_empty(pose_estimator):
    out = pose_estimator.estimate_single(np.array([]), camera_id="cam0", frame_id=0)
    assert out["poses"] == []
    assert out["person_detections"] == []


@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_pose_estimator_blank_frame_no_crash(pose_estimator, blank_frame):
    out = pose_estimator.estimate_single(blank_frame)
    assert isinstance(out["poses"], list)


# ---------------------------------------------------------------------------
# 8. estimate_batch — multiple frames keyed by camera_id
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_pose_estimator_batch_returns_dict(pose_estimator, blank_frame, white_frame):
    frames = [
        {"frame": blank_frame, "camera_id": "cam1", "frame_id": 0, "timestamp": time.time()},
        {"frame": white_frame, "camera_id": "cam2", "frame_id": 0, "timestamp": time.time()},
    ]
    results = pose_estimator.estimate_batch(frames)
    assert isinstance(results, dict)
    assert "cam1" in results
    assert "cam2" in results


@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_pose_estimator_batch_same_camera_accumulates(pose_estimator, blank_frame):
    frames = [
        {"frame": blank_frame, "camera_id": "cam1", "frame_id": 0, "timestamp": time.time()},
        {"frame": blank_frame, "camera_id": "cam1", "frame_id": 1, "timestamp": time.time()},
    ]
    results = pose_estimator.estimate_batch(frames)
    assert "cam1" in results
    assert isinstance(results["cam1"], list)


# ---------------------------------------------------------------------------
# 9. Every PoseResult has exactly 17 keypoints
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_pose_estimator_keypoint_count(pose_estimator, blank_frame):
    out = pose_estimator.estimate_single(blank_frame)
    for pose in out["poses"]:
        assert len(pose.keypoints) == 17, "Each PoseResult must have exactly 17 keypoints"


# ---------------------------------------------------------------------------
# 10. Bounding boxes within frame dimensions
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_pose_estimator_bbox_coords_within_frame(pose_estimator):
    h, w = 480, 640
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    out = pose_estimator.estimate_single(frame)
    for pose in out["poses"]:
        x1, y1, x2, y2 = pose.bbox
        assert x1 >= 0 and y1 >= 0
        assert x2 <= w + 10 and y2 <= h + 10


# ---------------------------------------------------------------------------
# 11. Single-pass mode outputs Detection objects for persons
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_single_pass_mode_enabled(config, blank_frame):
    cfg = {**config, "pose": {**config["pose"], "single_pass_mode": True}}
    estimator = PoseEstimator(config=cfg)
    out = estimator.estimate_single(blank_frame)
    assert isinstance(out["person_detections"], list)


@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_single_pass_mode_disabled_no_person_dets(config, blank_frame):
    cfg = {**config, "pose": {**config["pose"], "single_pass_mode": False}}
    estimator = PoseEstimator(config=cfg)
    out = estimator.estimate_single(blank_frame)
    assert out["person_detections"] == []


# ---------------------------------------------------------------------------
# 12. CombinedDetectionPipeline — frame → FrameAnalysis
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_combined_pipeline_flow(pipeline, blank_frame):
    analysis = pipeline.process_frame(blank_frame, camera_id="cam0", frame_id=0)
    assert isinstance(analysis, FrameAnalysis)
    assert analysis.camera_id == "cam0"
    assert analysis.frame_id == 0
    assert isinstance(analysis.person_detections, list)
    assert isinstance(analysis.object_detections, list)
    assert isinstance(analysis.poses, list)


# ---------------------------------------------------------------------------
# 13. CombinedDetectionPipeline stats include inference time
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_combined_pipeline_stats(pipeline, blank_frame):
    pipeline.process_frame(blank_frame)
    stats = pipeline.get_stats()
    assert "frames_processed" in stats
    assert stats["frames_processed"] >= 1
    assert "avg_inference_ms" in stats
    assert stats["avg_inference_ms"] >= 0.0


# ---------------------------------------------------------------------------
# 14. Visualization — draw_detections returns valid image
# ---------------------------------------------------------------------------

def test_visualization_detections(blank_frame, sample_detection):
    out = draw_detections(blank_frame, [sample_detection])
    assert out.shape == blank_frame.shape
    assert out.dtype == np.uint8


def test_visualization_detections_does_not_mutate_original(blank_frame, sample_detection):
    original = blank_frame.copy()
    draw_detections(blank_frame, [sample_detection])
    np.testing.assert_array_equal(blank_frame, original)


# ---------------------------------------------------------------------------
# 15. Visualization — draw_poses returns valid image
# ---------------------------------------------------------------------------

def test_visualization_poses(blank_frame):
    pose = _make_pose_result()
    out = draw_poses(blank_frame, [pose])
    assert out.shape == blank_frame.shape
    assert out.dtype == np.uint8


def test_visualization_poses_no_skeleton(blank_frame):
    pose = _make_pose_result()
    out = draw_poses(blank_frame, [pose], skeleton=False)
    assert out.shape == blank_frame.shape


# ---------------------------------------------------------------------------
# 16. Visualization — draw_frame_analysis returns valid image
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="yolov8m-pose.pt not found")
def test_visualization_combined(pipeline, blank_frame):
    analysis = pipeline.process_frame(blank_frame)
    out = draw_frame_analysis(blank_frame, analysis)
    assert out.shape == blank_frame.shape
    assert out.dtype == np.uint8
