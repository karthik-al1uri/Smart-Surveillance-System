"""Unit and integration tests for Phase 2: Object Detection.

MODEL_EXISTS is set to True — model-dependent tests are NOT skipped.
Tests assume models/yolov8m.pt is present in the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_EXISTS = True

from src.common.config import load_config, get_project_root
from src.detection.preprocessor import FramePreprocessor
from src.detection.yolo_detector import YOLODetector, Detection, DetectionResult
from src.detection.detection_pipeline import DetectionPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def detector(config):
    return YOLODetector(config=config)


@pytest.fixture(scope="module")
def pipeline(config):
    return DetectionPipeline(config=config)


@pytest.fixture
def blank_frame():
    """640x480 black BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame():
    """640x480 white BGR frame."""
    return np.full((480, 640, 3), 255, dtype=np.uint8)


@pytest.fixture
def person_like_frame():
    """Simple synthetic frame — no guarantee of detection, but valid input."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    frame[100:400, 260:380] = [200, 180, 160]
    return frame


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_loads(self, config):
        assert isinstance(config, dict)

    def test_detection_section_present(self, config):
        assert "detection" in config

    def test_model_path_configured(self, config):
        assert "model_path" in config["detection"]
        assert config["detection"]["model_path"] == "models/yolov8m.pt"

    def test_confidence_threshold_range(self, config):
        conf = config["detection"]["confidence_threshold"]
        assert 0.0 < conf < 1.0

    def test_iou_threshold_range(self, config):
        iou = config["detection"]["iou_threshold"]
        assert 0.0 < iou < 1.0

    def test_model_file_exists_on_disk(self):
        model_path = get_project_root() / "models" / "yolov8m.pt"
        assert model_path.exists(), (
            f"YOLOv8m weights not found at {model_path}. "
            "Place yolov8m.pt in the models/ directory."
        )


# ---------------------------------------------------------------------------
# Preprocessor tests
# ---------------------------------------------------------------------------

class TestFramePreprocessor:
    def test_output_shape_matches_input_size(self, blank_frame):
        preprocessor = FramePreprocessor(input_size=640)
        result, meta = preprocessor.preprocess_frame(blank_frame)
        assert result.shape == (640, 640, 3)

    def test_output_shape_smaller_size(self, blank_frame):
        preprocessor = FramePreprocessor(input_size=320)
        result, _ = preprocessor.preprocess_frame(blank_frame)
        assert result.shape == (320, 320, 3)

    def test_meta_contains_four_values(self, blank_frame):
        preprocessor = FramePreprocessor(input_size=640)
        _, meta = preprocessor.preprocess_frame(blank_frame)
        assert len(meta) == 4

    def test_scale_is_positive(self, blank_frame):
        preprocessor = FramePreprocessor(input_size=640)
        _, (sx, sy, pw, ph) = preprocessor.preprocess_frame(blank_frame)
        assert sx > 0 and sy > 0

    def test_empty_frame_raises(self):
        preprocessor = FramePreprocessor(input_size=640)
        with pytest.raises(ValueError):
            preprocessor.preprocess_frame(np.array([]))

    def test_wrong_channels_raises(self):
        preprocessor = FramePreprocessor(input_size=640)
        with pytest.raises(ValueError):
            preprocessor.preprocess_frame(np.zeros((480, 640), dtype=np.uint8))

    def test_batch_returns_correct_count(self, blank_frame, white_frame):
        preprocessor = FramePreprocessor(input_size=640)
        processed, metas = preprocessor.preprocess_batch([blank_frame, white_frame])
        assert len(processed) == 2
        assert len(metas) == 2

    def test_unscale_bbox_roundtrip(self, blank_frame):
        preprocessor = FramePreprocessor(input_size=640)
        _, meta = preprocessor.preprocess_frame(blank_frame)
        sx, sy, pw, ph = meta
        x1_model = int(100 * sx + pw)
        y1_model = int(100 * sy + ph)
        x2_model = int(200 * sx + pw)
        y2_model = int(200 * sy + ph)
        x1, y1, x2, y2 = preprocessor.unscale_bbox(
            (x1_model, y1_model, x2_model, y2_model), meta
        )
        assert abs(x1 - 100) <= 1
        assert abs(y1 - 100) <= 1
        assert abs(x2 - 200) <= 1
        assert abs(y2 - 200) <= 1


# ---------------------------------------------------------------------------
# Detection dataclass tests
# ---------------------------------------------------------------------------

class TestDetectionDataclasses:
    def test_detection_fields(self):
        d = Detection(class_id=0, class_name="person", bbox=(10, 20, 50, 80), confidence=0.9)
        assert d.class_id == 0
        assert d.class_name == "person"
        assert d.bbox == (10, 20, 50, 80)
        assert d.confidence == 0.9

    def test_detection_result_empty(self):
        result = DetectionResult()
        assert result.count == 0
        assert result.person_detections == []

    def test_detection_result_person_filter(self):
        dets = [
            Detection(class_id=0, class_name="person", bbox=(0, 0, 10, 10), confidence=0.9),
            Detection(class_id=2, class_name="car", bbox=(0, 0, 10, 10), confidence=0.8),
        ]
        result = DetectionResult(detections=dets)
        assert result.count == 2
        assert len(result.person_detections) == 1
        assert result.person_detections[0].class_name == "person"


# ---------------------------------------------------------------------------
# YOLODetector tests (model-dependent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="Model weights not available")
class TestYOLODetector:
    def test_detector_loads(self, detector):
        assert detector is not None

    def test_detect_returns_detection_result(self, detector, blank_frame):
        result = detector.detect(blank_frame)
        assert isinstance(result, DetectionResult)

    def test_detect_blank_frame_no_crash(self, detector, blank_frame):
        result = detector.detect(blank_frame)
        assert result.count >= 0

    def test_detect_batch_returns_list(self, detector, blank_frame, white_frame):
        results = detector.detect_batch([blank_frame, white_frame])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_detect_batch_empty_list(self, detector):
        results = detector.detect_batch([])
        assert results == []

    def test_detect_single_frame_result_has_frame_idx(self, detector, blank_frame):
        result = detector.detect(blank_frame, frame_idx=5)
        assert result.frame_idx == 5

    def test_detect_batch_frame_indices_correct(self, detector, blank_frame, white_frame):
        results = detector.detect_batch([blank_frame, white_frame])
        assert results[0].frame_idx == 0
        assert results[1].frame_idx == 1

    def test_inference_time_non_negative(self, detector, blank_frame):
        result = detector.detect(blank_frame)
        assert result.inference_time_ms >= 0

    def test_all_detections_within_target_classes(self, detector, config, person_like_frame):
        result = detector.detect(person_like_frame)
        target = set(config["detection"]["target_classes"])
        for det in result.detections:
            assert det.class_name in target, f"Unexpected class: {det.class_name}"

    def test_confidence_above_threshold(self, detector, config, person_like_frame):
        threshold = config["detection"]["confidence_threshold"]
        result = detector.detect(person_like_frame)
        for det in result.detections:
            assert det.confidence >= threshold


# ---------------------------------------------------------------------------
# DetectionPipeline tests (model-dependent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="Model weights not available")
class TestDetectionPipeline:
    def test_pipeline_initialises(self, pipeline):
        assert pipeline is not None

    def test_process_frame_returns_result(self, pipeline, blank_frame):
        result = pipeline.process_frame(blank_frame)
        assert isinstance(result, DetectionResult)

    def test_process_frame_with_idx(self, pipeline, blank_frame):
        result = pipeline.process_frame(blank_frame, frame_idx=7)
        assert result.frame_idx == 7

    def test_process_batch_returns_list(self, pipeline, blank_frame, white_frame):
        results = pipeline.process_batch([blank_frame, white_frame])
        assert len(results) == 2

    def test_process_batch_empty(self, pipeline):
        results = pipeline.process_batch([])
        assert results == []

    def test_process_batch_order_preserved(self, pipeline, blank_frame, white_frame):
        frames = [blank_frame, white_frame, blank_frame]
        results = pipeline.process_batch(frames)
        assert len(results) == 3
        assert results[0].frame_idx == 0
        assert results[1].frame_idx == 1
        assert results[2].frame_idx == 2
