# Completed Phases — Smart Surveillance System

*Phases are moved here from `pending_phases.md` upon completion.*
*Each entry includes the completion date and implementation notes.*

---

## Phase 2: Object Detection (YOLOv8)
**Status:** ✅ COMPLETED
**Completed:** 2026-06-29
**Branch:** phase-2/object-detection

**Implementation Notes:**
- YOLOv8m (medium, 50MB) used — COCO pretrained, CPU mode (Mac M-series, no CUDA)
- ultralytics v8.4.82 installed system-wide via pip
- Batched inference implemented with configurable `batch_size` (default 4)
- Letterbox preprocessing with unscaling back to original coords via `FramePreprocessor`
- `Detection` and `DetectionResult` dataclasses provide structured output
- `DetectionPipeline` wires preprocessor + detector into single callable
- `src/common/config.py` and `src/common/logger.py` created as shared infra
- No deviations from plan; ONNX/TensorRT export deferred to Phase 12
- Known limitation: CPU-only inference; ~1–3 FPS on Mac (GPU will be faster in production)

**Files Created/Modified:**
- `src/common/config.py` — YAML config loader with deep-merge env overrides
- `src/common/logger.py` — Structured logger factory
- `src/detection/preprocessor.py` — Letterbox resize + bbox unscaling
- `src/detection/yolo_detector.py` — YOLOv8 wrapper with Detection/DetectionResult dataclasses
- `src/detection/detection_pipeline.py` — Orchestrates preprocessor + detector
- `tests/test_detection.py` — 33 tests (config, preprocessor, dataclasses, detector, pipeline)
- `tests/test_ingestion.py` — Placeholder stub (Phase 1 not yet built)
- `scripts/demo_detection.py` — CLI demo script with annotated frame saving option
- `config/default.yaml` — Populated with detection, pose, ingestion, storage, logging settings
- `requirements.txt` — Created with pinned minimum versions

**Tasks completed:**
- [x] Implement `src/detection/yolo_detector.py` — Load YOLOv8 model, run inference on preprocessed frames
- [x] Add frame preprocessing (resize to 640x640, letterbox with padding)
- [x] Implement batched inference for multi-camera GPU efficiency
- [x] Configure confidence threshold (default 0.45) and NMS IoU threshold (default 0.5)
- [x] Output structured detections: class_id, class_name, bbox, confidence
- [x] Write `tests/test_detection.py` — 33 tests, all passing

---

## Phase 3: Pose Estimation (YOLOv8-Pose)
**Status:** ✅ COMPLETED
**Completed:** 2026-06-29
**Branch:** phase-3/pose-estimation

**Implementation Notes:**
- YOLOv8m-Pose (51MB) used in single-pass mode — detects persons AND extracts 17-point COCO skeleton in one forward pass
- `single_pass_mode: true` in config; PoseEstimator outputs both `PoseResult` and `Detection` objects for persons
- Standalone YOLOv8 still runs for non-person objects (weapons, bags, vehicles) — filtered to `_NON_PERSON_CLASSES`
- `CombinedDetectionPipeline` orchestrates both models and exposes `process_frame()` / background `start()`/`stop()` API
- Visualization utilities handle detections (colour-coded by class), pose keypoints (blue circles), skeleton connections (cyan lines)
- 30 tests written, all passing; 64/64 total tests passing (no regressions from Phase 2)
- Known limitation: CPU-only on Mac; single-pass significantly faster than running two models separately

**Files Created/Modified:**
- `src/detection/pose_structures.py` — `Keypoint`, `PoseResult`, `COCO_KEYPOINT_NAMES`, `COCO_SKELETON`
- `src/detection/pose_estimator.py` — YOLOv8-Pose wrapper with single-pass and batch modes
- `src/detection/combined_pipeline.py` — `FrameAnalysis` dataclass + `CombinedDetectionPipeline`
- `src/common/visualization.py` — `draw_detections`, `draw_poses`, `draw_frame_analysis`
- `tests/test_pose.py` — 30 tests covering dataclasses, COCO constants, estimator, pipeline, visualization
- `scripts/demo_pose.py` — CLI demo with optional annotated video output
- `config/default.yaml` — Added `pose.single_pass_mode` and `pipeline.skip_object_detection`

**Tasks completed:**
- [x] Implement `src/detection/pose_estimator.py` — YOLOv8-Pose single-pass + batch modes
- [x] Output per-person keypoints: 17 joints × (x, y, confidence)
- [x] Handle occluded/missing keypoints (low confidence kept, `is_visible` property)
- [x] Combined with Phase 2 via single-pass mode (YOLOv8-Pose does detection + pose)
- [x] Write `tests/test_pose.py` — 30 tests, all passing
