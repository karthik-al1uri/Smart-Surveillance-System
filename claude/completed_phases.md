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
