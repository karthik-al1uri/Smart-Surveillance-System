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

---

## Phase 4: Person Tracking (ByteTrack)
**Status:** ✅ COMPLETED
**Completed:** 2026-06-29
**Branch:** phase-4/person-tracking

**Implementation Notes:**
- ByteTrack implemented from scratch (IoU-based two-stage association, no external bytetrack library needed)
- Kalman filter via `filterpy` — constant-velocity model, 7D state (cx, cy, area, aspect, vx, vy, va)
- `lap.lapjv` used for Hungarian assignment (fast C-based solver); `scipy.optimize.linear_sum_assignment` as fallback
- `Track.keypoint_history` is a `deque(maxlen=64)` — each entry is a `(17, 3)` float32 array
- `get_track_keypoint_sequence(track_id, 16)` returns `(16, 17, 3)` array directly usable by Phase 5 LSTM
- Pose captured on track creation (frame 0) AND on every update — no missed frames
- `TrackedFrameAnalysis` replaces `FrameAnalysis` as pipeline output; `FrameAnalysis` retained internally
- `draw_tracks` uses hash-based consistent colours per track_id; movement trails via last-10 bbox centres
- 27 tracker tests + 91 total tests passing; 1 Phase 3 test updated for new return type
- Known limitation: IoU-only re-ID; appearance features deferred to Phase 12

**Files Created/Modified:**
- `src/detection/kalman_tracker.py` — `KalmanBoxTracker`, `bbox_to_z`, `x_to_bbox`
- `src/detection/tracker.py` — `Track` dataclass, `ByteTracker`, `compute_iou`, `compute_iou_matrix`, `linear_assignment`
- `src/detection/combined_pipeline.py` — Added `TrackedFrameAnalysis`; `process_frame` now returns it
- `src/common/visualization.py` — Added `draw_tracks`, `draw_tracked_frame`, `_track_color`
- `config/default.yaml` — Added `tracker` section
- `tests/test_tracker.py` — 27 tests covering Kalman, IoU, assignment, tracker lifecycle, keypoint history
- `tests/test_pose.py` — Updated `test_combined_pipeline_flow` to accept `TrackedFrameAnalysis`
- `scripts/demo_tracking.py` — CLI demo with per-frame track summary and annotated video output
- `requirements.txt` — Added `lap`, `scipy`, `filterpy`

**Tasks completed:**
- [x] Implement `src/detection/kalman_tracker.py` — Kalman filter for bbox state prediction
- [x] Implement `src/detection/tracker.py` — ByteTrack with two-stage IoU association
- [x] Maintain per-track history: bbox, keypoint, pose sequences (deque maxlen=64)
- [x] Keep keypoint history with `get_track_keypoint_sequence(id, 16)` → (16, 17, 3) for Phase 5
- [x] Handle track creation, update, lost, removal lifecycle
- [x] Write `tests/test_tracker.py` — 27 tests, all passing
- [x] 91/91 total tests passing, zero regressions

---

## Phase 5: Action Recognition (Keypoint LSTM Classifier)
**Status:** ✅ COMPLETED
**Completed:** 2026-06-30
**Branch:** phase-5/action-recognition

**Implementation Notes:**
- Bi-LSTM with attention pooling — 2 layers, 128 hidden units per direction, 256 combined
- Currently using DUMMY model (random weights) — not trained on real data
- Training pipeline ready — needs dataset (UCF-Crime / RWF-2000 / NTU RGB+D or custom)
- Sliding window: 16 frames, stride 8, 50% overlap → classification every ~4 seconds at 2fps
- 15 fine-grained ActionLabels mapped to 4 ActionCategories (Normal/Violent/Suspicious/Urgent)
- Hip-center normalization makes classifier position/scale invariant
- Missing keypoints linearly interpolated across frames; all-missing set to 0
- `ActionRecognitionPipeline` is decoupled from detection pipeline for modularity
- 119/119 total tests passing, zero regressions

**⚠️ TRAINING REQUIRED:**
To get meaningful predictions, train the model:
1. Prepare dataset: `python scripts/extract_keypoints.py --input_dir <raw_videos> --label <int>`
2. Train: `python training/train_action.py --dataset_dir training/datasets/processed/`
3. Evaluate: `python training/evaluate_action.py --model_path models/action_classifier_v1.pt --test_dir training/datasets/processed/test/`

**Files Created/Modified:**
- `src/recognition/action_classes.py` — `ActionCategory`, `ActionLabel`, `LABEL_TO_CATEGORY`, `ActionPrediction`
- `src/recognition/sliding_window.py` — `SlidingWindowManager`, `WindowData`
- `src/recognition/keypoint_lstm.py` — `KeypointLSTM`, `AttentionPooling`
- `src/recognition/keypoint_preprocessor.py` — `KeypointPreprocessor` (normalize + impute)
- `src/recognition/action_classifier.py` — `ActionClassifier` with dummy model support
- `src/recognition/recognition_pipeline.py` — `ActionRecognitionPipeline`
- `training/train_action.py` — Cross-entropy + Adam + ReduceLROnPlateau + early stopping
- `training/evaluate_action.py` — Per-class P/R/F1 + confusion matrix
- `training/export_action.py` — ONNX export
- `scripts/extract_keypoints.py` — Video → keypoint windows via full pipeline
- `scripts/create_dummy_dataset.py` — Synthetic sequences for 5 action classes
- `scripts/demo_action.py` — Full pipeline demo with action labels
- `tests/test_recognition.py` — 28 tests, all passing
- `config/default.yaml` — Added `action_recognition` section

**Tasks completed:**
- [x] Define action classes and ActionPrediction dataclass
- [x] Implement SlidingWindowManager with stride/overlap and quality filter
- [x] Implement KeypointLSTM with AttentionPooling
- [x] Implement KeypointPreprocessor with hip-center normalization
- [x] Implement ActionClassifier with dummy model fallback
- [x] Create training, evaluation, and ONNX export scripts
- [x] Create synthetic dataset generator and keypoint extraction script
- [x] Generate dummy model weights (models/action_classifier_v1.pt)
- [x] Implement ActionRecognitionPipeline (decoupled from detection)
- [x] Add action_recognition section to config/default.yaml
- [x] Write tests/test_recognition.py — 28 tests, all passing
- [x] 119/119 total tests passing, zero regressions

---

## Phase 6: Zone & Rule Engine
**Status:** ✅ COMPLETED
**Completed:** 2026-06-30
**Branch:** phase-6/zone-engine

**Implementation Notes:**
- Rule types implemented: `no_entry`, `loitering`, `intrusion`, `crowd_limit`, `abandoned_object`
- Zone scheduling with overnight support (e.g. 22:00–06:00 crosses midnight correctly)
- Shapely `Polygon` objects cached per `zone_id` — not recreated every frame
- Person position uses bbox bottom-center by default (feet position for "standing in zone")
- State persisted across frames: dwell times, intrusion triggers, cooldown timestamps
- State auto-cleaned when tracks disappear
- `ZoneManager.save_to_config()` persists zones back to YAML for operator workflow
- 35 zone tests + 154 total tests passing; zero regressions from earlier phases

**Files Created/Modified:**
- `src/scoring/zone_models.py` — `Zone`, `Rule`, `ZoneViolation`, `ZoneType`, `RuleType`
- `src/scoring/zone_manager.py` — `ZoneManager` with CRUD, YAML I/O, schedule logic
- `src/scoring/zone_engine.py` — `ZoneEngine` with all rule types, state, Shapely geometry
- `src/common/visualization.py` — Added `draw_zones`, `draw_violations`
- `tests/test_zone_engine.py` — 35 tests, all passing
- `scripts/demo_zones.py` — CLI demo with 3 predefined zones, violation printout
- `requirements.txt` — Added `shapely>=2.0.0`

**Tasks completed:**
- [x] Verify shapely dependency
- [x] Define `Zone`, `Rule`, `ZoneViolation` data structures
- [x] Implement `ZoneManager` with CRUD, config load/save, schedule checking
- [x] Implement `ZoneEngine` with `no_entry`, `loitering`, `intrusion`, `crowd_limit`, `abandoned_object`
- [x] Add zone visualization (`draw_zones`, `draw_violations`)
- [x] Write `tests/test_zone_engine.py` — 35 tests, all passing
- [x] Create `scripts/demo_zones.py`
- [x] 154/154 total tests passing, zero regressions

---

## Phase 7: Anomaly Scoring Engine
**Status:** ✅ COMPLETED
**Completed:** 2026-06-30
**Branch:** phase-7/anomaly-scoring

**Implementation Notes:**
- Weighted formula: `score = 0.35×action + 0.25×zone + 0.30×weapon + 0.10×time_of_day`
- Instant alerts for weapon detection (knife/gun/rifle/scissors) — bypass formula, score=1.0, ESCALATED
- Hysteresis: N consecutive high-score frames required before ALERT (default N=2)
- Cooldown: suppress duplicate alerts per (camera_id, track_id) pair (default 30s)
- ESCALATED events bypass cooldown; weapon alerts bypass both hysteresis and cooldown
- Per-camera config overrides: different thresholds/weights per camera
- Non-person events (abandoned object, crowd limit) scored with zone signal only
- State cleanup: stale tracks purged after 60s of inactivity
- 48 scoring tests + 202 total tests; zero regressions

**Known limitations:**
- Action signal value for SUSPICIOUS multiplied by 0.7, URGENT by 0.9 (per spec)
- Weapon cooldown uses a hash of class_name as pseudo-track-id (no actual track association)

**Files Created/Modified:**
- `src/scoring/scoring_models.py` — `AlertDecision`, `SignalType`, `ScoringSignal`, `ScoredEvent`, `ScoringConfig`
- `src/scoring/anomaly_scorer.py` — `AnomalyScorer` with full stateful logic
- `src/scoring/scoring_pipeline.py` — `ScoringPipeline` orchestrator with output queues
- `config/default.yaml` — Added `scoring` section + expanded `storage` section
- `tests/test_scoring.py` — 48 tests, all passing
- `scripts/demo_scoring.py` — Full pipeline demo with scoring printout

**Tasks completed:**
- [x] Define `AlertDecision`, `SignalType`, `ScoringSignal`, `ScoredEvent`, `ScoringConfig`
- [x] Implement `AnomalyScorer` with weighted scoring, hysteresis, cooldown, per-camera config
- [x] Add `scoring` section to `config/default.yaml` with per-camera override support
- [x] Implement `ScoringPipeline` orchestrator
- [x] Write `tests/test_scoring.py` — 48 tests, all passing
- [x] Create `scripts/demo_scoring.py`
- [x] 202/202 total tests passing, zero regressions

---

## Phase 8: Clip Capture & Storage Service
**Status:** ✅ COMPLETED
**Completed:** 2026-06-30
**Branch:** phase-8/clip-capture

**Implementation Notes:**
- Rolling buffer: JPEG-compressed frames in memory (default 20s × 10fps = 200 frames ≈ 10 MB/camera)
- Uncompressed budget for reference: 200 frames × 2.8 MB = ~560 MB/camera at 720p
- Async clip processing: `ClipCaptureService` queues requests and processes them after `post_seconds` elapsed (wall-clock), so post-event frames are always available
- Clip files named `{camera_id}_{YYYYMMDD_HHMMSS}_{event_id}.mp4` under `data/clips/{camera_id}/`
- Retention: clips auto-deleted after `retention_days` (default 30); enforcement runs every hour in background thread
- `AlertIntegration` glue layer: ALERT → priority=1, ESCALATED → priority=2
- Tests use `past = time.time() - 20.0` pattern so synthetic frame timestamps are always in-buffer and post-event delay is already elapsed
- 40 clip tests + 242 total tests; zero regressions

**Known limitations:**
- Encoder uses OpenCV `mp4v` codec (not H.264); H.264 requires FFmpeg or platform codec
- Memory is per-registered camera; unregistered cameras are silently dropped

**Files Created/Modified:**
- `src/alerts/clip_models.py` — `ClipRequest`, `ClipMetadata`, `StorageStats`
- `src/alerts/rolling_buffer.py` — `RollingFrameBuffer` with JPEG compression and thread safety
- `src/alerts/clip_encoder.py` — `ClipEncoder` using OpenCV `VideoWriter`
- `src/alerts/clip_capture.py` — `ClipCaptureService` with async background thread and retention
- `src/alerts/alert_integration.py` — `AlertIntegration` glue layer
- `config/default.yaml` — Expanded `storage` section (already done in Phase 7)
- `tests/test_clip_capture.py` — 40 tests, all passing
- `scripts/demo_clip_capture.py` — Full pipeline demo with clip output

**Tasks completed:**
- [x] Define `ClipRequest`, `ClipMetadata`, `StorageStats`
- [x] Implement `RollingFrameBuffer` with optional JPEG compression
- [x] Implement `ClipEncoder` (OpenCV MP4)
- [x] Implement `ClipCaptureService` with async processing and retention
- [x] Implement `AlertIntegration` glue layer
- [x] Write `tests/test_clip_capture.py` — 40 tests, all passing
- [x] Create `scripts/demo_clip_capture.py`
- [x] 242/242 total tests passing, zero regressions
