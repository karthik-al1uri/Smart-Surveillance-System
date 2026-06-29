# Pending Phases — Smart Surveillance System

---

## Phase 1: Stream Ingestion & Frame Extraction
**Status:** NOT STARTED
**Goal:** Connect to IP camera RTSP streams, decode video, extract frames, and push them into a processing queue.

**Tasks:**
- [ ] Implement `src/ingestion/stream_reader.py` — RTSP connection via OpenCV, with auto-reconnect on disconnect
- [ ] Implement `src/ingestion/frame_buffer.py` — Thread-safe frame queue with configurable max size and frame dropping
- [ ] Add frame downsampling logic (skip frames to achieve 2-4 fps effective rate)
- [ ] Add camera health monitoring (detect frozen/black frames)
- [ ] Support multiple simultaneous camera streams (one thread/process per camera)
- [ ] Write `tests/test_ingestion.py` 
- [ ] Add camera stream URLs to `config/default.yaml` 

**Key Decisions:**
- Use multiprocessing (one process per camera) for true parallelism
- Frame queue size: 30 frames max per camera (drop oldest if full)
- Target output rate: 2 fps per camera

⚠️ HUMAN INPUT REQUIRED:
- Provide at least one RTSP stream URL for testing (IP camera or test stream)
- Alternative: Use a local video file as a simulated stream for development

---

## Phase 2: Object Detection (YOLOv8)
**Status:** NOT STARTED
**Goal:** Detect persons, weapons, bags, and vehicles in each frame using YOLOv8.

**Tasks:**
- [ ] Implement `src/detection/yolo_detector.py` — Load YOLOv8 model, run inference on preprocessed frames
- [ ] Add frame preprocessing (resize to 640x640, normalize)
- [ ] Implement batched inference for multi-camera GPU efficiency
- [ ] Configure confidence threshold (default 0.45) and NMS IoU threshold (default 0.5)
- [ ] Output structured detections: class_id, class_name, bbox, confidence
- [ ] Write `tests/test_detection.py` with sample images

**Key Decisions:**
- Start with YOLOv8m (medium) — balance of speed and accuracy
- Use PyTorch first, export to ONNX/TensorRT for production later
- COCO pretrained initially; fine-tune for weapons in Phase 12+

⚠️ HUMAN INPUT REQUIRED:
- Download YOLOv8m weights: `yolov8m.pt` → place in `models/` directory
- Command: `pip install ultralytics && yolo predict model=yolov8m.pt` (will auto-download)
- If custom weapon detection is needed later, provide annotated weapon dataset

---

## Phase 3: Pose Estimation (YOLOv8-Pose)
**Status:** NOT STARTED
**Goal:** Extract 17-point COCO skeleton keypoints for each detected person.

**Tasks:**
- [ ] Implement `src/detection/pose_estimator.py` — Run YOLOv8-Pose on frames
- [ ] Output per-person keypoints: 17 joints × (x, y, confidence)
- [ ] Handle occluded/missing keypoints (low confidence filtering)
- [ ] Optionally combine with Phase 2 (YOLOv8-Pose does detection + pose in one pass)
- [ ] Write `tests/test_pose.py` 

**Key Decisions:**
- Use YOLOv8m-Pose for single-pass detection + pose (saves GPU compute)
- If using single-pass, Phase 2's standalone detector becomes a fallback/alternative

⚠️ HUMAN INPUT REQUIRED:
- Download YOLOv8m-Pose weights: `yolov8m-pose.pt` → place in `models/` 

---

## Phase 4: Person Tracking (ByteTrack / BoT-SORT)
**Status:** NOT STARTED
**Goal:** Assign consistent IDs to detected persons across frames. Maintain identity through brief occlusions.

**Tasks:**
- [ ] Implement `src/detection/tracker.py` — Integrate ByteTrack or BoT-SORT
- [ ] Maintain per-track history: bbox sequence, keypoint sequence, track state (active/lost/removed)
- [ ] Keep keypoint history buffer ≥ sliding window size (32 frames minimum)
- [ ] Handle track creation, update, and deletion lifecycle
- [ ] Write `tests/test_tracker.py` 

**Key Decisions:**
- Start with ByteTrack (simpler, IoU-based, no appearance features needed)
- Max lost frames before track deletion: 30 frames
- Re-identification: IoU-based only for now (appearance features in Phase 12+)

**No human input required for this phase.**

---

## Phase 5: Action Recognition (Keypoint Classifier)
**Status:** NOT STARTED
**Goal:** Classify activity of each tracked person over a temporal window (16-32 frames).

**Tasks:**
- [ ] Implement `src/recognition/keypoint_classifier.py` — LSTM/GRU network on keypoint sequences
- [ ] Implement `src/recognition/sliding_window.py` — Manage temporal windows per track
- [ ] Activity classes: Normal, Violent (fight/assault), Suspicious (loitering/intrusion), Urgent (fall/collapse)
- [ ] Output: class_label, confidence_score, sub_label
- [ ] Implement overlapping windows (classify every 8 frames, not every 16) to reduce latency
- [ ] Write `tests/test_recognition.py` 
- [ ] Create `training/models/keypoint_lstm.py` — Training script for the LSTM classifier

**Key Decisions:**
- Approach A (keypoint-based LSTM) for initial version — fast, lightweight, privacy-preserving
- Temporal window: 16 frames at 2 fps = 8 seconds
- Overlap stride: 8 frames (classify every 4 seconds)

⚠️ HUMAN INPUT REQUIRED:
- For training: Need action recognition dataset. Options:
  1. UCF-Crime dataset (download script in `scripts/download_ucf_crime.sh`)
  2. RWF-2000 dataset (violence detection)
  3. NTU RGB+D (skeleton-based action recognition)
  4. Custom annotated dataset from your own CCTV footage
- Provide preference on which dataset(s) to use
- Training requires GPU — confirm GPU availability for training

---

## Phase 6: Zone & Rule Engine
**Status:** NOT STARTED
**Goal:** Apply spatial and temporal rules to detections. Check restricted zone violations.

**Tasks:**
- [ ] Implement `src/scoring/zone_engine.py` — Point-in-polygon checks using Shapely
- [ ] Store zone polygons per camera in config/database
- [ ] Define rule schema: zone_id, time_window, allowed_classes, dwell_time_limit
- [ ] Evaluate rules: "Person in Zone A after 10 PM", "Loitering > 5 min in Zone B"
- [ ] Output rule violations with zone_id and violation details
- [ ] Write `tests/test_zone_engine.py` 

**No human input required for this phase.**

---

## Phase 7: Anomaly Scoring Engine
**Status:** NOT STARTED
**Goal:** Aggregate all signals into a severity score. Make alert decisions.

**Tasks:**
- [ ] Implement `src/scoring/anomaly_scorer.py` — Weighted scoring formula
- [ ] Inputs: action classification, object detection (weapons), zone violations, time context
- [ ] Configurable weights and thresholds per camera
- [ ] Cooldown logic: suppress duplicate alerts for ongoing events
- [ ] Hysteresis: require N consecutive high-confidence windows before alerting
- [ ] Write `tests/test_scoring.py` 

**Scoring formula (default):**
```
score = 0.4 × action_confidence + 0.3 × weapon_detected + 0.2 × zone_violation + 0.1 × time_risk
```

**No human input required for this phase.**

---

## Phase 8: Clip Capture & Storage Service
**Status:** NOT STARTED
**Goal:** When alert triggers, extract and store a 10-15 second video clip from the frame buffer.

**Tasks:**
- [ ] Implement `src/alerts/clip_capture.py` — Rolling frame buffer (20 sec per camera)
- [ ] On trigger: extract 10 sec before + 5 sec after event timestamp
- [ ] Encode clip as MP4 (H.264) using FFmpeg
- [ ] Store with metadata: event_id, camera_id, timestamp, file_path
- [ ] Implement retention policy (auto-delete after configurable days, default 30)
- [ ] Write `tests/test_clip_capture.py` 

**Key Decisions:**
- Storage: local disk for single-node, MinIO for multi-node
- Clip naming: `{camera_id}_{timestamp}_{event_id}.mp4` 

**No human input required for this phase.**

---

## Phase 9: Alert & Notification Service
**Status:** NOT STARTED
**Goal:** Deliver alerts through configured channels.

**Tasks:**
- [ ] Implement `src/alerts/alert_manager.py` — Central alert dispatcher
- [ ] Implement `src/alerts/notifiers/websocket.py` — Real-time dashboard push
- [ ] Implement `src/alerts/notifiers/webhook.py` — Third-party integration
- [ ] Implement `src/alerts/notifiers/email.py` — SMTP email alerts
- [ ] Add rate limiting (max 10 alerts per minute per camera)
- [ ] Add alert grouping (same incident across cameras)
- [ ] Add escalation rules (if no ACK in 5 min → escalate)
- [ ] Write `tests/test_alerts.py` 

⚠️ HUMAN INPUT REQUIRED:
- For email alerts: SMTP server credentials (host, port, username, password)
- For webhook: Target webhook URL(s) for testing
- These can be added later — the service should work with WebSocket-only initially

---

## Phase 10: Event Database & API Layer
**Status:** NOT STARTED
**Goal:** Persist all events and serve data via REST API.

**Tasks:**
- [ ] Implement `src/common/db.py` — PostgreSQL connection via SQLAlchemy
- [ ] Implement `src/common/models.py` — ORM models: cameras, events, alerts, feedback, logs
- [ ] Implement `src/api/main.py` — FastAPI app with CORS, auth middleware
- [ ] API routes: Camera CRUD, Event list/detail/search, Alert ACK/escalate/dismiss, Feedback submit, Zone CRUD, Config update, WebSocket live alerts
- [ ] Add Alembic for database migrations
- [ ] Write `tests/test_api.py` 

⚠️ HUMAN INPUT REQUIRED:
- PostgreSQL connection string (or confirm using Docker PostgreSQL)
- Confirm: should we use Docker Compose to spin up PostgreSQL automatically? (Recommended: YES)

---

## Phase 11: Operator Dashboard (React Frontend)
**Status:** NOT STARTED
**Goal:** Web-based UI for live monitoring, alert review, zone editing, and analytics.

**Tasks:**
- [ ] Initialize React app in `src/dashboard/` 
- [ ] Live Grid page: Multi-camera view with AI overlay
- [ ] Alert Queue page: Filterable alert list
- [ ] Event Detail page: Clip playback, frame overlay, feedback buttons
- [ ] Zone Editor page: Draw polygons on camera view
- [ ] Analytics page: Event heatmaps, trends
- [ ] Settings page: Camera management, thresholds, notifications, users
- [ ] WebSocket integration for real-time updates
- [ ] Role-based access control (admin/operator/viewer)

⚠️ HUMAN INPUT REQUIRED:
- UI/UX preferences or mockups (optional — can use defaults)
- Authentication method: basic auth, JWT, or OAuth?

---

## Phase 12: Model Management & Hot Swap
**Status:** NOT STARTED
**Goal:** Manage model versions, enable hot-swapping without restart.

**Tasks:**
- [ ] Implement `src/common/model_manager.py` — Model registry and loader
- [ ] Support model versioning and A/B testing
- [ ] Rollback mechanism if new model underperforms
- [ ] API endpoint to upload/activate new model versions

**No human input required for this phase.**

---

## Phase 13: Docker Containerization & Deployment
**Status:** NOT STARTED
**Goal:** Containerize all services for reproducible deployment.

**Tasks:**
- [ ] Write `docker/Dockerfile.inference` — GPU inference container
- [ ] Write `docker/Dockerfile.api` — API server container
- [ ] Write `docker/Dockerfile.dashboard` — Frontend container
- [ ] Write `docker-compose.yml` — Full stack orchestration
- [ ] Add NVIDIA Container Toolkit support for GPU passthrough
- [ ] Write `docs/deployment-guide.md` 

⚠️ HUMAN INPUT REQUIRED:
- Confirm target deployment hardware (GPU model, RAM, storage)
- Confirm: edge deployment (local server) or cloud (AWS/GCP)?

---

## Phase 14: Testing, Benchmarking & CI/CD
**Status:** NOT STARTED
**Goal:** Comprehensive testing and automated pipelines.

**Tasks:**
- [ ] Integration tests across full pipeline
- [ ] Benchmark inference speed (FPS per camera per GPU)
- [ ] Benchmark latency (event → alert delivery time)
- [ ] Set up GitHub Actions CI pipeline
- [ ] Add pre-commit hooks (black, ruff, mypy)
- [ ] Write `docs/training-roadmap.md` 

**No human input required for this phase.**
