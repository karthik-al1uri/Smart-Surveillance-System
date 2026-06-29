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
