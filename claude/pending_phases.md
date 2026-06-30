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
