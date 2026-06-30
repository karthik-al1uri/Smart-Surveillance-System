#!/usr/bin/env python3
"""
Full system audit for the Smart Surveillance System.

Run every component end-to-end, identify what works, what's broken, and what
still needs to be provided.  All output is written to audit/.

This script deliberately does NOT fix anything — it only tests and documents.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "audit"
TEST_CLIPS_DIR = AUDIT_DIR / "test_clips"


def ensure_dirs():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def run_shell(cmd: str, cwd: Path | None = None, log_file: Path | None = None, timeout: int | None = 300) -> tuple[int, str, str]:
    """Run a shell command, optionally logging to file.  Returns (rc, stdout, stderr)."""
    if cwd is None:
        cwd = PROJECT_ROOT
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        combined = f"TIMEOUT after {timeout}s\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        if log_file:
            log_file.write_text(combined, encoding="utf-8")
        return -1, stdout, stderr

    combined = f"COMMAND: {cmd}\nCWD: {cwd}\nEXIT CODE: {proc.returncode}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
    if log_file:
        log_file.write_text(combined, encoding="utf-8")
    return proc.returncode, proc.stdout, proc.stderr


def run_python(name: str, code: str, log_file: Path) -> str:
    """Execute a Python snippet via subprocess and write combined output."""
    rc, out, err = run_shell(
        f'{sys.executable} -c "{code}"',
        log_file=log_file,
        timeout=120,
    )
    return out + err


def section_header(title: str) -> str:
    return f"\n{'=' * 70}\n{title}\n{'=' * 70}\n"


def write_log(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# PART 1: Environment & Dependencies
# ---------------------------------------------------------------------------

def audit_environment():
    out = section_header("PART 1.1: Python Environment & Dependencies")
    out += f"Project root: {PROJECT_ROOT}\n"
    out += f"Python executable: {sys.executable}\n"

    rc, pyout, _ = run_shell(f"{sys.executable} --version")
    out += f"Python version: {pyout.strip()}\n"

    venv_path = PROJECT_ROOT / "venv"
    out += f"venv exists: {venv_path.exists()}\n"

    rc, pipout, _ = run_shell(f"{sys.executable} -m pip list --format=columns")
    out += f"\nInstalled packages:\n{pipout}\n"
    write_log(AUDIT_DIR / "dependency_list.txt", out)
    return out


def audit_tests():
    out = section_header("PART 1.2: Full Python Test Suite")
    rc, stdout, stderr = run_shell(
        f"{sys.executable} -m pytest tests/ -v --tb=short",
        log_file=AUDIT_DIR / "test_results.txt",
        timeout=600,
    )
    out += f"Exit code: {rc}\n"
    out += f"Output saved to audit/test_results.txt\n"
    write_log(AUDIT_DIR / "test_results_summary.txt", out)
    return out, stdout, stderr


def audit_frontend():
    out = section_header("PART 1.3: Frontend Build & Tests")
    dashboard = PROJECT_ROOT / "src" / "dashboard"

    if not dashboard.exists():
        out += "Dashboard directory not found — skipping frontend checks.\n"
        write_log(AUDIT_DIR / "frontend_build.txt", out)
        write_log(AUDIT_DIR / "frontend_tests.txt", out)
        return out

    run_shell("npm install", cwd=dashboard, log_file=AUDIT_DIR / "npm_install.txt", timeout=300)
    run_shell("npm run build", cwd=dashboard, log_file=AUDIT_DIR / "frontend_build.txt", timeout=300)
    run_shell("npm test", cwd=dashboard, log_file=AUDIT_DIR / "frontend_tests.txt", timeout=300)

    out += "npm install output: audit/npm_install.txt\n"
    out += "build output:      audit/frontend_build.txt\n"
    out += "test output:       audit/frontend_tests.txt\n"
    return out


# ---------------------------------------------------------------------------
# PART 1.4: Model Files
# ---------------------------------------------------------------------------

def audit_models():
    out = section_header("PART 1.4: Model Files & Load Check")
    models_dir = PROJECT_ROOT / "models"
    out += f"models/ directory exists: {models_dir.exists()}\n"
    if models_dir.exists():
        files = list(models_dir.iterdir())
        out += f"Files found: {len(files)}\n"
        for f in files:
            out += f"  {f.name} ({f.stat().st_size} bytes)\n"
    else:
        out += "  (directory missing)\n"

    yolo_path = models_dir / "yolov8m.pt"
    pose_path = models_dir / "yolov8m-pose.pt"
    action_path = models_dir / "action_classifier_v1.pt"

    if yolo_path.exists():
        try:
            from ultralytics import YOLO
            m = YOLO(str(yolo_path))
            out += f"\nYOLOv8m detection: OK — {len(m.names)} classes\n"
        except Exception as e:
            out += f"\nYOLOv8m detection: ERROR — {e}\n"
            out += traceback.format_exc()
    else:
        out += "\nYOLOv8m detection: MISSING — needs model file at models/yolov8m.pt\n"

    if pose_path.exists():
        try:
            from ultralytics import YOLO
            m = YOLO(str(pose_path))
            out += f"YOLOv8m-pose: OK — {len(m.names)} classes\n"
        except Exception as e:
            out += f"YOLOv8m-pose: ERROR — {e}\n"
    else:
        out += "YOLOv8m-pose: MISSING — needs model file at models/yolov8m-pose.pt\n"

    if action_path.exists():
        try:
            import torch
            data = torch.load(action_path, map_location="cpu")
            out += f"Action classifier: OK — keys={list(data.keys()) if isinstance(data, dict) else type(data)}\n"
            size = action_path.stat().st_size
            if size < 100_000:
                out += "  WARNING: file is very small — likely untrained dummy weights\n"
        except Exception as e:
            out += f"Action classifier: ERROR — {e}\n"
    else:
        out += "Action classifier: MISSING — needs training or model file at models/action_classifier_v1.pt\n"

    try:
        import torch
        out += f"\nDevice availability:\n"
        out += f"  CUDA available: {torch.cuda.is_available()}\n"
        if torch.cuda.is_available():
            out += f"  GPU: {torch.cuda.get_device_name(0)}\n"
            out += f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n"
        else:
            out += "  Running CPU-only mode\n"
    except Exception as e:
        out += f"\nDevice check: ERROR — {e}\n"

    write_log(AUDIT_DIR / "model_check.txt", out)
    return out


# ---------------------------------------------------------------------------
# PART 1.5: Database
# ---------------------------------------------------------------------------

def audit_database():
    out = section_header("PART 1.5: Database Check")
    try:
        import docker
    except Exception:
        pass

    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True, timeout=10)
        if "sss_postgres" in result.stdout:
            out += "PostgreSQL container sss_postgres: RUNNING\n"
        else:
            out += "PostgreSQL container sss_postgres: NOT RUNNING — will use SQLite fallback\n"
    except Exception as e:
        out += f"Docker check: {e} — will use SQLite fallback\n"

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.common.config import load_config
        from src.common.db import create_engine_from_config
        from sqlalchemy import text
        config = load_config()
        engine = create_engine_from_config(config.get("database", {}))
        out += f"Database URL: {engine.url}\n"
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            out += "Connection: OK\n"
    except Exception as e:
        out += f"Database connection: ERROR — {e}\n"
        out += traceback.format_exc()

    write_log(AUDIT_DIR / "db_check.txt", out)
    return out


# ---------------------------------------------------------------------------
# PART 2: Component Smoke Tests
# ---------------------------------------------------------------------------

def audit_component(name: str, test_fn, log_name: str) -> str:
    out = section_header(f"PART 2: {name}")
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        result = test_fn()
        out += str(result) if result else "(no output)\n"
    except Exception as e:
        out += f"ERROR — {e}\n"
        out += traceback.format_exc()
    write_log(AUDIT_DIR / log_name, out)
    return out


# Step 2.1: Ingestion

def test_ingestion():
    out = ""
    test_video = PROJECT_ROOT / "assets" / "test_video.mp4"
    out += f"Test video: {test_video}\n"
    out += f"  exists: {test_video.exists()}\n"

    if not test_video.exists():
        out += "Searching for any video file...\n"
        found = False
        for root, dirs, files in os.walk(PROJECT_ROOT):
            for f in files:
                if f.endswith((".mp4", ".avi", ".mkv")):
                    out += f"  Found: {os.path.join(root, f)}\n"
                    found = True
                    break
            if found:
                break
        if not found:
            out += "  No video files found.\n"

    try:
        import time
        import numpy as np
        from src.ingestion.frame_buffer import FrameBuffer
        buf = FrameBuffer(max_size=10)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        buf.put({"camera_id": "test", "frame_id": 0, "timestamp": time.time(), "frame": dummy, "resolution": (640, 480)})
        result = buf.get(timeout=1.0)
        out += f"Frame buffer: {'OK' if result is not None else 'FAILED'}\n"
    except Exception as e:
        out += f"Frame buffer: ERROR — {e}\n"

    if test_video.exists():
        try:
            import time
            from src.ingestion.stream_reader import StreamReader
            cam_config = {"id": "test_cam", "url": str(test_video), "name": "Test", "fps_override": 2, "enabled": True}
            reader = StreamReader(cam_config)
            reader.start()
            time.sleep(3)
            status = reader.get_status() if hasattr(reader, "get_status") else "N/A"
            reader.stop()
            out += f"Stream reader: OK (status: {status})\n"
        except Exception as e:
            out += f"Stream reader: ERROR — {e}\n"
    else:
        out += "Stream reader: SKIPPED (no test video)\n"
    return out


# Step 2.2: Detection

def test_detection():
    out = ""
    try:
        import numpy as np
        from src.common.config import load_config
        from src.detection.preprocessor import FramePreprocessor
        prep = FramePreprocessor(input_size=640)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed, meta = prep.preprocess_frame(test_frame)
        out += f"Preprocessor: OK (output shape: {processed.shape}, meta={meta})\n"
    except Exception as e:
        out += f"Preprocessor: ERROR — {e}\n"

    try:
        import numpy as np
        import cv2
        from src.common.config import load_config
        from src.detection.yolo_detector import YOLODetector
        config = load_config()
        detector = YOLODetector(config)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = detector.detect_single(test_frame)
        out += f"YOLO random noise: OK ({len(dets)} detections)\n"

        test_video = PROJECT_ROOT / "assets" / "test_video.mp4"
        if test_video.exists():
            cap = cv2.VideoCapture(str(test_video))
            ret, frame = cap.read()
            cap.release()
            if ret:
                real_dets = detector.detect_single(frame)
                out += f"YOLO on test video frame: OK ({len(real_dets)} detections)\n"
                for d in real_dets[:5]:
                    out += f"  {getattr(d, 'class_name', '?')}: {getattr(d, 'confidence', 0):.2f} bbox={getattr(d, 'bbox', None)}\n"
        else:
            out += "YOLO on real frame: SKIPPED (no test video)\n"
    except Exception as e:
        out += f"YOLO detector: ERROR — {e}\n"
        out += traceback.format_exc()
    return out


# Step 2.3: Pose estimation

def test_pose():
    out = ""
    try:
        import cv2
        import numpy as np
        from src.common.config import load_config
        from src.detection.pose_estimator import PoseEstimator
        config = load_config()
        estimator = PoseEstimator(config)
        test_video = PROJECT_ROOT / "assets" / "test_video.mp4"
        if test_video.exists():
            cap = cv2.VideoCapture(str(test_video))
            ret, frame = cap.read()
            cap.release()
            if ret:
                poses = estimator.estimate_single(frame)
                out += f"Pose estimator: OK ({len(poses)} persons)\n"
                for i, p in enumerate(poses[:3]):
                    kps = getattr(p, "keypoints", [])
                    visible = sum(1 for k in kps if getattr(k, "confidence", 0) > 0.3)
                    out += f"  Person {i+1}: {visible}/17 keypoints visible, bbox_conf={getattr(p, 'bbox_confidence', 0):.2f}\n"
        else:
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            poses = estimator.estimate_single(frame)
            out += f"Pose estimator on random noise: OK ({len(poses)} poses)\n"
    except Exception as e:
        out += f"Pose estimator: ERROR — {e}\n"
        out += traceback.format_exc()
    return out


# Step 2.4: Tracking

def test_tracker():
    out = ""
    try:
        from src.common.config import load_config
        from src.detection.tracker import ByteTracker
        config = load_config()
        tracker = ByteTracker(config)
        out += f"ByteTracker: OK (initialized)\n"
        methods = [m for m in dir(tracker) if not m.startswith("_")]
        out += f"  Methods: {methods[:20]}...\n"
    except Exception as e:
        out += f"ByteTracker: ERROR — {e}\n"
        out += traceback.format_exc()
    return out


# Step 2.5: Action Recognition

def test_action_recognition():
    out = ""
    try:
        import torch
        from src.recognition.keypoint_lstm import KeypointLSTM
        model = KeypointLSTM(input_size=51, hidden_size=128, num_layers=2, num_classes=15)
        dummy = torch.randn(1, 16, 51)
        output = model(dummy)
        out += f"KeypointLSTM forward: OK (output shape: {output.shape})\n"
    except Exception as e:
        out += f"KeypointLSTM: ERROR — {e}\n"

    try:
        import os
        from src.common.config import load_config
        from src.recognition.action_classifier import ActionClassifier
        config = load_config()
        clf = ActionClassifier(config)
        out += "Action classifier: OK (initialized)\n"
        model_path = config.get("action_recognition", {}).get("model_path", "models/action_classifier_v1.pt")
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            out += f"  Model file: {model_path} ({size / 1024:.0f} KB)\n"
            if size < 100_000:
                out += "  WARNING: model is very small — likely UNTRAINED dummy weights\n"
        else:
            out += f"  Model file not found: {model_path} — using random weights\n"
        out += "  STATUS: predictions will be RANDOM until trained on a real dataset\n"
    except Exception as e:
        out += f"Action classifier: ERROR — {e}\n"
        out += traceback.format_exc()

    try:
        import numpy as np
        from src.recognition.sliding_window import SlidingWindowManager
        from src.common.config import load_config
        config = load_config()
        wm = SlidingWindowManager(config)
        for i in range(20):
            kps = np.random.randn(17, 3).astype(np.float32)
            wm.update(track_id=1, keypoints=kps, frame_id=i)
        windows = wm.get_ready_windows()
        out += f"Sliding window: OK ({len(windows)} windows ready after 20 frames)\n"
    except Exception as e:
        out += f"Sliding window: ERROR — {e}\n"
        out += traceback.format_exc()
    return out


# Step 2.6: Zone Engine

def test_zone():
    out = ""
    try:
        from src.scoring.zone_models import Zone, ZoneType
        from src.scoring.zone_manager import ZoneManager
        from src.scoring.zone_engine import ZoneEngine
        zm = ZoneManager({})
        test_zone = Zone(
            zone_id="test_zone_1",
            camera_id="test_cam",
            name="Test Restricted Zone",
            zone_type=ZoneType.RESTRICTED,
            polygon=[(100, 100), (400, 100), (400, 400), (100, 400)],
        )
        zm.add_zone(test_zone)
        out += f"Zone manager: OK ({len(zm.get_all_zones())} zones)\n"
        engine = ZoneEngine(zm, {})
        if hasattr(engine, "is_point_in_zone"):
            result = engine.is_point_in_zone((250, 250), test_zone)
            out += f"Point-in-polygon: {result}\n"
        else:
            out += "Point-in-polygon: method not found\n"
    except Exception as e:
        out += f"Zone engine: ERROR — {e}\n"
        out += traceback.format_exc()
    return out


# Step 2.7: Anomaly Scorer

def test_scorer():
    out = ""
    try:
        import time
        from src.common.config import load_config
        from src.scoring.anomaly_scorer import AnomalyScorer
        from src.recognition.action_classes import ActionCategory, ActionLabel, ActionPrediction
        from src.scoring.scoring_models import ScoringSignal, SignalType
        config = load_config()
        scorer = AnomalyScorer(config)
        pred = ActionPrediction(
            track_id=1, camera_id="test_cam",
            timestamp=time.time(),
            category=ActionCategory.VIOLENT,
            label=ActionLabel.FIGHTING,
            confidence=0.85,
            category_probabilities={},
            window_start_frame=0, window_end_frame=15,
            keypoint_quality=0.9,
        )
        events = scorer.score_frame(
            camera_id="test_cam",
            timestamp=time.time(),
            action_predictions=[pred],
            zone_violations=[],
            object_detections=[],
        )
        out += "Anomaly scorer: OK\n"
        for e in events:
            out += f"  Score: {getattr(e, 'severity_score', 0):.3f} | Decision: {getattr(e, 'alert_decision', '?')} | Category: {getattr(e, 'event_category', '?')}\n"
    except Exception as e:
        out += f"Anomaly scorer: ERROR — {e}\n"
        out += traceback.format_exc()
    return out


# Step 2.8: Clip Capture

def test_clip_capture():
    out = ""
    try:
        import time
        import numpy as np
        from src.alerts.rolling_buffer import RollingFrameBuffer
        buf = RollingFrameBuffer(camera_id="test_cam", buffer_duration=5.0, target_fps=5.0)
        for i in range(30):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            buf.add_frame(frame, time.time() - 5 + (i * 0.2), i)
        time_range = buf.get_buffer_time_range()
        out += f"Rolling buffer: OK (covers {time_range[1] - time_range[0]:.1f}s)\n"

        mid = (time_range[0] + time_range[1]) / 2
        frames = buf.extract_clip(mid, pre_seconds=2, post_seconds=1)
        if frames:
            out += f"Clip extraction: OK ({len(frames)} frames)\n"
            try:
                from src.alerts.clip_encoder import ClipEncoder
                from src.alerts.clip_models import ClipRequest
                encoder = ClipEncoder({"storage": {"clip_dir": str(TEST_CLIPS_DIR), "clip_fps": 5, "clip_codec": "mp4v", "clip_quality": 85, "max_clip_duration": 20}})
                req = ClipRequest(event_id="test_evt", camera_id="test_cam", event_timestamp=mid)
                metadata = encoder.encode_clip(frames, req, str(TEST_CLIPS_DIR))
                if metadata:
                    out += f"Clip encoding: OK ({getattr(metadata, 'file_path', None)}, {getattr(metadata, 'file_size_bytes', 0)} bytes, {getattr(metadata, 'duration_seconds', 0):.1f}s)\n"
                else:
                    out += "Clip encoding: FAILED\n"
            except Exception as e:
                out += f"Clip encoding: ERROR — {e}\n"
                out += traceback.format_exc()
        else:
            out += "Clip extraction: FAILED (no frames)\n"
    except Exception as e:
        out += f"Clip capture: ERROR — {e}\n"
        out += traceback.format_exc()
    return out


# Step 2.9: Alert Service

def test_alert_service():
    out = ""
    try:
        import time
        from src.alerts.alert_builder import AlertBuilder
        from src.scoring.scoring_models import ScoredEvent, AlertDecision, SignalType, ScoringSignal
        builder = AlertBuilder({})
        event = ScoredEvent(
            event_id="test_001",
            camera_id="test_cam",
            track_id=1,
            timestamp=time.time(),
            severity_score=0.82,
            contributing_signals=[
                ScoringSignal(SignalType.ACTION_CLASSIFICATION, "action", 0.85, 0.35, 0.30, "Fighting detected")
            ],
            dominant_signal=SignalType.ACTION_CLASSIFICATION,
            event_category="violent",
            event_label="fighting",
            alert_decision=AlertDecision.ALERT,
        )
        alert = builder.build_alert(event)
        out += f"Alert builder: OK\n"
        out += f"  Title: {getattr(alert, 'title', '?')}\n"
        out += f"  Priority: {getattr(alert, 'priority', '?')}\n"
        desc = getattr(alert, "description", "") or ""
        out += f"  Description: {desc[:80]}...\n"
    except Exception as e:
        out += f"Alert builder: ERROR — {e}\n"
        out += traceback.format_exc()

    for name, cls in [
        ("WebSocket notifier", "src.alerts.notifiers.websocket_notifier.WebSocketNotifier"),
        ("Webhook notifier", "src.alerts.notifiers.webhook_notifier.WebhookNotifier"),
        ("Email notifier", "src.alerts.notifiers.email_notifier.EmailNotifier"),
    ]:
        try:
            mod_path, class_name = cls.rsplit(".", 1)
            mod = __import__(mod_path, fromlist=[class_name])
            Notifier = getattr(mod, class_name)
            if "WebSocket" in name:
                inst = Notifier({"notifications": {"websocket": {"enabled": True, "host": "0.0.0.0", "port": 18765, "ping_interval": 30, "ping_timeout": 10}}})
            elif "Webhook" in name:
                inst = Notifier({"notifications": {"webhook": {"enabled": False, "endpoints": []}}})
            else:
                inst = Notifier({"notifications": {"email": {"enabled": False, "smtp_host": "", "smtp_port": 587}}})
            out += f"{name}: OK (initialized, not started)\n"
        except Exception as e:
            out += f"{name}: ERROR — {e}\n"
    return out


# Step 2.10: API Server

def test_api():
    out = ""
    try:
        from fastapi.testclient import TestClient
        from src.api.main import create_app
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        from src.common.db_models import Base
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        app = create_app(config={
            "database": {"fallback_url": "sqlite:///:memory:", "auto_fallback": True, "echo_sql": False},
            "auth": {"secret_key": "audit-secret", "algorithm": "HS256", "access_token_expire_minutes": 60, "default_admin": {"username": "admin", "password": "admin", "role": "admin"}},
            "notifications": {"websocket": {"enabled": False}, "webhook": {"enabled": False, "endpoints": []}, "email": {"enabled": False, "smtp_host": ""}},
            "alerts": {"rate_limit": {"max_alerts_per_minute": 10, "max_alerts_per_minute_global": 30}, "grouping": {"enabled": False}, "escalation": {"enabled": False}, "history_size": 100},
            "scoring": {"escalation_threshold": 0.85},
            "dashboard_url": "http://localhost:3000",
        }, engine=engine)

        with TestClient(app) as client:
            resp = client.get("/api/v1/system/health")
            out += f"GET /system/health: {resp.status_code}\n"

            resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin"})
            out += f"POST /auth/login: {resp.status_code}\n"
            if resp.status_code == 200:
                token = resp.json().get("access_token", "")
                headers = {"Authorization": f"Bearer {token}"}
                out += "  Token obtained: OK\n"
                for endpoint in ["/api/v1/cameras", "/api/v1/events", "/api/v1/alerts", "/api/v1/models/status"]:
                    r = client.get(endpoint, headers=headers)
                    out += f"GET {endpoint}: {r.status_code}\n"
            else:
                out += f"  Login failed: {resp.text}\n"
    except Exception as e:
        out += f"API server: ERROR — {e}\n"
        out += traceback.format_exc()
    return out


# Step 2.11: Model Management

def test_model_management():
    out = ""
    try:
        from src.common.model_registry import ModelRegistry
        from src.common.model_manager_models import ModelType
        registry = ModelRegistry(str(PROJECT_ROOT / "models" / "registry.json"))
        models = registry.list_models()
        out += f"Model registry: OK ({len(models)} models registered)\n"
        for m in models:
            out += f"  {m.model_type.value}: v{m.version} — {m.path} [{m.status.value}]\n"
        for mt in ModelType:
            active = registry.get_active(mt)
            if active:
                out += f"  Active {mt.value}: v{active.version} OK\n"
            else:
                out += f"  Active {mt.value}: NONE\n"
    except Exception as e:
        out += f"Model management: ERROR — {e}\n"
        out += traceback.format_exc()
    return out


# ---------------------------------------------------------------------------
# PART 3: End-to-End Pipeline
# ---------------------------------------------------------------------------

def audit_e2e():
    out = section_header("PART 3: End-to-End Pipeline Test")
    test_video = PROJECT_ROOT / "assets" / "test_video.mp4"

    if not test_video.exists():
        out += "✗ No test video found at assets/test_video.mp4. Cannot run E2E test.\n"
        out += "  USER ACTION NEEDED: Provide a video file at assets/test_video.mp4\n"
        write_log(AUDIT_DIR / "e2e_test.txt", out)
        return out

    out += f"Test video: {test_video}\n"
    try:
        import cv2
        cap = cv2.VideoCapture(str(test_video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        out += f"  {width}x{height}, {native_fps:.0f}fps, {total_frames} frames ({total_frames / native_fps:.1f}s)\n"
    except Exception as e:
        out += f"  Video info error: {e}\n"
        write_log(AUDIT_DIR / "e2e_test.txt", out)
        return out

    try:
        import time
        import cv2
        import numpy as np
        from collections import Counter
        from src.common.config import load_config
        config = load_config()

        out += "\nLoading models...\n"
        detector = None
        pose_est = None
        tracker = None
        action_clf = None
        window_mgr = None
        zone_engine = None
        scorer = None

        try:
            from src.detection.yolo_detector import YOLODetector
            detector = YOLODetector(config)
            out += "  YOLO detector: OK\n"
        except Exception as e:
            out += f"  YOLO detector: ERROR — {e}\n"

        try:
            from src.detection.pose_estimator import PoseEstimator
            pose_est = PoseEstimator(config)
            out += "  Pose estimator: OK\n"
        except Exception as e:
            out += f"  Pose estimator: ERROR — {e}\n"

        try:
            from src.detection.tracker import ByteTracker
            tracker = ByteTracker(config)
            out += "  Tracker: OK\n"
        except Exception as e:
            out += f"  Tracker: ERROR — {e}\n"

        try:
            from src.recognition.action_classifier import ActionClassifier
            action_clf = ActionClassifier(config)
            out += "  Action classifier: OK\n"
        except Exception as e:
            out += f"  Action classifier: ERROR — {e}\n"

        try:
            from src.recognition.sliding_window import SlidingWindowManager
            window_mgr = SlidingWindowManager(config)
            out += "  Sliding window: OK\n"
        except Exception as e:
            out += f"  Sliding window: ERROR — {e}\n"

        try:
            from src.scoring.zone_engine import ZoneEngine
            from src.scoring.zone_manager import ZoneManager
            zone_mgr = ZoneManager(config)
            zone_engine = ZoneEngine(zone_mgr, config)
            out += "  Zone engine: OK\n"
        except Exception as e:
            out += f"  Zone engine: ERROR — {e}\n"

        try:
            from src.scoring.anomaly_scorer import AnomalyScorer
            scorer = AnomalyScorer(config)
            out += "  Anomaly scorer: OK\n"
        except Exception as e:
            out += f"  Anomaly scorer: ERROR — {e}\n"

        out += "\nProcessing frames...\n"
        cap = cv2.VideoCapture(str(test_video))
        max_frames = min(100, total_frames)
        target_fps = 2
        frame_skip = max(1, int(native_fps / target_fps))

        frame_count = 0
        detection_count = 0
        pose_count = 0
        track_ids_seen = set()
        action_predictions = []
        zone_violations_total = []
        scored_events = []
        errors = []

        start_time = time.time()
        for frame_idx in range(0, max_frames, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            timestamp = time.time()
            try:
                if detector:
                    detections = detector.detect_single(frame)
                    detection_count += len(detections)
                    person_dets = [d for d in detections if getattr(d, "class_name", "") == "person"]
                    object_dets = [d for d in detections if getattr(d, "class_name", "") != "person"]
                else:
                    detections = person_dets = object_dets = []

                if pose_est:
                    poses = pose_est.estimate_single(frame)
                    pose_count += len(poses)
                else:
                    poses = []

                tracks = []
                if tracker:
                    try:
                        from src.detection.combined_pipeline import FrameAnalysis
                        analysis = FrameAnalysis(
                            camera_id="test_cam", frame_id=frame_idx, timestamp=timestamp,
                            frame=frame, person_detections=person_dets, object_detections=object_dets,
                            poses=poses, inference_time_ms=0,
                        )
                        tracks = tracker.update(analysis)
                        for t in tracks:
                            track_ids_seen.add(getattr(t, "track_id", 0))
                    except Exception as te:
                        if frame_count == 1:
                            errors.append(f"Tracker update error: {te}")

                if window_mgr and action_clf:
                    for t in tracks:
                        kph = getattr(t, "keypoint_history", None)
                        if kph and len(kph) > 0:
                            latest_kp = kph[-1]
                            window_mgr.update(getattr(t, "track_id", 0), latest_kp, frame_idx)
                    ready_windows = window_mgr.get_ready_windows()
                    for w in ready_windows:
                        try:
                            pred = action_clf.classify(w)
                            action_predictions.append(pred)
                        except Exception as ae:
                            if len(errors) < 5:
                                errors.append(f"Action classify error: {ae}")

                if zone_engine:
                    try:
                        from src.detection.combined_pipeline import TrackedFrameAnalysis
                        tracked = TrackedFrameAnalysis(
                            camera_id="test_cam", frame_id=frame_idx, timestamp=timestamp,
                            frame=frame, person_detections=person_dets, object_detections=object_dets,
                            poses=poses, tracks=tracks, inference_time_ms=0, tracking_time_ms=0,
                        )
                        violations = zone_engine.evaluate(tracked)
                        zone_violations_total.extend(violations)
                    except Exception as ze:
                        if frame_count == 1:
                            errors.append(f"Zone engine error: {ze}")

                if scorer:
                    try:
                        events = scorer.score_frame(
                            camera_id="test_cam",
                            timestamp=timestamp,
                            action_predictions=action_predictions[-5:] if action_predictions else [],
                            zone_violations=zone_violations_total[-5:] if zone_violations_total else [],
                            object_detections=object_dets,
                        )
                        scored_events.extend(events)
                    except Exception as se:
                        if frame_count == 1:
                            errors.append(f"Scorer error: {se}")
            except Exception as e:
                if len(errors) < 10:
                    errors.append(f"Frame {frame_idx}: {e}")
        cap.release()
        elapsed = time.time() - start_time

        out += f"  Processed {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.1f} fps)\n"
        out += "\n" + "=" * 70 + "\nRESULTS\n" + "=" * 70 + "\n"
        out += f"Frames processed:      {frame_count}\n"
        out += f"Total detections:      {detection_count}\n"
        out += f"Total poses:           {pose_count}\n"
        out += f"Unique tracks:         {len(track_ids_seen)}\n"
        out += f"Action predictions:    {len(action_predictions)}\n"
        out += f"Zone violations:       {len(zone_violations_total)}\n"
        out += f"Scored events:         {len(scored_events)}\n"
        if scored_events:
            alerts = [e for e in scored_events if str(getattr(e, "alert_decision", "")).lower() in ("alert", "escalated")]
            suppressed = [e for e in scored_events if str(getattr(e, "alert_decision", "")).lower() == "suppressed"]
            out += f"  Alerts:     {len(alerts)}\n"
            out += f"  Suppressed: {len(suppressed)}\n"
            for e in alerts[:5]:
                out += f"    → {getattr(e, 'event_category', '?')}/{getattr(e, 'event_label', '?')} score={getattr(e, 'severity_score', 0):.2f} decision={getattr(e, 'alert_decision', '?')}\n"
        if action_predictions:
            cats = Counter(str(getattr(p, "category", getattr(p, "label", "unknown"))) for p in action_predictions)
            out += f"Action distribution:   {dict(cats)}\n"
            out += "  ⚠️  These are from an UNTRAINED model — predictions are meaningless\n"
        if errors:
            out += f"\nERRORS ({len(errors)}):\n"
            for e in errors:
                out += f"  ✗ {e}\n"
    except Exception as e:
        out += f"\nE2E pipeline: ERROR — {e}\n"
        out += traceback.format_exc()

    write_log(AUDIT_DIR / "e2e_test.txt", out)
    return out


# ---------------------------------------------------------------------------
# PART 4: Final Report
# ---------------------------------------------------------------------------

def generate_report():
    audit_files = [
        ("Dependency Check", "dependency_list.txt"),
        ("Test Results", "test_results.txt"),
        ("Frontend Build", "frontend_build.txt"),
        ("Frontend Tests", "frontend_tests.txt"),
        ("Model Check", "model_check.txt"),
        ("Database Check", "db_check.txt"),
        ("Phase 1: Ingestion", "phase1_test.txt"),
        ("Phase 2: Detection", "phase2_test.txt"),
        ("Phase 3: Pose", "phase3_test.txt"),
        ("Phase 4: Tracking", "phase4_test.txt"),
        ("Phase 5: Action Recognition", "phase5_test.txt"),
        ("Phase 6: Zone Engine", "phase6_test.txt"),
        ("Phase 7: Scoring", "phase7_test.txt"),
        ("Phase 8: Clip Capture", "phase8_test.txt"),
        ("Phase 9: Alert Service", "phase9_test.txt"),
        ("Phase 10: API", "phase10_test.txt"),
        ("Phase 12: Model Management", "phase12_test.txt"),
        ("End-to-End Test", "e2e_test.txt"),
    ]

    lines = []
    lines.append("=" * 70)
    lines.append("SMART SURVEILLANCE SYSTEM — FULL AUDIT REPORT")
    lines.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    for title, filename in audit_files:
        lines.append(f"--- {title} ---")
        path = AUDIT_DIR / filename
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            if len(content) > 3000:
                parts = content.splitlines()
                content = "\n".join(parts[:30]) + f"\n\n... ({len(parts) - 60} lines omitted) ...\n\n" + "\n".join(parts[-30:])
            lines.append(content)
        else:
            lines.append("  FILE NOT FOUND — test may not have run")
        lines.append("")

    lines.append("=" * 70)
    lines.append("SUMMARY: WHAT YOU NEED TO DO")
    lines.append("=" * 70)
    lines.append("")
    lines.append("ACTION ITEMS FOR THE USER:")
    lines.append("")
    lines.append("1. ACTION RECOGNITION TRAINING (REQUIRED for meaningful predictions)")
    lines.append("   The LSTM classifier has random weights. You need a labeled dataset.")
    lines.append("   Options:")
    lines.append("     a) NTU RGB+D 120 — largest skeleton action dataset")
    lines.append("     b) UCF-Crime — real CCTV footage (requires keypoint extraction)")
    lines.append("     c) Your own labeled CCTV clips")
    lines.append("   When ready, run the training scripts in training/.")
    lines.append("")
    lines.append("2. REAL CAMERA STREAMS (REQUIRED for production)")
    lines.append("   Add RTSP URLs to config/default.yaml under cameras.sources")
    lines.append("   Test with: python scripts/demo_ingestion.py")
    lines.append("")
    lines.append("3. MODEL WEIGHTS (REQUIRED for detection/pose)")
    lines.append("   Download or place the following in models/:")
    lines.append("     - yolov8m.pt (detection)")
    lines.append("     - yolov8m-pose.pt (pose estimation)")
    lines.append("     - action_classifier_v1.pt (trained classifier)")
    lines.append("")
    lines.append("4. NOTIFICATION CHANNELS (OPTIONAL)")
    lines.append("   - Webhook: add endpoints to config/default.yaml")
    lines.append("   - Email: add SMTP credentials to config/default.yaml")
    lines.append("")
    lines.append("5. PRODUCTION DATABASE (OPTIONAL — SQLite works for dev)")
    lines.append("   - Start PostgreSQL: docker-compose -f docker-compose.dev.yml up -d")
    lines.append("   - Run migrations: alembic upgrade head")
    lines.append("")
    lines.append("6. GPU DEPLOYMENT (OPTIONAL — CPU works but is slow)")
    lines.append("   - NVIDIA GPU + CUDA toolkit required")
    lines.append("   - Set detection.device and pose.device to 'cuda:0'")
    lines.append("")

    report = "\n".join(lines)
    (AUDIT_DIR / "FULL_AUDIT_REPORT.txt").write_text(report, encoding="utf-8")
    (AUDIT_DIR / "final_summary.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to: {AUDIT_DIR / 'FULL_AUDIT_REPORT.txt'}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    ensure_dirs()
    print("Starting full system audit...")
    print(f"All output will be written to {AUDIT_DIR}")

    audit_environment()
    audit_tests()
    audit_frontend()
    audit_models()
    audit_database()

    audit_component("Phase 1: Ingestion", test_ingestion, "phase1_test.txt")
    audit_component("Phase 2: Object Detection", test_detection, "phase2_test.txt")
    audit_component("Phase 3: Pose Estimation", test_pose, "phase3_test.txt")
    audit_component("Phase 4: Person Tracking", test_tracker, "phase4_test.txt")
    audit_component("Phase 5: Action Recognition", test_action_recognition, "phase5_test.txt")
    audit_component("Phase 6: Zone Engine", test_zone, "phase6_test.txt")
    audit_component("Phase 7: Anomaly Scoring", test_scorer, "phase7_test.txt")
    audit_component("Phase 8: Clip Capture", test_clip_capture, "phase8_test.txt")
    audit_component("Phase 9: Alert Service", test_alert_service, "phase9_test.txt")
    audit_component("Phase 10: API Server", test_api, "phase10_test.txt")
    audit_component("Phase 12: Model Management", test_model_management, "phase12_test.txt")

    audit_e2e()
    generate_report()


if __name__ == "__main__":
    main()
