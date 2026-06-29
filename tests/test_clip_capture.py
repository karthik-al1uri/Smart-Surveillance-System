"""Tests for Phase 8: Clip Capture & Storage Service.

All tests use synthetic data and pytest tmp_path for file I/O.
No real cameras or video sources are required.
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts.alert_integration import AlertIntegration
from src.alerts.clip_capture import ClipCaptureService
from src.alerts.clip_encoder import ClipEncoder
from src.alerts.clip_models import ClipMetadata, ClipRequest, StorageStats
from src.alerts.rolling_buffer import RollingFrameBuffer
from src.scoring.scoring_models import AlertDecision, ScoredEvent, SignalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(w: int = 64, h: int = 48) -> np.ndarray:
    return (np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))


def _make_config(tmp_path: Path) -> dict:
    return {
        "storage": {
            "clip_dir": str(tmp_path),
            "clip_codec": "mp4v",
            "clip_fps": 10,
            "clip_quality": 85,
            "max_clip_duration": 20,
            "retention_days": 30,
            "clip_buffer_duration": 20.0,
            "clip_buffer_fps": 10.0,
            "clip_buffer_compressed": False,
            "retention_check_interval": 3600,
        }
    }


def _make_scored_event(
    decision: AlertDecision = AlertDecision.ALERT,
    camera_id: str = "cam_01",
    track_id: int = 1,
) -> ScoredEvent:
    return ScoredEvent(
        event_id=f"evt_{uuid.uuid4().hex[:8]}",
        camera_id=camera_id,
        track_id=track_id,
        timestamp=time.time(),
        severity_score=0.75,
        contributing_signals=[],
        dominant_signal=SignalType.ACTION_CLASSIFICATION,
        event_category="violent",
        event_label="fighting",
        alert_decision=decision,
    )


def _make_frames_list(n: int = 100, start_ts: float = None) -> List[dict]:
    now = start_ts or time.time()
    fps = 10.0
    return [
        {"frame": _make_frame(), "timestamp": now + i / fps, "frame_id": i}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1–3: Data structures
# ---------------------------------------------------------------------------

def test_clip_request_dataclass():
    req = ClipRequest(event_id="evt_001", camera_id="cam_01", event_timestamp=time.time())
    assert req.pre_seconds == 10.0
    assert req.post_seconds == 5.0
    assert req.priority == 1


def test_clip_metadata_dataclass():
    now = time.time()
    meta = ClipMetadata(
        clip_id=str(uuid.uuid4()),
        event_id="evt_001",
        camera_id="cam_01",
        file_path="data/clips/cam_01/clip.mp4",
        file_size_bytes=1024 * 500,
        duration_seconds=14.8,
        start_timestamp=now - 10,
        end_timestamp=now + 5,
        event_timestamp=now,
        resolution=(640, 480),
        fps=10.0,
        codec="mp4v",
        created_at=now,
    )
    assert meta.duration_seconds == 14.8
    assert meta.resolution == (640, 480)
    assert meta.clip_id is not None


def test_storage_stats_dataclass():
    stats = StorageStats(
        total_clips=5,
        total_size_bytes=1024 * 1024 * 10,
        total_size_gb=round(1024 * 1024 * 10 / (1024 ** 3), 4),
        oldest_clip_timestamp=time.time() - 86400,
        newest_clip_timestamp=time.time(),
        clips_by_camera={"cam_01": 3, "cam_02": 2},
        storage_path="data/clips",
        retention_days=30,
    )
    assert stats.total_clips == 5
    assert stats.clips_by_camera["cam_01"] == 3


# ---------------------------------------------------------------------------
# 4–15: Rolling buffer
# ---------------------------------------------------------------------------

def test_rolling_buffer_add_frame():
    buf = RollingFrameBuffer("cam_01", buffer_duration=5.0, target_fps=10.0, compressed=False)
    buf.add_frame(_make_frame(), time.time(), 0)
    assert buf.get_stats()["frame_count"] == 1


def test_rolling_buffer_capacity():
    buf = RollingFrameBuffer("cam_01", buffer_duration=20.0, target_fps=10.0, compressed=False)
    now = time.time()
    # Add 250 frames spanning 25 seconds (1 per 0.1s)
    for i in range(250):
        buf.add_frame(_make_frame(), now + i * 0.1, i)
    stats = buf.get_stats()
    assert stats["frame_count"] <= stats["max_frames"]


def test_rolling_buffer_oldest_dropped():
    buf = RollingFrameBuffer("cam_01", buffer_duration=10.0, target_fps=10.0, compressed=False)
    now = time.time()
    # Fill to capacity (100 frames)
    for i in range(100):
        buf.add_frame(_make_frame(), now + i * 0.1, i)
    first_oldest, _ = buf.get_buffer_time_range()
    # Add 50 more — should bump the oldest
    for i in range(100, 150):
        buf.add_frame(_make_frame(), now + i * 0.1, i)
    new_oldest, _ = buf.get_buffer_time_range()
    # Oldest must have advanced (older frames were evicted)
    assert new_oldest >= first_oldest


def test_rolling_buffer_fps_downsampling():
    buf = RollingFrameBuffer("cam_01", buffer_duration=20.0, target_fps=10.0, compressed=False)
    now = time.time()
    # Add 300 frames at 30fps (every ~0.033s) over 10 seconds
    for i in range(300):
        buf.add_frame(_make_frame(), now + i / 30.0, i)
    stats = buf.get_stats()
    # Should store approximately target_fps × duration = 100 frames, not 300
    assert stats["frame_count"] <= 110  # a little slack for float rounding


def test_rolling_buffer_extract_clip():
    buf = RollingFrameBuffer("cam_01", buffer_duration=20.0, target_fps=10.0, compressed=False)
    now = time.time()
    for i in range(200):
        buf.add_frame(_make_frame(), now + i * 0.1, i)
    # Event at the middle
    event_ts = now + 10.0
    frames = buf.extract_clip(event_ts, pre_seconds=3.0, post_seconds=3.0)
    assert frames is not None
    assert len(frames) > 0
    for f in frames:
        assert now + 7.0 - 0.2 <= f["timestamp"] <= now + 13.0 + 0.2


def test_rolling_buffer_extract_clip_boundaries():
    buf = RollingFrameBuffer("cam_01", buffer_duration=20.0, target_fps=10.0, compressed=False)
    now = time.time()
    for i in range(50):
        buf.add_frame(_make_frame(), now + i * 0.1, i)
    # Event near start — pre-window extends before buffer start
    event_ts = now + 1.0
    frames = buf.extract_clip(event_ts, pre_seconds=10.0, post_seconds=2.0)
    # Returns partial clip (frames from buffer start)
    assert frames is not None
    assert len(frames) > 0


def test_rolling_buffer_extract_clip_too_old():
    buf = RollingFrameBuffer("cam_01", buffer_duration=10.0, target_fps=10.0, compressed=False)
    now = time.time()
    for i in range(100):
        buf.add_frame(_make_frame(), now + i * 0.1, i)
    # Event 60 seconds ago — outside buffer
    frames = buf.extract_clip(now - 60.0, pre_seconds=5.0, post_seconds=5.0)
    assert frames is None


def test_rolling_buffer_time_range():
    buf = RollingFrameBuffer("cam_01", buffer_duration=20.0, target_fps=10.0, compressed=False)
    now = time.time()
    for i in range(100):
        buf.add_frame(_make_frame(), now + i * 0.1, i)
    oldest, newest = buf.get_buffer_time_range()
    assert oldest <= newest
    assert newest - oldest >= 9.0  # should cover ~10 seconds


def test_rolling_buffer_compressed():
    buf_c = RollingFrameBuffer("cam_01", buffer_duration=5.0, target_fps=10.0, compressed=True, jpeg_quality=85)
    buf_u = RollingFrameBuffer("cam_01", buffer_duration=5.0, target_fps=10.0, compressed=False)
    now = time.time()
    for i in range(50):
        f = _make_frame(320, 240)
        buf_c.add_frame(f, now + i * 0.1, i)
        buf_u.add_frame(f, now + i * 0.1, i)
    stats_c = buf_c.get_stats()
    stats_u = buf_u.get_stats()
    # Compressed buffer should use less memory
    assert stats_c["estimated_memory_bytes"] < stats_u["estimated_memory_bytes"]


def test_rolling_buffer_compressed_quality():
    buf = RollingFrameBuffer("cam_01", buffer_duration=5.0, target_fps=10.0, compressed=True, jpeg_quality=85)
    frame = _make_frame(64, 48)
    now = time.time()
    buf.add_frame(frame, now, 0)
    frames = buf.extract_clip(now, pre_seconds=1.0, post_seconds=1.0)
    assert frames is not None and len(frames) == 1
    decoded = frames[0]["frame"]
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == frame.shape
    assert decoded.dtype == np.uint8


def test_rolling_buffer_thread_safety():
    buf = RollingFrameBuffer("cam_01", buffer_duration=10.0, target_fps=30.0, compressed=False)
    errors = []
    now = time.time()

    def writer():
        for i in range(100):
            try:
                buf.add_frame(_make_frame(), now + i * 0.033, i)
                time.sleep(0.001)
            except Exception as e:
                errors.append(e)

    def reader():
        for _ in range(20):
            try:
                buf.extract_clip(now + 1.0, 1.0, 1.0)
                time.sleep(0.005)
            except Exception as e:
                errors.append(e)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert errors == []


def test_rolling_buffer_clear():
    buf = RollingFrameBuffer("cam_01", buffer_duration=10.0, target_fps=10.0, compressed=False)
    now = time.time()
    for i in range(50):
        buf.add_frame(_make_frame(), now + i * 0.1, i)
    buf.clear()
    assert buf.get_stats()["frame_count"] == 0


# ---------------------------------------------------------------------------
# 16–21: Clip encoder
# ---------------------------------------------------------------------------

def test_encode_clip_creates_file(tmp_path):
    encoder = ClipEncoder({"storage": {"clip_dir": str(tmp_path), "clip_codec": "mp4v",
                                        "clip_fps": 10, "max_clip_duration": 20}})
    frames = _make_frames_list(30)
    req = ClipRequest(event_id="evt_test", camera_id="cam_01",
                      event_timestamp=frames[15]["timestamp"])
    meta = encoder.encode_clip(frames, req, str(tmp_path))
    assert meta is not None
    assert Path(meta.file_path).exists()


def test_encode_clip_metadata(tmp_path):
    encoder = ClipEncoder({"storage": {"clip_dir": str(tmp_path), "clip_codec": "mp4v",
                                        "clip_fps": 10, "max_clip_duration": 20}})
    frames = _make_frames_list(30)
    req = ClipRequest(event_id="evt_meta", camera_id="cam_01",
                      event_timestamp=frames[15]["timestamp"])
    meta = encoder.encode_clip(frames, req, str(tmp_path))
    assert meta is not None
    assert meta.resolution == (64, 48)
    assert meta.file_size_bytes > 0
    assert meta.duration_seconds > 0
    assert meta.camera_id == "cam_01"


def test_encode_clip_playable(tmp_path):
    import cv2 as cv
    encoder = ClipEncoder({"storage": {"clip_dir": str(tmp_path), "clip_codec": "mp4v",
                                        "clip_fps": 10, "max_clip_duration": 20}})
    frames = _make_frames_list(20)
    req = ClipRequest(event_id="evt_play", camera_id="cam_01",
                      event_timestamp=frames[10]["timestamp"])
    meta = encoder.encode_clip(frames, req, str(tmp_path))
    assert meta is not None
    cap = cv.VideoCapture(meta.file_path)
    assert cap.isOpened()
    ret, _ = cap.read()
    assert ret
    cap.release()


def test_encode_clip_directory_creation(tmp_path):
    nested = tmp_path / "new_dir" / "nested"
    encoder = ClipEncoder({"storage": {"clip_dir": str(nested), "clip_codec": "mp4v",
                                        "clip_fps": 10, "max_clip_duration": 20}})
    frames = _make_frames_list(10)
    req = ClipRequest(event_id="evt_dir", camera_id="cam_01",
                      event_timestamp=frames[5]["timestamp"])
    meta = encoder.encode_clip(frames, req, str(nested))
    assert meta is not None
    assert Path(meta.file_path).exists()


def test_encode_clip_empty_frames(tmp_path):
    encoder = ClipEncoder({"storage": {"clip_dir": str(tmp_path), "clip_codec": "mp4v",
                                        "clip_fps": 10, "max_clip_duration": 20}})
    req = ClipRequest(event_id="evt_empty", camera_id="cam_01",
                      event_timestamp=time.time())
    result = encoder.encode_clip([], req, str(tmp_path))
    assert result is None


def test_encode_clip_naming(tmp_path):
    encoder = ClipEncoder({"storage": {"clip_dir": str(tmp_path), "clip_codec": "mp4v",
                                        "clip_fps": 10, "max_clip_duration": 20}})
    frames = _make_frames_list(10)
    req = ClipRequest(event_id="evt_name", camera_id="cam_01",
                      event_timestamp=frames[5]["timestamp"])
    meta = encoder.encode_clip(frames, req, str(tmp_path))
    assert meta is not None
    fname = Path(meta.file_path).name
    assert "cam_01" in fname
    assert "evt_name" in fname
    assert fname.endswith(".mp4")


# ---------------------------------------------------------------------------
# 22–33: Clip capture service
# ---------------------------------------------------------------------------

def test_service_register_camera(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    assert "cam_01" in svc._buffers


def test_service_unregister_camera(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    svc.unregister_camera("cam_01")
    assert "cam_01" not in svc._buffers


def test_service_feed_frame(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    now = time.time()
    for i in range(10):
        svc.feed_frame("cam_01", _make_frame(), now + i * 0.1, i)
    assert svc._buffers["cam_01"].get_stats()["frame_count"] > 0


def test_service_feed_frame_unregistered(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    # Should not crash
    svc.feed_frame("cam_unknown", _make_frame(), time.time(), 0)


def test_service_request_and_process_clip(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    # Use past timestamps so post_seconds delay is already elapsed
    past = time.time() - 20.0
    for i in range(150):
        svc.feed_frame("cam_01", _make_frame(), past + i * 0.1, i)

    event_ts = past + 5.0
    req = ClipRequest(event_id="evt_proc", camera_id="cam_01",
                      event_timestamp=event_ts, pre_seconds=3.0, post_seconds=0.1)
    svc.request_clip(req)
    svc.process_pending_clips()  # post_seconds already elapsed

    meta = svc.get_clip_metadata("evt_proc")
    assert meta is not None
    assert Path(meta.file_path).exists()


def test_service_clip_delay(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    now = time.time()
    for i in range(50):
        svc.feed_frame("cam_01", _make_frame(), now + i * 0.1, i)

    # Event is in the future — post_seconds=100 means ready at now+102, far from now
    req = ClipRequest(event_id="evt_delay", camera_id="cam_01",
                      event_timestamp=now + 2.0, pre_seconds=1.0, post_seconds=100.0)
    svc.request_clip(req)
    # Process immediately — post_seconds definitely haven't elapsed
    svc.process_pending_clips()
    # Clip should NOT be produced yet
    assert svc.get_clip_metadata("evt_delay") is None


def test_service_clip_too_old(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    now = time.time()
    for i in range(50):
        svc.feed_frame("cam_01", _make_frame(), now + i * 0.1, i)

    # Event was 60 seconds ago — way outside the 20-second buffer
    old_ts = now - 60.0
    req = ClipRequest(event_id="evt_old", camera_id="cam_01",
                      event_timestamp=old_ts, pre_seconds=5.0, post_seconds=0.0)
    svc.request_clip(req)
    svc.process_pending_clips()
    # No clip should be captured
    assert svc.get_clip_metadata("evt_old") is None


def test_service_multiple_cameras(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    svc.register_camera("cam_02")
    past = time.time() - 20.0
    for i in range(150):
        svc.feed_frame("cam_01", _make_frame(), past + i * 0.1, i)
        svc.feed_frame("cam_02", _make_frame(), past + i * 0.1, i)

    for cam, eid in [("cam_01", "evt_c1"), ("cam_02", "evt_c2")]:
        req = ClipRequest(event_id=eid, camera_id=cam,
                          event_timestamp=past + 5.0, pre_seconds=2.0, post_seconds=0.1)
        svc.request_clip(req)
    svc.process_pending_clips()

    assert svc.get_clip_metadata("evt_c1") is not None
    assert svc.get_clip_metadata("evt_c2") is not None


def test_service_get_clip_metadata(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    past = time.time() - 20.0
    for i in range(150):
        svc.feed_frame("cam_01", _make_frame(), past + i * 0.1, i)
    req = ClipRequest(event_id="evt_meta2", camera_id="cam_01",
                      event_timestamp=past + 5.0, pre_seconds=2.0, post_seconds=0.1)
    svc.request_clip(req)
    svc.process_pending_clips()
    meta = svc.get_clip_metadata("evt_meta2")
    assert meta is not None
    assert meta.event_id == "evt_meta2"


def test_service_get_all_clips(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    past = time.time() - 20.0
    for i in range(200):
        svc.feed_frame("cam_01", _make_frame(), past + i * 0.1, i)
    for n in range(3):
        req = ClipRequest(event_id=f"evt_{n}", camera_id="cam_01",
                          event_timestamp=past + 3.0 + n, pre_seconds=1.0, post_seconds=0.1)
        svc.request_clip(req)
    svc.process_pending_clips()
    clips = svc.get_all_clips()
    assert len(clips) == 3


def test_service_get_all_clips_filtered(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    svc.register_camera("cam_02")
    past = time.time() - 20.0
    for i in range(150):
        svc.feed_frame("cam_01", _make_frame(), past + i * 0.1, i)
        svc.feed_frame("cam_02", _make_frame(), past + i * 0.1, i)
    for cam, eid in [("cam_01", "evt_f1"), ("cam_02", "evt_f2")]:
        req = ClipRequest(event_id=eid, camera_id=cam,
                          event_timestamp=past + 5.0, pre_seconds=2.0, post_seconds=0.1)
        svc.request_clip(req)
    svc.process_pending_clips()
    cam1_clips = svc.get_all_clips(camera_id="cam_01")
    assert all(c.camera_id == "cam_01" for c in cam1_clips)
    assert len(cam1_clips) == 1


def test_service_storage_stats(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    past = time.time() - 20.0
    for i in range(150):
        svc.feed_frame("cam_01", _make_frame(), past + i * 0.1, i)
    req = ClipRequest(event_id="evt_stats", camera_id="cam_01",
                      event_timestamp=past + 5.0, pre_seconds=2.0, post_seconds=0.1)
    svc.request_clip(req)
    svc.process_pending_clips()
    stats = svc.get_storage_stats()
    assert stats.total_clips == 1
    assert stats.total_size_bytes > 0
    assert "cam_01" in stats.clips_by_camera


# ---------------------------------------------------------------------------
# 34–36: Retention
# ---------------------------------------------------------------------------

def test_retention_deletes_old_clips(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    past = time.time() - 20.0
    for i in range(150):
        svc.feed_frame("cam_01", _make_frame(), past + i * 0.1, i)
    req = ClipRequest(event_id="evt_old_ret", camera_id="cam_01",
                      event_timestamp=past + 5.0, pre_seconds=2.0, post_seconds=0.1)
    svc.request_clip(req)
    svc.process_pending_clips()

    meta = svc.get_clip_metadata("evt_old_ret")
    assert meta is not None

    # Make it appear old
    svc._metadata["evt_old_ret"] = ClipMetadata(
        clip_id=meta.clip_id, event_id=meta.event_id,
        camera_id=meta.camera_id, file_path=meta.file_path,
        file_size_bytes=meta.file_size_bytes,
        duration_seconds=meta.duration_seconds,
        start_timestamp=meta.start_timestamp,
        end_timestamp=meta.end_timestamp,
        event_timestamp=time.time() - 40 * 86400,  # 40 days old
        resolution=meta.resolution, fps=meta.fps,
        codec=meta.codec, created_at=meta.created_at,
    )
    svc.enforce_retention()
    assert svc.get_clip_metadata("evt_old_ret") is None


def test_retention_keeps_recent_clips(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    past = time.time() - 20.0
    for i in range(150):
        svc.feed_frame("cam_01", _make_frame(), past + i * 0.1, i)
    req = ClipRequest(event_id="evt_recent", camera_id="cam_01",
                      event_timestamp=past + 5.0, pre_seconds=2.0, post_seconds=0.1)
    svc.request_clip(req)
    svc.process_pending_clips()
    svc.enforce_retention()
    assert svc.get_clip_metadata("evt_recent") is not None


def test_retention_file_cleanup(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_01")
    past = time.time() - 20.0
    for i in range(150):
        svc.feed_frame("cam_01", _make_frame(), past + i * 0.1, i)
    req = ClipRequest(event_id="evt_file_del", camera_id="cam_01",
                      event_timestamp=past + 5.0, pre_seconds=2.0, post_seconds=0.1)
    svc.request_clip(req)
    svc.process_pending_clips()

    meta = svc.get_clip_metadata("evt_file_del")
    assert meta is not None
    file_path = Path(meta.file_path)
    assert file_path.exists()

    svc._metadata["evt_file_del"] = ClipMetadata(
        clip_id=meta.clip_id, event_id=meta.event_id,
        camera_id=meta.camera_id, file_path=meta.file_path,
        file_size_bytes=meta.file_size_bytes,
        duration_seconds=meta.duration_seconds,
        start_timestamp=meta.start_timestamp,
        end_timestamp=meta.end_timestamp,
        event_timestamp=time.time() - 40 * 86400,
        resolution=meta.resolution, fps=meta.fps,
        codec=meta.codec, created_at=meta.created_at,
    )
    svc.enforce_retention()
    assert not file_path.exists()


# ---------------------------------------------------------------------------
# 37–40: Integration + full pipeline
# ---------------------------------------------------------------------------

def test_alert_integration_creates_clip_request(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    integration = AlertIntegration(svc, {"storage": {"clip_pre_seconds": 10, "clip_post_seconds": 5}})
    ev = _make_scored_event(AlertDecision.ALERT)
    submitted = integration.handle_scored_events([ev])
    assert len(submitted) == 1
    assert submitted[0].event_id == ev.event_id
    assert submitted[0].priority == 1


def test_alert_integration_no_clip_for_no_alert(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    integration = AlertIntegration(svc, {"storage": {}})
    ev = _make_scored_event(AlertDecision.NO_ALERT)
    submitted = integration.handle_scored_events([ev])
    assert len(submitted) == 0


def test_alert_integration_escalated_priority(tmp_path):
    svc = ClipCaptureService(_make_config(tmp_path))
    integration = AlertIntegration(svc, {"storage": {}})
    ev = _make_scored_event(AlertDecision.ESCALATED)
    submitted = integration.handle_scored_events([ev])
    assert len(submitted) == 1
    assert submitted[0].priority == 2


def test_full_pipeline_to_clip(tmp_path):
    """Frame buffer → feed → alert event → clip captured on disk."""
    svc = ClipCaptureService(_make_config(tmp_path))
    svc.register_camera("cam_test")
    integration = AlertIntegration(svc, {"storage": {"clip_pre_seconds": 2, "clip_post_seconds": 0.1}})

    past = time.time() - 20.0
    for i in range(150):
        svc.feed_frame("cam_test", _make_frame(), past + i * 0.1, i)

    ev = ScoredEvent(
        event_id="evt_pipeline",
        camera_id="cam_test",
        track_id=1,
        timestamp=past + 5.0,
        severity_score=0.80,
        contributing_signals=[],
        dominant_signal=SignalType.ACTION_CLASSIFICATION,
        event_category="violent",
        event_label="fighting",
        alert_decision=AlertDecision.ALERT,
    )
    integration.handle_scored_events([ev])
    svc.process_pending_clips()  # post_seconds already elapsed

    meta = svc.get_clip_metadata("evt_pipeline")
    assert meta is not None
    assert Path(meta.file_path).exists()
    assert meta.file_size_bytes > 0
