"""
Clip Capture Service.

Manages rolling frame buffers per camera, handles clip requests,
and coordinates encoding and storage.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.alerts.clip_encoder import ClipEncoder
from src.alerts.clip_models import ClipMetadata, ClipRequest, StorageStats
from src.alerts.rolling_buffer import RollingFrameBuffer
from src.common.logger import get_logger

logger = get_logger("alerts.clip_capture")


class ClipCaptureService:
    """Manages rolling frame buffers, clip requests, encoding, and retention.

    Typical usage:

    1. Call :meth:`register_camera` for each active camera at startup.
    2. Call :meth:`feed_frame` for every frame from the ingestion pipeline.
    3. Call :meth:`request_clip` when a :class:`~src.scoring.scoring_models.ScoredEvent`
       triggers an alert.
    4. Call :meth:`start` to launch the background processing thread.
    5. Query :meth:`get_clip_metadata` or :meth:`get_all_clips` as needed.

    Args:
        config: Full project config dict; reads the ``storage`` section.
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("storage", {})
        self._buffer_duration = float(cfg.get("clip_buffer_duration", 20.0))
        self._buffer_fps = float(cfg.get("clip_buffer_fps", 10.0))
        self._compressed = bool(cfg.get("clip_buffer_compressed", True))
        self._retention_days = int(cfg.get("retention_days", 30))
        self._retention_check_interval = float(cfg.get("retention_check_interval", 3600.0))
        self._clip_dir = Path(cfg.get("clip_dir", "data/clips"))

        self._encoder = ClipEncoder(config)

        self._buffers: Dict[str, RollingFrameBuffer] = {}
        self._pending: List[ClipRequest] = []
        self._metadata: Dict[str, ClipMetadata] = {}  # event_id → ClipMetadata

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_retention_check: float = 0.0

        logger.info(
            "ClipCaptureService ready: buffer=%.0fs@%.0ffps compressed=%s retention=%dd",
            self._buffer_duration, self._buffer_fps, self._compressed, self._retention_days,
        )

    # ------------------------------------------------------------------
    # Camera registration
    # ------------------------------------------------------------------

    def register_camera(self, camera_id: str) -> None:
        """Create a rolling frame buffer for a camera.

        Args:
            camera_id: Camera identifier.
        """
        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = RollingFrameBuffer(
                    camera_id=camera_id,
                    buffer_duration=self._buffer_duration,
                    target_fps=self._buffer_fps,
                    compressed=self._compressed,
                )
                logger.info("Camera '%s' registered with rolling buffer.", camera_id)

    def unregister_camera(self, camera_id: str) -> None:
        """Remove and clear a camera's rolling buffer.

        Args:
            camera_id: Camera identifier.
        """
        with self._lock:
            buf = self._buffers.pop(camera_id, None)
        if buf:
            buf.clear()
            logger.info("Camera '%s' unregistered.", camera_id)

    # ------------------------------------------------------------------
    # Frame ingestion
    # ------------------------------------------------------------------

    def feed_frame(self, camera_id: str, frame: np.ndarray, timestamp: float, frame_id: int) -> None:
        """Add a raw frame to the camera's rolling buffer.

        If the camera is not registered, a warning is logged and the frame
        is silently dropped.

        Args:
            camera_id: Camera identifier.
            frame: BGR numpy array.
            timestamp: Unix timestamp of the frame.
            frame_id: Sequential frame index from the ingestion pipeline.
        """
        buf = self._buffers.get(camera_id)
        if buf is None:
            logger.warning("feed_frame: camera '%s' not registered.", camera_id)
            return
        buf.add_frame(frame, timestamp, frame_id)

    # ------------------------------------------------------------------
    # Clip requests
    # ------------------------------------------------------------------

    def request_clip(self, clip_request: ClipRequest) -> None:
        """Queue a clip capture request.

        The clip will not be extracted immediately — the service waits for
        ``post_seconds`` to elapse after ``event_timestamp`` before encoding,
        so that post-event frames are available in the buffer.

        Args:
            clip_request: Populated :class:`~src.alerts.clip_models.ClipRequest`.
        """
        with self._lock:
            self._pending.append(clip_request)
        logger.debug(
            "Clip request queued: event=%s cam=%s ts=%.2f",
            clip_request.event_id, clip_request.camera_id, clip_request.event_timestamp,
        )

    def process_pending_clips(self) -> None:
        """Process queued clip requests that are ready.

        A request is ready when:
        ``now >= event_timestamp + post_seconds``

        Clips whose events are too old (frames no longer in buffer) are
        logged as errors and removed.
        """
        now = time.time()
        with self._lock:
            ready = [r for r in self._pending if now >= r.event_timestamp + r.post_seconds]
            for r in ready:
                self._pending.remove(r)

        for req in sorted(ready, key=lambda r: -r.priority):
            buf = self._buffers.get(req.camera_id)
            if buf is None:
                logger.error("Clip capture: camera '%s' not registered (event=%s).", req.camera_id, req.event_id)
                continue

            frames = buf.extract_clip(req.event_timestamp, req.pre_seconds, req.post_seconds)
            if frames is None:
                logger.warning(
                    "Clip capture failed — event too old for buffer (event=%s ts=%.2f).",
                    req.event_id, req.event_timestamp,
                )
                continue

            meta = self._encoder.encode_clip(frames, req)
            if meta:
                with self._lock:
                    self._metadata[req.event_id] = meta
                logger.info(
                    "Clip ready: %s (%.1fs, %dKB)", Path(meta.file_path).name,
                    meta.duration_seconds, meta.file_size_bytes // 1024,
                )

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background clip processing thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._background_loop, daemon=True, name="clip-capture")
        self._thread.start()
        logger.info("ClipCaptureService background thread started.")

    def stop(self) -> None:
        """Stop the background thread and flush remaining pending clips."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10.0)
        # Final flush
        self.process_pending_clips()
        logger.info("ClipCaptureService stopped.")

    def _background_loop(self) -> None:
        while self._running:
            self.process_pending_clips()
            now = time.time()
            if now - self._last_retention_check >= self._retention_check_interval:
                self.enforce_retention()
                self._last_retention_check = now
            time.sleep(1.0)

    # ------------------------------------------------------------------
    # Metadata / listing
    # ------------------------------------------------------------------

    def get_clip_metadata(self, event_id: str) -> Optional[ClipMetadata]:
        """Return metadata for a captured clip by event ID.

        Args:
            event_id: The ``event_id`` from :class:`~src.scoring.scoring_models.ScoredEvent`.
        """
        return self._metadata.get(event_id)

    def get_all_clips(self, camera_id: Optional[str] = None) -> List[ClipMetadata]:
        """Return all clip metadata, optionally filtered by camera.

        Args:
            camera_id: If provided, return only clips from this camera.
        """
        with self._lock:
            clips = list(self._metadata.values())
        if camera_id:
            clips = [c for c in clips if c.camera_id == camera_id]
        return clips

    def get_storage_stats(self) -> StorageStats:
        """Return current storage usage statistics.

        Returns:
            :class:`~src.alerts.clip_models.StorageStats` with counts, sizes,
            oldest/newest timestamps, and per-camera breakdown.
        """
        with self._lock:
            clips = list(self._metadata.values())

        total_size = sum(c.file_size_bytes for c in clips)
        by_cam: dict = {}
        for c in clips:
            by_cam[c.camera_id] = by_cam.get(c.camera_id, 0) + 1

        timestamps = [c.event_timestamp for c in clips]
        return StorageStats(
            total_clips=len(clips),
            total_size_bytes=total_size,
            total_size_gb=round(total_size / (1024 ** 3), 4),
            oldest_clip_timestamp=min(timestamps) if timestamps else None,
            newest_clip_timestamp=max(timestamps) if timestamps else None,
            clips_by_camera=by_cam,
            storage_path=str(self._clip_dir),
            retention_days=self._retention_days,
        )

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def enforce_retention(self) -> None:
        """Delete clips older than the configured retention period.

        Iterates all tracked clips and removes files from disk whose
        ``event_timestamp`` is older than ``retention_days`` ago.
        """
        cutoff = time.time() - self._retention_days * 86400
        with self._lock:
            to_delete = [eid for eid, m in self._metadata.items() if m.event_timestamp < cutoff]

        deleted = 0
        for eid in to_delete:
            with self._lock:
                meta = self._metadata.pop(eid, None)
            if meta is None:
                continue
            p = Path(meta.file_path)
            if p.exists():
                try:
                    p.unlink()
                    deleted += 1
                    logger.info("Retention: deleted %s", p.name)
                except OSError as exc:
                    logger.error("Retention: failed to delete %s: %s", p, exc)

        if deleted:
            logger.info("Retention enforcement: deleted %d clip(s).", deleted)
