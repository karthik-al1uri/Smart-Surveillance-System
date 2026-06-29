"""
Per-camera rolling frame buffer for clip extraction.

Keeps the last N seconds of raw frames in memory so clips can be captured
retroactively when an alert fires.

Memory budget (uncompressed):
    720p frame (1280×720 BGR) ≈ 2.8 MB
    200 frames × 2.8 MB ≈ 560 MB per camera  ← too high for multi-camera

With JPEG compression at quality=85:
    ~40–60 KB per frame
    200 frames × 50 KB ≈ 10 MB per camera  ← acceptable

Enable via ``clip_buffer_compressed: true`` in config (recommended).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.common.logger import get_logger

logger = get_logger("alerts.rolling_buffer")


class RollingFrameBuffer:
    """Thread-safe rolling buffer for one camera stream.

    Frames are stored in arrival order. When the buffer reaches capacity
    (``buffer_duration × target_fps`` frames) the oldest frame is dropped.
    Optionally stores JPEG-compressed bytes instead of raw numpy arrays to
    dramatically reduce memory usage.

    Args:
        camera_id: Identifier of the camera this buffer belongs to.
        buffer_duration: Seconds of video to retain (default 20 s).
        target_fps: Target storage frame rate; incoming frames are downsampled
            to this rate (default 10 fps).
        compressed: If ``True``, each frame is JPEG-encoded before storage and
            decoded on demand.
        jpeg_quality: JPEG quality (1–100) when ``compressed=True``.
    """

    def __init__(
        self,
        camera_id: str,
        buffer_duration: float = 20.0,
        target_fps: float = 10.0,
        compressed: bool = True,
        jpeg_quality: int = 85,
    ) -> None:
        self.camera_id = camera_id
        self._buffer_duration = buffer_duration
        self._target_fps = target_fps
        self._compressed = compressed
        self._jpeg_quality = jpeg_quality

        self._max_frames = int(buffer_duration * target_fps)
        self._min_interval = 1.0 / target_fps if target_fps > 0 else 0.0

        # Each entry: {"frame": np.ndarray | bytes, "timestamp": float, "frame_id": int}
        self._buffer: Deque[Dict] = deque(maxlen=self._max_frames)
        self._last_stored_ts: float = 0.0
        self._lock = threading.Lock()

        logger.debug(
            "RollingFrameBuffer[%s] ready: %.0fs × %.0ffps = %d max frames, compressed=%s",
            camera_id, buffer_duration, target_fps, self._max_frames, compressed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_frame(self, frame: np.ndarray, timestamp: float, frame_id: int) -> None:
        """Add a frame to the buffer.

        Frames arriving faster than ``target_fps`` are skipped; the buffer
        stores only approximately ``target_fps`` frames per second.

        Args:
            frame: BGR numpy array.
            timestamp: Unix timestamp of the frame.
            frame_id: Sequential frame index from the pipeline.
        """
        if timestamp - self._last_stored_ts < self._min_interval:
            return

        if self._compressed:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                logger.warning("JPEG encode failed for frame %d (cam=%s)", frame_id, self.camera_id)
                return
            stored: object = buf.tobytes()
        else:
            stored = frame.copy()

        entry = {"frame": stored, "timestamp": timestamp, "frame_id": frame_id}
        with self._lock:
            self._buffer.append(entry)
        self._last_stored_ts = timestamp

    def extract_clip(
        self,
        event_timestamp: float,
        pre_seconds: float = 10.0,
        post_seconds: float = 5.0,
    ) -> Optional[List[Dict]]:
        """Extract frames from the buffer around an event timestamp.

        Returns the subset of buffered frames in the window
        ``[event_timestamp - pre_seconds, event_timestamp + post_seconds]``.
        Returns ``None`` if the event is older than the buffer.

        Args:
            event_timestamp: Unix timestamp of the event.
            pre_seconds: Seconds before the event to include.
            post_seconds: Seconds after the event to include.

        Returns:
            List of dicts ``{"frame": np.ndarray, "timestamp": float, "frame_id": int}``,
            or ``None`` if the event is not covered by the buffer.
        """
        start_ts = event_timestamp - pre_seconds
        end_ts = event_timestamp + post_seconds

        with self._lock:
            if not self._buffer:
                return None
            oldest = self._buffer[0]["timestamp"]
            # If event pre-window is older than buffer start → we can't provide full clip
            # We still return partial clip (only what we have)
            # But if event itself is older than entire buffer → return None
            if event_timestamp < oldest:
                return None

            frames = [
                e for e in self._buffer
                if start_ts <= e["timestamp"] <= end_ts
            ]

        if not frames:
            return None

        # Decompress if needed
        result = []
        for e in frames:
            raw = e["frame"]
            if self._compressed and isinstance(raw, bytes):
                arr = np.frombuffer(raw, np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is None:
                    continue
                result.append({"frame": decoded, "timestamp": e["timestamp"], "frame_id": e["frame_id"]})
            else:
                result.append({"frame": raw, "timestamp": e["timestamp"], "frame_id": e["frame_id"]})

        return result if result else None

    def get_buffer_time_range(self) -> Tuple[float, float]:
        """Return ``(oldest_timestamp, newest_timestamp)`` of buffered frames.

        Returns ``(0.0, 0.0)`` if the buffer is empty.
        """
        with self._lock:
            if not self._buffer:
                return (0.0, 0.0)
            return (self._buffer[0]["timestamp"], self._buffer[-1]["timestamp"])

    def get_stats(self) -> dict:
        """Return buffer statistics.

        Returns:
            Dict with ``frame_count``, ``max_frames``, ``time_range_seconds``,
            ``estimated_memory_bytes``, and ``compressed``.
        """
        with self._lock:
            count = len(self._buffer)
            if count == 0:
                time_range = 0.0
                mem_estimate = 0
            else:
                time_range = self._buffer[-1]["timestamp"] - self._buffer[0]["timestamp"]
                sample = self._buffer[0]["frame"]
                if isinstance(sample, bytes):
                    avg_size = len(sample)
                else:
                    avg_size = sample.nbytes
                mem_estimate = count * avg_size

        return {
            "camera_id": self.camera_id,
            "frame_count": count,
            "max_frames": self._max_frames,
            "time_range_seconds": round(time_range, 2),
            "estimated_memory_bytes": mem_estimate,
            "compressed": self._compressed,
        }

    def clear(self) -> None:
        """Flush all buffered frames."""
        with self._lock:
            self._buffer.clear()
        self._last_stored_ts = 0.0
