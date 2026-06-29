"""
Encodes raw frames into MP4 video clips using OpenCV VideoWriter.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.alerts.clip_models import ClipMetadata, ClipRequest
from src.common.logger import get_logger

logger = get_logger("alerts.clip_encoder")


class ClipEncoder:
    """Encodes a list of frames into a timestamped MP4 file.

    Args:
        config: Full project config dict; reads the ``storage`` section.

    Configuration keys (all under ``storage``):
        ``clip_dir``: Root directory for all clips.
        ``clip_codec``: OpenCV FourCC codec string (default ``"mp4v"``).
        ``clip_fps``: Output frame rate (default ``10``).
        ``max_clip_duration``: Hard cap in seconds (default ``20``).
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("storage", {})
        self._clip_dir = Path(cfg.get("clip_dir", "data/clips"))
        self._codec = cfg.get("clip_codec", "mp4v")
        self._fps = float(cfg.get("clip_fps", 10.0))
        self._max_duration = float(cfg.get("max_clip_duration", 20.0))
        logger.debug(
            "ClipEncoder ready: dir=%s codec=%s fps=%.0f max_dur=%.0fs",
            self._clip_dir, self._codec, self._fps, self._max_duration,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_clip(
        self,
        frames: List[dict],
        clip_request: ClipRequest,
        output_dir: Optional[str] = None,
    ) -> Optional[ClipMetadata]:
        """Encode a list of frame dicts to an MP4 file.

        Args:
            frames: List of ``{"frame": np.ndarray, "timestamp": float, "frame_id": int}``.
            clip_request: The originating clip request (provides event_id, camera_id, timestamp).
            output_dir: Override for the output directory root.  Falls back to config ``clip_dir``.

        Returns:
            :class:`~src.alerts.clip_models.ClipMetadata` on success, or ``None`` on error.
        """
        if not frames:
            logger.warning("encode_clip called with empty frame list (event=%s).", clip_request.event_id)
            return None

        root = Path(output_dir) if output_dir else self._clip_dir
        camera_dir = root / clip_request.camera_id
        try:
            camera_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Cannot create clip directory %s: %s", camera_dir, exc)
            return None

        # Determine resolution from first valid frame
        first_frame = frames[0]["frame"]
        if not isinstance(first_frame, np.ndarray):
            logger.error("Frame is not a numpy array (event=%s).", clip_request.event_id)
            return None
        h, w = first_frame.shape[:2]

        # Enforce max_duration cap
        max_frames = int(self._max_duration * self._fps)
        frames = frames[:max_frames]

        # Build output path
        dt_str = datetime.fromtimestamp(clip_request.event_timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"{clip_request.camera_id}_{dt_str}_{clip_request.event_id}.mp4"
        out_path = camera_dir / filename

        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, self._fps, (w, h))
        if not writer.isOpened():
            logger.error("VideoWriter failed to open for %s.", out_path)
            return None

        try:
            for entry in frames:
                f = entry["frame"]
                if not isinstance(f, np.ndarray):
                    continue
                if f.shape[0] != h or f.shape[1] != w:
                    f = cv2.resize(f, (w, h))
                writer.write(f)
        except (OSError, cv2.error) as exc:
            logger.error("Error writing clip %s: %s", out_path, exc)
            writer.release()
            return None
        finally:
            writer.release()

        if not out_path.exists():
            logger.error("Clip file not found after write: %s", out_path)
            return None

        file_size = out_path.stat().st_size
        timestamps = [e["timestamp"] for e in frames if "timestamp" in e]
        start_ts = timestamps[0] if timestamps else clip_request.event_timestamp
        end_ts = timestamps[-1] if timestamps else clip_request.event_timestamp
        duration = end_ts - start_ts if end_ts > start_ts else len(frames) / self._fps

        meta = ClipMetadata(
            clip_id=str(uuid.uuid4()),
            event_id=clip_request.event_id,
            camera_id=clip_request.camera_id,
            file_path=str(out_path),
            file_size_bytes=file_size,
            duration_seconds=round(duration, 2),
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            event_timestamp=clip_request.event_timestamp,
            resolution=(w, h),
            fps=self._fps,
            codec=self._codec,
            created_at=time.time(),
        )
        logger.info(
            "Clip encoded: %s (%.1fs, %dKB)",
            out_path.name, duration, file_size // 1024,
        )
        return meta
