"""Sliding window manager for temporal action recognition.

Manages per-track windows and determines when to trigger classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.common.config import load_config
from src.common.logger import get_logger

logger = get_logger("recognition.sliding_window")


@dataclass
class WindowData:
    """A ready-to-classify temporal window of keypoints.

    Attributes:
        track_id: ID of the tracked person.
        keypoint_sequence: Shape ``(window_size, 17, 3)`` float32 array.
        start_frame: Index of the first frame in the window.
        end_frame: Index of the last frame in the window.
        avg_keypoint_confidence: Average keypoint confidence — quality metric.
    """

    track_id: int
    keypoint_sequence: np.ndarray
    start_frame: int
    end_frame: int
    avg_keypoint_confidence: float


class _TrackBuffer:
    """Internal rolling buffer for a single track."""

    def __init__(self, window_size: int) -> None:
        self.keypoints: List[np.ndarray] = []
        self.frame_ids: List[int] = []
        self.frames_since_last_emit: int = 0
        self.window_size = window_size

    def push(self, keypoints: np.ndarray, frame_id: int) -> None:
        self.keypoints.append(keypoints.copy())
        self.frame_ids.append(frame_id)
        self.frames_since_last_emit += 1

    def ready(self, stride: int) -> bool:
        return (
            len(self.keypoints) >= self.window_size
            and self.frames_since_last_emit >= stride
        )

    def get_window(self) -> tuple:
        seq = np.array(self.keypoints[-self.window_size:], dtype=np.float32)
        start = self.frame_ids[-self.window_size]
        end = self.frame_ids[-1]
        return seq, start, end

    def mark_emitted(self) -> None:
        self.frames_since_last_emit = 0


class SlidingWindowManager:
    """Manages per-track sliding windows for action classification.

    Config keys (under ``action_recognition``):
        - ``window_size`` (int, default 16): frames per window.
        - ``stride`` (int, default 8): frames between consecutive emissions.
        - ``min_keypoint_confidence`` (float, default 0.3): minimum average
          keypoint confidence to emit a window.
        - ``min_visible_keypoints`` (int, default 10): minimum keypoints with
          confidence > threshold averaged across the window frames.

    Args:
        config: Optional pre-loaded config dict.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or load_config()
        ar = cfg.get("action_recognition", {})
        self._window_size: int = ar.get("window_size", 16)
        self._stride: int = ar.get("stride", 8)
        self._min_kp_conf: float = ar.get("min_keypoint_confidence", 0.3)
        self._min_visible: int = ar.get("min_visible_keypoints", 10)

        self._buffers: Dict[int, _TrackBuffer] = {}
        self._stats = {"windows_created": 0, "windows_skipped_quality": 0, "windows_pending": 0}

        logger.info(
            "SlidingWindowManager ready: window=%d stride=%d min_conf=%.2f min_visible=%d",
            self._window_size, self._stride, self._min_kp_conf, self._min_visible,
        )

    def update(self, track_id: int, keypoints: np.ndarray, frame_id: int) -> None:
        """Feed one frame's keypoints (17×3) for a track.

        Args:
            track_id: Track identifier.
            keypoints: Array of shape ``(17, 3)`` — x, y, confidence.
            frame_id: Sequential frame index.
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = _TrackBuffer(self._window_size)
        self._buffers[track_id].push(keypoints, frame_id)

    def get_ready_windows(self) -> List[WindowData]:
        """Return all windows ready for classification and reset their stride counters.

        A window is ready when a track has ``>= window_size`` frames AND
        ``>= stride`` NEW frames since the last emission.  The window covers
        the LAST ``window_size`` frames (overlapping with the previous window).

        Returns:
            List of :class:`WindowData` objects that passed the quality filter.
        """
        ready: List[WindowData] = []
        for track_id, buf in self._buffers.items():
            if not buf.ready(self._stride):
                continue
            seq, start, end = buf.get_window()
            avg_conf = float(seq[:, :, 2].mean())
            avg_visible = float((seq[:, :, 2] >= self._min_kp_conf).sum(axis=1).mean())

            if avg_conf < self._min_kp_conf or avg_visible < self._min_visible:
                self._stats["windows_skipped_quality"] += 1
                buf.mark_emitted()
                logger.debug(
                    "Track %d window skipped: avg_conf=%.2f avg_visible=%.1f",
                    track_id, avg_conf, avg_visible,
                )
                continue

            buf.mark_emitted()
            self._stats["windows_created"] += 1
            ready.append(
                WindowData(
                    track_id=track_id,
                    keypoint_sequence=seq,
                    start_frame=start,
                    end_frame=end,
                    avg_keypoint_confidence=avg_conf,
                )
            )
            logger.debug("Track %d window ready: frames %d-%d conf=%.2f", track_id, start, end, avg_conf)

        self._stats["windows_pending"] = sum(
            1 for b in self._buffers.values() if len(b.keypoints) >= self._window_size
        )
        return ready

    def remove_track(self, track_id: int) -> None:
        """Remove the buffer for a track that has been deleted.

        Args:
            track_id: Track to remove.
        """
        if track_id in self._buffers:
            del self._buffers[track_id]
            logger.debug("SlidingWindowManager: track %d buffer removed.", track_id)

    def get_stats(self) -> dict:
        """Return window statistics.

        Returns:
            Dict with ``windows_created``, ``windows_skipped_quality``,
            ``windows_pending``, ``active_tracks``.
        """
        return {
            **self._stats,
            "active_tracks": len(self._buffers),
        }
