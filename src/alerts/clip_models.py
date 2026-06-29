"""
Data structures for clip capture and storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClipRequest:
    """Request to capture a video clip around an event.

    Attributes:
        event_id: Links to :class:`~src.scoring.scoring_models.ScoredEvent`.
        camera_id: Source camera.
        event_timestamp: Unix timestamp of the triggering event.
        pre_seconds: Seconds of video to include *before* the event.
        post_seconds: Seconds of video to include *after* the event.
        priority: Processing priority — higher = processed first.
    """

    event_id: str
    camera_id: str
    event_timestamp: float
    pre_seconds: float = 10.0
    post_seconds: float = 5.0
    priority: int = 1


@dataclass
class ClipMetadata:
    """Metadata about a captured and stored clip.

    Attributes:
        clip_id: Unique identifier (UUID).
        event_id: Links to :class:`~src.scoring.scoring_models.ScoredEvent`.
        camera_id: Source camera.
        file_path: Relative path: ``data/clips/cam_01/20240115_143022_evt123.mp4``.
        file_size_bytes: Size of the MP4 file on disk.
        duration_seconds: Clip duration in seconds.
        start_timestamp: Unix timestamp of the first frame.
        end_timestamp: Unix timestamp of the last frame.
        event_timestamp: When the event occurred (within the clip window).
        resolution: ``(width, height)`` of the clip.
        fps: Clip frame rate.
        codec: FourCC codec string, e.g. ``"mp4v"``.
        created_at: Unix timestamp of when the file was created.
    """

    clip_id: str
    event_id: str
    camera_id: str
    file_path: str
    file_size_bytes: int
    duration_seconds: float
    start_timestamp: float
    end_timestamp: float
    event_timestamp: float
    resolution: tuple          # (width, height)
    fps: float
    codec: str
    created_at: float


@dataclass
class StorageStats:
    """Storage usage statistics.

    Attributes:
        total_clips: Number of clips on disk.
        total_size_bytes: Sum of all clip file sizes.
        total_size_gb: Human-readable GB equivalent.
        oldest_clip_timestamp: Unix timestamp of the earliest clip.
        newest_clip_timestamp: Unix timestamp of the most recent clip.
        clips_by_camera: Mapping of ``camera_id → clip count``.
        storage_path: Root directory for clip storage.
        retention_days: Configured retention policy (days).
    """

    total_clips: int
    total_size_bytes: int
    total_size_gb: float
    oldest_clip_timestamp: Optional[float]
    newest_clip_timestamp: Optional[float]
    clips_by_camera: dict
    storage_path: str
    retention_days: int
