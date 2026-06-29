"""Pose estimation data structures and COCO keypoint definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


COCO_KEYPOINT_NAMES: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass
class Keypoint:
    """A single COCO skeleton keypoint.

    Attributes:
        x: Horizontal pixel coordinate in the original image.
        y: Vertical pixel coordinate in the original image.
        confidence: Detection confidence in ``[0, 1]``.
    """

    x: float
    y: float
    confidence: float

    @property
    def is_visible(self) -> bool:
        """Return ``True`` if confidence exceeds the default visibility threshold."""
        return self.confidence >= 0.3

    def as_array(self) -> List[float]:
        """Return ``[x, y, confidence]`` as a plain list."""
        return [self.x, self.y, self.confidence]


@dataclass
class PoseResult:
    """Skeleton pose for a single detected person in one frame.

    Attributes:
        camera_id: Identifier of the source camera.
        frame_id: Sequential frame index.
        timestamp: Unix timestamp of the frame.
        bbox: Person bounding box ``(x1, y1, x2, y2)`` in original image coords.
        bbox_confidence: Person detection confidence.
        keypoints: Exactly 17 COCO keypoints in standard order.
        keypoint_names: COCO keypoint name list (shared constant, not per-instance data).
    """

    camera_id: str
    frame_id: int
    timestamp: float
    bbox: Tuple[int, int, int, int]
    bbox_confidence: float
    keypoints: List[Keypoint] = field(default_factory=list)
    keypoint_names: List[str] = field(default_factory=lambda: COCO_KEYPOINT_NAMES)

    def __post_init__(self) -> None:
        if len(self.keypoints) != 17:
            raise ValueError(
                f"PoseResult requires exactly 17 keypoints, got {len(self.keypoints)}."
            )

    @property
    def visible_keypoint_count(self) -> int:
        """Return number of keypoints with confidence >= 0.3."""
        return sum(1 for kp in self.keypoints if kp.is_visible)

    def keypoints_as_array(self) -> List[List[float]]:
        """Return keypoints as a list of ``[x, y, confidence]`` lists (shape 17×3)."""
        return [kp.as_array() for kp in self.keypoints]
