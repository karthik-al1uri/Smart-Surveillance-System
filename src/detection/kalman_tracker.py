"""Kalman filter for bounding box state prediction.

Predicts the next position of a tracked person between detections using a
constant-velocity model with a 7-dimensional state vector.
"""

from __future__ import annotations

import numpy as np


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` bbox to measurement vector ``[cx, cy, s, r]``.

    Args:
        bbox: Array of shape ``(4,)`` with ``[x1, y1, x2, y2]``.

    Returns:
        Array of shape ``(4, 1)`` with ``[cx, cy, area, aspect_ratio]``.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([[cx], [cy], [s], [r]], dtype=np.float64)


def x_to_bbox(state: np.ndarray) -> np.ndarray:
    """Convert state vector ``[cx, cy, s, r, ...]`` back to ``[x1, y1, x2, y2]``.

    Args:
        state: Array of any shape with at least 4 elements (first 4 used).
            filterpy returns a ``(7, 1)`` column vector; we flatten it first.

    Returns:
        Array of shape ``(4,)`` with ``[x1, y1, x2, y2]``.
    """
    flat = np.asarray(state).flatten()
    cx, cy, s, r = float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
    if s <= 0:
        s = 1.0
    w = np.sqrt(s * abs(r))
    h = s / w if w > 0 else 1.0
    return np.array(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        dtype=np.float64,
    )


class KalmanBoxTracker:
    """Single-target Kalman filter tracker for a bounding box.

    State: ``[cx, cy, s, r, vcx, vcy, vs]`` (centre-x, centre-y, area,
    aspect-ratio, and their velocities; aspect-ratio velocity fixed at 0).

    Args:
        bbox: Initial detection bounding box as ``[x1, y1, x2, y2]``.
    """

    _count: int = 0

    def __init__(self, bbox: np.ndarray) -> None:
        from filterpy.kalman import KalmanFilter

        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = bbox_to_z(np.asarray(bbox, dtype=np.float64))

        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count

    def predict(self) -> np.ndarray:
        """Advance state by one time step and return the predicted bbox.

        Returns:
            Predicted bbox as ``[x1, y1, x2, y2]``.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        return x_to_bbox(self.kf.x)

    def update(self, bbox: np.ndarray) -> None:
        """Update the filter state with a new measurement.

        Args:
            bbox: Measured bbox as ``[x1, y1, x2, y2]``.
        """
        self.kf.update(bbox_to_z(np.asarray(bbox, dtype=np.float64)))

    def get_state(self) -> np.ndarray:
        """Return the current estimated bbox ``[x1, y1, x2, y2]``."""
        return x_to_bbox(self.kf.x)
