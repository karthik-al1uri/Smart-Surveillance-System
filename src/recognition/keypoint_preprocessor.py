"""Preprocesses raw keypoint sequences for LSTM input.

Handles normalization, missing-keypoint imputation, and tensor conversion.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.common.config import load_config
from src.common.logger import get_logger

logger = get_logger("recognition.preprocessor")

_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6


class KeypointPreprocessor:
    """Normalizes and prepares keypoint sequences for the LSTM classifier.

    Processing pipeline:

    1. **Hip-center normalization** — translate so the midpoint of left/right hip
       is at the origin; scale so torso length (shoulder-to-hip) = 1.0.
    2. **Missing keypoint imputation** — keypoints with confidence < 0.1 are
       linearly interpolated from neighbouring frames; if missing across all
       frames, set to 0.
    3. **Flatten** — reshape from ``(T, 17, 3)`` to ``(T, 51)``.
    4. **Tensor conversion** — float32 :class:`torch.Tensor`.

    Args:
        config: Optional pre-loaded config dict.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or load_config()
        ar = cfg.get("action_recognition", {})
        self._use_velocity = ar.get("use_velocity_features", False)
        self._missing_threshold: float = 0.1

    def preprocess(self, keypoint_sequence: np.ndarray) -> torch.Tensor:
        """Transform raw keypoint sequence to model-ready tensor.

        Args:
            keypoint_sequence: Array of shape ``(T, 17, 3)`` — x, y, confidence.

        Returns:
            Float32 tensor of shape ``(T, 51)``.
        """
        seq = keypoint_sequence.astype(np.float32).copy()
        seq = self._impute_missing(seq)
        seq = self._normalize(seq)
        flat = seq.reshape(seq.shape[0], -1)
        return torch.tensor(flat, dtype=torch.float32)

    def _impute_missing(self, seq: np.ndarray) -> np.ndarray:
        """Replace low-confidence keypoints with linear interpolation.

        Args:
            seq: Array ``(T, 17, 3)``.

        Returns:
            Array with missing keypoints imputed.
        """
        T, K, _ = seq.shape
        for k in range(K):
            conf = seq[:, k, 2]
            missing = conf < self._missing_threshold
            if missing.all():
                seq[:, k, :2] = 0.0
                continue
            if not missing.any():
                continue
            good_frames = np.where(~missing)[0]
            for t in range(T):
                if missing[t]:
                    left = good_frames[good_frames < t]
                    right = good_frames[good_frames > t]
                    if left.size and right.size:
                        t0, t1 = left[-1], right[0]
                        alpha = (t - t0) / (t1 - t0)
                        seq[t, k, :2] = (1 - alpha) * seq[t0, k, :2] + alpha * seq[t1, k, :2]
                    elif left.size:
                        seq[t, k, :2] = seq[left[-1], k, :2]
                    else:
                        seq[t, k, :2] = seq[right[0], k, :2]
        return seq

    def _normalize(self, seq: np.ndarray) -> np.ndarray:
        """Hip-center normalize each frame independently.

        Translates so mid-hip is at origin; scales so torso length = 1.0.

        Args:
            seq: Array ``(T, 17, 3)``.

        Returns:
            Normalized array, same shape.
        """
        for t in range(seq.shape[0]):
            kps = seq[t]
            hip_center = (kps[_LEFT_HIP, :2] + kps[_RIGHT_HIP, :2]) / 2.0

            shoulder_center = (kps[_LEFT_SHOULDER, :2] + kps[_RIGHT_SHOULDER, :2]) / 2.0
            torso_len = float(np.linalg.norm(shoulder_center - hip_center))
            if torso_len < 1e-6:
                torso_len = 1.0

            seq[t, :, :2] = (kps[:, :2] - hip_center) / torso_len

        return seq
