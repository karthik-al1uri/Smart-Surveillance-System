"""Frame preprocessing utilities for the detection pipeline.

Handles resizing, normalization, and batch preparation of raw frames
before passing them to the YOLOv8 inference engine.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


class FramePreprocessor:
    """Prepares raw BGR frames for YOLOv8 inference.

    Args:
        input_size: Target square size for model input (default 640).
    """

    def __init__(self, input_size: int = 640) -> None:
        self.input_size = input_size

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, int, int]]:
        """Resize and letterbox a single BGR frame to ``(input_size, input_size)``.

        Args:
            frame: Raw BGR image as a NumPy array with shape ``(H, W, 3)``.

        Returns:
            A tuple of:
            - ``resized``: Letterboxed BGR image of shape ``(input_size, input_size, 3)``.
            - ``meta``: Tuple of ``(scale, scale, pad_w, pad_h)`` needed to map
              detections back to original coordinates.

        Raises:
            ValueError: If the input frame is empty or has unexpected shape.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Received empty frame for preprocessing.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected BGR frame with shape (H, W, 3), got {frame.shape}.")

        h_orig, w_orig = frame.shape[:2]
        scale = min(self.input_size / h_orig, self.input_size / w_orig)
        new_w = int(round(w_orig * scale))
        new_h = int(round(h_orig * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (self.input_size - new_w) // 2
        pad_h = (self.input_size - new_h) // 2

        letterboxed = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        letterboxed[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        return letterboxed, (scale, scale, pad_w, pad_h)

    def preprocess_batch(
        self, frames: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float, int, int]]]:
        """Preprocess a batch of frames.

        Args:
            frames: List of raw BGR images.

        Returns:
            A tuple of:
            - ``processed``: List of letterboxed frames.
            - ``metas``: List of ``(scale_x, scale_y, pad_w, pad_h)`` tuples per frame.
        """
        processed: List[np.ndarray] = []
        metas: List[Tuple[float, float, int, int]] = []
        for frame in frames:
            proc, meta = self.preprocess_frame(frame)
            processed.append(proc)
            metas.append(meta)
        return processed, metas

    def unscale_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        meta: Tuple[float, float, int, int],
    ) -> Tuple[int, int, int, int]:
        """Map a detection bounding box from model space back to original image space.

        Args:
            bbox: ``(x1, y1, x2, y2)`` in model input coordinates.
            meta: ``(scale_x, scale_y, pad_w, pad_h)`` from :meth:`preprocess_frame`.

        Returns:
            ``(x1, y1, x2, y2)`` in original image pixel coordinates.
        """
        scale_x, scale_y, pad_w, pad_h = meta
        x1, y1, x2, y2 = bbox
        x1 = int((x1 - pad_w) / scale_x)
        y1 = int((y1 - pad_h) / scale_y)
        x2 = int((x2 - pad_w) / scale_x)
        y2 = int((y2 - pad_h) / scale_y)
        return x1, y1, x2, y2
