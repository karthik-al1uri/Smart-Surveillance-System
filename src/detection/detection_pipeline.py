"""End-to-end detection pipeline for the Smart Surveillance System.

Wires together the frame preprocessor and YOLOv8 detector into a single
callable that accepts raw BGR frames and returns structured results.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from src.common.config import load_config
from src.common.logger import get_logger
from src.common.model_manager import ModelManager
from src.detection.preprocessor import FramePreprocessor
from src.detection.yolo_detector import DetectionResult, YOLODetector

logger = get_logger("detection.pipeline")


class DetectionPipeline:
    """Orchestrates preprocessing and YOLOv8 inference for one or more frames.

    Args:
        config: Optional pre-loaded configuration dictionary. If ``None``,
            the default config is loaded automatically.
    """

    def __init__(self, config: Optional[Dict] = None, model_manager: Optional[ModelManager] = None) -> None:
        self._cfg = config or load_config()
        input_size: int = self._cfg["detection"]["input_size"]
        self._preprocessor = FramePreprocessor(input_size=input_size)
        self._detector = YOLODetector(config=self._cfg, model_manager=model_manager)
        self._model_manager = model_manager
        logger.info("DetectionPipeline initialised (input_size=%d).", input_size)

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> DetectionResult:
        """Run the full detection pipeline on a single BGR frame.

        Args:
            frame: Raw BGR image as a NumPy array.
            frame_idx: Optional frame index used for tracking.

        Returns:
            :class:`~src.detection.yolo_detector.DetectionResult` with detections
            in original image coordinates.
        """
        t0 = time.perf_counter()
        result = self._detector.detect(frame, frame_idx=frame_idx)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "frame_idx=%d | detections=%d | latency=%.1f ms",
            frame_idx,
            result.count,
            elapsed_ms,
        )
        return result

    def process_batch(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        """Run the full detection pipeline on a batch of BGR frames.

        Args:
            frames: List of raw BGR images (in order).

        Returns:
            List of :class:`~src.detection.yolo_detector.DetectionResult` objects,
            one per input frame, preserving input order.
        """
        if not frames:
            return []
        t0 = time.perf_counter()
        results = self._detector.detect_batch(frames)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_dets = sum(r.count for r in results)
        logger.info(
            "Batch processed: frames=%d | total_detections=%d | latency=%.1f ms",
            len(frames),
            total_dets,
            elapsed_ms,
        )
        return results
