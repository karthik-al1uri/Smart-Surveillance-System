"""YOLOv8 object detector for the Smart Surveillance System.

Loads a YOLOv8 model and runs inference on single frames or batches,
returning structured detection results.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.common.config import get_project_root, load_config
from src.common.logger import get_logger

logger = get_logger("detection.yolo")


@dataclass
class Detection:
    """A single object detection result.

    Attributes:
        class_id: Integer class index from the COCO label set.
        class_name: Human-readable class label (e.g. ``"person"``).
        bbox: Bounding box in original image coordinates as ``(x1, y1, x2, y2)``.
        confidence: Detection confidence score in ``[0, 1]``.
        frame_idx: Optional index of the source frame within a batch.
    """

    class_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    confidence: float
    frame_idx: int = 0


@dataclass
class DetectionResult:
    """Container for all detections from a single frame.

    Attributes:
        detections: List of :class:`Detection` objects.
        frame_idx: Index of this frame within a batch.
        inference_time_ms: Inference time in milliseconds for this frame.
    """

    detections: List[Detection] = field(default_factory=list)
    frame_idx: int = 0
    inference_time_ms: float = 0.0

    @property
    def person_detections(self) -> List[Detection]:
        """Return only detections with class ``"person"``."""
        return [d for d in self.detections if d.class_name == "person"]

    @property
    def count(self) -> int:
        """Return total number of detections."""
        return len(self.detections)


class YOLODetector:
    """YOLOv8 object detector wrapper.

    Loads the model from the path specified in config and exposes methods for
    single-frame and batched inference.

    Args:
        config: Optional pre-loaded configuration dictionary. If ``None``,
            the default config is loaded automatically.

    Raises:
        FileNotFoundError: If the model weights file does not exist.
        SystemExit: If ``ultralytics`` is not installed.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self._cfg = config or load_config()
        det_cfg = self._cfg["detection"]

        self._model_path = get_project_root() / det_cfg["model_path"]
        self._input_size: int = det_cfg["input_size"]
        self._conf_threshold: float = det_cfg["confidence_threshold"]
        self._iou_threshold: float = det_cfg["iou_threshold"]
        self._batch_size: int = det_cfg["batch_size"]
        self._device: str = det_cfg.get("device", "cpu")
        self._target_classes: List[str] = det_cfg.get("target_classes", [])

        self._validate_weights()
        self._model = self._load_model()
        self._class_names: Dict[int, str] = self._model.names

    def _validate_weights(self) -> None:
        """Ensure model weights exist before attempting to load."""
        if not self._model_path.exists():
            logger.error(
                "Model weights not found at '%s'. "
                "Place yolov8m.pt in the models/ directory. "
                "Download with: python -c \"from ultralytics import YOLO; YOLO('yolov8m.pt')\"",
                self._model_path,
            )
            sys.exit(1)

    def _load_model(self):
        """Load the YOLOv8 model from disk."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics is not installed. Run: pip install ultralytics")
            sys.exit(1)

        logger.info("Loading YOLOv8 model from '%s' on device '%s'", self._model_path, self._device)
        model = YOLO(str(self._model_path))
        logger.info("Model loaded. Classes available: %d", len(model.names))
        return model

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> DetectionResult:
        """Run detection on a single BGR frame.

        Args:
            frame: Raw BGR image as a NumPy array.
            frame_idx: Optional frame index for tracking.

        Returns:
            :class:`DetectionResult` with detections in original image coordinates.
        """
        results = self._model.predict(
            source=frame,
            imgsz=self._input_size,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            device=self._device,
            verbose=False,
        )
        return self._parse_results(results, [frame_idx])[0]

    def detect_batch(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        """Run detection on a batch of BGR frames.

        Frames are processed in chunks of ``batch_size`` for GPU efficiency.

        Args:
            frames: List of raw BGR images.

        Returns:
            List of :class:`DetectionResult`, one per input frame (same order).
        """
        all_results: List[DetectionResult] = []
        for chunk_start in range(0, len(frames), self._batch_size):
            chunk = frames[chunk_start : chunk_start + self._batch_size]
            indices = list(range(chunk_start, chunk_start + len(chunk)))
            raw = self._model.predict(
                source=chunk,
                imgsz=self._input_size,
                conf=self._conf_threshold,
                iou=self._iou_threshold,
                device=self._device,
                verbose=False,
            )
            all_results.extend(self._parse_results(raw, indices))
        return all_results

    def _parse_results(self, raw_results, frame_indices: List[int]) -> List[DetectionResult]:
        """Convert ultralytics result objects to :class:`DetectionResult` instances.

        Args:
            raw_results: List of ultralytics ``Results`` objects.
            frame_indices: Frame indices corresponding to each result.

        Returns:
            List of :class:`DetectionResult`.
        """
        output: List[DetectionResult] = []
        for raw, idx in zip(raw_results, frame_indices):
            detections: List[Detection] = []
            speed = raw.speed or {}
            infer_ms = speed.get("inference", 0.0)

            if raw.boxes is not None:
                for box in raw.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = self._class_names.get(cls_id, str(cls_id))

                    if self._target_classes and cls_name not in self._target_classes:
                        continue

                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = (int(v) for v in xyxy)
                    conf = float(box.conf[0].item())

                    detections.append(
                        Detection(
                            class_id=cls_id,
                            class_name=cls_name,
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            frame_idx=idx,
                        )
                    )

            output.append(
                DetectionResult(
                    detections=detections,
                    frame_idx=idx,
                    inference_time_ms=infer_ms,
                )
            )
        return output
