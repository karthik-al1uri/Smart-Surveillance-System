"""YOLOv8-Pose wrapper for skeleton keypoint extraction.

Runs detection + pose estimation in a single forward pass.
When ``single_pass_mode`` is enabled (default), also outputs
:class:`~src.detection.yolo_detector.Detection` objects for persons
so the standalone YOLOv8 detector is not needed for person detection.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional

import numpy as np

from src.common.config import get_project_root, load_config
from src.common.logger import get_logger
from src.detection.pose_structures import COCO_KEYPOINT_NAMES, Keypoint, PoseResult
from src.detection.yolo_detector import Detection

logger = get_logger("detection.pose")


class PoseEstimator:
    """YOLOv8-Pose model wrapper for single-frame and batched estimation.

    Args:
        config: Optional pre-loaded configuration dict.  Loaded automatically
            if ``None``.

    Raises:
        FileNotFoundError: If the model weights file does not exist.
        SystemExit: If ``ultralytics`` is not installed.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self._cfg = config or load_config()
        pose_cfg = self._cfg["pose"]

        self._model_path = get_project_root() / pose_cfg["model_path"]
        self._input_size: int = pose_cfg["input_size"]
        self._conf_threshold: float = pose_cfg["confidence_threshold"]
        self._iou_threshold: float = pose_cfg["iou_threshold"]
        self._device: str = pose_cfg.get("device", "cpu")
        self._kp_conf_threshold: float = pose_cfg.get("keypoint_confidence_threshold", 0.3)
        self._single_pass_mode: bool = pose_cfg.get("single_pass_mode", True)

        self._validate_weights()
        self._model = self._load_model()

    def _validate_weights(self) -> None:
        if not self._model_path.exists():
            logger.error(
                "Pose model weights not found at '%s'. "
                "Place yolov8m-pose.pt in the models/ directory.",
                self._model_path,
            )
            sys.exit(1)

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics is not installed. Run: pip install ultralytics")
            sys.exit(1)

        logger.info(
            "Loading YOLOv8-Pose model from '%s' on device '%s'",
            self._model_path,
            self._device,
        )
        model = YOLO(str(self._model_path))
        logger.info("Pose model loaded. Single-pass mode: %s", self._single_pass_mode)
        return model

    def get_model_info(self) -> Dict:
        """Return metadata about the loaded pose model.

        Returns:
            Dict with keys ``model_path``, ``device``, ``conf_threshold``,
            ``single_pass_mode``.
        """
        return {
            "model_path": str(self._model_path),
            "device": self._device,
            "conf_threshold": self._conf_threshold,
            "single_pass_mode": self._single_pass_mode,
        }

    def estimate_single(
        self,
        frame: np.ndarray,
        camera_id: str = "default",
        frame_id: int = 0,
        timestamp: Optional[float] = None,
    ) -> Dict:
        """Run pose estimation on a single BGR frame.

        Args:
            frame: Raw BGR image as a NumPy array.
            camera_id: Source camera identifier.
            frame_id: Frame sequence number.
            timestamp: Unix timestamp; defaults to current time.

        Returns:
            Dict with keys:
            - ``poses``: :class:`list` of :class:`PoseResult`
            - ``person_detections``: :class:`list` of :class:`Detection` (only
              when ``single_pass_mode`` is ``True``)
            - ``inference_time_ms``: float
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to estimate_single — returning empty results.")
            return {"poses": [], "person_detections": [], "inference_time_ms": 0.0}

        ts = timestamp if timestamp is not None else time.time()

        t0 = time.perf_counter()
        raw = self._model.predict(
            source=frame,
            imgsz=self._input_size,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            device=self._device,
            verbose=False,
        )
        infer_ms = (time.perf_counter() - t0) * 1000

        poses, person_dets = self._parse_single_result(
            raw[0], camera_id=camera_id, frame_id=frame_id, timestamp=ts
        )
        return {
            "poses": poses,
            "person_detections": person_dets if self._single_pass_mode else [],
            "inference_time_ms": infer_ms,
        }

    def estimate_batch(
        self,
        frames: List[Dict],
    ) -> Dict[str, List[PoseResult]]:
        """Run pose estimation on multiple frames, returning results keyed by camera_id.

        Args:
            frames: List of dicts with keys ``frame`` (np.ndarray), ``camera_id``
                (str), ``frame_id`` (int), ``timestamp`` (float).

        Returns:
            Dict mapping ``camera_id`` → list of :class:`PoseResult`.
        """
        results: Dict[str, List[PoseResult]] = {}
        for fd in frames:
            cam_id = fd.get("camera_id", "default")
            out = self.estimate_single(
                frame=fd["frame"],
                camera_id=cam_id,
                frame_id=fd.get("frame_id", 0),
                timestamp=fd.get("timestamp"),
            )
            if cam_id not in results:
                results[cam_id] = []
            results[cam_id].extend(out["poses"])
        return results

    def _parse_single_result(
        self,
        raw,
        camera_id: str,
        frame_id: int,
        timestamp: float,
    ):
        """Convert a single ultralytics result into PoseResult and Detection lists."""
        poses: List[PoseResult] = []
        person_dets: List[Detection] = []

        if raw.boxes is None or len(raw.boxes) == 0:
            return poses, person_dets

        boxes = raw.boxes
        kp_data = raw.keypoints.data if raw.keypoints is not None else None

        for i, box in enumerate(boxes):
            conf = float(box.conf[0].item())
            if conf < self._conf_threshold:
                continue

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = (int(v) for v in xyxy)

            if self._single_pass_mode:
                person_dets.append(
                    Detection(
                        class_id=0,
                        class_name="person",
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        frame_idx=frame_id,
                    )
                )

            if kp_data is not None and i < kp_data.shape[0]:
                kp_tensor = kp_data[i]
                keypoints = [
                    Keypoint(
                        x=float(kp_tensor[j, 0].item()),
                        y=float(kp_tensor[j, 1].item()),
                        confidence=float(kp_tensor[j, 2].item()),
                    )
                    for j in range(17)
                ]
            else:
                keypoints = [Keypoint(x=0.0, y=0.0, confidence=0.0) for _ in range(17)]

            poses.append(
                PoseResult(
                    camera_id=camera_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    bbox=(x1, y1, x2, y2),
                    bbox_confidence=conf,
                    keypoints=keypoints,
                )
            )

        return poses, person_dets
