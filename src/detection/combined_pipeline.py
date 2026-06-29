"""Combined detection and pose pipeline.

Runs YOLOv8-Pose for persons + YOLOv8 for non-person objects in an
efficient sequence, producing a unified :class:`FrameAnalysis` per frame.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.common.config import load_config
from src.common.logger import get_logger
from src.detection.pose_estimator import PoseEstimator
from src.detection.pose_structures import PoseResult
from src.detection.yolo_detector import Detection, YOLODetector

logger = get_logger("detection.combined_pipeline")

_NON_PERSON_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle",
    "backpack", "handbag", "knife", "scissors",
}


@dataclass
class FrameAnalysis:
    """All detection and pose results for a single frame.

    Attributes:
        camera_id: Source camera identifier.
        frame_id: Sequential frame index.
        timestamp: Unix timestamp of the frame.
        frame: Original BGR frame (retained for downstream clip capture).
        person_detections: Person bounding boxes from YOLOv8-Pose (single-pass).
        object_detections: Non-person object detections from standalone YOLOv8.
        poses: Skeleton keypoints per detected person.
        inference_time_ms: Total inference wall-time for this frame.
    """

    camera_id: str
    frame_id: int
    timestamp: float
    frame: np.ndarray
    person_detections: List[Detection] = field(default_factory=list)
    object_detections: List[Detection] = field(default_factory=list)
    poses: List[PoseResult] = field(default_factory=list)
    inference_time_ms: float = 0.0


class CombinedDetectionPipeline:
    """Orchestrates pose estimation and object detection for a stream of frames.

    Processing flow per frame:
    1. YOLOv8-Pose → person bounding boxes + skeleton keypoints.
    2. YOLOv8 (when ``skip_object_detection`` is ``False``) → non-person object
       detections (weapons, bags, vehicles).
    3. Combine into :class:`FrameAnalysis` and push to output queue.

    Args:
        config: Optional pre-loaded configuration dict.
        frame_buffer: Optional input :class:`queue.Queue` supplying frame dicts
            with keys ``frame``, ``camera_id``, ``frame_id``, ``timestamp``.
            If ``None``, call :meth:`process_frame` directly.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        frame_buffer: Optional[queue.Queue] = None,
    ) -> None:
        self._cfg = config or load_config()
        self._pose_estimator = PoseEstimator(config=self._cfg)
        self._skip_obj = self._cfg.get("pipeline", {}).get("skip_object_detection", False)

        if not self._skip_obj:
            self._obj_detector = YOLODetector(config=self._cfg)
        else:
            self._obj_detector = None

        self._frame_buffer = frame_buffer
        self._output_queue: queue.Queue = queue.Queue(maxsize=60)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._stats: Dict = {
            "frames_processed": 0,
            "total_inference_ms": 0.0,
            "total_persons": 0,
            "total_objects": 0,
        }
        logger.info(
            "CombinedDetectionPipeline ready. skip_object_detection=%s",
            self._skip_obj,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "default",
        frame_id: int = 0,
        timestamp: Optional[float] = None,
    ) -> FrameAnalysis:
        """Run the full detection+pose pipeline on a single BGR frame.

        Args:
            frame: Raw BGR image.
            camera_id: Source camera identifier.
            frame_id: Frame sequence number.
            timestamp: Unix timestamp; defaults to current time.

        Returns:
            :class:`FrameAnalysis` with all detection and pose results.
        """
        ts = timestamp if timestamp is not None else time.time()
        t0 = time.perf_counter()

        pose_out = self._pose_estimator.estimate_single(
            frame=frame, camera_id=camera_id, frame_id=frame_id, timestamp=ts
        )
        poses: List[PoseResult] = pose_out["poses"]
        person_dets: List[Detection] = pose_out["person_detections"]

        object_dets: List[Detection] = []
        if self._obj_detector is not None:
            det_result = self._obj_detector.detect(frame, frame_idx=frame_id)
            object_dets = [
                d for d in det_result.detections
                if d.class_name in _NON_PERSON_CLASSES
            ]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        self._stats["frames_processed"] += 1
        self._stats["total_inference_ms"] += elapsed_ms
        self._stats["total_persons"] += len(person_dets)
        self._stats["total_objects"] += len(object_dets)

        logger.debug(
            "camera=%s frame=%d persons=%d objects=%d poses=%d latency=%.1fms",
            camera_id, frame_id, len(person_dets), len(object_dets), len(poses), elapsed_ms,
        )

        return FrameAnalysis(
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp=ts,
            frame=frame,
            person_detections=person_dets,
            object_detections=object_dets,
            poses=poses,
            inference_time_ms=elapsed_ms,
        )

    def start(self) -> None:
        """Start the background processing loop (requires ``frame_buffer``)."""
        if self._frame_buffer is None:
            raise RuntimeError("frame_buffer must be provided to use start().")
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()
        logger.info("CombinedDetectionPipeline worker started.")

    def stop(self) -> None:
        """Signal the background loop to stop and wait for it to finish."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
        logger.info("CombinedDetectionPipeline worker stopped.")

    def get_results(self, timeout: float = 1.0) -> Optional[FrameAnalysis]:
        """Retrieve the next :class:`FrameAnalysis` from the output queue.

        Args:
            timeout: Seconds to wait for a result before returning ``None``.

        Returns:
            :class:`FrameAnalysis` or ``None`` if timed out.
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> Dict:
        """Return cumulative processing statistics.

        Returns:
            Dict with ``frames_processed``, ``total_inference_ms``,
            ``avg_inference_ms``, ``total_persons``, ``total_objects``.
        """
        n = self._stats["frames_processed"]
        return {
            **self._stats,
            "avg_inference_ms": (
                self._stats["total_inference_ms"] / n if n > 0 else 0.0
            ),
        }

    def _run_loop(self) -> None:
        """Background worker: pull frames, process, push results."""
        while not self._stop_event.is_set():
            try:
                frame_data = self._frame_buffer.get(timeout=0.5)
            except queue.Empty:
                continue

            analysis = self.process_frame(
                frame=frame_data["frame"],
                camera_id=frame_data.get("camera_id", "default"),
                frame_id=frame_data.get("frame_id", 0),
                timestamp=frame_data.get("timestamp"),
            )
            try:
                self._output_queue.put_nowait(analysis)
            except queue.Full:
                logger.warning("Output queue full — dropping frame %d.", analysis.frame_id)
