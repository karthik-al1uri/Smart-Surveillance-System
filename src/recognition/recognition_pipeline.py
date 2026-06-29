"""Action recognition pipeline.

Consumes tracked frame analyses, manages sliding windows,
and produces action predictions for each tracked person.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.common.config import load_config
from src.common.logger import get_logger
from src.detection.combined_pipeline import TrackedFrameAnalysis
from src.recognition.action_classes import ActionCategory, ActionPrediction
from src.recognition.action_classifier import ActionClassifier
from src.recognition.sliding_window import SlidingWindowManager, WindowData

logger = get_logger("recognition.pipeline")


class ActionRecognitionPipeline:
    """Consumes :class:`TrackedFrameAnalysis` frames and emits action predictions.

    Keeps a :class:`SlidingWindowManager` to accumulate per-track keypoint
    history and a :class:`ActionClassifier` to classify ready windows.

    This is intentionally a SEPARATE stage from
    :class:`~src.detection.combined_pipeline.CombinedDetectionPipeline` for
    modularity — detection/tracking and recognition can run at different rates
    or be disabled independently.

    Args:
        config: Optional pre-loaded config dict.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg = config or load_config()
        self._window_mgr = SlidingWindowManager(config=self._cfg)
        self._classifier = ActionClassifier(config=self._cfg)
        self._active = False

        self._stats: Dict = {
            "frames_processed": 0,
            "total_predictions": 0,
            "predictions_by_category": {cat.name: 0 for cat in ActionCategory},
        }

        logger.info("ActionRecognitionPipeline initialised.")

    def process(self, tracked_analysis: TrackedFrameAnalysis) -> List[ActionPrediction]:
        """Process one frame and return any ready action predictions.

        Args:
            tracked_analysis: :class:`TrackedFrameAnalysis` from the detection
                pipeline.

        Returns:
            List of :class:`~src.recognition.action_classes.ActionPrediction`
            objects — may be empty if no windows are ready yet.
        """
        self._stats["frames_processed"] += 1

        removed_ids = {
            t.track_id for t in self._window_mgr._buffers
            if t not in {tr.track_id for tr in tracked_analysis.tracks}
        } if hasattr(self._window_mgr, "_buffers") else set()

        active_ids = {t.track_id for t in tracked_analysis.tracks if t.state == "active"}
        buffer_ids = set(self._window_mgr._buffers.keys())
        stale = buffer_ids - active_ids - {
            t.track_id for t in tracked_analysis.tracks if t.state == "lost"
        }
        for tid in stale:
            self._window_mgr.remove_track(tid)

        for track in tracked_analysis.tracks:
            if track.state != "active":
                continue
            if track.keypoint_history:
                kp = list(track.keypoint_history)[-1]
                self._window_mgr.update(track.track_id, kp, tracked_analysis.frame_id)

        ready: List[WindowData] = self._window_mgr.get_ready_windows()
        if not ready:
            return []

        predictions = self._classifier.classify_batch(ready)
        for pred in predictions:
            pred.camera_id = tracked_analysis.camera_id

        self._stats["total_predictions"] += len(predictions)
        for pred in predictions:
            self._stats["predictions_by_category"][pred.category.name] += 1

        return predictions

    def start(self) -> None:
        """Mark the pipeline as active (lifecycle hook for future async use)."""
        self._active = True
        logger.info("ActionRecognitionPipeline started.")

    def stop(self) -> None:
        """Mark the pipeline as inactive and clear all track buffers."""
        self._active = False
        self._window_mgr._buffers.clear()
        logger.info("ActionRecognitionPipeline stopped.")

    def get_stats(self) -> dict:
        """Return processing statistics.

        Returns:
            Dict with ``frames_processed``, ``total_predictions``,
            ``predictions_by_category``, and window manager stats.
        """
        return {
            **self._stats,
            "window_stats": self._window_mgr.get_stats(),
        }
