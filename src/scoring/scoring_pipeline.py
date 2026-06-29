"""
Scoring pipeline: orchestrates the full decision loop.

Connects action recognition, zone engine, and object detection outputs
to the anomaly scorer.
"""

from __future__ import annotations

import queue
from typing import List, Optional

from src.common.logger import get_logger
from src.detection.combined_pipeline import TrackedFrameAnalysis
from src.recognition.action_classes import ActionPrediction
from src.recognition.recognition_pipeline import ActionRecognitionPipeline
from src.scoring.anomaly_scorer import AnomalyScorer
from src.scoring.scoring_models import AlertDecision, ScoredEvent
from src.scoring.zone_engine import ZoneEngine
from src.scoring.zone_models import ZoneViolation

logger = get_logger("scoring.scoring_pipeline")

_WEAPON_CLASSES = {"knife", "gun", "rifle", "scissors", "weapon"}


class ScoringPipeline:
    """Orchestrates zone evaluation, action scoring, and alert decision making.

    Args:
        config: Full project config dict.
        zone_engine: Initialised :class:`~src.scoring.zone_engine.ZoneEngine`.
        action_pipeline: Initialised
            :class:`~src.recognition.recognition_pipeline.ActionRecognitionPipeline`.
        anomaly_scorer: Initialised :class:`~src.scoring.anomaly_scorer.AnomalyScorer`.
    """

    def __init__(
        self,
        config: dict,
        zone_engine: ZoneEngine,
        action_pipeline: ActionRecognitionPipeline,
        anomaly_scorer: AnomalyScorer,
    ) -> None:
        self._zone_engine = zone_engine
        self._action_pipeline = action_pipeline
        self._scorer = anomaly_scorer

        # Queues for downstream consumers
        self._alert_queue: queue.Queue[ScoredEvent] = queue.Queue(maxsize=500)
        self._all_events_queue: queue.Queue[ScoredEvent] = queue.Queue(maxsize=2000)

        logger.info("ScoringPipeline initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        tracked_analysis: TrackedFrameAnalysis,
        action_predictions: List[ActionPrediction],
        zone_violations: List[ZoneViolation],
    ) -> List[ScoredEvent]:
        """Run the full decision loop for one analysis cycle.

        1. Extract weapon/dangerous object detections from ``tracked_analysis``.
        2. Score all signals via the anomaly scorer.
        3. Enqueue ALERT and ESCALATED events for downstream consumers.
        4. Return all scored events.

        Args:
            tracked_analysis: Output from the detection/tracking pipeline.
            action_predictions: Action predictions for this frame.
            zone_violations: Zone rule violations for this frame.

        Returns:
            All :class:`~src.scoring.scoring_models.ScoredEvent` objects for
            this cycle, regardless of decision.
        """
        weapon_detections = [
            d for d in tracked_analysis.object_detections
            if d.class_name.lower() in _WEAPON_CLASSES
        ]

        events = self._scorer.score_frame(
            camera_id=tracked_analysis.camera_id,
            timestamp=tracked_analysis.timestamp,
            action_predictions=action_predictions,
            zone_violations=zone_violations,
            object_detections=weapon_detections,
        )

        for event in events:
            if event.alert_decision in (AlertDecision.ALERT, AlertDecision.ESCALATED):
                logger.warning(
                    "ALERT [%s] cam=%s track=%s score=%.3f label=%s",
                    event.alert_decision.value.upper(),
                    event.camera_id,
                    event.track_id,
                    event.severity_score,
                    event.event_label,
                )
                try:
                    self._alert_queue.put_nowait(event)
                except queue.Full:
                    logger.error("Alert queue full — dropping event %s", event.event_id)

            try:
                self._all_events_queue.put_nowait(event)
            except queue.Full:
                pass  # Non-critical, best-effort

        return events

    def get_alert_events(self, timeout: float = 1.0) -> List[ScoredEvent]:
        """Drain the alert queue (ALERT + ESCALATED decisions only).

        Args:
            timeout: Max seconds to wait for the first event if queue is empty.

        Returns:
            List of alert events (may be empty).
        """
        return self._drain_queue(self._alert_queue, timeout)

    def get_all_events(self, timeout: float = 1.0) -> List[ScoredEvent]:
        """Drain all events including NO_ALERT and SUPPRESSED.

        Args:
            timeout: Max seconds to wait for the first event if queue is empty.

        Returns:
            List of all scored events (may be empty).
        """
        return self._drain_queue(self._all_events_queue, timeout)

    def get_stats(self) -> dict:
        """Return combined stats from all sub-components.

        Returns:
            Merged stats dict.
        """
        stats = {}
        stats["scorer"] = self._scorer.get_stats()
        stats["zone_engine"] = self._zone_engine.get_stats()
        stats["alert_queue_size"] = self._alert_queue.qsize()
        stats["all_events_queue_size"] = self._all_events_queue.qsize()
        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drain_queue(q: queue.Queue, timeout: float) -> List[ScoredEvent]:
        events: List[ScoredEvent] = []
        try:
            events.append(q.get(timeout=timeout))
        except queue.Empty:
            return events
        while True:
            try:
                events.append(q.get_nowait())
            except queue.Empty:
                break
        return events
