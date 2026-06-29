"""
Integration point: connects the scoring pipeline output to clip capture.

When a ScoredEvent has decision=ALERT or ESCALATED, triggers clip capture.
"""

from __future__ import annotations

from typing import List

from src.alerts.clip_capture import ClipCaptureService
from src.alerts.clip_models import ClipRequest
from src.common.logger import get_logger
from src.scoring.scoring_models import AlertDecision, ScoredEvent

logger = get_logger("alerts.alert_integration")


class AlertIntegration:
    """Lightweight glue layer between the scoring pipeline and clip capture.

    For each :class:`~src.scoring.scoring_models.ScoredEvent` with decision
    ``ALERT`` or ``ESCALATED``, a :class:`~src.alerts.clip_models.ClipRequest`
    is created and submitted to the
    :class:`~src.alerts.clip_capture.ClipCaptureService`.

    Args:
        clip_service: Initialised :class:`~src.alerts.clip_capture.ClipCaptureService`.
        config: Full project config dict; reads ``storage`` section for pre/post seconds.
    """

    def __init__(self, clip_service: ClipCaptureService, config: dict) -> None:
        self._clip_service = clip_service
        cfg = config.get("storage", {})
        self._pre_seconds = float(cfg.get("clip_pre_seconds", 10.0))
        self._post_seconds = float(cfg.get("clip_post_seconds", 5.0))
        logger.info(
            "AlertIntegration ready (pre=%.0fs post=%.0fs).",
            self._pre_seconds, self._post_seconds,
        )

    def handle_scored_events(self, events: List[ScoredEvent]) -> List[ClipRequest]:
        """Submit clip requests for all actionable scored events.

        Args:
            events: All scored events from one analysis cycle.

        Returns:
            List of :class:`~src.alerts.clip_models.ClipRequest` objects that
            were submitted (for inspection / testing).
        """
        submitted: List[ClipRequest] = []
        for event in events:
            if event.alert_decision not in (AlertDecision.ALERT, AlertDecision.ESCALATED):
                continue
            priority = 2 if event.alert_decision == AlertDecision.ESCALATED else 1
            req = ClipRequest(
                event_id=event.event_id,
                camera_id=event.camera_id,
                event_timestamp=event.timestamp,
                pre_seconds=self._pre_seconds,
                post_seconds=self._post_seconds,
                priority=priority,
            )
            self._clip_service.request_clip(req)
            submitted.append(req)
            logger.debug(
                "Clip request submitted: event=%s cam=%s decision=%s priority=%d",
                event.event_id, event.camera_id,
                event.alert_decision.value, priority,
            )
        return submitted
