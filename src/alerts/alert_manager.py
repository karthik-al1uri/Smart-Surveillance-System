"""
Central Alert Manager.
Coordinates alert creation, delivery across channels, status tracking,
rate limiting, grouping, and escalation.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Dict, List, Optional

from src.alerts.alert_builder import AlertBuilder
from src.alerts.alert_models import Alert, AlertPriority, AlertStatus
from src.alerts.notifiers.email_notifier import EmailNotifier
from src.alerts.notifiers.webhook_notifier import WebhookNotifier
from src.alerts.notifiers.websocket_notifier import WebSocketNotifier
from src.common.logger import get_logger
from src.scoring.scoring_models import AlertDecision, ScoredEvent

logger = get_logger("alerts.alert_manager")


class AlertManager:
    """Central coordinator for the alert lifecycle.

    Responsibilities:
    - Build :class:`~src.alerts.alert_models.Alert` objects from scored events
    - Rate-limit delivery (per-camera + global sliding windows)
    - Group similar alerts within a configurable time window
    - Deliver via all enabled channels
    - Track status and acknowledge/dismiss operations
    - Escalate unacknowledged alerts after a configurable timeout

    Args:
        config: Full project config dict.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        alert_cfg = config.get("alerts", {})
        rl_cfg = alert_cfg.get("rate_limit", {})
        group_cfg = alert_cfg.get("grouping", {})
        esc_cfg = alert_cfg.get("escalation", {})

        self._max_per_cam = int(rl_cfg.get("max_alerts_per_minute", 10))
        self._max_global = int(rl_cfg.get("max_alerts_per_minute_global", 30))
        self._grouping_enabled = bool(group_cfg.get("enabled", True))
        self._group_window = float(group_cfg.get("window_seconds", 10.0))
        self._group_same_cam = bool(group_cfg.get("same_camera_only", True))
        self._escalation_enabled = bool(esc_cfg.get("enabled", True))
        self._escalation_timeout = float(esc_cfg.get("timeout_seconds", 300.0))
        self._escalation_channels = esc_cfg.get("escalation_channels", ["email"])
        self._history_size = int(alert_cfg.get("history_size", 1000))

        self._builder = AlertBuilder(config)
        self._ws_notifier = WebSocketNotifier(config)
        self._webhook_notifier = WebhookNotifier(config)
        self._email_notifier = EmailNotifier(config)

        # alert_id → Alert
        self._alerts: Dict[str, Alert] = {}
        # Ordered list of alert_ids for history eviction
        self._history_order: collections.deque = collections.deque()

        # Sliding window timestamps for rate limiting
        # camera_id → deque of timestamps
        self._cam_window: Dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._global_window: collections.deque = collections.deque()

        # Channel delivery success counts
        self._channel_success: Dict[str, int] = {"websocket": 0, "webhook": 0, "email": 0}
        self._channel_attempts: Dict[str, int] = {"websocket": 0, "webhook": 0, "email": 0}

        self._lock = threading.Lock()
        self._running = False
        self._escalation_thread: Optional[threading.Thread] = None

        logger.info(
            "AlertManager ready (rate_limit=%d/cam %d/global grouping=%s escalation=%s).",
            self._max_per_cam, self._max_global,
            self._grouping_enabled, self._escalation_enabled,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start WebSocket server and background escalation checker."""
        self._ws_notifier.start()
        if self._escalation_enabled:
            self._running = True
            self._escalation_thread = threading.Thread(
                target=self._escalation_loop, daemon=True, name="alert-escalation"
            )
            self._escalation_thread.start()
        logger.info("AlertManager started.")

    def stop(self) -> None:
        """Stop all background threads and notifiers."""
        self._running = False
        if self._escalation_thread:
            self._escalation_thread.join(timeout=5.0)
        self._ws_notifier.stop()
        logger.info("AlertManager stopped.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_events(self, events: List[ScoredEvent]) -> List[Alert]:
        """Process scored events and deliver alerts for actionable ones.

        Args:
            events: All scored events from one analysis cycle.

        Returns:
            List of :class:`~src.alerts.alert_models.Alert` objects that were
            created (does not include suppressed/grouped events).
        """
        created: List[Alert] = []
        for event in events:
            if event.alert_decision not in (AlertDecision.ALERT, AlertDecision.ESCALATED):
                continue
            alert = self._process_single_event(event)
            if alert:
                created.append(alert)
        return created

    def _process_single_event(self, event: ScoredEvent) -> Optional[Alert]:
        now = time.time()

        # Rate limiting
        if not self._check_rate_limit(event.camera_id, now):
            logger.warning(
                "Rate limit exceeded for camera %s — alert suppressed.", event.camera_id
            )
            return None

        # Grouping
        if self._grouping_enabled:
            grouped = self._try_group(event, now)
            if grouped:
                logger.debug("Event grouped into existing alert %s.", grouped.alert_id)
                return None

        alert = self._builder.build_alert(event)
        self._register_alert(alert, now)
        self.deliver_alert(alert)
        return alert

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    def deliver_alert(self, alert: Alert) -> Dict[str, bool]:
        """Deliver an alert through all enabled channels.

        Args:
            alert: The alert to deliver.

        Returns:
            Dict mapping channel name to delivery success.
        """
        results: Dict[str, bool] = {}

        self._channel_attempts["websocket"] += 1
        ws_ok = self._ws_notifier.send_alert(alert)
        results["websocket"] = ws_ok
        if ws_ok:
            self._channel_success["websocket"] += 1

        self._channel_attempts["webhook"] += 1
        wh_ok = self._webhook_notifier.send_alert(alert)
        results["webhook"] = wh_ok
        if wh_ok:
            self._channel_success["webhook"] += 1

        self._channel_attempts["email"] += 1
        em_ok = self._email_notifier.send_alert(alert)
        results["email"] = em_ok
        if em_ok:
            self._channel_success["email"] += 1

        with self._lock:
            if any(results.values()):
                alert.status = AlertStatus.DELIVERED
                alert.delivered_at = time.time()
            else:
                alert.status = AlertStatus.FAILED
            alert.delivery_attempts = {k: 1 for k in results}

        logger.debug(
            "Alert %s delivered: ws=%s wh=%s em=%s",
            alert.alert_id, ws_ok, wh_ok, em_ok,
        )
        return results

    # ------------------------------------------------------------------
    # Status operations
    # ------------------------------------------------------------------

    def acknowledge_alert(self, alert_id: str, operator: str) -> bool:
        """Acknowledge an alert, cancelling its escalation timer.

        Args:
            alert_id: Alert to acknowledge.
            operator: Operator username.

        Returns:
            ``True`` if the alert was found and updated.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                return False
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = time.time()
            alert.acknowledged_by = operator
        logger.info("Alert %s acknowledged by %s.", alert_id, operator)
        return True

    def dismiss_alert(self, alert_id: str, operator: str, reason: str) -> bool:
        """Dismiss an alert (false positive, duplicate, resolved).

        Args:
            alert_id: Alert to dismiss.
            operator: Operator username.
            reason: One of ``"false_positive"``, ``"duplicate"``, ``"resolved"``.

        Returns:
            ``True`` if found and updated.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                return False
            alert.status = AlertStatus.DISMISSED
            alert.dismissed_at = time.time()
            alert.acknowledged_by = operator
            alert.dismiss_reason = reason
        logger.info("Alert %s dismissed by %s (reason=%s).", alert_id, operator, reason)
        return True

    def escalate_alert(self, alert_id: str) -> None:
        """Escalate an unacknowledged alert via escalation channels.

        Args:
            alert_id: Alert to escalate.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                return
            alert.status = AlertStatus.ESCALATED
            alert.escalated_at = time.time()

        logger.warning("Escalating alert %s.", alert_id)
        if "email" in self._escalation_channels:
            self._email_notifier.send_alert(alert)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Return a specific alert by ID."""
        return self._alerts.get(alert_id)

    def get_active_alerts(self) -> List[Alert]:
        """Return alerts that are PENDING or DELIVERED (unacknowledged)."""
        with self._lock:
            return [
                a for a in self._alerts.values()
                if a.status in (AlertStatus.PENDING, AlertStatus.DELIVERED)
            ]

    def get_alert_history(
        self,
        limit: int = 100,
        camera_id: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> List[Alert]:
        """Return recent alerts, optionally filtered.

        Args:
            limit: Maximum number of results.
            camera_id: Filter to one camera.
            priority: Filter by :class:`~src.alerts.alert_models.AlertPriority` value string.

        Returns:
            List of matching alerts, newest first.
        """
        with self._lock:
            alerts = list(self._alerts.values())
        if camera_id:
            alerts = [a for a in alerts if a.camera_id == camera_id]
        if priority:
            alerts = [a for a in alerts if a.priority.value == priority]
        alerts.sort(key=lambda a: a.created_at, reverse=True)
        return alerts[:limit]

    def get_stats(self) -> dict:
        """Return alert statistics.

        Returns:
            Dict with counts by status, per-channel success rates, and
            average response time.
        """
        with self._lock:
            alerts = list(self._alerts.values())

        by_status: Dict[str, int] = {}
        for a in alerts:
            by_status[a.status.value] = by_status.get(a.status.value, 0) + 1

        channel_rates: Dict[str, float] = {}
        for ch in ("websocket", "webhook", "email"):
            attempts = self._channel_attempts.get(ch, 0)
            success = self._channel_success.get(ch, 0)
            channel_rates[ch] = round(success / attempts, 3) if attempts else 0.0

        ack_times = [
            a.acknowledged_at - a.created_at
            for a in alerts
            if a.acknowledged_at and a.created_at
        ]
        avg_ack_time = sum(ack_times) / len(ack_times) if ack_times else None

        return {
            "total_alerts": len(alerts),
            "by_status": by_status,
            "channel_success_rates": channel_rates,
            "avg_acknowledgement_seconds": avg_ack_time,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_rate_limit(self, camera_id: str, now: float) -> bool:
        """Return ``True`` if the alert is within rate limits."""
        window_start = now - 60.0
        with self._lock:
            cam_q = self._cam_window[camera_id]
            while cam_q and cam_q[0] < window_start:
                cam_q.popleft()
            while self._global_window and self._global_window[0] < window_start:
                self._global_window.popleft()

            if len(cam_q) >= self._max_per_cam:
                return False
            if len(self._global_window) >= self._max_global:
                return False

            cam_q.append(now)
            self._global_window.append(now)
        return True

    def _try_group(self, event: ScoredEvent, now: float) -> Optional[Alert]:
        """Return an existing alert to group into, or ``None``.

        Args:
            event: Incoming scored event.
            now: Current wall-clock timestamp.

        Returns:
            Existing :class:`~src.alerts.alert_models.Alert` if grouping applies.
        """
        window_start = now - self._group_window
        with self._lock:
            for alert in list(self._alerts.values()):
                if alert.created_at < window_start:
                    continue
                if alert.event_category != event.event_category:
                    continue
                if self._group_same_cam and alert.camera_id != event.camera_id:
                    continue
                if alert.status in (AlertStatus.DISMISSED, AlertStatus.ACKNOWLEDGED):
                    continue
                # Attach to this group
                alert.severity_score = max(alert.severity_score, event.severity_score)
                # Update description to mention multiple occurrences
                if "Multiple events" not in alert.description:
                    alert.description = (
                        f"Multiple events detected (grouped). {alert.description}"
                    )
                return alert
        return None

    def _register_alert(self, alert: Alert, now: float) -> None:
        with self._lock:
            self._alerts[alert.alert_id] = alert
            self._history_order.append(alert.alert_id)
            # Evict oldest if over history_size
            while len(self._history_order) > self._history_size:
                old_id = self._history_order.popleft()
                self._alerts.pop(old_id, None)

    def _escalation_loop(self) -> None:
        while self._running:
            time.sleep(30.0)
            if not self._running:
                break
            now = time.time()
            with self._lock:
                candidates = [
                    a for a in self._alerts.values()
                    if a.status == AlertStatus.DELIVERED
                    and a.delivered_at is not None
                    and now - a.delivered_at >= self._escalation_timeout
                ]
            for alert in candidates:
                self.escalate_alert(alert.alert_id)
