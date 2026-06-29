"""
Email notification channel.
Sends formatted alert emails via SMTP.
"""

from __future__ import annotations

import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from jinja2 import Template

from src.alerts.alert_models import Alert, AlertPriority
from src.common.logger import get_logger

logger = get_logger("alerts.email_notifier")

_PRIORITY_ORDER = {
    AlertPriority.LOW: 0,
    AlertPriority.MEDIUM: 1,
    AlertPriority.HIGH: 2,
    AlertPriority.CRITICAL: 3,
}

_EMAIL_TEMPLATE = """\
Subject: [{{ priority }}] {{ title }}

SURVEILLANCE ALERT
──────────────────
Event:    {{ event_label }} ({{ event_category }})
Camera:   {{ camera_id }}
Time:     {{ formatted_time }}
Severity: {{ "%.2f"|format(severity_score) }}
Zone:     {{ zone_name or "N/A" }}

{{ description }}

View in dashboard: {{ dashboard_url }}/events/{{ event_id }}
"""


class EmailNotifier:
    """Sends formatted alert emails via SMTP.

    If SMTP host or username is not configured, the notifier operates in
    disabled mode and ``send_alert`` always returns ``False`` without crashing.

    Args:
        config: Full project config dict; reads ``notifications.email``.
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("notifications", {}).get("email", {})
        self._enabled: bool = bool(cfg.get("enabled", False))
        self._smtp_host: str = cfg.get("smtp_host", "")
        self._smtp_port: int = int(cfg.get("smtp_port", 587))
        self._username: str = cfg.get("smtp_username", "")
        self._password: str = cfg.get("smtp_password", "")
        self._use_tls: bool = bool(cfg.get("use_tls", True))
        self._from_address: str = cfg.get("from_address", "alerts@surveillance.local")
        self._recipients: List[str] = cfg.get("recipients", [])
        self._min_interval: float = float(cfg.get("min_interval_seconds", 300.0))
        min_pri_str: str = cfg.get("min_priority", "high")
        self._min_priority: AlertPriority = AlertPriority(min_pri_str)
        self._dashboard_url: str = config.get("dashboard_url", "http://localhost:3000")

        self._last_sent: float = 0.0
        self._template = Template(_EMAIL_TEMPLATE)

        if not self._smtp_host or not self._username:
            logger.info("Email notifications disabled — SMTP not configured.")
            self._enabled = False
        else:
            logger.info(
                "EmailNotifier ready: host=%s port=%d recipients=%s",
                self._smtp_host, self._smtp_port, self._recipients,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(self, alert: Alert) -> bool:
        """Send an email for the given alert.

        Returns ``False`` if:
        - SMTP is not configured / notifier is disabled
        - Rate limit is active
        - Alert priority is below the configured minimum
        - SMTP send fails

        Args:
            alert: :class:`~src.alerts.alert_models.Alert` to deliver.

        Returns:
            ``True`` on successful send.
        """
        if not self._enabled:
            return False

        if _PRIORITY_ORDER.get(alert.priority, 0) < _PRIORITY_ORDER.get(self._min_priority, 0):
            logger.debug(
                "Email skipped: priority %s < min %s.",
                alert.priority.value, self._min_priority.value,
            )
            return False

        now = time.time()
        if now - self._last_sent < self._min_interval:
            logger.debug("Email rate-limited (last sent %.0fs ago).", now - self._last_sent)
            return False

        if not self._recipients:
            logger.warning("No email recipients configured.")
            return False

        body = self._render_body(alert)
        subject = f"[{alert.priority.value.upper()}] {alert.title}"
        return self._send(subject, body)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _render_body(self, alert: Alert) -> str:
        formatted_time = datetime.fromtimestamp(alert.timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return self._template.render(
            priority=alert.priority.value.upper(),
            title=alert.title,
            event_label=alert.event_label,
            event_category=alert.event_category,
            camera_id=alert.camera_id,
            formatted_time=formatted_time,
            severity_score=alert.severity_score,
            zone_name=alert.zone_name,
            description=alert.description,
            dashboard_url=self._dashboard_url,
            event_id=alert.event_id,
        )

    def _send(self, subject: str, body: str) -> bool:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._from_address
        msg["To"] = ", ".join(self._recipients)
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=10) as smtp:
                if self._use_tls:
                    smtp.starttls()
                smtp.login(self._username, self._password)
                smtp.sendmail(self._from_address, self._recipients, msg.as_string())
            self._last_sent = time.time()
            logger.info("Email sent to %s.", self._recipients)
            return True
        except smtplib.SMTPException as exc:
            logger.error("SMTP error: %s", exc)
            return False
        except OSError as exc:
            logger.error("Email network error: %s", exc)
            return False
