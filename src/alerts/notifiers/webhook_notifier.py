"""
Webhook notification channel.
Sends alert data as HTTP POST to configured endpoints.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import List
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from src.alerts.alert_models import Alert
from src.common.logger import get_logger

logger = get_logger("alerts.webhook_notifier")


class WebhookNotifier:
    """Delivers alerts to one or more HTTP webhook endpoints.

    Args:
        config: Full project config dict; reads ``notifications.webhook``.
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("notifications", {}).get("webhook", {})
        self._enabled: bool = bool(cfg.get("enabled", False))
        raw_endpoints: list = cfg.get("endpoints", [])
        self._endpoints: List[dict] = []

        for ep in raw_endpoints:
            url = ep.get("url", "").strip()
            if not url:
                continue
            try:
                result = urlparse(url)
                if result.scheme not in ("http", "https"):
                    raise ValueError(f"Invalid scheme: {result.scheme}")
            except Exception as exc:
                logger.warning("Skipping invalid webhook URL %r: %s", url, exc)
                continue
            self._endpoints.append({
                "url": url,
                "secret": ep.get("secret", ""),
                "headers": ep.get("headers", {}),
                "timeout": int(ep.get("timeout", 10)),
                "retry_count": int(ep.get("retry_count", 3)),
                "retry_delay": float(ep.get("retry_delay", 5.0)),
            })

        logger.info(
            "WebhookNotifier ready (enabled=%s, endpoints=%d).",
            self._enabled, len(self._endpoints),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(self, alert: Alert) -> bool:
        """Send an alert to all configured webhook endpoints.

        Args:
            alert: :class:`~src.alerts.alert_models.Alert` to deliver.

        Returns:
            ``True`` if at least one endpoint accepted the payload.
        """
        if not self._enabled:
            return False
        if not self._endpoints:
            return False

        payload = json.dumps({
            "event": "alert.created",
            "payload": alert.to_dict(),
            "timestamp": time.time(),
        })
        any_success = False
        for ep in self._endpoints:
            if self._send_to_endpoint(payload, ep):
                any_success = True
        return any_success

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _send_to_endpoint(self, payload: str, ep: dict) -> bool:
        headers = {"Content-Type": "application/json"}
        headers.update(ep.get("headers", {}))
        if ep["secret"]:
            headers["X-Signature-256"] = self._compute_signature(payload, ep["secret"])

        data = payload.encode("utf-8")
        for attempt in range(1, ep["retry_count"] + 1):
            try:
                req = Request(ep["url"], data=data, headers=headers, method="POST")
                with urlopen(req, timeout=ep["timeout"]) as resp:
                    status = resp.status
                if 200 <= status < 300:
                    logger.info("Webhook delivered to %s (status=%d).", ep["url"], status)
                    return True
                logger.warning(
                    "Webhook %s returned non-2xx status %d (attempt %d/%d).",
                    ep["url"], status, attempt, ep["retry_count"],
                )
            except HTTPError as exc:
                logger.warning(
                    "Webhook %s HTTP error %d (attempt %d/%d).",
                    ep["url"], exc.code, attempt, ep["retry_count"],
                )
            except (URLError, OSError) as exc:
                logger.warning(
                    "Webhook %s connection error: %s (attempt %d/%d).",
                    ep["url"], exc, attempt, ep["retry_count"],
                )
            except Exception as exc:
                logger.error("Webhook %s unexpected error: %s", ep["url"], exc)

            if attempt < ep["retry_count"]:
                time.sleep(ep["retry_delay"])

        logger.error("Webhook %s failed after %d attempts.", ep["url"], ep["retry_count"])
        return False

    def _compute_signature(self, payload: str, secret: str) -> str:
        """Compute HMAC-SHA256 signature of ``payload`` using ``secret``.

        Args:
            payload: JSON string to sign.
            secret: Shared secret key.

        Returns:
            Hex digest prefixed with ``sha256=``.
        """
        sig = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={sig}"
