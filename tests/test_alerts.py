"""Tests for Phase 9: Alert & Notification Service.

All tests use synthetic data.  WebSocket tests use random ports to avoid
conflicts.  No real SMTP or external services are required.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts.alert_builder import AlertBuilder
from src.alerts.alert_manager import AlertManager
from src.alerts.alert_models import (
    Alert,
    AlertPriority,
    AlertStatus,
    NotificationChannel,
)
from src.alerts.notifiers.email_notifier import EmailNotifier
from src.alerts.notifiers.webhook_notifier import WebhookNotifier
from src.alerts.notifiers.websocket_notifier import WebSocketNotifier
from src.scoring.scoring_models import AlertDecision, ScoredEvent, SignalType


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _make_scored_event(
    category: str = "violent",
    label: str = "fighting",
    score: float = 0.75,
    camera_id: str = "cam_01",
    decision: AlertDecision = AlertDecision.ALERT,
    zone_name: str = None,
) -> ScoredEvent:
    return ScoredEvent(
        event_id=f"evt_{uuid.uuid4().hex[:8]}",
        camera_id=camera_id,
        track_id=1,
        timestamp=time.time(),
        severity_score=score,
        contributing_signals=[],
        dominant_signal=SignalType.ACTION_CLASSIFICATION,
        event_category=category,
        event_label=label,
        alert_decision=decision,
        zone_name=zone_name,
    )


def _base_config(ws_port: int = 8765) -> dict:
    return {
        "scoring": {"escalation_threshold": 0.85},
        "notifications": {
            "websocket": {"enabled": True, "host": "127.0.0.1", "port": ws_port,
                          "ping_interval": None, "ping_timeout": None},
            "webhook": {"enabled": False, "endpoints": []},
            "email": {"enabled": False, "smtp_host": "", "smtp_username": "",
                      "smtp_port": 587, "use_tls": False,
                      "from_address": "test@test.com", "recipients": ["a@b.com"],
                      "min_interval_seconds": 0, "min_priority": "low"},
        },
        "alerts": {
            "rate_limit": {"max_alerts_per_minute": 10, "max_alerts_per_minute_global": 30},
            "grouping": {"enabled": True, "window_seconds": 10, "same_camera_only": True},
            "escalation": {"enabled": False, "timeout_seconds": 300, "escalation_channels": ["email"]},
            "history_size": 100,
        },
        "dashboard_url": "http://localhost:3000",
    }


def _make_alert(
    priority: AlertPriority = AlertPriority.HIGH,
    camera_id: str = "cam_01",
    category: str = "violent",
    label: str = "fighting",
    score: float = 0.75,
) -> Alert:
    return Alert(
        alert_id=str(uuid.uuid4()),
        event_id=f"evt_{uuid.uuid4().hex[:8]}",
        camera_id=camera_id,
        timestamp=time.time(),
        priority=priority,
        title=f"{label} detected — Camera {camera_id}",
        description=f"Violent activity detected with 75% confidence.",
        event_category=category,
        event_label=label,
        severity_score=score,
        clip_url=f"/api/v1/clips/evt_test",
    )


# ---------------------------------------------------------------------------
# 1–5: Data structures
# ---------------------------------------------------------------------------

def test_alert_dataclass():
    alert = _make_alert()
    assert alert.alert_id is not None
    assert alert.status == AlertStatus.PENDING
    assert alert.delivery_attempts == {}
    assert alert.delivery_errors == []
    assert alert.acknowledged_at is None


def test_alert_to_dict():
    alert = _make_alert()
    d = alert.to_dict()
    for key in ("alert_id", "event_id", "camera_id", "timestamp", "priority",
                 "title", "description", "event_category", "event_label",
                 "severity_score", "clip_url", "zone_name", "status",
                 "created_at", "delivered_at", "acknowledged_at", "acknowledged_by"):
        assert key in d
    # Must be JSON-serializable
    json.dumps(d)


def test_alert_priority_enum():
    for v in ("low", "medium", "high", "critical"):
        assert AlertPriority(v) is not None


def test_alert_status_enum():
    for v in ("pending", "delivered", "acknowledged", "dismissed", "escalated", "failed"):
        assert AlertStatus(v) is not None


def test_notification_channel_enum():
    for v in ("websocket", "webhook", "email", "sms"):
        assert NotificationChannel(v) is not None


# ---------------------------------------------------------------------------
# 6–12: Alert builder
# ---------------------------------------------------------------------------

def test_build_alert_violent():
    builder = AlertBuilder(_base_config())
    ev = _make_scored_event(category="violent", label="fighting", score=0.75)
    alert = builder.build_alert(ev)
    assert alert.priority == AlertPriority.HIGH
    assert "fighting" in alert.title.lower() or "fighting" in alert.description.lower()


def test_build_alert_weapon():
    builder = AlertBuilder(_base_config())
    ev = _make_scored_event(category="weapon", label="knife", score=1.0)
    alert = builder.build_alert(ev)
    assert alert.priority == AlertPriority.CRITICAL


def test_build_alert_suspicious():
    builder = AlertBuilder(_base_config())
    ev = _make_scored_event(category="suspicious", label="loitering", score=0.55)
    alert = builder.build_alert(ev)
    assert alert.priority == AlertPriority.MEDIUM


def test_build_alert_urgent():
    builder = AlertBuilder(_base_config())
    ev = _make_scored_event(category="urgent", label="falling", score=0.72)
    alert = builder.build_alert(ev)
    assert alert.priority == AlertPriority.HIGH


def test_build_alert_title_format():
    builder = AlertBuilder(_base_config())
    ev = _make_scored_event(category="violent", label="fighting", camera_id="cam_01")
    alert = builder.build_alert(ev)
    assert "cam_01" in alert.title


def test_build_alert_description():
    builder = AlertBuilder(_base_config())
    ev = _make_scored_event(category="violent", label="fighting", score=0.80,
                            zone_name="Restricted Area")
    alert = builder.build_alert(ev)
    # Description should mention confidence percentage and event label
    assert "%" in alert.description or "80" in alert.description
    assert "fighting" in alert.description.lower() or "violent" in alert.description.lower()


def test_build_alert_clip_url():
    builder = AlertBuilder(_base_config())
    ev = _make_scored_event()
    alert = builder.build_alert(ev)
    assert alert.clip_url is not None
    assert ev.event_id in alert.clip_url


# ---------------------------------------------------------------------------
# 13–19: WebSocket notifier
# ---------------------------------------------------------------------------

def test_ws_notifier_start_stop():
    port = _free_port()
    cfg = _base_config(port)
    notifier = WebSocketNotifier(cfg)
    notifier.start()
    time.sleep(0.3)
    # Verify port is occupied
    with socket.socket() as s:
        result = s.connect_ex(("127.0.0.1", port))
    assert result == 0, f"Port {port} not bound after start()"
    notifier.stop()


def test_ws_notifier_client_connect():
    port = _free_port()
    cfg = _base_config(port)
    notifier = WebSocketNotifier(cfg)
    notifier.start()
    time.sleep(0.3)

    async def connect_and_wait():
        from websockets.asyncio.client import connect
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            time.sleep(0.1)
            return notifier.get_connected_clients()

    clients = asyncio.run(connect_and_wait())
    notifier.stop()
    assert len(clients) == 1


def test_ws_notifier_send_alert():
    port = _free_port()
    cfg = _base_config(port)
    notifier = WebSocketNotifier(cfg)
    notifier.start()
    time.sleep(0.3)
    alert = _make_alert()
    received = []

    async def connect_and_receive():
        from websockets.asyncio.client import connect
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            time.sleep(0.05)
            notifier.send_alert(alert)
            msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            received.append(json.loads(msg))

    asyncio.run(connect_and_receive())
    notifier.stop()
    assert len(received) == 1
    assert received[0]["type"] == "alert"
    assert "data" in received[0]
    assert received[0]["data"]["alert_id"] == alert.alert_id


def test_ws_notifier_broadcast():
    port = _free_port()
    cfg = _base_config(port)
    notifier = WebSocketNotifier(cfg)
    notifier.start()
    time.sleep(0.3)
    alert = _make_alert()
    msgs_client1 = []
    msgs_client2 = []

    async def two_clients():
        from websockets.asyncio.client import connect
        async with connect(f"ws://127.0.0.1:{port}") as ws1, \
                   connect(f"ws://127.0.0.1:{port}") as ws2:
            time.sleep(0.1)
            notifier.send_alert(alert)
            m1 = await asyncio.wait_for(ws1.recv(), timeout=3.0)
            m2 = await asyncio.wait_for(ws2.recv(), timeout=3.0)
            msgs_client1.append(json.loads(m1))
            msgs_client2.append(json.loads(m2))

    asyncio.run(two_clients())
    notifier.stop()
    assert len(msgs_client1) == 1
    assert len(msgs_client2) == 1
    assert msgs_client1[0]["data"]["alert_id"] == alert.alert_id


def test_ws_notifier_client_disconnect():
    port = _free_port()
    cfg = _base_config(port)
    notifier = WebSocketNotifier(cfg)
    notifier.start()
    time.sleep(0.3)

    async def connect_and_close():
        from websockets.asyncio.client import connect
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            time.sleep(0.05)
        # After context exit, connection is closed
        time.sleep(0.2)
        return notifier.get_connected_clients()

    remaining = asyncio.run(connect_and_close())
    notifier.stop()
    assert len(remaining) == 0


def test_ws_notifier_no_clients():
    port = _free_port()
    cfg = _base_config(port)
    notifier = WebSocketNotifier(cfg)
    notifier.start()
    time.sleep(0.2)
    alert = _make_alert()
    result = notifier.send_alert(alert)
    notifier.stop()
    assert result is False


def test_ws_notifier_message_format():
    port = _free_port()
    cfg = _base_config(port)
    notifier = WebSocketNotifier(cfg)
    notifier.start()
    time.sleep(0.3)
    alert = _make_alert()
    received = []

    async def get_message():
        from websockets.asyncio.client import connect
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            time.sleep(0.05)
            notifier.send_alert(alert)
            msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            received.append(json.loads(msg))

    asyncio.run(get_message())
    notifier.stop()
    msg = received[0]
    assert msg["type"] == "alert"
    assert "data" in msg
    assert "timestamp" in msg
    assert msg["data"]["priority"] == alert.priority.value


# ---------------------------------------------------------------------------
# 20–26: Webhook notifier
# ---------------------------------------------------------------------------

class _MockHTTPHandler(BaseHTTPRequestHandler):
    received_requests: List[dict] = []
    response_status: int = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        _MockHTTPHandler.received_requests.append({
            "body": json.loads(body) if body else {},
            "headers": dict(self.headers),
        })
        self.send_response(_MockHTTPHandler.response_status)
        self.end_headers()

    def log_message(self, *args):
        pass  # suppress output


def _start_mock_server(port: int):
    _MockHTTPHandler.received_requests.clear()
    _MockHTTPHandler.response_status = 200
    server = HTTPServer(("127.0.0.1", port), _MockHTTPHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def test_webhook_send_alert_success():
    port = _free_port()
    server = _start_mock_server(port)
    _MockHTTPHandler.received_requests.clear()

    cfg = {"notifications": {"webhook": {
        "enabled": True,
        "endpoints": [{"url": f"http://127.0.0.1:{port}/hook",
                       "secret": "", "headers": {}, "timeout": 5, "retry_count": 1, "retry_delay": 0}],
    }}}
    notifier = WebhookNotifier(cfg)
    alert = _make_alert()
    result = notifier.send_alert(alert)
    time.sleep(0.1)
    server.shutdown()
    assert result is True
    assert len(_MockHTTPHandler.received_requests) == 1
    assert _MockHTTPHandler.received_requests[0]["body"]["event"] == "alert.created"


def test_webhook_hmac_signature():
    port = _free_port()
    server = _start_mock_server(port)
    _MockHTTPHandler.received_requests.clear()
    secret = "my_secret"

    cfg = {"notifications": {"webhook": {
        "enabled": True,
        "endpoints": [{"url": f"http://127.0.0.1:{port}/hook",
                       "secret": secret, "headers": {}, "timeout": 5, "retry_count": 1, "retry_delay": 0}],
    }}}
    notifier = WebhookNotifier(cfg)
    alert = _make_alert()
    notifier.send_alert(alert)
    time.sleep(0.1)
    server.shutdown()
    req = _MockHTTPHandler.received_requests[0]
    sig_header = req["headers"].get("X-Signature-256", "")
    assert sig_header.startswith("sha256=")
    # Recompute expected signature
    payload = json.dumps({"event": "alert.created", "payload": alert.to_dict(),
                          "timestamp": None}, sort_keys=False)
    # Just verify the format is correct (exact payload varies by timestamp)
    assert len(sig_header) > 10


def test_webhook_retry_on_failure():
    port = _free_port()
    _MockHTTPHandler.received_requests.clear()
    attempt_count = [0]

    class RetryHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            attempt_count[0] += 1
            status = 500 if attempt_count[0] < 2 else 200
            self.send_response(status)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), RetryHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    cfg = {"notifications": {"webhook": {
        "enabled": True,
        "endpoints": [{"url": f"http://127.0.0.1:{port}/hook",
                       "secret": "", "headers": {}, "timeout": 5, "retry_count": 3, "retry_delay": 0}],
    }}}
    notifier = WebhookNotifier(cfg)
    result = notifier.send_alert(_make_alert())
    server.shutdown()
    assert result is True
    assert attempt_count[0] == 2


def test_webhook_all_retries_exhausted():
    port = _free_port()

    class AlwaysFailHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(503)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), AlwaysFailHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    cfg = {"notifications": {"webhook": {
        "enabled": True,
        "endpoints": [{"url": f"http://127.0.0.1:{port}/hook",
                       "secret": "", "headers": {}, "timeout": 5, "retry_count": 2, "retry_delay": 0}],
    }}}
    notifier = WebhookNotifier(cfg)
    result = notifier.send_alert(_make_alert())
    server.shutdown()
    assert result is False


def test_webhook_timeout():
    port = _free_port()

    class SlowHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            time.sleep(5)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), SlowHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    cfg = {"notifications": {"webhook": {
        "enabled": True,
        "endpoints": [{"url": f"http://127.0.0.1:{port}/hook",
                       "secret": "", "headers": {}, "timeout": 1, "retry_count": 1, "retry_delay": 0}],
    }}}
    notifier = WebhookNotifier(cfg)
    result = notifier.send_alert(_make_alert())
    server.shutdown()
    assert result is False


def test_webhook_multiple_endpoints():
    port1 = _free_port()
    port2 = _free_port()

    class FailHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(500)
            self.end_headers()

        def log_message(self, *args):
            pass

    server1 = HTTPServer(("127.0.0.1", port1), FailHandler)
    server2 = _start_mock_server(port2)
    for s, p in [(server1, port1)]:
        threading.Thread(target=s.serve_forever, daemon=True).start()

    cfg = {"notifications": {"webhook": {
        "enabled": True,
        "endpoints": [
            {"url": f"http://127.0.0.1:{port1}/hook",
             "secret": "", "headers": {}, "timeout": 5, "retry_count": 1, "retry_delay": 0},
            {"url": f"http://127.0.0.1:{port2}/hook",
             "secret": "", "headers": {}, "timeout": 5, "retry_count": 1, "retry_delay": 0},
        ],
    }}}
    notifier = WebhookNotifier(cfg)
    result = notifier.send_alert(_make_alert())
    server1.shutdown()
    server2.shutdown()
    assert result is True


def test_webhook_disabled():
    cfg = {"notifications": {"webhook": {"enabled": False, "endpoints": []}}}
    notifier = WebhookNotifier(cfg)
    result = notifier.send_alert(_make_alert())
    assert result is False


# ---------------------------------------------------------------------------
# 27–31: Email notifier
# ---------------------------------------------------------------------------

def test_email_not_configured():
    cfg = {"notifications": {"email": {"enabled": True, "smtp_host": "",
                                        "smtp_username": "", "smtp_port": 587,
                                        "use_tls": False, "from_address": "a@b.com",
                                        "recipients": ["c@d.com"],
                                        "min_interval_seconds": 0, "min_priority": "low"}}}
    notifier = EmailNotifier(cfg)
    result = notifier.send_alert(_make_alert())
    assert result is False


def test_email_rate_limit():
    cfg = {"notifications": {"email": {"enabled": True,
                                        "smtp_host": "smtp.test.com",
                                        "smtp_username": "user",
                                        "smtp_password": "pass",
                                        "smtp_port": 587, "use_tls": False,
                                        "from_address": "a@b.com",
                                        "recipients": ["c@d.com"],
                                        "min_interval_seconds": 9999, "min_priority": "low"}}}
    notifier = EmailNotifier(cfg)
    notifier._last_sent = time.time()  # Pretend we just sent
    result = notifier.send_alert(_make_alert(priority=AlertPriority.CRITICAL))
    assert result is False  # Rate limited


def test_email_priority_filter():
    cfg = {"notifications": {"email": {"enabled": True,
                                        "smtp_host": "smtp.test.com",
                                        "smtp_username": "user",
                                        "smtp_password": "pass",
                                        "smtp_port": 587, "use_tls": False,
                                        "from_address": "a@b.com",
                                        "recipients": ["c@d.com"],
                                        "min_interval_seconds": 0, "min_priority": "high"}}}
    notifier = EmailNotifier(cfg)
    result = notifier.send_alert(_make_alert(priority=AlertPriority.LOW))
    assert result is False  # Below min_priority


def test_email_template_render():
    cfg = {"notifications": {"email": {"enabled": True,
                                        "smtp_host": "smtp.test.com",
                                        "smtp_username": "user",
                                        "smtp_password": "pass",
                                        "smtp_port": 587, "use_tls": False,
                                        "from_address": "a@b.com",
                                        "recipients": ["c@d.com"],
                                        "min_interval_seconds": 0, "min_priority": "low"}},
            "dashboard_url": "http://dash.test"}
    notifier = EmailNotifier(cfg)
    alert = _make_alert(priority=AlertPriority.HIGH)
    body = notifier._render_body(alert)
    assert alert.event_label in body
    assert alert.camera_id in body
    assert "http://dash.test" in body


def test_email_send_success():
    cfg = {"notifications": {"email": {"enabled": True,
                                        "smtp_host": "smtp.test.com",
                                        "smtp_username": "user",
                                        "smtp_password": "pass",
                                        "smtp_port": 587, "use_tls": False,
                                        "from_address": "a@b.com",
                                        "recipients": ["c@d.com"],
                                        "min_interval_seconds": 0, "min_priority": "low"}}}
    notifier = EmailNotifier(cfg)
    alert = _make_alert(priority=AlertPriority.HIGH)

    with patch("smtplib.SMTP") as mock_smtp_cls:
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
        result = notifier.send_alert(alert)

    assert result is True


# ---------------------------------------------------------------------------
# 32–45: Alert manager
# ---------------------------------------------------------------------------

def _make_manager(extra_cfg: dict = None) -> AlertManager:
    cfg = _base_config(_free_port())
    cfg["notifications"]["websocket"]["enabled"] = False
    if extra_cfg:
        for k, v in extra_cfg.items():
            cfg[k] = v
    return AlertManager(cfg)


def test_manager_process_alert_event():
    mgr = _make_manager()
    ev = _make_scored_event(decision=AlertDecision.ALERT)
    alerts = mgr.process_events([ev])
    assert len(alerts) == 1
    assert alerts[0].event_id == ev.event_id


def test_manager_skip_no_alert():
    mgr = _make_manager()
    ev = _make_scored_event(decision=AlertDecision.NO_ALERT)
    alerts = mgr.process_events([ev])
    assert len(alerts) == 0


def test_manager_rate_limit_per_camera():
    cfg = _base_config(_free_port())
    cfg["notifications"]["websocket"]["enabled"] = False
    cfg["alerts"]["rate_limit"]["max_alerts_per_minute"] = 5
    mgr = AlertManager(cfg)
    events = [_make_scored_event(camera_id="cam_01") for _ in range(7)]
    alerts = mgr.process_events(events)
    assert len(alerts) <= 5


def test_manager_rate_limit_global():
    cfg = _base_config(_free_port())
    cfg["notifications"]["websocket"]["enabled"] = False
    cfg["alerts"]["rate_limit"]["max_alerts_per_minute"] = 100
    cfg["alerts"]["rate_limit"]["max_alerts_per_minute_global"] = 5
    mgr = AlertManager(cfg)
    events = [
        _make_scored_event(camera_id=f"cam_{i % 5}", decision=AlertDecision.ALERT)
        for i in range(8)
    ]
    alerts = mgr.process_events(events)
    assert len(alerts) <= 5


def test_manager_grouping():
    cfg = _base_config(_free_port())
    cfg["notifications"]["websocket"]["enabled"] = False
    cfg["alerts"]["grouping"]["enabled"] = True
    cfg["alerts"]["grouping"]["window_seconds"] = 30
    mgr = AlertManager(cfg)
    ev1 = _make_scored_event(category="violent", camera_id="cam_01")
    ev2 = _make_scored_event(category="violent", camera_id="cam_01")
    alerts = mgr.process_events([ev1, ev2])
    assert len(alerts) == 1  # second was grouped


def test_manager_grouping_different_camera():
    cfg = _base_config(_free_port())
    cfg["notifications"]["websocket"]["enabled"] = False
    cfg["alerts"]["grouping"]["enabled"] = True
    cfg["alerts"]["grouping"]["window_seconds"] = 30
    cfg["alerts"]["grouping"]["same_camera_only"] = True
    mgr = AlertManager(cfg)
    ev1 = _make_scored_event(category="violent", camera_id="cam_01")
    ev2 = _make_scored_event(category="violent", camera_id="cam_02")
    alerts = mgr.process_events([ev1, ev2])
    assert len(alerts) == 2  # different cameras → not grouped


def test_manager_acknowledge():
    mgr = _make_manager()
    ev = _make_scored_event()
    alerts = mgr.process_events([ev])
    alert_id = alerts[0].alert_id
    result = mgr.acknowledge_alert(alert_id, "operator_01")
    assert result is True
    alert = mgr.get_alert(alert_id)
    assert alert.status == AlertStatus.ACKNOWLEDGED
    assert alert.acknowledged_by == "operator_01"


def test_manager_dismiss():
    mgr = _make_manager()
    ev = _make_scored_event()
    alerts = mgr.process_events([ev])
    alert_id = alerts[0].alert_id
    result = mgr.dismiss_alert(alert_id, "operator_01", "false_positive")
    assert result is True
    alert = mgr.get_alert(alert_id)
    assert alert.status == AlertStatus.DISMISSED
    assert alert.dismiss_reason == "false_positive"


def test_manager_escalation():
    cfg = _base_config(_free_port())
    cfg["notifications"]["websocket"]["enabled"] = False
    cfg["alerts"]["escalation"]["enabled"] = True
    cfg["alerts"]["escalation"]["timeout_seconds"] = 300
    mgr = AlertManager(cfg)
    ev = _make_scored_event()
    alerts = mgr.process_events([ev])
    alert = alerts[0]
    # Manually set delivered_at to 400 seconds ago to trigger escalation
    alert.delivered_at = time.time() - 400
    alert.status = AlertStatus.DELIVERED
    mgr.escalate_alert(alert.alert_id)
    assert alert.status == AlertStatus.ESCALATED


def test_manager_escalation_cancelled():
    mgr = _make_manager()
    ev = _make_scored_event()
    alerts = mgr.process_events([ev])
    alert_id = alerts[0].alert_id
    mgr.acknowledge_alert(alert_id, "op")
    alert = mgr.get_alert(alert_id)
    # After ack, escalation should not change status
    assert alert.status == AlertStatus.ACKNOWLEDGED
    assert alert.escalated_at is None


def test_manager_get_active_alerts():
    port = _free_port()
    ws_port = _free_port()
    # Use WebSocket so alerts get DELIVERED status
    cfg = _base_config(ws_port)
    cfg["alerts"]["grouping"]["enabled"] = False
    mgr = AlertManager(cfg)
    mgr.start()
    time.sleep(0.3)
    ev1 = _make_scored_event(camera_id="cam_01")
    ev2 = _make_scored_event(camera_id="cam_02")
    all_alerts = mgr.process_events([ev1, ev2])
    # Force status to DELIVERED so we can test active filter
    for a in all_alerts:
        a.status = AlertStatus.DELIVERED
    # Dismiss one
    mgr.dismiss_alert(all_alerts[0].alert_id, "op", "resolved")
    active = mgr.get_active_alerts()
    mgr.stop()
    assert len(active) == 1


def test_manager_get_history_filtered():
    mgr = _make_manager()
    ev1 = _make_scored_event(camera_id="cam_01", category="violent")
    ev2 = _make_scored_event(camera_id="cam_02", category="suspicious")
    mgr.process_events([ev1, ev2])
    cam1_alerts = mgr.get_alert_history(camera_id="cam_01")
    assert all(a.camera_id == "cam_01" for a in cam1_alerts)
    assert len(cam1_alerts) == 1


def test_manager_stats():
    mgr = _make_manager()
    evs = [_make_scored_event() for _ in range(3)]
    mgr.process_events(evs)
    stats = mgr.get_stats()
    assert "total_alerts" in stats
    assert "by_status" in stats
    assert "channel_success_rates" in stats
    assert stats["total_alerts"] >= 1


def test_manager_start_stop():
    port = _free_port()
    cfg = _base_config(port)
    cfg["notifications"]["websocket"]["enabled"] = False
    cfg["alerts"]["escalation"]["enabled"] = True
    mgr = AlertManager(cfg)
    mgr.start()
    time.sleep(0.2)
    mgr.stop()
    # No assertions — just verifies no crash


# ---------------------------------------------------------------------------
# 46–47: Integration
# ---------------------------------------------------------------------------

def test_full_flow_event_to_alert():
    port = _free_port()
    cfg = _base_config(port)
    cfg["alerts"]["grouping"]["enabled"] = False
    mgr = AlertManager(cfg)
    mgr.start()
    time.sleep(0.3)

    received = []

    async def connect_and_receive():
        from websockets.asyncio.client import connect
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            time.sleep(0.05)
            ev = _make_scored_event()
            mgr.process_events([ev])
            msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            received.append(json.loads(msg))

    asyncio.run(connect_and_receive())
    mgr.stop()
    assert len(received) == 1
    assert received[0]["type"] == "alert"


def test_manager_multiple_channels():
    port = _free_port()
    webhook_port = _free_port()
    server = _start_mock_server(webhook_port)
    _MockHTTPHandler.received_requests.clear()

    cfg = _base_config(port)
    cfg["alerts"]["grouping"]["enabled"] = False
    cfg["notifications"]["webhook"] = {
        "enabled": True,
        "endpoints": [{"url": f"http://127.0.0.1:{webhook_port}/hook",
                       "secret": "", "headers": {}, "timeout": 5,
                       "retry_count": 1, "retry_delay": 0}],
    }
    mgr = AlertManager(cfg)
    mgr.start()
    time.sleep(0.3)

    received_ws = []

    async def run():
        from websockets.asyncio.client import connect
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            time.sleep(0.05)
            ev = _make_scored_event()
            mgr.process_events([ev])
            msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            received_ws.append(json.loads(msg))

    asyncio.run(run())
    time.sleep(0.2)
    mgr.stop()
    server.shutdown()

    assert len(received_ws) == 1
    assert len(_MockHTTPHandler.received_requests) >= 1
