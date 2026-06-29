"""
WebSocket notification channel.
Pushes real-time alerts to connected dashboard clients.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Dict, List, Optional, Set

from src.alerts.alert_models import Alert
from src.common.logger import get_logger

logger = get_logger("alerts.websocket_notifier")

try:
    import websockets
    from websockets.asyncio.server import ServerConnection, serve
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False
    logger.warning("websockets package not available — WebSocket notifier disabled.")


class WebSocketNotifier:
    """Broadcasts alerts to all connected WebSocket clients.

    Args:
        config: Full project config dict; reads ``notifications.websocket``.
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("notifications", {}).get("websocket", {})
        self._enabled: bool = bool(cfg.get("enabled", True))
        self._host: str = cfg.get("host", "0.0.0.0")
        self._port: int = int(cfg.get("port", 8765))
        self._ping_interval: int = int(cfg.get("ping_interval", 30))
        self._ping_timeout: int = int(cfg.get("ping_timeout", 10))

        self._clients: Dict[str, object] = {}  # client_id → websocket
        self._lock = threading.Lock()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(
            "WebSocketNotifier ready (enabled=%s host=%s port=%d).",
            self._enabled, self._host, self._port,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the WebSocket server in a background daemon thread."""
        if not self._enabled or not _WS_AVAILABLE:
            logger.info("WebSocket notifier not started (disabled or unavailable).")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="ws-notifier"
        )
        self._thread.start()
        # Give the event loop time to bind the port
        time.sleep(0.2)
        logger.info("WebSocket server started on ws://%s:%d.", self._host, self._port)

    def stop(self) -> None:
        """Stop the WebSocket server and disconnect all clients."""
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("WebSocket notifier stopped.")

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def send_alert(self, alert: Alert) -> bool:
        """Broadcast an alert to all connected clients.

        Args:
            alert: :class:`~src.alerts.alert_models.Alert` to broadcast.

        Returns:
            ``True`` if at least one client received the message.
        """
        message = json.dumps({
            "type": "alert",
            "data": alert.to_dict(),
            "timestamp": time.time(),
        })
        return self._broadcast(message)

    def send_system_status(self, status: dict) -> bool:
        """Send a system health/status update to all clients.

        Args:
            status: Arbitrary status dict.

        Returns:
            ``True`` if at least one client received the message.
        """
        message = json.dumps({
            "type": "system_status",
            "data": status,
            "timestamp": time.time(),
        })
        return self._broadcast(message)

    def send_to_client(self, client_id: str, data: dict) -> bool:
        """Send a message to a specific connected client.

        Args:
            client_id: Target client identifier.
            data: Dict to serialize and send.

        Returns:
            ``True`` on success.
        """
        with self._lock:
            ws = self._clients.get(client_id)
        if ws is None or self._loop is None:
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._send_single(client_id, ws, json.dumps(data)), self._loop
        )
        try:
            return future.result(timeout=3.0)
        except Exception:
            return False

    def get_connected_clients(self) -> List[str]:
        """Return list of currently connected client IDs."""
        with self._lock:
            return list(self._clients.keys())

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _broadcast(self, message: str) -> bool:
        if self._loop is None or not self._clients:
            return False
        with self._lock:
            items = list(self._clients.items())
        if not items:
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._do_broadcast(message, items), self._loop
        )
        try:
            return future.result(timeout=5.0)
        except Exception as exc:
            logger.error("Broadcast error: %s", exc)
            return False

    async def _do_broadcast(self, message: str, items: list) -> bool:
        success = False
        dead = []
        for client_id, ws in items:
            try:
                await ws.send(message)
                success = True
            except Exception:
                dead.append(client_id)
        with self._lock:
            for cid in dead:
                self._clients.pop(cid, None)
        return success

    async def _send_single(self, client_id: str, ws, message: str) -> bool:
        try:
            await ws.send(message)
            return True
        except Exception:
            with self._lock:
                self._clients.pop(client_id, None)
            return False

    async def _handler(self, websocket) -> None:
        client_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._clients[client_id] = websocket
        logger.info("WS client connected: %s (total=%d)", client_id, len(self._clients))
        try:
            async for _ in websocket:
                pass  # ignore incoming messages for now
        except Exception:
            pass
        finally:
            with self._lock:
                self._clients.pop(client_id, None)
            logger.info("WS client disconnected: %s", client_id)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        try:
            async with serve(
                self._handler,
                self._host,
                self._port,
                ping_interval=self._ping_interval,
                ping_timeout=self._ping_timeout,
            ) as server:
                self._server = server
                await server.serve_forever()
        except Exception as exc:
            logger.error("WebSocket server error: %s", exc)
