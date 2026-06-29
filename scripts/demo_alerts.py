"""
Demo: Alert service with WebSocket delivery.
Starts WebSocket server, generates sample alerts, and delivers them.
Connect a WebSocket client to ws://localhost:8765 to receive alerts.

Usage:
    python scripts/demo_alerts.py [--port 8765] [--simulate-events 10]
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts.alert_manager import AlertManager
from src.alerts.alert_models import AlertStatus
from src.common.config import load_config
from src.common.logger import get_logger
from src.scoring.scoring_models import AlertDecision, ScoredEvent, SignalType

logger = get_logger("demo.alerts")

_SCENARIOS = [
    {"category": "violent", "label": "fighting", "score": 0.78, "decision": AlertDecision.ALERT},
    {"category": "weapon", "label": "knife", "score": 1.00, "decision": AlertDecision.ESCALATED},
    {"category": "violent", "label": "fighting", "score": 0.65, "decision": AlertDecision.ALERT},
    {"category": "suspicious", "label": "loitering", "score": 0.58, "decision": AlertDecision.ALERT,
     "zone_name": "Restricted Zone A"},
    {"category": "urgent", "label": "falling", "score": 0.73, "decision": AlertDecision.ALERT},
    {"category": "violent", "label": "fighting", "score": 0.72, "decision": AlertDecision.ALERT},
    {"category": "weapon", "label": "gun", "score": 1.00, "decision": AlertDecision.ESCALATED},
    {"category": "suspicious", "label": "loitering", "score": 0.56, "decision": AlertDecision.ALERT},
    {"category": "violent", "label": "fighting", "score": 0.68, "decision": AlertDecision.ALERT},
    {"category": "urgent", "label": "falling", "score": 0.70, "decision": AlertDecision.ALERT},
]


def _make_event(scenario: dict, camera_id: str) -> ScoredEvent:
    return ScoredEvent(
        event_id=f"evt_{uuid.uuid4().hex[:8]}",
        camera_id=camera_id,
        track_id=1,
        timestamp=time.time(),
        severity_score=scenario["score"],
        contributing_signals=[],
        dominant_signal=SignalType.ACTION_CLASSIFICATION,
        event_category=scenario["category"],
        event_label=scenario["label"],
        alert_decision=scenario["decision"],
        zone_name=scenario.get("zone_name"),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSS Alert Demo")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--simulate-events", type=int, default=10)
    return p.parse_args()


def _symbol(decision: AlertDecision) -> str:
    return "⚠️ " if decision == AlertDecision.ESCALATED else ""


def main() -> None:
    args = _parse_args()
    config = load_config()
    config["notifications"]["websocket"]["port"] = args.port
    config["notifications"]["websocket"]["ping_interval"] = None
    config["notifications"]["websocket"]["ping_timeout"] = None
    config["notifications"]["webhook"]["enabled"] = False
    config["notifications"]["email"]["enabled"] = False
    config["alerts"]["grouping"]["enabled"] = True
    config["alerts"]["grouping"]["window_seconds"] = 5
    config["alerts"]["escalation"]["enabled"] = False

    mgr = AlertManager(config)
    mgr.start()
    time.sleep(0.5)

    print(f"\nWebSocket server started on ws://localhost:{args.port}")
    print("Connect a client to receive alerts (e.g.: websocat ws://localhost:" + str(args.port) + ")")
    print(f"\nGenerating {args.simulate_events} simulated events...\n")

    cameras = ["cam_01", "cam_02", "cam_03"]
    n = min(args.simulate_events, len(_SCENARIOS))
    total_created = 0
    total_suppressed = 0
    total_grouped = 0

    for i in range(n):
        scenario = _SCENARIOS[i % len(_SCENARIOS)]
        cam = cameras[i % len(cameras)]
        ev = _make_event(scenario, cam)

        before = len(mgr._alerts)
        alerts = mgr.process_events([ev])
        after = len(mgr._alerts)

        sym = _symbol(scenario["decision"])
        if not alerts:
            total_suppressed += 1
            print(f"  [{i+1}/{n}] Suppressed: rate limit or grouped ({cam})")
        else:
            alert = alerts[0]
            total_created += 1
            ws_clients = len(mgr._ws_notifier.get_connected_clients())
            ws_ok = "✓" if ws_clients > 0 else "✓ (0 clients)"
            print(
                f"  [{i+1}/{n}] {sym}Alert: {alert.title}\n"
                f"    Priority: {alert.priority.value.upper()} | Score: {scenario['score']:.2f}\n"
                f"    → WebSocket: {ws_ok} | Webhook: disabled | Email: disabled"
            )

        time.sleep(0.3)

    print(f"\n{'═'*52}")
    print("  === Alert Summary ===")
    print(f"  Created:    {total_created}")
    print(f"  Suppressed: {total_suppressed}")
    stats = mgr.get_stats()
    for status, count in stats.get("by_status", {}).items():
        print(f"  {status.capitalize():<14}: {count}")
    print(f"{'═'*52}")

    print("\nServer running for 30 seconds — connect a client now.")
    print("Press Ctrl+C to stop early.\n")
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()
        print("\nDemo stopped.")


if __name__ == "__main__":
    main()
