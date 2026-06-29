"""
Demo: Start the FastAPI server and exercise all API endpoints.

Usage:
    python scripts/demo_api.py [--port 8000] [--db-url sqlite:///data/sss_dev.db]
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSS API Demo")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--db-url", default="sqlite:///data/sss_dev.db")
    return p.parse_args()


def _demo_requests(base_url: str, token: str) -> None:
    """Run a sequence of demo API calls and print results."""
    import urllib.request
    import json

    def get(path: str) -> dict:
        req = urllib.request.Request(
            f"{base_url}{path}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def post(path: str, data: dict, extra_headers: dict = None) -> dict:
        body = json.dumps(data).encode()
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {token}"}
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(f"{base_url}{path}", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    print("\n" + "═" * 60)
    print("  Smart Surveillance System — API Demo")
    print("═" * 60)

    # System health
    health = get("/api/v1/system/health")
    print(f"\n[Health]   status={health['status']}")

    # Cameras
    cams = get("/api/v1/cameras")
    print(f"[Cameras]  count={len(cams)}")
    for c in cams[:3]:
        print(f"           {c['id']:10s}  {c['name']}")

    # Events
    evs = get("/api/v1/events?limit=5")
    total = evs.get("total", 0)
    print(f"[Events]   total={total} (showing latest 5)")
    for e in evs.get("events", [])[:5]:
        print(f"           [{e['event_category']:12s}] {e['event_label']:15s}  "
              f"score={e['severity_score']:.2f}  cam={e['camera_id']}")

    # Alerts
    alerts = get("/api/v1/alerts?limit=5")
    atotal = alerts.get("total", 0)
    print(f"[Alerts]   total={atotal} (showing latest 5)")
    for a in alerts.get("alerts", [])[:5]:
        print(f"           [{a['priority']:8s}] {a['status']:12s}  {a['title'][:50]}")

    # Stats
    estats = get("/api/v1/events/stats")
    print(f"[Stats]    events total={estats['total']}")
    for cat, cnt in estats.get("by_category", {}).items():
        print(f"           {cat:15s}: {cnt}")

    astats = get("/api/v1/alerts/stats")
    print(f"[Stats]    alerts by_status={astats.get('by_status', {})}")

    print("\n" + "═" * 60)
    print("  Demo complete. Server still running.")
    print("  API docs at: " + base_url + "/docs")
    print("═" * 60 + "\n")


def main() -> None:
    args = _parse_args()

    from src.common.config import load_config
    from src.api.main import create_app
    from scripts.seed_database import seed

    # Seed the database first
    print(f"\nSeeding database ({args.db_url})...")
    seed(args.db_url)

    # Override DB URL in config
    cfg = load_config()
    cfg["database"]["fallback_url"] = args.db_url

    app = create_app(cfg)

    import uvicorn

    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning"),
        daemon=True,
    )
    server_thread.start()
    time.sleep(1.5)  # Wait for server to start

    base_url = f"http://127.0.0.1:{args.port}"
    print(f"\nServer running at {base_url}")
    print(f"API docs:  {base_url}/docs")
    print(f"OpenAPI:   {base_url}/openapi.json")

    # Obtain token
    import urllib.request, json as _json
    login_data = _json.dumps({"username": "admin", "password": "admin"}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/v1/auth/login",
        data=login_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        token = _json.loads(r.read())["access_token"]

    _demo_requests(base_url, token)

    print("Press Ctrl+C to stop the server.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
