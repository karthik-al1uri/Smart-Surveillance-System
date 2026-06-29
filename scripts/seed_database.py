"""
Seed the database with sample cameras, events, alerts, and users.

Usage:
    python scripts/seed_database.py [--db-url sqlite:///data/sss_dev.db]
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import choice, randint, uniform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.repositories import (
    AlertRepository,
    CameraRepository,
    EventRepository,
    UserRepository,
)
from src.common.db import init_db
from src.common.logger import get_logger

logger = get_logger("scripts.seed_database")

_CAMERAS = [
    {"id": "cam_01", "name": "Front Entrance",   "stream_url": "rtsp://192.168.1.101/stream",
     "location": "Building A — Front", "indoor": False},
    {"id": "cam_02", "name": "Rear Exit",        "stream_url": "rtsp://192.168.1.102/stream",
     "location": "Building A — Rear", "indoor": False},
    {"id": "cam_03", "name": "Lobby",            "stream_url": "rtsp://192.168.1.103/stream",
     "location": "Building A — Ground Floor", "indoor": True},
    {"id": "cam_04", "name": "Car Park North",   "stream_url": "rtsp://192.168.1.104/stream",
     "location": "Car Park — North", "indoor": False},
    {"id": "cam_05", "name": "Server Room",      "stream_url": "rtsp://192.168.1.105/stream",
     "location": "Floor 3 — Server Room", "indoor": True},
]

_EVENTS = [
    ("violent", "fighting",   0.82, "alert"),
    ("weapon",  "knife",      0.99, "alert"),
    ("suspicious", "loitering", 0.57, "alert"),
    ("urgent",  "falling",    0.71, "alert"),
    ("normal",  "walking",    0.15, "no_alert"),
    ("violent", "fighting",   0.90, "alert"),
    ("suspicious", "tailgating", 0.63, "alert"),
    ("normal",  "standing",   0.08, "no_alert"),
    ("weapon",  "gun",        1.00, "alert"),
    ("urgent",  "falling",    0.68, "alert"),
]

_USERS = [
    {"username": "operator1", "password": "operator1pass", "role": "operator",
     "full_name": "Alice Tan"},
    {"username": "operator2", "password": "operator2pass", "role": "operator",
     "full_name": "Bob Singh"},
    {"username": "viewer1",   "password": "viewer1pass",   "role": "viewer",
     "full_name": "Carol Lee"},
]


def seed(db_url: str) -> None:
    engine = create_engine(db_url, connect_args={"check_same_thread": False}
                           if "sqlite" in db_url else {})
    init_db(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()

    # ---- Users ----
    user_repo = UserRepository(session)
    if user_repo.count() == 0:
        user_repo.create_user("admin", "admin", "admin", "System Administrator")
        print("  Created admin user.")
    for u in _USERS:
        if not user_repo.get_user(u["username"]):
            user_repo.create_user(u["username"], u["password"], u["role"], u["full_name"])
    session.commit()
    print(f"  Users: {user_repo.count()} total")

    # ---- Cameras ----
    cam_repo = CameraRepository(session)
    for c in _CAMERAS:
        if not cam_repo.get_camera(c["id"]):
            cam_repo.create_camera(c)
    session.commit()
    print(f"  Cameras: {len(cam_repo.list_cameras())} total")

    # ---- Events ----
    ev_repo = EventRepository(session)
    _, existing = ev_repo.list_events(limit=1)
    if existing == 0:
        base_time = datetime.now(timezone.utc) - timedelta(hours=6)
        for i, (cat, label, score, decision) in enumerate(_EVENTS * 5):
            cam = choice(_CAMERAS)
            ev_repo.create_event({
                "id": str(uuid.uuid4()),
                "camera_id": cam["id"],
                "timestamp": base_time + timedelta(minutes=i * 7),
                "event_category": cat,
                "event_label": label,
                "severity_score": score,
                "alert_decision": decision,
                "track_id": randint(1, 20),
                "contributing_signals": [{"signal_type": "action_classification",
                                          "value": score, "weight": 1.0}],
                "dominant_signal": "action_classification",
            })
        session.commit()
        _, total = ev_repo.list_events(limit=1)
        print(f"  Events: {total} created")
    else:
        print(f"  Events: {existing} already exist, skipping.")

    # ---- Alerts ----
    alert_repo = AlertRepository(session)
    events_with_alert, _ = ev_repo.list_events(category="violent", limit=5)
    for ev in events_with_alert:
        alert_repo.create_alert({
            "event_id": ev.id,
            "priority": choice(["high", "critical"]),
            "title": f"Alert — {ev.event_label} on {ev.camera_id}",
            "description": f"Detected {ev.event_label} with score {ev.severity_score:.2f}.",
            "status": choice(["pending", "delivered", "acknowledged"]),
        })
    session.commit()
    stats = alert_repo.get_alert_stats()
    print(f"  Alerts: {stats['total']} total")

    session.close()
    print("\nDatabase seeded successfully.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed SSS database")
    p.add_argument("--db-url", default="sqlite:///data/sss_dev.db")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"\nSeeding database: {args.db_url}\n")
    seed(args.db_url)
