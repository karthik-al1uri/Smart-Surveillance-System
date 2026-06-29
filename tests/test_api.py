"""Tests for Phase 10: Event Database & API Layer.

All tests use SQLite in-memory — no PostgreSQL or Docker required.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.auth import create_access_token, hash_password, verify_password, verify_token
from src.api.dependencies import get_db, set_session_factory
from src.api.main import create_app
from src.api.repositories import (
    AlertRepository,
    CameraRepository,
    EventRepository,
    UserRepository,
)
from src.common.db import get_session_factory, init_db
from src.common.db_models import AlertRecord, Base, Camera, Event, User

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TEST_CONFIG = {
    "database": {"fallback_url": "sqlite:///:memory:", "auto_fallback": True, "echo_sql": False},
    "auth": {
        "secret_key": "test-secret",
        "algorithm": "HS256",
        "access_token_expire_minutes": 60,
        "default_admin": {"username": "admin", "password": "admin", "role": "admin"},
    },
    "notifications": {
        "websocket": {"enabled": False},
        "webhook": {"enabled": False, "endpoints": []},
        "email": {"enabled": False, "smtp_host": "", "smtp_username": ""},
    },
    "alerts": {
        "rate_limit": {"max_alerts_per_minute": 10, "max_alerts_per_minute_global": 30},
        "grouping": {"enabled": False, "window_seconds": 10, "same_camera_only": True},
        "escalation": {"enabled": False, "timeout_seconds": 300, "escalation_channels": []},
        "history_size": 100,
    },
    "scoring": {"escalation_threshold": 0.85},
    "dashboard_url": "http://localhost:3000",
}


@pytest.fixture(scope="function")
def db_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    factory = sessionmaker(bind=db_engine)
    session = factory()
    yield session
    session.close()


@pytest.fixture(scope="function")
def client(db_engine):
    app = create_app(_TEST_CONFIG, engine=db_engine)

    with TestClient(app) as c:
        yield c


def get_session(factory):
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return _ctx()


def _admin_token(client: TestClient) -> str:
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin"})
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _make_camera(repo: CameraRepository, cam_id: str = "cam_01") -> Camera:
    return repo.create_camera({
        "id": cam_id, "name": f"Camera {cam_id}",
        "stream_url": f"rtsp://test/{cam_id}", "enabled": True,
    })


def _make_event(repo: EventRepository, camera_id: str = "cam_01") -> Event:
    return repo.create_event({
        "id": str(uuid.uuid4()),
        "camera_id": camera_id,
        "timestamp": datetime.now(timezone.utc),
        "event_category": "violent",
        "event_label": "fighting",
        "severity_score": 0.75,
        "alert_decision": "alert",
    })


# ---------------------------------------------------------------------------
# 1–6: Database models
# ---------------------------------------------------------------------------

def test_create_camera(db_session):
    repo = CameraRepository(db_session)
    cam = _make_camera(repo)
    db_session.commit()
    assert cam.id == "cam_01"
    assert cam.name == "Camera cam_01"


def test_create_event(db_session):
    repo_cam = CameraRepository(db_session)
    repo_ev = EventRepository(db_session)
    _make_camera(repo_cam)
    ev = _make_event(repo_ev)
    db_session.commit()
    assert ev.event_category == "violent"
    assert ev.severity_score == 0.75


def test_event_camera_relationship(db_session):
    CameraRepository(db_session).create_camera({
        "id": "cam_r", "name": "R Cam", "stream_url": "rtsp://r", "enabled": True,
    })
    ev = EventRepository(db_session).create_event({
        "id": str(uuid.uuid4()), "camera_id": "cam_r",
        "timestamp": datetime.now(timezone.utc),
        "event_category": "violent", "event_label": "fighting", "severity_score": 0.5,
    })
    db_session.commit()
    assert ev.camera_id == "cam_r"


def test_create_alert_record(db_session):
    CameraRepository(db_session).create_camera({
        "id": "cam_a", "name": "A", "stream_url": "rtsp://a", "enabled": True,
    })
    ev = EventRepository(db_session).create_event({
        "id": "ev-alert", "camera_id": "cam_a",
        "timestamp": datetime.now(timezone.utc),
        "event_category": "violent", "event_label": "fighting", "severity_score": 0.8,
    })
    rec = AlertRepository(db_session).create_alert({
        "event_id": ev.id, "priority": "high", "title": "Test alert",
    })
    db_session.commit()
    assert rec.event_id == ev.id
    assert rec.priority == "high"


def test_create_user(db_session):
    user = UserRepository(db_session).create_user("testuser", "password123", "operator")
    db_session.commit()
    assert user.username == "testuser"
    assert user.hashed_password != "password123"


def test_user_unique_username(db_session):
    repo = UserRepository(db_session)
    repo.create_user("dupuser", "pass1", "operator")
    db_session.commit()
    with pytest.raises(IntegrityError):
        repo.create_user("dupuser", "pass2", "operator")


# ---------------------------------------------------------------------------
# 7–18: Repositories
# ---------------------------------------------------------------------------

def test_event_repo_create_and_get(db_session):
    CameraRepository(db_session).create_camera({
        "id": "cam_01", "name": "C1", "stream_url": "rtsp://x", "enabled": True,
    })
    ev = _make_event(EventRepository(db_session))
    db_session.commit()
    fetched = EventRepository(db_session).get_event(ev.id)
    assert fetched.id == ev.id


def test_event_repo_list_paginated(db_session):
    CameraRepository(db_session).create_camera({
        "id": "cam_01", "name": "C1", "stream_url": "rtsp://x", "enabled": True,
    })
    repo = EventRepository(db_session)
    for _ in range(20):
        _make_event(repo)
    db_session.commit()
    events, total = repo.list_events(limit=10, offset=0)
    assert len(events) == 10
    assert total == 20


def test_event_repo_filter_by_camera(db_session):
    for cid in ["cam_01", "cam_02"]:
        CameraRepository(db_session).create_camera({
            "id": cid, "name": cid, "stream_url": f"rtsp://{cid}", "enabled": True,
        })
    repo = EventRepository(db_session)
    for _ in range(3):
        _make_event(repo, "cam_01")
    for _ in range(2):
        _make_event(repo, "cam_02")
    db_session.commit()
    evs, total = repo.list_events(camera_id="cam_01")
    assert total == 3
    assert all(e.camera_id == "cam_01" for e in evs)


def test_event_repo_filter_by_category(db_session):
    CameraRepository(db_session).create_camera({
        "id": "cam_01", "name": "C", "stream_url": "rtsp://x", "enabled": True,
    })
    repo = EventRepository(db_session)
    repo.create_event({
        "id": str(uuid.uuid4()), "camera_id": "cam_01",
        "timestamp": datetime.now(timezone.utc),
        "event_category": "weapon", "event_label": "knife", "severity_score": 1.0,
    })
    _make_event(repo)  # violent
    db_session.commit()
    evs, total = repo.list_events(category="weapon")
    assert total == 1
    assert evs[0].event_category == "weapon"


def test_event_repo_filter_by_time(db_session):
    CameraRepository(db_session).create_camera({
        "id": "cam_01", "name": "C", "stream_url": "rtsp://x", "enabled": True,
    })
    repo = EventRepository(db_session)
    t1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
    repo.create_event({
        "id": str(uuid.uuid4()), "camera_id": "cam_01",
        "timestamp": t1, "event_category": "violent",
        "event_label": "fighting", "severity_score": 0.5,
    })
    repo.create_event({
        "id": str(uuid.uuid4()), "camera_id": "cam_01",
        "timestamp": t2, "event_category": "violent",
        "event_label": "fighting", "severity_score": 0.5,
    })
    db_session.commit()
    evs, total = repo.list_events(start_time=t2)
    assert total == 1


def test_event_repo_stats(db_session):
    CameraRepository(db_session).create_camera({
        "id": "cam_01", "name": "C", "stream_url": "rtsp://x", "enabled": True,
    })
    repo = EventRepository(db_session)
    for _ in range(3):
        _make_event(repo)
    db_session.commit()
    stats = repo.get_event_stats()
    assert stats["total"] == 3
    assert "violent" in stats["by_category"]


def test_event_repo_update_feedback(db_session):
    CameraRepository(db_session).create_camera({
        "id": "cam_01", "name": "C", "stream_url": "rtsp://x", "enabled": True,
    })
    repo = EventRepository(db_session)
    ev = _make_event(repo)
    db_session.commit()
    updated = repo.update_feedback(ev.id, is_correct=False,
                                   corrected_label="loitering", operator="op1")
    db_session.commit()
    assert updated.feedback_correct is False
    assert updated.feedback_label == "loitering"


def test_camera_repo_crud(db_session):
    repo = CameraRepository(db_session)
    cam = _make_camera(repo)
    db_session.commit()
    assert repo.get_camera("cam_01") is not None
    repo.update_camera("cam_01", {"name": "Updated Name"})
    db_session.commit()
    assert repo.get_camera("cam_01").name == "Updated Name"
    repo.delete_camera("cam_01")
    db_session.commit()
    assert repo.get_camera("cam_01") is None


def test_camera_repo_update_zones(db_session):
    repo = CameraRepository(db_session)
    _make_camera(repo)
    db_session.commit()
    zones = [{"zone_id": "z1", "name": "Entry", "zone_type": "restricted",
               "polygon": [[0, 0], [1, 0], [1, 1]]}]
    repo.update_zones("cam_01", zones)
    db_session.commit()
    cam = repo.get_camera("cam_01")
    assert len(cam.zones_config) == 1


def test_alert_repo_create_and_list(db_session):
    CameraRepository(db_session).create_camera({
        "id": "cam_01", "name": "C", "stream_url": "rtsp://x", "enabled": True,
    })
    ev = _make_event(EventRepository(db_session))
    db_session.commit()
    repo = AlertRepository(db_session)
    repo.create_alert({"event_id": ev.id, "priority": "high", "title": "T1"})
    repo.create_alert({"event_id": ev.id, "priority": "medium", "title": "T2"})
    db_session.commit()
    records, total = repo.list_alerts()
    assert total == 2


def test_alert_repo_update_status(db_session):
    CameraRepository(db_session).create_camera({
        "id": "cam_01", "name": "C", "stream_url": "rtsp://x", "enabled": True,
    })
    ev = _make_event(EventRepository(db_session))
    db_session.commit()
    repo = AlertRepository(db_session)
    rec = repo.create_alert({"event_id": ev.id, "priority": "high"})
    db_session.commit()
    repo.update_status(rec.id, "acknowledged", operator="admin")
    db_session.commit()
    updated = repo.get_alert(rec.id)
    assert updated.status == "acknowledged"
    assert updated.acknowledged_by == "admin"


def test_user_repo_authenticate(db_session):
    repo = UserRepository(db_session)
    repo.create_user("authuser", "secret123", "operator")
    db_session.commit()
    assert repo.authenticate("authuser", "secret123") is not None
    assert repo.authenticate("authuser", "wrongpass") is None


# ---------------------------------------------------------------------------
# 19–23: Authentication
# ---------------------------------------------------------------------------

def test_create_access_token():
    from src.api.auth import configure_auth
    configure_auth({"secret_key": "test-secret", "algorithm": "HS256",
                    "access_token_expire_minutes": 60})
    token = create_access_token({"sub": "user1", "role": "operator"})
    assert isinstance(token, str)
    assert len(token) > 20


def test_verify_valid_token():
    from src.api.auth import configure_auth
    configure_auth({"secret_key": "test-secret", "algorithm": "HS256",
                    "access_token_expire_minutes": 60})
    token = create_access_token({"sub": "user1", "role": "admin"})
    payload = verify_token(token)
    assert payload["sub"] == "user1"
    assert payload["role"] == "admin"


def test_verify_expired_token():
    from datetime import timedelta
    from jose import JWTError
    from src.api.auth import configure_auth
    configure_auth({"secret_key": "test-secret", "algorithm": "HS256",
                    "access_token_expire_minutes": 60})
    token = create_access_token({"sub": "user1"}, expires_delta=timedelta(seconds=-1))
    with pytest.raises(JWTError):
        verify_token(token)


def test_verify_invalid_token():
    from jose import JWTError
    from src.api.auth import configure_auth
    configure_auth({"secret_key": "test-secret", "algorithm": "HS256",
                    "access_token_expire_minutes": 60})
    with pytest.raises(JWTError):
        verify_token("not.a.valid.token")


def test_password_hash():
    hashed = hash_password("mypassword")
    assert hashed != "mypassword"
    assert verify_password("mypassword", hashed)
    assert not verify_password("wrongpassword", hashed)


# ---------------------------------------------------------------------------
# 24–47: API endpoints (TestClient)
# ---------------------------------------------------------------------------

def test_login_success(client):
    resp = client.post("/api/v1/auth/login",
                       json={"username": "admin", "password": "admin"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_wrong_password(client):
    resp = client.post("/api/v1/auth/login",
                       json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401


def test_cameras_list(client):
    token = _admin_token(client)
    resp = client.get("/api/v1/cameras", headers=_auth(token))
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_cameras_create(client):
    token = _admin_token(client)
    resp = client.post("/api/v1/cameras", headers=_auth(token),
                       json={"id": "cam_01", "name": "Front", "stream_url": "rtsp://x"})
    assert resp.status_code == 201
    assert resp.json()["id"] == "cam_01"


def test_cameras_get(client):
    token = _admin_token(client)
    client.post("/api/v1/cameras", headers=_auth(token),
                json={"id": "cam_02", "name": "Back", "stream_url": "rtsp://y"})
    resp = client.get("/api/v1/cameras/cam_02", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["name"] == "Back"


def test_cameras_update(client):
    token = _admin_token(client)
    client.post("/api/v1/cameras", headers=_auth(token),
                json={"id": "cam_03", "name": "Old", "stream_url": "rtsp://z"})
    resp = client.put("/api/v1/cameras/cam_03", headers=_auth(token),
                      json={"name": "New Name"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New Name"


def test_cameras_delete(client):
    token = _admin_token(client)
    client.post("/api/v1/cameras", headers=_auth(token),
                json={"id": "cam_del", "name": "Del", "stream_url": "rtsp://del"})
    resp = client.delete("/api/v1/cameras/cam_del", headers=_auth(token))
    assert resp.status_code == 200
    assert client.get("/api/v1/cameras/cam_del",
                      headers=_auth(token)).status_code == 404


def test_cameras_zones_update(client):
    token = _admin_token(client)
    client.post("/api/v1/cameras", headers=_auth(token),
                json={"id": "cam_z", "name": "Z", "stream_url": "rtsp://z"})
    resp = client.put("/api/v1/cameras/cam_z/zones", headers=_auth(token),
                      json={"zones": [{"zone_id": "z1", "name": "Entry",
                                       "zone_type": "restricted",
                                       "polygon": [[0, 0], [1, 0], [1, 1]]}]})
    assert resp.status_code == 200


def test_events_list(client):
    token = _admin_token(client)
    resp = client.get("/api/v1/events", headers=_auth(token))
    assert resp.status_code == 200
    assert "events" in resp.json()


def test_events_list_filtered(client):
    token = _admin_token(client)
    resp = client.get("/api/v1/events?camera_id=cam_01&category=violent",
                      headers=_auth(token))
    assert resp.status_code == 200


def test_events_detail(client, db_engine):
    token = _admin_token(client)
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    with get_session(factory) as session:
        CameraRepository(session).create_camera({
            "id": "cam_ev", "name": "E", "stream_url": "rtsp://e", "enabled": True,
        })
        ev = _make_event(EventRepository(session), "cam_ev")
    resp = client.get(f"/api/v1/events/{ev.id}", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["id"] == ev.id


def test_events_feedback(client, db_engine):
    token = _admin_token(client)
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    with get_session(factory) as session:
        CameraRepository(session).create_camera({
            "id": "cam_fb", "name": "FB", "stream_url": "rtsp://fb", "enabled": True,
        })
        ev = _make_event(EventRepository(session), "cam_fb")
    resp = client.post(f"/api/v1/events/{ev.id}/feedback",
                       headers=_auth(token),
                       json={"is_correct": False, "corrected_label": "loitering"})
    assert resp.status_code == 200


def test_events_stats(client):
    token = _admin_token(client)
    resp = client.get("/api/v1/events/stats", headers=_auth(token))
    assert resp.status_code == 200
    assert "total" in resp.json()


def test_alerts_list(client):
    token = _admin_token(client)
    resp = client.get("/api/v1/alerts", headers=_auth(token))
    assert resp.status_code == 200
    assert "alerts" in resp.json()


def test_alerts_active(client, db_engine):
    token = _admin_token(client)
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    with get_session(factory) as session:
        CameraRepository(session).create_camera({
            "id": "cam_al", "name": "AL", "stream_url": "rtsp://al", "enabled": True,
        })
        ev = _make_event(EventRepository(session), "cam_al")
        AlertRepository(session).create_alert({
            "event_id": ev.id, "priority": "high", "status": "delivered",
        })
    resp = client.get("/api/v1/alerts/active", headers=_auth(token))
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_alerts_acknowledge(client, db_engine):
    token = _admin_token(client)
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    with get_session(factory) as session:
        CameraRepository(session).create_camera({
            "id": "cam_ack", "name": "ACK", "stream_url": "rtsp://ack", "enabled": True,
        })
        ev = _make_event(EventRepository(session), "cam_ack")
        rec = AlertRepository(session).create_alert({
            "event_id": ev.id, "priority": "high",
        })
    resp = client.post(f"/api/v1/alerts/{rec.id}/acknowledge",
                       headers=_auth(token), json={"operator": "admin"})
    assert resp.status_code == 200


def test_alerts_dismiss(client, db_engine):
    token = _admin_token(client)
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    with get_session(factory) as session:
        CameraRepository(session).create_camera({
            "id": "cam_dis", "name": "DIS", "stream_url": "rtsp://dis", "enabled": True,
        })
        ev = _make_event(EventRepository(session), "cam_dis")
        rec = AlertRepository(session).create_alert({
            "event_id": ev.id, "priority": "medium",
        })
    resp = client.post(f"/api/v1/alerts/{rec.id}/dismiss",
                       headers=_auth(token),
                       json={"operator": "admin", "reason": "false_positive"})
    assert resp.status_code == 200


def test_clips_list(client):
    token = _admin_token(client)
    resp = client.get("/api/v1/clips", headers=_auth(token))
    assert resp.status_code == 200


def test_clips_storage_stats(client):
    token = _admin_token(client)
    resp = client.get("/api/v1/clips/storage", headers=_auth(token))
    assert resp.status_code == 200
    assert "total_clips" in resp.json()


def test_system_health(client):
    token = _admin_token(client)
    resp = client.get("/api/v1/system/health", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_users_crud(client):
    token = _admin_token(client)
    # Create
    resp = client.post("/api/v1/users", headers=_auth(token),
                       json={"username": "op1", "password": "pass123", "role": "operator"})
    assert resp.status_code == 201
    user_id = resp.json()["id"]
    # List
    resp = client.get("/api/v1/users", headers=_auth(token))
    assert resp.status_code == 200
    assert any(u["username"] == "op1" for u in resp.json())
    # Update
    resp = client.put(f"/api/v1/users/{user_id}", headers=_auth(token),
                      json={"full_name": "Operator One"})
    assert resp.status_code == 200
    # Delete
    resp = client.delete(f"/api/v1/users/{user_id}", headers=_auth(token))
    assert resp.status_code == 200


def test_auth_required(client):
    resp = client.get("/api/v1/cameras")
    assert resp.status_code == 401


def test_admin_required(client):
    token = _admin_token(client)
    # Create a non-admin user
    client.post("/api/v1/users", headers=_auth(token),
                json={"username": "viewer1", "password": "view123", "role": "viewer"})
    viewer_resp = client.post("/api/v1/auth/login",
                              json={"username": "viewer1", "password": "view123"})
    viewer_token = viewer_resp.json()["access_token"]
    # Try admin-only route
    resp = client.post("/api/v1/cameras", headers=_auth(viewer_token),
                       json={"id": "x", "name": "X", "stream_url": "rtsp://x"})
    assert resp.status_code == 403


def test_cors_headers(client):
    resp = client.options("/api/v1/cameras",
                          headers={"Origin": "http://localhost:3000",
                                   "Access-Control-Request-Method": "GET"})
    # CORS middleware should add headers
    assert resp.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")
