"""
API tests for the Phase 12 model management endpoints.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_model_manager, get_model_registry
from src.api.main import create_app
from src.common.config import load_config
from src.common.model_manager import ModelManager
from src.common.model_manager_models import ModelType, ModelVersion
from src.common.model_registry import ModelRegistry


@pytest.fixture
def test_client():
    """Provide a test client with a fresh in-memory DB and temp registry."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config()
        cfg["auth"] = {
            "secret_key": "test-secret",
            "algorithm": "HS256",
            "access_token_expire_minutes": 60,
            "default_admin": {"username": "admin", "password": "admin", "role": "admin"},
        }
        cfg["model_management"] = {"registry_path": str(Path(tmp) / "registry.json")}

        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        app = create_app(config=cfg, engine=engine)

        registry = ModelRegistry(str(Path(tmp) / "registry.json"))
        manager = ModelManager(registry)
        app.dependency_overrides[get_model_registry] = lambda: registry
        app.dependency_overrides[get_model_manager] = lambda: manager

        with TestClient(app) as client:
            # Login as admin
            resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin"})
            assert resp.status_code == 200, resp.text
            token = resp.json()["access_token"]
            client.headers = {"Authorization": f"Bearer {token}"}
            yield client


def test_list_models_empty(test_client):
    resp = test_client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["models"] == []
    assert data["total"] == 0


def test_register_model(test_client):
    payload = {
        "model_id": "det_v1",
        "model_type": "detector",
        "version": "1.0",
        "path": "models/dummy.pt",
        "description": "test",
    }
    resp = test_client.post("/api/v1/models", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["model_id"] == "det_v1"
    assert data["status"] == "registered"


def test_get_model(test_client):
    test_client.post("/api/v1/models", json={
        "model_id": "det_v1", "model_type": "detector", "version": "1.0", "path": "models/dummy.pt"
    })
    resp = test_client.get("/api/v1/models/det_v1")
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "det_v1"


def test_get_model_not_found(test_client):
    resp = test_client.get("/api/v1/models/nope")
    assert resp.status_code == 404


def test_activate_and_get_active(test_client):
    test_client.post("/api/v1/models", json={
        "model_id": "det_v1", "model_type": "detector", "version": "1.0", "path": "models/dummy.pt"
    })
    resp = test_client.post("/api/v1/models/det_v1/activate")
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    resp = test_client.get("/api/v1/models/active/detector")
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "det_v1"


def test_retire_model(test_client):
    test_client.post("/api/v1/models", json={
        "model_id": "det_v1", "model_type": "detector", "version": "1.0", "path": "models/dummy.pt"
    })
    resp = test_client.post("/api/v1/models/det_v1/retire")
    assert resp.status_code == 200
    assert resp.json()["status"] == "retired"


def test_delete_model(test_client):
    test_client.post("/api/v1/models", json={
        "model_id": "det_v1", "model_type": "detector", "version": "1.0", "path": "models/dummy.pt"
    })
    resp = test_client.delete("/api/v1/models/det_v1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "det_v1"


def test_status(test_client):
    test_client.post("/api/v1/models", json={
        "model_id": "det_v1", "model_type": "detector", "version": "1.0", "path": "models/dummy.pt"
    })
    test_client.post("/api/v1/models/det_v1/activate")
    resp = test_client.get("/api/v1/models/status")
    assert resp.status_code == 200
    assert resp.json()["detector"]["model_id"] == "det_v1"
