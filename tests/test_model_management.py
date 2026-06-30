"""
Tests for model management (Phase 12): registry, model manager, and hot-swap.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.common.model_manager import ModelManager, _StubModel
from src.common.model_manager_models import (
    ModelStatus,
    ModelSwapResult,
    ModelType,
    ModelVersion,
)
from src.common.model_registry import ModelRegistry


@pytest.fixture
def tmp_registry():
    """Provide a temporary registry file and cleanup."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "registry.json"
        yield str(path)


def make_version(model_id: str, version: str, mtype: ModelType, path: str = "dummy.pt"):
    return ModelVersion(model_id=model_id, model_type=mtype, version=version, path=path)


class TestModelRegistry:
    def test_register_and_get(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        v = reg.register(make_version("det_v1", "1.0", ModelType.DETECTOR))
        assert v.status == ModelStatus.REGISTERED
        assert reg.get("det_v1") == v

    def test_register_duplicate_raises(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        reg.register(make_version("det_v1", "1.0", ModelType.DETECTOR))
        with pytest.raises(ValueError):
            reg.register(make_version("det_v1", "1.1", ModelType.DETECTOR))

    def test_activate_retires_previous(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        v1 = reg.register(make_version("det_v1", "1.0", ModelType.DETECTOR))
        v2 = reg.register(make_version("det_v2", "1.1", ModelType.DETECTOR))
        reg.activate(v1.model_id)
        assert reg.get(v1.model_id).status == ModelStatus.ACTIVE
        reg.activate(v2.model_id)
        assert reg.get(v1.model_id).status == ModelStatus.RETIRED
        assert reg.get(v2.model_id).status == ModelStatus.ACTIVE

    def test_get_active(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        v1 = reg.register(make_version("det_v1", "1.0", ModelType.DETECTOR))
        reg.activate(v1.model_id)
        assert reg.get_active(ModelType.DETECTOR) == v1
        assert reg.get_active(ModelType.POSE_ESTIMATOR) is None

    def test_list_filters(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        reg.register(make_version("det_v1", "1.0", ModelType.DETECTOR))
        reg.register(make_version("pose_v1", "1.0", ModelType.POSE_ESTIMATOR))
        assert len(reg.list_models(model_type=ModelType.DETECTOR)) == 1
        assert len(reg.list_models()) == 2

    def test_delete(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        reg.register(make_version("det_v1", "1.0", ModelType.DETECTOR))
        reg.delete("det_v1")
        assert reg.get("det_v1") is None

    def test_persistence_round_trip(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        reg.register(make_version("det_v1", "1.0", ModelType.DETECTOR))
        reg2 = ModelRegistry(tmp_registry)
        assert reg2.get("det_v1") is not None
        assert reg2.get("det_v1").model_type == ModelType.DETECTOR


class TestModelManager:
    def test_load_active_returns_stub_when_file_missing(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        reg.register(make_version("det_v1", "1.0", ModelType.DETECTOR, path="missing.pt"))
        reg.activate("det_v1")
        mgr = ModelManager(reg)
        model = mgr.load(ModelType.DETECTOR)
        assert isinstance(model, _StubModel)
        assert mgr.get_version(ModelType.DETECTOR).model_id == "det_v1"

    def test_swap_to_model(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        reg.register(make_version("det_v1", "1.0", ModelType.DETECTOR, path="missing.pt"))
        reg.activate("det_v1")
        reg.register(make_version("det_v2", "2.0", ModelType.DETECTOR, path="missing.pt"))
        mgr = ModelManager(reg)
        mgr.load(ModelType.DETECTOR)

        result = mgr.swap("det_v2")
        assert result.success is True
        assert result.new_version == "2.0"
        assert result.previous_version == "1.0"
        assert mgr.get_version(ModelType.DETECTOR).model_id == "det_v2"

    def test_swap_unknown_model_returns_failure(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        mgr = ModelManager(reg)
        result = mgr.swap("does_not_exist")
        assert result.success is False
        assert "not found" in (result.error or "").lower()

    def test_rollback(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        reg.register(make_version("det_v1", "1.0", ModelType.DETECTOR, path="missing.pt"))
        reg.activate("det_v1")
        reg.register(make_version("det_v2", "2.0", ModelType.DETECTOR, path="missing.pt"))
        mgr = ModelManager(reg)
        mgr.load(ModelType.DETECTOR)
        mgr.swap("det_v2")

        result = mgr.rollback(ModelType.DETECTOR)
        assert result is not None
        assert result.success is True
        assert result.new_version == "1.0"
        assert result.previous_version == "2.0"
        assert mgr.get_version(ModelType.DETECTOR).model_id == "det_v1"

    def test_rollback_with_no_history_returns_none(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        mgr = ModelManager(reg)
        assert mgr.rollback(ModelType.DETECTOR) is None

    def test_get_status(self, tmp_registry):
        reg = ModelRegistry(tmp_registry)
        reg.register(make_version("det_v1", "1.0", ModelType.DETECTOR, path="missing.pt"))
        reg.activate("det_v1")
        mgr = ModelManager(reg)
        mgr.load(ModelType.DETECTOR)
        status = mgr.get_status()
        assert status["detector"]["model_id"] == "det_v1"
        assert status["pose_estimator"] is None


class TestAutoRegister:
    def test_auto_register_skips_existing(self, tmp_registry, tmp_path):
        reg = ModelRegistry(tmp_registry)
        # Create a fake model file
        model_file = tmp_path / "detector.pt"
        model_file.write_text("fake")
        reg.register(ModelVersion(model_id="detector", model_type=ModelType.DETECTOR, version="auto", path=str(model_file)))
        registered = reg.auto_register_existing(str(tmp_path))
        assert registered == []

    def test_auto_register_finds_new_files(self, tmp_registry, tmp_path):
        reg = ModelRegistry(tmp_registry)
        (tmp_path / "detector_v2.pt").write_text("fake")
        (tmp_path / "pose_v1.pt").write_text("fake")
        registered = reg.auto_register_existing(str(tmp_path))
        assert len(registered) == 2
        types = {m.model_type for m in registered}
        assert ModelType.DETECTOR in types
        assert ModelType.POSE_ESTIMATOR in types


class TestModelVersionSerialization:
    def test_to_dict_and_from_dict(self):
        v = ModelVersion(
            model_id="det_v1",
            model_type=ModelType.DETECTOR,
            version="1.0",
            path="models/det_v1.pt",
            description="test",
            status=ModelStatus.ACTIVE,
        )
        restored = ModelVersion.from_dict(v.to_dict())
        assert restored.model_id == v.model_id
        assert restored.model_type == v.model_type
        assert restored.status == v.status

    def test_swap_result_to_dict(self):
        r = ModelSwapResult(success=True, model_type=ModelType.DETECTOR, previous_version="1.0", new_version="2.0", duration_seconds=0.12)
        d = r.to_dict()
        assert d["success"] is True
        assert d["new_version"] == "2.0"
