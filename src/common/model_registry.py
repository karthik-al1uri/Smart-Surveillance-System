"""
Model registry — persists model metadata to a JSON file.

Tracks all registered model versions, their lifecycle status, and
which version is currently active for each model type.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

from src.common.logger import get_logger
from src.common.model_manager_models import ModelStatus, ModelType, ModelVersion

logger = get_logger("model_registry")

DEFAULT_REGISTRY_PATH = "models/registry.json"


class ModelRegistry:
    """Thread-safe JSON-backed registry for model versions."""

    def __init__(self, registry_path: str = DEFAULT_REGISTRY_PATH) -> None:
        self._path = Path(registry_path)
        self._lock = Lock()
        self._models: dict[str, ModelVersion] = {}  # model_id -> ModelVersion
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the registry from disk (no-op if file doesn't exist)."""
        if not self._path.exists():
            logger.info("Registry file not found — starting empty: %s", self._path)
            return
        try:
            with open(self._path) as fh:
                data = json.load(fh)
            for entry in data.get("models", []):
                mv = ModelVersion.from_dict(entry)
                self._models[mv.model_id] = mv
            logger.info("Loaded %d model(s) from registry", len(self._models))
        except Exception as exc:
            logger.error("Failed to load registry: %s", exc)

    def save(self) -> None:
        """Persist the registry to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as fh:
                json.dump({"models": [m.to_dict() for m in self._models.values()]}, fh, indent=2)
            tmp.replace(self._path)
            logger.debug("Registry saved to %s", self._path)
        except Exception as exc:
            logger.error("Failed to save registry: %s", exc)
            if tmp.exists():
                tmp.unlink()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, model: ModelVersion) -> ModelVersion:
        """Register a new model version. Raises if model_id already exists."""
        with self._lock:
            if model.model_id in self._models:
                raise ValueError(f"Model ID already registered: {model.model_id}")
            self._models[model.model_id] = model
            self.save()
            logger.info("Registered model %s (%s v%s)", model.model_id, model.model_type.value, model.version)
            return model

    def get(self, model_id: str) -> Optional[ModelVersion]:
        """Return a model version by ID, or None."""
        with self._lock:
            return self._models.get(model_id)

    def get_active(self, model_type: ModelType) -> Optional[ModelVersion]:
        """Return the currently active model for a given type, or None."""
        with self._lock:
            for mv in self._models.values():
                if mv.model_type == model_type and mv.status == ModelStatus.ACTIVE:
                    return mv
            return None

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
    ) -> list[ModelVersion]:
        """Return models, optionally filtered by type and/or status."""
        with self._lock:
            results = list(self._models.values())
        if model_type is not None:
            results = [m for m in results if m.model_type == model_type]
        if status is not None:
            results = [m for m in results if m.status == status]
        return results

    def activate(self, model_id: str) -> ModelVersion:
        """Mark a model as active; retires any previously active model of the same type."""
        with self._lock:
            target = self._models.get(model_id)
            if target is None:
                raise KeyError(f"Model not found: {model_id}")
            # Allow reactivating a retired model (e.g. rollback) by resetting its status.
            if target.status == ModelStatus.RETIRED:
                target.status = ModelStatus.REGISTERED

            # Retire current active model for this type
            for mv in self._models.values():
                if mv.model_type == target.model_type and mv.status == ModelStatus.ACTIVE:
                    mv.status = ModelStatus.RETIRED
                    logger.info("Retired model %s", mv.model_id)

            target.status = ModelStatus.ACTIVE
            target.activated_at = datetime.now(timezone.utc).isoformat()
            self.save()
            logger.info("Activated model %s", model_id)
            return target

    def retire(self, model_id: str) -> ModelVersion:
        """Mark a model as retired."""
        with self._lock:
            target = self._models.get(model_id)
            if target is None:
                raise KeyError(f"Model not found: {model_id}")
            target.status = ModelStatus.RETIRED
            self.save()
            logger.info("Retired model %s", model_id)
            return target

    def delete(self, model_id: str) -> None:
        """Remove a model from the registry (does not delete the file on disk)."""
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Model not found: {model_id}")
            del self._models[model_id]
            self.save()
            logger.info("Deleted model %s from registry", model_id)

    def auto_register_existing(self, models_dir: str = "models") -> list[ModelVersion]:
        """Scan a directory and register any .pt/.onnx files not already registered."""
        registered: list[ModelVersion] = []
        models_path = Path(models_dir)
        if not models_path.exists():
            return registered

        existing_paths = {mv.path for mv in self._models.values()}

        for f in models_path.rglob("*"):
            if f.suffix not in {".pt", ".onnx", ".pkl"} or not f.is_file():
                continue
            path_str = str(f)
            if path_str in existing_paths:
                continue

            # Infer model type from filename heuristic
            name = f.stem.lower()
            if "pose" in name or "keypoint" in name:
                mtype = ModelType.POSE_ESTIMATOR
            elif "action" in name or "class" in name or "lstm" in name:
                mtype = ModelType.ACTION_CLASSIFIER
            else:
                mtype = ModelType.DETECTOR

            model_id = f.stem
            version = "auto"
            try:
                mv = self.register(
                    ModelVersion(
                        model_id=model_id,
                        model_type=mtype,
                        version=version,
                        path=path_str,
                        description=f"Auto-registered from {path_str}",
                    )
                )
                registered.append(mv)
            except ValueError:
                pass  # already registered

        return registered
