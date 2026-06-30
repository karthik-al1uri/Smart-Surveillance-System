"""
Model manager — thread-safe hot-swap of ML models in the inference pipeline.

Wraps model loading and provides atomic swap, rollback, and status reporting
without stopping any in-flight inference.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from src.common.logger import get_logger
from src.common.model_manager_models import (
    ModelStatus,
    ModelSwapResult,
    ModelType,
    ModelVersion,
)
from src.common.model_registry import ModelRegistry

logger = get_logger("model_manager")


class ModelManager:
    """
    Thread-safe runtime manager for loaded model objects.

    Maintains one loaded model object per ModelType.  Hot-swapping replaces
    the model object atomically under an RLock so concurrent inference threads
    always see a consistent model.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._lock = RLock()
        # model_type -> (ModelVersion, loaded_model_object)
        self._loaded: dict[ModelType, tuple[ModelVersion, Any]] = {}
        self._history: dict[ModelType, list[ModelVersion]] = {t: [] for t in ModelType}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(mv: ModelVersion) -> Any:
        """Load a model from disk.  Returns the raw framework object.

        Supports:
        - .pt  → Ultralytics YOLO or torch.load
        - .onnx → onnxruntime InferenceSession
        - .pkl  → pickle (scikit-learn classifiers)
        - stub  → returns a mock for testing when file is absent
        """
        path = Path(mv.path)
        if not path.exists():
            logger.warning("Model file not found, using stub: %s", mv.path)
            return _StubModel(mv)

        suffix = path.suffix.lower()
        try:
            if suffix == ".pt":
                try:
                    from ultralytics import YOLO
                    model = YOLO(str(path))
                    logger.info("Loaded YOLO model: %s", mv.model_id)
                    return model
                except ImportError:
                    import torch
                    model = torch.load(str(path), map_location="cpu")
                    logger.info("Loaded torch model: %s", mv.model_id)
                    return model

            elif suffix == ".onnx":
                import onnxruntime as ort
                sess = ort.InferenceSession(str(path))
                logger.info("Loaded ONNX model: %s", mv.model_id)
                return sess

            elif suffix == ".pkl":
                import pickle
                with open(path, "rb") as fh:
                    model = pickle.load(fh)
                logger.info("Loaded pickle model: %s", mv.model_id)
                return model

        except Exception as exc:
            logger.error("Failed to load model %s: %s", mv.model_id, exc)
            raise

        raise ValueError(f"Unsupported model format: {suffix}")

    def load(self, model_type: ModelType) -> Optional[Any]:
        """Load the currently active model for a type (if any) into memory."""
        mv = self._registry.get_active(model_type)
        if mv is None:
            logger.info("No active model for %s", model_type.value)
            return None
        model_obj = self._load_model(mv)
        with self._lock:
            self._loaded[model_type] = (mv, model_obj)
            self._history[model_type] = [mv]
        return model_obj

    # ------------------------------------------------------------------
    # Hot-swap
    # ------------------------------------------------------------------

    def swap(self, model_id: str) -> ModelSwapResult:
        """Hot-swap to a different model version by ID.

        1. Looks up the model in the registry.
        2. Loads the new model object.
        3. Atomically replaces the current model under the lock.
        4. Activates the new version in the registry.
        5. Returns a ModelSwapResult.
        """
        start = time.monotonic()
        mv = self._registry.get(model_id)
        if mv is None:
            return ModelSwapResult(
                success=False,
                model_type=ModelType.DETECTOR,
                previous_version=None,
                new_version=model_id,
                duration_seconds=time.monotonic() - start,
                error=f"Model not found in registry: {model_id}",
            )

        with self._lock:
            current = self._loaded.get(mv.model_type)
            prev_version = current[0].version if current else None

            try:
                new_obj = self._load_model(mv)
            except Exception as exc:
                return ModelSwapResult(
                    success=False,
                    model_type=mv.model_type,
                    previous_version=prev_version,
                    new_version=mv.version,
                    duration_seconds=time.monotonic() - start,
                    error=str(exc),
                )

            # Keep old entry in history for rollback
            if current is not None:
                self._history[mv.model_type].append(current[0])

            self._loaded[mv.model_type] = (mv, new_obj)

        # Update registry outside the inference-path lock to minimise contention
        self._registry.activate(model_id)

        duration = time.monotonic() - start
        logger.info(
            "Hot-swapped %s: %s → %s in %.3fs",
            mv.model_type.value, prev_version, mv.version, duration,
        )
        return ModelSwapResult(
            success=True,
            model_type=mv.model_type,
            previous_version=prev_version,
            new_version=mv.version,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, model_type: ModelType) -> Optional[ModelSwapResult]:
        """Revert to the previous loaded model version (in-memory only)."""
        with self._lock:
            history = self._history.get(model_type, [])
            if len(history) < 1:
                logger.warning("No rollback target for %s", model_type.value)
                return None

            prev_mv = history.pop()
            # The previous object is no longer cached; reload it
            start = time.monotonic()
            try:
                prev_obj = self._load_model(prev_mv)
            except Exception as exc:
                return ModelSwapResult(
                    success=False,
                    model_type=model_type,
                    previous_version=None,
                    new_version=prev_mv.version,
                    duration_seconds=time.monotonic() - start,
                    error=str(exc),
                )

            current = self._loaded.get(model_type)
            current_version = current[0].version if current else None
            self._loaded[model_type] = (prev_mv, prev_obj)
            duration = time.monotonic() - start

        # Reactivate in registry
        try:
            self._registry.activate(prev_mv.model_id)
        except Exception:
            pass

        logger.info(
            "Rolled back %s: %s → %s in %.3fs",
            model_type.value, current_version, prev_mv.version, duration,
        )
        return ModelSwapResult(
            success=True,
            model_type=model_type,
            previous_version=current_version,
            new_version=prev_mv.version,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get(self, model_type: ModelType) -> Optional[Any]:
        """Return the currently loaded model object for a type, or None."""
        with self._lock:
            entry = self._loaded.get(model_type)
            return entry[1] if entry else None

    def get_version(self, model_type: ModelType) -> Optional[ModelVersion]:
        """Return the ModelVersion metadata for the currently loaded model."""
        with self._lock:
            entry = self._loaded.get(model_type)
            return entry[0] if entry else None

    def get_status(self) -> dict:
        """Return a status snapshot of all loaded models."""
        with self._lock:
            return {
                mt.value: {
                    "model_id": entry[0].model_id,
                    "version": entry[0].version,
                    "status": entry[0].status.value,
                    "path": entry[0].path,
                }
                if (entry := self._loaded.get(mt)) else None
                for mt in ModelType
            }


class _StubModel:
    """Placeholder returned when a model file doesn't exist (dev/test mode)."""

    def __init__(self, mv: ModelVersion) -> None:
        self._mv = mv

    def __call__(self, *args: Any, **kwargs: Any) -> list:
        logger.debug("StubModel called for %s — returning empty results", self._mv.model_id)
        return []

    def __repr__(self) -> str:
        return f"<StubModel model_id={self._mv.model_id!r}>"
