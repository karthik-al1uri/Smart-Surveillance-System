"""
Model management data structures for Phase 12.

Defines enums and dataclasses used by the model registry and model manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ModelType(str, Enum):
    """Supported model types in the inference pipeline."""

    DETECTOR = "detector"
    POSE_ESTIMATOR = "pose_estimator"
    ACTION_CLASSIFIER = "action_classifier"


class ModelStatus(str, Enum):
    """Lifecycle status of a registered model version."""

    REGISTERED = "registered"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Represents a single versioned model artifact."""

    model_id: str
    model_type: ModelType
    version: str
    path: str
    description: str = ""
    status: ModelStatus = ModelStatus.REGISTERED
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    activated_at: Optional[str] = None
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "version": self.version,
            "path": self.path,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at,
            "activated_at": self.activated_at,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        return cls(
            model_id=data["model_id"],
            model_type=ModelType(data["model_type"]),
            version=data["version"],
            path=data["path"],
            description=data.get("description", ""),
            status=ModelStatus(data.get("status", ModelStatus.REGISTERED.value)),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            activated_at=data.get("activated_at"),
            metrics=data.get("metrics", {}),
        )


@dataclass
class ModelSwapResult:
    """Result of a model hot-swap operation."""

    success: bool
    model_type: ModelType
    previous_version: Optional[str]
    new_version: str
    duration_seconds: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "model_type": self.model_type.value,
            "previous_version": self.previous_version,
            "new_version": self.new_version,
            "duration_seconds": round(self.duration_seconds, 4),
            "error": self.error,
        }
