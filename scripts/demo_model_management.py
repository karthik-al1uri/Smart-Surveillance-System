#!/usr/bin/env python3
"""
Demo script for Phase 12: Model Management & Hot Swap.

Registers two model versions, activates the first, performs a hot-swap to the
second, and then rolls back.  All against an empty temporary registry so it is
safe to run repeatedly without affecting production state.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.model_manager import ModelManager
from src.common.model_manager_models import ModelType, ModelVersion
from src.common.model_registry import ModelRegistry


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        registry_path = Path(tmp) / "registry.json"
        registry = ModelRegistry(str(registry_path))
        manager = ModelManager(registry)

        # Register two detector versions (point at non-existent files; stubs will be used)
        v1 = registry.register(
            ModelVersion(
                model_id="detector_v1",
                model_type=ModelType.DETECTOR,
                version="1.0.0",
                path=str(Path(tmp) / "detector_v1.pt"),
                description="Initial detector",
            )
        )
        v2 = registry.register(
            ModelVersion(
                model_id="detector_v2",
                model_type=ModelType.DETECTOR,
                version="2.0.0",
                path=str(Path(tmp) / "detector_v2.pt"),
                description="Improved detector",
            )
        )

        print("Registered models:")
        for m in registry.list_models():
            print(f"  - {m.model_id} ({m.model_type.value} v{m.version}) -> {m.status.value}")

        # Activate v1 and load into the manager
        registry.activate(v1.model_id)
        manager.load(ModelType.DETECTOR)
        print("\nLoaded model:", manager.get_version(ModelType.DETECTOR).model_id)

        # Hot-swap to v2
        print("\nHot-swapping to detector_v2 ...")
        swap_result = manager.swap(v2.model_id)
        print(f"  success={swap_result.success}")
        print(f"  previous={swap_result.previous_version}")
        print(f"  new={swap_result.new_version}")
        print(f"  duration={swap_result.duration_seconds:.4f}s")
        print("  active model:", manager.get_version(ModelType.DETECTOR).model_id)

        # Rollback to v1
        print("\nRolling back to previous model ...")
        rollback_result = manager.rollback(ModelType.DETECTOR)
        if rollback_result:
            print(f"  success={rollback_result.success}")
            print(f"  previous={rollback_result.previous_version}")
            print(f"  new={rollback_result.new_version}")
            print(f"  duration={rollback_result.duration_seconds:.4f}s")
            print("  active model:", manager.get_version(ModelType.DETECTOR).model_id)

        # Runtime status snapshot
        print("\nRuntime status:")
        for model_type, info in manager.get_status().items():
            print(f"  {model_type}: {info}")


if __name__ == "__main__":
    main()
