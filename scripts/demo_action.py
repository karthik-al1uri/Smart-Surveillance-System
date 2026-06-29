"""Demo: Full pipeline from video → detection + pose + tracking + action recognition.

Shows per-person activity classification overlaid on video.

Usage:
    python scripts/demo_action.py [--source <path_or_index>] [--save output.mp4]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.common.config import get_project_root, load_config
from src.common.logger import get_logger
from src.common.visualization import draw_tracked_frame
from src.detection.combined_pipeline import CombinedDetectionPipeline
from src.recognition.action_classes import ActionCategory, ActionLabel
from src.recognition.action_classifier import ActionClassifier
from src.recognition.recognition_pipeline import ActionRecognitionPipeline

logger = get_logger("demo.action")

_WARNING_CATEGORIES = {ActionCategory.VIOLENT, ActionCategory.URGENT, ActionCategory.SUSPICIOUS}
_CATEGORY_COLORS = {
    ActionCategory.NORMAL: (0, 200, 0),
    ActionCategory.VIOLENT: (0, 0, 255),
    ActionCategory.SUSPICIOUS: (0, 165, 255),
    ActionCategory.URGENT: (0, 100, 255),
}


def _check_weights(config: dict) -> None:
    root = get_project_root()
    missing = []
    for key, label in [("detection", "yolov8m.pt"), ("pose", "yolov8m-pose.pt")]:
        p = root / config[key]["model_path"]
        if not p.exists():
            missing.append((label, p))
    if missing:
        for label, p in missing:
            print(f"\n❌  Missing: {label} at {p}", file=sys.stderr)
        sys.exit(1)


def _ensure_dummy_model(config: dict) -> dict:
    root = get_project_root()
    model_path = root / config.get("action_recognition", {}).get("model_path", "models/action_classifier_v1.pt")
    if not model_path.exists():
        logger.warning("Action model not found at %s — creating dummy weights.", model_path)
        tmp_cfg = {**config, "action_recognition": {**config.get("action_recognition", {}),
                                                     "model_path": str(model_path)}}
        clf = ActionClassifier(config=tmp_cfg)
        clf.create_dummy_model(str(model_path))
    return config


def _draw_action_labels(
    frame: np.ndarray,
    tracks: list,
    pred_map: dict,
) -> np.ndarray:
    out = frame.copy()
    for track in tracks:
        pred = pred_map.get(track.track_id)
        if pred is None:
            continue
        x1, y1, _, _ = track.bbox
        text = f"{pred.label.name} ({pred.category.name}) {pred.confidence:.0%}"
        color = _CATEGORY_COLORS.get(pred.category, (200, 200, 200))
        cv2.putText(out, text, (x1, max(y1 - 20, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSS Action Recognition Demo")
    p.add_argument("--source", default="0")
    p.add_argument("--save", default="")
    p.add_argument("--max-frames", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config()
    _check_weights(config)
    config = _ensure_dummy_model(config)

    det_pipeline = CombinedDetectionPipeline(config=config)
    rec_pipeline = ActionRecognitionPipeline(config=config)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", args.source)
        sys.exit(1)

    writer = None
    if args.save:
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    frame_idx = 0
    all_preds = []
    latest_pred_map: dict = {}
    t_start = time.perf_counter()

    logger.info("Starting action recognition demo. Press Ctrl+C to stop.")
    try:
        while frame_idx < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            tracked = det_pipeline.process_frame(frame, camera_id="demo", frame_id=frame_idx)
            preds = rec_pipeline.process(tracked)

            for pred in preds:
                latest_pred_map[pred.track_id] = pred
                all_preds.append(pred)
                tag = " ⚠️" if pred.category in _WARNING_CATEGORIES else ""
                print(
                    f"[Frame {frame_idx:04d}] Track ID {pred.track_id} | "
                    f"Action: {pred.label.name} ({pred.category.name}) | "
                    f"Confidence: {pred.confidence:.2f}{tag}"
                )

            if writer is not None:
                annotated = draw_tracked_frame(frame, tracked)
                annotated = _draw_action_labels(annotated, tracked.tracks, latest_pred_map)
                writer.write(annotated)

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()
        if writer:
            writer.release()

    elapsed = time.perf_counter() - t_start
    by_cat: dict = {}
    for p in all_preds:
        by_cat[p.category.name] = by_cat.get(p.category.name, 0) + 1

    total = len(all_preds)
    print(f"\n{'═'*50}")
    print("  === Action Recognition Summary ===")
    print(f"  Frames processed   : {frame_idx}")
    print(f"  Total predictions  : {total}")
    print("  By category:")
    for cat_name, count in sorted(by_cat.items()):
        pct = 100 * count / total if total else 0
        print(f"    {cat_name:<12}: {count} ({pct:.1f}%)")
    print()
    print("  NOTE: Using untrained dummy model — predictions are not meaningful.")
    print("  Train with: python training/train_action.py --dataset_dir <your_dataset>")
    print(f"{'═'*50}\n")


if __name__ == "__main__":
    main()
