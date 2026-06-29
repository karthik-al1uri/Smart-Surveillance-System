"""End-to-end detection demo script.

Runs YOLOv8 object detection on a video file or webcam stream and prints
detection results to the console.  Optionally saves annotated frames.

Usage:
    python scripts/demo_detection.py --source <path_or_index> [--save]

Examples:
    python scripts/demo_detection.py --source 0               # webcam
    python scripts/demo_detection.py --source video.mp4       # video file
    python scripts/demo_detection.py --source video.mp4 --save
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.common.config import get_project_root, load_config
from src.common.logger import get_logger
from src.detection.detection_pipeline import DetectionPipeline

logger = get_logger("demo.detection")


def _check_weights(config: dict) -> None:
    """Exit with clear instructions if model weights are missing."""
    model_path = get_project_root() / config["detection"]["model_path"]
    if not model_path.exists():
        print(
            f"\n❌  Model weights not found at: {model_path}\n"
            "    Place yolov8m.pt in the models/ directory.\n"
            "    Download command:\n"
            "      python -c \"from ultralytics import YOLO; YOLO('yolov8m.pt')\"\n"
            "    Then move it:  mv yolov8m.pt models/\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSS Detection Demo")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: file path or webcam index (default: 0)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output frames to demo_output/ directory",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum number of frames to process (default: 100)",
    )
    return parser.parse_args()


def _draw_detections(frame, result) -> None:
    """Draw bounding boxes and labels on frame in-place."""
    for det in result.detections:
        x1, y1, x2, y2 = det.bbox
        label = f"{det.class_name} {det.confidence:.2f}"
        color = (0, 255, 0) if det.class_name == "person" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main() -> None:
    args = _parse_args()
    config = load_config()
    _check_weights(config)

    pipeline = DetectionPipeline(config=config)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", args.source)
        sys.exit(1)

    output_dir = PROJECT_ROOT / "demo_output"
    if args.save:
        output_dir.mkdir(exist_ok=True)
        logger.info("Saving annotated frames to %s", output_dir)

    frame_idx = 0
    total_detections = 0
    t_start = time.perf_counter()

    logger.info("Starting detection demo. Press Ctrl+C to stop.")
    try:
        while frame_idx < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of source reached.")
                break

            result = pipeline.process_frame(frame, frame_idx=frame_idx)
            total_detections += result.count

            persons = result.person_detections
            print(
                f"[Frame {frame_idx:04d}] detections={result.count} "
                f"persons={len(persons)} "
                f"infer={result.inference_time_ms:.1f}ms"
            )
            for det in result.detections:
                print(f"  └─ {det.class_name} conf={det.confidence:.3f} bbox={det.bbox}")

            if args.save:
                annotated = frame.copy()
                _draw_detections(annotated, result)
                cv2.imwrite(str(output_dir / f"frame_{frame_idx:04d}.jpg"), annotated)

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()

    elapsed = time.perf_counter() - t_start
    fps = frame_idx / elapsed if elapsed > 0 else 0
    print(
        f"\n{'─'*50}\n"
        f"  Frames processed : {frame_idx}\n"
        f"  Total detections : {total_detections}\n"
        f"  Elapsed time     : {elapsed:.2f}s\n"
        f"  Effective FPS    : {fps:.2f}\n"
        f"{'─'*50}"
    )


if __name__ == "__main__":
    main()
