"""Demo: Full pipeline from video → detection + pose estimation.

Optionally saves annotated output video.

Usage:
    python scripts/demo_pose.py [--source <path_or_index>] [--save <output.mp4>]

Examples:
    python scripts/demo_pose.py --source 0
    python scripts/demo_pose.py --source video.mp4
    python scripts/demo_pose.py --source video.mp4 --save output.mp4
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
from src.common.visualization import draw_frame_analysis
from src.detection.combined_pipeline import CombinedDetectionPipeline

logger = get_logger("demo.pose")


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
        print("Place weights in the models/ directory and retry.\n", file=sys.stderr)
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSS Pose Estimation Demo")
    parser.add_argument("--source", default="0", help="Video source (file path or webcam index)")
    parser.add_argument("--save", default="", help="Save annotated video to this path")
    parser.add_argument("--max-frames", type=int, default=200, help="Max frames to process")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config()
    _check_weights(config)

    pipeline = CombinedDetectionPipeline(config=config)

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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_src, (w, h))
        logger.info("Saving annotated video to %s", args.save)

    frame_idx = 0
    total_persons = 0
    total_kp_visible = 0
    total_kp_count = 0
    t_start = time.perf_counter()

    logger.info("Starting pose demo. Press Ctrl+C to stop.")
    try:
        while frame_idx < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of source.")
                break

            analysis = pipeline.process_frame(frame, camera_id="demo", frame_id=frame_idx)

            n_persons = len(analysis.person_detections)
            n_objects = len(analysis.object_detections)
            total_persons += n_persons

            print(
                f"Frame {frame_idx:04d} | "
                f"Persons: {n_persons} | Objects: {n_objects}"
            )
            for pi, pose in enumerate(analysis.poses):
                vis = pose.visible_keypoint_count
                total_kp_visible += vis
                total_kp_count += 17
                det = analysis.person_detections[pi] if pi < len(analysis.person_detections) else None
                conf_str = f"conf={det.confidence:.2f}" if det else ""
                print(
                    f"  Person {pi + 1}: bbox={list(pose.bbox)} {conf_str}"
                    f" | Keypoints: {vis}/17 visible"
                )
            for det in analysis.object_detections:
                print(f"  Object: {det.class_name} ({det.confidence:.2f}) {list(det.bbox)}")

            if writer is not None:
                annotated = draw_frame_analysis(frame, analysis)
                writer.write(annotated)

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - t_start
    fps = frame_idx / elapsed if elapsed > 0 else 0
    avg_kp = (total_kp_visible / total_kp_count * 17) if total_kp_count > 0 else 0

    print(
        f"\n{'─'*55}\n"
        f"  Frames processed         : {frame_idx}\n"
        f"  Total persons detected   : {total_persons}\n"
        f"  Avg keypoints visible    : {avg_kp:.1f} / 17\n"
        f"  Elapsed time             : {elapsed:.2f}s\n"
        f"  Inference FPS            : {fps:.2f}\n"
        f"{'─'*55}"
    )


if __name__ == "__main__":
    main()
