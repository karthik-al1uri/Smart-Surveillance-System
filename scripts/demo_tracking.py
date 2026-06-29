"""Demo: Full pipeline from video → detection + pose + tracking.

Shows consistent person IDs across frames with movement trails.

Usage:
    python scripts/demo_tracking.py [--source <path_or_index>] [--save <output.mp4>]

Examples:
    python scripts/demo_tracking.py --source 0
    python scripts/demo_tracking.py --source video.mp4
    python scripts/demo_tracking.py --source video.mp4 --save tracked.mp4
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
from src.common.visualization import draw_tracked_frame
from src.detection.combined_pipeline import CombinedDetectionPipeline

logger = get_logger("demo.tracking")


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
    parser = argparse.ArgumentParser(description="SSS Tracking Demo")
    parser.add_argument("--source", default="0", help="Video source (file path or webcam index)")
    parser.add_argument("--save", default="", help="Save annotated video to this path")
    parser.add_argument("--max-frames", type=int, default=150, help="Max frames to process")
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
    all_track_ids: set = set()
    total_infer_ms = 0.0
    total_track_ms = 0.0
    t_start = time.perf_counter()

    logger.info("Starting tracking demo. Press Ctrl+C to stop.")
    try:
        while frame_idx < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of source.")
                break

            analysis = pipeline.process_frame(frame, camera_id="demo", frame_id=frame_idx)
            total_infer_ms += analysis.inference_time_ms
            total_track_ms += analysis.tracking_time_ms

            active = [t for t in analysis.tracks if t.state == "active"]
            lost = [t for t in analysis.tracks if t.state == "lost"]
            for t in analysis.tracks:
                all_track_ids.add(t.track_id)

            print(
                f"Frame {frame_idx:04d} | "
                f"Active tracks: {len(active)} | Lost: {len(lost)}"
            )
            for track in active:
                kp_ready = pipeline._tracker.get_track_keypoint_sequence(track.track_id, 16)
                kp_tag = f"kp_seq_ready=YES ({track.track_id})" if kp_ready is not None else ""
                print(
                    f"  Track ID {track.track_id}: "
                    f"bbox={list(track.bbox)} "
                    f"conf={track.confidence:.2f} "
                    f"age={track.age} "
                    f"| {len(track.keypoint_history)}/17+ keypoints "
                    f"{kp_tag}"
                )
            for track in lost:
                print(
                    f"  Track ID {track.track_id}: LOST "
                    f"({track.time_since_update} frames) "
                    f"last_bbox={list(track.bbox)}"
                )
            for det in analysis.object_detections:
                print(f"  Object: {det.class_name} ({det.confidence:.2f}) {list(det.bbox)}")

            if writer is not None:
                annotated = draw_tracked_frame(frame, analysis)
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
    avg_infer = total_infer_ms / frame_idx if frame_idx > 0 else 0
    avg_track = total_track_ms / frame_idx if frame_idx > 0 else 0

    kp_ready_count = 0
    if pipeline._tracker:
        for tid in all_track_ids:
            if pipeline._tracker.get_track_keypoint_sequence(tid, 16) is not None:
                kp_ready_count += 1

    print(
        f"\n{'═'*55}\n"
        f"  === Tracking Summary ===\n"
        f"  Frames processed         : {frame_idx}\n"
        f"  Total unique tracks      : {len(all_track_ids)}\n"
        f"  Currently active         : {len([t for t in (pipeline._tracker._tracks if pipeline._tracker else []) if t.state == 'active'])}\n"
        f"  Avg infer+tracking time  : {avg_infer + avg_track:.1f}ms ({1000 / (avg_infer + avg_track + 1e-9):.1f} FPS)\n"
        f"  Keypoint seq ready (≥16) : {kp_ready_count} tracks\n"
        f"  Elapsed time             : {elapsed:.2f}s\n"
        f"{'═'*55}"
    )


if __name__ == "__main__":
    main()
