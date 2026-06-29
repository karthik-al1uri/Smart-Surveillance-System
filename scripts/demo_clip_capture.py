"""
Demo: Full pipeline with clip capture.

Runs video through detection → scoring → clip capture.
Simulates alerts and shows captured clips.

Usage:
    python scripts/demo_clip_capture.py \\
        [--source assets/test_video.mp4] \\
        [--output_dir data/demo_clips] \\
        [--max-frames N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.alerts.alert_integration import AlertIntegration
from src.alerts.clip_capture import ClipCaptureService
from src.common.config import get_project_root, load_config
from src.common.logger import get_logger
from src.detection.combined_pipeline import CombinedDetectionPipeline
from src.recognition.recognition_pipeline import ActionRecognitionPipeline
from src.scoring.anomaly_scorer import AnomalyScorer
from src.scoring.scoring_models import AlertDecision
from src.scoring.scoring_pipeline import ScoringPipeline
from src.scoring.zone_engine import ZoneEngine
from src.scoring.zone_manager import ZoneManager
from src.scoring.zone_models import Rule, RuleType, Zone, ZoneType

logger = get_logger("demo.clip_capture")

CAMERA_ID = "demo_cam"


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


def _build_demo_zones(w: int, h: int) -> ZoneManager:
    zm = ZoneManager()
    z = Zone(
        zone_id="entry_zone",
        camera_id=CAMERA_ID,
        name="Entry Zone",
        zone_type=ZoneType.RESTRICTED,
        polygon=[
            (int(w * 0.55), int(h * 0.05)),
            (int(w * 0.98), int(h * 0.05)),
            (int(w * 0.98), int(h * 0.55)),
            (int(w * 0.55), int(h * 0.55)),
        ],
    )
    zm.add_zone(z)
    zm.add_rule(Rule("r_noe", "entry_zone", RuleType.NO_ENTRY, cooldown_seconds=15.0))
    zm.add_rule(Rule("r_loi", "entry_zone", RuleType.LOITERING,
                     max_duration_seconds=10.0, cooldown_seconds=30.0))
    return zm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSS Clip Capture Demo")
    p.add_argument("--source", default="0")
    p.add_argument("--output_dir", default="data/demo_clips")
    p.add_argument("--max-frames", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config()
    _check_weights(config)

    # Lower threshold for demo
    config["scoring"]["alert_threshold"] = 0.10
    config["scoring"]["hysteresis_count"] = 1
    config["storage"]["clip_dir"] = args.output_dir
    config["storage"]["clip_buffer_duration"] = 25.0
    config["storage"]["clip_buffer_compressed"] = True

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", args.source)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det_pipeline = CombinedDetectionPipeline(config=config)
    action_pipeline = ActionRecognitionPipeline(config=config)
    zone_manager = _build_demo_zones(w, h)
    zone_engine = ZoneEngine(zone_manager)
    scorer = AnomalyScorer(config)
    scoring_pipeline = ScoringPipeline(config, zone_engine, action_pipeline, scorer)

    clip_service = ClipCaptureService(config)
    clip_service.register_camera(CAMERA_ID)
    clip_service.start()

    alert_integration = AlertIntegration(clip_service, config)

    frame_idx = 0
    alert_count = 0

    logger.info("Starting clip capture demo. Press Ctrl+C to stop.")
    try:
        while frame_idx < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            ts = time.time()
            clip_service.feed_frame(CAMERA_ID, frame, ts, frame_idx)

            tracked = det_pipeline.process_frame(frame, camera_id=CAMERA_ID, frame_id=frame_idx)
            action_preds = action_pipeline.process(tracked)
            zone_viols = zone_engine.evaluate(tracked)
            scored_events = scoring_pipeline.process(tracked, action_preds, zone_viols)

            submitted = alert_integration.handle_scored_events(scored_events)
            for req in submitted:
                alert_count += 1
                ev = next(e for e in scored_events if e.event_id == req.event_id)
                print(f"\n[Frame {frame_idx:04d}] ALERT triggered — event_id={req.event_id},"
                      f" camera={req.camera_id}, score={ev.severity_score:.2f}")
                print(f"  → Clip requested: {req.pre_seconds:.0f}s before + {req.post_seconds:.0f}s after event")

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()

    # Wait briefly for last post-event frames then stop
    time.sleep(6.0)
    clip_service.stop()

    # Print captured clips
    all_clips = clip_service.get_all_clips()
    stats = clip_service.get_storage_stats()

    print(f"\n{'═'*60}")
    print("  === Captured Clips ===")
    for n, clip in enumerate(all_clips, 1):
        size_mb = clip.file_size_bytes / (1024 * 1024)
        print(f"  {n}. {clip.event_id} → {Path(clip.file_path).name}"
              f"  ({clip.duration_seconds:.1f}s, {size_mb:.1f}MB)")

    print(f"\n  Storage: {stats.total_clips} clip(s),"
          f" {stats.total_size_bytes / (1024*1024):.1f} MB total")
    print(f"  Output dir: {stats.storage_path}")
    print(f"\n  Frames processed : {frame_idx}")
    print(f"  Alerts triggered : {alert_count}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
