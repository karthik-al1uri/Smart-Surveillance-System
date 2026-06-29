"""Demo: Zone engine with predefined zones on test video.

Defines example zones and shows violations as persons move through them.

Usage:
    python scripts/demo_zones.py [--source <path_or_index>] [--save output.mp4]
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
from src.common.visualization import draw_tracked_frame, draw_violations, draw_zones
from src.detection.combined_pipeline import CombinedDetectionPipeline
from src.scoring.zone_engine import ZoneEngine
from src.scoring.zone_manager import ZoneManager
from src.scoring.zone_models import Rule, RuleType, Zone, ZoneType

logger = get_logger("demo.zones")


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


def _build_demo_zones(frame_w: int, frame_h: int) -> ZoneManager:
    """Create three demo zones scaled to the frame dimensions."""
    zm = ZoneManager()

    z1 = Zone(
        zone_id="loading_dock",
        camera_id="demo",
        name="Loading Dock",
        zone_type=ZoneType.RESTRICTED,
        polygon=[
            (int(frame_w * 0.05), int(frame_h * 0.60)),
            (int(frame_w * 0.35), int(frame_h * 0.60)),
            (int(frame_w * 0.35), int(frame_h * 0.95)),
            (int(frame_w * 0.05), int(frame_h * 0.95)),
        ],
    )
    zm.add_zone(z1)
    zm.add_rule(Rule("r_dock_noe", "loading_dock", RuleType.NO_ENTRY, cooldown_seconds=30.0))
    zm.add_rule(Rule("r_dock_loi", "loading_dock", RuleType.LOITERING,
                     max_duration_seconds=30.0, cooldown_seconds=60.0))
    zm.add_rule(Rule("r_dock_int", "loading_dock", RuleType.INTRUSION, cooldown_seconds=0.0))

    z2 = Zone(
        zone_id="entrance",
        camera_id="demo",
        name="Main Entrance",
        zone_type=ZoneType.PERIMETER,
        polygon=[
            (int(frame_w * 0.40), int(frame_h * 0.05)),
            (int(frame_w * 0.70), int(frame_h * 0.05)),
            (int(frame_w * 0.70), int(frame_h * 0.40)),
            (int(frame_w * 0.40), int(frame_h * 0.40)),
        ],
    )
    zm.add_zone(z2)
    zm.add_rule(Rule("r_ent_int", "entrance", RuleType.INTRUSION, cooldown_seconds=0.0))
    zm.add_rule(Rule("r_ent_crowd", "entrance", RuleType.CROWD_LIMIT,
                     max_persons=3, cooldown_seconds=30.0))

    z3 = Zone(
        zone_id="server_room",
        camera_id="demo",
        name="Server Room",
        zone_type=ZoneType.RESTRICTED,
        polygon=[
            (int(frame_w * 0.72), int(frame_h * 0.50)),
            (int(frame_w * 0.98), int(frame_h * 0.50)),
            (int(frame_w * 0.98), int(frame_h * 0.98)),
            (int(frame_w * 0.72), int(frame_h * 0.98)),
        ],
    )
    zm.add_zone(z3)
    zm.add_rule(Rule("r_srv_noe", "server_room", RuleType.NO_ENTRY, cooldown_seconds=15.0))

    return zm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSS Zone Engine Demo")
    p.add_argument("--source", default="0")
    p.add_argument("--save", default="")
    p.add_argument("--max-frames", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config()
    _check_weights(config)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", args.source)
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipeline = CombinedDetectionPipeline(config=config)
    zone_manager = _build_demo_zones(frame_w, frame_h)
    engine = ZoneEngine(zone_manager)

    writer = None
    if args.save:
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_src, (frame_w, frame_h))
        logger.info("Saving annotated video to %s", args.save)

    frame_idx = 0
    all_violations = []
    zone_list = zone_manager.get_all_zones()
    dwell_status: dict = {}

    logger.info("Starting zone demo. Press Ctrl+C to stop.")
    try:
        while frame_idx < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            tracked = pipeline.process_frame(frame, camera_id="demo", frame_id=frame_idx)
            violations = engine.evaluate(tracked)

            for track in tracked.tracks:
                for zone in zone_list:
                    pos = engine.get_person_position(track.bbox, "bottom_center")
                    if engine.is_point_in_zone(pos, zone):
                        key = (track.track_id, zone.zone_id)
                        if key not in dwell_status:
                            dwell_status[key] = frame_idx
                            print(
                                f"[Frame {frame_idx:04d}] Track ID {track.track_id} "
                                f"entered Zone '{zone.name}' ({zone.zone_type.value})"
                            )
                        elapsed = frame_idx - dwell_status[key]
                        if elapsed > 0 and elapsed % 30 == 0:
                            print(
                                f"[Frame {frame_idx:04d}] Track ID {track.track_id} "
                                f"— Dwell in '{zone.name}': {elapsed} frames"
                            )

            for vio in violations:
                all_violations.append(vio)
                print(
                    f"[Frame {frame_idx:04d}] \u26a0  VIOLATION: {vio.rule_type.value.upper()} "
                    f"— {vio.details}"
                )

            if writer is not None:
                annotated = draw_tracked_frame(frame, tracked)
                annotated = draw_zones(annotated, zone_list, active_only=False)
                if violations:
                    annotated = draw_violations(annotated, violations)
                writer.write(annotated)

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()
        if writer:
            writer.release()

    by_type: dict = {}
    for v in all_violations:
        by_type[v.rule_type.value] = by_type.get(v.rule_type.value, 0) + 1

    print(f"\n{'═'*52}")
    print("  === Zone Engine Summary ===")
    print(f"  Frames processed  : {frame_idx}")
    print(f"  Total violations  : {len(all_violations)}")
    print("  By rule type:")
    for rt, count in sorted(by_type.items()):
        print(f"    {rt:<22}: {count}")
    print(f"{'═'*52}\n")


if __name__ == "__main__":
    main()
