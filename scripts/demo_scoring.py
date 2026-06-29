"""
Demo: Full pipeline — detection + pose + tracking + action + zones + scoring.

Shows the complete decision-making process including hysteresis, cooldown,
and instant weapon alerts.

Usage:
    python scripts/demo_scoring.py [--source <path_or_index>] [--max-frames N]
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
from src.detection.combined_pipeline import CombinedDetectionPipeline
from src.recognition.recognition_pipeline import ActionRecognitionPipeline
from src.scoring.anomaly_scorer import AnomalyScorer
from src.scoring.scoring_models import AlertDecision
from src.scoring.scoring_pipeline import ScoringPipeline
from src.scoring.zone_engine import ZoneEngine
from src.scoring.zone_manager import ZoneManager
from src.scoring.zone_models import Rule, RuleType, Zone, ZoneType

logger = get_logger("demo.scoring")


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
    zm = ZoneManager()
    z1 = Zone(
        zone_id="restricted_area",
        camera_id="demo",
        name="Restricted Area",
        zone_type=ZoneType.RESTRICTED,
        polygon=[
            (int(frame_w * 0.60), int(frame_h * 0.05)),
            (int(frame_w * 0.98), int(frame_h * 0.05)),
            (int(frame_w * 0.98), int(frame_h * 0.50)),
            (int(frame_w * 0.60), int(frame_h * 0.50)),
        ],
    )
    zm.add_zone(z1)
    zm.add_rule(Rule("r1_noe", "restricted_area", RuleType.NO_ENTRY, cooldown_seconds=30.0))
    zm.add_rule(Rule("r1_int", "restricted_area", RuleType.INTRUSION, cooldown_seconds=0.0))
    zm.add_rule(Rule("r1_loi", "restricted_area", RuleType.LOITERING,
                     max_duration_seconds=20.0, cooldown_seconds=60.0))
    return zm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSS Scoring Demo")
    p.add_argument("--source", default="0")
    p.add_argument("--max-frames", type=int, default=200)
    return p.parse_args()


def _fmt_signals(signals) -> str:
    parts = []
    for s in signals:
        parts.append(f"{s.signal_type.value.split('_')[0]}={s.value:.2f}×{s.weight:.2f}={s.weighted_value:.3f}")
    return " | ".join(parts) if parts else "none"


def main() -> None:
    args = _parse_args()
    config = load_config()
    _check_weights(config)

    # Lower threshold to trigger more alerts in demo
    config["scoring"]["alert_threshold"] = 0.10
    config["scoring"]["hysteresis_count"] = 1

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", args.source)
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det_pipeline = CombinedDetectionPipeline(config=config)
    action_pipeline = ActionRecognitionPipeline(config=config)
    zone_manager = _build_demo_zones(frame_w, frame_h)
    zone_engine = ZoneEngine(zone_manager)
    scorer = AnomalyScorer(config)
    scoring_pipeline = ScoringPipeline(config, zone_engine, action_pipeline, scorer)

    frame_idx = 0
    total_alerts = 0
    total_suppressed = 0
    total_escalated = 0
    category_counts: dict = {}

    logger.info("Starting scoring demo. Press Ctrl+C to stop.")
    try:
        while frame_idx < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            tracked = det_pipeline.process_frame(frame, camera_id="demo", frame_id=frame_idx)
            action_preds = action_pipeline.process(tracked)
            zone_viols = zone_engine.evaluate(tracked)
            scored_events = scoring_pipeline.process(tracked, action_preds, zone_viols)

            actionable = [e for e in scored_events
                          if e.alert_decision in (AlertDecision.ALERT, AlertDecision.ESCALATED,
                                                   AlertDecision.SUPPRESSED)]

            if actionable:
                print(f"\n[Frame {frame_idx:04d}] === Scoring Results ===")
                for ev in scored_events:
                    if ev.alert_decision == AlertDecision.NO_ALERT:
                        if ev.track_id is not None:
                            print(f"  Track {ev.track_id}: score={ev.severity_score:.3f} → NO_ALERT"
                                  f" ({ev.event_category}/{ev.event_label})")
                        continue
                    symbol = "⚠️ " if ev.alert_decision == AlertDecision.ESCALATED else ""
                    print(f"  {symbol}Track {ev.track_id}: score={ev.severity_score:.3f}"
                          f" → {ev.alert_decision.value.upper()}"
                          f" ({ev.event_category}/{ev.event_label})")
                    sigs_str = _fmt_signals(ev.contributing_signals)
                    if sigs_str:
                        print(f"    Signals: {sigs_str}")
                    if ev.suppression_reason:
                        print(f"    Suppressed: {ev.suppression_reason}")
                    if ev.zone_name:
                        print(f"    Zone: {ev.zone_name}")

            for ev in scored_events:
                if ev.alert_decision == AlertDecision.ALERT:
                    total_alerts += 1
                elif ev.alert_decision == AlertDecision.SUPPRESSED:
                    total_suppressed += 1
                elif ev.alert_decision == AlertDecision.ESCALATED:
                    total_escalated += 1
                cat = ev.event_category
                category_counts[cat] = category_counts.get(cat, 0) + 1

            # Check instant weapon alerts specifically
            for ev in scored_events:
                if ev.event_category == "weapon":
                    print(f"\n[Frame {frame_idx:04d}] === INSTANT ALERT ===")
                    print(f"  ⚠️  WEAPON DETECTED: {ev.event_label}")
                    print(f"  Score: {ev.severity_score:.2f} → {ev.alert_decision.value.upper()}")

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()

    stats = scorer.get_stats()
    total_scored = stats["total_events_scored"]

    print(f"\n{'═'*52}")
    print("  === Scoring Summary ===")
    print(f"  Frames processed     : {frame_idx}")
    print(f"  Total events scored  : {total_scored}")
    print(f"  Alerts fired         : {total_alerts}")
    print(f"  Alerts suppressed    : {total_suppressed}")
    print(f"  Alerts escalated     : {total_escalated}")
    print("  By category          :")
    for cat, count in sorted(category_counts.items()):
        print(f"    {cat:<20}: {count}")
    cam_avgs = stats.get("avg_score_by_camera", {})
    if cam_avgs:
        for cam, avg in cam_avgs.items():
            print(f"  Avg score ({cam})   : {avg:.4f}")
    print(f"{'═'*52}\n")


if __name__ == "__main__":
    main()
