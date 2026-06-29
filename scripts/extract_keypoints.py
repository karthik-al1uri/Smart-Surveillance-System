"""Extract keypoints from video datasets for LSTM training.

Processes video files → runs YOLOv8-Pose → tracks persons →
extracts keypoint windows and saves as numpy arrays.

Usage:
    python scripts/extract_keypoints.py \\
        --input_dir training/datasets/raw/fight/ \\
        --output_dir training/datasets/processed/ \\
        --label 10 \\
        --window_size 16

Expected raw dataset structure::

    training/datasets/raw/
    ├── fight/          → label: FIGHTING (10)
    │   ├── video1.mp4
    │   └── video2.avi
    ├── normal/         → label: STANDING (0) / WALKING (1)
    ├── fall/           → label: FALLING (30)
    └── loitering/      → label: LOITERING (20)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.common.config import load_config
from src.common.logger import get_logger
from src.detection.combined_pipeline import CombinedDetectionPipeline
from src.recognition.sliding_window import SlidingWindowManager

logger = get_logger("scripts.extract_keypoints")

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_from_video(
    video_path: str,
    pipeline: CombinedDetectionPipeline,
    window_mgr: SlidingWindowManager,
) -> List[np.ndarray]:
    """Run pose extraction and tracking on a single video.

    Args:
        video_path: Path to the input video file.
        pipeline: Initialised :class:`CombinedDetectionPipeline`.
        window_mgr: Sliding window manager (cleared before use).

    Returns:
        List of ``(window_size, 17, 3)`` float32 arrays.
    """
    import cv2

    window_mgr._buffers.clear()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []

    frame_id = 0
    all_windows: List[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        analysis = pipeline.process_frame(frame, camera_id="extract", frame_id=frame_id)
        for track in analysis.tracks:
            if track.keypoint_history:
                kp = list(track.keypoint_history)[-1]
                window_mgr.update(track.track_id, kp, frame_id)
        for win in window_mgr.get_ready_windows():
            all_windows.append(win.keypoint_sequence)
        frame_id += 1

    cap.release()
    logger.info("%s → %d windows", video_path, len(all_windows))
    return all_windows


def extract(
    input_dir: str,
    output_dir: str,
    label: int,
    window_size: int = 16,
    stride: int = 8,
) -> None:
    """Extract keypoints from all videos in a directory.

    Args:
        input_dir: Directory containing raw video files.
        output_dir: Directory where numpy arrays will be saved.
        label: Integer ActionLabel value to assign to all windows.
        window_size: Frames per classification window.
        stride: Frames between consecutive windows.
    """
    config = load_config()
    config["action_recognition"] = {
        "window_size": window_size,
        "stride": stride,
        "min_keypoint_confidence": 0.3,
        "min_visible_keypoints": 8,
    }
    pipeline = CombinedDetectionPipeline(config=config)
    window_mgr = SlidingWindowManager(config=config)

    videos = [
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in _VIDEO_EXTS
    ]
    logger.info("Found %d videos in %s", len(videos), input_dir)

    all_seqs: List[np.ndarray] = []
    for v in videos:
        seqs = extract_from_video(str(v), pipeline, window_mgr)
        all_seqs.extend(seqs)

    if not all_seqs:
        logger.warning("No windows extracted from %s — check video files.", input_dir)
        return

    sequences = np.array(all_seqs, dtype=np.float32)
    labels = np.full(len(sequences), label, dtype=np.int64)

    os.makedirs(output_dir, exist_ok=True)
    seq_path = os.path.join(output_dir, "sequences.npy")
    lbl_path = os.path.join(output_dir, "labels.npy")

    if os.path.exists(seq_path):
        old_seqs = np.load(seq_path)
        old_lbls = np.load(lbl_path)
        sequences = np.concatenate([old_seqs, sequences], axis=0)
        labels = np.concatenate([old_lbls, labels], axis=0)

    np.save(seq_path, sequences)
    np.save(lbl_path, labels)
    logger.info("Saved %d sequences → %s", len(sequences), output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract keypoint sequences from videos")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--label", type=int, required=True, help="ActionLabel int value")
    p.add_argument("--window_size", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    args = p.parse_args()
    extract(args.input_dir, args.output_dir, args.label, args.window_size, args.stride)
