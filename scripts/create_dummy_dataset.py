"""Create a small synthetic dataset for pipeline testing.

Generates fake keypoint sequences with known patterns.

Usage:
    python scripts/create_dummy_dataset.py \\
        --output_dir training/datasets/dummy/ \\
        --samples_per_class 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.recognition.action_classes import ActionLabel

WINDOW_SIZE = 16
N_KEYPOINTS = 17

_BASE_POSE = np.array([
    [0.50, 0.20, 0.9],
    [0.48, 0.18, 0.9],
    [0.52, 0.18, 0.9],
    [0.46, 0.19, 0.8],
    [0.54, 0.19, 0.8],
    [0.43, 0.30, 0.9],
    [0.57, 0.30, 0.9],
    [0.40, 0.42, 0.9],
    [0.60, 0.42, 0.9],
    [0.38, 0.54, 0.8],
    [0.62, 0.54, 0.8],
    [0.45, 0.55, 0.9],
    [0.55, 0.55, 0.9],
    [0.44, 0.70, 0.9],
    [0.56, 0.70, 0.9],
    [0.43, 0.85, 0.9],
    [0.57, 0.85, 0.9],
], dtype=np.float32)


def _make_standing(n: int) -> np.ndarray:
    base = np.tile(_BASE_POSE, (WINDOW_SIZE, 1, 1))
    noise = np.random.normal(0, 0.005, (n, WINDOW_SIZE, N_KEYPOINTS, 3)).astype(np.float32)
    noise[:, :, :, 2] = 0.0
    return np.tile(base, (n, 1, 1, 1)) + noise


def _make_walking(n: int) -> np.ndarray:
    seqs = []
    for _ in range(n):
        seq = _BASE_POSE[None].repeat(WINDOW_SIZE, axis=0).copy()
        for t in range(WINDOW_SIZE):
            phase = (2 * np.pi * t) / WINDOW_SIZE
            seq[t, 13, 1] += 0.05 * np.sin(phase)
            seq[t, 14, 1] += 0.05 * np.sin(phase + np.pi)
            seq[t, 15, 1] += 0.04 * np.sin(phase)
            seq[t, 16, 1] += 0.04 * np.sin(phase + np.pi)
        seq += np.random.normal(0, 0.005, seq.shape).astype(np.float32)
        seqs.append(seq)
    return np.array(seqs, dtype=np.float32)


def _make_fighting(n: int) -> np.ndarray:
    seqs = []
    for _ in range(n):
        seq = _BASE_POSE[None].repeat(WINDOW_SIZE, axis=0).copy()
        for t in range(WINDOW_SIZE):
            phase = (2 * np.pi * t) / (WINDOW_SIZE / 2)
            seq[t, 7, 0] += 0.15 * np.sin(phase)
            seq[t, 7, 1] -= 0.10 * np.cos(phase)
            seq[t, 9, 0] += 0.18 * np.sin(phase + 0.5)
            seq[t, 8, 0] -= 0.15 * np.sin(phase)
            seq[t, 10, 0] -= 0.18 * np.sin(phase + 0.5)
        seq += np.random.normal(0, 0.01, seq.shape).astype(np.float32)
        seqs.append(seq)
    return np.array(seqs, dtype=np.float32)


def _make_falling(n: int) -> np.ndarray:
    seqs = []
    for _ in range(n):
        seq = _BASE_POSE[None].repeat(WINDOW_SIZE, axis=0).copy()
        for t in range(WINDOW_SIZE):
            drop = t / (WINDOW_SIZE - 1)
            rotate = drop * (np.pi / 2)
            seq[t, :, 1] += drop * 0.3
            seq[t, :, 0] += np.sin(rotate) * seq[t, :, 1] * 0.2
        seq += np.random.normal(0, 0.008, seq.shape).astype(np.float32)
        seqs.append(seq)
    return np.array(seqs, dtype=np.float32)


def _make_loitering(n: int) -> np.ndarray:
    seqs = []
    for _ in range(n):
        seq = _BASE_POSE[None].repeat(WINDOW_SIZE, axis=0).copy()
        drift = np.cumsum(np.random.normal(0, 0.005, (WINDOW_SIZE, 2)), axis=0).astype(np.float32)
        seq[:, :, :2] += drift[:, None, :]
        seq += np.random.normal(0, 0.003, seq.shape).astype(np.float32)
        seqs.append(seq)
    return np.array(seqs, dtype=np.float32)


_GENERATORS = {
    ActionLabel.STANDING: (_make_standing, 0),
    ActionLabel.WALKING: (_make_walking, 1),
    ActionLabel.FIGHTING: (_make_fighting, 10),
    ActionLabel.FALLING: (_make_falling, 30),
    ActionLabel.LOITERING: (_make_loitering, 20),
}


def create_dummy_dataset(output_dir: str, samples_per_class: int = 50) -> None:
    """Generate and save a synthetic dataset.

    Args:
        output_dir: Root directory (train/val/test subdirs created inside).
        samples_per_class: Number of sequences per action label.
    """
    np.random.seed(42)
    all_seqs: List[np.ndarray] = []
    all_labels: List[int] = []

    for lbl, (gen_fn, raw_val) in _GENERATORS.items():
        seqs = gen_fn(samples_per_class)
        all_seqs.append(seqs)
        all_labels.extend([raw_val] * samples_per_class)

    sequences = np.concatenate(all_seqs, axis=0)
    labels = np.array(all_labels, dtype=np.int64)

    idx = np.random.permutation(len(sequences))
    sequences, labels = sequences[idx], labels[idx]

    n = len(sequences)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": (sequences[:n_train], labels[:n_train]),
        "val": (sequences[n_train:n_train + n_val], labels[n_train:n_train + n_val]),
        "test": (sequences[n_train + n_val:], labels[n_train + n_val:]),
    }

    class_distribution: dict = {}
    for split_name, (seqs, lbls) in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        np.save(os.path.join(split_dir, "sequences.npy"), seqs)
        np.save(os.path.join(split_dir, "labels.npy"), lbls)
        print(f"Saved {split_name}: {len(seqs)} sequences → {split_dir}")

    for lbl, (_, raw_val) in _GENERATORS.items():
        class_distribution[lbl.name] = int((labels == raw_val).sum())
    dist_path = os.path.join(output_dir, "class_distribution.json")
    with open(dist_path, "w") as f:
        json.dump(class_distribution, f, indent=2)
    print(f"\nClass distribution: {class_distribution}")
    print(f"Dataset saved to {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="training/datasets/dummy/")
    p.add_argument("--samples_per_class", type=int, default=50)
    args = p.parse_args()
    create_dummy_dataset(args.output_dir, args.samples_per_class)
