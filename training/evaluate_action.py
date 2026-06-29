"""Evaluate trained action classifier on a test set.

Usage:
    python training/evaluate_action.py \\
        --model_path models/action_classifier_v1.pt \\
        --test_dir training/datasets/processed/test/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.logger import get_logger
from src.recognition.action_classes import ALL_ACTION_LABELS, LABEL_INDEX, NUM_CLASSES
from src.recognition.keypoint_lstm import KeypointLSTM
from src.recognition.keypoint_preprocessor import KeypointPreprocessor
from training.train_action import ActionDataset, _load_split

logger = get_logger("training.evaluate_action")


def evaluate(model_path: str, test_dir: str, batch_size: int = 32, device_str: str = "cpu") -> dict:
    """Evaluate model on a test split.

    Args:
        model_path: Path to saved ``.pt`` checkpoint.
        test_dir: Directory containing ``sequences.npy`` and ``labels.npy``.
        batch_size: Evaluation batch size.
        device_str: PyTorch device string.

    Returns:
        Dict with accuracy, per-class precision/recall/F1, and confusion matrix.
    """
    device = torch.device(device_str)
    preprocessor = KeypointPreprocessor(config={})
    test_ds = _load_split(test_dir, preprocessor)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    info = checkpoint.get("model_info", {})
    model = KeypointLSTM(
        hidden_size=info.get("hidden_size", 128),
        num_layers=info.get("num_layers", 2),
        num_classes=NUM_CLASSES,
    ).to(device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    accuracy = float((preds == targets).mean())

    label_names = [lbl.name for lbl in ALL_ACTION_LABELS]
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(targets, preds):
        confusion[t, p] += 1

    per_class = {}
    for c in range(NUM_CLASSES):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum()) - tp
        fn = int(confusion[c, :].sum()) - tp
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        per_class[label_names[c]] = {"precision": precision, "recall": recall, "f1": f1, "support": int(confusion[c].sum())}

    result = {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "label_names": label_names,
    }

    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(f"{'Label':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
    print("-" * 50)
    for name, m in per_class.items():
        print(f"{name:<20} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['support']:6d}")

    results_dir = PROJECT_ROOT / "training" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Evaluation report saved to %s", out_path)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    evaluate(args.model_path, args.test_dir, args.batch_size, args.device)
