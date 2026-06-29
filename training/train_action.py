"""Training script for the keypoint LSTM action classifier.

Usage:
    python training/train_action.py \\
        --dataset_dir training/datasets/processed/ \\
        --output_path models/action_classifier_v1.pt \\
        --epochs 50 \\
        --batch_size 32 \\
        --lr 0.001

Dataset format:
    training/datasets/processed/
    ├── train/
    │   ├── sequences.npy    # (N, T, 17, 3) float32
    │   └── labels.npy       # (N,) int64 — ActionLabel values (raw int)
    ├── val/
    │   ├── sequences.npy
    │   └── labels.npy
    └── class_distribution.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.common.logger import get_logger
from src.recognition.action_classes import ALL_ACTION_LABELS, LABEL_INDEX, NUM_CLASSES
from src.recognition.keypoint_lstm import KeypointLSTM
from src.recognition.keypoint_preprocessor import KeypointPreprocessor

logger = get_logger("training.train_action")


class ActionDataset(Dataset):
    """PyTorch dataset loading keypoint sequences from numpy files.

    Args:
        sequences: Array of shape ``(N, T, 17, 3)``.
        labels: Array of shape ``(N,)`` with raw ActionLabel int values.
        preprocessor: Preprocessor to apply to each sequence.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        preprocessor: KeypointPreprocessor,
    ) -> None:
        self.sequences = sequences
        self.labels = labels
        self.preprocessor = preprocessor
        self._label_to_idx = {int(lbl): i for lbl, i in LABEL_INDEX.items()}

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq_tensor = self.preprocessor.preprocess(self.sequences[idx])
        raw_label = int(self.labels[idx])
        class_idx = self._label_to_idx.get(raw_label, 0)
        return seq_tensor, torch.tensor(class_idx, dtype=torch.long)


def _load_split(split_dir: str, preprocessor: KeypointPreprocessor) -> ActionDataset:
    seq_path = os.path.join(split_dir, "sequences.npy")
    lbl_path = os.path.join(split_dir, "labels.npy")
    seqs = np.load(seq_path)
    lbls = np.load(lbl_path)
    return ActionDataset(seqs, lbls, preprocessor)


def _compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    label_to_idx = {int(lbl): i for lbl, i in LABEL_INDEX.items()}
    mapped = np.array([label_to_idx.get(int(l), 0) for l in labels])
    counts = np.bincount(mapped, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def _f1_per_class(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum()
        fp = ((preds == c) & (targets != c)).sum()
        fn = ((preds != c) & (targets == c)).sum()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1s.append(float(f1))
    return np.array(f1s)


def train(
    dataset_dir: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    device_str: str = "cpu",
    patience: int = 10,
) -> None:
    """Run the full training loop.

    Args:
        dataset_dir: Root directory with train/ and val/ subdirectories.
        output_path: Where to save the best model checkpoint.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate.
        hidden_size: LSTM hidden units.
        num_layers: Stacked LSTM layers.
        dropout: Dropout probability.
        device_str: PyTorch device string.
        patience: Early stopping patience (epochs without val F1 improvement).
    """
    device = torch.device(device_str)
    preprocessor = KeypointPreprocessor(config={})

    train_ds = _load_split(os.path.join(dataset_dir, "train"), preprocessor)
    val_ds = _load_split(os.path.join(dataset_dir, "val"), preprocessor)
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    class_weights = _compute_class_weights(train_ds.labels, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = KeypointLSTM(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=NUM_CLASSES,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_f1 = -1.0
    no_improve = 0
    history = []

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(x)
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_targets.append(y.cpu().numpy())
        val_loss /= len(val_ds)
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        val_acc = float((preds == targets).mean())
        f1s = _f1_per_class(preds, targets, NUM_CLASSES)
        mean_f1 = float(f1s.mean())

        scheduler.step(val_loss)
        logger.info(
            "Epoch %3d | train_loss=%.4f val_loss=%.4f val_acc=%.3f mean_f1=%.3f",
            epoch, train_loss, val_loss, val_acc, mean_f1,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "mean_f1": mean_f1,
            "f1_per_class": f1s.tolist(),
        })

        if mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_f1": mean_f1,
                    "model_info": model.get_model_info(),
                },
                output_path,
            )
            logger.info("Best model saved (val_f1=%.4f).", mean_f1)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    results_dir = PROJECT_ROOT / "training" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    hist_path = results_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", hist_path)
    logger.info("Training complete. Best val_f1=%.4f  Model saved to %s", best_val_f1, output_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train keypoint LSTM action classifier")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--output_path", default="models/action_classifier_v1.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--patience", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device_str=args.device,
        patience=args.patience,
    )
