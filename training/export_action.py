"""Export trained model to ONNX for production deployment.

Usage:
    python training/export_action.py \\
        --model_path models/action_classifier_v1.pt \\
        --output_path models/action_classifier_v1.onnx \\
        --window_size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.common.logger import get_logger
from src.recognition.action_classes import NUM_CLASSES
from src.recognition.keypoint_lstm import KeypointLSTM

logger = get_logger("training.export_action")


def export_onnx(model_path: str, output_path: str, window_size: int = 16, device_str: str = "cpu") -> None:
    """Export a trained KeypointLSTM to ONNX.

    Args:
        model_path: Path to ``.pt`` checkpoint.
        output_path: Path for the output ``.onnx`` file.
        window_size: Temporal window length (number of frames).
        device_str: PyTorch device string.
    """
    device = torch.device(device_str)
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

    dummy_input = torch.randn(1, window_size, 51, device=device)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["keypoint_sequence"],
        output_names=["action_logits"],
        dynamic_axes={
            "keypoint_sequence": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
        opset_version=17,
    )
    logger.info("Model exported to ONNX: %s", output_path)
    print(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--window_size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    export_onnx(args.model_path, args.output_path, args.window_size, args.device)
