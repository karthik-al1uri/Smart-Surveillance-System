"""Action recognition classifier.

Wraps the LSTM model with pre/post-processing for production use.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.common.config import get_project_root, load_config
from src.common.logger import get_logger
from src.recognition.action_classes import (
    ALL_ACTION_LABELS,
    INDEX_TO_LABEL,
    LABEL_INDEX,
    LABEL_TO_CATEGORY,
    NUM_CLASSES,
    ActionCategory,
    ActionLabel,
    ActionPrediction,
)
from src.recognition.keypoint_lstm import KeypointLSTM
from src.recognition.keypoint_preprocessor import KeypointPreprocessor
from src.recognition.sliding_window import WindowData

logger = get_logger("recognition.classifier")


class ActionClassifier:
    """Wraps the KeypointLSTM model for inference.

    On construction, attempts to load weights from ``model_path``.  If the
    file does not exist, the model is initialised with random weights and a
    WARNING is logged.

    Args:
        config: Optional pre-loaded config dict.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or load_config()
        ar = cfg.get("action_recognition", {})
        root = get_project_root()

        model_path_rel = ar.get("model_path", "models/action_classifier_v1.pt")
        self._model_path = str(root / model_path_rel)
        self._device = torch.device(ar.get("device", "cpu"))
        self._num_classes: int = ar.get("num_classes", NUM_CLASSES)
        self._hidden_size: int = ar.get("hidden_size", 128)
        self._num_layers: int = ar.get("num_layers", 2)
        self._dropout: float = ar.get("dropout", 0.3)
        self._bidirectional: bool = ar.get("bidirectional", True)
        self._conf_threshold: float = ar.get("confidence_threshold", 0.4)
        self._window_size: int = ar.get("window_size", 16)

        self._preprocessor = KeypointPreprocessor(config=cfg)

        self._model = KeypointLSTM(
            input_size=51,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            num_classes=self._num_classes,
            dropout=self._dropout,
            bidirectional=self._bidirectional,
        ).to(self._device)

        self.load_model(self._model_path)
        self._model.eval()

    def load_model(self, model_path: str) -> None:
        """Load trained weights from disk.

        If the file does not exist, a WARNING is logged and random weights
        are retained — the pipeline still works but predictions are random.

        Args:
            model_path: Absolute or relative path to the ``.pt`` file.
        """
        if not os.path.exists(model_path):
            logger.warning(
                "Model file not found at '%s'. Using untrained model — predictions "
                "will be random. Train with: python training/train_action.py",
                model_path,
            )
            return
        state = torch.load(model_path, map_location=self._device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            self._model.load_state_dict(state["model_state_dict"])
        else:
            self._model.load_state_dict(state)
        logger.info("Action classifier weights loaded from '%s'.", model_path)

    def classify(self, window: WindowData) -> ActionPrediction:
        """Classify one temporal window.

        Args:
            window: :class:`~src.recognition.sliding_window.WindowData` instance.

        Returns:
            :class:`~src.recognition.action_classes.ActionPrediction`.
        """
        return self.classify_batch([window])[0]

    def classify_batch(self, windows: List[WindowData]) -> List[ActionPrediction]:
        """Classify a batch of windows.

        Args:
            windows: List of :class:`~src.recognition.sliding_window.WindowData`.

        Returns:
            List of :class:`~src.recognition.action_classes.ActionPrediction`,
            one per window.
        """
        if not windows:
            return []

        tensors = [
            self._preprocessor.preprocess(w.keypoint_sequence)
            for w in windows
        ]
        batch = torch.stack(tensors).to(self._device)

        with torch.no_grad():
            logits = self._model(batch)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        predictions: List[ActionPrediction] = []
        for i, window in enumerate(windows):
            class_probs = probs[i]
            top_idx = int(class_probs.argmax())
            top_conf = float(class_probs[top_idx])
            top_label = INDEX_TO_LABEL[top_idx]
            top_category = LABEL_TO_CATEGORY[top_label]

            if top_label != ActionLabel.STANDING and top_conf < self._conf_threshold:
                top_label = ActionLabel.STANDING
                top_category = ActionCategory.NORMAL

            category_probs: Dict[ActionCategory, float] = {cat: 0.0 for cat in ActionCategory}
            for j, lbl in enumerate(ALL_ACTION_LABELS):
                cat = LABEL_TO_CATEGORY[lbl]
                category_probs[cat] = category_probs.get(cat, 0.0) + float(class_probs[j])

            predictions.append(
                ActionPrediction(
                    track_id=window.track_id,
                    camera_id="default",
                    timestamp=time.time(),
                    category=top_category,
                    label=top_label,
                    confidence=top_conf,
                    category_probabilities=category_probs,
                    window_start_frame=window.start_frame,
                    window_end_frame=window.end_frame,
                    keypoint_quality=window.avg_keypoint_confidence,
                )
            )
        return predictions

    def create_dummy_model(self, save_path: str) -> None:
        """Save the current (randomly initialised) model weights to disk.

        This allows the full pipeline to run end-to-end before any real
        training has taken place.

        Args:
            save_path: Path where the ``.pt`` file will be written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "model_info": self._model.get_model_info(),
                "is_dummy": True,
                "note": "Randomly initialised — train with training/train_action.py",
            },
            save_path,
        )
        logger.info("Dummy model saved to '%s'.", save_path)

    def get_model_info(self) -> dict:
        """Return model metadata including parameter count.

        Returns:
            Dict with architecture hyperparameters and parameter counts.
        """
        info = self._model.get_model_info()
        info["model_path"] = self._model_path
        info["device"] = str(self._device)
        info["confidence_threshold"] = self._conf_threshold
        return info
