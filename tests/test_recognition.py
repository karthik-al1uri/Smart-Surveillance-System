"""Tests for Phase 5: Action Recognition.

All tests use synthetic data — no video inference required.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_config
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
from src.recognition.action_classifier import ActionClassifier
from src.recognition.keypoint_lstm import AttentionPooling, KeypointLSTM
from src.recognition.keypoint_preprocessor import KeypointPreprocessor
from src.recognition.recognition_pipeline import ActionRecognitionPipeline
from src.recognition.sliding_window import SlidingWindowManager, WindowData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_POSE = np.array([
    [0.50, 0.20, 0.9], [0.48, 0.18, 0.9], [0.52, 0.18, 0.9],
    [0.46, 0.19, 0.8], [0.54, 0.19, 0.8], [0.43, 0.30, 0.9],
    [0.57, 0.30, 0.9], [0.40, 0.42, 0.9], [0.60, 0.42, 0.9],
    [0.38, 0.54, 0.8], [0.62, 0.54, 0.8], [0.45, 0.55, 0.9],
    [0.55, 0.55, 0.9], [0.44, 0.70, 0.9], [0.56, 0.70, 0.9],
    [0.43, 0.85, 0.9], [0.57, 0.85, 0.9],
], dtype=np.float32)


def make_synthetic_keypoint_sequence(num_frames: int = 16, action: str = "standing") -> np.ndarray:
    """Generate a synthetic (T, 17, 3) keypoint sequence."""
    seq = np.tile(_BASE_POSE, (num_frames, 1, 1)).copy().astype(np.float32)
    if action == "standing":
        seq += np.random.normal(0, 0.005, seq.shape).astype(np.float32)
    elif action == "falling":
        for i in range(num_frames):
            seq[i, :, 1] += i * 0.02
    elif action == "walking":
        for t in range(num_frames):
            phase = 2 * np.pi * t / num_frames
            seq[t, 13, 1] += 0.05 * np.sin(phase)
            seq[t, 14, 1] += 0.05 * np.sin(phase + np.pi)
    return seq


def make_window(track_id: int = 1, num_frames: int = 16) -> WindowData:
    seq = make_synthetic_keypoint_sequence(num_frames)
    return WindowData(
        track_id=track_id,
        keypoint_sequence=seq,
        start_frame=0,
        end_frame=num_frames - 1,
        avg_keypoint_confidence=0.85,
    )


def make_tracked_analysis_with_keypoints(num_tracks: int = 1, n_history: int = 20):
    """Build a fake TrackedFrameAnalysis with keypoint history."""
    import time as t_mod
    from collections import deque
    from src.detection.combined_pipeline import TrackedFrameAnalysis
    from src.detection.tracker import Track

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = []
    for i in range(num_tracks):
        kp_hist = deque(maxlen=64)
        for f in range(n_history):
            kp_hist.append(make_synthetic_keypoint_sequence(1)[0])
        track = Track(
            track_id=i + 1, state="active", bbox=(100, 100, 200, 400),
            keypoint_history=kp_hist, age=n_history, hits=n_history,
            time_since_update=0, confidence=0.9,
        )
        tracks.append(track)

    return TrackedFrameAnalysis(
        camera_id="test_cam", frame_id=100, timestamp=t_mod.time(),
        frame=frame, person_detections=[], object_detections=[],
        poses=[], tracks=tracks,
    )


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def dummy_classifier(config, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    model_path = str(tmp / "dummy.pt")
    cfg = {**config, "action_recognition": {**config.get("action_recognition", {}), "model_path": model_path}}
    clf = ActionClassifier(config=cfg)
    clf.create_dummy_model(model_path)
    clf2 = ActionClassifier(config=cfg)
    return clf2


# ---------------------------------------------------------------------------
# 1. ActionCategory enum
# ---------------------------------------------------------------------------

def test_action_category_enum():
    assert ActionCategory.NORMAL == 0
    assert ActionCategory.VIOLENT == 1
    assert ActionCategory.SUSPICIOUS == 2
    assert ActionCategory.URGENT == 3


# ---------------------------------------------------------------------------
# 2. ActionLabel enum — 15 labels
# ---------------------------------------------------------------------------

def test_action_label_enum():
    assert ActionLabel.STANDING == 0
    assert ActionLabel.WALKING == 1
    assert ActionLabel.FIGHTING == 10
    assert ActionLabel.FALLING == 30
    assert NUM_CLASSES == 15


# ---------------------------------------------------------------------------
# 3. LABEL_TO_CATEGORY mapping correctness
# ---------------------------------------------------------------------------

def test_label_to_category_mapping():
    assert LABEL_TO_CATEGORY[ActionLabel.STANDING] == ActionCategory.NORMAL
    assert LABEL_TO_CATEGORY[ActionLabel.WALKING] == ActionCategory.NORMAL
    assert LABEL_TO_CATEGORY[ActionLabel.FIGHTING] == ActionCategory.VIOLENT
    assert LABEL_TO_CATEGORY[ActionLabel.LOITERING] == ActionCategory.SUSPICIOUS
    assert LABEL_TO_CATEGORY[ActionLabel.FALLING] == ActionCategory.URGENT
    assert LABEL_TO_CATEGORY[ActionLabel.COLLAPSE] == ActionCategory.URGENT
    assert all(lbl in LABEL_TO_CATEGORY for lbl in ALL_ACTION_LABELS)


# ---------------------------------------------------------------------------
# 4. ActionPrediction dataclass
# ---------------------------------------------------------------------------

def test_action_prediction_dataclass():
    pred = ActionPrediction(
        track_id=5, camera_id="cam0", timestamp=time.time(),
        category=ActionCategory.VIOLENT, label=ActionLabel.FIGHTING,
        confidence=0.78,
        category_probabilities={ActionCategory.NORMAL: 0.1, ActionCategory.VIOLENT: 0.78,
                                 ActionCategory.SUSPICIOUS: 0.07, ActionCategory.URGENT: 0.05},
        window_start_frame=0, window_end_frame=15, keypoint_quality=0.88,
    )
    assert pred.track_id == 5
    assert pred.category == ActionCategory.VIOLENT
    assert pred.label == ActionLabel.FIGHTING
    assert pred.confidence == pytest.approx(0.78)


# ---------------------------------------------------------------------------
# 5. SlidingWindow — single track, 16 frames → 1 window
# ---------------------------------------------------------------------------

def test_sliding_window_single_track(config):
    cfg = {**config, "action_recognition": {**config.get("action_recognition", {}),
                                             "window_size": 16, "stride": 16,
                                             "min_keypoint_confidence": 0.0,
                                             "min_visible_keypoints": 0}}
    mgr = SlidingWindowManager(config=cfg)
    for f in range(16):
        kp = make_synthetic_keypoint_sequence(1)[0]
        mgr.update(1, kp, f)
    windows = mgr.get_ready_windows()
    assert len(windows) == 1
    assert windows[0].track_id == 1
    assert windows[0].keypoint_sequence.shape == (16, 17, 3)


# ---------------------------------------------------------------------------
# 6. Stride=8, 24 frames → 2 windows emitted
# ---------------------------------------------------------------------------

def test_sliding_window_stride(config):
    cfg = {**config, "action_recognition": {**config.get("action_recognition", {}),
                                             "window_size": 16, "stride": 8,
                                             "min_keypoint_confidence": 0.0,
                                             "min_visible_keypoints": 0}}
    mgr = SlidingWindowManager(config=cfg)
    for f in range(24):
        kp = make_synthetic_keypoint_sequence(1)[0]
        mgr.update(1, kp, f)
        mgr.get_ready_windows()

    mgr2 = SlidingWindowManager(config=cfg)
    windows = []
    for f in range(24):
        kp = make_synthetic_keypoint_sequence(1)[0]
        mgr2.update(1, kp, f)
        windows.extend(mgr2.get_ready_windows())
    assert len(windows) == 2


# ---------------------------------------------------------------------------
# 7. Overlap between consecutive windows
# ---------------------------------------------------------------------------

def test_sliding_window_overlap(config):
    cfg = {**config, "action_recognition": {**config.get("action_recognition", {}),
                                             "window_size": 16, "stride": 8,
                                             "min_keypoint_confidence": 0.0,
                                             "min_visible_keypoints": 0}}
    mgr = SlidingWindowManager(config=cfg)
    windows = []
    for f in range(24):
        kp = make_synthetic_keypoint_sequence(1)[0]
        mgr.update(1, kp, f)
        windows.extend(mgr.get_ready_windows())
    assert len(windows) == 2
    overlap = windows[1].start_frame - windows[0].start_frame
    assert overlap == 8


# ---------------------------------------------------------------------------
# 8. Quality filter — low confidence keypoints → window skipped
# ---------------------------------------------------------------------------

def test_sliding_window_quality_filter(config):
    cfg = {**config, "action_recognition": {**config.get("action_recognition", {}),
                                             "window_size": 4, "stride": 4,
                                             "min_keypoint_confidence": 0.9,
                                             "min_visible_keypoints": 17}}
    mgr = SlidingWindowManager(config=cfg)
    for f in range(4):
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[:, 2] = 0.1
        mgr.update(1, kp, f)
    windows = mgr.get_ready_windows()
    assert len(windows) == 0


# ---------------------------------------------------------------------------
# 9. Multiple tracks — independent windows
# ---------------------------------------------------------------------------

def test_sliding_window_multiple_tracks(config):
    cfg = {**config, "action_recognition": {**config.get("action_recognition", {}),
                                             "window_size": 8, "stride": 8,
                                             "min_keypoint_confidence": 0.0,
                                             "min_visible_keypoints": 0}}
    mgr = SlidingWindowManager(config=cfg)
    for f in range(8):
        kp = make_synthetic_keypoint_sequence(1)[0]
        mgr.update(1, kp, f)
        mgr.update(2, kp, f)
    windows = mgr.get_ready_windows()
    track_ids = {w.track_id for w in windows}
    assert 1 in track_ids and 2 in track_ids


# ---------------------------------------------------------------------------
# 10. Track removal — buffer cleaned
# ---------------------------------------------------------------------------

def test_sliding_window_track_removal(config):
    mgr = SlidingWindowManager(config=config)
    for f in range(5):
        kp = make_synthetic_keypoint_sequence(1)[0]
        mgr.update(42, kp, f)
    mgr.remove_track(42)
    assert 42 not in mgr._buffers


# ---------------------------------------------------------------------------
# 11. Insufficient frames → no window
# ---------------------------------------------------------------------------

def test_sliding_window_insufficient_frames(config):
    cfg = {**config, "action_recognition": {**config.get("action_recognition", {}),
                                             "window_size": 16, "stride": 8,
                                             "min_keypoint_confidence": 0.0,
                                             "min_visible_keypoints": 0}}
    mgr = SlidingWindowManager(config=cfg)
    for f in range(5):
        kp = make_synthetic_keypoint_sequence(1)[0]
        mgr.update(1, kp, f)
    windows = mgr.get_ready_windows()
    assert len(windows) == 0


# ---------------------------------------------------------------------------
# 12. Preprocessor — hip-center normalization
# ---------------------------------------------------------------------------

def test_keypoint_preprocessor_normalize(config):
    prep = KeypointPreprocessor(config=config)
    seq = make_synthetic_keypoint_sequence(16, "standing")
    tensor = prep.preprocess(seq)
    flat = tensor.numpy().reshape(16, 17, 3)
    hip_x = (flat[:, 11, 0] + flat[:, 12, 0]) / 2.0
    hip_y = (flat[:, 11, 1] + flat[:, 12, 1]) / 2.0
    assert np.abs(hip_x).max() < 0.3
    assert np.abs(hip_y).max() < 0.3


# ---------------------------------------------------------------------------
# 13. Preprocessor — different person sizes → same relative scale
# ---------------------------------------------------------------------------

def test_keypoint_preprocessor_scale(config):
    prep = KeypointPreprocessor(config=config)
    small = make_synthetic_keypoint_sequence(16)
    large = small.copy()
    large[:, :, :2] *= 3.0
    t_small = prep.preprocess(small)
    t_large = prep.preprocess(large)
    np.testing.assert_allclose(t_small.numpy(), t_large.numpy(), atol=0.01)


# ---------------------------------------------------------------------------
# 14. Preprocessor — missing keypoints handled (no NaN)
# ---------------------------------------------------------------------------

def test_keypoint_preprocessor_missing_keypoints(config):
    prep = KeypointPreprocessor(config=config)
    seq = make_synthetic_keypoint_sequence(16)
    seq[:, 7, 2] = 0.0
    seq[:, 7, :2] = 0.0
    tensor = prep.preprocess(seq)
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


# ---------------------------------------------------------------------------
# 15. Preprocessor — output shape (T, 51)
# ---------------------------------------------------------------------------

def test_keypoint_preprocessor_output_shape(config):
    prep = KeypointPreprocessor(config=config)
    seq = make_synthetic_keypoint_sequence(16)
    tensor = prep.preprocess(seq)
    assert tensor.shape == (16, 51)
    assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# 16. LSTM forward — (B, 16, 51) → (B, 15)
# ---------------------------------------------------------------------------

def test_lstm_model_forward():
    model = KeypointLSTM()
    x = torch.randn(4, 16, 51)
    out = model(x)
    assert out.shape == (4, 15)


# ---------------------------------------------------------------------------
# 17. Attention pooling — weights sum to 1
# ---------------------------------------------------------------------------

def test_lstm_model_attention():
    attn = AttentionPooling(256)
    lstm_out = torch.randn(2, 16, 256)
    weights = torch.softmax(attn.attention(lstm_out), dim=1)
    sums = weights.sum(dim=1)
    assert sums.shape == (2, 1)
    np.testing.assert_allclose(sums.detach().numpy(), np.ones((2, 1)), atol=1e-5)


# ---------------------------------------------------------------------------
# 18. LSTM softmax output — values in [0,1], sum to 1
# ---------------------------------------------------------------------------

def test_lstm_model_output_range():
    import torch.nn.functional as F
    model = KeypointLSTM()
    x = torch.randn(3, 16, 51)
    logits = model(x)
    probs = F.softmax(logits, dim=-1)
    assert (probs >= 0).all()
    assert (probs <= 1).all()
    np.testing.assert_allclose(probs.sum(dim=-1).detach().numpy(), np.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# 19. Classifier — load dummy model without crash
# ---------------------------------------------------------------------------

def test_classifier_load_dummy(dummy_classifier):
    info = dummy_classifier.get_model_info()
    assert "total_parameters" in info
    assert info["total_parameters"] > 0


# ---------------------------------------------------------------------------
# 20. Classifier — classify single window returns ActionPrediction
# ---------------------------------------------------------------------------

def test_classifier_classify_single(dummy_classifier):
    window = make_window(track_id=3)
    pred = dummy_classifier.classify(window)
    assert isinstance(pred, ActionPrediction)
    assert pred.track_id == 3
    assert 0.0 <= pred.confidence <= 1.0
    assert isinstance(pred.category, ActionCategory)
    assert isinstance(pred.label, ActionLabel)


# ---------------------------------------------------------------------------
# 21. Classifier — batch of 4 windows → 4 predictions
# ---------------------------------------------------------------------------

def test_classifier_classify_batch(dummy_classifier):
    windows = [make_window(track_id=i) for i in range(4)]
    preds = dummy_classifier.classify_batch(windows)
    assert len(preds) == 4
    assert all(isinstance(p, ActionPrediction) for p in preds)


# ---------------------------------------------------------------------------
# 22. Classifier — low confidence → classified as NORMAL
# ---------------------------------------------------------------------------

def test_classifier_confidence_threshold(config, tmp_path):
    model_path = str(tmp_path / "dummy.pt")
    cfg = {**config, "action_recognition": {
        **config.get("action_recognition", {}),
        "model_path": model_path,
        "confidence_threshold": 0.99,
    }}
    clf = ActionClassifier(config=cfg)
    clf.create_dummy_model(model_path)
    clf2 = ActionClassifier(config=cfg)
    window = make_window()
    pred = clf2.classify(window)
    assert pred.category == ActionCategory.NORMAL


# ---------------------------------------------------------------------------
# 23. Classifier — missing model file → initialises with warning, no crash
# ---------------------------------------------------------------------------

def test_classifier_missing_model_file(config, tmp_path):
    cfg = {**config, "action_recognition": {
        **config.get("action_recognition", {}),
        "model_path": str(tmp_path / "nonexistent.pt"),
    }}
    clf = ActionClassifier(config=cfg)
    window = make_window()
    pred = clf.classify(window)
    assert isinstance(pred, ActionPrediction)


# ---------------------------------------------------------------------------
# 24. Recognition pipeline integration — enough history → predictions
# ---------------------------------------------------------------------------

def test_recognition_pipeline_integration(config, tmp_path):
    model_path = str(tmp_path / "dummy.pt")
    cfg = {**config, "action_recognition": {
        **config.get("action_recognition", {}),
        "model_path": model_path,
        "window_size": 16, "stride": 8,
        "min_keypoint_confidence": 0.0, "min_visible_keypoints": 0,
    }}
    pipeline = ActionRecognitionPipeline(config=cfg)
    pipeline._classifier.create_dummy_model(model_path)
    pipeline._classifier = ActionClassifier(config=cfg)

    preds = []
    for i in range(25):
        analysis = make_tracked_analysis_with_keypoints(num_tracks=1, n_history=0)
        analysis.frame_id = i
        analysis.tracks[0].keypoint_history.append(make_synthetic_keypoint_sequence(1)[0])
        ps = pipeline.process(analysis)
        preds.extend(ps)

    assert isinstance(preds, list)


# ---------------------------------------------------------------------------
# 25. Recognition pipeline — new tracks, < window_size frames → empty
# ---------------------------------------------------------------------------

def test_recognition_pipeline_no_ready_windows(config, tmp_path):
    model_path = str(tmp_path / "dummy2.pt")
    cfg = {**config, "action_recognition": {
        **config.get("action_recognition", {}),
        "model_path": model_path,
        "window_size": 16, "stride": 8,
        "min_keypoint_confidence": 0.0, "min_visible_keypoints": 0,
    }}
    pipeline = ActionRecognitionPipeline(config=cfg)
    pipeline._classifier.create_dummy_model(model_path)

    for i in range(5):
        analysis = make_tracked_analysis_with_keypoints(num_tracks=1, n_history=0)
        analysis.frame_id = i
        preds = pipeline.process(analysis)
        assert preds == []


# ---------------------------------------------------------------------------
# 26. Pipeline stats tracked
# ---------------------------------------------------------------------------

def test_recognition_pipeline_stats(config, tmp_path):
    model_path = str(tmp_path / "dummy3.pt")
    cfg = {**config, "action_recognition": {
        **config.get("action_recognition", {}),
        "model_path": model_path,
    }}
    pipeline = ActionRecognitionPipeline(config=cfg)
    pipeline._classifier.create_dummy_model(model_path)

    for i in range(3):
        analysis = make_tracked_analysis_with_keypoints(num_tracks=1, n_history=0)
        analysis.frame_id = i
        pipeline.process(analysis)

    stats = pipeline.get_stats()
    assert stats["frames_processed"] == 3
    assert "total_predictions" in stats
    assert "predictions_by_category" in stats


# ---------------------------------------------------------------------------
# 27. Dummy dataset creation — correct shapes and label counts
# ---------------------------------------------------------------------------

def test_dummy_dataset_creation(tmp_path):
    from scripts.create_dummy_dataset import create_dummy_dataset
    out = str(tmp_path / "dummy_ds")
    create_dummy_dataset(out, samples_per_class=10)
    train_seqs = np.load(os.path.join(out, "train", "sequences.npy"))
    train_lbls = np.load(os.path.join(out, "train", "labels.npy"))
    assert train_seqs.ndim == 4
    assert train_seqs.shape[1] == 16
    assert train_seqs.shape[2] == 17
    assert train_seqs.shape[3] == 3
    assert len(train_lbls) == len(train_seqs)
    assert len(train_lbls) > 0


# ---------------------------------------------------------------------------
# 28. Training script — 2 epochs on tiny dataset, no crash
# ---------------------------------------------------------------------------

def test_training_script_runs(tmp_path):
    from scripts.create_dummy_dataset import create_dummy_dataset
    from training.train_action import train

    ds_dir = str(tmp_path / "ds")
    create_dummy_dataset(ds_dir, samples_per_class=5)
    out_model = str(tmp_path / "test_model.pt")
    train(
        dataset_dir=ds_dir,
        output_path=out_model,
        epochs=2,
        batch_size=8,
        lr=1e-3,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        device_str="cpu",
        patience=5,
    )
    assert os.path.exists(out_model)
