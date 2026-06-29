"""Action recognition class definitions and mappings.

Central registry of all activity categories the system can detect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional


class ActionCategory(IntEnum):
    """Top-level activity categories."""

    NORMAL = 0
    VIOLENT = 1
    SUSPICIOUS = 2
    URGENT = 3


class ActionLabel(IntEnum):
    """Fine-grained activity labels."""

    STANDING = 0
    WALKING = 1
    SITTING = 2
    RUNNING = 3

    FIGHTING = 10
    PUNCHING = 11
    KICKING = 12
    WEAPON_USE = 13

    LOITERING = 20
    INTRUSION = 21
    ABANDONED_OBJECT = 22
    TRESPASSING = 23

    FALLING = 30
    COLLAPSE = 31
    LYING_DOWN = 32


LABEL_TO_CATEGORY: Dict[ActionLabel, ActionCategory] = {
    ActionLabel.STANDING: ActionCategory.NORMAL,
    ActionLabel.WALKING: ActionCategory.NORMAL,
    ActionLabel.SITTING: ActionCategory.NORMAL,
    ActionLabel.RUNNING: ActionCategory.NORMAL,
    ActionLabel.FIGHTING: ActionCategory.VIOLENT,
    ActionLabel.PUNCHING: ActionCategory.VIOLENT,
    ActionLabel.KICKING: ActionCategory.VIOLENT,
    ActionLabel.WEAPON_USE: ActionCategory.VIOLENT,
    ActionLabel.LOITERING: ActionCategory.SUSPICIOUS,
    ActionLabel.INTRUSION: ActionCategory.SUSPICIOUS,
    ActionLabel.ABANDONED_OBJECT: ActionCategory.SUSPICIOUS,
    ActionLabel.TRESPASSING: ActionCategory.SUSPICIOUS,
    ActionLabel.FALLING: ActionCategory.URGENT,
    ActionLabel.COLLAPSE: ActionCategory.URGENT,
    ActionLabel.LYING_DOWN: ActionCategory.URGENT,
}

ALL_ACTION_LABELS = list(ActionLabel)
NUM_CLASSES = len(ALL_ACTION_LABELS)

LABEL_INDEX: Dict[ActionLabel, int] = {lbl: i for i, lbl in enumerate(ALL_ACTION_LABELS)}
INDEX_TO_LABEL: Dict[int, ActionLabel] = {i: lbl for lbl, i in LABEL_INDEX.items()}


@dataclass
class ActionPrediction:
    """Output of the action recognition model for a single temporal window.

    Attributes:
        track_id: ID of the tracked person.
        camera_id: Source camera.
        timestamp: Unix timestamp of the prediction.
        category: Top-level category (Normal/Violent/Suspicious/Urgent).
        label: Fine-grained action label.
        confidence: Confidence of the top predicted class (0–1).
        category_probabilities: Probability per :class:`ActionCategory`.
        window_start_frame: First frame index of the temporal window.
        window_end_frame: Last frame index of the temporal window.
        keypoint_quality: Average keypoint confidence across the window.
    """

    track_id: int
    camera_id: str
    timestamp: float
    category: ActionCategory
    label: ActionLabel
    confidence: float
    category_probabilities: Dict[ActionCategory, float]
    window_start_frame: int
    window_end_frame: int
    keypoint_quality: float = 0.0
