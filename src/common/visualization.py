"""Visualization utilities for drawing detections and poses on frames.

Used by demo scripts and the operator dashboard overlay.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from src.detection.pose_structures import COCO_SKELETON, PoseResult
from src.detection.yolo_detector import Detection

_WEAPON_CLASSES = {"knife", "scissors", "gun"}
_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

_COLOR_PERSON = (0, 255, 0)
_COLOR_WEAPON = (0, 0, 255)
_COLOR_OTHER = (0, 165, 255)
_COLOR_KEYPOINT = (255, 100, 0)
_COLOR_SKELETON = (0, 255, 255)
_COLOR_TEXT_BG = (0, 0, 0)
_COLOR_TEXT_FG = (255, 255, 255)


def _detection_color(det: Detection) -> Tuple[int, int, int]:
    if det.class_name in _WEAPON_CLASSES:
        return _COLOR_WEAPON
    if det.class_name == "person":
        return _COLOR_PERSON
    return _COLOR_OTHER


def _label_strip(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """Draw a text label with a dark background strip for readability."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x, y - th - baseline), (x + tw, y + baseline), _COLOR_TEXT_BG, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    color: Tuple[int, int, int] = _COLOR_PERSON,
) -> np.ndarray:
    """Draw bounding boxes with class labels and confidence scores.

    Args:
        frame: BGR image to draw on (copied internally — original not mutated).
        detections: List of :class:`~src.detection.yolo_detector.Detection` objects.
        color: Default box colour (overridden per class when colour is auto-derived).

    Returns:
        Annotated BGR image copy.
    """
    out = frame.copy()
    for det in detections:
        c = _detection_color(det)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        _label_strip(out, label, (x1, max(y1 - 2, 10)), _COLOR_TEXT_FG)
    return out


def draw_poses(
    frame: np.ndarray,
    poses: List[PoseResult],
    skeleton: bool = True,
) -> np.ndarray:
    """Draw keypoints and optional skeleton connections for each detected person.

    Args:
        frame: BGR image to draw on (copied internally).
        poses: List of :class:`~src.detection.pose_structures.PoseResult` objects.
        skeleton: If ``True``, draw limb connections between keypoints.

    Returns:
        Annotated BGR image copy.
    """
    out = frame.copy()
    for pose in poses:
        kps = pose.keypoints

        if skeleton:
            for idx_a, idx_b in COCO_SKELETON:
                kp_a, kp_b = kps[idx_a], kps[idx_b]
                if kp_a.confidence >= 0.3 and kp_b.confidence >= 0.3:
                    pt_a = (int(kp_a.x), int(kp_a.y))
                    pt_b = (int(kp_b.x), int(kp_b.y))
                    cv2.line(out, pt_a, pt_b, _COLOR_SKELETON, 2, cv2.LINE_AA)

        for kp in kps:
            pt = (int(kp.x), int(kp.y))
            if kp.confidence >= 0.3:
                cv2.circle(out, pt, 3, _COLOR_KEYPOINT, -1, cv2.LINE_AA)
            else:
                cv2.circle(out, pt, 3, _COLOR_KEYPOINT, 1, cv2.LINE_AA)

    return out


def _track_color(track_id: int) -> Tuple[int, int, int]:
    """Hash track_id to a consistent BGR colour so the same person always has the same colour."""
    import hashlib
    h = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = h & 0x0000FF
    return (b, g, r)


def draw_tracks(
    frame: np.ndarray,
    tracks: list,
    show_trail: bool = True,
) -> np.ndarray:
    """Draw tracked persons with persistent IDs, optional movement trails, and pose skeletons.

    Args:
        frame: BGR image to draw on (copied internally).
        tracks: List of :class:`~src.detection.tracker.Track` objects.
        show_trail: If ``True``, draw the last 10 bbox centre positions as a fading trail.

    Returns:
        Annotated BGR image copy.
    """
    out = frame.copy()
    for track in tracks:
        color = _track_color(track.track_id)
        state_color = color if track.state == "active" else (0, 165, 255)

        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), state_color, 2)

        state_tag = "" if track.state == "active" else " LOST"
        label = f"ID:{track.track_id}{state_tag} {track.confidence:.2f}"
        _label_strip(out, label, (x1, max(y1 - 2, 10)), _COLOR_TEXT_FG)

        if show_trail and len(track.bbox_history) > 1:
            centres = [
                (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))
                for b in list(track.bbox_history)[-10:]
            ]
            for i in range(1, len(centres)):
                alpha = i / len(centres)
                c = tuple(int(v * alpha) for v in color)
                cv2.line(out, centres[i - 1], centres[i], c, 2, cv2.LINE_AA)

        if track.pose_history:
            latest_pose = list(track.pose_history)[-1]
            kps = latest_pose.keypoints
            for idx_a, idx_b in COCO_SKELETON:
                kp_a, kp_b = kps[idx_a], kps[idx_b]
                if kp_a.confidence >= 0.3 and kp_b.confidence >= 0.3:
                    cv2.line(out, (int(kp_a.x), int(kp_a.y)), (int(kp_b.x), int(kp_b.y)),
                             _COLOR_SKELETON, 2, cv2.LINE_AA)
            for kp in kps:
                if kp.confidence >= 0.3:
                    cv2.circle(out, (int(kp.x), int(kp.y)), 3, _COLOR_KEYPOINT, -1, cv2.LINE_AA)

    return out


def draw_tracked_frame(frame: np.ndarray, analysis) -> np.ndarray:
    """Draw everything from a :class:`~src.detection.combined_pipeline.TrackedFrameAnalysis`.

    Draws tracked persons (with IDs, trails, pose skeletons) and object detections.

    Args:
        frame: BGR source frame.
        analysis: :class:`~src.detection.combined_pipeline.TrackedFrameAnalysis` instance.

    Returns:
        Fully annotated BGR image copy.
    """
    out = draw_tracks(frame, analysis.tracks)
    out = draw_detections(out, analysis.object_detections)

    infer_ms = getattr(analysis, "inference_time_ms", 0.0)
    track_ms = getattr(analysis, "tracking_time_ms", 0.0)
    active = sum(1 for t in analysis.tracks if t.state == "active")
    lost = sum(1 for t in analysis.tracks if t.state == "lost")
    info = (
        f"Tracks:{active} Lost:{lost} "
        f"Objects:{len(analysis.object_detections)} "
        f"Det:{infer_ms:.0f}ms Track:{track_ms:.1f}ms"
    )
    _label_strip(out, info, (6, 18), _COLOR_TEXT_FG, font_scale=0.5)
    return out


def draw_frame_analysis(frame: np.ndarray, analysis) -> np.ndarray:
    """Draw everything from a :class:`~src.detection.combined_pipeline.FrameAnalysis`.

    Draws person detections (green), object detections (colour-coded), and pose
    skeletons.

    Args:
        frame: BGR source frame.
        analysis: :class:`~src.detection.combined_pipeline.FrameAnalysis` instance.

    Returns:
        Fully annotated BGR image copy.
    """
    out = draw_detections(frame, analysis.person_detections)
    out = draw_detections(out, analysis.object_detections)
    out = draw_poses(out, analysis.poses)

    info = (
        f"Persons:{len(analysis.person_detections)} "
        f"Objects:{len(analysis.object_detections)} "
        f"Poses:{len(analysis.poses)} "
        f"{analysis.inference_time_ms:.0f}ms"
    )
    _label_strip(out, info, (6, 18), _COLOR_TEXT_FG, font_scale=0.5)
    return out
