"""ByteTrack-based multi-object tracker.

Assigns consistent IDs to detected persons across frames.
Maintains keypoint history for temporal action recognition (Phase 5).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.common.config import load_config
from src.common.logger import get_logger
from src.detection.combined_pipeline import FrameAnalysis
from src.detection.kalman_tracker import KalmanBoxTracker, x_to_bbox
from src.detection.pose_structures import PoseResult
from src.detection.yolo_detector import Detection

logger = get_logger("detection.tracker")


# ---------------------------------------------------------------------------
# IoU utilities
# ---------------------------------------------------------------------------

def compute_iou(bbox1: Tuple, bbox2: Tuple) -> float:
    """Compute Intersection over Union between two bboxes ``[x1, y1, x2, y2]``.

    Args:
        bbox1: First bounding box.
        bbox2: Second bounding box.

    Returns:
        IoU score in ``[0, 1]``.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_iou_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Compute N×M IoU matrix between two sets of bboxes.

    Args:
        bboxes1: Array of shape ``(N, 4)``.
        bboxes2: Array of shape ``(M, 4)``.

    Returns:
        IoU matrix of shape ``(N, M)``.
    """
    n, m = len(bboxes1), len(bboxes2)
    matrix = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            matrix[i, j] = compute_iou(bboxes1[i], bboxes2[j])
    return matrix


# ---------------------------------------------------------------------------
# Hungarian assignment
# ---------------------------------------------------------------------------

def linear_assignment(
    cost_matrix: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Run Hungarian (linear) assignment on a cost matrix.

    Tries ``lap.lapjv`` first; falls back to
    ``scipy.optimize.linear_sum_assignment``.

    Args:
        cost_matrix: Array of shape ``(N, M)`` — lower = better match.

    Returns:
        Tuple of:
        - ``matches``: list of ``(row, col)`` index pairs.
        - ``unmatched_rows``: row indices with no assignment.
        - ``unmatched_cols``: col indices with no assignment.
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    try:
        import lap
        _, row_inds, col_inds = lap.lapjv(cost_matrix, extend_cost=True)
        matches = [(r, col_inds[r]) for r in range(len(row_inds)) if col_inds[r] >= 0]
        unmatched_rows = [r for r in range(cost_matrix.shape[0]) if col_inds[r] < 0]
        unmatched_cols = [c for c in range(cost_matrix.shape[1]) if row_inds[c] < 0]
    except Exception:
        from scipy.optimize import linear_sum_assignment
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        matched_set = set(zip(row_inds.tolist(), col_inds.tolist()))
        matches = list(matched_set)
        unmatched_rows = [r for r in range(cost_matrix.shape[0]) if r not in row_inds]
        unmatched_cols = [c for c in range(cost_matrix.shape[1]) if c not in col_inds]

    return matches, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# Track dataclass
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """A single tracked person with persistent identity across frames.

    Attributes:
        track_id: Unique persistent integer ID.
        state: One of ``"active"``, ``"lost"``, ``"removed"``.
        bbox: Current estimated bounding box ``(x1, y1, x2, y2)``.
        bbox_history: Rolling deque of past bboxes (maxlen=history_length).
        keypoint_history: Rolling deque of past 17×3 keypoint arrays.
        pose_history: Rolling deque of past :class:`PoseResult` objects.
        age: Total frames since track creation.
        hits: Total successful detection matches.
        time_since_update: Frames elapsed since last matched detection.
        confidence: Detection confidence of the latest match.
        camera_id: Source camera identifier.
        created_at: Unix timestamp of first detection.
        last_seen_at: Unix timestamp of last matched detection.
    """

    track_id: int
    state: str
    bbox: Tuple[int, int, int, int]
    bbox_history: deque = field(default_factory=deque)
    keypoint_history: deque = field(default_factory=deque)
    pose_history: deque = field(default_factory=deque)
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    confidence: float = 0.0
    camera_id: str = "default"
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """ByteTrack-style multi-object tracker.

    Maintains a pool of :class:`Track` objects across frames, using IoU-based
    matching with a two-stage association (high-confidence then low-confidence
    detections) as described in the ByteTrack paper.

    Args:
        config: Optional pre-loaded configuration dict.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        cfg = config or load_config()
        t = cfg.get("tracker", {})
        self._max_lost: int = t.get("max_lost_frames", 30)
        self._iou_high: float = t.get("iou_threshold_high", 0.3)
        self._iou_low: float = t.get("iou_threshold_low", 0.5)
        self._conf_split: float = t.get("confidence_split", 0.5)
        self._min_hits: int = t.get("min_hits", 3)
        self._history_length: int = t.get("history_length", 64)

        self._tracks: List[Track] = []
        self._kalman_map: Dict[int, KalmanBoxTracker] = {}
        self._next_id: int = 1

        self._stats: Dict = {
            "total_created": 0,
            "total_removed": 0,
        }

        logger.info(
            "ByteTracker initialised: max_lost=%d conf_split=%.2f history=%d",
            self._max_lost, self._conf_split, self._history_length,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_analysis: FrameAnalysis) -> List[Track]:
        """Process one frame and update all tracks.

        Args:
            frame_analysis: :class:`FrameAnalysis` containing person detections
                and poses for the current frame.

        Returns:
            List of all tracks that are ``"active"`` or ``"lost"``.
        """
        person_dets = frame_analysis.person_detections
        poses = frame_analysis.poses
        ts = frame_analysis.timestamp

        pose_map: Dict[Tuple, PoseResult] = {}
        for pose in poses:
            pose_map[tuple(pose.bbox)] = pose

        high_dets = [d for d in person_dets if d.confidence >= self._conf_split]
        low_dets = [d for d in person_dets if d.confidence < self._conf_split]

        active = [t for t in self._tracks if t.state in ("active", "lost")]
        pred_bboxes = self._predict_all(active)

        matched_high, unmatched_tracks, unmatched_high = self._associate(
            active, pred_bboxes, high_dets, iou_thresh=self._iou_high
        )

        remaining_tracks = [active[i] for i in unmatched_tracks]
        remaining_pred = [pred_bboxes[i] for i in unmatched_tracks]

        matched_low, still_unmatched_tracks, unmatched_low = self._associate(
            remaining_tracks, remaining_pred, low_dets, iou_thresh=self._iou_low
        )

        for t_idx, d_idx in matched_high:
            self._update_track(active[t_idx], high_dets[d_idx], ts, pose_map)

        for t_idx, d_idx in matched_low:
            self._update_track(remaining_tracks[t_idx], low_dets[d_idx], ts, pose_map)

        for d_idx in unmatched_high:
            self._create_track(high_dets[d_idx], ts, pose_map=pose_map)

        lost_track_ids = {remaining_tracks[i].track_id for i in still_unmatched_tracks}
        for track in self._tracks:
            if track.track_id in lost_track_ids or (
                track.state == "lost" and track.track_id not in
                {t.track_id for t in active if t.track_id not in lost_track_ids}
            ):
                if track.state != "removed":
                    track.state = "lost"
                    track.time_since_update += 1

        for track in self._tracks:
            if track.state == "lost" and track.time_since_update > self._max_lost:
                track.state = "removed"
                self._stats["total_removed"] += 1
                logger.debug("Track %d removed after %d lost frames.", track.track_id, track.time_since_update)

        return [t for t in self._tracks if t.state in ("active", "lost")]

    def get_active_tracks(self) -> List[Track]:
        """Return only confirmed active tracks (hits >= min_hits, state=active)."""
        return [
            t for t in self._tracks
            if t.state == "active" and t.hits >= self._min_hits
        ]

    def get_track(self, track_id: int) -> Optional[Track]:
        """Return the track with the given ID, or ``None`` if not found."""
        for t in self._tracks:
            if t.track_id == track_id:
                return t
        return None

    def get_track_keypoint_sequence(
        self, track_id: int, window_size: int = 16
    ) -> Optional[np.ndarray]:
        """Return the last ``window_size`` keypoint frames as a numpy array.

        This is the direct input format for Phase 5's action classifier.

        Args:
            track_id: ID of the target track.
            window_size: Number of frames requested.

        Returns:
            NumPy array of shape ``(window_size, 17, 3)`` or ``None`` if the
            track has fewer than ``window_size`` keypoint frames available.
        """
        track = self.get_track(track_id)
        if track is None or len(track.keypoint_history) < window_size:
            return None
        seq = list(track.keypoint_history)[-window_size:]
        return np.array(seq, dtype=np.float32)

    def reset(self) -> None:
        """Clear all tracks and reset internal state."""
        self._tracks.clear()
        self._kalman_map.clear()
        self._next_id = 1
        self._stats = {"total_created": 0, "total_removed": 0}
        logger.info("ByteTracker reset.")

    def get_stats(self) -> Dict:
        """Return cumulative tracking statistics."""
        active = sum(1 for t in self._tracks if t.state == "active")
        lost = sum(1 for t in self._tracks if t.state == "lost")
        removed = sum(1 for t in self._tracks if t.state == "removed")
        return {
            "total_created": self._stats["total_created"],
            "total_removed": self._stats["total_removed"],
            "currently_active": active,
            "currently_lost": lost,
            "currently_removed": removed,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_all(self, tracks: List[Track]) -> List[np.ndarray]:
        preds = []
        for track in tracks:
            kf = self._kalman_map.get(track.track_id)
            if kf is not None:
                pred = kf.predict()
            else:
                pred = np.array(track.bbox, dtype=np.float64)
            preds.append(pred)
            track.age += 1
        return preds

    def _associate(
        self,
        tracks: List[Track],
        pred_bboxes: List[np.ndarray],
        detections: List[Detection],
        iou_thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_arr = np.array([b for b in pred_bboxes], dtype=np.float64)
        det_arr = np.array([list(d.bbox) for d in detections], dtype=np.float64)
        iou_mat = compute_iou_matrix(track_arr, det_arr)
        cost_mat = 1.0 - iou_mat

        cost_mat[iou_mat < iou_thresh] = 1e9

        matches, unmatched_t, unmatched_d = linear_assignment(cost_mat)
        valid_matches = [(t, d) for t, d in matches if iou_mat[t, d] >= iou_thresh]
        newly_unmatched_t = [t for t, d in matches if iou_mat[t, d] < iou_thresh]
        unmatched_t = unmatched_t + newly_unmatched_t

        return valid_matches, unmatched_t, unmatched_d

    def _update_track(
        self,
        track: Track,
        det: Detection,
        ts: float,
        pose_map: Dict,
    ) -> None:
        kf = self._kalman_map.get(track.track_id)
        bbox_arr = np.array(list(det.bbox), dtype=np.float64)
        if kf is not None:
            kf.update(bbox_arr)
            new_bbox_arr = kf.get_state()
        else:
            new_bbox_arr = bbox_arr

        new_bbox = tuple(int(v) for v in new_bbox_arr)
        track.bbox = new_bbox
        track.confidence = det.confidence
        track.state = "active"
        track.hits += 1
        track.time_since_update = 0
        track.last_seen_at = ts

        track.bbox_history.append(new_bbox)

        pose = pose_map.get(tuple(det.bbox))
        if pose is None and pose_map:
            det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
            det_cy = (det.bbox[1] + det.bbox[3]) / 2.0
            best_key, best_dist = None, float("inf")
            for key in pose_map:
                pcx = (key[0] + key[2]) / 2.0
                pcy = (key[1] + key[3]) / 2.0
                d = (det_cx - pcx) ** 2 + (det_cy - pcy) ** 2
                if d < best_dist:
                    best_dist, best_key = d, key
            if best_dist < 50 ** 2:
                pose = pose_map.get(best_key)
        if pose is not None:
            track.pose_history.append(pose)
            kp_arr = np.array(pose.keypoints_as_array(), dtype=np.float32)
            track.keypoint_history.append(kp_arr)

    def _create_track(self, det: Detection, ts: float, pose_map: Optional[Dict] = None) -> None:
        bbox_arr = np.array(list(det.bbox), dtype=np.float64)
        kf = KalmanBoxTracker(bbox_arr)
        tid = self._next_id
        self._next_id += 1

        kp_hist: deque = deque(maxlen=self._history_length)
        pose_hist: deque = deque(maxlen=self._history_length)

        if pose_map:
            pose = pose_map.get(tuple(det.bbox))
            if pose is None:
                det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
                det_cy = (det.bbox[1] + det.bbox[3]) / 2.0
                best_key, best_dist = None, float("inf")
                for key in pose_map:
                    pcx = (key[0] + key[2]) / 2.0
                    pcy = (key[1] + key[3]) / 2.0
                    d = (det_cx - pcx) ** 2 + (det_cy - pcy) ** 2
                    if d < best_dist:
                        best_dist, best_key = d, key
                if best_dist < 50 ** 2:
                    pose = pose_map.get(best_key)
            if pose is not None:
                pose_hist.append(pose)
                kp_hist.append(np.array(pose.keypoints_as_array(), dtype=np.float32))

        track = Track(
            track_id=tid,
            state="active",
            bbox=det.bbox,
            bbox_history=deque([det.bbox], maxlen=self._history_length),
            keypoint_history=kp_hist,
            pose_history=pose_hist,
            age=1,
            hits=1,
            time_since_update=0,
            confidence=det.confidence,
            camera_id=getattr(det, "camera_id", "default"),
            created_at=ts,
            last_seen_at=ts,
        )

        self._kalman_map[tid] = kf
        self._tracks.append(track)
        self._stats["total_created"] += 1
        logger.debug("New track %d created at bbox=%s conf=%.2f", tid, det.bbox, det.confidence)
