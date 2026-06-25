"""
Simplified SORT (Simple Online and Realtime Tracking) tracker.

Uses a linear velocity Kalman filter for motion prediction and the
Hungarian algorithm for detection-to-track assignment based on IoU.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger('safety_monitor')


class KalmanBoxTracker:
    """Kalman filter tracker for a single bounding box.

    Maintains state [x, y, w, h, dx, dy] where (x, y) is the
    center, (w, h) is the size, and (dx, dy) is the velocity.
    Uses a simple constant-velocity model.
    """

    count = 0

    def __init__(self, bbox: Tuple):
        """Initialize tracker with a bounding box.

        Args:
            bbox: (x1, y1, x2, y2) bounding box.
        """
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        # State: [cx, cy, w, h, dx, dy]
        self.state = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history = [bbox]

    def predict(self) -> np.ndarray:
        """Predict next bounding box position.

        Returns:
            Predicted (x1, y1, x2, y2) bounding box.
        """
        # Apply velocity
        self.state[0] += self.state[4]  # cx += dx
        self.state[1] += self.state[5]  # cy += dy

        # Prevent negative dimensions
        self.state[2] = max(self.state[2], 1)
        self.state[3] = max(self.state[3], 1)

        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox: Tuple):
        """Update tracker with observed bounding box.

        Args:
            bbox: (x1, y1, x2, y2) observed bounding box.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        # Compute velocity (simple exponential moving average)
        alpha = 0.5
        self.state[4] = alpha * (cx - self.state[0]) + (1 - alpha) * self.state[4]
        self.state[5] = alpha * (cy - self.state[1]) + (1 - alpha) * self.state[5]

        # Update position and size
        self.state[0] = cx
        self.state[1] = cy
        self.state[2] = w
        self.state[3] = h

        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)
        if len(self.history) > 30:
            self.history = self.history[-30:]

    def get_state(self) -> Tuple:
        """Return current bounding box estimate.

        Returns:
            (x1, y1, x2, y2) bounding box.
        """
        cx, cy, w, h = self.state[:4]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return (int(x1), int(y1), int(x2), int(y2))


class SORTTracker:
    """Simplified SORT (Simple Online and Realtime Tracking).

    Uses Kalman filters for motion prediction and the Hungarian
    algorithm for optimal detection-to-track assignment based on IoU.

    Usage:
        tracker = SORTTracker(max_age=30)
        # Each frame:
        results = tracker.update([(x1,y1,x2,y2), ...])
        for track_id, bbox in results.items():
            print(f"Track {track_id}: {bbox}")
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """Initialize SORT tracker.

        Args:
            max_age: Max frames a track can live without a match.
            min_hits: Min consecutive hits before a track is reported.
            iou_threshold: Minimum IoU for valid assignment.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, detections: List[Tuple]) -> OrderedDict:
        """Update tracker with new detections.

        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes.

        Returns:
            OrderedDict of {track_id: (x1, y1, x2, y2)}.
        """
        # Step 1: Predict new locations for existing trackers
        predicted = []
        to_remove = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            predicted.append(pred)
            # Remove tracker if predicted box has invalid dimensions
            if pred[2] <= pred[0] or pred[3] <= pred[1]:
                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            self.trackers.pop(i)
            predicted.pop(i)

        # Step 2: Compute IoU matrix and match
        if len(detections) > 0 and len(predicted) > 0:
            iou_matrix = np.zeros(
                (len(predicted), len(detections)), dtype=np.float32
            )
            for t, trk_box in enumerate(predicted):
                for d, det_box in enumerate(detections):
                    iou_matrix[t, d] = self._iou(trk_box, det_box)

            # Hungarian algorithm (minimize cost = 1 - IoU)
            matched_rows, matched_cols = linear_sum_assignment(
                1.0 - iou_matrix
            )

            matched_tracks = set()
            matched_dets = set()

            for row, col in zip(matched_rows, matched_cols):
                if iou_matrix[row, col] >= self.iou_threshold:
                    self.trackers[row].update(detections[col])
                    matched_tracks.add(row)
                    matched_dets.add(col)

            # Unmatched detections -> new trackers
            for d in range(len(detections)):
                if d not in matched_dets:
                    self.trackers.append(KalmanBoxTracker(detections[d]))

            # Unmatched trackers -> increase time_since_update (already done in predict)

        elif len(detections) > 0:
            # No existing trackers, create new ones for all detections
            for det in detections:
                self.trackers.append(KalmanBoxTracker(det))

        # Step 3: Remove dead trackers and collect results
        results = OrderedDict()
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

        for trk in self.trackers:
            if trk.hits >= self.min_hits or trk.time_since_update == 0:
                results[trk.id] = trk.get_state()

        return results

    @staticmethod
    def _iou(bb1: Tuple, bb2: Tuple) -> float:
        """Calculate Intersection over Union between two bounding boxes.

        Args:
            bb1, bb2: (x1, y1, x2, y2) bounding boxes.

        Returns:
            IoU value (0.0 to 1.0).
        """
        x1 = max(bb1[0], bb2[0])
        y1 = max(bb1[1], bb2[1])
        x2 = min(bb1[2], bb2[2])
        y2 = min(bb1[3], bb2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, bb1[2] - bb1[0]) * max(0, bb1[3] - bb1[1])
        area2 = max(0, bb2[2] - bb2[0]) * max(0, bb2[3] - bb2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def reset(self):
        """Reset tracker, clearing all tracks."""
        self.trackers.clear()
        KalmanBoxTracker.count = 0
