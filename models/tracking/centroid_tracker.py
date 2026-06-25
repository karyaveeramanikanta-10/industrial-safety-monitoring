"""
Centroid-based object tracker for multi-person tracking.

Assigns unique IDs to detected persons and maintains tracking across
frames by matching closest centroids using Euclidean distance.
Based on the PyImageSearch centroid tracker algorithm.
"""

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger('safety_monitor')


class CentroidTracker:
    """Track objects across video frames using centroid distance matching.

    Assigns unique integer IDs to each detected object and tracks them
    by computing pairwise Euclidean distances between existing object
    centroids and new detection centroids each frame.

    Usage:
        tracker = CentroidTracker(max_disappeared=50)
        # Each frame:
        objects = tracker.update([(x1,y1,x2,y2), ...])
        for object_id, centroid in objects.items():
            print(f"Object {object_id} at {centroid}")
    """

    def __init__(self, max_disappeared: int = 50,
                 max_distance: int = 50):
        """Initialize centroid tracker.

        Args:
            max_disappeared: Maximum consecutive frames an object can be
                            missing before being deregistered.
            max_distance: Maximum pixel distance for centroid matching.
                         Matches beyond this distance are rejected.
        """
        self.nextObjectID = 0
        self.objects: OrderedDict = OrderedDict()       # ID -> centroid
        self.disappeared: OrderedDict = OrderedDict()   # ID -> count
        self.bboxes: OrderedDict = OrderedDict()        # ID -> bbox
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: np.ndarray,
                 bbox: Optional[Tuple] = None):
        """Register a new object with a unique ID.

        Args:
            centroid: (cX, cY) center coordinates.
            bbox: Optional (x1, y1, x2, y2) bounding box.
        """
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        if bbox is not None:
            self.bboxes[self.nextObjectID] = bbox
        self.nextObjectID += 1

    def deregister(self, objectID: int):
        """Remove a tracked object.

        Args:
            objectID: The ID of the object to remove.
        """
        del self.objects[objectID]
        del self.disappeared[objectID]
        if objectID in self.bboxes:
            del self.bboxes[objectID]

    def update(self, rects: List[Tuple]) -> OrderedDict:
        """Update tracker with new frame detections.

        Core tracking algorithm:
        1. If no detections, increment disappeared counts
        2. If no existing objects, register all detections
        3. Otherwise, compute distance matrix and greedily match
        4. Handle unmatched objects and new detections

        Args:
            rects: List of (x1, y1, x2, y2) bounding boxes.

        Returns:
            OrderedDict of {objectID: centroid} for currently tracked objects.
        """
        # No detections — mark all existing objects as disappeared
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        # Compute centroids from input bounding boxes
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # No existing objects — register all new detections
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
            return self.objects

        # Match existing objects to new detections
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        # Compute pairwise Euclidean distance matrix
        D = dist.cdist(np.array(objectCentroids), inputCentroids)

        # Sort rows by minimum distance value (greedy matching)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            # Skip match if distance exceeds threshold
            if D[row, col] > self.max_distance:
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = inputCentroids[col]
            self.bboxes[objectID] = rects[col]
            self.disappeared[objectID] = 0

            usedRows.add(row)
            usedCols.add(col)

        # Handle unmatched existing objects (disappeared)
        unusedRows = set(range(D.shape[0])).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.max_disappeared:
                self.deregister(objectID)

        # Handle unmatched new detections (new objects)
        unusedCols = set(range(D.shape[1])).difference(usedCols)
        for col in unusedCols:
            self.register(inputCentroids[col], rects[col])

        return self.objects

    def get_bbox(self, objectID: int) -> Optional[Tuple]:
        """Get bounding box for a tracked object.

        Args:
            objectID: The object's tracking ID.

        Returns:
            (x1, y1, x2, y2) bounding box or None.
        """
        return self.bboxes.get(objectID)

    def get_all_bboxes(self) -> Dict[int, Tuple]:
        """Get all currently tracked bounding boxes.

        Returns:
            Dict of {objectID: (x1, y1, x2, y2)}.
        """
        return dict(self.bboxes)

    def get_object_count(self) -> int:
        """Get current number of tracked objects."""
        return len(self.objects)

    def reset(self):
        """Reset tracker, clearing all tracked objects."""
        self.nextObjectID = 0
        self.objects.clear()
        self.disappeared.clear()
        self.bboxes.clear()
