"""
Performance metrics tracking for Industrial Safety Monitoring System.

Provides FPS counting and detection performance tracking utilities.
"""

import time
from collections import deque
from typing import Dict, Set, List


class FPSCounter:
    """Track frames per second with rolling average.

    Computes FPS using a sliding window of frame timestamps
    for smooth, accurate readings.
    """

    def __init__(self, avg_count: int = 30):
        """Initialize FPS counter.

        Args:
            avg_count: Number of frames to average over.
        """
        self._times = deque(maxlen=avg_count)
        self._start_time = None
        self._fps = 0.0

    def start(self):
        """Mark the start of a frame."""
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Mark the end of a frame and compute FPS.

        Returns:
            Current FPS value.
        """
        if self._start_time is None:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        self._times.append(elapsed)
        self._start_time = None

        if len(self._times) > 0:
            avg_time = sum(self._times) / len(self._times)
            self._fps = 1.0 / avg_time if avg_time > 0 else 0.0
        return self._fps

    @property
    def fps(self) -> float:
        """Get current FPS value."""
        return self._fps

    def reset(self):
        """Reset the counter."""
        self._times.clear()
        self._start_time = None
        self._fps = 0.0


class DetectionMetrics:
    """Track detection and violation performance metrics.

    Maintains running counts of detections, violations, compliance,
    and per-worker statistics for dashboard display.
    """

    def __init__(self):
        self.total_frames: int = 0
        self.total_detections: int = 0
        self.total_violations: int = 0
        self.violations_by_type: Dict[str, int] = {}
        self.workers_seen: Set[int] = set()
        self.compliant_frames: int = 0
        self._worker_violations: Dict[int, int] = {}

    def update(self, num_detections: int, violations: List[Dict],
               worker_ids: List[int]):
        """Update metrics with a frame's results.

        Args:
            num_detections: Number of persons detected in frame.
            violations: List of violation dicts with 'worker_id' and 'type'.
            worker_ids: List of worker IDs detected in frame.
        """
        self.total_frames += 1
        self.total_detections += num_detections
        self.workers_seen.update(worker_ids)

        if violations:
            self.total_violations += len(violations)
            for v in violations:
                vtype = v.get('type', 'unknown')
                self.violations_by_type[vtype] = \
                    self.violations_by_type.get(vtype, 0) + 1
                wid = v.get('worker_id', -1)
                self._worker_violations[wid] = \
                    self._worker_violations.get(wid, 0) + 1
        else:
            if num_detections > 0:
                self.compliant_frames += 1

    def get_compliance_rate(self) -> float:
        """Get overall compliance rate as a percentage.

        Returns:
            Compliance rate (0-100).
        """
        if self.total_frames == 0:
            return 100.0
        frames_with_detections = max(1, self.total_frames)
        return (self.compliant_frames / frames_with_detections) * 100.0

    def get_summary(self) -> Dict:
        """Get comprehensive metrics summary.

        Returns:
            Dict with all metric values.
        """
        return {
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'total_violations': self.total_violations,
            'unique_workers': len(self.workers_seen),
            'violations_by_type': dict(self.violations_by_type),
            'compliance_rate': self.get_compliance_rate(),
            'worker_violations': dict(self._worker_violations),
        }

    def reset(self):
        """Reset all metrics."""
        self.total_frames = 0
        self.total_detections = 0
        self.total_violations = 0
        self.violations_by_type.clear()
        self.workers_seen.clear()
        self.compliant_frames = 0
        self._worker_violations.clear()
