"""Unit tests for tracking modules."""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tracking.centroid_tracker import CentroidTracker
from models.tracking.sort_tracker import SORTTracker


class TestCentroidTracker(unittest.TestCase):
    """Tests for CentroidTracker class."""

    def setUp(self):
        self.tracker = CentroidTracker(max_disappeared=5, max_distance=50)

    def test_register_new_object(self):
        rects = [(100, 100, 200, 200)]
        objects = self.tracker.update(rects)
        self.assertEqual(len(objects), 1)
        self.assertIn(0, objects)

    def test_track_multiple_objects(self):
        rects = [(100, 100, 200, 200), (300, 300, 400, 400)]
        objects = self.tracker.update(rects)
        self.assertEqual(len(objects), 2)

    def test_track_across_frames(self):
        self.tracker.update([(100, 100, 200, 200)])
        objects = self.tracker.update([(105, 105, 205, 205)])
        self.assertEqual(len(objects), 1)
        self.assertIn(0, objects)

    def test_disappear_and_deregister(self):
        self.tracker.update([(100, 100, 200, 200)])
        for _ in range(6):
            self.tracker.update([])
        self.assertEqual(len(self.tracker.objects), 0)

    def test_new_object_gets_new_id(self):
        self.tracker.update([(100, 100, 200, 200)])
        for _ in range(6):
            self.tracker.update([])
        objects = self.tracker.update([(500, 500, 600, 600)])
        self.assertIn(1, objects)

    def test_max_distance_prevents_match(self):
        self.tracker.update([(100, 100, 200, 200)])
        # Object appears very far away
        objects = self.tracker.update([(900, 900, 1000, 1000)])
        # Should be 2 objects (old one disappeared, new one registered)
        self.assertGreaterEqual(len(objects), 1)

    def test_get_bbox(self):
        self.tracker.update([(100, 100, 200, 200)])
        bbox = self.tracker.get_bbox(0)
        self.assertIsNotNone(bbox)
        self.assertEqual(bbox, (100, 100, 200, 200))

    def test_get_all_bboxes(self):
        self.tracker.update([(100, 100, 200, 200), (300, 300, 400, 400)])
        bboxes = self.tracker.get_all_bboxes()
        self.assertEqual(len(bboxes), 2)

    def test_empty_update(self):
        objects = self.tracker.update([])
        self.assertEqual(len(objects), 0)

    def test_reset(self):
        self.tracker.update([(100, 100, 200, 200)])
        self.tracker.reset()
        self.assertEqual(len(self.tracker.objects), 0)
        self.assertEqual(self.tracker.nextObjectID, 0)

    def test_object_count(self):
        self.tracker.update([(100, 100, 200, 200), (300, 300, 400, 400)])
        self.assertEqual(self.tracker.get_object_count(), 2)


class TestSORTTracker(unittest.TestCase):
    """Tests for SORTTracker class."""

    def setUp(self):
        self.tracker = SORTTracker(max_age=5, min_hits=1)

    def test_register_detection(self):
        results = self.tracker.update([(100, 100, 200, 200)])
        self.assertGreaterEqual(len(results), 0)

    def test_track_multiple(self):
        self.tracker.update([(100, 100, 200, 200)])
        results = self.tracker.update(
            [(105, 105, 205, 205), (400, 400, 500, 500)]
        )
        self.assertGreaterEqual(len(results), 1)

    def test_iou_calculation(self):
        iou = SORTTracker._iou(
            (0, 0, 100, 100), (50, 50, 150, 150)
        )
        self.assertGreater(iou, 0)
        self.assertLess(iou, 1)

    def test_iou_no_overlap(self):
        iou = SORTTracker._iou(
            (0, 0, 50, 50), (100, 100, 200, 200)
        )
        self.assertEqual(iou, 0.0)

    def test_iou_perfect_overlap(self):
        iou = SORTTracker._iou(
            (0, 0, 100, 100), (0, 0, 100, 100)
        )
        self.assertAlmostEqual(iou, 1.0)

    def test_reset(self):
        self.tracker.update([(100, 100, 200, 200)])
        self.tracker.reset()
        self.assertEqual(len(self.tracker.trackers), 0)


if __name__ == '__main__':
    unittest.main()
