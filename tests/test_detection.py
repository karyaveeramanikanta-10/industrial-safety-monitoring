"""Unit tests for person and PPE detection modules."""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.person_detector.person_detector import PersonDetector
from models.ppe_detector.ppe_detector import PPEDetector


class TestPersonDetector(unittest.TestCase):
    """Tests for PersonDetector class."""

    def setUp(self):
        self.detector = PersonDetector(confidence_threshold=0.5)

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertTrue(self.detector.is_loaded)

    def test_detect_returns_list(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        self.assertIsInstance(result, list)

    def test_detect_with_empty_frame(self):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        self.assertIsInstance(result, list)

    def test_detect_with_none_frame(self):
        result = self.detector.detect(None)
        self.assertEqual(result, [])

    def test_detection_format(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(frame)
        for det in results:
            self.assertIn('bbox', det)
            self.assertIn('confidence', det)
            self.assertEqual(len(det['bbox']), 4)
            self.assertGreaterEqual(det['confidence'], 0)
            self.assertLessEqual(det['confidence'], 1)

    def test_backend_property(self):
        backend = self.detector.backend
        self.assertIn(backend, ['tensorflow', 'hog', 'none'])

    def test_confidence_threshold(self):
        detector = PersonDetector(confidence_threshold=0.9)
        self.assertEqual(detector.confidence_threshold, 0.9)


class TestPPEDetector(unittest.TestCase):
    """Tests for PPEDetector class."""

    def setUp(self):
        self.detector = PPEDetector(required_ppe=['helmet', 'vest'])

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.required_ppe, ['helmet', 'vest'])

    def test_detect_returns_dict(self):
        crop = np.zeros((200, 100, 3), dtype=np.uint8)
        result = self.detector.detect(crop)
        self.assertIsInstance(result, dict)

    def test_detect_has_all_ppe_items(self):
        crop = np.zeros((200, 100, 3), dtype=np.uint8)
        result = self.detector.detect(crop)
        expected_items = [
            'helmet', 'vest', 'mask', 'goggles',
            'gloves', 'ear_protection', 'shoes'
        ]
        for item in expected_items:
            self.assertIn(item, result)
            self.assertIn('detected', result[item])
            self.assertIn('confidence', result[item])

    def test_detect_with_none_crop(self):
        result = self.detector.detect(None)
        self.assertIsInstance(result, dict)
        for item_data in result.values():
            self.assertFalse(item_data['detected'])

    def test_detect_with_small_crop(self):
        crop = np.zeros((5, 5, 3), dtype=np.uint8)
        result = self.detector.detect(crop)
        self.assertIsInstance(result, dict)

    def test_compliance_check_compliant(self):
        ppe_status = {
            'helmet': {'detected': True, 'confidence': 0.8, 'region': 'head'},
            'vest': {'detected': True, 'confidence': 0.7, 'region': 'torso'},
        }
        result = self.detector.check_compliance(ppe_status)
        self.assertTrue(result['compliant'])
        self.assertEqual(len(result['missing']), 0)

    def test_compliance_check_violation(self):
        ppe_status = {
            'helmet': {'detected': True, 'confidence': 0.8, 'region': 'head'},
            'vest': {'detected': False, 'confidence': 0.05, 'region': 'torso'},
        }
        result = self.detector.check_compliance(ppe_status)
        self.assertFalse(result['compliant'])
        self.assertIn('vest', result['missing'])
        self.assertIn('helmet', result['present'])

    def test_set_required_ppe(self):
        self.detector.set_required_ppe(['helmet', 'mask', 'gloves'])
        self.assertEqual(
            self.detector.required_ppe, ['helmet', 'mask', 'gloves']
        )

    def test_set_required_ppe_filters_invalid(self):
        self.detector.set_required_ppe(['helmet', 'invalid_item', 'vest'])
        self.assertNotIn('invalid_item', self.detector.required_ppe)

    def test_yellow_helmet_detection(self):
        """Test that a yellow region at the top is detected as a helmet."""
        crop = np.zeros((200, 100, 3), dtype=np.uint8)
        # Paint top 15% yellow (HSV: H=30, S=200, V=200)
        # In BGR: Yellow is roughly (0, 200, 200)
        crop[0:30, :] = (0, 200, 200)
        result = self.detector.detect(crop)
        # The yellow region should have some confidence for helmet
        self.assertGreater(result['helmet']['confidence'], 0)


if __name__ == '__main__':
    unittest.main()
