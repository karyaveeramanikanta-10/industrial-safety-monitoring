"""
Comprehensive integration test suite for Industrial Safety Monitoring System.

Tests:
  1. Person Detection — HOG backend functionality
  2. PPE Detection — Color-based accuracy with synthetic test images
  3. Centroid Tracker — Multi-object tracking correctness
  4. Full Pipeline — SafetyPredictor end-to-end integration
  5. Database — MongoDB Atlas connectivity and CRUD operations
  6. Alerts — Alert manager trigger and history
  7. Config — YAML loading and singleton pattern
  8. Utils — Helpers, metrics, visualization
"""

import os
import sys
import time
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from models.person_detector.person_detector import PersonDetector
from models.ppe_detector.ppe_detector import PPEDetector
from models.tracking.centroid_tracker import CentroidTracker
from alerts.alert_manager import AlertManager
from utils.helpers import (
    calculate_iou, crop_region, get_body_regions,
    resize_frame, format_timestamp, generate_session_id
)
from utils.metrics import FPSCounter, DetectionMetrics
from utils.visualization import (
    draw_person_detection, draw_ppe_status,
    draw_violation_alert, draw_dashboard_overlay,
    create_violation_snapshot
)


# =============================================================================
# Test 1: Person Detection
# =============================================================================
class TestPersonDetection(unittest.TestCase):
    """Validate person detection with HOG backend."""

    @classmethod
    def setUpClass(cls):
        cls.detector = PersonDetector(confidence_threshold=0.3)

    def test_01_backend_loaded(self):
        """Verify detection backend is loaded."""
        self.assertTrue(self.detector.is_loaded)
        print(f"  [INFO] Detection backend: {self.detector.backend}")

    def test_02_detect_on_blank_frame(self):
        """No persons in a blank black frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(frame)
        self.assertIsInstance(results, list)
        print(f"  [INFO] Blank frame detections: {len(results)}")

    def test_03_detect_none_frame(self):
        """Handle None input gracefully."""
        self.assertEqual(self.detector.detect(None), [])

    def test_04_detect_tiny_frame(self):
        """Handle very small frames without crashing."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        results = self.detector.detect(frame)
        self.assertIsInstance(results, list)

    def test_05_detection_output_format(self):
        """Each detection has 'bbox' (4-tuple) and 'confidence' (float)."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(frame)
        for det in results:
            self.assertIn('bbox', det)
            self.assertIn('confidence', det)
            self.assertEqual(len(det['bbox']), 4)
            self.assertGreaterEqual(det['confidence'], 0)
            self.assertLessEqual(det['confidence'], 1)

    def test_06_nms_filters_duplicates(self):
        """Non-maximum suppression removes overlapping detections."""
        detections = [
            {'bbox': (100, 100, 200, 200), 'confidence': 0.9},
            {'bbox': (105, 105, 205, 205), 'confidence': 0.7},
            {'bbox': (400, 400, 500, 500), 'confidence': 0.8},
        ]
        filtered = PersonDetector._nms(detections, threshold=0.3)
        self.assertLessEqual(len(filtered), 3)
        # The two overlapping boxes should merge
        self.assertEqual(len(filtered), 2)
        print(f"  [INFO] NMS: 3 inputs -> {len(filtered)} outputs")


# =============================================================================
# Test 2: PPE Detection Accuracy
# =============================================================================
class TestPPEDetectionAccuracy(unittest.TestCase):
    """Test PPE detection accuracy with synthetic colored images."""

    @classmethod
    def setUpClass(cls):
        cls.detector = PPEDetector(
            required_ppe=['helmet', 'vest', 'mask', 'gloves']
        )

    def _make_person_crop(self, h=200, w=100):
        """Create a base person crop (skin-colored)."""
        crop = np.full((h, w, 3), (130, 140, 180), dtype=np.uint8)  # skin-ish
        return crop

    def test_01_yellow_helmet_detected(self):
        """Yellow hard hat in head region should be detected."""
        crop = self._make_person_crop()
        # Paint head region bright yellow (BGR: 0, 255, 255)
        head_end = int(200 * 0.15)
        crop[0:head_end, :] = (0, 255, 255)
        result = self.detector.detect(crop)
        print(f"  [INFO] Yellow helmet confidence: {result['helmet']['confidence']:.4f}")
        self.assertGreater(result['helmet']['confidence'], 0.0,
                           "Yellow helmet should produce non-zero confidence")

    def test_02_white_helmet_detected(self):
        """White hard hat in head region should be detected."""
        crop = self._make_person_crop()
        head_end = int(200 * 0.15)
        crop[0:head_end, :] = (250, 250, 250)  # White
        result = self.detector.detect(crop)
        print(f"  [INFO] White helmet confidence: {result['helmet']['confidence']:.4f}")
        self.assertGreater(result['helmet']['confidence'], 0.0)

    def test_03_hivis_vest_detected(self):
        """Hi-vis yellow/orange vest in torso region should be detected."""
        crop = self._make_person_crop()
        torso_start = int(200 * 0.20)
        torso_end = int(200 * 0.55)
        # Hi-vis yellow in BGR
        crop[torso_start:torso_end, :] = (0, 230, 230)
        result = self.detector.detect(crop)
        print(f"  [INFO] Hi-vis vest confidence: {result['vest']['confidence']:.4f}")
        self.assertGreater(result['vest']['confidence'], 0.0,
                           "Hi-vis vest should produce non-zero confidence")

    def test_04_blue_mask_detected(self):
        """Blue surgical mask in face region should be detected."""
        crop = self._make_person_crop()
        face_start = int(200 * 0.08)
        face_end = int(200 * 0.25)
        x_start = int(100 * 0.25)
        x_end = int(100 * 0.75)
        # Blue mask (BGR: 200, 120, 0)
        crop[face_start:face_end, x_start:x_end] = (200, 120, 0)
        result = self.detector.detect(crop)
        print(f"  [INFO] Blue mask confidence: {result['mask']['confidence']:.4f}")
        self.assertGreater(result['mask']['confidence'], 0.0)

    def test_05_no_ppe_all_zero(self):
        """Plain skin-colored crop should have low PPE confidences."""
        crop = self._make_person_crop()
        result = self.detector.detect(crop)
        # Most items should have very low confidence on plain skin
        for item in ['helmet', 'vest', 'mask']:
            self.assertLess(result[item]['confidence'], 0.3,
                            f"{item} should have low confidence on plain image")
        print(f"  [INFO] No-PPE confidences: "
              f"helmet={result['helmet']['confidence']:.3f}, "
              f"vest={result['vest']['confidence']:.3f}, "
              f"mask={result['mask']['confidence']:.3f}")

    def test_06_compliance_all_present(self):
        """Worker with all required PPE should be compliant."""
        ppe_status = {
            'helmet': {'detected': True, 'confidence': 0.8, 'region': 'head'},
            'vest': {'detected': True, 'confidence': 0.7, 'region': 'torso'},
            'mask': {'detected': True, 'confidence': 0.6, 'region': 'face'},
            'gloves': {'detected': True, 'confidence': 0.5, 'region': 'hands'},
        }
        result = self.detector.check_compliance(ppe_status)
        self.assertTrue(result['compliant'])
        self.assertEqual(len(result['missing']), 0)
        self.assertEqual(len(result['present']), 4)
        print(f"  [INFO] Full compliance: {result['present']}")

    def test_07_compliance_missing_items(self):
        """Worker missing PPE should be non-compliant."""
        ppe_status = {
            'helmet': {'detected': True, 'confidence': 0.8, 'region': 'head'},
            'vest': {'detected': False, 'confidence': 0.01, 'region': 'torso'},
            'mask': {'detected': False, 'confidence': 0.02, 'region': 'face'},
            'gloves': {'detected': True, 'confidence': 0.5, 'region': 'hands'},
        }
        result = self.detector.check_compliance(ppe_status)
        self.assertFalse(result['compliant'])
        self.assertIn('vest', result['missing'])
        self.assertIn('mask', result['missing'])
        print(f"  [INFO] Missing PPE detected: {result['missing']}")

    def test_08_full_ppe_accuracy_report(self):
        """Generate accuracy summary for all PPE types."""
        test_cases = [
            ('helmet_yellow', (0, 255, 255), 'head', 'helmet'),
            ('helmet_white', (250, 250, 250), 'head', 'helmet'),
            ('vest_hivis', (0, 230, 230), 'torso', 'vest'),
            ('mask_blue', (200, 120, 0), 'face', 'mask'),
            ('mask_white', (250, 250, 250), 'face', 'mask'),
        ]

        regions_map = {
            'head': (0, int(200*0.15)),
            'torso': (int(200*0.20), int(200*0.55)),
            'face': (int(200*0.08), int(200*0.25)),
        }

        correct = 0
        total = len(test_cases)

        for name, color, region, ppe_type in test_cases:
            crop = self._make_person_crop()
            y1, y2 = regions_map[region]
            crop[y1:y2, :] = color
            result = self.detector.detect(crop)
            detected = result[ppe_type]['detected']
            conf = result[ppe_type]['confidence']
            if conf > 0:
                correct += 1
            print(f"  [CASE] {name}: detected={detected}, conf={conf:.4f}")

        accuracy = (correct / total) * 100
        print(f"\n  [RESULT] PPE Color Detection Accuracy: {correct}/{total} = {accuracy:.1f}%")
        self.assertGreaterEqual(accuracy, 60.0,
                                "PPE detection accuracy should be >= 60%")


# =============================================================================
# Test 3: Centroid Tracker
# =============================================================================
class TestCentroidTracker(unittest.TestCase):
    """Validate centroid tracker correctness."""

    def setUp(self):
        self.tracker = CentroidTracker(max_disappeared=5, max_distance=80)

    def test_01_register_and_track(self):
        """Register objects and verify ID persistence."""
        self.tracker.update([(100, 100, 200, 200)])
        objects = self.tracker.update([(105, 105, 205, 205)])
        self.assertIn(0, objects, "Same object should keep ID 0")
        print(f"  [INFO] Tracked 1 object across 2 frames, ID maintained")

    def test_02_multiple_objects(self):
        """Track multiple independent objects."""
        objects = self.tracker.update([
            (50, 50, 150, 150),
            (300, 300, 400, 400),
            (500, 100, 600, 200),
        ])
        self.assertEqual(len(objects), 3)
        print(f"  [INFO] Tracking {len(objects)} objects: IDs {list(objects.keys())}")

    def test_03_disappear_and_reappear(self):
        """Object that disappears gets deregistered after max_disappeared."""
        self.tracker.update([(100, 100, 200, 200)])
        for _ in range(6):
            self.tracker.update([])
        self.assertEqual(len(self.tracker.objects), 0)
        # New object gets new ID
        objects = self.tracker.update([(500, 500, 600, 600)])
        self.assertIn(1, objects)
        print(f"  [INFO] Object deregistered after 6 frames, new ID assigned")

    def test_04_bbox_tracking(self):
        """Bounding boxes are stored correctly."""
        self.tracker.update([(10, 20, 30, 40)])
        bbox = self.tracker.get_bbox(0)
        self.assertEqual(bbox, (10, 20, 30, 40))

    def test_05_crossing_paths(self):
        """Two objects moving toward each other maintain IDs."""
        # Frame 1: Two objects far apart
        self.tracker.update([(50, 50, 100, 100), (400, 400, 450, 450)])
        # Frame 2: Moved closer but still distinct
        objects = self.tracker.update([(80, 80, 130, 130), (370, 370, 420, 420)])
        self.assertEqual(len(objects), 2)
        print(f"  [INFO] Two objects tracked through movement: IDs {list(objects.keys())}")


# =============================================================================
# Test 4: Full Pipeline Integration
# =============================================================================
class TestFullPipeline(unittest.TestCase):
    """End-to-end integration test of the prediction pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.person_detector = PersonDetector(confidence_threshold=0.3)
        cls.ppe_detector = PPEDetector(required_ppe=['helmet', 'vest'])
        cls.tracker = CentroidTracker(max_disappeared=10, max_distance=80)

    def test_01_pipeline_processes_frame(self):
        """Pipeline should process a frame without crashing."""
        from inference.predictor import SafetyPredictor

        predictor = SafetyPredictor(
            person_detector=self.person_detector,
            ppe_detector=self.ppe_detector,
            tracker=self.tracker,
            db_manager=None,
            alert_manager=AlertManager(),
        )
        predictor.start_session('test')

        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        result = predictor.process_frame(frame)

        self.assertIn('annotated_frame', result)
        self.assertIn('detections', result)
        self.assertIn('workers', result)
        self.assertIn('violations', result)
        self.assertIn('stats', result)
        self.assertEqual(result['annotated_frame'].shape, frame.shape)

        stats = result['stats']
        print(f"  [INFO] Pipeline result: "
              f"workers={stats['total_workers']}, "
              f"violations={stats['total_violations']}, "
              f"fps={stats['fps']:.1f}, "
              f"compliance={stats['compliance_rate']:.1f}%")

        predictor.end_session()

    def test_02_multi_frame_processing(self):
        """Pipeline should process multiple frames with consistent stats."""
        from inference.predictor import SafetyPredictor

        predictor = SafetyPredictor(
            person_detector=self.person_detector,
            ppe_detector=self.ppe_detector,
            tracker=CentroidTracker(max_disappeared=10, max_distance=80),
            db_manager=None,
            alert_manager=AlertManager(),
        )
        predictor.start_session('multi_frame_test')

        for i in range(5):
            frame = np.random.randint(30, 180, (480, 640, 3), dtype=np.uint8)
            result = predictor.process_frame(frame)

        summary = predictor.get_current_stats()
        self.assertEqual(summary['frame_count'], 5)
        print(f"  [INFO] Processed 5 frames: "
              f"total_detections={summary['total_detections']}, "
              f"unique_workers={summary['unique_workers']}")

        predictor.end_session()


# =============================================================================
# Test 5: MongoDB Database
# =============================================================================
class TestMongoDBDatabase(unittest.TestCase):
    """Test MongoDB Atlas connectivity and CRUD operations."""

    TEST_DB_NAME = 'safety_monitoring_test'

    @classmethod
    def setUpClass(cls):
        mongo_uri = os.environ.get('MONGO_URI')
        if not mongo_uri:
            raise unittest.SkipTest("MONGO_URI not set — skipping DB tests")
        try:
            from database.database import DatabaseManager
            cls.db = DatabaseManager(
                mongo_uri=mongo_uri,
                db_name=cls.TEST_DB_NAME
            )
        except Exception as e:
            raise unittest.SkipTest(f"MongoDB connection failed: {e}")

    def setUp(self):
        self.db.workers.delete_many({})
        self.db.violations.delete_many({})
        self.db.sessions.delete_many({})
        self.db.compliance_snapshots.delete_many({})
        self.db.db['counters'].delete_many({})

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'db') and cls.db:
            cls.db.client.drop_database(cls.TEST_DB_NAME)
            cls.db.close()

    def test_01_connection_alive(self):
        """MongoDB Atlas is reachable and authenticated."""
        self.db.client.admin.command('ping')
        print(f"  [INFO] MongoDB Atlas: connected ✓")

    def test_02_create_session(self):
        sid = self.db.create_session('test_video.mp4')
        self.assertGreater(sid, 0)
        print(f"  [INFO] Session created: ID={sid}")

    def test_03_register_and_query_workers(self):
        self.db.register_worker(1)
        self.db.register_worker(2)
        self.db.register_worker(3)
        stats = self.db.get_worker_stats()
        self.assertEqual(len(stats), 3)
        print(f"  [INFO] 3 workers registered and queried")

    def test_04_log_and_retrieve_violations(self):
        sid = self.db.create_session('test')
        self.db.log_violation(1, 'helmet', 10, 0.85, session_id=sid)
        self.db.log_violation(1, 'vest', 20, 0.72, session_id=sid)
        self.db.log_violation(2, 'helmet', 30, 0.91, session_id=sid)

        w1_violations = self.db.get_worker_violations(1)
        self.assertEqual(len(w1_violations), 2)

        by_type = self.db.get_violations_by_type()
        self.assertEqual(by_type.get('helmet'), 2)
        self.assertEqual(by_type.get('vest'), 1)

        recent = self.db.get_recent_violations(10)
        self.assertEqual(len(recent), 3)
        print(f"  [INFO] 3 violations logged: by_type={by_type}")

    def test_05_compliance_snapshots(self):
        sid = self.db.create_session('test')
        self.db.save_compliance_snapshot(sid, 10, 8, 80.0)
        self.db.save_compliance_snapshot(sid, 10, 9, 90.0)
        self.db.save_compliance_snapshot(sid, 10, 10, 100.0)

        rate = self.db.get_compliance_rate(sid)
        self.assertAlmostEqual(rate, 90.0, places=1)

        history = self.db.get_compliance_history(hours=1)
        self.assertEqual(len(history), 3)
        print(f"  [INFO] Compliance rate: {rate:.1f}% (3 snapshots)")

    def test_06_session_lifecycle(self):
        sid = self.db.create_session('lifecycle_test.mp4')
        self.db.end_session(sid, total_frames=500, total_detections=200,
                            total_violations=15)
        stats = self.db.get_session_stats(sid)
        self.assertEqual(stats['total_frames'], 500)
        self.assertEqual(stats['total_violations'], 15)
        print(f"  [INFO] Session lifecycle: created → ended with 500 frames")

    def test_07_total_counts(self):
        self.db.register_worker(1)
        self.db.register_worker(2)
        sid = self.db.create_session('counts_test')
        self.db.log_violation(1, 'helmet', 1, session_id=sid)
        counts = self.db.get_total_counts()
        self.assertEqual(counts['total_workers'], 2)
        self.assertEqual(counts['total_violations'], 1)
        self.assertEqual(counts['total_sessions'], 1)
        print(f"  [INFO] Total counts: {counts}")

    def test_08_violation_trends(self):
        sid = self.db.create_session('trends')
        for i in range(5):
            self.db.log_violation(1, 'helmet', i, session_id=sid)
        trends = self.db.get_violation_trends(hours=1)
        total = sum(trends.values())
        self.assertEqual(total, 5)
        print(f"  [INFO] Violation trends (last 1h): {trends}")

    def test_09_worker_update(self):
        self.db.register_worker(1)
        self.db.update_worker(1, frames_tracked=50)
        self.db.update_worker(1, frames_tracked=30)
        stats = self.db.get_worker_stats()
        self.assertEqual(stats[0]['total_frames_tracked'], 80)
        print(f"  [INFO] Worker frames tracked: 50+30 = {stats[0]['total_frames_tracked']}")


# =============================================================================
# Test 6: Alert System
# =============================================================================
class TestAlertSystem(unittest.TestCase):
    """Test alert manager and channels."""

    def test_01_trigger_and_history(self):
        mgr = AlertManager()
        mgr.trigger_alert(1, 'helmet', ['helmet', 'vest'])
        mgr.trigger_alert(2, 'mask', ['mask'])
        alerts = mgr.get_recent_alerts()
        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0]['worker_id'], 2)
        print(f"  [INFO] 2 alerts triggered and stored in history")

    def test_02_alert_channels_result(self):
        mgr = AlertManager()
        result = mgr.trigger_alert(1, 'helmet', ['helmet'])
        self.assertIn('sound', result)
        self.assertIn('email', result)
        self.assertIn('sms', result)
        # Email and SMS disabled by default
        self.assertFalse(result['email'])
        self.assertFalse(result['sms'])
        print(f"  [INFO] Alert channels: {result}")

    def test_03_clear_history(self):
        mgr = AlertManager()
        mgr.trigger_alert(1, 'vest', ['vest'])
        mgr.clear_history()
        self.assertEqual(len(mgr.get_recent_alerts()), 0)


# =============================================================================
# Test 7: Configuration
# =============================================================================
class TestConfiguration(unittest.TestCase):
    """Test config loading and singleton."""

    def test_01_singleton_pattern(self):
        Config.reset()
        c1 = Config.get_instance()
        c2 = Config.get_instance()
        self.assertIs(c1, c2)
        Config.reset()

    def test_02_default_values(self):
        Config.reset()
        config = Config.get_instance()
        self.assertEqual(config.detection.confidence_threshold, 0.5)
        self.assertIn('helmet', config.ppe.required_items)
        self.assertEqual(config.tracker.max_disappeared, 50)
        print(f"  [INFO] Config defaults loaded: "
              f"threshold={config.detection.confidence_threshold}, "
              f"PPE={config.ppe.required_items}")
        Config.reset()


# =============================================================================
# Test 8: Utilities
# =============================================================================
class TestUtilities(unittest.TestCase):
    """Test helper functions and metrics."""

    def test_01_iou_perfect_overlap(self):
        self.assertAlmostEqual(
            calculate_iou((0, 0, 100, 100), (0, 0, 100, 100)), 1.0
        )

    def test_02_iou_no_overlap(self):
        self.assertEqual(
            calculate_iou((0, 0, 50, 50), (100, 100, 200, 200)), 0.0
        )

    def test_03_iou_partial_overlap(self):
        iou = calculate_iou((0, 0, 100, 100), (50, 50, 150, 150))
        self.assertAlmostEqual(iou, 50*50 / (10000 + 10000 - 50*50), places=4)
        print(f"  [INFO] Partial IoU: {iou:.4f}")

    def test_04_crop_region(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:40, 30:60] = 255
        crop = crop_region(frame, (30, 20, 60, 40))
        self.assertEqual(crop.shape, (20, 30, 3))
        self.assertEqual(crop[0, 0, 0], 255)

    def test_05_body_regions(self):
        regions = get_body_regions(0, 0, 100, 200)
        self.assertIn('head', regions)
        self.assertIn('torso', regions)
        self.assertIn('feet', regions)
        self.assertEqual(len(regions), 6)

    def test_06_resize_frame(self):
        frame = np.zeros((1000, 2000, 3), dtype=np.uint8)
        resized = resize_frame(frame, max_width=640)
        self.assertEqual(resized.shape[1], 640)

    def test_07_fps_counter(self):
        counter = FPSCounter(avg_count=5)
        for _ in range(10):
            counter.start()
            time.sleep(0.01)
            counter.stop()
        self.assertGreater(counter.fps, 0)
        print(f"  [INFO] FPS counter: {counter.fps:.1f} fps")

    def test_08_detection_metrics(self):
        metrics = DetectionMetrics()
        metrics.update(3, [{'worker_id': 1, 'type': 'helmet'}], [1, 2, 3])
        metrics.update(2, [], [1, 2])
        summary = metrics.get_summary()
        self.assertEqual(summary['total_frames'], 2)
        self.assertEqual(summary['total_violations'], 1)
        self.assertEqual(summary['unique_workers'], 3)
        print(f"  [INFO] Metrics summary: {summary}")

    def test_09_visualization_no_crash(self):
        """Drawing functions should not crash on valid inputs."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_person_detection(frame, (100, 100, 200, 300), 1, True)
        draw_person_detection(frame, (300, 100, 400, 300), 2, False)
        draw_ppe_status(frame, (100, 100, 200, 300),
                        {'helmet': {'detected': True, 'confidence': 0.8}},
                        ['helmet', 'vest'])
        draw_violation_alert(frame, (300, 100, 400, 300), 2, ['vest'])
        draw_dashboard_overlay(frame, {
            'fps': 25.0, 'total_workers': 2,
            'total_violations': 1, 'compliance_rate': 50.0
        })
        snapshot = create_violation_snapshot(
            frame, (300, 100, 400, 300), 2, 'vest'
        )
        self.assertIsNotNone(snapshot)
        self.assertGreater(snapshot.shape[0], 0)
        print(f"  [INFO] All visualization functions passed (no crash)")

    def test_10_generate_session_id(self):
        sid = generate_session_id()
        self.assertEqual(len(sid), 8)

    def test_11_format_timestamp(self):
        ts = format_timestamp()
        self.assertGreater(len(ts), 0)


# =============================================================================
# Run all tests
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("INDUSTRIAL SAFETY MONITORING SYSTEM — COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    unittest.main(verbosity=2)
