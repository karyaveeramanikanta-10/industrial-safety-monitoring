"""
Main prediction pipeline for Industrial Safety Monitoring System.

Orchestrates person detection, tracking, PPE detection, compliance
checking, violation logging, and alert triggering in a unified pipeline.
"""

import cv2
import numpy as np
import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from utils.helpers import crop_region, get_body_regions, get_timestamp_filename
from utils.visualization import (
    draw_person_detection, draw_ppe_status,
    draw_violation_alert, draw_dashboard_overlay,
    create_violation_snapshot
)
from utils.metrics import FPSCounter, DetectionMetrics

logger = logging.getLogger('safety_monitor')


class SafetyPredictor:
    """Main prediction pipeline for real-time safety monitoring.

    Orchestrates the complete processing pipeline per frame:
    1. Person detection (SSD MobileNet V2 / HOG)
    2. Worker tracking (Centroid / SORT)
    3. PPE detection (Color/Region heuristics)
    4. Compliance checking
    5. Violation logging & alerts
    6. Frame annotation

    Usage:
        predictor = SafetyPredictor(detector, ppe_detector, tracker, db, alerts)
        predictor.start_session('webcam')
        results = predictor.process_frame(frame)
    """

    def __init__(self, person_detector, ppe_detector, tracker,
                 db_manager=None, alert_manager=None, config=None):
        """Initialize the safety predictor pipeline.

        Args:
            person_detector: PersonDetector instance.
            ppe_detector: PPEDetector instance.
            tracker: CentroidTracker or SORTTracker instance.
            db_manager: Optional DatabaseManager for violation logging.
            alert_manager: Optional AlertManager for alerts.
            config: Optional Config instance.
        """
        self.person_detector = person_detector
        self.ppe_detector = ppe_detector
        self.tracker = tracker
        self.db_manager = db_manager
        self.alert_manager = alert_manager
        self.config = config

        self.frame_count = 0
        self.session_id = None
        self.worker_ppe_history: Dict[int, List] = {}
        self.violation_cooldown: Dict[Tuple, float] = {}
        self.fps_counter = FPSCounter()
        self.metrics = DetectionMetrics()

        # Cooldown settings
        self._cooldown_seconds = 30
        if config and hasattr(config, 'alerts'):
            self._cooldown_seconds = getattr(
                config.alerts, 'cooldown_seconds', 30
            )

    def start_session(self, video_source: str = 'webcam'):
        """Initialize a new monitoring session.

        Args:
            video_source: Description of the video source.
        """
        if self.db_manager:
            self.session_id = self.db_manager.create_session(str(video_source))
        self.frame_count = 0
        self.worker_ppe_history.clear()
        self.violation_cooldown.clear()
        self.metrics.reset()
        self.tracker.reset()
        logger.info(f"Started monitoring session {self.session_id}")

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single video frame through the full pipeline.

        Args:
            frame: BGR image (numpy array).

        Returns:
            Dict with keys:
                'annotated_frame': Annotated BGR image
                'detections': List of person detections
                'workers': Dict of worker results
                'violations': List of new violations
                'stats': Real-time statistics
        """
        self.fps_counter.start()
        self.frame_count += 1
        annotated = frame.copy()
        violations = []
        workers = {}

        # Step 1: Detect persons
        detections = self.person_detector.detect(frame)

        # Step 2: Update tracker with detection bounding boxes
        rects = [d['bbox'] for d in detections]
        tracked_objects = self.tracker.update(rects)

        # Step 3: For each tracked worker, run PPE detection
        compliant_count = 0
        worker_ids = []

        for object_id, centroid in tracked_objects.items():
            bbox = self.tracker.get_bbox(object_id)
            if bbox is None:
                continue

            worker_ids.append(object_id)

            # Register worker in database
            if self.db_manager:
                self.db_manager.register_worker(object_id)
                self.db_manager.update_worker(object_id)

            # Crop person region
            person_crop = crop_region(frame, bbox)
            if person_crop is None or person_crop.size == 0:
                continue

            # Run PPE detection
            ppe_status = self.ppe_detector.detect(person_crop)
            compliance = self.ppe_detector.check_compliance(ppe_status)

            is_compliant = compliance['compliant']
            missing_ppe = compliance['missing']

            if is_compliant:
                compliant_count += 1

            # Store worker result
            workers[object_id] = {
                'bbox': bbox,
                'centroid': centroid,
                'ppe_status': ppe_status,
                'compliant': is_compliant,
                'missing_ppe': missing_ppe,
                'present_ppe': compliance['present'],
            }

            # Step 4: Draw annotations
            draw_person_detection(annotated, bbox, object_id, is_compliant)
            draw_ppe_status(
                annotated, bbox, ppe_status,
                self.ppe_detector.required_ppe
            )

            # Step 5: Handle violations
            if not is_compliant:
                draw_violation_alert(annotated, bbox, object_id, missing_ppe)
                new_violations = self._handle_violation(
                    object_id, missing_ppe, frame, bbox
                )
                violations.extend(new_violations)

            # Update PPE history for worker
            if object_id not in self.worker_ppe_history:
                self.worker_ppe_history[object_id] = []
            self.worker_ppe_history[object_id].append(ppe_status)
            if len(self.worker_ppe_history[object_id]) > 30:
                self.worker_ppe_history[object_id] = \
                    self.worker_ppe_history[object_id][-30:]

        # Step 6: Compute stats
        fps = self.fps_counter.stop()
        total_workers = len(tracked_objects)
        compliance_rate = (
            (compliant_count / total_workers * 100)
            if total_workers > 0 else 100.0
        )

        stats = {
            'total_workers': total_workers,
            'compliant_workers': compliant_count,
            'total_violations': self.metrics.total_violations + len(violations),
            'fps': fps,
            'compliance_rate': compliance_rate,
            'frame_number': self.frame_count,
        }

        # Update metrics
        self.metrics.update(
            len(detections),
            [{'worker_id': v['worker_id'], 'type': v['type']}
             for v in violations],
            worker_ids
        )

        # Draw dashboard overlay
        draw_dashboard_overlay(annotated, stats)

        # Save compliance snapshot periodically (every 30 frames)
        if (self.db_manager and self.session_id and
                self.frame_count % 30 == 0 and total_workers > 0):
            self.db_manager.save_compliance_snapshot(
                self.session_id, total_workers,
                compliant_count, compliance_rate
            )

        return {
            'annotated_frame': annotated,
            'detections': detections,
            'workers': workers,
            'violations': violations,
            'stats': stats,
        }

    def _check_violation_cooldown(self, worker_id: int,
                                    violation_type: str) -> bool:
        """Check if enough time has passed since last alert.

        Args:
            worker_id: Worker tracking ID.
            violation_type: Type of PPE violation.

        Returns:
            True if alert should fire (cooldown expired).
        """
        key = (worker_id, violation_type)
        now = time.time()
        last_time = self.violation_cooldown.get(key, 0)

        if now - last_time >= self._cooldown_seconds:
            self.violation_cooldown[key] = now
            return True
        return False

    def _handle_violation(self, worker_id: int, missing_ppe: List[str],
                           frame: np.ndarray, bbox: Tuple) -> List[Dict]:
        """Handle detected PPE violations.

        Args:
            worker_id: Worker tracking ID.
            missing_ppe: List of missing PPE item names.
            frame: Current video frame.
            bbox: Worker bounding box.

        Returns:
            List of violation dicts that were actually triggered.
        """
        triggered = []

        for ppe_item in missing_ppe:
            if not self._check_violation_cooldown(worker_id, ppe_item):
                continue

            violation = {
                'worker_id': worker_id,
                'type': ppe_item,
                'frame_number': self.frame_count,
                'timestamp': datetime.now().isoformat(),
            }

            # Save violation snapshot
            snapshot_path = None
            try:
                snapshot = create_violation_snapshot(
                    frame, bbox, worker_id, ppe_item
                )
                snapshot_dir = os.path.join('data', 'violation_logs')
                os.makedirs(snapshot_dir, exist_ok=True)
                snapshot_filename = get_timestamp_filename(
                    f'violation_w{worker_id}_{ppe_item}'
                )
                snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
                cv2.imwrite(snapshot_path, snapshot)
                violation['snapshot_path'] = snapshot_path
            except Exception as e:
                logger.error(f"Failed to save violation snapshot: {e}")

            # Log to database
            if self.db_manager:
                try:
                    self.db_manager.log_violation(
                        worker_id=worker_id,
                        violation_type=ppe_item,
                        frame_number=self.frame_count,
                        confidence=0.0,
                        snapshot_path=snapshot_path,
                        session_id=self.session_id,
                    )
                except Exception as e:
                    logger.error(f"Failed to log violation to DB: {e}")

            # Trigger alerts
            if self.alert_manager:
                try:
                    self.alert_manager.trigger_alert(
                        worker_id=worker_id,
                        violation_type=ppe_item,
                        missing_ppe=missing_ppe,
                        timestamp=violation['timestamp'],
                        frame=frame,
                    )
                except Exception as e:
                    logger.error(f"Failed to trigger alert: {e}")

            triggered.append(violation)
            logger.warning(
                f"VIOLATION: Worker #{worker_id} missing {ppe_item} "
                f"at frame {self.frame_count}"
            )

        return triggered

    def end_session(self):
        """End the current monitoring session."""
        if self.db_manager and self.session_id:
            summary = self.metrics.get_summary()
            self.db_manager.end_session(
                self.session_id,
                total_frames=summary['total_frames'],
                total_detections=summary['total_detections'],
                total_violations=summary['total_violations'],
            )
        logger.info(
            f"Session {self.session_id} ended. "
            f"Frames: {self.frame_count}, "
            f"Violations: {self.metrics.total_violations}"
        )

    def get_current_stats(self) -> Dict:
        """Get current monitoring statistics."""
        summary = self.metrics.get_summary()
        summary['fps'] = self.fps_counter.fps
        summary['session_id'] = self.session_id
        summary['frame_count'] = self.frame_count
        return summary
