"""
Video file processing for safety monitoring.

Processes recorded video files with progress tracking and optional
output video writing.
"""

import cv2
import time
import os
import logging
import threading
from typing import Optional, Dict

logger = logging.getLogger('safety_monitor')


class VideoProcessor:
    """Process recorded video files for safety monitoring.

    Supports any video format readable by OpenCV. Runs processing
    in a background thread and tracks progress.

    Usage:
        processor = VideoProcessor(predictor, 'video.mp4')
        processor.start()
        while not processor.is_complete:
            frame = processor.get_frame()
            progress = processor.progress
        processor.stop()
    """

    def __init__(self, predictor, video_path: str,
                 output_path: Optional[str] = None):
        """Initialize video processor.

        Args:
            predictor: SafetyPredictor instance.
            video_path: Path to the input video file.
            output_path: Optional path for annotated output video.
        """
        self.predictor = predictor
        self.video_path = video_path
        self.output_path = output_path
        self.cap = None
        self.writer = None
        self.running = False
        self.progress = 0.0
        self.total_frames = 0
        self.processed_frames = 0
        self.current_frame = None
        self.current_results = None
        self.lock = threading.Lock()
        self._thread = None

    def start(self):
        """Start video processing."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup output video writer
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, fps, (width, height)
            )

        self.running = True
        self.processed_frames = 0
        self.progress = 0.0
        self.predictor.start_session(self.video_path)

        self._thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._thread.start()
        logger.info(
            f"Video processing started: {self.video_path} "
            f"({self.total_frames} frames)"
        )

    def _process_loop(self):
        """Main processing loop (runs in background thread)."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                self.progress = 1.0
                break

            try:
                results = self.predictor.process_frame(frame)
                annotated = results.get('annotated_frame', frame)

                with self.lock:
                    self.current_frame = annotated
                    self.current_results = results

                if self.writer:
                    self.writer.write(annotated)

            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                with self.lock:
                    self.current_frame = frame

            self.processed_frames += 1
            if self.total_frames > 0:
                self.progress = self.processed_frames / self.total_frames

            # Match original video FPS
            time.sleep(0.01)

        # Processing complete
        self._cleanup()
        logger.info(
            f"Video processing complete: {self.processed_frames} "
            f"frames processed"
        )

    def get_frame(self) -> Optional[object]:
        """Get the latest processed frame."""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def get_results(self) -> Optional[Dict]:
        """Get the latest processing results."""
        with self.lock:
            return self.current_results

    def stop(self):
        """Stop video processing."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._cleanup()

    def _cleanup(self):
        """Release video resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.writer:
            self.writer.release()
            self.writer = None
        self.predictor.end_session()

    @property
    def is_complete(self) -> bool:
        """Check if video processing is finished."""
        return not self.running and self.progress >= 1.0

    @property
    def is_running(self) -> bool:
        """Check if the processor is currently running."""
        return self.running
