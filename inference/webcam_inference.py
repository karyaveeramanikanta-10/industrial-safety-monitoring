"""
Webcam-based video processing for safety monitoring.

Provides threaded webcam capture with non-blocking frame access
for integration with the Streamlit dashboard.
"""

import cv2
import time
import logging
import threading
from typing import Optional, Dict

logger = logging.getLogger('safety_monitor')


class WebcamProcessor:
    """Process live webcam feed for safety monitoring.

    Runs video capture and processing in a separate daemon thread
    for non-blocking operation with the Streamlit UI.

    Usage:
        processor = WebcamProcessor(predictor, source=0)
        processor.start()
        while processor.is_running:
            frame = processor.get_frame()
            results = processor.get_results()
        processor.stop()
    """

    def __init__(self, predictor, source: int = 0,
                 width: int = 640, height: int = 480):
        """Initialize webcam processor.

        Args:
            predictor: SafetyPredictor instance.
            source: Camera device index (0 = default webcam).
            width: Capture width in pixels.
            height: Capture height in pixels.
        """
        self.predictor = predictor
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.current_frame = None
        self.current_results = None
        self.lock = threading.Lock()
        self._thread = None

    def start(self):
        """Start webcam capture and processing."""
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam {self.source}. "
                f"Check that a camera is connected."
            )

        self.running = True
        self.predictor.start_session(f'webcam:{self.source}')

        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._thread.start()
        logger.info(f"Webcam {self.source} started ({self.width}x{self.height})")

    def _capture_loop(self):
        """Main capture and processing loop (runs in background thread)."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from webcam")
                time.sleep(0.1)
                continue

            try:
                results = self.predictor.process_frame(frame)
                with self.lock:
                    self.current_frame = results.get('annotated_frame', frame)
                    self.current_results = results
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                with self.lock:
                    self.current_frame = frame
                    self.current_results = None

            time.sleep(0.01)

    def get_frame(self) -> Optional[object]:
        """Get the latest processed frame (thread-safe).

        Returns:
            Annotated BGR frame or None if no frame available.
        """
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def get_results(self) -> Optional[Dict]:
        """Get the latest processing results (thread-safe).

        Returns:
            Results dict from predictor or None.
        """
        with self.lock:
            return self.current_results

    def stop(self):
        """Stop webcam capture and cleanup."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.predictor.end_session()
        logger.info("Webcam stopped")

    @property
    def is_running(self) -> bool:
        """Check if the processor is currently running."""
        return self.running
