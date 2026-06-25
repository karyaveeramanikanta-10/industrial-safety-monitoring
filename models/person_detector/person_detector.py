"""
Person detection module using SSD MobileNet V2.

Provides a multi-tier fallback detection system:
1. TensorFlow SavedModel (best accuracy)
2. OpenCV HOG person detector (no model download needed)

The system works out of the box with the HOG fallback and automatically
upgrades when the TensorFlow model is available.
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger('safety_monitor')


class PersonDetector:
    """Detect persons in video frames using SSD MobileNet V2.

    Supports multiple backends with automatic fallback:
    - TensorFlow 2 SavedModel (highest accuracy)
    - OpenCV HOG descriptor (zero-dependency fallback)

    Usage:
        detector = PersonDetector(confidence_threshold=0.5)
        detections = detector.detect(frame)
        # Returns: [{'bbox': (x1,y1,x2,y2), 'confidence': 0.95}, ...]
    """

    PERSON_CLASS_ID = 1  # COCO class ID for 'person'
    MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'

    def __init__(self, model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5):
        """Initialize person detector.

        Args:
            model_path: Path to saved_model directory. Auto-detected if None.
            confidence_threshold: Minimum confidence score for detections.
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.detect_fn = None
        self._backend = 'none'
        self._hog = None
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None):
        """Load detection model with tiered fallback.

        Priority: TensorFlow SavedModel -> OpenCV HOG
        """
        # Try TensorFlow SavedModel
        if self._try_load_tensorflow(model_path):
            return

        # Fallback to OpenCV HOG
        self._load_hog_detector()

    def _try_load_tensorflow(self, model_path: Optional[str] = None) -> bool:
        """Attempt to load TensorFlow SavedModel.

        Args:
            model_path: Path to saved_model directory.

        Returns:
            True if loaded successfully.
        """
        try:
            import tensorflow as tf

            # Search paths for the model
            search_paths = []
            if model_path:
                search_paths.append(model_path)
                search_paths.append(os.path.join(model_path, 'saved_model'))

            # Default locations
            search_paths.extend([
                os.path.join('models', 'person_detector', self.MODEL_NAME,
                             'saved_model'),
                os.path.join('models', 'person_detector', 'saved_model'),
                os.path.join('models', 'person_detector',
                             'ssd_person_model'),
            ])

            for path in search_paths:
                if os.path.exists(path):
                    logger.info(f"Loading TensorFlow model from {path}...")
                    self.model = tf.saved_model.load(path)
                    self.detect_fn = self.model
                    self._backend = 'tensorflow'
                    logger.info("TensorFlow SSD MobileNet V2 loaded "
                                "successfully")
                    return True

            logger.info("TensorFlow model not found on disk. "
                        "Falling back to HOG detector.")
            return False

        except ImportError:
            logger.warning("TensorFlow not installed. "
                           "Using HOG fallback detector.")
            return False
        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}. "
                         f"Using HOG fallback.")
            return False

    def _load_hog_detector(self):
        """Load OpenCV HOG person detector as fallback."""
        try:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(
                cv2.HOGDescriptor_getDefaultPeopleDetector()
            )
            self._backend = 'hog'
            logger.info("OpenCV HOG person detector loaded (fallback mode)")
        except Exception as e:
            logger.error(f"Failed to load HOG detector: {e}")
            self._backend = 'none'

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect persons in a frame.

        Args:
            frame: BGR image (numpy array).

        Returns:
            List of dicts: [{'bbox': (x1,y1,x2,y2), 'confidence': float}]
        """
        if frame is None or frame.size == 0:
            return []

        if self._backend == 'tensorflow':
            return self._detect_tf(frame)
        elif self._backend == 'hog':
            return self._detect_hog(frame)
        else:
            logger.warning("No detection backend available")
            return []

    def _detect_tf(self, frame: np.ndarray) -> List[Dict]:
        """Detection using TensorFlow SavedModel.

        Args:
            frame: BGR image.

        Returns:
            List of person detections.
        """
        import tensorflow as tf

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        detections = self.detect_fn(input_tensor)

        # Extract results
        boxes = detections['detection_boxes'].numpy()[0]
        classes = detections['detection_classes'].numpy()[0]
        scores = detections['detection_scores'].numpy()[0]
        num_det = int(detections['num_detections'].numpy()[0])

        h, w = frame.shape[:2]
        results = []

        for i in range(num_det):
            if (int(classes[i]) == self.PERSON_CLASS_ID and
                    scores[i] >= self.confidence_threshold):
                ymin, xmin, ymax, xmax = boxes[i]
                bbox = (
                    int(xmin * w), int(ymin * h),
                    int(xmax * w), int(ymax * h)
                )
                results.append({
                    'bbox': bbox,
                    'confidence': float(scores[i])
                })

        return results

    def _detect_hog(self, frame: np.ndarray) -> List[Dict]:
        """Detection using OpenCV HOG descriptor.

        Args:
            frame: BGR image.

        Returns:
            List of person detections.
        """
        if self._hog is None:
            return []

        # HOG needs minimum image size (at least 64x128)
        h, w = frame.shape[:2]
        if h < 128 or w < 64:
            return []

        scale = 1.0
        if w > 640:
            scale = 640 / w
            resized = cv2.resize(frame, (640, int(h * scale)))
        else:
            resized = frame

        # Detect people
        rects, weights = self._hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )

        results = []
        for (x, y, bw, bh), weight in zip(rects, weights):
            confidence = float(min(weight[0] / 2.0, 1.0))
            if confidence >= self.confidence_threshold:
                # Scale back to original size
                bbox = (
                    int(x / scale), int(y / scale),
                    int((x + bw) / scale), int((y + bh) / scale)
                )
                results.append({
                    'bbox': bbox,
                    'confidence': confidence
                })

        # Apply non-maximum suppression
        if results:
            results = self._nms(results, threshold=0.4)

        return results

    @staticmethod
    def _nms(detections: List[Dict], threshold: float = 0.4) -> List[Dict]:
        """Apply Non-Maximum Suppression to filter overlapping detections.

        Args:
            detections: List of detection dicts.
            threshold: IoU threshold for suppression.

        Returns:
            Filtered detections list.
        """
        if not detections:
            return []

        boxes = np.array([d['bbox'] for d in detections], dtype=np.float32)
        scores = np.array([d['confidence'] for d in detections])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]

    @property
    def backend(self) -> str:
        """Get the active detection backend name."""
        return self._backend

    @property
    def is_loaded(self) -> bool:
        """Check if any detection backend is loaded."""
        return self._backend != 'none'

    def download_model(self):
        """Download SSD model from TF2 Detection Model Zoo.

        Downloads and extracts the SSD MobileNet V2 model to the
        default model directory.
        """
        import urllib.request
        import tarfile

        url = (
            'http://download.tensorflow.org/models/object_detection/tf2/'
            '20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
        )
        save_dir = os.path.join('models', 'person_detector')
        os.makedirs(save_dir, exist_ok=True)

        tar_path = os.path.join(save_dir, f'{self.MODEL_NAME}.tar.gz')

        logger.info(f"Downloading model from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        logger.info("Download complete. Extracting...")

        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(save_dir)

        os.remove(tar_path)
        logger.info(f"Model extracted to {save_dir}/{self.MODEL_NAME}")

        # Reload with TensorFlow
        self._try_load_tensorflow()
