"""
Utility helper functions for Industrial Safety Monitoring System.

Provides common operations like frame resizing, directory setup,
body region extraction, and IoU calculation.
"""

import os
import uuid
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional


def resize_frame(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """Resize frame maintaining aspect ratio.

    Args:
        frame: Input BGR image.
        max_width: Maximum width in pixels.

    Returns:
        Resized frame.
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    ratio = max_width / w
    new_h = int(h * ratio)
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def generate_session_id() -> str:
    """Generate a unique session identifier.

    Returns:
        UUID-based session string.
    """
    return str(uuid.uuid4())[:8]


def format_timestamp(ts=None, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Format a timestamp for display.

    Args:
        ts: datetime object or None for current time.
        fmt: strftime format string.

    Returns:
        Formatted timestamp string.
    """
    if ts is None:
        ts = datetime.now()
    elif isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return ts
    return ts.strftime(fmt)


def ensure_directories():
    """Create all required project directories."""
    dirs = [
        'data/raw',
        'data/processed',
        'data/annotations',
        'data/violation_logs',
        'datasets/train/images',
        'datasets/train/labels',
        'datasets/val/images',
        'datasets/val/labels',
        'datasets/test/images',
        'datasets/test/labels',
        'logs',
        'static/images',
        'static/icons',
        'static/css',
        'database',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """Calculate Intersection over Union between two bounding boxes.

    Args:
        box1: (x1, y1, x2, y2) first bounding box.
        box2: (x1, y1, x2, y2) second bounding box.

    Returns:
        IoU value between 0.0 and 1.0.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union


def crop_region(frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
    """Safely crop a region from frame with boundary checks.

    Args:
        frame: Input image.
        bbox: (x1, y1, x2, y2) bounding box.

    Returns:
        Cropped region or None if invalid.
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def get_body_regions(x1: int, y1: int, x2: int, y2: int) -> Dict[str, Tuple]:
    """Split a person bounding box into body regions for PPE detection.

    Divides the person bounding box proportionally into head, face,
    torso, hands, ears, and feet regions.

    Args:
        x1, y1, x2, y2: Person bounding box coordinates.

    Returns:
        Dict mapping region name to (x1, y1, x2, y2) coordinates.
    """
    h = y2 - y1
    w = x2 - x1

    return {
        'head': (
            x1 + int(w * 0.15), y1,
            x2 - int(w * 0.15), y1 + int(h * 0.15)
        ),
        'face': (
            x1 + int(w * 0.25), y1 + int(h * 0.08),
            x2 - int(w * 0.25), y1 + int(h * 0.25)
        ),
        'ears': (
            x1, y1 + int(h * 0.05),
            x2, y1 + int(h * 0.18)
        ),
        'torso': (
            x1 + int(w * 0.05), y1 + int(h * 0.20),
            x2 - int(w * 0.05), y1 + int(h * 0.55)
        ),
        'hands': (
            x1, y1 + int(h * 0.45),
            x2, y1 + int(h * 0.70)
        ),
        'feet': (
            x1 + int(w * 0.1), y1 + int(h * 0.85),
            x2 - int(w * 0.1), y2
        ),
    }


def get_timestamp_filename(prefix: str = 'snapshot', ext: str = '.jpg') -> str:
    """Generate a timestamped filename.

    Args:
        prefix: Filename prefix.
        ext: File extension.

    Returns:
        Filename string like 'snapshot_20240115_143052.jpg'.
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{ts}{ext}"


def clamp(value, min_val, max_val):
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))
