"""
Visualization utilities for Industrial Safety Monitoring System.

Provides functions for drawing bounding boxes, PPE status indicators,
violation alerts, and dashboard overlays on video frames.
"""

import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional


# Color scheme (BGR format for OpenCV)
COLORS = {
    'compliant':   (118, 230, 0),     # Green
    'violation':   (68, 23, 255),      # Red
    'warning':     (0, 196, 255),      # Amber
    'info':        (235, 158, 52),     # Blue
    'text_bg':     (30, 30, 30),       # Dark gray
    'white':       (255, 255, 255),
    'black':       (0, 0, 0),
    'person_box':  (0, 255, 0),
    'overlay_bg':  (20, 20, 40),
}

# PPE status text labels
PPE_LABELS = {
    'helmet': 'Helmet',
    'vest': 'Vest',
    'mask': 'Mask',
    'goggles': 'Goggles',
    'gloves': 'Gloves',
    'shoes': 'Shoes',
    'ear_protection': 'Ear Prot.',
}


def draw_rounded_rect(frame: np.ndarray, pt1: Tuple, pt2: Tuple,
                       color: Tuple, thickness: int = 2,
                       radius: int = 10, fill: bool = False):
    """Draw a rounded rectangle on the frame.

    Args:
        frame: Image to draw on.
        pt1: Top-left corner (x, y).
        pt2: Bottom-right corner (x, y).
        color: BGR color tuple.
        thickness: Line thickness (-1 for filled).
        radius: Corner radius.
        fill: If True, fill the rectangle.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    if fill:
        # Draw filled rounded rectangle using overlapping rectangles and circles
        cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(frame, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(frame, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(frame, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(frame, (x2 - r, y2 - r), r, color, -1)
    else:
        # Draw outline
        cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_person_detection(frame: np.ndarray, bbox: Tuple, worker_id: int,
                           is_compliant: bool = True):
    """Draw bounding box and ID label for a detected person.

    Args:
        frame: Image to draw on.
        bbox: (x1, y1, x2, y2) bounding box.
        worker_id: Worker tracking ID.
        is_compliant: Whether the worker is PPE-compliant.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = COLORS['compliant'] if is_compliant else COLORS['violation']

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw ID label with background
    label = f"Worker #{worker_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    label_y = max(y1 - 10, th + 5)
    # Background rectangle
    cv2.rectangle(
        frame,
        (x1, label_y - th - 5),
        (x1 + tw + 10, label_y + 5),
        color, -1
    )
    # Text
    cv2.putText(frame, label, (x1 + 5, label_y - 2),
                font, font_scale, COLORS['white'], thickness)


def draw_ppe_status(frame: np.ndarray, bbox: Tuple,
                     ppe_status: Dict, required_ppe: List[str]):
    """Draw PPE status indicators next to the person bounding box.

    Shows checkmarks for detected PPE items and X marks for missing ones.

    Args:
        frame: Image to draw on.
        bbox: (x1, y1, x2, y2) bounding box.
        ppe_status: Dict of {ppe_item: {'detected': bool, 'confidence': float}}.
        required_ppe: List of required PPE item names.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    line_height = 18
    margin = 5

    # Position to the right of the bounding box
    start_x = x2 + 5
    start_y = y1 + 15

    for i, ppe_item in enumerate(required_ppe):
        y_pos = start_y + i * line_height
        label_name = PPE_LABELS.get(ppe_item, ppe_item)

        item_data = ppe_status.get(ppe_item, {})
        detected = item_data.get('detected', False)

        if detected:
            icon = "[+]"
            color = COLORS['compliant']
        else:
            icon = "[X]"
            color = COLORS['violation']

        text = f"{icon} {label_name}"

        # Draw background
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(
            frame,
            (start_x - 2, y_pos - th - 2),
            (start_x + tw + 4, y_pos + 4),
            COLORS['text_bg'], -1
        )

        cv2.putText(frame, text, (start_x, y_pos),
                    font, font_scale, color, thickness)


def draw_violation_alert(frame: np.ndarray, bbox: Tuple,
                          worker_id: int, missing_ppe: List[str]):
    """Draw a prominent violation alert overlay.

    Args:
        frame: Image to draw on.
        bbox: (x1, y1, x2, y2) bounding box.
        worker_id: Worker tracking ID.
        missing_ppe: List of missing PPE item names.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]

    # Flashing border effect
    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                  COLORS['violation'], 3)

    # Alert banner at top of bounding box
    missing_text = ", ".join(
        PPE_LABELS.get(p, p) for p in missing_ppe[:3]
    )
    if len(missing_ppe) > 3:
        missing_text += f" +{len(missing_ppe) - 3} more"

    alert_text = f"VIOLATION: Missing {missing_text}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(alert_text, font, font_scale, thickness)

    banner_y = max(y1 - 35, 0)
    # Semi-transparent red banner
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x1 - 3, banner_y),
        (max(x1 + tw + 10, x2 + 3), banner_y + th + 12),
        COLORS['violation'], -1
    )
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, alert_text,
                (x1 + 2, banner_y + th + 5),
                font, font_scale, COLORS['white'], thickness)


def draw_dashboard_overlay(frame: np.ndarray, stats: Dict):
    """Draw a semi-transparent statistics overlay on the video frame.

    Displays FPS, worker count, violations, and compliance rate.

    Args:
        frame: Image to draw on.
        stats: Dict with 'fps', 'total_workers', 'total_violations',
               'compliance_rate' keys.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Background panel at top-left
    panel_w = 250
    panel_h = 100
    cv2.rectangle(overlay, (5, 5), (panel_w, panel_h),
                  COLORS['overlay_bg'], -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw border
    cv2.rectangle(frame, (5, 5), (panel_w, panel_h),
                  COLORS['info'], 1)

    # Stats text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    y_offset = 25

    fps = stats.get('fps', 0)
    workers = stats.get('total_workers', 0)
    violations = stats.get('total_violations', 0)
    compliance = stats.get('compliance_rate', 0)

    lines = [
        (f"FPS: {fps:.1f}", COLORS['info']),
        (f"Workers: {workers}", COLORS['white']),
        (f"Violations: {violations}",
         COLORS['violation'] if violations > 0 else COLORS['compliant']),
        (f"Compliance: {compliance:.1f}%",
         COLORS['compliant'] if compliance >= 80 else COLORS['warning']),
    ]

    for text, color in lines:
        cv2.putText(frame, text, (15, y_offset),
                    font, font_scale, color, thickness)
        y_offset += 20


def create_violation_snapshot(frame: np.ndarray, bbox: Tuple,
                               worker_id: int,
                               violation_type: str) -> np.ndarray:
    """Create a snapshot image focused on a violation.

    Crops the violation area and adds metadata annotations.

    Args:
        frame: Full video frame.
        bbox: (x1, y1, x2, y2) of the person.
        worker_id: Worker tracking ID.
        violation_type: Type of PPE violation.

    Returns:
        Annotated snapshot image.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Expand crop area with padding
    pad = 40
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)

    snapshot = frame[cy1:cy2, cx1:cx2].copy()

    if snapshot.size == 0:
        snapshot = frame.copy()

    sh, sw = snapshot.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Add timestamp banner at bottom
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    banner_h = 30
    banner = np.zeros((banner_h, sw, 3), dtype=np.uint8)
    banner[:] = COLORS['text_bg']
    cv2.putText(banner, f"Worker #{worker_id} | {violation_type} | {ts}",
                (5, 20), font, 0.4, COLORS['white'], 1)

    snapshot = np.vstack([snapshot, banner])
    return snapshot


def draw_body_regions_debug(frame: np.ndarray, bbox: Tuple,
                             regions: Dict[str, Tuple]):
    """Debug visualization: draw body region boundaries.

    Args:
        frame: Image to draw on.
        bbox: Person bounding box.
        regions: Dict of region name to (x1, y1, x2, y2).
    """
    region_colors = {
        'head':  (0, 255, 255),   # Yellow
        'face':  (255, 0, 255),   # Magenta
        'ears':  (255, 255, 0),   # Cyan
        'torso': (0, 165, 255),   # Orange
        'hands': (0, 255, 0),     # Green
        'feet':  (255, 0, 0),     # Blue
    }

    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, (rx1, ry1, rx2, ry2) in regions.items():
        color = region_colors.get(name, (200, 200, 200))
        cv2.rectangle(frame, (int(rx1), int(ry1)),
                      (int(rx2), int(ry2)), color, 1)
        cv2.putText(frame, name, (int(rx1), int(ry1) - 3),
                    font, 0.3, color, 1)
