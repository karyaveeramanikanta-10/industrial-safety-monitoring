"""
PPE (Personal Protective Equipment) detection module.

Uses color-based region analysis to detect safety equipment on
cropped person images. Divides the person bounding box into body
regions and checks for PPE-colored pixels using HSV thresholding.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger('safety_monitor')


class PPEDetector:
    """Detect Personal Protective Equipment using color-based region analysis.

    Analyzes cropped person images by:
    1. Dividing into body regions (head, torso, face, hands, feet, ears)
    2. Converting each region to HSV color space
    3. Checking for PPE-characteristic colors using thresholding
    4. Applying morphological operations for noise reduction
    5. Computing pixel ratio to determine detection confidence

    Usage:
        detector = PPEDetector(required_ppe=['helmet', 'vest'])
        ppe_status = detector.detect(person_crop)
        compliance = detector.check_compliance(ppe_status)
    """

    # Body region proportions (fraction of person height/width)
    BODY_REGIONS = {
        'head':  {'y_start': 0.0,  'y_end': 0.15, 'x_start': 0.15, 'x_end': 0.85},
        'face':  {'y_start': 0.08, 'y_end': 0.25, 'x_start': 0.25, 'x_end': 0.75},
        'ears':  {'y_start': 0.05, 'y_end': 0.18, 'x_start': 0.0,  'x_end': 1.0},
        'torso': {'y_start': 0.20, 'y_end': 0.55, 'x_start': 0.05, 'x_end': 0.95},
        'hands': {'y_start': 0.45, 'y_end': 0.70, 'x_start': 0.0,  'x_end': 1.0},
        'feet':  {'y_start': 0.85, 'y_end': 1.0,  'x_start': 0.1,  'x_end': 0.9},
    }

    # Maps PPE items to body regions
    PPE_REGION_MAP = {
        'helmet': 'head',
        'vest': 'torso',
        'mask': 'face',
        'goggles': 'face',
        'gloves': 'hands',
        'ear_protection': 'ears',
        'shoes': 'feet',
    }

    # Minimum color area ratios for positive detection
    DEFAULT_MIN_AREA_RATIOS = {
        'helmet': 0.12,
        'vest': 0.18,
        'mask': 0.10,
        'goggles': 0.05,
        'gloves': 0.08,
        'ear_protection': 0.05,
        'shoes': 0.10,
    }

    def __init__(self, color_ranges: Optional[Dict] = None,
                 required_ppe: Optional[List[str]] = None,
                 min_area_ratios: Optional[Dict[str, float]] = None):
        """Initialize PPE detector.

        Args:
            color_ranges: Dict of PPE color HSV ranges. Uses defaults if None.
            required_ppe: List of required PPE items. Defaults to ['helmet', 'vest'].
            min_area_ratios: Minimum color area ratios per PPE item.
        """
        self.color_ranges = color_ranges or self._default_color_ranges()
        self.required_ppe = required_ppe or ['helmet', 'vest']
        self.min_area_ratios = min_area_ratios or self.DEFAULT_MIN_AREA_RATIOS.copy()

    @staticmethod
    def _default_color_ranges() -> Dict:
        """Return default HSV color ranges for PPE items."""
        return {
            'helmet': {
                'white':  {'lower': np.array([0, 0, 180]),
                           'upper': np.array([180, 30, 255])},
                'yellow': {'lower': np.array([20, 80, 80]),
                           'upper': np.array([35, 255, 255])},
                'blue':   {'lower': np.array([100, 80, 80]),
                           'upper': np.array([130, 255, 255])},
                'red':    {'lower': np.array([0, 100, 100]),
                           'upper': np.array([10, 255, 255])},
                'orange': {'lower': np.array([10, 100, 100]),
                           'upper': np.array([20, 255, 255])},
            },
            'vest': {
                'hi_vis_yellow': {'lower': np.array([20, 100, 100]),
                                  'upper': np.array([35, 255, 255])},
                'hi_vis_orange': {'lower': np.array([5, 100, 100]),
                                  'upper': np.array([18, 255, 255])},
            },
            'gloves': {
                'blue':   {'lower': np.array([100, 50, 50]),
                           'upper': np.array([130, 255, 255])},
                'orange': {'lower': np.array([5, 100, 100]),
                           'upper': np.array([20, 255, 255])},
            },
            'mask': {
                'white': {'lower': np.array([0, 0, 180]),
                          'upper': np.array([180, 30, 255])},
                'blue':  {'lower': np.array([90, 50, 50]),
                          'upper': np.array([130, 255, 255])},
            },
            'goggles': {
                'clear': {'lower': np.array([0, 0, 200]),
                          'upper': np.array([180, 30, 255])},
            },
            'ear_protection': {
                'yellow': {'lower': np.array([20, 80, 80]),
                           'upper': np.array([35, 255, 255])},
                'red':    {'lower': np.array([0, 100, 100]),
                           'upper': np.array([10, 255, 255])},
            },
            'shoes': {
                'black': {'lower': np.array([0, 0, 0]),
                          'upper': np.array([180, 255, 50])},
                'brown': {'lower': np.array([10, 50, 20]),
                          'upper': np.array([20, 200, 150])},
            },
        }

    def detect(self, person_crop: np.ndarray) -> Dict:
        """Detect PPE items in a cropped person image.

        Args:
            person_crop: Cropped BGR image of a single person.

        Returns:
            Dict mapping PPE item to detection results:
            {
                'helmet': {'detected': True, 'confidence': 0.35, 'region': 'head'},
                'vest': {'detected': False, 'confidence': 0.05, 'region': 'torso'},
                ...
            }
        """
        if person_crop is None or person_crop.size == 0:
            return self._empty_result()

        h, w = person_crop.shape[:2]
        if h < 20 or w < 10:
            return self._empty_result()

        results = {}

        for ppe_item in self.PPE_REGION_MAP:
            region_name = self.PPE_REGION_MAP[ppe_item]
            roi = self._extract_region(person_crop, region_name)

            if roi is None or roi.size == 0:
                results[ppe_item] = {
                    'detected': False,
                    'confidence': 0.0,
                    'region': region_name,
                }
                continue

            # Get color ranges for this PPE item
            color_variants = self.color_ranges.get(ppe_item, {})
            min_ratio = self.min_area_ratios.get(ppe_item, 0.10)

            detected, confidence = self._detect_color(
                roi, color_variants, min_ratio
            )

            results[ppe_item] = {
                'detected': detected,
                'confidence': confidence,
                'region': region_name,
            }

        return results

    def _extract_region(self, image: np.ndarray,
                         region_name: str) -> Optional[np.ndarray]:
        """Extract a body region from the person crop.

        Args:
            image: Full person crop image.
            region_name: Name of the body region.

        Returns:
            Cropped region image or None.
        """
        region = self.BODY_REGIONS.get(region_name)
        if region is None:
            return None

        h, w = image.shape[:2]
        y1 = int(h * region['y_start'])
        y2 = int(h * region['y_end'])
        x1 = int(w * region['x_start'])
        x2 = int(w * region['x_end'])

        # Clamp to image bounds
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))

        roi = image[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    def _detect_color(self, roi: np.ndarray,
                       color_variants: Dict,
                       min_ratio: float) -> Tuple[bool, float]:
        """Check for specific colors in a region of interest.

        Args:
            roi: Region of interest image (BGR).
            color_variants: Dict of color name -> {'lower': [], 'upper': []}.
            min_ratio: Minimum pixel ratio for positive detection.

        Returns:
            Tuple of (detected: bool, max_confidence: float).
        """
        if roi is None or roi.size == 0:
            return False, 0.0

        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        except cv2.error:
            return False, 0.0

        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0:
            return False, 0.0

        # Morphological kernel for noise reduction
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color_name, ranges in color_variants.items():
            lower = ranges.get('lower')
            upper = ranges.get('upper')

            if lower is None or upper is None:
                continue

            # Convert lists to numpy arrays if needed
            if isinstance(lower, list):
                lower = np.array(lower, dtype=np.uint8)
            if isinstance(upper, list):
                upper = np.array(upper, dtype=np.uint8)

            mask = cv2.inRange(hsv, lower, upper)

            # Morphological operations to reduce noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Combine masks
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Calculate the pixel ratio
        color_pixels = cv2.countNonZero(combined_mask)
        ratio = color_pixels / total_pixels

        detected = ratio >= min_ratio
        return detected, round(ratio, 4)

    def check_compliance(self, ppe_status: Dict) -> Dict:
        """Check if all required PPE items are detected.

        Args:
            ppe_status: Dict from detect() method.

        Returns:
            Dict with compliance results:
            {
                'compliant': bool,
                'missing': ['vest'],
                'present': ['helmet'],
                'details': {...}
            }
        """
        missing = []
        present = []

        for item in self.required_ppe:
            item_data = ppe_status.get(item, {})
            if item_data.get('detected', False):
                present.append(item)
            else:
                missing.append(item)

        return {
            'compliant': len(missing) == 0,
            'missing': missing,
            'present': present,
            'details': ppe_status,
        }

    def set_required_ppe(self, required_items: List[str]):
        """Update the list of required PPE items.

        Args:
            required_items: New list of required PPE item names.
        """
        valid_items = set(self.PPE_REGION_MAP.keys())
        self.required_ppe = [
            item for item in required_items if item in valid_items
        ]
        logger.info(f"Updated required PPE: {self.required_ppe}")

    def _empty_result(self) -> Dict:
        """Return empty detection result for all PPE items."""
        return {
            item: {'detected': False, 'confidence': 0.0, 'region': region}
            for item, region in self.PPE_REGION_MAP.items()
        }
