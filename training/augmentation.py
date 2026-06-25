"""
Data augmentation pipelines for training.

Provides a variety of image augmentation transformations that
preserve bounding box annotations.
"""

import cv2
import numpy as np
import random
from typing import List, Tuple, Optional


class DataAugmentor:
    """Apply data augmentation transformations for training.

    Supports geometric and photometric augmentations with
    proper bounding box transformation.

    Usage:
        augmentor = DataAugmentor()
        aug_image, aug_boxes = augmentor.augment(image, bboxes)
    """

    def __init__(self, augmentations: Optional[List[str]] = None):
        """Initialize augmentor.

        Args:
            augmentations: List of augmentation names to apply.
                          Uses all available if None.
        """
        self.augmentations = augmentations or [
            'horizontal_flip', 'brightness', 'contrast',
            'gaussian_noise', 'rotation', 'scale'
        ]

    def augment(self, image: np.ndarray,
                bboxes: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply random augmentations to image and bounding boxes.

        Each augmentation is applied with 50% probability.

        Args:
            image: Input image (numpy array, BGR or RGB).
            bboxes: Optional array of [x1, y1, x2, y2] boxes (normalized).

        Returns:
            Tuple of (augmented_image, transformed_bboxes).
        """
        aug_image = image.copy()
        aug_boxes = bboxes.copy() if bboxes is not None else None

        for aug_name in self.augmentations:
            if random.random() < 0.5:
                method = getattr(self, aug_name, None)
                if method:
                    result = method(aug_image, aug_boxes)
                    if isinstance(result, tuple):
                        aug_image, aug_boxes = result
                    else:
                        aug_image = result

        return aug_image, aug_boxes

    def horizontal_flip(self, image: np.ndarray,
                         bboxes: Optional[np.ndarray] = None,
                         p: float = 0.5
                         ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Horizontally flip image and bounding boxes.

        Args:
            image: Input image.
            bboxes: Normalized [x1, y1, x2, y2] boxes.
            p: Probability of applying (already handled by augment()).

        Returns:
            Flipped image and transformed boxes.
        """
        flipped = cv2.flip(image, 1)

        if bboxes is not None and len(bboxes) > 0:
            new_boxes = bboxes.copy()
            new_boxes[:, 0] = 1.0 - bboxes[:, 2]  # new_x1 = 1 - old_x2
            new_boxes[:, 2] = 1.0 - bboxes[:, 0]  # new_x2 = 1 - old_x1
            return flipped, new_boxes

        return flipped, bboxes

    def adjust_brightness(self, image: np.ndarray,
                          bboxes: Optional[np.ndarray] = None,
                          factor_range: Tuple[float, float] = (0.7, 1.3)
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Adjust image brightness randomly.

        Args:
            image: Input image.
            bboxes: Bounding boxes (unchanged).
            factor_range: Min and max brightness multiplier.

        Returns:
            Brightness-adjusted image and unchanged boxes.
        """
        factor = random.uniform(*factor_range)
        adjusted = np.clip(image.astype(np.float32) * factor, 0, 255)
        return adjusted.astype(np.uint8), bboxes

    # Alias for the augment dispatcher
    brightness = adjust_brightness

    def adjust_contrast(self, image: np.ndarray,
                        bboxes: Optional[np.ndarray] = None,
                        factor_range: Tuple[float, float] = (0.7, 1.3)
                        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Adjust image contrast randomly.

        Args:
            image: Input image.
            bboxes: Bounding boxes (unchanged).
            factor_range: Min and max contrast multiplier.

        Returns:
            Contrast-adjusted image and unchanged boxes.
        """
        factor = random.uniform(*factor_range)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        adjusted = np.clip(
            (image.astype(np.float32) - mean) * factor + mean,
            0, 255
        )
        return adjusted.astype(np.uint8), bboxes

    contrast = adjust_contrast

    def add_gaussian_noise(self, image: np.ndarray,
                           bboxes: Optional[np.ndarray] = None,
                           mean: float = 0, std: float = 10
                           ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Add Gaussian noise to image.

        Args:
            image: Input image.
            bboxes: Bounding boxes (unchanged).
            mean: Noise mean.
            std: Noise standard deviation.

        Returns:
            Noisy image and unchanged boxes.
        """
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8), bboxes

    gaussian_noise = add_gaussian_noise

    def rotate(self, image: np.ndarray,
               bboxes: Optional[np.ndarray] = None,
               angle_range: Tuple[float, float] = (-15, 15)
               ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Rotate image and adjust bounding boxes.

        Args:
            image: Input image.
            bboxes: Normalized [x1, y1, x2, y2] boxes.
            angle_range: Min and max rotation angle in degrees.

        Returns:
            Rotated image and adjusted boxes.
        """
        h, w = image.shape[:2]
        angle = random.uniform(*angle_range)
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                 borderMode=cv2.BORDER_REFLECT_101)

        if bboxes is not None and len(bboxes) > 0:
            # Transform box corners
            new_boxes = []
            for box in bboxes:
                corners = np.array([
                    [box[0] * w, box[1] * h],
                    [box[2] * w, box[1] * h],
                    [box[2] * w, box[3] * h],
                    [box[0] * w, box[3] * h],
                ], dtype=np.float32)

                ones = np.ones((4, 1))
                corners_h = np.hstack([corners, ones])
                transformed = M.dot(corners_h.T).T

                x_min = max(0, np.min(transformed[:, 0]) / w)
                y_min = max(0, np.min(transformed[:, 1]) / h)
                x_max = min(1, np.max(transformed[:, 0]) / w)
                y_max = min(1, np.max(transformed[:, 1]) / h)

                if x_max > x_min and y_max > y_min:
                    new_boxes.append([x_min, y_min, x_max, y_max])

            bboxes = np.array(new_boxes, dtype=np.float32) if new_boxes \
                else np.zeros((0, 4), dtype=np.float32)

        return rotated, bboxes

    rotation = rotate

    def random_scale(self, image: np.ndarray,
                     bboxes: Optional[np.ndarray] = None,
                     scale_range: Tuple[float, float] = (0.8, 1.2)
                     ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Random scale augmentation.

        Args:
            image: Input image.
            bboxes: Normalized boxes (unchanged for scale).
            scale_range: Min and max scale factor.

        Returns:
            Scaled image and unchanged boxes.
        """
        h, w = image.shape[:2]
        scale = random.uniform(*scale_range)
        new_h, new_w = int(h * scale), int(w * scale)

        scaled = cv2.resize(image, (new_w, new_h))

        # Pad or crop to original size
        result = np.zeros_like(image)
        copy_h = min(new_h, h)
        copy_w = min(new_w, w)
        result[:copy_h, :copy_w] = scaled[:copy_h, :copy_w]

        return result, bboxes

    scale = random_scale
