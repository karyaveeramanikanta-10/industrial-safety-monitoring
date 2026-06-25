"""
Dataset loading and preprocessing for training.

Supports YOLO-format labels (txt files) and provides
tf.data.Dataset creation for efficient training.
"""

import cv2
import numpy as np
import os
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger('safety_monitor')


class SafetyDataset:
    """Load and preprocess safety monitoring datasets.

    Supports standard YOLO format directory structure:
        datasets/train/images/  — image files
        datasets/train/labels/  — label files (one per image)

    Label format (YOLO): class_id x_center y_center width height
    (all values normalized to 0-1)

    Usage:
        dataset = SafetyDataset('datasets/', split='train')
        tf_dataset = dataset.create_tf_dataset(batch_size=16)
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, data_dir: str, split: str = 'train',
                 input_size: Tuple[int, int] = (320, 320)):
        """Initialize dataset loader.

        Args:
            data_dir: Root dataset directory.
            split: Dataset split ('train', 'val', 'test').
            input_size: Target image size (height, width).
        """
        self.data_dir = data_dir
        self.split = split
        self.input_size = input_size
        self.images: List[str] = []
        self.labels: List[str] = []
        self._load_data()

    def _load_data(self):
        """Load image and label file paths from directory."""
        images_dir = os.path.join(self.data_dir, self.split, 'images')
        labels_dir = os.path.join(self.data_dir, self.split, 'labels')

        if not os.path.exists(images_dir):
            logger.warning(f"Images directory not found: {images_dir}")
            return

        for filename in sorted(os.listdir(images_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                continue

            image_path = os.path.join(images_dir, filename)
            label_name = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)

            self.images.append(image_path)
            self.labels.append(
                label_path if os.path.exists(label_path) else None
            )

        logger.info(
            f"Loaded {len(self.images)} images for split '{self.split}'"
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, dict]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image, labels_dict).
        """
        image = cv2.imread(self.images[idx])
        if image is None:
            logger.warning(f"Failed to load image: {self.images[idx]}")
            image = np.zeros(
                (self.input_size[0], self.input_size[1], 3), dtype=np.uint8
            )

        image = cv2.resize(image, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse labels
        boxes = []
        classes = []
        label_path = self.labels[idx]

        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Convert to (x1, y1, x2, y2) normalized
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2

                        boxes.append([x1, y1, x2, y2])
                        classes.append(cls_id)

        return image, {
            'boxes': np.array(boxes, dtype=np.float32) if boxes
                     else np.zeros((0, 4), dtype=np.float32),
            'classes': np.array(classes, dtype=np.int32) if classes
                       else np.zeros((0,), dtype=np.int32),
        }

    def create_tf_dataset(self, batch_size: int = 16,
                          shuffle: bool = True):
        """Create a tf.data.Dataset for training.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle the dataset.

        Returns:
            tf.data.Dataset yielding (images, (classes, boxes)) batches.
        """
        import tensorflow as tf

        def generator():
            indices = list(range(len(self)))
            if shuffle:
                np.random.shuffle(indices)
            for idx in indices:
                image, labels = self[idx]
                # Normalize image to [0, 1]
                image = image.astype(np.float32) / 255.0

                # For the simplified model, use the first box or zeros
                if len(labels['classes']) > 0:
                    cls = np.zeros(2, dtype=np.float32)
                    cls[min(labels['classes'][0], 1)] = 1.0
                    box = labels['boxes'][0]
                else:
                    cls = np.array([1.0, 0.0], dtype=np.float32)
                    box = np.zeros(4, dtype=np.float32)

                yield image, (cls, box)

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(self.input_size[0], self.input_size[1], 3),
                    dtype=tf.float32
                ),
                (
                    tf.TensorSpec(shape=(2,), dtype=tf.float32),
                    tf.TensorSpec(shape=(4,), dtype=tf.float32),
                ),
            )
        )

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
