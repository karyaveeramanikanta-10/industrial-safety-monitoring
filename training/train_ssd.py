"""
SSD model training pipeline.

Provides functionality to train or fine-tune an SSD MobileNet V2
model for person detection or PPE detection on custom datasets.

Usage:
    python -m training.train_ssd --data_dir datasets/ --epochs 50
"""

import os
import logging
import argparse
from datetime import datetime
from typing import Optional

logger = logging.getLogger('safety_monitor')


class SSDTrainer:
    """Train SSD model with MobileNetV2 backbone for object detection.

    Supports building a custom SSD model, training on labeled datasets,
    and saving/loading checkpoints.
    """

    def __init__(self, num_classes: int = 2,
                 input_shape: tuple = (320, 320, 3),
                 learning_rate: float = 0.001,
                 batch_size: int = 16):
        """Initialize SSD trainer.

        Args:
            num_classes: Number of detection classes (including background).
            input_shape: Model input shape (height, width, channels).
            learning_rate: Initial learning rate.
            batch_size: Training batch size.
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = None
        self.history = None

    def build_model(self):
        """Build SSD model with MobileNetV2 backbone.

        Uses MobileNetV2 as the feature extractor and adds custom
        detection heads for localization and classification.
        """
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras import layers, Model

        # Base feature extractor
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        base_model.trainable = False

        # Feature extraction layers
        inputs = base_model.input
        features = base_model.output

        # Detection head
        x = layers.GlobalAveragePooling2D()(features)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Classification output
        class_output = layers.Dense(
            self.num_classes, activation='softmax', name='classification'
        )(x)

        # Bounding box regression output (4 coordinates)
        bbox_output = layers.Dense(
            4, activation='sigmoid', name='bbox_regression'
        )(x)

        self.model = Model(
            inputs=inputs,
            outputs=[class_output, bbox_output]
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss={
                'classification': 'categorical_crossentropy',
                'bbox_regression': 'mse',
            },
            loss_weights={
                'classification': 1.0,
                'bbox_regression': 5.0,
            },
            metrics={
                'classification': 'accuracy',
            }
        )

        logger.info(
            f"SSD model built: {self.num_classes} classes, "
            f"input shape {self.input_shape}"
        )
        self.model.summary(print_fn=logger.info)
        return self.model

    def train(self, train_dataset, val_dataset, epochs: int = 50,
              checkpoint_dir: str = 'models/checkpoints'):
        """Train the model.

        Args:
            train_dataset: Training tf.data.Dataset.
            val_dataset: Validation tf.data.Dataset.
            epochs: Number of training epochs.
            checkpoint_dir: Directory for checkpoints.

        Returns:
            Training history object.
        """
        import tensorflow as tf

        os.makedirs(checkpoint_dir, exist_ok=True)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(checkpoint_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    checkpoint_dir, 'tensorboard',
                    datetime.now().strftime('%Y%m%d_%H%M%S')
                )
            ),
        ]

        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        logger.info(f"Training completed after {len(self.history.epoch)} epochs")
        return self.history

    def save_model(self, path: str):
        """Save trained model to file.

        Args:
            path: Output file path (.h5 or SavedModel directory).
        """
        if self.model is None:
            logger.error("No model to save")
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a trained model from file.

        Args:
            path: Path to saved model (.h5 or SavedModel directory).
        """
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description='Train SSD MobileNet V2 model for safety monitoring'
    )
    parser.add_argument(
        '--data_dir', default='datasets/',
        help='Dataset directory with train/val/test splits'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--num_classes', type=int, default=2,
        help='Number of classes (including background)'
    )
    parser.add_argument(
        '--output', default='models/person_detector/ssd_person_model.h5',
        help='Output model path'
    )
    args = parser.parse_args()

    from training.data_loader import SafetyDataset

    # Initialize trainer
    trainer = SSDTrainer(
        num_classes=args.num_classes,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    trainer.build_model()

    # Load datasets
    train_data = SafetyDataset(args.data_dir, split='train')
    val_data = SafetyDataset(args.data_dir, split='val')

    train_tf = train_data.create_tf_dataset(batch_size=args.batch_size)
    val_tf = val_data.create_tf_dataset(batch_size=args.batch_size, shuffle=False)

    # Train
    trainer.train(train_tf, val_tf, epochs=args.epochs)

    # Save model
    trainer.save_model(args.output)
    print(f"Training complete. Model saved to {args.output}")


if __name__ == '__main__':
    main()
