"""
Model evaluation module.

Evaluates trained detection models using standard metrics:
precision, recall, F1-score, and mean Average Precision (mAP).
"""

import numpy as np
import logging
import os
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger('safety_monitor')


class ModelEvaluator:
    """Evaluate trained object detection models.

    Computes precision, recall, F1-score, and mAP on test datasets.

    Usage:
        evaluator = ModelEvaluator(model, test_dataset)
        results = evaluator.evaluate()
        evaluator.generate_report('evaluation_report.txt')
    """

    def __init__(self, model=None, test_dataset=None):
        """Initialize evaluator.

        Args:
            model: Trained model with a predict() method.
            test_dataset: Test dataset iterable yielding (image, labels).
        """
        self.model = model
        self.test_dataset = test_dataset
        self.results: Dict = {}

    def evaluate(self) -> Dict:
        """Run evaluation and compute all metrics.

        Returns:
            Dict with precision, recall, f1, mAP values.
        """
        if self.model is None or self.test_dataset is None:
            logger.error("Model or test dataset not provided")
            return {}

        all_predictions = []
        all_ground_truths = []

        for images, labels in self.test_dataset:
            predictions = self.model.predict(images)
            all_predictions.append(predictions)
            all_ground_truths.append(labels)

        # Compute metrics
        try:
            from sklearn.metrics import (
                precision_score, recall_score, f1_score
            )

            preds_flat = np.concatenate(
                [p[0].argmax(axis=-1) for p in all_predictions]
            )
            labels_flat = np.concatenate(
                [l[0].argmax(axis=-1) if l[0].ndim > 1
                 else l[0] for l in all_ground_truths]
            )

            self.results = {
                'precision': float(precision_score(
                    labels_flat, preds_flat, average='weighted',
                    zero_division=0
                )),
                'recall': float(recall_score(
                    labels_flat, preds_flat, average='weighted',
                    zero_division=0
                )),
                'f1_score': float(f1_score(
                    labels_flat, preds_flat, average='weighted',
                    zero_division=0
                )),
                'total_samples': len(preds_flat),
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self.results = {
                'error': str(e),
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
            }

        logger.info(f"Evaluation results: {self.results}")
        return self.results

    def compute_map(self, predictions: List, ground_truths: List,
                     iou_threshold: float = 0.5) -> float:
        """Compute mean Average Precision (mAP).

        Args:
            predictions: List of predicted bounding boxes per image.
            ground_truths: List of ground truth bounding boxes per image.
            iou_threshold: IoU threshold for matching.

        Returns:
            mAP value (0.0 to 1.0).
        """
        aps = []
        for preds, gts in zip(predictions, ground_truths):
            if len(gts) == 0:
                if len(preds) == 0:
                    aps.append(1.0)
                else:
                    aps.append(0.0)
                continue

            if len(preds) == 0:
                aps.append(0.0)
                continue

            # Sort predictions by confidence (descending)
            sorted_preds = sorted(preds, key=lambda x: x.get('confidence', 0),
                                  reverse=True)

            tp = np.zeros(len(sorted_preds))
            fp = np.zeros(len(sorted_preds))
            matched = set()

            for i, pred in enumerate(sorted_preds):
                best_iou = 0
                best_gt = -1

                for j, gt in enumerate(gts):
                    if j in matched:
                        continue
                    iou = self._iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j

                if best_iou >= iou_threshold and best_gt >= 0:
                    tp[i] = 1
                    matched.add(best_gt)
                else:
                    fp[i] = 1

            # Compute precision-recall curve
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / len(gts)

            # AP via 11-point interpolation
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                p_at_r = precision[recall >= t]
                ap += (max(p_at_r) if len(p_at_r) > 0 else 0) / 11

            aps.append(ap)

        return float(np.mean(aps)) if aps else 0.0

    @staticmethod
    def _iou(box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def generate_report(self, output_path: str = 'data/processed/evaluation_report.txt'):
        """Generate detailed evaluation report.

        Args:
            output_path: Path for the report file.
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        lines = [
            "=" * 60,
            "Model Evaluation Report",
            "=" * 60,
            "",
        ]

        for key, value in self.results.items():
            if isinstance(value, float):
                lines.append(f"{key:20s}: {value:.4f}")
            else:
                lines.append(f"{key:20s}: {value}")

        lines.extend(["", "=" * 60])
        report = '\n'.join(lines)

        with open(output_path, 'w') as f:
            f.write(report)

        logger.info(f"Evaluation report saved to {output_path}")
        return output_path
