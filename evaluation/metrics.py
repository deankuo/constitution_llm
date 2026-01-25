"""
Evaluation Metrics for Political Indicator Predictions

This module provides metrics for evaluating prediction quality:
- Accuracy
- Precision, Recall, F1 (per-class and macro)
- Confusion matrix
- Agreement metrics
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    label: str
    precision: float
    recall: float
    f1: float
    support: int  # Number of true instances


@dataclass
class IndicatorMetrics:
    """Complete metrics for an indicator."""
    indicator: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: Dict[str, ClassMetrics]
    confusion_matrix: Dict[str, Dict[str, int]]
    total_samples: int
    valid_samples: int


def calculate_accuracy(
    predictions: List[str],
    ground_truth: List[str]
) -> float:
    """
    Calculate accuracy (proportion of correct predictions).

    Args:
        predictions: List of predicted values
        ground_truth: List of true values

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for p, g in zip(predictions, ground_truth) if str(p) == str(g))
    return correct / len(predictions)


def calculate_confusion_matrix(
    predictions: List[str],
    ground_truth: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Dict[str, int]]:
    """
    Calculate confusion matrix.

    Args:
        predictions: List of predicted values
        ground_truth: List of true values
        labels: Optional list of all possible labels

    Returns:
        Confusion matrix as nested dict {true_label: {pred_label: count}}
    """
    if labels is None:
        labels = sorted(set(str(p) for p in predictions) | set(str(g) for g in ground_truth))

    matrix = {true: {pred: 0 for pred in labels} for true in labels}

    for pred, true in zip(predictions, ground_truth):
        pred_str = str(pred)
        true_str = str(true)
        if true_str in matrix and pred_str in matrix[true_str]:
            matrix[true_str][pred_str] += 1

    return matrix


def calculate_precision_recall_f1(
    predictions: List[str],
    ground_truth: List[str],
    label: str
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 for a specific class.

    Args:
        predictions: List of predicted values
        ground_truth: List of true values
        label: The class label to calculate metrics for

    Returns:
        Tuple of (precision, recall, f1)
    """
    label_str = str(label)

    true_positives = sum(
        1 for p, g in zip(predictions, ground_truth)
        if str(p) == label_str and str(g) == label_str
    )

    predicted_positives = sum(1 for p in predictions if str(p) == label_str)
    actual_positives = sum(1 for g in ground_truth if str(g) == label_str)

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def evaluate_indicator(
    predictions: List[str],
    ground_truth: List[str],
    indicator: str,
    labels: Optional[List[str]] = None
) -> IndicatorMetrics:
    """
    Calculate all metrics for an indicator.

    Args:
        predictions: List of predicted values
        ground_truth: List of true values
        indicator: Name of the indicator
        labels: Optional list of valid labels

    Returns:
        IndicatorMetrics with all calculated metrics
    """
    # Filter out None/invalid predictions
    valid_pairs = [
        (p, g) for p, g in zip(predictions, ground_truth)
        if p is not None and g is not None
    ]

    if not valid_pairs:
        return IndicatorMetrics(
            indicator=indicator,
            accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            per_class={},
            confusion_matrix={},
            total_samples=len(predictions),
            valid_samples=0
        )

    valid_preds, valid_truth = zip(*valid_pairs)
    valid_preds = [str(p) for p in valid_preds]
    valid_truth = [str(g) for g in valid_truth]

    # Determine labels if not provided
    if labels is None:
        labels = sorted(set(valid_preds) | set(valid_truth))

    # Calculate accuracy
    accuracy = calculate_accuracy(valid_preds, valid_truth)

    # Calculate confusion matrix
    conf_matrix = calculate_confusion_matrix(valid_preds, valid_truth, labels)

    # Calculate per-class metrics
    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for label in labels:
        precision, recall, f1 = calculate_precision_recall_f1(valid_preds, valid_truth, label)
        support = sum(1 for g in valid_truth if str(g) == str(label))

        per_class[label] = ClassMetrics(
            label=label,
            precision=precision,
            recall=recall,
            f1=f1,
            support=support
        )

        if support > 0:  # Only include classes with support in macro average
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    # Calculate macro averages
    macro_precision = np.mean(precisions) if precisions else 0.0
    macro_recall = np.mean(recalls) if recalls else 0.0
    macro_f1 = np.mean(f1s) if f1s else 0.0

    return IndicatorMetrics(
        indicator=indicator,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=conf_matrix,
        total_samples=len(predictions),
        valid_samples=len(valid_pairs)
    )


def calculate_agreement(
    predictions_1: List[str],
    predictions_2: List[str]
) -> float:
    """
    Calculate agreement rate between two sets of predictions.

    Args:
        predictions_1: First set of predictions
        predictions_2: Second set of predictions

    Returns:
        Agreement rate (0.0 to 1.0)
    """
    if len(predictions_1) != len(predictions_2):
        raise ValueError("Both prediction lists must have same length")

    if len(predictions_1) == 0:
        return 0.0

    agreements = sum(
        1 for p1, p2 in zip(predictions_1, predictions_2)
        if str(p1) == str(p2)
    )

    return agreements / len(predictions_1)


def calculate_cohens_kappa(
    predictions_1: List[str],
    predictions_2: List[str],
    labels: Optional[List[str]] = None
) -> float:
    """
    Calculate Cohen's Kappa for inter-rater agreement.

    Args:
        predictions_1: First set of predictions
        predictions_2: Second set of predictions
        labels: Optional list of all possible labels

    Returns:
        Cohen's Kappa score
    """
    if len(predictions_1) != len(predictions_2):
        raise ValueError("Both prediction lists must have same length")

    n = len(predictions_1)
    if n == 0:
        return 0.0

    # Convert to strings
    p1 = [str(p) for p in predictions_1]
    p2 = [str(p) for p in predictions_2]

    if labels is None:
        labels = sorted(set(p1) | set(p2))

    # Calculate observed agreement
    observed = sum(1 for a, b in zip(p1, p2) if a == b) / n

    # Calculate expected agreement
    counter_1 = Counter(p1)
    counter_2 = Counter(p2)

    expected = sum(
        (counter_1.get(label, 0) / n) * (counter_2.get(label, 0) / n)
        for label in labels
    )

    # Calculate kappa
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0

    kappa = (observed - expected) / (1 - expected)
    return kappa


def format_metrics_report(metrics: IndicatorMetrics) -> str:
    """
    Format metrics as a readable report.

    Args:
        metrics: IndicatorMetrics to format

    Returns:
        Formatted string report
    """
    lines = [
        f"\n{'='*60}",
        f"Metrics for: {metrics.indicator}",
        f"{'='*60}",
        f"Samples: {metrics.valid_samples} valid / {metrics.total_samples} total",
        f"",
        f"Overall Metrics:",
        f"  Accuracy:         {metrics.accuracy:.4f}",
        f"  Macro Precision:  {metrics.macro_precision:.4f}",
        f"  Macro Recall:     {metrics.macro_recall:.4f}",
        f"  Macro F1:         {metrics.macro_f1:.4f}",
        f"",
        f"Per-Class Metrics:",
    ]

    for label, class_metrics in sorted(metrics.per_class.items()):
        lines.append(f"  Class '{label}' (n={class_metrics.support}):")
        lines.append(f"    Precision: {class_metrics.precision:.4f}")
        lines.append(f"    Recall:    {class_metrics.recall:.4f}")
        lines.append(f"    F1:        {class_metrics.f1:.4f}")

    lines.append(f"")
    lines.append(f"Confusion Matrix (rows=true, cols=pred):")

    # Format confusion matrix
    labels = sorted(metrics.confusion_matrix.keys())
    header = "      " + "  ".join(f"{l:>5}" for l in labels)
    lines.append(header)

    for true_label in labels:
        row_values = [metrics.confusion_matrix[true_label].get(pred, 0) for pred in labels]
        row = f"{true_label:>5} " + "  ".join(f"{v:>5}" for v in row_values)
        lines.append(row)

    lines.append(f"{'='*60}\n")

    return "\n".join(lines)
