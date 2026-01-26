"""
Evaluation and metrics for prediction results.

This package contains:
- metrics: Accuracy, F1, per-class metrics
- analyzer: Result analysis and reporting
- notebook_utils: Easy-to-use functions for Jupyter notebooks
"""

__version__ = '1.0.0'

from evaluation.notebook_utils import (
    load_predictions,
    filter_data,
    calculate_polity_accuracy,
    evaluate_indicator,
    evaluate_all_indicators,
    plot_accuracy_comparison,
    quick_eval,
    compare_datasets,
    compare_experiments,
    plot_binary_comparison,
    plot_multiclass_comparison,
    INDICATORS_WITH_GROUND_TRUTH,
    BINARY_INDICATORS,
    MULTICLASS_INDICATORS
)

__all__ = [
    'load_predictions',
    'filter_data',
    'calculate_polity_accuracy',
    'evaluate_indicator',
    'evaluate_all_indicators',
    'plot_accuracy_comparison',
    'quick_eval',
    'compare_datasets',
    'compare_experiments',
    'plot_binary_comparison',
    'plot_multiclass_comparison',
    'INDICATORS_WITH_GROUND_TRUTH',
    'BINARY_INDICATORS',
    'MULTICLASS_INDICATORS'
]
