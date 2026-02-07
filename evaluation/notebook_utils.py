"""
Evaluation utilities for Jupyter notebooks.

This module provides easy-to-use functions for evaluating model predictions
in Jupyter notebooks with visualization and filtering capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


# Indicators with ground truth (exclude constitution)
INDICATORS_WITH_GROUND_TRUTH = [
    'sovereign', 'powersharing', 'assembly',
    'appointment', 'tenure', 'exit'
]

# Binary (2-class) indicators
BINARY_INDICATORS = ['sovereign', 'powersharing', 'assembly', 'exit']

# 3-class indicators
MULTICLASS_INDICATORS = ['appointment', 'tenure']


def load_predictions(filepath: str) -> pd.DataFrame:
    """
    Load prediction results from CSV.

    Args:
        filepath: Path to the predictions CSV file

    Returns:
        DataFrame with predictions
    """
    df = pd.read_csv(filepath)
    return df


def filter_data(
    df: pd.DataFrame,
    start_year: Optional[Tuple[int, int]] = None,
    end_year: Optional[Tuple[int, int]] = None,
    region_v1: Optional[List[str]] = None,
    region_v2: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter dataset by temporal and regional criteria.

    Args:
        df: Input DataFrame
        start_year: (min, max) tuple for filtering start_year
        end_year: (min, max) tuple for filtering end_year
        region_v1: List of region_v1 values to include
        region_v2: List of region_v2 values to include

    Returns:
        Filtered DataFrame

    Example:
        >>> filtered = filter_data(
        ...     df,
        ...     start_year=(1800, 2000),
        ...     region_v1=['Asia']
        ... )
    """
    filtered = df.copy()

    if start_year is not None:
        min_start, max_start = start_year
        filtered = filtered[(filtered['start_year'] >= min_start) &
                           (filtered['start_year'] <= max_start)]

    if end_year is not None:
        min_end, max_end = end_year
        filtered = filtered[(filtered['end_year'] >= min_end) &
                           (filtered['end_year'] <= max_end)]

    if region_v1 is not None:
        filtered = filtered[filtered['region_v1'].isin(region_v1)]

    if region_v2 is not None:
        filtered = filtered[filtered['region_v2'].isin(region_v2)]

    return filtered


def calculate_polity_accuracy(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate per-polity accuracy (ratio of correct predictions).

    Args:
        df: DataFrame with predictions
        indicators: List of indicators to evaluate (default: all with ground truth)

    Returns:
        DataFrame with added 'polity_accuracy' column

    Example:
        >>> df_with_accuracy = calculate_polity_accuracy(df)
        >>> print(df_with_accuracy[['territorynamehistorical', 'polity_accuracy']].head())
    """
    if indicators is None:
        indicators = INDICATORS_WITH_GROUND_TRUTH

    df = df.copy()

    # Count correct predictions per polity
    correct_counts = []
    total_counts = []

    for idx, row in df.iterrows():
        correct = 0
        total = 0

        for ind in indicators:
            # Check if both ground truth and prediction exist
            if ind in df.columns and f'{ind}_prediction' in df.columns:
                gt = row[ind]
                pred = row[f'{ind}_prediction']

                # Skip if either is missing
                if pd.notna(gt) and pd.notna(pred):
                    total += 1
                    if str(gt) == str(pred):
                        correct += 1

        correct_counts.append(correct)
        total_counts.append(total)

    # Calculate accuracy
    df['polity_correct_predictions'] = correct_counts
    df['polity_total_indicators'] = total_counts
    df['polity_accuracy'] = df.apply(
        lambda row: round(row['polity_correct_predictions'] / row['polity_total_indicators'], 3)
        if row['polity_total_indicators'] > 0 else np.nan,
        axis=1
    )

    return df


def evaluate_indicator(
    df: pd.DataFrame,
    indicator: str,
    show_confusion_matrix: bool = True,
    show_report: bool = True,
    figsize: Tuple[int, int] = (8, 6)
) -> Dict:
    """
    Evaluate a single indicator with confusion matrix and metrics.

    Args:
        df: DataFrame with predictions
        indicator: Name of the indicator to evaluate
        show_confusion_matrix: Whether to plot confusion matrix
        show_report: Whether to print classification report
        figsize: Figure size for confusion matrix plot

    Returns:
        Dictionary with evaluation metrics

    Example:
        >>> metrics = evaluate_indicator(df, 'sovereign')
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    # Get ground truth and predictions
    pred_col = f'{indicator}_prediction'

    if indicator not in df.columns:
        raise ValueError(f"Ground truth column '{indicator}' not found in DataFrame")

    if pred_col not in df.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found in DataFrame")

    # Filter out rows with missing values
    valid_mask = df[indicator].notna() & df[pred_col].notna()
    y_true = df.loc[valid_mask, indicator].astype(str)
    y_pred = df.loc[valid_mask, pred_col].astype(str)

    if len(y_true) == 0:
        print(f"⚠️ No valid data for {indicator}")
        return {}

    # Determine number of classes
    unique_labels = sorted(set(y_true) | set(y_pred))
    is_multiclass = len(unique_labels) > 2

    # Calculate metrics
    accuracy = round(accuracy_score(y_true, y_pred), 3)

    if is_multiclass:
        f1_macro = round(f1_score(y_true, y_pred, average='macro', zero_division=0), 3)
        f1_weighted = round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 3)
        metrics = {
            'indicator': indicator,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'n_samples': len(y_true),
            'n_classes': len(unique_labels)
        }
    else:
        f1 = round(f1_score(y_true, y_pred, pos_label=unique_labels[-1], zero_division=0), 3)
        metrics = {
            'indicator': indicator,
            'accuracy': accuracy,
            'f1': f1,
            'n_samples': len(y_true),
            'n_classes': len(unique_labels)
        }

    # Print metrics
    print(f"\n{'='*60}")
    print(f"📊 Evaluation Results: {indicator.upper()}")
    print(f"{'='*60}")
    print(f"Samples: {metrics['n_samples']}")
    print(f"Classes: {metrics['n_classes']} ({', '.join(unique_labels)})")
    print(f"Accuracy: {accuracy:.3f}")

    if is_multiclass:
        print(f"F1 (macro): {f1_macro:.3f}")
        print(f"F1 (weighted): {f1_weighted:.3f}")
    else:
        print(f"F1 Score: {f1:.3f}")

    # Confusion matrix
    if show_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix: {indicator.upper()}', fontsize=14, fontweight='bold')
        plt.ylabel('Ground Truth', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.show()

    # Classification report
    if show_report:
        print(f"\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, labels=unique_labels, zero_division=0))

    return metrics


def evaluate_all_indicators(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None,
    show_plots: bool = True,
    figsize: Tuple[int, int] = (8, 6)
) -> pd.DataFrame:
    """
    Evaluate all indicators and return summary metrics.

    Args:
        df: DataFrame with predictions
        indicators: List of indicators to evaluate (default: all with ground truth)
        show_plots: Whether to show confusion matrices
        figsize: Figure size for each plot

    Returns:
        DataFrame with summary metrics for each indicator

    Example:
        >>> summary = evaluate_all_indicators(df)
        >>> print(summary)
    """
    if indicators is None:
        indicators = INDICATORS_WITH_GROUND_TRUTH

    results = []

    for indicator in indicators:
        try:
            metrics = evaluate_indicator(
                df,
                indicator,
                show_confusion_matrix=show_plots,
                show_report=False,
                figsize=figsize
            )
            results.append(metrics)
        except Exception as e:
            print(f"⚠️ Error evaluating {indicator}: {e}")

    summary_df = pd.DataFrame(results)

    # Display summary table
    print(f"\n{'='*60}")
    print("📈 SUMMARY: All Indicators")
    print(f"{'='*60}\n")
    print(summary_df.to_string(index=False))
    print()

    return summary_df


def plot_accuracy_comparison(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot accuracy comparison across indicators.

    Args:
        df: DataFrame with predictions
        indicators: List of indicators to compare
        figsize: Figure size

    Example:
        >>> plot_accuracy_comparison(df)
    """
    if indicators is None:
        indicators = INDICATORS_WITH_GROUND_TRUTH

    accuracies = []
    labels = []

    for indicator in indicators:
        pred_col = f'{indicator}_prediction'

        if indicator in df.columns and pred_col in df.columns:
            valid_mask = df[indicator].notna() & df[pred_col].notna()
            if valid_mask.sum() > 0:
                y_true = df.loc[valid_mask, indicator].astype(str)
                y_pred = df.loc[valid_mask, pred_col].astype(str)
                acc = round(accuracy_score(y_true, y_pred), 3)
                accuracies.append(acc)
                labels.append(indicator)

    # Create bar plot
    plt.figure(figsize=figsize)
    colors = ['#1f77b4' if ind not in MULTICLASS_INDICATORS else '#ff7f0e'
              for ind in labels]

    bars = plt.bar(range(len(labels)), accuracies, color=colors, alpha=0.8)
    plt.xlabel('Indicator', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Prediction Accuracy by Indicator', fontsize=14, fontweight='bold')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{acc:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.8, label='Binary (2-class)'),
        Patch(facecolor='#ff7f0e', alpha=0.8, label='Multi-class (3-class)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.show()


# Convenience function for notebooks
def quick_eval(
    filepath: str,
    indicators: Optional[List[str]] = None,
    start_year: Optional[Tuple[int, int]] = None,
    end_year: Optional[Tuple[int, int]] = None,
    region_v1: Optional[List[str]] = None,
    region_v2: Optional[List[str]] = None,
    add_polity_accuracy: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quick evaluation workflow for notebooks.

    Args:
        filepath: Path to predictions CSV
        indicators: List of indicators to evaluate
        start_year: Filter by start year range
        end_year: Filter by end year range
        region_v1: Filter by region_v1 values
        region_v2: Filter by region_v2 values
        add_polity_accuracy: Whether to add polity accuracy column

    Returns:
        (filtered_df, summary_metrics)

    Example:
        >>> df, summary = quick_eval(
        ...     'data/results/predictions.csv',
        ...     start_year=(1800, 2000),
        ...     region_v1=['Europe']
        ... )
    """
    # Load data
    print("📂 Loading data...")
    df = load_predictions(filepath)
    print(f"   Loaded {len(df)} polities")

    # Filter data
    if any([start_year, end_year, region_v1, region_v2]):
        print("\n🔍 Applying filters...")
        df = filter_data(df, start_year, end_year, region_v1, region_v2)
        print(f"   Filtered to {len(df)} polities")

    # Add polity accuracy
    if add_polity_accuracy:
        print("\n📊 Calculating polity-level accuracy...")
        df = calculate_polity_accuracy(df, indicators)
        mean_polity_acc = df['polity_accuracy'].mean()
        print(f"   Mean polity accuracy: {mean_polity_acc:.3f}")

    # Evaluate indicators
    print("\n" + "="*60)
    print("🎯 Evaluating Indicators")
    print("="*60)

    summary = evaluate_all_indicators(df, indicators, show_plots=True)

    # Show comparison plot
    print("\n📊 Generating comparison plot...")
    plot_accuracy_comparison(df, indicators)

    return df, summary


# =============================================================================
# MULTI-DATASET COMPARISON
# =============================================================================

def compute_metrics_for_dataset(
    df: pd.DataFrame,
    indicator: str,
    include_precision_recall: bool = False
) -> Dict[str, float]:
    """
    Compute metrics for a single indicator in a dataset.

    Args:
        df: DataFrame with predictions
        indicator: Name of the indicator
        include_precision_recall: Whether to include precision and recall (for binary only)

    Returns:
        Dictionary with metrics
    """
    pred_col = f'{indicator}_prediction'

    if indicator not in df.columns or pred_col not in df.columns:
        return {}

    # Filter valid rows
    valid_mask = df[indicator].notna() & df[pred_col].notna()
    y_true = df.loc[valid_mask, indicator].astype(str)
    y_pred = df.loc[valid_mask, pred_col].astype(str)

    if len(y_true) == 0:
        return {}

    unique_labels = sorted(set(y_true) | set(y_pred))
    is_binary = len(unique_labels) == 2

    metrics = {
        'indicator': indicator,
        'accuracy': round(accuracy_score(y_true, y_pred), 3),
        'n_samples': len(y_true)
    }

    if is_binary:
        # Binary classification
        pos_label = unique_labels[-1]  # Usually '1'
        metrics['f1'] = round(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0), 3)

        if include_precision_recall:
            metrics['precision'] = round(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0), 3)
            metrics['recall'] = round(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0), 3)
    else:
        # Multi-class
        metrics['f1_macro'] = round(f1_score(y_true, y_pred, average='macro', zero_division=0), 3)
        metrics['f1_weighted'] = round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 3)

    return metrics


def compare_datasets(
    datasets: Dict[str, Union[str, pd.DataFrame]],
    indicators: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare metrics across multiple datasets.

    Args:
        datasets: Dictionary mapping dataset labels to either:
                 - File paths (str): e.g., 'data/baseline.csv'
                 - DataFrames (pd.DataFrame): Already loaded in notebook
                 Examples:
                   {'Baseline': 'data/baseline.csv', 'CoVe': df_cove}
                   {'Exp1': df1, 'Exp2': df2, 'Exp3': df3}
        indicators: List of indicators to compare (default: all with ground truth)

    Returns:
        Tuple of (binary_metrics_df, multiclass_metrics_df)

    Example:
        >>> # Using file paths
        >>> datasets = {
        ...     'Baseline': 'data/results/baseline.csv',
        ...     'Self-Consistency': 'data/results/sc.csv',
        ... }
        >>> binary_df, multiclass_df = compare_datasets(datasets)

        >>> # Using DataFrames
        >>> df1 = pd.read_csv('data/exp1.csv')
        >>> df2 = pd.read_csv('data/exp2.csv')
        >>> datasets = {'Exp1': df1, 'Exp2': df2}
        >>> binary_df, multiclass_df = compare_datasets(datasets)

        >>> # Mixed (file paths and DataFrames)
        >>> datasets = {
        ...     'Baseline': 'data/baseline.csv',
        ...     'Filtered': df_filtered  # Already loaded and filtered
        ... }
        >>> binary_df, multiclass_df = compare_datasets(datasets)
    """
    if indicators is None:
        indicators = INDICATORS_WITH_GROUND_TRUTH

    # Load all datasets (handle both paths and DataFrames)
    dfs = {}
    for label, data in datasets.items():
        if isinstance(data, pd.DataFrame):
            print(f"📊 Using DataFrame: {label} ({len(data)} rows)")
            dfs[label] = data
        elif isinstance(data, str):
            print(f"📂 Loading {label}: {data}")
            dfs[label] = load_predictions(data)
        else:
            raise ValueError(
                f"Dataset '{label}' must be either a file path (str) or DataFrame, "
                f"got {type(data)}"
            )

    # Validate all datasets have the same indicators
    for label, df in dfs.items():
        for indicator in indicators:
            pred_col = f'{indicator}_prediction'
            if indicator not in df.columns:
                raise ValueError(f"Dataset '{label}' missing ground truth column '{indicator}'")
            if pred_col not in df.columns:
                raise ValueError(f"Dataset '{label}' missing prediction column '{pred_col}'")

    print(f"✅ All datasets have required indicators: {indicators}\n")

    # Compute metrics for each dataset
    binary_results = []
    multiclass_results = []

    for label, df in dfs.items():
        for indicator in indicators:
            is_binary = indicator in BINARY_INDICATORS

            # Compute metrics
            metrics = compute_metrics_for_dataset(
                df,
                indicator,
                include_precision_recall=is_binary
            )

            if metrics:
                metrics['dataset'] = label

                if is_binary:
                    binary_results.append(metrics)
                else:
                    multiclass_results.append(metrics)

    binary_df = pd.DataFrame(binary_results) if binary_results else pd.DataFrame()
    multiclass_df = pd.DataFrame(multiclass_results) if multiclass_results else pd.DataFrame()

    return binary_df, multiclass_df


def plot_binary_comparison(
    binary_metrics: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 10)
):
    """
    Plot comparison of binary indicators across datasets.

    Creates 4 subplots: accuracy, precision, recall, f1-score.
    Each subplot shows all datasets for comparison.

    Args:
        binary_metrics: DataFrame from compare_datasets (binary indicators)
        figsize: Figure size

    Example:
        >>> binary_df, _ = compare_datasets(datasets)
        >>> plot_binary_comparison(binary_df)
    """
    if binary_metrics.empty:
        print("⚠️ No binary metrics to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Binary Indicators Performance Comparison', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1-Score')
    ]

    datasets = binary_metrics['dataset'].unique()
    indicators = sorted(binary_metrics['indicator'].unique())
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(datasets)))

    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]

        # Prepare data for plotting
        x = np.arange(len(indicators))
        width = 0.8 / len(datasets)

        for i, dataset in enumerate(datasets):
            dataset_data = binary_metrics[binary_metrics['dataset'] == dataset]
            values = []

            for indicator in indicators:
                row = dataset_data[dataset_data['indicator'] == indicator]
                if not row.empty and metric_key in row.columns:
                    values.append(row[metric_key].values[0])
                else:
                    values.append(0)

            offset = (i - len(datasets) / 2) * width + width / 2
            ax.bar(x + offset, values, width, label=dataset, color=colors[i], alpha=0.8)

        ax.set_xlabel('Indicator', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(indicators, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)

        # Add value labels on bars
        for i, dataset in enumerate(datasets):
            dataset_data = binary_metrics[binary_metrics['dataset'] == dataset]
            for j, indicator in enumerate(indicators):
                row = dataset_data[dataset_data['indicator'] == indicator]
                if not row.empty and metric_key in row.columns:
                    value = row[metric_key].values[0]
                    offset = (i - len(datasets) / 2) * width + width / 2
                    ax.text(
                        j + offset,
                        value + 0.02,
                        f'{value:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )

    plt.tight_layout()
    plt.show()


def plot_multiclass_comparison(
    multiclass_metrics: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot comparison of multi-class indicators across datasets.

    Shows accuracy, f1_macro, and f1_weighted side-by-side.

    Args:
        multiclass_metrics: DataFrame from compare_datasets (multi-class indicators)
        figsize: Figure size

    Example:
        >>> _, multiclass_df = compare_datasets(datasets)
        >>> plot_multiclass_comparison(multiclass_df)
    """
    if multiclass_metrics.empty:
        print("⚠️ No multi-class metrics to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Multi-Class Indicators Performance Comparison', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ('accuracy', 'Accuracy'),
        ('f1_macro', 'F1 (Macro)'),
        ('f1_weighted', 'F1 (Weighted)')
    ]

    datasets = multiclass_metrics['dataset'].unique()
    indicators = sorted(multiclass_metrics['indicator'].unique())
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(datasets)))

    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Prepare data for plotting
        x = np.arange(len(indicators))
        width = 0.8 / len(datasets)

        for i, dataset in enumerate(datasets):
            dataset_data = multiclass_metrics[multiclass_metrics['dataset'] == dataset]
            values = []

            for indicator in indicators:
                row = dataset_data[dataset_data['indicator'] == indicator]
                if not row.empty and metric_key in row.columns:
                    values.append(row[metric_key].values[0])
                else:
                    values.append(0)

            offset = (i - len(datasets) / 2) * width + width / 2
            ax.bar(x + offset, values, width, label=dataset, color=colors[i], alpha=0.8)

        ax.set_xlabel('Indicator', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(indicators, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)

        # Add value labels on bars
        for i, dataset in enumerate(datasets):
            dataset_data = multiclass_metrics[multiclass_metrics['dataset'] == dataset]
            for j, indicator in enumerate(indicators):
                row = dataset_data[dataset_data['indicator'] == indicator]
                if not row.empty and metric_key in row.columns:
                    value = row[metric_key].values[0]
                    offset = (i - len(datasets) / 2) * width + width / 2
                    ax.text(
                        j + offset,
                        value + 0.02,
                        f'{value:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )

    plt.tight_layout()
    plt.show()


def compare_experiments(
    datasets: Dict[str, Union[str, pd.DataFrame]],
    indicators: Optional[List[str]] = None,
    show_summary: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete workflow to compare multiple experiments.

    Args:
        datasets: Dictionary mapping experiment labels to either:
                 - File paths (str): CSV file paths
                 - DataFrames (pd.DataFrame): Already loaded/filtered DataFrames
                 Can mix both types in the same dictionary
        indicators: List of indicators to compare (default: all with ground truth)
        show_summary: Whether to print summary tables

    Returns:
        Tuple of (binary_metrics_df, multiclass_metrics_df)

    Example:
        >>> # Using file paths
        >>> datasets = {
        ...     'Baseline': 'data/results/baseline.csv',
        ...     'Self-Consistency': 'data/results/sc.csv',
        ...     'CoVe': 'data/results/cove.csv'
        ... }
        >>> binary_df, multiclass_df = compare_experiments(datasets)

        >>> # Using DataFrames (e.g., after filtering in notebook)
        >>> df_baseline = pd.read_csv('data/baseline.csv')
        >>> df_filtered = filter_data(df_baseline, start_year=(1800, 2000))
        >>> datasets = {
        ...     'Full Dataset': df_baseline,
        ...     'Filtered (1800-2000)': df_filtered
        ... }
        >>> binary_df, multiclass_df = compare_experiments(datasets)

        >>> # Mixed inputs
        >>> datasets = {
        ...     'Baseline': 'data/baseline.csv',
        ...     'Filtered Europe': df_europe,  # Pre-filtered DataFrame
        ...     'Filtered Asia': df_asia       # Pre-filtered DataFrame
        ... }
        >>> binary_df, multiclass_df = compare_experiments(datasets)
    """
    print("="*80)
    print("🔬 MULTI-DATASET COMPARISON")
    print("="*80)
    print(f"Comparing {len(datasets)} datasets:\n")
    for label, data in datasets.items():
        if isinstance(data, pd.DataFrame):
            print(f"  • {label}: DataFrame ({len(data)} rows)")
        elif isinstance(data, str):
            print(f"  • {label}: {data}")
        else:
            print(f"  • {label}: {type(data)}")
    print()

    # Compute metrics
    binary_metrics, multiclass_metrics = compare_datasets(datasets, indicators)

    # Show summary tables
    if show_summary:
        if not binary_metrics.empty:
            print("\n" + "="*80)
            print("📊 BINARY INDICATORS SUMMARY")
            print("="*80)
            print(binary_metrics.to_string(index=False))
            print()

        if not multiclass_metrics.empty:
            print("\n" + "="*80)
            print("📊 MULTI-CLASS INDICATORS SUMMARY")
            print("="*80)
            print(multiclass_metrics.to_string(index=False))
            print()

    # Generate plots
    print("\n" + "="*80)
    print("📈 GENERATING COMPARISON PLOTS")
    print("="*80)

    if not binary_metrics.empty:
        print("\n1️⃣ Binary Indicators Plot (Accuracy, Precision, Recall, F1-Score)...")
        plot_binary_comparison(binary_metrics)

    if not multiclass_metrics.empty:
        print("\n2️⃣ Multi-Class Indicators Plot (Accuracy, F1-Macro, F1-Weighted)...")
        plot_multiclass_comparison(multiclass_metrics)

    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETED")
    print("="*80)

    return binary_metrics, multiclass_metrics
