"""
Result Analysis and Reporting

This module provides tools for analyzing prediction results:
- Compare predictions across models
- Generate summary reports
- Identify disagreements
- Export analysis results
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import INDICATOR_LABELS, INDICATORS_WITH_GROUND_TRUTH
from evaluation.metrics import (
    evaluate_indicator,
    calculate_agreement,
    calculate_cohens_kappa,
    format_metrics_report,
    IndicatorMetrics
)


@dataclass
class ModelComparison:
    """Results of comparing two models."""
    model_1: str
    model_2: str
    indicator: str
    agreement_rate: float
    cohens_kappa: float
    disagreements: List[Dict]


@dataclass
class ExperimentAnalysis:
    """Complete analysis of an experiment."""
    experiment_name: str
    timestamp: datetime
    total_polities: int
    indicators_analyzed: List[str]
    metrics_by_indicator: Dict[str, IndicatorMetrics]
    model_comparisons: List[ModelComparison]
    summary_stats: Dict[str, Any]


class ResultAnalyzer:
    """
    Analyzer for prediction results.

    This class provides tools for:
    - Evaluating predictions against ground truth
    - Comparing predictions across models
    - Identifying and analyzing disagreements
    - Generating summary reports

    Example:
        analyzer = ResultAnalyzer()

        # Load results
        results_df = pd.read_csv('results.csv')
        ground_truth_df = pd.read_csv('ground_truth.csv')

        # Evaluate
        metrics = analyzer.evaluate_predictions(
            results_df,
            ground_truth_df,
            indicators=['sovereign', 'assembly']
        )

        # Generate report
        analyzer.generate_report(metrics, 'analysis_report.md')
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.results_cache: Dict[str, pd.DataFrame] = {}

    def evaluate_predictions(
        self,
        predictions_df: pd.DataFrame,
        ground_truth_df: Optional[pd.DataFrame] = None,
        indicators: Optional[List[str]] = None,
        prediction_suffix: str = '',
        truth_suffix: str = ''
    ) -> Dict[str, IndicatorMetrics]:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions_df: DataFrame with predictions
            ground_truth_df: DataFrame with ground truth (if separate)
            indicators: List of indicators to evaluate
            prediction_suffix: Suffix for prediction columns (e.g., '_gemini')
            truth_suffix: Suffix for ground truth columns

        Returns:
            Dictionary mapping indicator names to their metrics
        """
        if indicators is None:
            indicators = INDICATORS_WITH_GROUND_TRUTH

        # Merge if ground truth is separate
        if ground_truth_df is not None:
            # Assuming they can be merged on index or a common key
            df = predictions_df.copy()
            for col in ground_truth_df.columns:
                if col not in df.columns:
                    df[col] = ground_truth_df[col].values
        else:
            df = predictions_df

        metrics = {}

        for indicator in indicators:
            pred_col = f"{indicator}{prediction_suffix}" if prediction_suffix else indicator
            truth_col = f"{indicator}{truth_suffix}" if truth_suffix else f"{indicator}_truth"

            # Try different column name patterns
            possible_pred_cols = [pred_col, indicator, f"{indicator}_predicted"]
            possible_truth_cols = [truth_col, f"{indicator}_gt", f"gt_{indicator}", indicator]

            pred_col_found = None
            truth_col_found = None

            for col in possible_pred_cols:
                if col in df.columns:
                    pred_col_found = col
                    break

            for col in possible_truth_cols:
                if col in df.columns and col != pred_col_found:
                    truth_col_found = col
                    break

            if pred_col_found is None:
                print(f"Warning: No prediction column found for {indicator}")
                continue

            if truth_col_found is None:
                print(f"Warning: No ground truth column found for {indicator}")
                continue

            predictions = df[pred_col_found].tolist()
            ground_truth = df[truth_col_found].tolist()

            labels = INDICATOR_LABELS.get(indicator, ['0', '1'])

            metrics[indicator] = evaluate_indicator(
                predictions=predictions,
                ground_truth=ground_truth,
                indicator=indicator,
                labels=labels
            )

        return metrics

    def compare_models(
        self,
        df: pd.DataFrame,
        model_1_suffix: str,
        model_2_suffix: str,
        indicators: Optional[List[str]] = None
    ) -> List[ModelComparison]:
        """
        Compare predictions from two different models.

        Args:
            df: DataFrame with predictions from both models
            model_1_suffix: Suffix for model 1 columns (e.g., '_gemini')
            model_2_suffix: Suffix for model 2 columns (e.g., '_gpt')
            indicators: List of indicators to compare

        Returns:
            List of ModelComparison results
        """
        if indicators is None:
            indicators = INDICATORS_WITH_GROUND_TRUTH

        comparisons = []

        for indicator in indicators:
            col_1 = f"{indicator}{model_1_suffix}"
            col_2 = f"{indicator}{model_2_suffix}"

            if col_1 not in df.columns or col_2 not in df.columns:
                print(f"Warning: Missing columns for {indicator} comparison")
                continue

            preds_1 = df[col_1].astype(str).tolist()
            preds_2 = df[col_2].astype(str).tolist()

            agreement = calculate_agreement(preds_1, preds_2)
            kappa = calculate_cohens_kappa(preds_1, preds_2)

            # Find disagreements
            disagreements = []
            for idx, (p1, p2) in enumerate(zip(preds_1, preds_2)):
                if p1 != p2:
                    disagreements.append({
                        'index': idx,
                        'polity': df.iloc[idx].get('territorynamehistorical', f'row_{idx}'),
                        f'{model_1_suffix}': p1,
                        f'{model_2_suffix}': p2
                    })

            comparisons.append(ModelComparison(
                model_1=model_1_suffix,
                model_2=model_2_suffix,
                indicator=indicator,
                agreement_rate=agreement,
                cohens_kappa=kappa,
                disagreements=disagreements[:100]  # Limit to first 100
            ))

        return comparisons

    def analyze_disagreements(
        self,
        df: pd.DataFrame,
        model_suffixes: List[str],
        indicator: str,
        min_models_disagree: int = 2
    ) -> pd.DataFrame:
        """
        Analyze polities where models disagree.

        Args:
            df: DataFrame with predictions
            model_suffixes: List of model suffixes to compare
            indicator: Indicator to analyze
            min_models_disagree: Minimum number of models that must disagree

        Returns:
            DataFrame of disagreements with details
        """
        cols = [f"{indicator}{suffix}" for suffix in model_suffixes]
        valid_cols = [c for c in cols if c in df.columns]

        if len(valid_cols) < 2:
            return pd.DataFrame()

        # Find rows with disagreement
        disagreement_rows = []

        for idx, row in df.iterrows():
            predictions = {col: str(row[col]) for col in valid_cols}
            unique_preds = set(predictions.values()) - {'nan', 'None', ''}

            if len(unique_preds) >= min_models_disagree:
                disagreement_rows.append({
                    'index': idx,
                    'polity': row.get('territorynamehistorical', ''),
                    'start_year': row.get('start_year', ''),
                    'end_year': row.get('end_year', ''),
                    **predictions,
                    'n_unique_predictions': len(unique_preds)
                })

        return pd.DataFrame(disagreement_rows)

    def generate_report(
        self,
        metrics: Dict[str, IndicatorMetrics],
        output_path: str,
        comparisons: Optional[List[ModelComparison]] = None,
        title: str = "Prediction Analysis Report"
    ) -> None:
        """
        Generate a formatted analysis report.

        Args:
            metrics: Dictionary of indicator metrics
            output_path: Path for output file (.md or .txt)
            comparisons: Optional model comparisons to include
            title: Report title
        """
        lines = [
            f"# {title}",
            f"",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Summary",
            f"",
            f"| Indicator | Accuracy | Macro F1 | Valid Samples |",
            f"|-----------|----------|----------|---------------|",
        ]

        for indicator, ind_metrics in sorted(metrics.items()):
            lines.append(
                f"| {indicator} | {ind_metrics.accuracy:.4f} | "
                f"{ind_metrics.macro_f1:.4f} | {ind_metrics.valid_samples} |"
            )

        lines.append("")
        lines.append("## Detailed Metrics")
        lines.append("")

        for indicator, ind_metrics in sorted(metrics.items()):
            lines.append(format_metrics_report(ind_metrics))

        if comparisons:
            lines.append("## Model Comparisons")
            lines.append("")
            lines.append("| Indicator | Model 1 | Model 2 | Agreement | Kappa |")
            lines.append("|-----------|---------|---------|-----------|-------|")

            for comp in comparisons:
                lines.append(
                    f"| {comp.indicator} | {comp.model_1} | {comp.model_2} | "
                    f"{comp.agreement_rate:.4f} | {comp.cohens_kappa:.4f} |"
                )

            lines.append("")

        # Write report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        print(f"Report saved to: {output_path}")

    def export_analysis(
        self,
        analysis: ExperimentAnalysis,
        output_dir: str
    ) -> None:
        """
        Export complete analysis to multiple formats.

        Args:
            analysis: ExperimentAnalysis to export
            output_dir: Directory for output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export metrics as JSON
        metrics_data = {
            indicator: {
                'accuracy': m.accuracy,
                'macro_f1': m.macro_f1,
                'macro_precision': m.macro_precision,
                'macro_recall': m.macro_recall,
                'per_class': {
                    label: {
                        'precision': cm.precision,
                        'recall': cm.recall,
                        'f1': cm.f1,
                        'support': cm.support
                    }
                    for label, cm in m.per_class.items()
                },
                'confusion_matrix': m.confusion_matrix
            }
            for indicator, m in analysis.metrics_by_indicator.items()
        }

        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)

        # Export comparisons as CSV
        if analysis.model_comparisons:
            comparison_rows = []
            for comp in analysis.model_comparisons:
                comparison_rows.append({
                    'indicator': comp.indicator,
                    'model_1': comp.model_1,
                    'model_2': comp.model_2,
                    'agreement_rate': comp.agreement_rate,
                    'cohens_kappa': comp.cohens_kappa,
                    'n_disagreements': len(comp.disagreements)
                })
            pd.DataFrame(comparison_rows).to_csv(
                output_path / 'model_comparisons.csv',
                index=False
            )

        # Export summary
        summary = {
            'experiment_name': analysis.experiment_name,
            'timestamp': analysis.timestamp.isoformat(),
            'total_polities': analysis.total_polities,
            'indicators_analyzed': analysis.indicators_analyzed,
            'summary_stats': analysis.summary_stats
        }

        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Analysis exported to: {output_dir}")


def quick_evaluate(
    results_path: str,
    ground_truth_path: Optional[str] = None,
    indicators: Optional[List[str]] = None
) -> Dict[str, IndicatorMetrics]:
    """
    Quick evaluation function for command line use.

    Args:
        results_path: Path to results CSV
        ground_truth_path: Optional path to ground truth CSV
        indicators: Optional list of indicators to evaluate

    Returns:
        Dictionary of indicator metrics
    """
    analyzer = ResultAnalyzer()

    results_df = pd.read_csv(results_path)

    ground_truth_df = None
    if ground_truth_path:
        ground_truth_df = pd.read_csv(ground_truth_path)

    metrics = analyzer.evaluate_predictions(
        results_df,
        ground_truth_df,
        indicators
    )

    # Print summary
    print("\nQuick Evaluation Results:")
    print("=" * 50)
    for indicator, m in metrics.items():
        print(f"{indicator}: Accuracy={m.accuracy:.4f}, F1={m.macro_f1:.4f}")
    print("=" * 50)

    return metrics
