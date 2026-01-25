"""
Batch Processing Runner

This module provides batch processing capabilities for running predictions
on multiple polities with checkpoint support.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import pandas as pd
from tqdm import tqdm

from config import COL_TERRITORY_NAME, COL_START_YEAR, COL_END_YEAR, REQUIRED_COLUMNS
from pipeline.predictor import Predictor, PolityPrediction, PredictionConfig
from utils.cost_tracker import CostTracker
from utils.logger import ExperimentLogger


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    checkpoint_interval: int = 50  # Save checkpoint every N polities
    delay_between_calls: float = 1.0  # Seconds between API calls
    max_retries: int = 3
    retry_delay: float = 5.0
    save_intermediate: bool = True
    output_formats: List[str] = field(default_factory=lambda: ['csv', 'json'])


@dataclass
class BatchProgress:
    """Track batch processing progress."""
    total: int
    completed: int = 0
    failed: int = 0
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def remaining(self) -> int:
        return self.total - self.completed - self.failed

    @property
    def success_rate(self) -> float:
        if self.completed + self.failed == 0:
            return 0.0
        return self.completed / (self.completed + self.failed)


class BatchRunner:
    """
    Batch processor for running predictions on multiple polities.

    This class handles:
    - Loading and validating input data
    - Processing polities with progress tracking
    - Checkpoint saving and recovery
    - Output in multiple formats (CSV, JSON)
    - Cost and performance tracking

    Example:
        predictor = Predictor(config, api_keys)
        runner = BatchRunner(
            predictor,
            BatchConfig(checkpoint_interval=100),
            output_path='data/results/experiment_001.csv'
        )

        df = pd.read_csv('data/plt_polity_data_v2.csv')
        results = runner.run(df)
    """

    def __init__(
        self,
        predictor: Predictor,
        config: Optional[BatchConfig] = None,
        output_path: Optional[str] = None,
        logger: Optional[ExperimentLogger] = None
    ):
        """
        Initialize batch runner.

        Args:
            predictor: Predictor instance to use
            config: Batch processing configuration
            output_path: Path for output file
            logger: Optional experiment logger
        """
        self.predictor = predictor
        self.config = config or BatchConfig()
        self.output_path = output_path or 'data/results/batch_results.csv'
        self.logger = logger

        self.results: List[Dict] = []
        self.progress: Optional[BatchProgress] = None
        self.checkpoint_files: List[str] = []

        # Ensure output directory exists
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        df: pd.DataFrame,
        resume_from: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run predictions on all polities in the DataFrame.

        Args:
            df: Input DataFrame with polity data
            resume_from: Optional index to resume from (for recovery)

        Returns:
            DataFrame with prediction results
        """
        # Validate input
        self._validate_input(df)

        # Initialize progress
        self.progress = BatchProgress(total=len(df))
        start_idx = resume_from or 0

        if self.logger:
            self.logger.log_config({
                'total_polities': len(df),
                'model': self.predictor.config.model,
                'mode': self.predictor.config.mode.value,
                'indicators': self.predictor.config.indicators,
                'verification': self.predictor.config.verify.value
            })

        print(f"Starting batch processing of {len(df)} polities...")
        print(f"Checkpoint interval: {self.config.checkpoint_interval}")
        print(f"Output path: {self.output_path}")

        # Process each polity
        for idx in tqdm(range(start_idx, len(df)), desc="Processing polities"):
            row = df.iloc[idx]

            try:
                result = self._process_single_polity(row, idx)
                self.results.append(result)
                self.progress.completed += 1

            except Exception as e:
                print(f"\nError processing {row.get(COL_TERRITORY_NAME, 'unknown')}: {e}")
                self.progress.failed += 1

                # Add error result
                error_result = self._create_error_result(row, str(e))
                self.results.append(error_result)

                if self.logger:
                    self.logger.log_error(f"Failed: {row.get(COL_TERRITORY_NAME, 'unknown')}: {e}")

            # Save checkpoint if needed
            if self.config.save_intermediate and (idx + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(idx + 1)

            # Rate limiting delay
            time.sleep(self.config.delay_between_calls)

        # Save final results
        results_df = self._save_final_results()

        # Clean up checkpoints
        self._cleanup_checkpoints()

        # Log completion
        if self.logger:
            self.logger.log_finish()
            self.logger.log_metrics({
                'total_processed': self.progress.completed + self.progress.failed,
                'successful': self.progress.completed,
                'failed': self.progress.failed,
                'success_rate': self.progress.success_rate
            })

        print(f"\nBatch processing complete!")
        print(f"Successful: {self.progress.completed}, Failed: {self.progress.failed}")
        print(f"Results saved to: {self.output_path}")

        return results_df

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required columns."""
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _process_single_polity(self, row: pd.Series, idx: int) -> Dict:
        """Process a single polity and return result dict."""
        polity = row[COL_TERRITORY_NAME]
        start_year = int(row[COL_START_YEAR])
        end_year = int(row[COL_END_YEAR])

        # Get prediction (cost tracking is handled internally by predictor)
        prediction = self.predictor.predict(polity, start_year, end_year)

        # Convert to dict and merge with original row data
        result = row.to_dict()
        result.update(prediction.to_dict())

        return result

    def _create_error_result(self, row: pd.Series, error_msg: str) -> Dict:
        """Create result dict for failed prediction."""
        result = row.to_dict()

        for indicator in self.predictor.config.indicators:
            result[indicator] = None
            result[f'{indicator}_reasoning'] = f'Error: {error_msg}'
            result[f'{indicator}_confidence'] = None

        return result

    def _save_checkpoint(self, count: int) -> None:
        """Save checkpoint of current results."""
        checkpoint_path = f"{self.output_path}.checkpoint_{count}.csv"

        temp_df = pd.DataFrame(self.results)
        temp_df.to_csv(checkpoint_path, index=False)

        self.checkpoint_files.append(checkpoint_path)
        print(f"\n  Checkpoint saved: {checkpoint_path}")

    def _save_final_results(self) -> pd.DataFrame:
        """Save final results in all configured formats."""
        results_df = pd.DataFrame(self.results)

        # Save CSV
        if 'csv' in self.config.output_formats:
            results_df.to_csv(self.output_path, index=False)
            print(f"  Saved CSV: {self.output_path}")

        # Save JSON
        if 'json' in self.config.output_formats:
            json_path = self.output_path.replace('.csv', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"  Saved JSON: {json_path}")

        # Save cost report in data/logs/ directory
        # Create logs directory if it doesn't exist
        logs_dir = Path('data/logs')
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Get filename from output path
        output_filename = Path(self.output_path).stem
        cost_report_path = logs_dir / f'{output_filename}_costs.json'
        self.predictor.cost_tracker.save_report(str(cost_report_path))
        print(f"  Saved cost report: {cost_report_path}")

        return results_df

    def _cleanup_checkpoints(self) -> None:
        """Remove checkpoint files after successful completion."""
        for checkpoint_file in self.checkpoint_files:
            try:
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                    print(f"  Removed checkpoint: {checkpoint_file}")
            except Exception as e:
                print(f"  Warning: Could not remove checkpoint {checkpoint_file}: {e}")

        self.checkpoint_files = []


def load_polity_data(file_path: str) -> pd.DataFrame:
    """
    Load preprocessed polity data from CSV file.

    Args:
        file_path: Path to the CSV file containing polity data

    Returns:
        DataFrame with polity information

    Raises:
        ValueError: If required columns are missing
        FileNotFoundError: If the file doesn't exist
    """
    print(f"Loading polity data from {file_path}...")

    df = pd.read_csv(file_path)

    # Validate required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Data loaded successfully! Total polities: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


def run_batch_experiment(
    input_path: str,
    output_path: str,
    predictor_config: PredictionConfig,
    api_keys: Dict[str, str],
    batch_config: Optional[BatchConfig] = None,
    test_size: Optional[int] = None,
    experiment_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to run a complete batch experiment.

    Args:
        input_path: Path to input CSV
        output_path: Path for output results
        predictor_config: Configuration for predictor
        api_keys: API keys dictionary
        batch_config: Optional batch configuration
        test_size: Optional limit on number of polities (for testing)
        experiment_name: Optional name for experiment logging

    Returns:
        DataFrame with results
    """
    # Load data
    df = load_polity_data(input_path)

    # Apply test limit if specified
    if test_size:
        df = df.head(test_size)
        print(f"Test mode: Processing only {test_size} polities")

    # Create predictor
    predictor = Predictor(predictor_config, api_keys)

    # Create logger if experiment name provided
    logger = None
    if experiment_name:
        logger = ExperimentLogger(experiment_name)

    # Create runner
    runner = BatchRunner(
        predictor=predictor,
        config=batch_config or BatchConfig(),
        output_path=output_path,
        logger=logger
    )

    # Run
    return runner.run(df)
