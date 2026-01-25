"""
Logging utilities for Constitution Analysis Pipeline.

This module provides a consistent logging interface across all modules.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Default log format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SIMPLE_FORMAT = '%(levelname)s: %(message)s'


def setup_logger(
    name: str = 'constitution_llm',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = DEFAULT_FORMAT,
    console: bool = True
) -> logging.Logger:
    """
    Set up and return a configured logger.

    Args:
        name: Logger name (use __name__ for module-specific loggers)
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Log message format
        console: Whether to log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(format_string)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = 'constitution_llm') -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If no handlers, set up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger


class ExperimentLogger:
    """
    Logger for tracking experiment progress and results.

    This class provides structured logging for experiments including
    progress tracking, metrics logging, and timing information.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = 'data/results/logs',
        level: int = logging.INFO
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
            level: Logging level
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now()

        # Create unique log file name
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        log_file = f"{log_dir}/{experiment_name}_{timestamp}.log"

        self.logger = setup_logger(
            name=f'experiment.{experiment_name}',
            level=level,
            log_file=log_file
        )

        self.logger.info(f"Experiment '{experiment_name}' started at {self.start_time}")

    def log_config(self, config: dict) -> None:
        """Log experiment configuration."""
        self.logger.info("Experiment Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def log_progress(self, current: int, total: int, message: str = '') -> None:
        """Log progress update."""
        percentage = (current / total) * 100 if total > 0 else 0
        self.logger.info(f"Progress: {current}/{total} ({percentage:.1f}%) {message}")

    def log_prediction(
        self,
        polity: str,
        indicator: str,
        prediction: str,
        confidence: Optional[int] = None
    ) -> None:
        """Log a single prediction."""
        conf_str = f" (confidence: {confidence})" if confidence else ""
        self.logger.debug(f"Prediction: {polity} | {indicator}={prediction}{conf_str}")

    def log_error(self, message: str, exc_info: bool = True) -> None:
        """Log an error."""
        self.logger.error(message, exc_info=exc_info)

    def log_warning(self, message: str) -> None:
        """Log a warning."""
        self.logger.warning(message)

    def log_metrics(self, metrics: dict) -> None:
        """Log evaluation metrics."""
        self.logger.info("Evaluation Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")

    def log_cost(self, model: str, cost: float) -> None:
        """Log API cost."""
        self.logger.info(f"API Cost - {model}: ${cost:.4f}")

    def log_finish(self) -> None:
        """Log experiment completion."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.logger.info(f"Experiment '{self.experiment_name}' completed")
        self.logger.info(f"Duration: {duration}")


# Create a default logger for the package
default_logger = get_logger('constitution_llm')
