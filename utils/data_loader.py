"""
Unified data loading utility supporting CSV and JSONL formats.

Usage:
    from utils.data_loader import load_dataframe

    df = load_dataframe("data/input.csv")    # CSV
    df = load_dataframe("data/input.jsonl")  # JSONL
"""

import json
from io import StringIO
from pathlib import Path

import pandas as pd


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV or JSONL file, auto-detected by extension.

    Args:
        file_path: Path to the input file (.csv or .jsonl)

    Returns:
        pandas DataFrame

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file extension is not supported or file is empty
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {file_path}")

    path_lower = str(path).lower()

    if path_lower.endswith('.jsonl'):
        df = _load_jsonl(path)
    elif path_lower.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {file_path}. "
            "Expected .csv or .jsonl extension."
        )

    return df


def _load_jsonl(path: Path) -> pd.DataFrame:
    """
    Load a JSONL file into a DataFrame, handling empty lines and encoding.

    Reads the file explicitly to avoid the pandas FutureWarning about
    passing literal JSON strings, and filters out blank lines that would
    cause ``ValueError: Expected object or value``.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e

    if not records:
        raise ValueError(f"No valid JSON records found in: {path}")

    return pd.DataFrame(records)
