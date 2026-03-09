"""
Unified data loading utility supporting CSV and JSONL formats.

Usage:
    from utils.data_loader import load_dataframe

    df = load_dataframe("data/input.csv")    # CSV
    df = load_dataframe("data/input.jsonl")  # JSONL
"""

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
        ValueError: If the file extension is not supported
    """
    path_lower = file_path.lower()

    if path_lower.endswith('.jsonl'):
        df = pd.read_json(file_path, lines=True)
    elif path_lower.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {file_path}. "
            "Expected .csv or .jsonl extension."
        )

    return df
