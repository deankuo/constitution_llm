#!/usr/bin/env python3
"""
Convert a CSV file to JSONL (JSON Lines) format.

Usage:
    python scripts/csv_to_jsonl.py -i data/input.csv -o data/input.jsonl
    python scripts/csv_to_jsonl.py -i data/input.csv                    # auto-names output
"""

import argparse
import os

import pandas as pd


def csv_to_jsonl(input_path: str, output_path: str) -> None:
    """
    Convert a CSV file to JSONL format.

    Args:
        input_path: Path to the input CSV file
        output_path: Path for the output JSONL file
    """
    df = pd.read_csv(input_path)
    df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    print(f"Converted {len(df)} rows: {input_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV to JSONL format"
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output JSONL file path (default: same name with .jsonl extension)'
    )

    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        base, _ = os.path.splitext(args.input)
        output_path = base + '.jsonl'

    csv_to_jsonl(args.input, output_path)


if __name__ == '__main__':
    main()
