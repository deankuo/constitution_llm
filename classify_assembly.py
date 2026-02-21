"""
Assembly Extended Classifier
=============================

Downstream post-processing script that extends binary assembly predictions
(0/1) to a three-label scheme (0/1/2).

Run AFTER the main pipeline has produced predictions with an
`assembly_prediction` column.

Label meanings
--------------
  0  No assembly exists             (pass-through, no API call)
  1  Assembly exists, no competitive factions or parties
  2  Assembly exists WITH competitive factions or parties

Usage
-----
  python classify_assembly.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv \\
      --model  gemini-2.5-pro \\
      --parallel-rows 4          # optional: process N rows concurrently

The script adds two new columns to the output CSV:
  assembly_extended_prediction  : "0", "1", or "2"
  assembly_extended_confidence  : integer 1-100 (null for pass-through rows)
  assembly_extended_reasoning   : reasoning text  (null for pass-through rows)

The original columns are never modified.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from models.llm_clients import create_llm
from utils.json_parser import parse_json_response

load_dotenv()


# =============================================================================
# PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a professional political scientist and historian specializing in \
legislative institutions across different historical periods.

Your task is to further classify an assembly that is KNOWN TO EXIST in a given polity. \
You must determine whether this assembly featured competitive factions or organized political parties.

## Definition of Competitive Factions or Parties (Label 2)

An assembly is coded as having **competitive factions or parties (2)** if:
- Organized groups or factions competed for influence within the assembly
- Distinct political blocs, parties, or factions held seats and competed for policy outcomes
- Examples: Roman political factions (Optimates vs Populares), English Whigs and Tories, \
modern parliamentary parties, organized voting blocs in a legislature

An assembly is coded as **existing without competitive factions (1)** if:
- The assembly exists and functions (role in selection/taxation/policy, independent, regular)
- But no organized competing factions or parties are present
- Members vote as individuals rather than as part of organized blocs

## Output Requirements

Provide a JSON object with exactly these fields:
- "assembly_extended": Must be exactly "1" or "2" (string)
- "reasoning": Your step-by-step reasoning (string)
- "confidence_score": Integer from 1 to 100

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }
"""

USER_PROMPT_TEMPLATE = """An assembly is KNOWN TO EXIST in this polity. Determine whether it \
featured competitive factions or organized political parties.

**Polity:** {polity}
**Leader:** {name}
**Reign Period:** {start_year}-{end_year}

Determine whether this assembly had competitive factions or parties:
- 2: Assembly exists AND has competitive factions or organized parties
- 1: Assembly exists but NO competitive factions or organized parties

⚠️ **IMPORTANT:** The assembly is confirmed to exist. Only classify whether competitive \
factions/parties are present (→ 2) or absent (→ 1). Focus on THIS LEADER'S REIGN.

Respond with a single JSON object:
{{"assembly_extended": "1 or 2", "reasoning": "your analysis", "confidence_score": 1-100}}
"""


# =============================================================================
# CLASSIFIER
# =============================================================================

class AssemblyExtendedClassifier:
    """
    Extends binary assembly predictions to a three-label scheme.

    Rows where assembly_prediction == "0" are passed through without
    an API call (assembly_extended_prediction = "0").

    Rows where assembly_prediction == "1" receive a second LLM call
    to determine label 1 (no competitive factions) or 2 (with factions).
    """

    # Column names for the prediction output
    PRED_COL = "assembly_extended_prediction"
    CONF_COL = "assembly_extended_confidence"
    REASON_COL = "assembly_extended_reasoning"

    def __init__(
        self,
        model: str,
        api_keys: Dict[str, str],
        assembly_col: str = "assembly_prediction",
        max_workers: int = 1,
        delay: float = 1.0,
    ):
        """
        Args:
            model:         LLM model identifier (e.g. "gemini-2.5-pro")
            api_keys:      API key dictionary
            assembly_col:  Name of the column holding binary assembly predictions
            max_workers:   Number of rows to process concurrently (1 = sequential)
            delay:         Seconds between sequential calls / between parallel windows
        """
        self.model = model
        self.llm = create_llm(model, api_keys)
        self.assembly_col = assembly_col
        self.max_workers = max_workers
        self.delay = delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run classification and return df with three new columns added.

        Args:
            df: DataFrame with at least `assembly_col`, territorynamehistorical,
                name, start_year, end_year columns.

        Returns:
            Copy of df with assembly_extended_prediction/confidence/reasoning added.
        """
        df = df.copy()

        # Initialise output columns
        df[self.PRED_COL] = None
        df[self.CONF_COL] = None
        df[self.REASON_COL] = None

        # Pass-through rows where assembly == 0
        mask_no_assembly = df[self.assembly_col].astype(str) == "0"
        df.loc[mask_no_assembly, self.PRED_COL] = "0"

        # Rows that need an LLM call
        rows_to_classify = df[~mask_no_assembly].copy()
        total = len(rows_to_classify)

        if total == 0:
            print("No rows with assembly=1 found. Nothing to classify.")
            return df

        print(f"Classifying {total} rows with assembly=1 (workers={self.max_workers})...")

        if self.max_workers > 1:
            results = self._classify_parallel(rows_to_classify)
        else:
            results = self._classify_sequential(rows_to_classify)

        for orig_idx, pred, conf, reason in results:
            df.at[orig_idx, self.PRED_COL] = pred
            df.at[orig_idx, self.CONF_COL] = conf
            df.at[orig_idx, self.REASON_COL] = reason

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, row: pd.Series) -> Tuple[str, Optional[int], str]:
        """
        Make a single LLM call for one row.

        Returns:
            (prediction, confidence_score, reasoning)
        """
        polity = str(row.get("territorynamehistorical", "Unknown Polity"))
        name = str(row.get("name", "Unknown Leader"))
        start_year = row.get("start_year", "?")
        end_year = row.get("end_year", "?")

        user_prompt = USER_PROMPT_TEMPLATE.format(
            polity=polity,
            name=name,
            start_year=start_year,
            end_year=end_year,
        )

        response = self.llm.call(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )

        parsed = parse_json_response(response.content, verbose=False)
        pred = str(parsed.get("assembly_extended", "1"))
        # Clamp to valid labels
        if pred not in ("1", "2"):
            pred = "1"
        conf = parsed.get("confidence_score")
        reason = parsed.get("reasoning", "")
        return pred, conf, reason

    def _classify_sequential(
        self, rows: pd.DataFrame
    ) -> List[Tuple[int, str, Optional[int], str]]:
        """Classify rows one at a time."""
        results = []
        for orig_idx, row in tqdm(rows.iterrows(), total=len(rows), desc="Classifying"):
            try:
                pred, conf, reason = self._call_llm(row)
            except Exception as e:
                print(f"\nError on row {orig_idx}: {e}")
                pred, conf, reason = "1", None, f"Error: {e}"
            results.append((orig_idx, pred, conf, reason))
            time.sleep(self.delay)
        return results

    def _classify_parallel(
        self, rows: pd.DataFrame
    ) -> List[Tuple[int, str, Optional[int], str]]:
        """Classify rows in parallel windows of max_workers size."""
        all_results: List[Tuple[int, str, Optional[int], str]] = []
        indices = list(rows.index)
        windows = [indices[i:i + self.max_workers] for i in range(0, len(indices), self.max_workers)]
        lock = Lock()

        with tqdm(total=len(indices), desc="Classifying") as pbar:
            for window in windows:
                window_results: List[Tuple[int, str, Optional[int], str]] = []

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_idx = {
                        executor.submit(self._call_llm, rows.loc[orig_idx]): orig_idx
                        for orig_idx in window
                    }
                    for future in as_completed(future_to_idx):
                        orig_idx = future_to_idx[future]
                        try:
                            pred, conf, reason = future.result()
                        except Exception as e:
                            print(f"\nError on row {orig_idx}: {e}")
                            pred, conf, reason = "1", None, f"Error: {e}"
                        window_results.append((orig_idx, pred, conf, reason))

                # Sort by original index to preserve order
                window_results.sort(key=lambda x: x[0])
                all_results.extend(window_results)
                pbar.update(len(window))

                if window != windows[-1]:
                    time.sleep(self.delay)

        return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Assembly Extended Classifier\n"
            "\n"
            "Post-processing script that extends binary assembly predictions (0/1)\n"
            "to a three-label scheme (0/1/2).\n"
            "\n"
            "  0  No assembly (pass-through, no API call)\n"
            "  1  Assembly exists, no competitive factions/parties\n"
            "  2  Assembly exists WITH competitive factions/parties\n"
            "\n"
            "Must be run AFTER the main pipeline has produced a predictions CSV\n"
            "that contains an assembly_prediction column."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python classify_assembly.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv

  # Use a different model
  python classify_assembly.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv \\
      --model  gpt-4o

  # Process 4 rows in parallel
  python classify_assembly.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv \\
      --parallel-rows 4

  # Custom assembly column name (if your pipeline used a different column)
  python classify_assembly.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv \\
      --assembly-col assembly
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to predictions CSV from the main pipeline"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to write the extended predictions CSV"
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-2.5-pro",
        help="LLM model identifier (default: gemini-2.5-pro)"
    )
    parser.add_argument(
        "--assembly-col",
        default="assembly_prediction",
        help="Column name holding binary assembly predictions (default: assembly_prediction)"
    )
    parser.add_argument(
        "--parallel-rows",
        type=int,
        default=1,
        metavar="N",
        help="Number of rows to process in parallel (default: 1 = sequential)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between API calls / between parallel windows (default: 1.0)"
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N rows (for testing)"
    )

    args = parser.parse_args()

    # --- Load API keys ---
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
    }

    # --- Load input data ---
    print(f"Loading predictions from: {args.input}")
    df = pd.read_csv(args.input)

    if args.assembly_col not in df.columns:
        raise ValueError(
            f"Column '{args.assembly_col}' not found in input CSV. "
            f"Available columns: {list(df.columns)}"
        )

    if args.test:
        df = df.head(args.test)
        print(f"Test mode: processing only first {args.test} rows")

    print(f"Loaded {len(df)} rows. "
          f"assembly=0: {(df[args.assembly_col].astype(str) == '0').sum()}, "
          f"assembly=1: {(df[args.assembly_col].astype(str) == '1').sum()}")

    # --- Run classifier ---
    classifier = AssemblyExtendedClassifier(
        model=args.model,
        api_keys=api_keys,
        assembly_col=args.assembly_col,
        max_workers=args.parallel_rows,
        delay=args.delay,
    )

    result_df = classifier.classify(df)

    # --- Save output ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved extended predictions to: {args.output}")

    # --- Summary ---
    pred_counts = result_df[AssemblyExtendedClassifier.PRED_COL].value_counts().sort_index()
    print("\nassembly_extended_prediction distribution:")
    for label, count in pred_counts.items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
