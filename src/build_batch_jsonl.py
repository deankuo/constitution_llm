#!/usr/bin/env python3
"""
Build a JSONL file of Gemini Batch API requests from a dataset.

Self-consistency (SC) convention — matches main.py:
  n_samples = number of ADDITIONAL SC calls (not counting the initial prediction)
  Total requests per (row, indicator) = n_samples + 1
  sc_idx=0 → initial prediction (temperature=1.0)
  sc_idx=1..n_samples → SC samples (all temperature=1.0 by default)

  Example: --n-samples 2 → 3 requests per (row, indicator) → 3 votes in majority vote

Custom ID format: "{row_idx}|{indicator}|{sc_idx}"
  row_idx:   0-based positional index in the input file
  indicator: e.g. "constitution", "sovereign", "elections"
  sc_idx:    0-based index; 0 = initial prediction, 1..n = SC samples

Three task types:
  constitution  — uses prompts/constitution.py
  indicators    — uses MultiplePromptBuilder (one request per indicator per row)
  elections     — filters rows where assembly_prediction == "2"; others get
                  elections_prediction = "0" pass-through in the runner

Usage
-----
  python src/build_batch_jsonl.py \\
      --task constitution \\
      --input data/plt_leaders_data.csv \\
      --output data/temp/batch_constitution.jsonl \\
      --n-samples 2

  python src/build_batch_jsonl.py \\
      --task indicators \\
      --indicators sovereign federalism checks collegiality petition assembly entry exit symbolism \\
      --input data/plt_leaders_data.csv \\
      --output data/temp/batch_indicators.jsonl \\
      --n-samples 2

  python src/build_batch_jsonl.py \\
      --task elections \\
      --input data/results/exp001.csv \\
      --output data/temp/batch_elections.jsonl \\
      --n-samples 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    COL_TERRITORY_NAME, COL_LEADER_NAME, COL_START_YEAR, COL_END_YEAR,
    DEFAULT_MAX_TOKENS, DEFAULT_TOP_P,
)
from utils.data_loader import load_dataframe


# ---------------------------------------------------------------------------
# SC temperature schedule
# ---------------------------------------------------------------------------

def _all_temperatures(n_samples: int, sc_temperatures: list[float] = None) -> list[float]:
    """Return n_samples+1 temperatures: initial call then n_samples SC calls.

    Matches SelfConsistencyConfig default: all SC calls at temperature 1.0.
    sc_idx=0 uses initial_temp (1.0); sc_idx=1..n_samples use sc_temperatures.
    """
    if sc_temperatures is None:
        sc_temperatures = [1.0] * n_samples
    return [1.0] + list(sc_temperatures[:n_samples])


# ---------------------------------------------------------------------------
# Request dict builder
# ---------------------------------------------------------------------------

def _make_request_line(
    custom_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    n_samples: int,
) -> dict:
    """One JSONL line in Gemini InlinedRequestDict format.

    Embeds custom_id and n_samples in metadata so the runner can:
    - Match responses via metadata echo in InlinedResponse.metadata
    - Detect n_samples mismatches between build and run steps
    """
    return {
        "contents": [{"parts": [{"text": user_prompt}], "role": "user"}],
        "config": {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": DEFAULT_TOP_P,
            "response_mime_type": "application/json",
        },
        "metadata": {"custom_id": custom_id, "n_samples": str(n_samples)},
    }


# ---------------------------------------------------------------------------
# Row field extraction
# ---------------------------------------------------------------------------

def _row_fields(row: pd.Series) -> tuple[str, str, int | None, int | None]:
    """Extract (polity, name, start_year, end_year) from a dataset row."""
    polity = str(row.get(COL_TERRITORY_NAME) or "Unknown Polity")
    name = str(row.get(COL_LEADER_NAME) or "Unknown Leader")
    raw_start = row.get(COL_START_YEAR)
    raw_end = row.get(COL_END_YEAR)
    start_year = int(raw_start) if pd.notna(raw_start) else None
    end_year = int(raw_end) if pd.notna(raw_end) else None
    return polity, name, start_year, end_year


# ---------------------------------------------------------------------------
# Per-task builders
# ---------------------------------------------------------------------------

def build_constitution_requests(
    df: pd.DataFrame,
    n_samples: int,
    max_tokens: int,
    sc_temperatures: list[float] = None,
) -> list[dict]:
    """Build requests for the constitution task.

    Total requests per row = n_samples + 1 (initial + n_samples SC samples).
    """
    from prompts.constitution import get_prompt as get_constitution_prompt

    all_temps = _all_temperatures(n_samples, sc_temperatures)
    requests = []

    for row_idx in tqdm(range(len(df)), desc="constitution"):
        polity, name, start_year, end_year = _row_fields(df.iloc[row_idx])
        sys_p, usr_p = get_constitution_prompt(polity, name, start_year, end_year)

        for sc_idx, temp in enumerate(all_temps):
            custom_id = f"{row_idx}|constitution|{sc_idx}"
            requests.append(_make_request_line(custom_id, sys_p, usr_p, temp, max_tokens, n_samples))

    return requests


def build_indicator_requests(
    df: pd.DataFrame,
    indicators: list[str],
    n_samples: int,
    max_tokens: int,
    sc_temperatures: list[float] = None,
) -> list[dict]:
    """Build requests for non-constitution indicators using MultiplePromptBuilder.

    Total requests per (row, indicator) = n_samples + 1.
    """
    from prompts.multiple_builder import MultiplePromptBuilder

    if "constitution" in indicators:
        print("  Note: 'constitution' removed from indicators task (use --task constitution).")
        indicators = [i for i in indicators if i != "constitution"]

    all_temps = _all_temperatures(n_samples, sc_temperatures)
    builder = MultiplePromptBuilder(indicators=indicators, reasoning=True)
    requests = []

    for row_idx in tqdm(range(len(df)), desc="indicators"):
        polity, name, start_year, end_year = _row_fields(df.iloc[row_idx])
        prompts = builder.build(polity, name, start_year, end_year)

        for prompt in prompts:
            indicator = prompt.indicators[0]
            for sc_idx, temp in enumerate(all_temps):
                custom_id = f"{row_idx}|{indicator}|{sc_idx}"
                requests.append(
                    _make_request_line(
                        custom_id, prompt.system_prompt, prompt.user_prompt, temp, max_tokens, n_samples
                    )
                )

    return requests


def build_elections_requests(
    df: pd.DataFrame,
    n_samples: int,
    max_tokens: int,
    sc_temperatures: list[float] = None,
    prompt_version: str = "multiple",
) -> list[dict]:
    """Build requests for the elections downstream task.

    Only rows with assembly_prediction == 2 get an LLM call.
    Others are handled as pass-through (elections_prediction = "0") in the runner.
    Total requests per eligible row = n_samples + 1.
    """
    from pipeline.post_processing import (
        ELECTIONS_SYSTEM_PROMPT,
        ELECTIONS_USER_PROMPT_TEMPLATE,
        _SINGLE_PROMPTS,
    )

    if "assembly_prediction" not in df.columns:
        raise ValueError("Elections task requires 'assembly_prediction' column in input.")

    eligible_mask = pd.to_numeric(df["assembly_prediction"], errors="coerce") == 2

    if prompt_version in _SINGLE_PROMPTS:
        sys_tmpl, usr_tmpl = _SINGLE_PROMPTS[prompt_version]
    else:
        sys_tmpl, usr_tmpl = ELECTIONS_SYSTEM_PROMPT, ELECTIONS_USER_PROMPT_TEMPLATE

    all_temps = _all_temperatures(n_samples, sc_temperatures)
    requests = []
    skipped = 0

    for row_idx in tqdm(range(len(df)), desc="elections"):
        if not eligible_mask.iloc[row_idx]:
            skipped += 1
            continue

        row = df.iloc[row_idx]
        polity = str(
            row.get(COL_TERRITORY_NAME) or row.get("territorynamehistorical") or "Unknown Polity"
        )
        name = str(
            row.get(COL_LEADER_NAME) or row.get("name") or "Unknown Leader"
        )
        start_year = row.get(COL_START_YEAR) or row.get("start_year") or "?"
        end_year = row.get(COL_END_YEAR) or row.get("end_year") or "?"

        usr_p = usr_tmpl.format(
            polity=polity, name=name, start_year=start_year, end_year=end_year
        )

        for sc_idx, temp in enumerate(all_temps):
            custom_id = f"{row_idx}|elections|{sc_idx}"
            requests.append(
                _make_request_line(custom_id, sys_tmpl, usr_p, temp, max_tokens, n_samples)
            )

    n_eligible = eligible_mask.sum()
    print(f"  {n_eligible} eligible rows (assembly=2), {skipped} pass-through rows")
    return requests


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a JSONL file of Gemini batch requests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task", choices=["constitution", "indicators", "elections"], required=True,
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV or JSONL dataset.")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path.")
    parser.add_argument(
        "--indicators", nargs="+",
        default=[
            "sovereign", "federalism", "checks", "collegiality",
            "petition", "assembly", "entry", "exit", "symbolism",
        ],
        help="Indicators to include (task=indicators only).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=0,
        help=(
            "Additional SC samples per (row, indicator). "
            "0 = no SC (single call). "
            "2 = 1 initial + 2 SC = 3 total votes. "
            "Matches --n-samples in main.py."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--prompt-version", default="multiple",
        help="Elections prompt variant: 'multiple' or 'v1'/'v2'/'v3'.",
    )
    parser.add_argument(
        "--test", type=int, default=None,
        help="Process only the first N rows (for quick sanity checks).",
    )
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    df = load_dataframe(args.input)
    if args.test:
        df = df.head(args.test)
        print(f"Test mode: using {len(df)} rows")

    all_temps = _all_temperatures(args.n_samples)
    total_per_group = args.n_samples + 1
    print(
        f"Task: {args.task} | Rows: {len(df)} | "
        f"n_samples={args.n_samples} | Total per group: {total_per_group} | "
        f"Temperatures: {all_temps}"
    )

    if args.task == "constitution":
        requests = build_constitution_requests(df, args.n_samples, args.max_tokens)
    elif args.task == "indicators":
        print(f"Indicators: {args.indicators}")
        requests = build_indicator_requests(df, args.indicators, args.n_samples, args.max_tokens)
    elif args.task == "elections":
        requests = build_elections_requests(
            df, args.n_samples, args.max_tokens, prompt_version=args.prompt_version
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    file_size_mb = Path(args.output).stat().st_size / 1024 / 1024
    n_groups = len(requests) // total_per_group
    print(f"\nWrote: {args.output}")
    print(f"  {n_groups} groups × {total_per_group} requests = {len(requests)} total")
    print(f"  File size: {file_size_mb:.1f} MB")
    if file_size_mb > 1800:
        print(
            "  WARNING: File exceeds ~1.8 GB. Consider splitting by indicator "
            "or using --test for smaller runs."
        )


if __name__ == "__main__":
    main()
