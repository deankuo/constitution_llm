#!/usr/bin/env python3
"""
Build a JSONL file of Gemini Batch API requests from a dataset.

Self-consistency (SC) convention — matches main.py:
  n_samples = number of ADDITIONAL SC calls (not counting the initial prediction)
  Total requests per group = n_samples + 1
  sc_idx=0 → initial prediction (temperature=1.0)
  sc_idx=1..n_samples → SC samples (all temperature=1.0 by default)

  Example: --n-samples 2 → 3 requests per group → 3 votes in majority vote

Custom ID format:
  constitution/elections:  "{row_idx}|{indicator}|{sc_idx}"
  indicators task:         "{row_idx}|single|{sc_idx}"  (one request covers ALL indicators)
    row_idx:   0-based positional index in the input file
    sc_idx:    0-based index; 0 = initial prediction, 1..n = SC samples
  The selected indicators list is embedded in metadata["indicators"] for the runner.

Three task types:
  constitution  — uses prompts/constitution.py
  indicators    — uses SinglePromptBuilder (one request per row covers all indicators)
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
      --indicators sovereign federalism checks_local checks_military checks_clergy checks_aristocracy checks_bourgeoisie checks_bureaucracy checks_judiciary checks_assembly checks_council collegiality petition assembly entry exit symbolism \\
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

MAX_CHUNK_BYTES = 1_900 * 1024 * 1024  # 1.9 GB — safely under Gemini's 2 GB batch limit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    COL_TERRITORY_NAME, COL_LEADER_NAME, COL_START_YEAR, COL_END_YEAR,
    DEFAULT_MAX_TOKENS, DEFAULT_TOP_P, DEFAULT_VERIFICATION_CONFIG,
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
# JSONL writer with auto-chunking
# ---------------------------------------------------------------------------

def _write_chunks(
    requests: list[dict],
    output_path: str,
    max_bytes: int = MAX_CHUNK_BYTES,
) -> list[Path]:
    """Write requests to one or more JSONL files, splitting when size exceeds max_bytes.

    If all requests fit in one file, writes to output_path unchanged.
    Otherwise writes {stem}_chunk001.jsonl, {stem}_chunk002.jsonl, ...

    Returns list of written file paths.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize all lines once; measure byte size
    lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in requests]
    total_bytes = sum(len(l.encode("utf-8")) for l in lines)

    if total_bytes <= max_bytes:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return [path]

    # Chunked write
    chunks: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for line in lines:
        lb = len(line.encode("utf-8"))
        if current and current_size + lb > max_bytes:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(line)
        current_size += lb
    if current:
        chunks.append(current)

    written = []
    for i, chunk_lines in enumerate(chunks):
        chunk_path = path.parent / f"{path.stem}_chunk{i + 1:03d}{path.suffix}"
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.writelines(chunk_lines)
        written.append(chunk_path)

    return written


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
            req = _make_request_line(custom_id, sys_p, usr_p, temp, max_tokens, n_samples)
            if sc_idx == 0:
                req["metadata"]["row_data"] = json.dumps(
                    df.iloc[row_idx].to_dict(), ensure_ascii=False, default=str
                )
            requests.append(req)

    return requests


def build_indicator_requests(
    df: pd.DataFrame,
    indicators: list[str],
    n_samples: int,
    max_tokens: int,
    sc_temperatures: list[float] = None,
    reasoning: bool = True,
    prompt_version: str = "v1",
) -> list[dict]:
    """Build requests for non-constitution indicators using SinglePromptBuilder.

    One request per row covers ALL selected indicators in a single prompt.
    Custom_id: "{row_idx}|single|{sc_idx}"; indicators list embedded in metadata.
    Total requests per row = n_samples + 1.

    Args:
        reasoning:      Include reasoning fields in output (default True). Set False
                        to reduce output tokens and cost.
        prompt_version: Which SinglePromptBuilder variant to use:
                        'v1' (full definitions), 'v2' (tabular), 'v3' (compact).
    """
    from prompts.single_builder import SinglePromptBuilder, SinglePromptBuilderV2, SinglePromptBuilderV3

    _BUILDER_CLS = {"v1": SinglePromptBuilder, "v2": SinglePromptBuilderV2, "v3": SinglePromptBuilderV3}
    BuilderCls = _BUILDER_CLS.get(prompt_version, SinglePromptBuilder)
    if prompt_version not in _BUILDER_CLS:
        print(f"  Warning: unknown --prompt-version '{prompt_version}', falling back to v1.")

    if "constitution" in indicators:
        print("  Note: 'constitution' removed from indicators task (use --task constitution).")
        indicators = [i for i in indicators if i != "constitution"]

    all_temps = _all_temperatures(n_samples, sc_temperatures)
    builder = BuilderCls(indicators=indicators, reasoning=reasoning)
    indicators_json = json.dumps(indicators)
    requests = []

    for row_idx in tqdm(range(len(df)), desc="indicators"):
        polity, name, start_year, end_year = _row_fields(df.iloc[row_idx])
        prompts = builder.build(polity, name, start_year, end_year)
        prompt = prompts[0]  # SinglePromptBuilder returns exactly one PromptOutput

        for sc_idx, temp in enumerate(all_temps):
            custom_id = f"{row_idx}|single|{sc_idx}"
            req = _make_request_line(
                custom_id, prompt.system_prompt, prompt.user_prompt, temp, max_tokens, n_samples
            )
            req["metadata"]["indicators"] = indicators_json
            if sc_idx == 0:
                req["metadata"]["row_data"] = json.dumps(
                    df.iloc[row_idx].to_dict(), ensure_ascii=False, default=str
                )
            requests.append(req)

    return requests


def build_elections_requests(
    df: pd.DataFrame,
    n_samples: int,
    max_tokens: int,
    sc_temperatures: list[float] = None,
    prompt_version: str = "v1",
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
            req = _make_request_line(custom_id, sys_tmpl, usr_p, temp, max_tokens, n_samples)
            if sc_idx == 0:
                req["metadata"]["row_data"] = json.dumps(
                    row.to_dict(), ensure_ascii=False, default=str
                )
            requests.append(req)

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
            "sovereign", "federalism",
            "checks_local", "checks_military", "checks_clergy", "checks_aristocracy",
            "checks_bourgeoisie", "checks_bureaucracy", "checks_judiciary",
            "checks_assembly", "checks_council",
            "collegiality", "petition", "assembly", "entry", "exit", "symbolism",
        ],
        help="Indicators to include (task=indicators only).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=2,
        help=(
            "Additional SC samples per (row, indicator). "
            "0 = no SC (single call). "
            "2 = 1 initial + 2 SC = 3 total votes. "
            "Matches --n-samples in main.py."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--prompt-version", default="v1",
        help=(
            "Prompt variant to use.\n"
            "  indicators task: 'v1' (full definitions, default), 'v2' (tabular), 'v3' (compact/token-efficient).\n"
            "  elections task:  'v1' (default, matches post_processing.py), 'v2', 'v3', or 'multiple'."
        ),
    )
    parser.add_argument(
        "--reasoning",
        type=lambda x: x.lower() == "true",
        default=False,
        help=(
            "Include reasoning fields in the prompt output (default: False). "
            "Set to False to reduce output tokens — omits {indicator}_reasoning from the response JSON."
        ),
    )
    parser.add_argument(
        "--max-size-mb", type=int, default=1900,
        help=(
            "Maximum JSONL chunk size in MB (default: 1900 = 1.9 GB). "
            "If the output exceeds this, it is split into {stem}_chunk001.jsonl, etc. "
            "Gemini Batch API limit is 2 GB per job."
        ),
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
        print(f"Indicators: {args.indicators} | prompt_version={args.prompt_version} | reasoning={args.reasoning}")
        requests = build_indicator_requests(
            df, args.indicators, args.n_samples, args.max_tokens,
            reasoning=args.reasoning, prompt_version=args.prompt_version,
        )
    elif args.task == "elections":
        requests = build_elections_requests(
            df, args.n_samples, args.max_tokens, prompt_version=args.prompt_version
        )

    max_bytes = args.max_size_mb * 1024 * 1024
    written_files = _write_chunks(requests, args.output, max_bytes)

    n_groups = len(requests) // total_per_group if total_per_group else len(requests)
    total_mb = sum(p.stat().st_size for p in written_files) / 1024 / 1024
    print(f"\n{n_groups} groups × {total_per_group} requests = {len(requests)} total ({total_mb:.1f} MB)")
    if len(written_files) == 1:
        print(f"Wrote: {written_files[0]}")
    else:
        print(f"Split into {len(written_files)} chunks (limit: {args.max_size_mb} MB each):")
        for p in written_files:
            print(f"  {p}  ({p.stat().st_size / 1024 / 1024:.1f} MB)")

    # Build manifest: records the build configuration alongside the JSONL for reproducibility.
    # The runner adds model and job-level info at submit time (see _provenance.json output).
    # Note: model is not known at build time — pass it to the runner via --model.
    manifest: dict = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "n_samples": args.n_samples,
        "total_requests": len(requests),
        "total_groups": n_groups,
        "source_file": str(Path(args.input).resolve()),
        "output_files": [str(p) for p in written_files],
    }
    if args.task == "indicators":
        manifest["indicators"] = args.indicators
        manifest["prompt_version"] = args.prompt_version
        manifest["reasoning"] = args.reasoning
    elif args.task == "elections":
        manifest["prompt_version"] = args.prompt_version
    manifest_path = Path(written_files[0]).parent / "logs" / (Path(args.output).stem + "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
