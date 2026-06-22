#!/usr/bin/env python3
"""
Gemini Batch API runner — single-job, SC-embedded approach.

Two entry points
----------------
run_inline_batch(df, ...)
    Called by main.py when --use-batch is set. Builds requests in memory,
    submits as ONE Gemini batch job, parses SC, returns enriched DataFrame.
    No JSONL file needed; no checkpoints.

run_from_jsonl(jsonl_path, ...)
    Standalone use: reads a pre-built JSONL (from src/build_batch_jsonl.py),
    submits as one job, parses SC, writes CSV + JSON.

Self-consistency convention — matches main.py / SelfConsistencyConfig
----------------------------------------------------------------------
  n_samples = number of ADDITIONAL SC calls (not counting the initial)
  Total requests per (row, indicator) = n_samples + 1
    sc_idx=0  → initial prediction (temp=1.0)
    sc_idx=1..n_samples → SC samples (temp=1.0 each, default)
  Total votes in majority = n_samples + 1

  n_samples=0 → no SC (single call), no _verified/_agreement/_uncertainty columns
  n_samples=2 → 3 votes, write _verified/_agreement/_uncertainty

Custom ID format: "{row_idx}|{indicator}|{sc_idx}"

Usage (standalone)
------------------
  python pipeline/jsonl_batch_runner.py \\
      --jsonl data/temp/batch_constitution.jsonl \\
      --input data/plt_leaders_data.csv \\
      --output data/results/exp001.csv \\
      --model gemini-3.1-pro-preview \\
      --n-samples 2
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    INDICATOR_LABELS,
    COL_TERRITORY_NAME, COL_LEADER_NAME, COL_START_YEAR, COL_END_YEAR,
    DEFAULT_MAX_TOKENS, DEFAULT_TOP_P,
)
from utils.data_loader import load_dataframe
from utils.json_parser import (
    parse_json_response,
    validate_constitution_response,
    validate_indicator_response,
)

BATCH_DISCOUNT = 0.5


# ---------------------------------------------------------------------------
# Custom ID helpers
# ---------------------------------------------------------------------------

def _parse_custom_id(cid: str) -> tuple[int, str, int]:
    """Parse "{row_idx}|{indicator}|{sc_idx}" → (row_idx, indicator, sc_idx)."""
    parts = cid.split("|")
    if len(parts) != 3:
        raise ValueError(f"Unexpected custom_id format: {cid!r}")
    return int(parts[0]), parts[1], int(parts[2])


# ---------------------------------------------------------------------------
# SC aggregation
# ---------------------------------------------------------------------------

def _normalize_pred(pred, indicator: str) -> str:
    """Normalize a raw parsed prediction to a stable comparable string."""
    if indicator == "checks" and isinstance(pred, list):
        return json.dumps(sorted(int(x) for x in pred))
    if isinstance(pred, float):
        return str(int(pred))
    return str(pred)


def _aggregate_sc(
    votes: list[str],
    indicator: str,
) -> tuple[Optional[str], float, str]:
    """Majority vote over SC votes. Matches SelfConsistencyVerification._aggregate_predictions().

    Returns (final_pred_str, agreement_ratio, uncertainty).
    uncertainty: 'none' (unanimous) | 'low' (majority ≥ 2) | 'high' (all differ)

    checks (multi-select): when all differ, use intersection of all vote sets.
    Others: when all differ, fall back to votes[0] (the sc_idx=0 initial prediction).
    """
    if not votes:
        return None, 0.0, "high"

    n = len(votes)
    counter = Counter(votes)
    winner, winner_count = counter.most_common(1)[0]
    agreement = winner_count / n

    if winner_count == n:
        uncertainty = "none"
    elif winner_count >= 2:
        uncertainty = "low"
    else:
        uncertainty = "high"
        if indicator == "checks":
            try:
                sets = [set(json.loads(v)) for v in votes]
                common = set.intersection(*sets)
                winner = json.dumps(sorted(common)) if common else None
            except Exception:
                winner = votes[0]
        else:
            winner = votes[0]  # fall back to sc_idx=0 (initial prediction)

    return winner, agreement, uncertainty


def _denormalize_pred(pred: Optional[str], indicator: str):
    """Convert a normalized prediction string back to the storage format."""
    if pred is None:
        return None
    if indicator == "checks":
        try:
            return json.loads(pred)
        except (json.JSONDecodeError, TypeError):
            return pred
    return pred


# ---------------------------------------------------------------------------
# Request building (mirrors build_batch_jsonl.py, but in-memory)
# ---------------------------------------------------------------------------

def _all_temperatures(n_samples: int, sc_temperatures: list[float] = None) -> list[float]:
    """Return n_samples+1 temperatures: [initial_temp] + sc_temperatures[:n_samples]."""
    if sc_temperatures is None:
        sc_temperatures = [1.0] * n_samples
    return [1.0] + list(sc_temperatures[:n_samples])


def _make_req(custom_id: str, system_prompt: str, user_prompt: str,
              temperature: float, max_tokens: int, n_samples: int) -> dict:
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


def _row_fields(row: pd.Series) -> tuple[str, str, Optional[int], Optional[int]]:
    polity = str(row.get(COL_TERRITORY_NAME) or "Unknown Polity")
    name = str(row.get(COL_LEADER_NAME) or "Unknown Leader")
    raw_start = row.get(COL_START_YEAR)
    raw_end = row.get(COL_END_YEAR)
    start_year = int(raw_start) if pd.notna(raw_start) else None
    end_year = int(raw_end) if pd.notna(raw_end) else None
    return polity, name, start_year, end_year


def _build_requests_in_memory(
    df: pd.DataFrame,
    indicators: list[str],
    n_samples: int,
    max_tokens: int,
    sc_temperatures: list[float] = None,
    prompt_builder=None,
) -> list[dict]:
    """Build all batch requests in memory.

    If prompt_builder is provided (e.g. monkey-patched for forced search),
    it is used for all indicators including constitution.
    Otherwise uses get_constitution_prompt + MultiplePromptBuilder.
    """
    all_temps = _all_temperatures(n_samples, sc_temperatures)
    requests = []

    if prompt_builder is not None:
        # Custom builder (e.g. forced search path with monkey-patched build())
        for row_idx in tqdm(range(len(df)), desc="building requests"):
            polity, name, start_year, end_year = _row_fields(df.iloc[row_idx])
            prompts = prompt_builder.build(polity, name, start_year, end_year)
            for prompt in prompts:
                if len(prompt.indicators) != 1:
                    continue  # batch only supports multiple mode (one indicator per prompt)
                indicator = prompt.indicators[0]
                for sc_idx, temp in enumerate(all_temps):
                    cid = f"{row_idx}|{indicator}|{sc_idx}"
                    requests.append(_make_req(cid, prompt.system_prompt, prompt.user_prompt, temp, max_tokens, n_samples))
        return requests

    # Standard path: use default builders
    from prompts.constitution import get_prompt as get_constitution_prompt
    from prompts.multiple_builder import MultiplePromptBuilder

    constitution_in = "constitution" in indicators
    other_indicators = [i for i in indicators if i != "constitution"]

    if constitution_in:
        for row_idx in tqdm(range(len(df)), desc="constitution"):
            polity, name, start_year, end_year = _row_fields(df.iloc[row_idx])
            sys_p, usr_p = get_constitution_prompt(polity, name, start_year, end_year)
            for sc_idx, temp in enumerate(all_temps):
                cid = f"{row_idx}|constitution|{sc_idx}"
                requests.append(_make_req(cid, sys_p, usr_p, temp, max_tokens, n_samples))

    if other_indicators:
        builder = MultiplePromptBuilder(indicators=other_indicators, reasoning=True)
        for row_idx in tqdm(range(len(df)), desc="indicators"):
            polity, name, start_year, end_year = _row_fields(df.iloc[row_idx])
            prompts = builder.build(polity, name, start_year, end_year)
            for prompt in prompts:
                indicator = prompt.indicators[0]
                for sc_idx, temp in enumerate(all_temps):
                    cid = f"{row_idx}|{indicator}|{sc_idx}"
                    requests.append(_make_req(cid, prompt.system_prompt, prompt.user_prompt, temp, max_tokens, n_samples))

    return requests


# ---------------------------------------------------------------------------
# Gemini submission and polling
# ---------------------------------------------------------------------------

def _get_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def _extract_text(response) -> str:
    """Extract text from a Gemini GenerateContentResponse."""
    if hasattr(response, "text"):
        try:
            return response.text
        except (ValueError, AttributeError):
            pass
    if hasattr(response, "candidates"):
        try:
            return response.candidates[0].content.parts[0].text
        except (IndexError, AttributeError):
            pass
    return str(response)


def _submit_and_wait(
    client,
    model: str,
    requests: list[dict],
    display_name: str,
    poll_interval: int = 30,
) -> dict[str, str]:
    """Submit one batch job and wait for completion. Returns custom_id → response_text.

    Requests must include metadata.custom_id for ID matching.
    Falls back to positional index if metadata is not echoed by Gemini.
    """
    # Strip metadata from what we send; keep it in our copy for positional fallback
    api_requests = [
        {"contents": r["contents"], "config": r["config"], "metadata": r.get("metadata")}
        for r in requests
    ]
    our_cids = [r["metadata"]["custom_id"] for r in requests]

    print(f"  Submitting {len(requests)} requests as one batch job ...")
    job = client.batches.create(
        model=model,
        src=api_requests,
        config={"display_name": display_name},
    )
    print(f"  Job: {job.name}")

    while True:
        job = client.batches.get(name=job.name)
        state = str(getattr(job.state, "name", str(job.state))).upper()
        if any(k in state for k in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")):
            break
        print(f"  State: {state} — polling again in {poll_interval}s ...")
        time.sleep(poll_interval)

    print(f"  Final state: {state}")
    results: dict[str, str] = {}

    if "SUCCEEDED" not in state:
        print(f"  WARNING: job ended in {state}; no results collected.")
        return results

    inlined = getattr(getattr(job, "dest", None), "inlined_responses", None)
    if inlined is None:
        print("  WARNING: no inlined_responses in job.dest.")
        return results

    n_failed = 0
    for i, inline_resp in enumerate(inlined):
        if i >= len(our_cids):
            break

        # Prefer metadata echo; fall back to positional match
        meta = getattr(inline_resp, "metadata", None)
        cid = meta["custom_id"] if (meta and "custom_id" in meta) else our_cids[i]

        if getattr(inline_resp, "error", None):
            n_failed += 1
            results[cid] = ""
            continue

        resp = inline_resp.response
        results[cid] = _extract_text(resp) if resp else ""

    print(f"  Collected {len(results)} responses ({n_failed} failed requests)")
    return results


# ---------------------------------------------------------------------------
# Response parsing per indicator type
# ---------------------------------------------------------------------------

def _parse_constitution(response_text: str) -> tuple[str, str, Optional[int], dict]:
    """Returns (prediction_str, reasoning, confidence, extra_fields)."""
    parsed = parse_json_response(response_text, verbose=False)
    v = validate_constitution_response(parsed)
    pred = v.get("constitution")
    pred_str = str(int(float(pred))) if pred is not None else ""
    extra = {
        "constitution_document_name": v.get("document_name"),
        "constitution_year": v.get("constitution_year"),
        "constitution_document_types": v.get("document_types"),
    }
    return pred_str, v.get("reasoning", ""), v.get("confidence_score"), extra


def _parse_indicator(response_text: str, indicator: str) -> tuple[str, str, Optional[int]]:
    """Returns (prediction_str, reasoning, confidence)."""
    valid_labels = [str(l) for l in INDICATOR_LABELS.get(indicator, ["0", "1"])]
    parsed = parse_json_response(response_text, verbose=False)
    v = validate_indicator_response(parsed, indicator, valid_labels)
    raw_pred = v.get(indicator)
    pred_str = _normalize_pred(raw_pred, indicator) if raw_pred is not None else ""
    return pred_str, v.get("reasoning", ""), v.get("confidence_score")


def _parse_elections(response_text: str) -> tuple[str, str, Optional[int]]:
    """Returns (prediction_str, reasoning, confidence)."""
    parsed = parse_json_response(response_text, verbose=False)
    pred = str(parsed.get("elections", "0"))
    if pred not in ("0", "1", "2"):
        pred = "0"
    reasoning = parsed.get("reasoning", parsed.get("elections_reasoning", ""))
    confidence = parsed.get("confidence_score", parsed.get("elections_confidence"))
    return pred, reasoning, confidence


# ---------------------------------------------------------------------------
# SC aggregation and DataFrame merge
# ---------------------------------------------------------------------------

def _aggregate_and_merge(
    raw_results: dict[str, str],
    df: pd.DataFrame,
    n_samples: int,
) -> pd.DataFrame:
    """Parse all responses, aggregate SC votes, merge into df."""
    n_rows = len(df)

    # Group by (row_idx, indicator) → {sc_idx: response_text}
    grouped: dict[tuple[int, str], dict[int, str]] = defaultdict(dict)
    for cid, resp_text in raw_results.items():
        try:
            row_idx, indicator, sc_idx = _parse_custom_id(cid)
            grouped[(row_idx, indicator)][sc_idx] = resp_text
        except Exception as e:
            print(f"  WARNING: cannot parse custom_id {cid!r}: {e}")

    row_updates: dict[int, dict] = defaultdict(dict)

    for (row_idx, indicator), sc_map in tqdm(grouped.items(), desc="aggregating SC"):
        votes: list[str] = []
        initial_reasoning = ""
        initial_confidence = None
        extra_fields: dict = {}

        for sc_idx in sorted(sc_map.keys()):
            resp_text = sc_map[sc_idx]
            if not resp_text:
                continue

            if indicator == "constitution":
                pred_str, reasoning, confidence, extra = _parse_constitution(resp_text)
                if sc_idx == 0:
                    initial_reasoning = reasoning
                    initial_confidence = confidence
                    extra_fields = extra
            elif indicator == "elections":
                pred_str, reasoning, confidence = _parse_elections(resp_text)
                if sc_idx == 0:
                    initial_reasoning = reasoning
                    initial_confidence = confidence
            else:
                pred_str, reasoning, confidence = _parse_indicator(resp_text, indicator)
                if sc_idx == 0:
                    initial_reasoning = reasoning
                    initial_confidence = confidence

            if pred_str:
                votes.append(pred_str)

        initial_pred = votes[0] if votes else None
        initial_stored = _denormalize_pred(initial_pred, indicator)

        updates = row_updates[row_idx]
        updates[f"{indicator}_prediction"] = initial_stored
        updates[f"{indicator}_reasoning"] = initial_reasoning
        updates[f"{indicator}_confidence"] = initial_confidence
        updates.update(extra_fields)

        if n_samples > 0:
            final_pred_str, agreement, uncertainty = _aggregate_sc(votes, indicator)
            final_pred = _denormalize_pred(final_pred_str, indicator)
            updates[f"{indicator}_verified"] = final_pred
            updates[f"{indicator}_agreement"] = round(agreement, 3)
            updates[f"{indicator}_uncertainty"] = uncertainty

    # Merge into result DataFrame (df already reset_index'd before this call)
    result_df = df.copy()
    for row_idx, cols in row_updates.items():
        if row_idx >= n_rows:
            continue
        for col, val in cols.items():
            if col not in result_df.columns:
                result_df[col] = None
            result_df.iloc[row_idx, result_df.columns.get_loc(col)] = val

    return result_df


# ---------------------------------------------------------------------------
# Public entry point — called by main.py
# ---------------------------------------------------------------------------

def run_inline_batch(
    df: pd.DataFrame,
    indicators: list[str],
    model: str,
    api_key: str,
    n_samples: int,
    output_path: str,
    prompt_builder=None,
    sc_temperatures: list[float] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    poll_interval: int = 30,
) -> pd.DataFrame:
    """Build prompts in memory, submit as one Gemini batch job, aggregate SC.

    Called by main.py when --use-batch is set. Replaces GeminiBatchRunner.

    Args:
        df:              Input DataFrame (already filtered/sliced by main.py).
        indicators:      Which indicators to predict.
        model:           Gemini model identifier.
        api_key:         GEMINI_API_KEY.
        n_samples:       Additional SC calls. Total votes = n_samples + 1.
                         n_samples=0 → single call, no SC columns.
        output_path:     Path for CSV + JSON output (written here AND returned).
        prompt_builder:  Optional custom prompt builder (for --search-mode forced).
                         If None, uses get_constitution_prompt + MultiplePromptBuilder.
        sc_temperatures: Temperature list for SC samples (default: all 1.0).
        max_tokens:      Max output tokens per request.
        poll_interval:   Seconds between batch job status polls.

    Returns:
        Enriched DataFrame with prediction/reasoning/confidence columns,
        and optionally _verified/_agreement/_uncertainty columns when n_samples > 0.
    """
    df = df.reset_index(drop=True)
    client = _get_client(api_key)

    # Build requests
    total_per_group = n_samples + 1
    all_temps = _all_temperatures(n_samples, sc_temperatures)
    print(
        f"\n[Batch] Building requests: {len(indicators)} indicators × {len(df)} rows × "
        f"{total_per_group} SC calls (temps={all_temps}) = "
        f"~{len(indicators) * len(df) * total_per_group} total"
    )
    requests = _build_requests_in_memory(
        df=df,
        indicators=indicators,
        n_samples=n_samples,
        max_tokens=max_tokens,
        sc_temperatures=sc_temperatures,
        prompt_builder=prompt_builder,
    )
    print(f"[Batch] Built {len(requests)} requests")

    # Submit and wait
    display_name = f"const-llm-{Path(output_path).stem}-{int(time.time())}"
    raw_results = _submit_and_wait(client, model, requests, display_name, poll_interval)

    # Report missing rows
    expected_cids = {r["metadata"]["custom_id"] for r in requests}
    missing_cids = expected_cids - set(raw_results.keys())
    empty_cids = {cid for cid, text in raw_results.items() if not text}
    if missing_cids or empty_cids:
        print(f"\n[Batch] Missing/failed responses: {len(missing_cids | empty_cids)}")
        failed_rows = sorted({_parse_custom_id(cid)[0] for cid in missing_cids | empty_cids})
        print(f"  Failed row indices: {failed_rows[:20]}{'...' if len(failed_rows) > 20 else ''}")
        print("  Re-run these rows by filtering the input and rebuilding the JSONL.")

    # Aggregate SC and merge
    result_df = _aggregate_and_merge(raw_results, df, n_samples)

    # Write outputs
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    json_path = str(output_path).replace(".csv", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Batch] Saved: {output_path}")
    print(f"[Batch] Saved: {json_path}")

    return result_df


# ---------------------------------------------------------------------------
# Standalone entry point — reads pre-built JSONL
# ---------------------------------------------------------------------------

def run_from_jsonl(
    jsonl_path: str,
    input_path: str,
    output_path: str,
    model: str,
    api_key: str,
    n_samples: int,
    poll_interval: int = 30,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> pd.DataFrame:
    """Read a pre-built JSONL, submit as one job, aggregate SC, write output."""
    print(f"Loading input: {input_path}")
    df = load_dataframe(input_path).reset_index(drop=True)

    print(f"Loading requests: {jsonl_path}")
    requests: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                requests.append(json.loads(line))

    if not requests:
        raise ValueError(f"JSONL file is empty: {jsonl_path}")

    # Verify n_samples matches what was embedded at build time
    first_meta = requests[0].get("metadata", {})
    embedded_n = first_meta.get("n_samples")
    if embedded_n is not None and int(embedded_n) != n_samples:
        raise ValueError(
            f"--n-samples {n_samples} does not match n_samples={embedded_n} "
            f"embedded in {jsonl_path}. Rebuild or pass the correct --n-samples."
        )

    total_per_group = n_samples + 1
    print(f"  {len(requests)} requests ({len(requests) // total_per_group} groups × {total_per_group})")

    client = _get_client(api_key)
    display_name = f"const-llm-{Path(output_path).stem}-{int(time.time())}"
    raw_results = _submit_and_wait(client, model, requests, display_name, poll_interval)

    # Report failures
    expected_cids = {r["metadata"]["custom_id"] for r in requests}
    missing = expected_cids - set(raw_results.keys())
    empty = {cid for cid, text in raw_results.items() if not text}
    if missing or empty:
        print(f"\nMissing/failed: {len(missing | empty)} responses")
        failed_rows = sorted({_parse_custom_id(cid)[0] for cid in missing | empty})
        print(f"  Failed row indices: {failed_rows[:20]}{'...' if len(failed_rows) > 20 else ''}")

    result_df = _aggregate_and_merge(raw_results, df, n_samples)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    json_path = str(output_path).replace(".csv", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {output_path}")
    print(f"Saved: {json_path}")

    return result_df


# ---------------------------------------------------------------------------
# CLI (standalone use)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Submit a pre-built JSONL to Gemini Batch API (single job, no checkpoints).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--jsonl", required=True, help="Path to JSONL built by src/build_batch_jsonl.py")
    parser.add_argument("--input", "-i", required=True, help="Original input CSV/JSONL (for metadata).")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path.")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument(
        "--n-samples", type=int, default=0,
        help=(
            "Additional SC samples. Must match the value used during build. "
            "n_samples=0 → no SC. n_samples=2 → 3 total votes."
        ),
    )
    parser.add_argument("--poll-interval", type=int, default=30)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GEMINI_API_KEY environment variable is not set.")

    run_from_jsonl(
        jsonl_path=args.jsonl,
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        api_key=api_key,
        n_samples=args.n_samples,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
