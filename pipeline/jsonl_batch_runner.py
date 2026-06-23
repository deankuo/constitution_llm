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

Custom ID format:
    constitution/elections:  "{row_idx}|{indicator}|{sc_idx}"
    single-mode indicators:  "{row_idx}|single|{sc_idx}"  (one request covers ALL indicators)
    indicators list is embedded in metadata["indicators"] for _aggregate_and_merge

Usage (standalone)
------------------
python pipeline/jsonl_batch_runner.py \\
    --input data/temp/batch_constitution.jsonl \\
    --dataset data/plt_leaders_data.csv \\
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

BATCH_DISCOUNT = 0.6
MAX_BATCH_BYTES = 1_900 * 1024 * 1024  # 1.9 GB — safely under Gemini's 2 GB per-job limit


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
# Request chunking (for Gemini's 2 GB per-job limit)
# ---------------------------------------------------------------------------

def _chunk_requests(
    requests: list[dict],
    max_bytes: int = MAX_BATCH_BYTES,
) -> list[list[dict]]:
    """Split requests into chunks whose serialized size stays under max_bytes.

    Returns a list of sub-lists. If all requests fit in one chunk, returns [[...all...]].
    """
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_size = 0
    for req in requests:
        lb = len(json.dumps(req, ensure_ascii=False).encode("utf-8")) + 1  # +1 for newline
        if current and current_size + lb > max_bytes:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(req)
        current_size += lb
    if current:
        chunks.append(current)
    return chunks or [[]]


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
    reasoning: bool = True,
    prompt_version: str = "v1",
) -> list[dict]:
    """Build all batch requests in memory.

    If prompt_builder is provided (e.g. forced search path), it is used as-is.
      - Single-mode prompts (len(indicators) > 1): custom_id "{row_idx}|single|{sc_idx}",
        indicators list embedded in metadata["indicators"].
      - Multiple-mode prompts (len(indicators) == 1): custom_id "{row_idx}|{indicator}|{sc_idx}".

    Standard path (no prompt_builder):
      - constitution → get_constitution_prompt, custom_id "{row_idx}|constitution|{sc_idx}"
      - other indicators → SinglePromptBuilder (all indicators in one prompt per row),
        custom_id "{row_idx}|single|{sc_idx}", metadata["indicators"] set.
    """
    all_temps = _all_temperatures(n_samples, sc_temperatures)
    requests = []

    if prompt_builder is not None:
        # Custom builder (e.g. forced search path)
        for row_idx in tqdm(range(len(df)), desc="building requests"):
            polity, name, start_year, end_year = _row_fields(df.iloc[row_idx])
            prompts = prompt_builder.build(polity, name, start_year, end_year)
            for prompt in prompts:
                if len(prompt.indicators) == 1:
                    # Multiple mode: one indicator per prompt
                    indicator = prompt.indicators[0]
                    for sc_idx, temp in enumerate(all_temps):
                        cid = f"{row_idx}|{indicator}|{sc_idx}"
                        requests.append(_make_req(cid, prompt.system_prompt, prompt.user_prompt, temp, max_tokens, n_samples))
                else:
                    # Single mode: all indicators in one prompt
                    indicators_json = json.dumps(prompt.indicators)
                    for sc_idx, temp in enumerate(all_temps):
                        cid = f"{row_idx}|single|{sc_idx}"
                        req = _make_req(cid, prompt.system_prompt, prompt.user_prompt, temp, max_tokens, n_samples)
                        req["metadata"]["indicators"] = indicators_json
                        requests.append(req)
        return requests

    # Standard path: constitution → own prompt; others → SinglePromptBuilder (version selectable)
    from prompts.constitution import get_prompt as get_constitution_prompt

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
        from prompts.single_builder import SinglePromptBuilder, SinglePromptBuilderV2, SinglePromptBuilderV3
        _BUILDER_CLS = {"v1": SinglePromptBuilder, "v2": SinglePromptBuilderV2, "v3": SinglePromptBuilderV3}
        BuilderCls = _BUILDER_CLS.get(prompt_version, SinglePromptBuilder)
        builder = BuilderCls(indicators=other_indicators, reasoning=reasoning)
        indicators_json = json.dumps(other_indicators)
        for row_idx in tqdm(range(len(df)), desc="indicators"):
            polity, name, start_year, end_year = _row_fields(df.iloc[row_idx])
            prompt = builder.build(polity, name, start_year, end_year)[0]
            for sc_idx, temp in enumerate(all_temps):
                cid = f"{row_idx}|single|{sc_idx}"
                req = _make_req(cid, prompt.system_prompt, prompt.user_prompt, temp, max_tokens, n_samples)
                req["metadata"]["indicators"] = indicators_json
                requests.append(req)

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
    request_metadata: Optional[dict[str, dict]] = None,
) -> pd.DataFrame:
    """Parse all responses, aggregate SC votes, merge into df.

    For single-mode responses (indicator == "single"), the same full response
    text is re-used for each indicator extracted from metadata["indicators"].
    validate_indicator_response already handles {indicator}_reasoning /
    {indicator}_confidence keys emitted by SinglePromptBuilder.
    """
    n_rows = len(df)

    # Expand single-mode responses: one combined response → N per-indicator entries.
    # Each indicator gets the same full JSON text; _parse_indicator extracts only its keys.
    expanded: dict[str, str] = {}
    for cid, resp_text in raw_results.items():
        try:
            row_idx, indicator, sc_idx = _parse_custom_id(cid)
        except Exception as e:
            print(f"  WARNING: cannot parse custom_id {cid!r}: {e}")
            continue
        if indicator == "single":
            meta = (request_metadata or {}).get(cid, {})
            inds = json.loads(meta.get("indicators", "[]"))
            for ind in inds:
                expanded[f"{row_idx}|{ind}|{sc_idx}"] = resp_text
        else:
            expanded[cid] = resp_text

    # Group by (row_idx, indicator) → {sc_idx: response_text}
    grouped: dict[tuple[int, str], dict[int, str]] = defaultdict(dict)
    for cid, resp_text in expanded.items():
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
    reasoning: bool = True,
    prompt_version: str = "v1",
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
        f"\n[Batch] Building requests: {len(indicators)} indicators over {len(df)} rows "
        f"({total_per_group} SC calls each, temps={all_temps}) ..."
    )
    requests = _build_requests_in_memory(
        df=df,
        indicators=indicators,
        n_samples=n_samples,
        max_tokens=max_tokens,
        sc_temperatures=sc_temperatures,
        prompt_builder=prompt_builder,
        reasoning=reasoning,
        prompt_version=prompt_version,
    )
    print(f"[Batch] Built {len(requests)} requests")

    # Build metadata lookup for _aggregate_and_merge (needed for single-mode expansion)
    request_metadata = {r["metadata"]["custom_id"]: r["metadata"] for r in requests}

    # Chunk requests to stay under Gemini's 2 GB per-job limit, then submit sequentially
    chunks = _chunk_requests(requests)
    display_base = f"const-llm-{Path(output_path).stem}-{int(time.time())}"
    raw_results: dict[str, str] = {}
    for i, chunk in enumerate(chunks):
        chunk_label = f"chunk {i + 1}/{len(chunks)}" if len(chunks) > 1 else "single job"
        display_name = f"{display_base}-c{i + 1:03d}" if len(chunks) > 1 else display_base
        size_mb = sum(len(json.dumps(r, ensure_ascii=False).encode()) + 1 for r in chunk) / 1024 / 1024
        print(f"\n[Batch] Submitting {chunk_label}: {len(chunk)} requests ({size_mb:.1f} MB) ...")
        chunk_results = _submit_and_wait(client, model, chunk, display_name, poll_interval)
        raw_results.update(chunk_results)

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
    result_df = _aggregate_and_merge(raw_results, df, n_samples, request_metadata)

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
    jsonl_path,  # str or list[str] — one file or multiple chunk files
    output_path: str,
    model: str,
    api_key: str,
    n_samples: int,
    input_path: Optional[str] = None,
    poll_interval: int = 30,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> pd.DataFrame:
    """Read one or more pre-built JSONL chunk files, submit each as its own batch job,
    aggregate SC across all results, and write output.

    jsonl_path can be a single file path or a list of paths (for chunked builds).

    input_path is optional. When provided, predictions are merged into the full
    original DataFrame (all original columns preserved). When omitted, a minimal
    DataFrame is constructed from the JSONL row indices — the output will contain
    only prediction/confidence/reasoning columns keyed by positional row index.
    """
    if input_path is not None:
        print(f"Loading input: {input_path}")
        df = load_dataframe(input_path).reset_index(drop=True)
    else:
        df = None  # will be built after loading requests

    jsonl_paths = [jsonl_path] if isinstance(jsonl_path, str) else list(jsonl_path)

    all_requests: list[dict] = []
    for jpath in jsonl_paths:
        print(f"Loading requests: {jpath}")
        with open(jpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_requests.append(json.loads(line))

    if not all_requests:
        raise ValueError(f"No requests found in: {jsonl_paths}")

    # Verify n_samples matches what was embedded at build time
    first_meta = all_requests[0].get("metadata", {})
    embedded_n = first_meta.get("n_samples")
    if embedded_n is not None and int(embedded_n) != n_samples:
        raise ValueError(
            f"--n-samples {n_samples} does not match n_samples={embedded_n} "
            f"embedded in JSONL. Rebuild or pass the correct --n-samples."
        )

    total_per_group = n_samples + 1
    print(f"  {len(all_requests)} total requests ({len(all_requests) // total_per_group} groups × {total_per_group})")

    # Build DataFrame from JSONL when --dataset is not provided.
    # sc_idx=0 requests carry row_data embedded by build_batch_jsonl.py; use those to
    # reconstruct the full original dataset (all columns, correct dtypes from str conversion).
    # Falls back to a minimal row_idx-only DataFrame for older JSONL files without row_data.
    if df is None:
        row_data_by_idx: dict[int, dict] = {}
        for r in all_requests:
            meta = r.get("metadata", {})
            cid = meta.get("custom_id", "")
            try:
                ri, _, si = _parse_custom_id(cid)
                if si == 0 and "row_data" in meta:
                    row_data_by_idx[ri] = json.loads(meta["row_data"])
            except Exception:
                pass

        if row_data_by_idx:
            max_idx = max(row_data_by_idx.keys())
            rows = [row_data_by_idx.get(i, {}) for i in range(max_idx + 1)]
            df = pd.DataFrame(rows)
            print(f"  Reconstructed {len(df)} rows from JSONL row_data ({len(df.columns)} columns)")
        else:
            max_row_idx = max(
                int(r["metadata"]["custom_id"].split("|")[0])
                for r in all_requests
                if "metadata" in r and "custom_id" in r["metadata"]
            )
            df = pd.DataFrame({"row_idx": range(max_row_idx + 1)})
            print(f"  No row_data in JSONL; output will contain predictions only ({max_row_idx + 1} rows)")

    # Build metadata lookup for _aggregate_and_merge (needed for single-mode expansion)
    request_metadata = {r["metadata"]["custom_id"]: r["metadata"] for r in all_requests}

    # Chunk by size and submit each chunk as a separate batch job
    chunks = _chunk_requests(all_requests)
    client = _get_client(api_key)
    display_base = f"const-llm-{Path(output_path).stem}-{int(time.time())}"
    raw_results: dict[str, str] = {}
    for i, chunk in enumerate(chunks):
        chunk_label = f"chunk {i + 1}/{len(chunks)}" if len(chunks) > 1 else "single job"
        display_name = f"{display_base}-c{i + 1:03d}" if len(chunks) > 1 else display_base
        size_mb = sum(len(json.dumps(r, ensure_ascii=False).encode()) + 1 for r in chunk) / 1024 / 1024
        print(f"\nSubmitting {chunk_label}: {len(chunk)} requests ({size_mb:.1f} MB) ...")
        chunk_results = _submit_and_wait(client, model, chunk, display_name, poll_interval)
        raw_results.update(chunk_results)

    result_df = _aggregate_and_merge(raw_results, df, n_samples, request_metadata)

    # Response-level failures (CID missing from results or response text empty)
    expected_cids = {r["metadata"]["custom_id"] for r in all_requests}
    missing = expected_cids - set(raw_results.keys())
    empty = {cid for cid, text in raw_results.items() if not text}
    response_failed_rows = sorted({_parse_custom_id(cid)[0] for cid in missing | empty})

    # Prediction-level nulls: response received but truncated/unparseable for some indicators.
    # Only check indicators actually requested in this batch — not columns inherited from a prior
    # merged run or indicators from a different task (e.g. constitution columns when running
    # indicators task). Requested indicators are in metadata["indicators"] (single mode) or
    # directly in the custom_id indicator slot (constitution/elections).
    requested_indicators: set[str] = set()
    for meta in request_metadata.values():
        if "indicators" in meta:
            try:
                requested_indicators.update(json.loads(meta["indicators"]))
            except (json.JSONDecodeError, TypeError):
                pass
        cid = meta.get("custom_id", "")
        try:
            _, ind, _ = _parse_custom_id(cid)
            if ind not in ("single",):
                requested_indicators.add(ind)
        except Exception:
            pass

    pred_cols = [
        f"{ind}_prediction" for ind in requested_indicators
        if f"{ind}_prediction" in result_df.columns
    ]
    null_row_idxs: set[int] = set()
    if pred_cols:
        null_row_idxs = set(result_df.index[result_df[pred_cols].isnull().any(axis=1)].tolist())

    all_failed_rows = sorted(null_row_idxs | set(response_failed_rows))
    retry_path = None

    if all_failed_rows:
        print(f"\nFailed rows: {len(all_failed_rows)} total")
        if response_failed_rows:
            print(f"  Response-level (missing/empty): {response_failed_rows[:10]}"
                  f"{'...' if len(response_failed_rows) > 10 else ''}")
        extra_null = sorted(null_row_idxs - set(response_failed_rows))
        if extra_null:
            print(f"  Prediction nulls (truncated/parse failure): {extra_null[:10]}"
                  f"{'...' if len(extra_null) > 10 else ''}")
        # Write all SC calls for failed rows so they can be resubmitted as a new batch job.
        # Resubmitting from the same JSONL is reproducibility-safe (same prompts, same sampling
        # distribution). Do NOT fall back to sync calls — they use a different serving path.
        failed_row_set = set(all_failed_rows)
        failed_requests = [
            r for r in all_requests
            if _parse_custom_id(r["metadata"]["custom_id"])[0] in failed_row_set
        ]
        retry_path = Path(output_path).parent / (Path(output_path).stem + "_failed_requests.jsonl")
        with open(retry_path, "w", encoding="utf-8") as f:
            for r in failed_requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Retry JSONL: {retry_path} ({len(all_failed_rows)} rows, {len(failed_requests)} requests)")
        print(f"  Re-run: python pipeline/jsonl_batch_runner.py --input {retry_path} --output ...")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    json_path = str(output_path).replace(".csv", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {output_path}")
    print(f"Saved: {json_path}")

    # Provenance record: captures what the JSONL cannot (model, job names, run time, failures)
    import datetime as _dt
    prov = {
        "model": model,
        "run_timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "n_samples": n_samples,
        "input_jsonl": jsonl_paths,
        "total_requests": len(all_requests),
        "success_count": sum(1 for v in raw_results.values() if v),
        "response_failed_cids": sorted(missing | empty),
        "prediction_null_rows": sorted(null_row_idxs),
        "all_failed_rows": all_failed_rows,
        "retry_jsonl": str(retry_path) if retry_path else None,
    }
    prov_path = Path(output_path).parent / (Path(output_path).stem + "_provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print(f"Saved: {prov_path}")

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
    parser.add_argument(
        "--input", required=True, nargs="+",
        help=(
            "Path(s) to JSONL built by src/build_batch_jsonl.py. "
            "Pass multiple files when the build was split into chunks "
            "(e.g. --input data/temp/batch_chunk001.jsonl data/temp/batch_chunk002.jsonl). "
            "Each chunk is submitted as a separate Gemini batch job."
        ),
    )
    parser.add_argument(
        "--dataset", default=None,
        help=(
            "Original input CSV/JSONL dataset (optional). "
            "When provided, predictions are merged into the full original DataFrame "
            "(all original columns are preserved in the output). "
            "When omitted, a minimal DataFrame is built from the JSONL row indices — "
            "the output contains only prediction/confidence/reasoning columns."
        ),
    )
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
        jsonl_path=args.input if len(args.input) > 1 else args.input[0],
        output_path=args.output,
        model=args.model,
        api_key=api_key,
        n_samples=args.n_samples,
        input_path=args.dataset,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
