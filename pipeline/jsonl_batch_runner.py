#!/usr/bin/env python3
"""
Gemini Batch API runner — single-job, SC-embedded approach.

Entry point
-----------
run_from_jsonl(jsonl_path, ...)
    Reads a pre-built JSONL (from src/build_batch_jsonl.py), submits it as
    one batch job per chunk, parses SC, writes CSV + JSON. This is the ONLY
    batch path — main.py runs synchronously; batch runs always go through
    build_batch_jsonl.py + this runner.

Self-consistency convention — matches main.py / SelfConsistencyConfig
----------------------------------------------------------------------
    n_samples = number of ADDITIONAL SC calls (not counting the initial)
    Total requests per (row, indicator) = n_samples + 1
    sc_idx=0  → initial prediction (temp=1.0)
    sc_idx=1..n_samples → SC samples (temp=1.0 each, default)
    Total votes in majority = n_samples + 1

    n_samples=0 → no SC (single call), no _prediction (majority)/_agreement/_uncertainty columns
    n_samples=2 → 3 votes, write _prediction (majority vote)/_agreement/_uncertainty

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

Retrying failed rows (large runs, e.g. 13k+ rows)
--------------------------------------------------
run_from_jsonl detects two kinds of per-row failure and reports both:
  - response-level: request missing from the batch output, or response text empty
  - prediction-level: response received but a requested indicator's _prediction
    is still null after parsing (truncated/malformed JSON)
It writes {output_stem}_failed_requests.jsonl (ready to resubmit) and
{output_stem}_provenance.json (model, timestamps, exact failed row list).

To retry and merge back into the SAME combined dataset:
    python pipeline/jsonl_batch_runner.py \\
        --input data/results/exp001_failed_requests.jsonl \\
        --dataset data/results/exp001.csv \\
        --output data/results/exp001.csv \\
        --model gemini-3.1-pro-preview --n-samples 2

--dataset MUST be the prior run's --output (not the original raw input) —
_aggregate_and_merge only overwrites rows present in --input, so every
already-successful row is carried through untouched. Repeat until
_failed_requests.jsonl is no longer produced.

Recovering from a killed local process (job already submitted)
-----------------------------------------------------------------
Gemini batch jobs run server-side and keep going even if the local runner is
killed (e.g. a Jupyter kernel interrupt). If a job was already submitted,
don't resubmit — attach to it by name and just poll/download:
    python pipeline/jsonl_batch_runner.py \\
        --input data/temp/batch_chunk001.jsonl data/temp/batch_chunk002.jsonl \\
        --dataset data/plt_leaders_data.csv \\
        --output data/results/exp001.csv \\
        --attach-jobs batches/abc123 batches/def456
--attach-jobs takes one job name per --input chunk, in submission order. Find
job names/states with `client.batches.list()` (google-genai SDK) if the
notebook output with the printed "Job: ..." line was lost.
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

    When all differ, fall back to votes[0] (the sc_idx=0 initial prediction).
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
        winner = votes[0]  # fall back to sc_idx=0 (initial prediction)

    return winner, agreement, uncertainty


def _denormalize_pred(pred: Optional[str], indicator: str):
    """Convert a normalized prediction string back to the storage format."""
    if pred is None:
        return None
    return pred


def _ids_equal(a, b) -> bool:
    """Compare two id values across CSV round-trip dtype changes (123 vs '123' vs 123.0)."""
    if str(a) == str(b):
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Gemini submission and polling
# ---------------------------------------------------------------------------

def _get_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def _extract_text(response: dict) -> str:
    """Extract text from a raw (camelCase) GenerateContentResponse dict.

    Operates on the raw dict downloaded from a file-based batch job's output
    file, not an SDK-typed object — pydantic's strict GenerateContentResponse
    model rejects newly-added API fields it doesn't know about yet (e.g.
    usageMetadata.serviceTier), so we deliberately avoid model_validate() here.
    """
    try:
        return response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


def _extract_grounding(response: dict) -> tuple:
    """Extract (queries_str, urls_str) from a raw (camelCase) response dict.

    Returns (None, None) when grounding was not used or metadata is absent.
    """
    try:
        g_meta = response["candidates"][0].get("groundingMetadata")
        if not g_meta:
            return None, None
        queries = g_meta.get("webSearchQueries") or []
        queries_str = " | ".join(queries) if queries else None
        chunks = g_meta.get("groundingChunks") or []
        urls = []
        for chunk in chunks:
            web = chunk.get("web")
            if web:
                title = web.get("title", "") or ""
                uri = web.get("uri", "") or ""
                entry = f"{title} ({uri})" if title else uri
                if entry:
                    urls.append(entry)
        urls_str = " | ".join(urls) if urls else None
        return queries_str or None, urls_str or None
    except (KeyError, IndexError, TypeError):
        return None, None


def _requests_use_grounding(requests: list) -> bool:
    """Detect if batch requests include google_search or google_search_retrieval."""
    if not requests:
        return False
    tools = requests[0].get("request", {}).get("tools", [])
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and (
                "google_search" in tool or "google_search_retrieval" in tool
            ):
                return True
    return False


def _poll_and_download(
    client,
    job,
    our_cids: list[str],
    poll_interval: int = 30,
    grounding_data: Optional[dict] = None,
) -> dict[str, str]:
    """Poll a batch job (already created or attached-to) until it finishes, then
    download and parse its output file. Returns custom_id ("key") → response_text.

    The output file is JSONL where each line is either
    {"key", "metadata"?, "response": <GenerateContentResponse>} or
    {"key", "error": {"code", "message"}} for individual request failures.
    """
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

    dest = getattr(job, "dest", None)
    if not dest or not dest.file_name:
        print("  WARNING: no output file in job.dest.")
        return results

    raw = client.files.download(file=dest.file_name).decode("utf-8")
    n_failed = 0
    for i, line in enumerate(raw.strip().split("\n")):
        if not line:
            continue
        record = json.loads(line)
        cid = record.get("key", our_cids[i] if i < len(our_cids) else None)
        if cid is None:
            continue

        if "error" in record:
            n_failed += 1
            results[cid] = ""
            continue

        resp_dict = record.get("response")
        if not resp_dict:
            results[cid] = ""
            continue

        results[cid] = _extract_text(resp_dict)
        if grounding_data is not None:
            q, u = _extract_grounding(resp_dict)
            if q or u:
                grounding_data[cid] = (q, u)

    print(f"  Collected {len(results)} responses ({n_failed} failed requests)")
    return results


def _submit_and_wait(
    client,
    model: str,
    requests: list[dict],
    display_name: str,
    poll_interval: int = 30,
    grounding_data: Optional[dict] = None,
) -> dict[str, str]:
    """Upload requests as a JSONL file, submit one batch job, wait for completion.

    Returns custom_id ("key") → response_text.

    Uses the file-upload path (client.files.upload + src=<file name>) rather
    than inline submission: inline requests are capped at 20MB total request
    size, while file uploads support up to 2GB — see
    https://ai.google.dev/gemini-api/docs/batch-api. Each request must already
    be in {"key", "request", "metadata"} shape (see src/build_batch_jsonl.py).
    """
    import tempfile

    our_cids = [r["key"] for r in requests]

    tmp_path = None
    uploaded_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
            for r in requests:
                tmp.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp_path = tmp.name

        print(f"  Uploading {len(requests)} requests as a JSONL file ...")
        uploaded = client.files.upload(file=tmp_path, config={"mime_type": "jsonl"})
        uploaded_name = uploaded.name

        job = client.batches.create(
            model=model,
            src=uploaded_name,
            config={"display_name": display_name},
        )
        print(f"  Job: {job.name}")

        return _poll_and_download(client, job, our_cids, poll_interval, grounding_data)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if uploaded_name:
            try:
                client.files.delete(name=uploaded_name)
            except Exception:
                pass


def _attach_and_wait(
    client,
    job_name: str,
    requests: list[dict],
    poll_interval: int = 30,
    grounding_data: Optional[dict] = None,
) -> dict[str, str]:
    """Attach to an ALREADY-SUBMITTED batch job by name and collect its results.

    Recovery path for when the local process (e.g. a Jupyter kernel) was killed
    after submission — Gemini batch jobs run server-side and keep going
    independently of the local runner, so the job may already be running or
    finished. Skips upload/create entirely; only polls + downloads.
    """
    job = client.batches.get(name=job_name)
    state = str(getattr(job.state, "name", str(job.state))).upper()
    print(f"  Attached to job: {job_name} (state: {state})")
    our_cids = [r["key"] for r in requests]
    return _poll_and_download(client, job, our_cids, poll_interval, grounding_data)


# ---------------------------------------------------------------------------
# Response parsing per indicator type
# ---------------------------------------------------------------------------

def _parse_constitution(response_text: str) -> tuple[str, str, Optional[int], dict]:
    """Returns (prediction_str, reasoning, confidence, extra_fields).

    extra_fields keys: 'document_name', 'constitution_year', 'document_types'
    (no column-name prefix; caller appends _SC{n} suffix).
    """
    parsed = parse_json_response(response_text, verbose=False)
    v = validate_constitution_response(parsed)
    pred = v.get("constitution")
    pred_str = str(int(float(pred))) if pred is not None else ""
    extra = {
        "document_name": v.get("document_name"),
        "constitution_year": v.get("constitution_year"),
        "document_types": v.get("document_types"),
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
    grounding_data: Optional[dict] = None,
    include_reasoning: bool = True,
) -> pd.DataFrame:
    """Parse all responses, aggregate SC votes, merge into df.

    For single-mode responses (indicator == "single"), the same full response
    text is re-used for each indicator extracted from metadata["indicators"].
    validate_indicator_response already handles {indicator}_reasoning /
    {indicator}_confidence keys emitted by SinglePromptBuilder.

    include_reasoning=False (build ran with --reasoning false) omits ALL
    reasoning columns from the output instead of writing empty ones.

    grounding_data is not None ⇔ grounding was enabled at build time. In that
    case the {task}_search_queries / {task}_urls_used columns are ALWAYS
    emitted (None/NaN when Gemini did not search for a row), so missing search
    data reads as NA rather than as an absent column.
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

    # Task label for search-metadata columns: a "single" request covers multiple
    # indicators with ONE LLM call, so its search/grounding result is a per-task
    # property, not per-indicator — writing it once as "indicators_*" instead of
    # duplicating identical values across every combined indicator's own columns.
    def _task_label(indicator: str) -> str:
        return "indicators" if indicator not in ("constitution", "elections") else indicator

    # Grounding data keyed by (row_idx, task_label, sc_idx) — built from the
    # ORIGINAL (non-expanded) cid, so a "single" call contributes exactly one
    # entry regardless of how many indicators it covers.
    grounding_by_task: dict[tuple[int, str, int], tuple[str, str]] = {}
    if grounding_data:
        for cid, (q, u) in grounding_data.items():
            try:
                row_idx, indicator, sc_idx = _parse_custom_id(cid)
            except Exception:
                continue
            grounding_by_task[(row_idx, _task_label(indicator), sc_idx)] = (q, u)

    # Collect pre-search metadata (fetched at build time, stored in request metadata
    # sc_idx=0), keyed by (row_idx, task_label) for the same reason as grounding above.
    pre_search_by_row: dict[tuple[int, str], tuple[str, str]] = {}
    if request_metadata:
        for cid, meta in request_metadata.items():
            try:
                row_idx, indicator, sc_idx = _parse_custom_id(cid)
            except Exception:
                continue
            if sc_idx == 0 and ("search_queries" in meta or "search_urls" in meta):
                pre_search_by_row[(row_idx, _task_label(indicator))] = (
                    meta.get("search_queries") or "",
                    meta.get("search_urls") or "",
                )

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
        # Maps sc_idx → (pred_str, reasoning, confidence, extra_fields)
        sc_slot_data: dict[int, tuple] = {}

        for sc_idx in sorted(sc_map.keys()):
            resp_text = sc_map[sc_idx]
            if not resp_text:
                continue

            if indicator == "constitution":
                pred_str, reasoning, confidence, extra = _parse_constitution(resp_text)
            elif indicator == "elections":
                pred_str, reasoning, confidence = _parse_elections(resp_text)
                extra = {}
            else:
                pred_str, reasoning, confidence = _parse_indicator(resp_text, indicator)
                extra = {}

            sc_slot_data[sc_idx] = (pred_str, reasoning, confidence, extra)
            if pred_str:
                votes.append(pred_str)

        updates = row_updates[row_idx]

        if n_samples == 0:
            # No SC: plain column names.
            sc_idx_0 = sc_slot_data.get(0, ("", "", None, {}))
            pred0, reasoning0, confidence0, extra0 = sc_idx_0
            updates[f"{indicator}_prediction"] = _denormalize_pred(pred0, indicator) if pred0 else None
            if include_reasoning:
                updates[f"{indicator}_reasoning"] = reasoning0
            updates[f"{indicator}_confidence"] = confidence0
            if indicator == "constitution":
                updates["constitution_document_name"] = extra0.get("document_name")
                updates["constitution_year"] = extra0.get("constitution_year")
                updates["constitution_document_types"] = extra0.get("document_types")
        else:
            # SC mode: _SCN columns for each slot; _prediction = majority vote.
            # sc_idx=0 → SC1, sc_idx=1 → SC2, ..., sc_idx=N → SC{N+1}
            for sc_idx in range(0, n_samples + 1):
                slot_n = sc_idx + 1  # SC1 = sc_idx 0, SC2 = sc_idx 1, ...
                slot = sc_slot_data.get(sc_idx)
                if slot is not None:
                    pred_str, reasoning, confidence, extra = slot
                    updates[f"{indicator}_SC{slot_n}"] = _denormalize_pred(pred_str, indicator) if pred_str else None
                    if indicator != "constitution":
                        if include_reasoning:
                            updates[f"{indicator}_reasoning_SC{slot_n}"] = reasoning
                        updates[f"{indicator}_confidence_SC{slot_n}"] = confidence
                    else:
                        updates[f"constitution_document_name_SC{slot_n}"] = extra.get("document_name")
                        updates[f"constitution_year_SC{slot_n}"] = extra.get("constitution_year")
                        updates[f"constitution_document_types_SC{slot_n}"] = extra.get("document_types")
                else:
                    updates[f"{indicator}_SC{slot_n}"] = None
                    if indicator != "constitution":
                        if include_reasoning:
                            updates[f"{indicator}_reasoning_SC{slot_n}"] = None
                        updates[f"{indicator}_confidence_SC{slot_n}"] = None
                    else:
                        updates[f"constitution_document_name_SC{slot_n}"] = None
                        updates[f"constitution_year_SC{slot_n}"] = None
                        updates[f"constitution_document_types_SC{slot_n}"] = None

            final_pred_str, agreement, uncertainty = _aggregate_sc(votes, indicator)
            updates[f"{indicator}_prediction"] = _denormalize_pred(final_pred_str, indicator)
            updates[f"{indicator}_agreement"] = round(agreement, 3)
            updates[f"{indicator}_uncertainty"] = uncertainty

        # Pre-search metadata (fetched at build time, one entry per row regardless of SC slots)
        task_label = _task_label(indicator)
        if (row_idx, task_label) in pre_search_by_row:
            q, u = pre_search_by_row[(row_idx, task_label)]
            if q:
                updates[f"{task_label}_search_queries"] = q
            if u:
                updates[f"{task_label}_urls_used"] = u

        # When grounding is enabled, guarantee the search columns exist for every
        # row (None when Gemini did not search); real values are filled in by the
        # grounding_by_task pass below.
        if grounding_data is not None:
            if n_samples == 0:
                updates.setdefault(f"{task_label}_search_queries", None)
                updates.setdefault(f"{task_label}_urls_used", None)
            else:
                for sc_idx in range(0, n_samples + 1):
                    slot_n = sc_idx + 1
                    updates.setdefault(f"{task_label}_search_queries_SC{slot_n}", None)
                    updates.setdefault(f"{task_label}_urls_used_SC{slot_n}", None)

    # Grounding metadata — one column pair per (row, task), not per indicator: a
    # "single" call covers multiple indicators with ONE grounding result, so this
    # is written once as "indicators_*" rather than duplicated across each of
    # sovereign_*, federalism_*, assembly_*, etc.
    for (row_idx, task_label, sc_idx), (q, u) in grounding_by_task.items():
        updates = row_updates[row_idx]
        if n_samples == 0:
            updates[f"{task_label}_search_queries"] = q
            updates[f"{task_label}_urls_used"] = u
        else:
            slot_n = sc_idx + 1
            updates[f"{task_label}_search_queries_SC{slot_n}"] = q
            updates[f"{task_label}_urls_used_SC{slot_n}"] = u

    # Merge into result DataFrame (df already reset_index'd before this call)
    result_df = df.copy()

    # Pre-allocate all new columns at once to avoid DataFrame fragmentation.
    # Use insertion order from row_updates (Python 3.7+ dict ordering) so that
    # {indicator}_prediction appears left of {indicator}_SC1, _SC2, etc.
    seen_cols = set(result_df.columns)
    ordered_new_cols = []
    for row_idx, cols in row_updates.items():
        if row_idx < n_rows:
            for col in cols:
                if col not in seen_cols:
                    ordered_new_cols.append(col)
                    seen_cols.add(col)
    if ordered_new_cols:
        result_df = pd.concat(
            [result_df, pd.DataFrame(index=result_df.index, columns=ordered_new_cols)],
            axis=1,
        )

    # Pre-existing columns being written into (e.g. merging retry results into a prior
    # run's output CSV) may have been inferred as numeric dtype (float64) by read_csv
    # because earlier failed rows left NaNs mixed in with valid values. Predictions are
    # parsed as str, so assigning into such a column raises pandas' LossySetitemError.
    # Newly-created columns (above) are already object dtype and unaffected.
    touched_cols = {col for cols in row_updates.values() for col in cols}
    for col in touched_cols:
        if col in result_df.columns and result_df[col].dtype != object:
            result_df[col] = result_df[col].astype(object)

    # Batch by column instead of setting one cell at a time: with ~135k rows and
    # dozens of indicator columns, per-cell .iloc/get_loc calls (millions of them)
    # took over an hour. Grouping into one vectorized .loc assignment per column
    # only touches the same (row, col) pairs the original loop did, so rows/cols
    # absent from a given update dict are left untouched (required for the
    # retry-merge guarantee that only retried rows get overwritten).
    col_to_rowvals: dict[str, dict[int, object]] = defaultdict(dict)
    for row_idx, cols in row_updates.items():
        if row_idx >= n_rows:
            continue
        for col, val in cols.items():
            col_to_rowvals[col][row_idx] = val

    for col, rowvals in col_to_rowvals.items():
        result_df.loc[list(rowvals.keys()), col] = pd.Series(rowvals)

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
    attach_jobs: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Read one or more pre-built JSONL chunk files, submit each as its own batch job,
    aggregate SC across all results, and write output.

    jsonl_path can be a single file path or a list of paths (for chunked builds).

    input_path is optional. When provided, predictions are merged into the full
    original DataFrame (all original columns preserved). When omitted, a minimal
    DataFrame is constructed from the JSONL row indices — the output will contain
    only prediction/confidence/reasoning columns keyed by positional row index.

    attach_jobs: recovery path for a killed local process. When provided, must have
    one Gemini batch job name per chunk (same order the JSONL re-chunks into) — skips
    upload/create and instead polls/downloads results from those already-submitted jobs.
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

    # Reasoning flag embedded at build time (build_batch_jsonl --reasoning false).
    # When False, reasoning columns are omitted from the output entirely.
    include_reasoning = str(first_meta.get("reasoning", "true")).lower() != "false"
    if not include_reasoning:
        print("  reasoning=False in JSONL metadata — reasoning columns will be omitted.")

    total_per_group = n_samples + 1
    print(f"  {len(all_requests)} total requests ({len(all_requests) // total_per_group} groups × {total_per_group})")

    # Build DataFrame from JSONL when --dataset is not provided.
    # sc_idx=0 requests carry row_data embedded by build_batch_jsonl.py; use those to
    # reconstruct the full original dataset (all columns, correct dtypes from str conversion).
    # Falls back to a minimal row_idx-only DataFrame for older JSONL files without row_data.
    if df is None:
        row_data_by_idx: dict[int, dict] = {}
        for r in all_requests:
            cid = r.get("key", "")
            meta = r.get("metadata", {})
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
                int(r["key"].split("|")[0])
                for r in all_requests
                if "key" in r
            )
            df = pd.DataFrame({"row_idx": range(max_row_idx + 1)})
            print(f"  No row_data in JSONL; output will contain predictions only ({max_row_idx + 1} rows)")

    # Build metadata lookup for _aggregate_and_merge (needed for single-mode expansion)
    request_metadata = {r["key"]: r.get("metadata", {}) for r in all_requests}

    # Row-identity guard: sc_idx=0 requests embed the full original row in
    # metadata.row_data. Verify each embedded identifier against the dataset
    # row at the same positional index BEFORE submitting — merging is
    # positional, so a reordered/filtered --dataset would otherwise silently
    # attach predictions to the wrong leaders. Guards on "id" (the unique
    # key); falls back to "slug_id" only when the dataset has no "id" column
    # (slug_id is informative but NOT unique — repeated across spells).
    if input_path is not None and "id" in df.columns:
        _guard_cols = ["id"]
    elif input_path is not None and "slug_id" in df.columns:
        _guard_cols = ["slug_id"]
    else:
        _guard_cols = []
    if _guard_cols:
        _mismatches: list[tuple] = []
        _checked = 0
        for r in all_requests:
            meta = r.get("metadata", {})
            if "row_data" not in meta:
                continue
            try:
                _ri, _, _si = _parse_custom_id(r.get("key", ""))
            except Exception:
                continue
            if _si != 0:
                continue
            try:
                _embedded = json.loads(meta["row_data"])
            except (json.JSONDecodeError, TypeError):
                continue
            if _ri >= len(df):
                _mismatches.append((_ri, "<any>", _embedded.get(_guard_cols[0]), "<row_idx beyond dataset>"))
                continue
            _row_checked = False
            for _col in _guard_cols:
                if _col not in _embedded:
                    continue
                _row_checked = True
                if not _ids_equal(_embedded[_col], df.iloc[_ri][_col]):
                    _mismatches.append((_ri, _col, _embedded[_col], df.iloc[_ri][_col]))
            _checked += 1 if _row_checked else 0
        if _mismatches:
            _preview = "; ".join(
                f"row {ri} [{col}]: JSONL={a!r} vs dataset={b!r}" for ri, col, a, b in _mismatches[:5]
            )
            raise ValueError(
                f"Row-identity check FAILED for {len(_mismatches)} value(s) — the --dataset row "
                f"order no longer matches this JSONL (merging is positional). "
                f"First mismatches: {_preview}. Use the dataset the JSONL was built from "
                f"(or, on retry rounds, the prior run's --output) without reordering, "
                f"filtering, or re-sorting rows."
            )
        if _checked:
            print(f"  Row-identity check passed: {_checked} rows verified via {_guard_cols}.")

    # Auto-detect grounding from JSONL content
    use_grounding = _requests_use_grounding(all_requests)
    if use_grounding:
        print("  Detected Google Search grounding in JSONL — grounding metadata will be collected.")

    # Chunk by size and submit each chunk as a separate batch job
    chunks = _chunk_requests(all_requests)

    if attach_jobs is not None and len(attach_jobs) != len(chunks):
        raise ValueError(
            f"--attach-jobs has {len(attach_jobs)} job name(s) but this JSONL re-chunks into "
            f"{len(chunks)} chunk(s) — pass one job name per chunk, in the same order the "
            f"original build/submit produced them."
        )

    client = _get_client(api_key)
    display_base = f"const-llm-{Path(output_path).stem}-{int(time.time())}"
    raw_results: dict[str, str] = {}
    all_grounding: dict[str, tuple] = {}
    for i, chunk in enumerate(chunks):
        chunk_label = f"chunk {i + 1}/{len(chunks)}" if len(chunks) > 1 else "single job"
        size_mb = sum(len(json.dumps(r, ensure_ascii=False).encode()) + 1 for r in chunk) / 1024 / 1024
        if attach_jobs is not None:
            print(f"\nAttaching to existing job for {chunk_label}: {len(chunk)} requests ({size_mb:.1f} MB) ...")
            chunk_results = _attach_and_wait(client, attach_jobs[i], chunk, poll_interval,
                                             grounding_data=all_grounding if use_grounding else None)
        else:
            display_name = f"{display_base}-c{i + 1:03d}" if len(chunks) > 1 else display_base
            print(f"\nSubmitting {chunk_label}: {len(chunk)} requests ({size_mb:.1f} MB) ...")
            chunk_results = _submit_and_wait(client, model, chunk, display_name, poll_interval,
                                             grounding_data=all_grounding if use_grounding else None)
        raw_results.update(chunk_results)

    # Free the request bodies now that responses are collected — only the small
    # per-cid metadata (already extracted into request_metadata) is needed from
    # here on. Holding both `all_requests` (grounding-heavy prompts, can be
    # multi-GB for 100k+ row batches) and the merge/aggregation structures at
    # once is what pushed a 135k-row grounding batch past available RAM (a
    # Jetsam/OOM kill was observed in production on a 16GB machine, mid-merge).
    total_requests = len(all_requests)
    expected_cids = set(request_metadata.keys())
    requested_rows: set[int] = set()
    for cid in request_metadata:
        try:
            requested_rows.add(_parse_custom_id(cid)[0])
        except Exception:
            pass
    del all_requests, chunks

    result_df = _aggregate_and_merge(raw_results, df, n_samples, request_metadata,
                                      grounding_data=all_grounding if use_grounding else None,
                                      include_reasoning=include_reasoning)

    # Response-level failures (CID missing from results or response text empty)
    missing = expected_cids - set(raw_results.keys())
    empty = {cid for cid, text in raw_results.items() if not text}
    response_failed_rows = sorted({_parse_custom_id(cid)[0] for cid in missing | empty})

    # Prediction-level nulls: response received but truncated/unparseable for some indicators.
    # Only check indicators actually requested in this batch — not columns inherited from a prior
    # merged run or indicators from a different task (e.g. constitution columns when running
    # indicators task). Requested indicators are in metadata["indicators"] (single mode) or
    # directly in the custom_id indicator slot (constitution/elections).
    requested_indicators: set[str] = set()
    for cid, meta in request_metadata.items():
        if "indicators" in meta:
            try:
                requested_indicators.update(json.loads(meta["indicators"]))
            except (json.JSONDecodeError, TypeError):
                pass
        try:
            _, ind, _ = _parse_custom_id(cid)
            if ind not in ("single",):
                requested_indicators.add(ind)
        except Exception:
            pass

    # requested_rows (rows actually covered by this JSONL) was computed above,
    # right before all_requests was freed. Null-checking is restricted to these:
    # a subset build (elections gating, sanity_check re-runs, --test) must not
    # flag rows that were never requested in this batch as failures.

    # Elections pass-through: the build filters to assembly_prediction == 2, so
    # non-requested rows with assembly != 2 get elections_prediction = "0" with
    # no LLM call (same convention as pipeline/post_processing.py).
    if "elections" in requested_indicators and "assembly_prediction" in result_df.columns:
        if "elections_prediction" not in result_df.columns:
            result_df["elections_prediction"] = None
        _assembly_num = pd.to_numeric(result_df["assembly_prediction"], errors="coerce")
        _passthrough = (
            result_df["elections_prediction"].isnull()
            & (_assembly_num != 2)
            & ~result_df.index.isin(list(requested_rows))
        )
        result_df.loc[_passthrough, "elections_prediction"] = "0"
        if _passthrough.any():
            print(f"  Elections pass-through (assembly != 2): {int(_passthrough.sum())} rows set to 0")

    pred_cols = [
        f"{ind}_prediction" for ind in requested_indicators
        if f"{ind}_prediction" in result_df.columns
    ]
    null_row_idxs: set[int] = set()
    if pred_cols:
        null_row_idxs = set(
            result_df.index[result_df[pred_cols].isnull().any(axis=1)].tolist()
        ) & requested_rows

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
        # all_requests was freed after download to bound memory (see above) — re-stream the
        # JSONL files from disk instead of holding every request body in memory the whole run;
        # this only runs when there ARE failures, so the extra I/O is the rare-path cost.
        failed_row_set = set(all_failed_rows)
        failed_requests = []
        for jpath in jsonl_paths:
            with open(jpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    try:
                        if _parse_custom_id(r["key"])[0] in failed_row_set:
                            failed_requests.append(r)
                    except Exception:
                        continue
        retry_path = Path(output_path).parent / (Path(output_path).stem + "_failed_requests.jsonl")
        retry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(retry_path, "w", encoding="utf-8") as f:
            for r in failed_requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Retry JSONL: {retry_path} ({len(all_failed_rows)} rows, {len(failed_requests)} requests)")
        print(
            f"  Re-run: python pipeline/jsonl_batch_runner.py --input {retry_path} "
            f"--dataset {output_path} --output {output_path} ...\n"
            f"    NOTE: --dataset MUST point at this run's output ({output_path}), not the original\n"
            f"    raw input — otherwise only the {len(all_failed_rows)} retried rows survive and every\n"
            f"    already-successful row is dropped from the merged result."
        )

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
        "run_timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_samples": n_samples,
        "input_jsonl": jsonl_paths,
        "total_requests": total_requests,
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
            "the output contains only prediction/confidence/reasoning columns.\n"
            "RETRY ROUNDS: when re-running a '..._failed_requests.jsonl', pass the PRIOR "
            "run's --output CSV here (not the original raw input) so only the retried rows "
            "are overwritten and every already-successful row is preserved."
        ),
    )
    parser.add_argument("--output", "-o", required=True, help="Output CSV path.")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument(
        "--n-samples", type=int, default=0,
        help=(
            "Additional SC samples. Must match the value used during build. "
            "Default 0 → no SC (single call). n_samples=2 → 3 total votes."
        ),
    )
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument(
        "--attach-jobs", nargs="+", default=None,
        help=(
            "Recovery path: one or more ALREADY-SUBMITTED Gemini batch job names "
            "(e.g. --attach-jobs batches/abc123 batches/def456), one per --input chunk, "
            "in the same order. Skips upload/create and just polls/downloads results — "
            "use this after a local process (e.g. Jupyter kernel) was killed mid-run, "
            "since the remote batch job keeps running independently. Find job names/states "
            "with client.batches.list() via the google-genai SDK."
        ),
    )
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
        attach_jobs=args.attach_jobs,
    )


if __name__ == "__main__":
    main()
