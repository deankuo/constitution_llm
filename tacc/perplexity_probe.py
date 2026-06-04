#!/usr/bin/env python3
"""
Knowledge probe for historical leaders and polities.

Supports two backends auto-detected from --model:
  - Open-source (vLLM): JSON-constrained generation via FSM; pass a local model directory.
  - Commercial (Gemini / GPT / Claude / Bedrock): prompt-based JSON generation via API.

One LLM call is made per question per row. Output CSV contains gen_* columns with the
model's answers alongside ground-truth columns for offline comparison.
"""

import csv
import os
import sys
import json
import logging
import argparse
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field
from tqdm import tqdm

# ── vLLM imports (only used for local open-source models) ────────────────────
try:
    from vllm import LLM, SamplingParams as _SP

    try:
        from vllm.sampling_params import StructuredOutputsParams as _SOParams

        def _json_sp(schema: type) -> _SP:
            return _SP(
                max_tokens=200, temperature=0.0,
                structured_outputs=_SOParams(json=schema.model_json_schema()),
            )
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams as _SOParams  # type: ignore[assignment]

        def _json_sp(schema: type) -> _SP:
            return _SP(
                max_tokens=200, temperature=0.0,
                guided_decoding=_SOParams(json=schema.model_json_schema()),
            )

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

# Allow importing from the parent project (models/, config.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)


# ── Answer schemas ────────────────────────────────────────────────────────────
_BC_NOTE = "Use a negative integer for BC years, e.g., -44 for 44 BC."


class _YearAnswer(BaseModel):
    year: int = Field(description=f"Calendar year. {_BC_NOTE}")


class _TenureAnswer(BaseModel):
    years: int = Field(ge=0, le=300, description="Duration of rule in whole years (not a calendar year).")


class _NameAnswer(BaseModel):
    name: str = Field(max_length=120, description="Full name of the ruler.")


class _PolityAnswer(BaseModel):
    polity: str = Field(max_length=120, description="Name of the polity or territory.")


# ── Shared system prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a historical knowledge assistant. "
    "Answer only with a valid JSON object and no other text."
)

# ── Question templates ────────────────────────────────────────────────────────
# Design rule: never include two of {entry_year, exit_year, tenure} in the same
# question when one of them is the answer — the model could compute it arithmetically.
#
# `requires`: context fields that must be non-None to form the question (not the answer field).
# Tenure is computed from entry/exit if absent from the CSV.

STATEMENTS = [
    {
        "label":      "entry_year",
        "requires":   ["name", "polity", "exit_year"],
        "question":   (
            "{name} ruled {polity} and left power in {exit_year}.\n"
            "In which year did {name} first come to power in {polity}?\n"
            'Respond with: {{"year": <integer>}}\n'
            "Use negative integers for BC years."
        ),
        "ans_schema": _YearAnswer,
        "ans_field":  "year",
    },
    {
        "label":      "exit_year",
        "requires":   ["name", "polity", "entry_year"],
        "question":   (
            "{name} came to power in {polity} in {entry_year}.\n"
            "In which year did {name}'s rule of {polity} end?\n"
            'Respond with: {{"year": <integer>}}\n'
            "Use negative integers for BC years."
        ),
        "ans_schema": _YearAnswer,
        "ans_field":  "year",
    },
    {
        "label":      "tenure_from_entry",
        "requires":   ["name", "polity", "entry_year"],
        "question":   (
            "{name} came to power in {polity} in {entry_year}.\n"
            "How many years did {name} hold power in {polity}?\n"
            'Respond with: {{"years": <integer>}}'
        ),
        "ans_schema": _TenureAnswer,
        "ans_field":  "years",
    },
    {
        "label":      "tenure_from_exit",
        "requires":   ["name", "polity", "exit_year"],
        "question":   (
            "{name} ruled {polity} and left power in {exit_year}.\n"
            "How many years did {name} hold power in {polity}?\n"
            'Respond with: {{"years": <integer>}}'
        ),
        "ans_schema": _TenureAnswer,
        "ans_field":  "years",
    },
    {
        "label":      "leader_name",
        "requires":   ["polity", "entry_year", "exit_year"],
        "question":   (
            "A ruler of {polity} came to power in {entry_year} and left power in {exit_year}.\n"
            "Who was this ruler?\n"
            'Respond with: {{"name": "<full name>"}}'
        ),
        "ans_schema": _NameAnswer,
        "ans_field":  "name",
    },
    {
        "label":      "polity_name",
        "requires":   ["name", "entry_year", "exit_year"],
        "question":   (
            "{name} came to power in {entry_year} and left power in {exit_year}.\n"
            "Which polity or territory did {name} rule during this period?\n"
            'Respond with: {{"polity": "<name>"}}'
        ),
        "ans_schema": _PolityAnswer,
        "ans_field":  "polity",
    },
]

LABELS = [s["label"] for s in STATEMENTS]
OUT_FIELDS = (
    ["row_idx", "polity", "leader", "entry_year", "exit_year", "tenure", "region"]
    + [f"gen_{lbl}" for lbl in LABELS]
)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Knowledge probe — tests factual recall about historical leaders"
    )
    p.add_argument(
        "--model", required=True,
        help=(
            "Model identifier. "
            "Local directory path → open-source via vLLM "
            "(e.g. ../llm_models/Qwen2.5-32B-Instruct). "
            "Model name → commercial API "
            "(e.g. gemini-2.5-pro, gpt-4o, us.anthropic.claude-sonnet-4-5-20250929-v1:0)."
        ),
    )
    p.add_argument("--input",  required=True, help="Input CSV")
    p.add_argument("--output", required=True, help="Output CSV (checkpoint/resume supported)")

    # vLLM-only (ignored for commercial models)
    g = p.add_argument_group("vLLM options (local open-source models only)")
    g.add_argument("--tensor-parallel-size", type=int, default=2,
                   help="Number of GPUs for tensor parallelism (default: 2)")
    g.add_argument("--max-model-len", type=int, default=512,
                   help="Max context length for KV cache allocation (default: 512)")
    g.add_argument("--dtype", choices=["bfloat16", "float16", "auto"], default="bfloat16",
                   help="Model weight dtype (default: bfloat16)")
    g.add_argument("--batch-size", type=int, default=1000,
                   help="Rows per vLLM generate call (default: 1000)")

    # Commercial-only (ignored for vLLM)
    g2 = p.add_argument_group("Commercial API options")
    g2.add_argument("--parallel-rows", type=int, default=4,
                    help="Number of rows to process in parallel (default: 4)")
    g2.add_argument("--delay", type=float, default=1.0,
                    help="Seconds to wait between API calls within a row (default: 1.0)")

    p.add_argument("--test-n", type=int, default=None,
                   help="Process only the first N rows — for smoke testing (default: full dataset)")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return p.parse_args()


# ── Model detection and loading ───────────────────────────────────────────────

def _is_local_model(model: str) -> bool:
    """True if model points to a local directory (open-source / vLLM path)."""
    return Path(model).is_dir()


def load_model(args) -> tuple:
    """
    Returns ("vllm", llm) for local models or ("commercial", llm) for API models.
    Commercial path uses create_llm() from models/llm_clients.py and reads API keys
    from environment variables (OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY,
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN).
    """
    if _is_local_model(args.model):
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is not installed. Run: pip install vllm"
            )
        log.info(
            f"Local model: {args.model}  "
            f"tp={args.tensor_parallel_size}  dtype={args.dtype}"
        )
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            trust_remote_code=True,
        )
        log.info("Model loaded")
        return ("vllm", llm)

    from models.llm_clients import create_llm, detect_provider
    api_keys = {
        "openai":               os.getenv("OPENAI_API_KEY"),
        "gemini":               os.getenv("GEMINI_API_KEY"),
        "anthropic":            os.getenv("ANTHROPIC_API_KEY"),
        "aws_access_key_id":    os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_session_token":    os.getenv("AWS_SESSION_TOKEN"),
    }
    provider = detect_provider(args.model)
    log.info(f"Commercial model: {args.model}  provider={provider}")
    return ("commercial", create_llm(args.model, api_keys))


# ── Data helpers ──────────────────────────────────────────────────────────────

def parse_fields(row: dict) -> dict:
    """Map CSV columns → template fields; 'NA'/empty → None.

    Tenure is computed as exit_year - entry_year when the CSV has no tenure column.
    """
    def get(key):
        v = row.get(key, "").strip()
        return v if v.upper() not in ("NA", "") else None

    entry  = get("leader_first_year")
    exit_  = get("leader_last_year")
    tenure = get("tenure")

    if tenure is None and entry is not None and exit_ is not None:
        try:
            tenure = str(int(exit_) - int(entry))
        except (ValueError, TypeError):
            pass

    return {
        "name":       get("leader_name"),
        "polity":     get("polity_name"),
        "region":     get("polity_region"),
        "entry_year": entry,
        "exit_year":  exit_,
        "tenure":     tenure,
    }


def _can_form(stmt: dict, fields: dict) -> bool:
    return all(fields.get(r) is not None for r in stmt["requires"])


def _format_question(stmt: dict, fields: dict) -> str:
    return stmt["question"].format(**fields)


def _make_base_rec(row_idx: int, fields: dict) -> dict:
    return {
        "row_idx":    row_idx,
        "polity":     fields.get("polity") or "",
        "leader":     fields.get("name") or "",
        "entry_year": fields.get("entry_year") or "",
        "exit_year":  fields.get("exit_year") or "",
        "tenure":     fields.get("tenure") or "",
        "region":     fields.get("region") or "",
    }


# ── JSON extraction ───────────────────────────────────────────────────────────

def parse_answer(gen_text: str, ans_field: str) -> str:
    """Extract ans_field value from a JSON response string.

    Handles markdown code fences and falls back to regex if json.loads fails.
    """
    if not gen_text:
        return ""
    text = re.sub(r"```(?:json)?\s*", "", gen_text.strip())
    text = re.sub(r"```\s*$", "", text.strip()).strip()

    def _extract(data: dict) -> str:
        v = data.get(ans_field)
        return str(v) if v is not None else ""

    try:
        return _extract(json.loads(text))
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[^}]+\}", text)
    if m:
        try:
            return _extract(json.loads(m.group()))
        except json.JSONDecodeError:
            pass

    log.debug(f"Could not parse JSON from: {gen_text[:120]!r}")
    return ""


# ── vLLM batch processing (gen-only, no PPL) ─────────────────────────────────

def process_vllm_batch(llm, indexed_rows: list[tuple[int, dict]]) -> list[dict]:
    parsed = [(row_idx, parse_fields(row)) for row_idx, row in indexed_rows]

    prompts: list[str] = []
    sp_list: list      = []
    keys:    list      = []

    for offset, (_, fields) in enumerate(parsed):
        for stmt in STATEMENTS:
            if not _can_form(stmt, fields):
                continue
            prompts.append(_format_question(stmt, fields))
            sp_list.append(_json_sp(stmt["ans_schema"]))
            keys.append((offset, stmt["label"], stmt["ans_field"]))

    gen_texts = []
    if prompts:
        outputs   = llm.generate(prompts, sp_list, use_tqdm=False)
        gen_texts = [o.outputs[0].text.strip() for o in outputs]

    gen_map = {
        (off, lbl): parse_answer(txt, field)
        for (off, lbl, field), txt in zip(keys, gen_texts)
    }

    results = []
    for offset, (row_idx, fields) in enumerate(parsed):
        rec = _make_base_rec(row_idx, fields)
        for stmt in STATEMENTS:
            rec[f"gen_{stmt['label']}"] = gen_map.get((offset, stmt["label"]), "")
        results.append(rec)

    return results


# ── Commercial single-row processing ─────────────────────────────────────────

def process_one_commercial(llm, row_idx: int, row: dict, delay: float = 1.0) -> tuple[dict, int]:
    """Process all questions for a single row via commercial API (one call per question).

    Returns (rec, n_errors) so callers can tally failures.
    """
    fields   = parse_fields(row)
    rec      = _make_base_rec(row_idx, fields)
    n_errors = 0

    for i, stmt in enumerate(STATEMENTS):
        if not _can_form(stmt, fields):
            rec[f"gen_{stmt['label']}"] = ""
            continue
        if i > 0 and delay > 0:
            time.sleep(delay)
        try:
            response = llm.call(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=_format_question(stmt, fields),
                temperature=0.0,
                max_tokens=1024,
                response_schema=stmt["ans_schema"],
            )
            rec[f"gen_{stmt['label']}"] = parse_answer(response.content, stmt["ans_field"])
        except Exception as e:
            log.warning(f"API error row {row_idx} [{stmt['label']}]: {e}")
            rec[f"gen_{stmt['label']}"] = ""
            n_errors += 1
        
        # print(response)

    return rec, n_errors


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_done(output_path: Path) -> set[int]:
    """Return row indices where ALL gen_* columns are non-empty.

    Using all() instead of any() ensures partially-completed rows (e.g. where
    some API calls failed) are retried rather than permanently skipped.
    """
    if not output_path.exists():
        return set()
    with open(output_path) as f:
        return {
            int(r["row_idx"])
            for r in csv.DictReader(f)
            if r.get("row_idx") and all(r.get(f"gen_{lbl}", "") for lbl in LABELS)
        }


def has_scoreable_data(row: dict) -> bool:
    """True if at least one question can be formed for this row."""
    fields = parse_fields(row)
    return any(_can_form(stmt, fields) for stmt in STATEMENTS)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("google_genai").setLevel(logging.ERROR)

    mode, llm = load_model(args)

    with open(args.input) as f:
        all_rows = list(csv.DictReader(f))
    if args.test_n:
        all_rows = all_rows[: args.test_n]
        log.info(f"Test mode: {args.test_n} rows")
    log.info(f"Input rows: {len(all_rows)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)

    pending = [
        (idx, row) for idx, row in enumerate(all_rows)
        if idx not in done and has_scoreable_data(row)
    ]
    skipped = sum(1 for _, row in enumerate(all_rows) if not has_scoreable_data(row))
    log.info(
        f"Total: {len(all_rows)}  |  already done: {len(done)}  |  "
        f"no data (skipped): {skipped}  |  to process: {len(pending)}"
    )

    fh = open(out_path, "a" if done else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
    if not done:
        writer.writeheader()

    try:
        if mode == "vllm":
            buffer: list[tuple[int, dict]] = []
            with tqdm(total=len(pending), unit="row", desc="Probing") as pbar:
                for idx, row in pending:
                    buffer.append((idx, row))
                    if len(buffer) < args.batch_size:
                        continue
                    writer.writerows(process_vllm_batch(llm, buffer))
                    fh.flush()
                    pbar.update(len(buffer))
                    buffer = []
                if buffer:
                    writer.writerows(process_vllm_batch(llm, buffer))
                    fh.flush()
                    pbar.update(len(buffer))

        else:  # commercial
            total_errors = 0
            with tqdm(total=len(pending), unit="row", desc="Probing") as pbar:
                with ThreadPoolExecutor(max_workers=args.parallel_rows) as executor:
                    futures = {
                        executor.submit(
                            process_one_commercial, llm, idx, row, args.delay
                        ): idx
                        for idx, row in pending
                    }
                    for fut in as_completed(futures):
                        try:
                            rec, n_errors = fut.result()
                            writer.writerow(rec)
                            fh.flush()
                            total_errors += n_errors
                        except Exception as e:
                            log.error(f"Row {futures[fut]} failed: {e}")
                        pbar.update(1)
            if total_errors:
                log.warning(
                    f"{total_errors} API call(s) failed — rerun to retry incomplete rows"
                )

    finally:
        fh.close()

    log.info(f"Done → {out_path}")


if __name__ == "__main__":
    main()
