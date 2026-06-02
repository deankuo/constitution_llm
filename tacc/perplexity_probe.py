#!/usr/bin/env python3
"""
Perplexity-based knowledge probe for historical leaders and polities.

For each factual statement template, runs two vLLM passes per batch:
  1. PPL pass  — prompt_logprobs on (context + scored), measures surprise.
  2. Gen pass  — JSON-constrained Q&A via vLLM structured outputs; the model
                 is forced to emit a valid Pydantic-schema JSON object so the
                 answer is always machine-readable without post-hoc parsing.

Lower PPL  = model finds the fact unsurprising = entity likely known.
Higher PPL = model is uncertain about the fact = entity likely unknown.
correct_*  = 1 if the model's JSON answer matches the ground-truth value.
"""

import csv
import math
import logging
import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ── vLLM structured-output API (changed in v0.12) ────────────────────────────
# v0.8–0.11 uses GuidedDecodingParams; v0.12+ renamed it StructuredOutputsParams.
try:
    from vllm.sampling_params import StructuredOutputsParams as _SOParams

    def _json_sp(schema: type) -> SamplingParams:
        return SamplingParams(
            max_tokens=300, temperature=0.0,
            structured_outputs=_SOParams(json=schema.model_json_schema()),
        )
except ImportError:
    from vllm.sampling_params import GuidedDecodingParams as _SOParams  # type: ignore[assignment]

    def _json_sp(schema: type) -> SamplingParams:
        return SamplingParams(
            max_tokens=300, temperature=0.0,
            guided_decoding=_SOParams(json=schema.model_json_schema()),
        )

log = logging.getLogger(__name__)

# ── Answer schemas (one Pydantic model per probe type) ────────────────────────
# vLLM's JSON constrained decoding uses these schemas to force the model to emit
# valid JSON — no free-form text can appear in gen_* columns.
#
# ConfigDict(extra="forbid") → additionalProperties:false in the JSON schema,
# so the model cannot append extra fields and the JSON stays short.
# Field(description=...) is embedded in the schema and guides value semantics.
# Field(ge/le/max_length) constrains valid values in the FSM, preventing
# nonsensical outputs like tenure=1456 or a 500-token name string.

_BC_NOTE = "Use a negative integer for BC years, e.g., -44 for 44 BC."

class _YearAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    year: int = Field(description=f"Calendar year. {_BC_NOTE}")

class _ReignAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_year: int = Field(description=f"First year of rule. {_BC_NOTE}")
    end_year:   int = Field(description=f"Last year of rule.  {_BC_NOTE}")

class _TenureAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    years: int = Field(description="Duration of rule in whole years (not a calendar year). Typical range 0–300.")

class _NameAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(max_length=120, description="Full name of the ruler.")


# ── Probe statement templates ─────────────────────────────────────────────────
# Design principle: entity names go in `context` (unscored) to avoid conflating
# tokenization difficulty of rare names with factual knowledge.
# `scored` contains only ground-truth facts — this is what PPL measures.
#
# To add/remove probes: edit ONLY this list. No other code changes needed.
# `requires`:      fields that must be non-NA; statement is skipped otherwise.
# `gen_q`:         Q&A prompt for the gen pass; the model answers in JSON.
# `ans_schema`:    Pydantic class whose JSON schema constrains the gen output.
#
# Available fields: name, polity, entry_year, exit_year, tenure, region
# NA rates in plt_leaders_for_llm_20260120.csv:
#   entry_year: ~0%  |  exit_year: ~2%  |  tenure: ~2%  |  region: ~51%
STATEMENTS = [
    {
        "label":      "reign_range",
        "context":    "{name}, a ruler of {polity},",
        "scored":     " came to power in {entry_year} and ruled until {exit_year}.",
        "requires":   ["entry_year", "exit_year"],
        "gen_q":      "What were the start and end years of {name}'s rule of {polity}? Use negative integers for BC years.",
        "ans_schema": _ReignAnswer,
    },
    {
        "label":      "entry_year",
        "context":    "{name} became the ruler of {polity}",
        "scored":     " in {entry_year}.",
        "requires":   ["entry_year"],
        "gen_q":      "What year did {name} first take power in {polity}? Use a negative integer for BC years.",
        "ans_schema": _YearAnswer,
    },
    {
        "label":      "exit_year",
        "context":    "{name}'s rule of {polity} ended",
        "scored":     " in {exit_year}.",
        "requires":   ["exit_year"],
        "gen_q":      "What year did {name}'s rule of {polity} end? Use a negative integer for BC years.",
        "ans_schema": _YearAnswer,
    },
    {
        "label":      "tenure_length",
        "context":    "In {polity}, {name}",
        "scored":     " held power for {tenure} years.",
        "requires":   ["tenure"],
        "gen_q":      "How many years did {name} hold power in {polity}?",
        "ans_schema": _TenureAnswer,
    },
    {
        "label":      "leader_name",
        "context":    "From {entry_year} to {exit_year}, {polity} was ruled by",
        "scored":     " {name}.",
        "requires":   ["entry_year", "exit_year", "name"],
        "gen_q":      "Who ruled {polity} from {entry_year} to {exit_year}?",
        "ans_schema": _NameAnswer,
    },
]

LABELS     = [s["label"] for s in STATEMENTS]
OUT_FIELDS = (
    ["row_idx", "polity", "leader", "entry_year", "exit_year", "region"]
    + [f"ppl_{lbl}"     for lbl in LABELS]
    + [f"gen_{lbl}"     for lbl in LABELS]
    + [f"correct_{lbl}" for lbl in LABELS]
    + ["n_stmts", "geo_mean_ppl", "n_correct"]
)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="PPL knowledge probe — scores factual statements to measure model knowledge"
    )
    p.add_argument("--model",               default="../llm_models/Qwen2.5-32B-Instruct",
                   help="Local path to model directory")
    p.add_argument("--input",               required=True,
                   help="Input CSV (plt_leaders_for_llm_20260120.csv schema)")
    p.add_argument("--output",              required=True,
                   help="Output CSV (supports checkpoint/resume)")
    p.add_argument("--tensor-parallel-size",type=int, default=2,
                   help="Number of GPUs for tensor parallelism (default: 2)")
    p.add_argument("--max-model-len",       type=int, default=512,
                   help="Max token length for KV cache allocation (default: 512). "
                        "Short statements rarely exceed 150 tokens; lower = less VRAM.")
    p.add_argument("--dtype",               choices=["bfloat16", "float16", "auto"],
                   default="bfloat16",
                   help="Model weight dtype (default: bfloat16)")
    p.add_argument("--batch-size",          type=int, default=1000,
                   help="Rows per vLLM call; controls checkpoint frequency (default: 1000)")
    p.add_argument("--checkpoint-interval", type=int, default=5000,
                   help="Log progress every N processed rows (default: 5000)")
    p.add_argument("--test-n",              type=int, default=None,
                   help="Process only first N rows — for smoke testing (default: full dataset)")
    p.add_argument("--log-level",           choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(args):
    log.info(f"Loading model: {args.model}  tp={args.tensor_parallel_size}  dtype={args.dtype}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    log.info("Model ready")
    return llm, tok


# ── Data helpers ──────────────────────────────────────────────────────────────

def parse_fields(row: dict) -> dict:
    """Map CSV columns → template fields; 'NA'/empty → None."""
    def get(key):
        v = row.get(key, "").strip()
        return v if v.upper() not in ("NA", "") else None

    return {
        "name":       get("leader_name"),
        "polity":     get("polity_name"),
        "region":     get("polity_region"),
        "entry_year": get("leader_first_year"),
        "exit_year":  get("leader_last_year"),
        "tenure":     get("tenure"),
    }


def format_statement(stmt: dict, fields: dict) -> tuple[str, str, str] | None:
    """Return (context, scored, gen_q) or None if any required field is missing."""
    if any(fields.get(r) is None for r in stmt["requires"]):
        return None
    try:
        return (
            stmt["context"].format(**fields),
            stmt["scored"].format(**fields),
            stmt["gen_q"].format(**fields),
        )
    except KeyError:
        return None


# ── Scoring (PPL + generation) ────────────────────────────────────────────────

_PPL_PARAMS = SamplingParams(
    max_tokens=1,       # minimum to trigger prompt_logprobs; no real generation needed
    prompt_logprobs=1,  # top-1 logprob per prompt token; actual token always included
)


def score_batch(
    llm, tok, pairs: list[tuple[str, str, str, type]]
) -> tuple[list[float], list[str]]:
    """
    For each (context, scored, gen_q, ans_schema) tuple return (ppl, gen_text).

    Pass 1 — PPL: feed (context + scored) with prompt_logprobs; sum NLL over
    the scored span only (positions >= ctx_len).
    Pass 2 — Gen: feed gen_q with per-request JSON-constrained SamplingParams
    derived from ans_schema; the model is forced to emit a valid JSON object.
    """
    full_texts, ctx_lens = [], []
    gen_prompts, gen_sp_list = [], []

    for ctx, scored, gen_q, ans_schema in pairs:
        ctx_ids  = tok.encode(ctx,          add_special_tokens=True)
        full_ids = tok.encode(ctx + scored, add_special_tokens=True)

        # Context must be an exact token prefix; a mismatch shifts the scoring
        # window and silently corrupts PPL values.
        if full_ids[: len(ctx_ids)] != ctx_ids:
            log.warning(
                f"Context is not an exact token prefix — scored span may be off by "
                f"1–2 tokens. ctx={ctx!r}"
            )

        full_texts.append(ctx + scored)
        ctx_lens.append(len(ctx_ids))
        gen_prompts.append(gen_q)
        gen_sp_list.append(_json_sp(ans_schema))

    # Pass 1: PPL
    ppl_outputs = llm.generate(full_texts, _PPL_PARAMS, use_tqdm=False)
    ppls = []
    for output, ctx_len in zip(ppl_outputs, ctx_lens):
        prompt_ids = output.prompt_token_ids
        prompt_lp  = output.prompt_logprobs
        nll_values = []
        for pos in range(ctx_len, len(prompt_ids)):
            lp_dict = prompt_lp[pos] if prompt_lp else None
            if lp_dict is None:
                continue
            actual_id = prompt_ids[pos]
            if actual_id in lp_dict:
                nll_values.append(-lp_dict[actual_id].logprob)
        ppls.append(
            math.exp(sum(nll_values) / len(nll_values)) if nll_values else float("inf")
        )

    # Pass 2: JSON-constrained generation — one SamplingParams per probe schema
    gen_outputs = llm.generate(gen_prompts, gen_sp_list, use_tqdm=False)
    gen_texts   = [o.outputs[0].text.strip() for o in gen_outputs]

    return ppls, gen_texts


def _year_matches(expected_str: str, actual_int: int) -> bool:
    """True if the model's integer equals the CSV year (handles BC abs-value)."""
    exp = expected_str.strip()
    return str(actual_int) == exp or str(abs(actual_int)) == exp.lstrip("-")


def check_correct(stmt: dict, fields: dict, gen_text: str) -> int:
    """1 if the JSON answer in gen_text matches the ground-truth value.

    The gen pass uses JSON-constrained decoding, so gen_text is always a valid
    JSON object matching ans_schema.  We use model_validate_json() to parse it
    and then compare field values exactly.

    For BC/BCE years the CSV stores negative integers ("-44"); the model may
    output either -44 or 44 (treating it as unsigned), so we accept both.

    Name probe: full case-insensitive match, with single-word fallback for
    partial names ("Brankovic" matching "Vuk Grgurevic Brankovic").
    """
    if not gen_text:
        return 0

    schema = stmt.get("ans_schema")
    try:
        data = schema.model_validate_json(gen_text)
    except Exception:
        return 0

    if schema is _NameAnswer:
        name = fields.get("name", "")
        if not name:
            return 0
        gen_lower = data.name.lower()
        # Weak condition: the expected leader name appears anywhere in the answer
        # (the model often appends context like "Napoleon Bonaparte, who...")
        if name.lower() in gen_lower:
            return 1
        # Fallback: any significant word (>3 chars) from the expected name
        for word in name.lower().split():
            if len(word) > 3 and word in gen_lower:
                return 1
        return 0

    if schema is _ReignAnswer:
        return int(
            _year_matches(fields.get("entry_year", ""), data.start_year)
            and _year_matches(fields.get("exit_year", ""),  data.end_year)
        )

    if schema is _YearAnswer:
        req = stmt["requires"][0]
        return int(_year_matches(fields.get(req, ""), data.year))

    if schema is _TenureAnswer:
        return int(str(data.years) == str(fields.get("tenure", "")).strip())

    return 0


def extract_gen_value(schema: type, gen_text: str) -> str:
    """Parse the JSON gen output and return a clean human-readable string for the CSV.

    Primary path: Pydantic model_validate_json.
    Fallback: raw json.loads — handles cases where Pydantic validation fails
    (e.g., the FSM emitted valid JSON but the value is outside a Field constraint)
    so the CSV is never silently empty when the JSON itself is readable.
    """
    if not gen_text:
        return ""

    def _raw_extract(text: str) -> str:
        import json as _json
        try:
            raw = _json.loads(text)
        except Exception:
            return ""
        if schema is _YearAnswer:
            v = raw.get("year")
        elif schema is _ReignAnswer:
            sy, ey = raw.get("start_year"), raw.get("end_year")
            return f"{sy}-{ey}" if sy is not None and ey is not None else ""
        elif schema is _TenureAnswer:
            v = raw.get("years")
        elif schema is _NameAnswer:
            v = raw.get("name")
        else:
            return ""
        return str(v) if v is not None else ""

    try:
        data = schema.model_validate_json(gen_text)
        if schema is _YearAnswer:
            return str(data.year)
        if schema is _ReignAnswer:
            return f"{data.start_year}-{data.end_year}"
        if schema is _TenureAnswer:
            return str(data.years)
        if schema is _NameAnswer:
            return data.name
    except Exception:
        # Pydantic rejected (range violation, type mismatch, etc.) — try raw JSON.
        val = _raw_extract(gen_text)
        if val:
            log.debug("extract_gen_value fallback for %s: %r → %r", schema.__name__, gen_text[:80], val)
        return val
    return ""


# ── Batch processing ──────────────────────────────────────────────────────────

def process_batch(llm, tok, indexed_rows: list[tuple[int, dict]]) -> list[dict]:
    """PPL-score and generate completions for all statements in the batch."""
    indices = [i for i, _ in indexed_rows]
    rows    = [r for _, r in indexed_rows]
    parsed  = [parse_fields(r) for r in rows]

    pairs, keys = [], []
    for offset, fields in enumerate(parsed):
        for stmt in STATEMENTS:
            triple = format_statement(stmt, fields)
            if triple:
                ctx, scored, gen_q = triple
                pairs.append((ctx, scored, gen_q, stmt["ans_schema"]))
                keys.append((offset, stmt["label"]))

    if pairs:
        ppls, gen_texts = score_batch(llm, tok, pairs)
    else:
        ppls, gen_texts = [], []

    ppl_map = {k: p for k, p in zip(keys, ppls)}
    gen_map = {k: g for k, g in zip(keys, gen_texts)}

    results = []
    for offset, (row_idx, row, fields) in enumerate(zip(indices, rows, parsed)):
        rec = {
            "row_idx":    row_idx,
            "polity":     row.get("territorynamehistorical", ""),
            "leader":     row.get("name_clean", ""),
            "entry_year": row.get("entrydateyear", ""),
            "exit_year":  row.get("exitdateyear", ""),
            "region":     row.get("region", ""),
        }
        stmt_ppls = []
        n_correct = 0
        for stmt in STATEMENTS:
            key    = (offset, stmt["label"])
            ppl    = ppl_map.get(key)
            gen    = gen_map.get(key)   # raw JSON string from the model
            scored = ppl is not None    # True iff statement was scoreable

            rec[f"ppl_{stmt['label']}"] = f"{ppl:.4f}" if scored else ""
            # Store the extracted value, not the raw JSON, so the CSV is human-readable
            rec[f"gen_{stmt['label']}"] = (
                extract_gen_value(stmt["ans_schema"], gen) if scored else ""
            )
            if scored:
                # check_correct still receives raw JSON so model_validate_json works
                correct = check_correct(stmt, fields, gen or "")
                rec[f"correct_{stmt['label']}"] = correct
                stmt_ppls.append(ppl)
                n_correct += correct
            else:
                rec[f"correct_{stmt['label']}"] = ""

        rec["n_stmts"]     = len(stmt_ppls)
        rec["geo_mean_ppl"] = (
            f"{math.exp(sum(math.log(p) for p in stmt_ppls) / len(stmt_ppls)):.4f}"
            if stmt_ppls else ""
        )
        rec["n_correct"] = n_correct if stmt_ppls else ""
        results.append(rec)

    return results


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_done(output_path: Path) -> set[int]:
    """Return row indices that were successfully scored (n_stmts > 0).

    Rows with n_stmts=0 are re-queued so they get another attempt if the
    source data later has values, or if a transient error produced empty scores.
    """
    if not output_path.exists():
        return set()
    with open(output_path) as f:
        return {
            int(r["row_idx"])
            for r in csv.DictReader(f)
            if r.get("row_idx") and int(r.get("n_stmts") or 0) > 0
        }


def has_scoreable_data(row: dict) -> bool:
    """Return True if at least one statement can be scored for this row."""
    fields = parse_fields(row)
    return any(format_statement(stmt, fields) is not None for stmt in STATEMENTS)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    llm, tok = load_model(args)

    with open(args.input) as f:
        all_rows = list(csv.DictReader(f))
    if args.test_n:
        all_rows = all_rows[: args.test_n]
        log.info(f"Test mode: {args.test_n} rows")
    log.info(f"Input rows: {len(all_rows)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)

    # Only queue rows that have scoreable data and haven't been successfully scored yet
    pending = [
        (idx, row) for idx, row in enumerate(all_rows)
        if idx not in done and has_scoreable_data(row)
    ]
    skipped = sum(1 for idx, row in enumerate(all_rows) if not has_scoreable_data(row))
    log.info(
        f"Total rows: {len(all_rows)}  |  already scored: {len(done)}  |  "
        f"no data (skipped): {skipped}  |  to score: {len(pending)}"
    )

    fh     = open(out_path, "a" if done else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
    if not done:
        writer.writeheader()

    buffer: list[tuple[int, dict]] = []
    processed = 0

    try:
        with tqdm(total=len(pending), unit="row", desc="Scoring") as pbar:
            for idx, row in pending:
                buffer.append((idx, row))

                if len(buffer) < args.batch_size:
                    continue

                results = process_batch(llm, tok, buffer)
                writer.writerows(results)
                fh.flush()
                pbar.update(len(buffer))
                processed += len(buffer)
                buffer = []

            if buffer:
                results = process_batch(llm, tok, buffer)
                writer.writerows(results)
                fh.flush()
                pbar.update(len(buffer))
                processed += len(buffer)

    finally:
        fh.close()

    log.info(f"Done. Scored {processed} new rows → {out_path}")


if __name__ == "__main__":
    main()
