"""Token usage and USD cost of a Gemini Batch API run.

Works on the raw output lines a batch job writes ({"key", "response"} or
{"key", "error"}); usageMetadata is camelCase because the file is the raw REST
payload, not an SDK object. Pricing comes from utils.cost_tracker.PRICING.
Generalises the ad-hoc cell in main.ipynb.
"""
import json
from typing import Iterable, Optional

from utils.cost_tracker import CostTracker

USAGE_KEYS = ("prompt", "candidates", "thoughts", "cached", "total")
_FIELD = {"prompt": "promptTokenCount", "candidates": "candidatesTokenCount",
          "thoughts": "thoughtsTokenCount", "cached": "cachedContentTokenCount",
          "total": "totalTokenCount"}


def new_usage_acc() -> dict:
    acc = {k: 0 for k in USAGE_KEYS}
    acc["n_responses"] = 0
    return acc


def extract_usage(record: dict) -> Optional[dict]:
    """Usage of one output line, or None for an error line / missing response."""
    if not isinstance(record, dict) or "error" in record:
        return None
    resp = record.get("response")
    if not isinstance(resp, dict):
        return None
    um = resp.get("usageMetadata") or {}
    return {k: int(um.get(f, 0) or 0) for k, f in _FIELD.items()}


def add_usage(acc: dict, usage: Optional[dict]) -> None:
    if usage is None:
        return
    for k in USAGE_KEYS:
        acc[k] += usage.get(k, 0)
    acc["n_responses"] += 1


def usage_from_lines(lines: Iterable[str]) -> dict:
    acc = new_usage_acc()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        add_usage(acc, extract_usage(json.loads(line)))
    return acc


def thoughts_included_in_candidates(acc: dict) -> bool:
    """True when totalTokenCount shows thoughts are already inside candidates.

    Gemini reports total = prompt + candidates + thoughts when they are separate.
    If total is within 1% of prompt + candidates instead, billing thoughts again
    would double count them.
    """
    total = acc.get("total", 0)
    if not total or not acc.get("thoughts"):
        return False
    separate = acc["prompt"] + acc["candidates"] + acc["thoughts"]
    return abs(separate - total) >= total * 0.01


def cost_summary(acc: dict, model: str, batch_discount: float = 0.5) -> dict:
    """USD cost for the accumulated usage (input = prompt minus cached)."""
    inside = thoughts_included_in_candidates(acc)
    cost = CostTracker().calculate_cost(
        model,
        input_tokens=max(0, acc["prompt"] - acc["cached"]),
        output_tokens=acc["candidates"],
        cached_tokens=acc["cached"],
        thinking_tokens=0 if inside else acc["thoughts"],
        batch_discount=batch_discount,
    )
    return {
        "model": model,
        "usage": {k: acc[k] for k in USAGE_KEYS},
        "n_responses": acc["n_responses"],
        "batch_discount": batch_discount,
        "thoughts_included_in_candidates": inside,
        "cost_usd": round(cost, 4),
    }


def format_cost(summary: dict) -> str:
    u = summary["usage"]
    note = " (thoughts counted inside candidates)" if summary["thoughts_included_in_candidates"] else ""
    return (f"Usage over {summary['n_responses']:,} responses: prompt {u['prompt']:,} | "
            f"candidates {u['candidates']:,} | thoughts {u['thoughts']:,} | cached {u['cached']:,}{note}\n"
            f"Estimated cost: ${summary['cost_usd']:.2f} USD "
            f"({summary['model']}, batch discount x{summary['batch_discount']})")
