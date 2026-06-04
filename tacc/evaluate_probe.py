#!/usr/bin/env python3
"""
Evaluate knowledge probe output against ground truth.

Reads the CSV produced by perplexity_probe.py and computes per-question accuracy
using configurable buffer zones for numeric answers and partial matching for names.

Ground truth → predicted column mapping:
  entry_year  → gen_entry_year
  exit_year   → gen_exit_year
  tenure      → gen_tenure_from_entry  (context: entry year given)
  tenure      → gen_tenure_from_exit   (context: exit year given)
  leader      → gen_leader_name
  polity      → gen_polity_name

Correctness rules:
  year questions   : |predicted - ground_truth| <= --year-window   (default ±3)
  tenure questions : |predicted - ground_truth| <= --tenure-window  (default ±5)
  name / polity    : ground truth string (case-insensitive) appears anywhere in the
                     predicted string, or any significant word (>3 chars) matches

Usage:
  python evaluate_probe.py --input probe_output.csv
  python evaluate_probe.py --input probe_output.csv --year-window 5 --tenure-window 10
"""

import csv
import argparse

# ── Evaluation specs ──────────────────────────────────────────────────────────
# Maps each gen_* column to its ground truth column and comparison type.

EVAL_SPECS = [
    {"gen_col": "gen_entry_year",        "gt_col": "entry_year", "type": "year"},
    {"gen_col": "gen_exit_year",         "gt_col": "exit_year",  "type": "year"},
    {"gen_col": "gen_tenure_from_entry", "gt_col": "tenure",     "type": "tenure"},
    {"gen_col": "gen_tenure_from_exit",  "gt_col": "tenure",     "type": "tenure"},
    {"gen_col": "gen_leader_name",       "gt_col": "leader",     "type": "name"},
    {"gen_col": "gen_polity_name",       "gt_col": "polity",     "type": "name"},
]


# ── Comparison helpers ────────────────────────────────────────────────────────

def _numeric_correct(gt: str, pred: str, window: int) -> bool:
    try:
        return abs(int(pred) - int(gt)) <= window
    except (ValueError, TypeError):
        return False


def _name_correct(gt: str, pred: str) -> bool:
    gt_lower   = gt.lower().strip()
    pred_lower = pred.lower().strip()
    if gt_lower in pred_lower:
        return True
    for word in gt_lower.split():
        if len(word) > 3 and word in pred_lower:
            return True
    return False


def is_correct(spec: dict, gt: str, pred: str, year_window: int, tenure_window: int):
    """Return 1 (correct), 0 (wrong), or '' (missing gt or prediction)."""
    if not gt.strip() or not pred.strip():
        return ""
    if spec["type"] == "year":
        return int(_numeric_correct(gt, pred, year_window))
    if spec["type"] == "tenure":
        return int(_numeric_correct(gt, pred, tenure_window))
    if spec["type"] == "name":
        return int(_name_correct(gt, pred))
    return ""


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate knowledge probe output against ground truth"
    )
    p.add_argument("--input",         required=True,
                   help="Probe output CSV (from perplexity_probe.py)")
    p.add_argument("--year-window",   type=int, default=3,
                   help="Tolerance ±N years for entry/exit year questions (default: 3)")
    p.add_argument("--tenure-window", type=int, default=5,
                   help="Tolerance ±N years for tenure questions (default: 5)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.input, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No rows in input.")
        return

    # Check which gen_* columns are present (probe may have been run on a subset)
    available = {s["gen_col"] for s in EVAL_SPECS if s["gen_col"] in rows[0]}
    if not available:
        print("No gen_* columns found. Is this a probe output CSV?")
        return

    # Per-question accumulators: {gen_col: [n_correct, n_evaluated]}
    counts: dict[str, list[int]] = {s["gen_col"]: [0, 0] for s in EVAL_SPECS}

    for row in rows:
        for spec in EVAL_SPECS:
            if spec["gen_col"] not in available:
                continue
            gt   = row.get(spec["gt_col"], "")
            pred = row.get(spec["gen_col"], "")
            c    = is_correct(spec, gt, pred, args.year_window, args.tenure_window)
            if c != "":
                counts[spec["gen_col"]][0] += c
                counts[spec["gen_col"]][1] += 1

    # ── Accuracy summary ───────────────────────────────────────────────────
    print(
        f"\nEvaluation  "
        f"(year ±{args.year_window} | tenure ±{args.tenure_window} | "
        f"name/polity: partial match)\n"
    )
    col_w    = max(len(s["gen_col"]) for s in EVAL_SPECS if s["gen_col"] in available)
    divider  = "-" * (col_w + 34)
    print(f"{'Question':<{col_w}}   {'Correct':>7}   {'Total':>5}   Accuracy")
    print(divider)

    overall_correct = overall_total = 0
    for spec in EVAL_SPECS:
        if spec["gen_col"] not in available:
            continue
        n_correct, n_total = counts[spec["gen_col"]]
        acc = f"{n_correct / n_total:.1%}" if n_total else "n/a"
        print(f"{spec['gen_col']:<{col_w}}   {n_correct:>7}   {n_total:>5}   {acc}")
        overall_correct += n_correct
        overall_total   += n_total

    print(divider)
    overall_acc = f"{overall_correct / overall_total:.1%}" if overall_total else "n/a"
    print(f"{'Overall':<{col_w}}   {overall_correct:>7}   {overall_total:>5}   {overall_acc}\n")



if __name__ == "__main__":
    main()
