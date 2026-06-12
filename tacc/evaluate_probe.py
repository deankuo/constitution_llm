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
  python evaluate_probe.py --input a.csv b.csv --labels ModelA ModelB
  python evaluate_probe.py --input probe_output.csv --year-window 5 --tenure-window 10
"""

import csv
import json
import argparse
from pathlib import Path
import sys

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
        return abs(round(float(pred)) - round(float(gt))) <= window
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
    p.add_argument("--input",         required=True, nargs="+",
                   help="One or more probe output CSVs (from perplexity_probe.py)")
    p.add_argument("--labels",        nargs="+",
                   help="Display names for each input file (defaults to filename stems)")
    p.add_argument("--year-window",   type=int, default=3,
                   help="Tolerance ±N years for entry/exit year questions (default: 3)")
    p.add_argument("--tenure-window", type=int, default=5,
                   help="Tolerance ±N years for tenure questions (default: 5)")
    p.add_argument("--output", default=None,
                   help="Path to save evaluation results as JSON (e.g. results.json)")
    return p.parse_args()


# ── Per-file evaluation ───────────────────────────────────────────────────────

def evaluate_file(path: str, year_window: int, tenure_window: int):
    """Return (available_cols, counts_dict, augmented_rows) for one CSV file."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return set(), {}, []

    available = {s["gen_col"] for s in EVAL_SPECS if s["gen_col"] in rows[0]}
    counts: dict[str, list[int]] = {s["gen_col"]: [0, 0] for s in EVAL_SPECS}

    for row in rows:
        for spec in EVAL_SPECS:
            if spec["gen_col"] not in available:
                continue
            gt   = row.get(spec["gt_col"], "")
            pred = row.get(spec["gen_col"], "")
            c    = is_correct(spec, gt, pred, year_window, tenure_window)
            row[f"correct_{spec['gen_col']}"] = c
            if c != "":
                counts[spec["gen_col"]][0] += c
                counts[spec["gen_col"]][1] += 1

        valid = [row[f"correct_{s['gen_col']}"] for s in EVAL_SPECS
                 if f"correct_{s['gen_col']}" in row and row[f"correct_{s['gen_col']}"] != ""]
        row["avg_accuracy"] = sum(valid) / len(valid) if valid else ""

    return available, counts, rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    labels = args.labels or [Path(p).stem for p in args.input]
    if len(labels) != len(args.input):
        print(f"Error: --labels count ({len(labels)}) must match --input count ({len(args.input)}).")
        return

    results = []
    all_available: set[str] = set()
    for path, label in zip(args.input, labels):
        available, counts, augmented_rows = evaluate_file(path, args.year_window, args.tenure_window)
        if not available:
            print(f"Warning: no gen_* columns or no rows in {path!r} — skipping.")
            continue
        results.append((label, available, counts, augmented_rows, path))
        all_available |= available

    if not results:
        print("No valid input files.")
        return

    if args.output:
        if len(results) > 1:
            print("Error: --output only supports a single --input file.")
            sys.exit(1)

        ext = Path(args.output).suffix.lower()
        if ext not in (".json", ".csv"):
            print(f"Error: --output must end in .json or .csv (got {ext!r})")
            sys.exit(1)

        _, _, _, augmented_rows, _ = results[0]

        if ext == ".csv":
            fieldnames = list(augmented_rows[0].keys())
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(augmented_rows)
        else:
            Path(args.output).write_text(json.dumps(augmented_rows, indent=2))

        print(f"Results saved to {args.output}")

    print(
        f"\nEvaluation  "
        f"(year ±{args.year_window} | tenure ±{args.tenure_window} | "
        f"name/polity: partial match)\n"
    )

    # ── Single-file: original compact layout ──────────────────────────────
    if len(results) == 1:
        label, available, counts, _, _ = results[0]
        col_w   = max(len(s["gen_col"]) for s in EVAL_SPECS if s["gen_col"] in available)
        divider = "-" * (col_w + 34)
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
        return

    # ── Multi-file: side-by-side accuracy comparison ───────────────────────
    col_w  = max(len(s["gen_col"]) for s in EVAL_SPECS if s["gen_col"] in all_available)
    # "100.0% (9999)" = 13 chars; labels may be longer
    cell_w = max(max(len(lbl) for lbl, *_ in results), 13)
    n_cols = len(results)
    divider = "-" * (col_w + 3 + n_cols * (cell_w + 3))

    def fmt_cell(n_correct: int, n_total: int) -> str:
        acc = f"{n_correct / n_total:.1%}" if n_total else "n/a"
        return f"{acc} ({n_total})"

    header = f"{'Question':<{col_w}}   " + "   ".join(f"{lbl:>{cell_w}}" for lbl, *_ in results)
    print(header)
    print(divider)

    overall: list[list[int]] = [[0, 0] for _ in results]
    for spec in EVAL_SPECS:
        if spec["gen_col"] not in all_available:
            continue
        cells = []
        for i, (label, available, counts, _, _) in enumerate(results):
            if spec["gen_col"] not in available:
                cells.append(f"{'—':>{cell_w}}")
            else:
                n_correct, n_total = counts[spec["gen_col"]]
                cells.append(f"{fmt_cell(n_correct, n_total):>{cell_w}}")
                overall[i][0] += n_correct
                overall[i][1] += n_total
        print(f"{spec['gen_col']:<{col_w}}   " + "   ".join(cells))

    print(divider)
    overall_cells = [f"{fmt_cell(n_correct, n_total):>{cell_w}}" for n_correct, n_total in overall]
    print(f"{'Overall':<{col_w}}   " + "   ".join(overall_cells))
    print()


if __name__ == "__main__":
    main()
