#!/usr/bin/env python3
"""
Aggregate per-leader PPL scores to polity level.
Reads the output of perplexity_probe.py and writes a polity-level summary.

Usage:
    python test/aggregate_ppl.py \
        --input  data/plt_leaders_ppl.csv \
        --output data/plt_polities_ppl.csv
"""

import csv
import math
import argparse
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate leader-level PPL to polity level")
    p.add_argument("--input",  required=True,  help="Leader-level PPL CSV from perplexity_probe.py")
    p.add_argument("--output", required=True,  help="Output polity-level summary CSV")
    p.add_argument("--min-leaders", type=int, default=1,
                   help="Minimum leaders per polity to include in output (default: 1)")
    return p.parse_args()


def geo_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    args = parse_args()

    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    # Detect which ppl_* columns are present
    ppl_cols = [k for k in rows[0].keys() if k.startswith("ppl_")]

    # Group by polity
    polity_data: dict[str, dict] = defaultdict(lambda: {
        "region": "",
        "n_leaders": 0,
        "geo_mean_ppls": [],           # per-leader geo_mean_ppl values
        **{col: [] for col in ppl_cols},
    })

    for row in tqdm(rows, desc="Aggregating", unit="row"):
        polity = row["polity"]
        if not polity:
            continue
        g = polity_data[polity]
        g["region"] = g["region"] or row.get("region", "")
        g["n_leaders"] += 1

        if row.get("geo_mean_ppl"):
            g["geo_mean_ppls"].append(float(row["geo_mean_ppl"]))

        for col in ppl_cols:
            if row.get(col):
                g[col].append(float(row[col]))

    # Build output rows
    out_fields = (
        ["polity", "region", "n_leaders"]
        + [f"geo_mean_{col}" for col in ppl_cols]
        + ["geo_mean_ppl"]
    )

    out_rows = []
    for polity, g in sorted(polity_data.items()):
        if g["n_leaders"] < args.min_leaders:
            continue
        rec = {
            "polity":     polity,
            "region":     g["region"],
            "n_leaders":  g["n_leaders"],
            "geo_mean_ppl": (
                f"{geo_mean(g['geo_mean_ppls']):.4f}" if g["geo_mean_ppls"] else ""
            ),
        }
        for col in ppl_cols:
            vals = g[col]
            rec[f"geo_mean_{col}"] = f"{geo_mean(vals):.4f}" if vals else ""
        out_rows.append(rec)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} polity-level rows → {args.output}")


if __name__ == "__main__":
    main()
