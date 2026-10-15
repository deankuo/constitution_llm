#!/usr/bin/env python3
"""Sample leader rows from the classic cases for the general-constraint pilot.

Python port of the case windows in stats/classic_cases.R (that file stays the
canonical definition for the R side; keep the two in sync). Each case keeps at
most --per-case rows, one row per distinct leader, drawn with a fixed seed.
The output keeps every input column and adds `case` and `original_row_idx`
(position in the input file) so a batch build can use --original-idx-col and
the results can be joined back to stats/plt_irt.rds by `id`.

  python src/select_classic_cases.py --input data/plt_constraints.csv \\
      --output data/temp/classic_cases_sample.csv --per-case 4 --seed 42
"""
import argparse

import pandas as pd

# case -> (polity_name variants, start, end); tenure kept if it OVERLAPS the window
CLASSIC_CASE_SPEC = {
    "Athens (democracy)": (["Athens", "Athenai Attika"], -508, -322),
    "Rome (early Republic)": (["Roman Republic"], -509, -134),
    "Rome (late Republic)": (["Roman Republic"], -133, -27),
    "Roman Empire (Principate)": (["Roman Empire"], -27, 284),
    "Rome (Dominate)": (["Roman Empire"], 284, 476),
    "Sparta": (["Kingdom Of Sparta"], -785, -192),
    "Qin": (["Qin Dynasty", "Before Qin"], -350, -206),
    "Achaemenid Persia": (["Achaemenid Empire", "Achaemenid Kingdom"], -559, -330),
    "Venice (pre-Serrata)": (["Venice"], 697, 1297),
    "Venice (post-Serrata)": (["Venice"], 1298, 1797),
    "Icelandic Commonwealth": (["Iceland"], 930, 1262),
    "Holy Roman Empire": (["Holy Roman Empire", "Holy_Roman_Empire", "Germano Roman Empire"], 800, 1806),
    "Abbasid Caliphate": (["The Abbasid Caliphate", "The Caliphate"], 750, 1258),
    "Heian Japan": (["Japan"], 794, 1185),
    "Dutch Republic": (["Netherlands", "The Netherlands", "United Provinces"], 1581, 1795),
    "Poland-Lithuania": (["Poland-Lithuania", "Poland", "Lithuania"], 1569, 1795),
    "France (Louis XIV)": (["France"], 1643, 1715),
    "Tokugawa Japan": (["Japan", "Tokugawa"], 1603, 1868),
    "Ottoman Empire": (["Ottoman", "Ottoman Empire"], 1300, 1922),
    "United Kingdom (1900)": (["United Kingdom", "Great Britain", "England"], 1885, 1914),
    "United States (1900)": (["United States Of America"], 1885, 1914),
    "Soviet Union (Stalin)": (["Soviet Union", "Soviet Russian Republic"], 1924, 1953),
    "Nazi Germany": (["Germany", "German_Reich"], 1933, 1945),
    "Sweden (contemporary)": (["Sweden"], 1990, 2020),
}


def select_classic_cases(df: pd.DataFrame, per_case: int = 4, seed: int = 42) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    picked = []
    used_ids: set = set()
    for case, (polities, start, end) in CLASSIC_CASE_SPEC.items():
        m = (df["polity_name"].isin(polities)
             & (df["leader_last_year"] >= start) & (df["leader_first_year"] <= end)
             & ~df["id"].isin(used_ids))
        sub = df[m].drop_duplicates("leader_name")
        if sub.empty:
            continue
        sub = sub.sample(n=min(per_case, len(sub)), random_state=seed)
        sub = sub.assign(case=case, original_row_idx=sub.index)
        used_ids.update(sub["id"].tolist())
        picked.append(sub)
    out = pd.concat(picked, ignore_index=True) if picked else df.iloc[0:0].assign(case=None, original_row_idx=None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-case", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    out = select_classic_cases(df, args.per_case, args.seed)
    out.to_csv(args.output, index=False)
    print(out.groupby("case", sort=False).size().to_string())
    print(f"\n{len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
