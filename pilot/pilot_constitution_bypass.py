"""Pilot: constitution-bypass question (conditional on constitution == 2).

Two prompt versions x 3 reps per case, no search, sync Gemini calls.
Both versions use the final question wording from prompts/constitution_bypass.py
(the Nixon / Lincoln "does not count" examples are part of the question itself):
  C = question only
  D = question + early/middle-era examples (Sulla/Caesar, Charles I vs Cicero, Faliero)
The prompt module is shared with `build_batch_jsonl.py --task constitution_bypass`,
so what is tested here is exactly what the batch run sends.

Earlier pilot (versions A/B, "letter and/or spirit" wording): Lincoln was coded 1
in 6/6 runs against the professor's judgement of 0; Nixon and Ebert were unstable.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

from models.llm_clients import create_llm  # noqa: E402
from prompts.constitution_bypass import get_prompt, select_type2_docs  # noqa: E402
from utils.json_parser import parse_json_response  # noqa: E402

MODEL = "gemini-3.1-pro-preview"
N_REPS = 3
OUT = "data/temp/pilot_constitution_bypass_v2.csv"
ALL_VERSIONS = {"C": False, "D": True}  # version -> with_examples
# Set from CLI in main(); defaults reproduce the original C-vs-D pilot with reasoning.
VERSIONS = dict(ALL_VERSIONS)
REASONING = True

# id -> (group, provisional expected label)
# NOTE: Charles I and Marino Faliero appear in the version-D examples, so their
# rows test recall of the example, not generalisation.
CASES = {
    # expected 1
    14285: ("expected_1", "1"),   # Hitler 1933-34 (Weimar)
    2459: ("expected_1", "1"),    # Stalin 1941-46 (1936 constitution)
    2501: ("expected_1", "1"),    # Brezhnev 1966-82
    12512: ("expected_1", "1"),   # Pinochet
    286: ("expected_1", "1"),     # Porfirio Diaz 1884-1910
    15953: ("expected_1", "1"),   # Mussolini
    8280: ("expected_1", "1"),    # Kim Il Sung
    15031: ("expected_1", "1"),   # Saddam Hussein
    6849: ("expected_1", "1"),    # Fujimori
    33924: ("expected_1", "1"),   # Ceausescu 1965-67
    7791: ("expected_1", "1"),    # Peron 1946-55
    13761: ("expected_1", "1"),   # Napoleon III (1851 coup) - sanity check
    19898: ("expected_1", "1"),   # Cromwell 1653-58
    19872: ("expected_1", "1"),   # Charles I 1625-49 (Personal Rule) - in D examples
    19918: ("expected_1", "1"),   # James II 1685-89 (dispensing power)
    13730: ("expected_1", "1"),   # Charles X of France (July Ordinances 1830)
    # expected 0
    4975: ("expected_0", "0"),    # Eisenhower
    4986: ("expected_0", "0"),    # Obama
    13926: ("expected_0", "0"),   # Mitterrand
    14293: ("expected_0", "0"),   # Adenauer
    8154: ("expected_0", "0"),    # Nehru
    4957: ("expected_0", "0"),    # Lincoln (habeas corpus) - professor: 0; in question text
    4978: ("expected_0", "0"),    # Nixon - professor: 0; in question text
    66935: ("expected_0", "0"),   # Francesco Foscari, Venice 1423-57 (deposed BY the councils)
    66925: ("expected_0", "0"),   # Marino Faliero 1354-55 (failed coup) - in D examples
    # borderline (letter vs spirit, emergency powers, court defiance)
    14271: ("borderline", "?"),   # Ebert (Art. 48, 1923)
    14279: ("borderline", "?"),   # Hindenburg 1925-34
    8167: ("borderline", "?"),    # Indira Gandhi 1966-77 (Emergency)
    4948: ("borderline", "?"),    # Andrew Jackson (Worcester v. Georgia, Bank veto)
    4973: ("borderline", "?"),    # FDR (court-packing failed; internment)
    65002: ("borderline", "?"),   # Sigismund III Vasa (rokosz of 1606-08)
    # pre-modern / gate-quality
    140917: ("premodern", "?"),   # Telepinu
    117440: ("premodern", "?"),   # Solon
    117519: ("premodern", "?"),   # Pericles
    117523: ("premodern", "?"),   # Cleon
    141218: ("premodern", "?"),   # Josiah (Deuteronomic Code)
    145: ("premodern", "?"),      # Marcos de Torres y Rueda, New Spain governor
}


def fmt_year(y):
    return None if pd.isna(y) else int(y)


def run_one(llm, row, version, rep):
    doc_name, doc_year = select_type2_docs(
        row.constitution_document_name, row.constitution_year, row.constitution_document_types
    )
    sys_p, usr_p = get_prompt(
        row.polity_name, row.leader_name,
        fmt_year(row.leader_first_year), fmt_year(row.leader_last_year),
        doc_name, doc_year, reasoning=REASONING, with_examples=VERSIONS[version],
    )
    out = {"id": row.id, "version": version, "rep": rep, "docs_shown": doc_name,
           "pred": None, "confidence": None, "reasoning": None, "error": None}
    try:
        resp = llm.call(sys_p, usr_p, max_tokens=32768)
        parsed = parse_json_response(resp.content)
        out["pred"] = str(parsed.get("constitution_bypass")) if parsed else None
        out["confidence"] = parsed.get("confidence_score") if parsed else None
        out["reasoning"] = parsed.get("reasoning") if parsed else resp.content[:500]
    except Exception as e:  # keep going; report nulls explicitly
        out["error"] = repr(e)[:300]
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", choices=sorted(ALL_VERSIONS), default=sorted(ALL_VERSIONS),
                    help="prompt versions to run (C = question only, D = question + examples; default both)")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="drop the reasoning field from the prompt/output (prediction + confidence only)")
    ap.add_argument("--reps", type=int, default=N_REPS, help=f"repetitions per case (default {N_REPS})")
    ap.add_argument("--out", default=OUT, help=f"output CSV (default {OUT})")
    return ap.parse_args()


def main():
    global VERSIONS, REASONING, N_REPS, OUT
    args = parse_args()
    VERSIONS = {v: ALL_VERSIONS[v] for v in args.versions}
    REASONING = not args.no_reasoning
    N_REPS, OUT = args.reps, args.out
    print(f"model={MODEL} versions={list(VERSIONS)} reasoning={REASONING} reps={N_REPS} out={OUT}", flush=True)

    df = pd.read_csv("data/plt_constraints.csv", low_memory=False)
    df = df[df.id.isin(CASES)].drop_duplicates("id").set_index("id", drop=False)
    missing = set(CASES) - set(df.index)
    assert not missing, f"missing ids: {missing}"
    assert (df.constitution_prediction == 2).all(), "all pilot rows must pass the gate"

    llm = create_llm(MODEL, {"gemini": os.getenv("GEMINI_API_KEY")})  # no grounding
    jobs = [(df.loc[i], v, r) for i in CASES for v in VERSIONS for r in range(1, N_REPS + 1)]
    results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(run_one, llm, *j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if k % 20 == 0:
                print(f"{k}/{len(jobs)} done", flush=True)

    res = pd.DataFrame(results)
    meta = df[["id", "polity_name", "leader_name", "leader_first_year", "leader_last_year",
               "constitution_document_name", "constitution_year",
               "constitution_document_types"]].reset_index(drop=True)
    meta["group"] = meta.id.map(lambda i: CASES[i][0])
    meta["expected"] = meta.id.map(lambda i: CASES[i][1])
    res = res.merge(meta, on="id").sort_values(["group", "id", "version", "rep"])
    res.to_csv(OUT, index=False)
    print(f"wrote {OUT}; nulls={res.pred.isna().sum()} errors={res.error.notna().sum()}")
    summary = (res.assign(pred=res.pred.astype(str))
               .groupby(["group", "leader_name", "version"]).pred
               .agg("".join).unstack("version"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
