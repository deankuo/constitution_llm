"""Pilot: constitution-bypass question (conditional on constitution == 2).

Two prompt versions x 3 reps per case, no search, sync Gemini calls.
Version A = professor's reframed wording verbatim ("letter AND spirit").
Version B = A + "letter OR spirit" + one sentence on formally legal hollowing-out.
"""
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
from utils.json_parser import parse_json_response  # noqa: E402

MODEL = "gemini-3.1-pro-preview"
N_REPS = 3
OUT = "data/temp/pilot_constitution_bypass.csv"

# id -> (group, provisional expected label)
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
    # expected 0
    4975: ("expected_0", "0"),    # Eisenhower
    4986: ("expected_0", "0"),    # Obama
    13926: ("expected_0", "0"),   # Mitterrand
    14293: ("expected_0", "0"),   # Adenauer
    8154: ("expected_0", "0"),    # Nehru
    # borderline (letter vs spirit, emergency powers)
    14271: ("borderline", "?"),   # Ebert (Art. 48, 1923)
    14279: ("borderline", "?"),   # Hindenburg 1925-34
    8167: ("borderline", "?"),    # Indira Gandhi 1966-77 (Emergency)
    4957: ("borderline", "?"),    # Lincoln (habeas corpus)
    4978: ("borderline", "?"),    # Nixon
    # pre-modern / gate-quality
    140917: ("premodern", "?"),   # Telepinu
    117440: ("premodern", "?"),   # Solon
    117519: ("premodern", "?"),   # Pericles
    141218: ("premodern", "?"),   # Josiah (Deuteronomic Code)
    145: ("premodern", "?"),      # Marcos de Torres y Rueda, New Spain governor
}

QUESTION_A = (
    "If there is a written constitution with laws for governance (Constitutionalism=2), "
    "are there reasons to suppose that constitutional limits on the executive are ignored "
    "or bypassed in practice (in violation of the letter and spirit of the law)? "
    "Code 1 if yes, 0 otherwise."
)
QUESTION_B = (
    "If there is a written constitution with laws for governance (Constitutionalism=2), "
    "are there reasons to suppose that constitutional limits on the executive are ignored "
    "or bypassed in practice (in violation of the letter or spirit of the law)? "
    "Code 1 if yes, 0 otherwise. This includes cases where constitutional limits are "
    "hollowed out through formally legal procedures (such as delegating legislative power "
    "to the executive, or prolonged use of emergency powers) in violation of the spirit "
    "of the constitution."
)

SYSTEM_TEMPLATE = """You are a political scientist and historian coding executive constraints for historical leaders.

A prior coding step has determined that this polity had a written constitution with laws for governance \
(Constitutionalism = 2) during the leader's tenure: a written document that includes rules of governance \
and limits on the authority of the executive.

## Question

{question}

## Notes

- Focus on THIS leader's tenure, not the polity's history as a whole.
- The document list you receive may also include ordinary codes of law (civil, criminal, etc.). \
Judge only the constitutional document(s) that set rules of governance and limits on the executive.

## Output Requirements

Provide a JSON object with exactly these fields:
- "constitution_bypass": Must be exactly "0" or "1" (string)
- "reasoning": Your step-by-step reasoning (string)
- "confidence_score": Integer from 1 to 100

Respond with ONLY a JSON object, starting with {{ and ending with }}.
"""

USER_TEMPLATE = """**Polity:** {polity}
**Leader:** {name}
**Tenure Period:** {start_year}-{end_year}
**Constitutional document(s) identified:** {doc_name}
**Year(s):** {doc_year}

Respond with a single JSON object:
{{"constitution_bypass": "0 or 1", "reasoning": "your analysis", "confidence_score": 1-100}}
"""

PROMPTS = {
    "A": SYSTEM_TEMPLATE.format(question=QUESTION_A),
    "B": SYSTEM_TEMPLATE.format(question=QUESTION_B),
}


def fmt_year(y):
    return "present" if pd.isna(y) else str(int(y))


def run_one(llm, row, version, rep):
    user = USER_TEMPLATE.format(
        polity=row.polity_name,
        name=row.leader_name,
        start_year=fmt_year(row.leader_first_year),
        end_year=fmt_year(row.leader_last_year),
        doc_name=row.constitution_document_name,
        doc_year=row.constitution_year,
    )
    out = {"id": row.id, "version": version, "rep": rep,
           "pred": None, "confidence": None, "reasoning": None, "error": None}
    try:
        resp = llm.call(PROMPTS[version], user, max_tokens=32768)
        parsed = parse_json_response(resp.content)
        out["pred"] = str(parsed.get("constitution_bypass")) if parsed else None
        out["confidence"] = parsed.get("confidence_score") if parsed else None
        out["reasoning"] = parsed.get("reasoning") if parsed else resp.content[:500]
    except Exception as e:  # keep going; report nulls explicitly
        out["error"] = repr(e)[:300]
    return out


def main():
    df = pd.read_csv("data/plt_constraints.csv", low_memory=False)
    df = df[df.id.isin(CASES)].drop_duplicates("id").set_index("id", drop=False)
    missing = set(CASES) - set(df.index)
    assert not missing, f"missing ids: {missing}"
    assert (df.constitution_prediction == 2).all(), "all pilot rows must pass the gate"

    llm = create_llm(MODEL, {"gemini": os.getenv("GEMINI_API_KEY")})  # no grounding
    jobs = [(df.loc[i], v, r) for i in CASES for v in PROMPTS for r in range(1, N_REPS + 1)]
    results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(run_one, llm, *j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if k % 20 == 0:
                print(f"{k}/{len(jobs)} done", flush=True)

    res = pd.DataFrame(results)
    meta = df[["id", "polity_name", "leader_name", "leader_first_year", "leader_last_year",
               "constitution_document_name", "constitution_year"]].reset_index(drop=True)
    meta["group"] = meta.id.map(lambda i: CASES[i][0])
    meta["expected"] = meta.id.map(lambda i: CASES[i][1])
    res = res.merge(meta, on="id").sort_values(["group", "id", "version", "rep"])
    res.to_csv(OUT, index=False)
    print(f"wrote {OUT}; nulls={res.pred.isna().sum()} errors={res.error.notna().sum()}")


if __name__ == "__main__":
    main()
