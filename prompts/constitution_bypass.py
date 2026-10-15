"""Constitution-bypass question: conditional on constitution == 2.

Shared by pilot/pilot_constitution_bypass.py (sync) and the
``--task constitution_bypass`` batch builder so both send identical prompts.

Question wording is researcher-maintained (Gerring); do not edit without asking.
Row inputs come from a prior constitution run: ``constitution_document_name``,
``constitution_year`` and ``constitution_document_types`` are parallel
semicolon-separated lists, and only the type-2 entries (full constitutions)
are shown to the model.
"""
import re

import pandas as pd

# Final wording. The two late-era "does not count" examples (Nixon, Lincoln)
# are part of the question itself.
QUESTION = (
    "If there is a written constitution with laws for governance (Constitutionalism=2), "
    "are there reasons to suppose that constitutional limits on the executive are "
    "systematically ignored or bypassed in practice (in violation of the letter or spirit "
    "of the law)? Do not include minor or occasional violations such as Nixon's dirty "
    "tricks during Watergate or Lincoln's wartime suspension of habeas corpus. "
    "Code 1 if yes, 0 otherwise."
)

# Early- and middle-era anchors, one pole each, kept to two sentences so the
# prompt stays short. Toggled by get_prompt(with_examples=...).
EXAMPLES = (
    "Systematic bypass (code 1): Sulla's or Caesar's dictatorship in the late Roman Republic; "
    "Charles I's Personal Rule (1629-1640) governing without Parliament in violation of the "
    "Petition of Right. Not systematic (code 0): Cicero's execution of the Catiline "
    "conspirators without trial (63 BC); Marino Faliero's failed coup in Venice (1355), after "
    "which the Promissione Ducale continued to bind the doge."
)

SYSTEM_PROMPT_TEMPLATE = """You are a political scientist and historian coding executive constraints for historical leaders.

A prior coding step has determined that this polity had a written constitution with laws for governance \
(Constitutionalism = 2) during the leader's tenure: a written document that includes rules of governance \
and limits on the authority of the executive.

## Question

{question}
{examples}
## Notes

- Focus on THIS leader's tenure, not the polity's history as a whole.
- Judge the constitutional document(s) listed for this leader.

## Output Requirements

Provide a JSON object with exactly these fields:
- "constitution_bypass": Must be exactly "0" or "1" (string)
- "reasoning": Your step-by-step reasoning (string)
- "confidence_score": Integer from 1 to 100

Respond with ONLY a JSON object, starting with {{ and ending with }}.
"""

USER_PROMPT_TEMPLATE = """**Polity:** {polity}
**Leader:** {name}
**Tenure Period:** {start_year}-{end_year}
**Constitutional document(s) identified:** {doc_name}
**Year(s):** {doc_year}

Respond with a single JSON object:
{{"constitution_bypass": "0 or 1", "reasoning": "your analysis", "confidence_score": 1-100}}
"""


def _split(value) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [p.strip() for p in str(value).split(";")]


def select_type2_docs(doc_name, doc_year, doc_types) -> tuple[str, str]:
    """Keep only the type-2 entries of the parallel name/year/type lists.

    Falls back to the full lists when types are missing, the lists are not
    the same length, or no entry is type 2 — never silently drops everything.
    Returns ("unknown", "unknown") when no document name is present.
    """
    names, years, types = _split(doc_name), _split(doc_year), _split(doc_types)
    if not names:
        return "unknown", "unknown"
    years_str = "; ".join(years) if years else "unknown"
    if len(types) != len(names) or (years and len(years) != len(names)):
        return "; ".join(names), years_str
    keep = [i for i, t in enumerate(types) if t == "2"]
    if not keep:
        return "; ".join(names), years_str
    kept_names = "; ".join(names[i] for i in keep)
    kept_years = "; ".join(years[i] for i in keep) if years else "unknown"
    return kept_names, kept_years


def _strip_reasoning(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    sys_out = re.sub(r'\n- "reasoning":[^\n]*', "", system_prompt)
    usr_out = re.sub(r'"reasoning": "[^"]*",\s*', "", user_prompt)
    return sys_out, usr_out


def get_prompt(polity: str, name: str, start_year, end_year, doc_name: str, doc_year: str,
               reasoning: bool = True, with_examples: bool = True) -> tuple[str, str]:
    """Return (system_prompt, user_prompt). Years of None render as "unknown"."""
    examples = f"\n{EXAMPLES}\n" if with_examples else ""
    sys_p = SYSTEM_PROMPT_TEMPLATE.format(question=QUESTION, examples=examples)
    usr_p = USER_PROMPT_TEMPLATE.format(
        polity=polity, name=name,
        start_year="unknown" if start_year is None else start_year,
        end_year="unknown" if end_year is None else end_year,
        doc_name=doc_name, doc_year=doc_year,
    )
    if not reasoning:
        sys_p, usr_p = _strip_reasoning(sys_p, usr_p)
    return sys_p, usr_p


if __name__ == "__main__":  # print the full prompts: uv run python prompts/constitution_bypass.py [--no-examples]
    import sys
    ex = "--no-examples" not in sys.argv
    s, u = get_prompt("United States Of America", "Abraham Lincoln", 1861, 1865,
                      "United States Constitution", "1787", with_examples=ex)
    print("=" * 30, f"SYSTEM (with_examples={ex})", "=" * 30); print(s)
    print("=" * 30, "USER", "=" * 30); print(u)
