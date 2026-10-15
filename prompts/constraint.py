"""General executive-constraint concept prompt (direct concept coding).

Instead of coding separate indicators and aggregating them (IRT / lexical
index), the model is asked to rate the overall concept on a numeric scale.
Two scales are available for the pilot: "7" (1-7, anchored like Polity's
XCONST) and "10" (0-10).

DEFINITION is the researcher's text (Gerring) verbatim — do not edit without
asking. Only the scale anchors, coding notes and output format are ours.
"""
import re

DEFINITION = """Constraint

Definition: the degree to which executive power is institutionally constrained by domestic actors.

(Executive: the individual(s) in charge of administration at the apex of a polity, e.g., a monarch, president, or prime minister. Typically, executive power is vested in a single individual; occasionally, it is held collectively as in the Swiss presidency.)

If the executive can make policy decisions without input from other actors (beyond kith and kin) then s/he is unconstrained. Power in such a polity may be exercised in a personal, and highly arbitrary, fashion.

Constraints on executive decisionmaking may arise from a variety of institutions, so long as they enjoy some degree of independence. This includes a formal constitution, a judiciary, legislature, bureaucracy, religious body, local governments, a federal separation of central and local authority, citizen petitions, and elections.

Constraints may be indicated by methods of executive entry (e.g., by election rather than inheritance or seizure) and exit (e.g., by regular methods rather than coercion or death), by tenure in office (longer tenure generally indicating fewer constraints), as well as by methods of governing and by the symbols attached to executive authority.

Constraints may be de jure or de facto, but they must be operational. If an apparent constraint is ignored or bypassed then it is not really serving to constrain the executive's decisionmaking power and should be discounted.

Constraints refer to decisionmaking power rather than implementation power or state capacity, which in many states was, and is, limited. Logistical constraints imposed by technology, revenue, infrastructure, or bureaucracy means that executives do not always get what they want. This is irrelevant for present purposes. Executive constraints refers only to decisionmaking power, the ability to establish the de jure and de facto law of the land.

Examples of polities in which executives were (usually) highly constrained include the Greek poleis; the republics of Rome, Venice, Genoa, Florence, and Holland; the constitutional monarchies of England, Sweden during the Age of Liberty, Poland-Lithuania; and democracies in the modern era."""

SCALES = {
    "7": {
        "label": "1 to 7",
        "range": "1-7",
        "lo": 1, "hi": 7,
        "anchors": """Rate the executive's constraint on a scale from 1 to 7:
- 1 = Unlimited authority. The executive decides alone; no independent domestic actor can block or substantially modify decisions. Rule is personal and may be arbitrary.
- 3 = Slight to moderate limitations. Independent actors exist and are routinely consulted, but the executive can override them at will; constraints rarely bind.
- 5 = Substantial limitations. Independent institutions can block or substantially modify important decisions in at least some policy areas, and the executive must usually bargain to get their way.
- 7 = Executive parity or subordination. Independent institutions have effective authority equal to or greater than the executive in most areas; the executive cannot establish the law of the land alone.
- Use 2, 4 and 6 for cases that fall between the anchors.""",
    },
    "10": {
        "label": "0 to 10",
        "range": "0-10",
        "lo": 0, "hi": 10,
        "anchors": """Rate the executive's constraint on a scale from 0 to 10:
- 0 = Completely unconstrained. The executive decides alone; no independent domestic actor can block or substantially modify decisions.
- 5 = Moderately constrained. Independent institutions can block or substantially modify important decisions in some policy areas, but the executive prevails in most.
- 10 = Fully constrained. Independent institutions have effective authority equal to or greater than the executive in most areas; the executive cannot establish the law of the land alone.
- Use intermediate values for intermediate cases.""",
    },
}

SYSTEM_PROMPT_TEMPLATE = """You are a political scientist and historian coding executive constraints for historical leaders.

## Concept

{definition}

## Scale

{anchors}

## Coding Notes

- The executive is the leader named below. Judge THIS leader's actual (de facto) situation during the tenure given, not the polity's reputation across its whole history.
- Count only constraints that were operational during the tenure; discount those that were ignored or bypassed.

## Output Requirements

Provide a JSON object with exactly these fields:
- "constraint": Integer from {range_}
- "reasoning": Your step-by-step reasoning (string)
- "confidence_score": Integer from 1 to 100

Respond with ONLY a JSON object, starting with {{ and ending with }}.
"""

USER_PROMPT_TEMPLATE = """**Polity:** {polity}
**Leader:** {name}
**Tenure Period:** {start_year}-{end_year}

Respond with a single JSON object:
{{"constraint": {range_}, "reasoning": "your analysis", "confidence_score": 1-100}}
"""


def _strip_reasoning(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    sys_out = re.sub(r'\n- "reasoning":[^\n]*', "", system_prompt)
    usr_out = re.sub(r'"reasoning": "[^"]*",\s*', "", user_prompt)
    return sys_out, usr_out


def get_prompt(polity: str, name: str, start_year, end_year,
               scale: str = "7", reasoning: bool = True) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the given scale ("7" or "10")."""
    scale = str(scale)
    if scale not in SCALES:
        raise ValueError(f"Unknown scale {scale!r}; choose one of {sorted(SCALES)}")
    sc = SCALES[scale]
    sys_p = SYSTEM_PROMPT_TEMPLATE.format(definition=DEFINITION, anchors=sc["anchors"],
                                          range_=sc["label"])
    usr_p = USER_PROMPT_TEMPLATE.format(
        polity=polity, name=name,
        start_year="unknown" if start_year is None else start_year,
        end_year="unknown" if end_year is None else end_year,
        range_=sc["range"],
    )
    if not reasoning:
        sys_p, usr_p = _strip_reasoning(sys_p, usr_p)
    return sys_p, usr_p


if __name__ == "__main__":  # print the full prompts: uv run python prompts/constraint.py [7|10]
    import sys
    sc = sys.argv[1] if len(sys.argv) > 1 else "7"
    s, u = get_prompt("Venice", "Francesco Foscari", 1423, 1457, scale=sc)
    print("=" * 30, f"SYSTEM (scale {sc})", "=" * 30); print(s)
    print("=" * 30, "USER", "=" * 30); print(u)
