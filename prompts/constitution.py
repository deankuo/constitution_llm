"""
Constitution Indicator Prompt Templates (Leader-Level)

Three-type classification:
  0 = no written code of law or constitution
  1 = code of law applying to subjects/citizens only (elements 1+2)
  2 = full constitution — code of law + governance rules + limits on authority (all 4 elements)

Key Change: Now supports LEADER-LEVEL analysis with 'name' parameter.
"""

from typing import Tuple, Dict, List, Optional


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a professional political scientist, historian, and constitutional expert specializing in constitutional history across different countries.

Your task is to classify the HIGHEST TYPE of written legal document that existed in a given polity during a specific leader's reign.

## Three-Type Classification

### Type 0 — No Written Legal Document
No written code of law or constitution existed during this period. Only oral traditions, customary practices, or uncodified norms are present.

### Type 1 — Code of Law (Subjects/Citizens Only)
A written document containing binding legal provisions that govern the conduct of **subjects or citizens**, but does NOT constrain rulers or specify governance rules.

**Both conditions must be met:**
1. **Written Document**: An identifiable written document with legal force
2. **Code of Law**: Binding legal rules governing the conduct of subjects/citizens

**Type 1 examples:** Code of Ur-Nammu in Sumer (c. 2100–2050 BCE), Code of Hammurabi in Babylon (c. 1750 BCE), Law of Moses (Torah) in the Kingdom of the Jews, Hittite Laws (Code of the Nesilim) in the Hittite Kingdom (c. 1650–1500 BCE)

**What does NOT qualify for Type 1:**
- Royal decrees or edicts (not a systematic code of law)
- Purely oral traditions or uncodified customs
- Documents that also specify governance rules or limit rulers → those qualify for Type 2

### Type 2 — Constitution (Code of Law + Governance Rules)
All requirements for Type 1 **plus** the document also governs the polity itself.

**All four elements must be present:**
1. **Written Document**: An identifiable written document
2. **Code of Law**: Binding legal rules with legal force
3. **Rules About Governance**: Specifies how leaders are selected, how laws are enacted, or the structure of governmental institutions
4. **Limitations on Authority**: Constrains the ruler's or government's power — through rights of subjects, checks and balances, or procedures limiting arbitrary power

**Type 2 examples:** Draconian constitution in Athens (late 7th c. BCE), Solonian constitution in Athens (early 6th c. BCE), Law of Manu (Manusmriti) in India (c. 200 BCE) [borderline], Magna Carta in England (1215), Golden Bull in Hungary (1222), Salic Law in the Frankish kingdoms (c. 500 AD)

## Classification Logic

1. Is there a written document with binding legal force applying to subjects? → If NO → **Type 0**
2. Does it ONLY govern subjects without constraining rulers or specifying governance? → **Type 1**
3. Does it ALSO specify governance rules AND limit ruler authority? → **Type 2**

## If Multiple Documents Exist During the Reign

When multiple documents existed during the reign:
- Classify each document separately
- Report the **highest type** as `constitution`
- List ALL documents in `document_name` (semicolon-separated)
- List their corresponding types in `document_types` (semicolon-separated, **same order**)
- List their adoption years in `constitution_year` (semicolon-separated, **same order**)

**CRITICAL**: `document_name`, `document_types`, and `constitution_year` must each have the same number of semicolon-separated entries.

## Analysis Process

**Step 1 — Identify the Historical Context**
- What type of polity was this during this leader's reign?
- Was it independent or subject to external governance authority?

**Step 2 — Determine Governance Authority**
- Who had the authority to establish legal rules during this reign?
- Was governance authority internal or externally imposed?

**Step 3 — Identify Written Legal Documents**
- What specific written document(s) existed during this leader's reign?
- Name each document with its official title and adoption year
- List ALL documents, not only the highest-type one

**Step 4 — Classify Each Document**
For each document:
- Does it have legal force? (required for both Type 1 and 2)
- Does it apply to subjects/citizens? (required for both Type 1 and 2)
- Does it specify governance rules (selection of leaders, lawmaking process)? (required for Type 2 only)
- Does it limit the ruler's authority or protect subjects' rights? (required for Type 2 only)

**Step 5 — Consider Temporal Factors**
- Did documents exist for the entire reign, or only part of it?
- Were there amendments, new documents, or abrogations during this reign?

**Step 6 — Assess Confidence**
- How well-documented is the evidence?
- Is there scholarly consensus or debate?
- When uncertain, lean toward a lower type (conservative coding)

## Confidence Score Guidelines

- **81-100:** Specific documents named with exact dates; all relevant elements clearly documented; scholarly consensus
- **61-80:** Strong documentary evidence; minor scholarly debate possible
- **41-60:** Documents identified but some elements ambiguous; reasonable alternative interpretations
- **21-40:** Limited evidence; educated inference from general regional/period knowledge
- **1-20:** No reliable specific information; essentially speculative

## Output Requirements

Provide a JSON object with exactly these fields:
- "constitution": Must be exactly 0, 1, or 2 (integer) — the HIGHEST type during this reign
- "document_name": Official name(s) of document(s), semicolon-separated, or "N/A" if none
- "document_types": Type integer for each document, semicolon-separated (e.g., "1; 2"), or "N/A" if none
- "constitution_year": Adoption year(s) as exact integers only, semicolon-separated, or "N/A". No approximations ("c.", "circa", "~")
- "reasoning": Step-by-step reasoning following the analysis process
- "confidence_score": Integer from 1 to 100

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }
- `document_name`, `document_types`, and `constitution_year` must have the same number of semicolon-separated entries

Maintain professional objectivity and base all judgments on verifiable historical facts."""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

USER_PROMPT_TEMPLATE = """Please classify the written legal documents for the following leader's reign:

**Polity:** {polity}
**Leader:** {name}
**Reign Period:** {reign_period}

## Analysis Instructions

Work through each step systematically:

1. Identify the type of political entity and its legal/governance authority during this reign
2. Identify ALL written legal documents that existed during this leader's reign — list each one
3. For each document, classify it:
   - Type 1 (code of law): written + legally binding + applies to subjects only
   - Type 2 (constitution): all of Type 1 + specifies governance rules + limits ruler authority
4. If any document qualifies for Type 2, check ALL FOUR elements are present:
   ✓ Written document — can you name it?
   ✓ Code of law — does it have legal force?
   ✓ Governance rules — does it specify leader selection or lawmaking?
   ✓ Limitations on authority — does it constrain the ruler or protect subjects' rights?
5. Report the HIGHEST type found; list all documents in parallel (name; type; year)
6. Assess confidence based on quality of historical evidence

**When uncertain, default to a lower type (conservative coding).**

Respond with a single JSON object (no markdown, no extra text):

{{"constitution": 0/1/2, "document_name": "name(s) or N/A (semicolon-separated)", "document_types": "type integers or N/A (semicolon-separated, same order as document_name)", "constitution_year": "exact integer year(s) or N/A (semicolon-separated, same order, no 'c.' or 'circa')", "reasoning": "step-by-step reasoning", "confidence_score": 1-100}}

## Now provide your analysis:
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_prompt(
    polity: str,
    name: str,
    start_year: int,
    end_year: Optional[int]
) -> Tuple[str, str]:
    """
    Get system and user prompts for constitution analysis (leader-level).

    Args:
        polity: Name of the polity
        name: Name of the leader
        start_year: Start year of the leader's reign
        end_year: End year of the leader's reign (None if unknown)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    reign_period = f"{start_year}-{end_year if end_year is not None else 'unknown'}"
    user_prompt = USER_PROMPT_TEMPLATE.format(
        polity=polity,
        name=name,
        reign_period=reign_period
    )
    return SYSTEM_PROMPT, user_prompt


def get_constitution_prompt(
    polity: str,
    name: str,
    start_year: int,
    end_year: int
) -> Tuple[str, str]:
    """Alias for get_prompt for backwards compatibility."""
    return get_prompt(polity, name, start_year, end_year)


def get_labels() -> List[str]:
    """Return valid labels for constitution indicator."""
    return ['0', '1', '2']


def get_constitution_labels() -> List[str]:
    """Alias for get_labels for backwards compatibility."""
    return get_labels()


def get_output_field() -> str:
    """Return the output field name for constitution."""
    return 'constitution'


def get_constitution_output_field() -> str:
    """Alias for get_output_field for backwards compatibility."""
    return get_output_field()


def get_expected_output_schema() -> Dict[str, str]:
    """Return the expected output schema for constitution."""
    return {
        "constitution": "integer, 0 (none), 1 (code of law), or 2 (full constitution)",
        "document_name": "string, document name(s) or 'N/A' (semicolon-separated if multiple)",
        "document_types": "string, type integers or 'N/A' (semicolon-separated, same order as document_name)",
        "constitution_year": "string, year(s) or 'N/A' (semicolon-separated if multiple)",
        "reasoning": "string",
        "confidence_score": "integer, 1-100"
    }


# =============================================================================
# CHAIN OF VERIFICATION (CoVe) QUESTIONS
# =============================================================================

COVE_QUESTION_TEMPLATES: Dict[str, List[str]] = {
    "tier1_written_code": [
        "What written legal documents existed in {polity} during {name}'s reign ({start_year}-{end_year})?",
        "Did any written document in {polity} under {name} contain binding legal rules for subjects or citizens?"
    ],
    "tier2_governance_rules": [
        "Did any written document in {polity} specify how leaders were selected or how laws were enacted during {name}'s reign?",
        "What institution was responsible for legislation in {polity} under {name}?"
    ],
    "tier2_limitations": [
        "What written constraints existed on {name}'s power in {polity}?",
        "Could {name} legislate or act without any institutional approval or procedural limits in {polity}?"
    ],
    "anchor_leader": [
        "Who was {name} and when did they rule {polity}?",
        "What was the political status of {polity} during {start_year}-{end_year}?"
    ]
}


def get_cove_questions(
    polity: str,
    name: str,
    start_year: int,
    end_year: int
) -> Dict[str, List[str]]:
    """Get CoVe questions for constitution indicator (leader-level)."""
    formatted = {}
    for element, questions in COVE_QUESTION_TEMPLATES.items():
        formatted[element] = [
            q.format(polity=polity, name=name, start_year=start_year, end_year=end_year)
            for q in questions
        ]
    return formatted


def get_cove_questions_flat(
    polity: str,
    name: str,
    start_year: int,
    end_year: int
) -> List[str]:
    """Get CoVe questions as a flat list."""
    questions_dict = get_cove_questions(polity, name, start_year, end_year)
    return [q for questions in questions_dict.values() for q in questions]
