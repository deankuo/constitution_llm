"""
Constitution Indicator Prompt Templates (Leader-Level)

This module contains the complex prompt template for the Constitution indicator.
Constitution uses a more detailed analysis framework with 4 required elements.

The Constitution indicator is special:
- No ground truth available (requires manual evaluation)
- More complex prompt with detailed criteria
- Primary target for Chain of Verification (CoVe)

Key Change: Now supports LEADER-LEVEL analysis with 'name' parameter.
"""

from typing import Tuple, Dict, List, Optional


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a professional political scientist, historian, and constitutional expert specializing in constitutional history across different countries.

Your task is to determine whether a given polity had a constitution during a specific leader's reign based on the polity name, leader name, and the leader's reign period provided.

## Definition of Constitution

A constitution is understood as a set of rules setting forth how a polity is governed.

**A constitution MUST have ALL FOUR of these elements to qualify as "Yes":**

1. **Written Document(s)**
   - There must be identifiable written documents (statutes, treaties, charters, basic laws, constitutions)
   - Purely oral traditions, customary practices, or uncodified norms do NOT qualify
   - Exception: If a polity has multiple written documents that collectively serve as a constitution, this counts

2. **Code of Law**
   - The document(s) must contain legal provisions and binding rules
   - These must have legal force and authority within the polity

3. **Rules About Governance**
   - The document(s) must specify how the polity is governed, including:
     * How leaders are selected, chosen, or succeed to power
     * How laws are determined, enacted, or promulgated
     * The structure and organization of governmental institutions

4. **Limitations on Authority**
   - The document(s) must include constraints on the ruler's or government's power, such as:
     * Rights of citizens or subjects
     * Checks and balances between institutions
     * Procedures that limit arbitrary exercise of power
     * Protections against abuse of authority

**CRITICAL: If ANY of these four elements is absent or unclear, you MUST answer "No".**

## What Does NOT Qualify as a Constitution

- **Legal codes only**: Criminal or civil codes (e.g., Hammurabi's Code) without governance structures or limitations
- **Royal decrees**: Proclamations or edicts that do not establish systematic governance rules
- **Colonial charters**: Administrative documents imposed externally without local constitutional framework or limitations on authority
- **Treaties**: International agreements that do not establish internal governance (unless they serve as the constitutional basis)
- **Uncodified traditions**: Customary law or practices not recorded in written form
- **Advisory documents**: Guidelines or recommendations without legal force

## Analysis Process

Follow this systematic approach:

Step 1 **Identify the Historical Context**
   - What type of polity was this? (independent state, colony, protectorate, vassal state, etc.)
   - What was its political status during this leader's reign?

Step 2 **Determine Governance Authority**
   - Who had the authority to establish rules of governance during this leader's reign?
   - Was governance authority internal or externally imposed?
   - If external, did the polity have any autonomous constitutional framework?

Step 3 **Identify Written Constitutional Documents**
   - What specific written document(s) established rules for governance during this leader's reign?
   - Name the document(s) with their official titles
   - When were they adopted or enacted?
   - If multiple constitutions existed during this reign, list ALL of them

Step 4 **Evaluate Against the Four Criteria**
   - Element 1 (Written): Is there a specific written document you can name?
   - Element 2 (Legal Code): Does it contain binding legal provisions?
   - Element 3 (Governance Rules): Does it specify how leaders are chosen and how laws are made?
   - Element 4 (Limitations): Does it constrain the government's power or protect rights?

   **If you cannot confirm YES for all four elements, the answer is "No".**

Step 5 **Consider Temporal Factors**
   - Did constitutional document(s) exist during this leader's reign?
   - Were there changes, amendments, or new constitutions adopted during this reign?
   - If multiple constitutions, record ALL that were in effect during this leader's reign

Step 6 **Assess Confidence Based on Historical Evidence**
   - How clear and well-documented is the evidence?
   - Are you identifying specific documents or making inferences?
   - Is there scholarly consensus or debate?
   - **Be conservative: when in doubt, lean toward "No"**

## Confidence Score Guidelines

- **81-100 (Very High):** You can name specific documents with exact dates; all four elements are clearly documented; scholarly consensus exists
- **61-80 (High):** Strong documentary evidence; all four elements are present; minor scholarly debate possible
- **41-60 (Moderate):** You can identify documents but some elements are ambiguous; reasonable alternative interpretations exist
- **21-40 (Low):** Very limited evidence; you are making educated guesses based on general knowledge of the region/period
- **1-20 (Very Low):** No reliable specific information; essentially speculating based on typical patterns

**If your confidence is below 3, seriously consider whether the answer should be "No".**

## Output Requirements

Provide a JSON object with exactly these fields:
- "constitution": Must be exactly "Yes" or "No" (string)
- "document_name": Official name(s) of the constitutional document(s), or "N/A" if none. If multiple documents, separate with semicolons (e.g., "Document A; Document B")
- "constitution_year": Year(s) of adoption as exact integers only, or "N/A" if none. Do NOT use approximations like "c.", "circa", or "approximately". If multiple years, separate with semicolons (e.g., "1789; 1791"). The year must fall within or before the leader's reign period.
- "reasoning": Your step-by-step reasoning following the analysis process (string)
- "confidence_score": Integer from 1 to 100 based on evidence quality

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }

Maintain professional objectivity and base all judgments on verifiable historical facts."""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

USER_PROMPT_TEMPLATE = """Please analyze the constitutional status of the following leader's reign:

**Polity:** {polity}
**Leader:** {name}
**Reign Period:** {reign_period}

## Analysis Instructions

Work through each step systematically:

1. Identify the type of political entity and its status during this leader's reign
2. Determine who had authority over its governance (internal vs external) under this leader
3. Identify specific written documents that established governance rules during this leader's reign
4. Check EACH of the four required elements:
   ✓ Written document(s) - can you name them?
   ✓ Code of law - do they have legal force?
   ✓ Governance rules - do they specify how leaders are chosen and laws are made?
   ✓ Limitations on authority - do they constrain power or protect rights?
5. If multiple constitutions existed during this reign, record ALL of them (separate with semicolons)
6. Assess your confidence based on the quality of historical evidence

**Remember: If you cannot confirm ALL FOUR elements, answer "No".**

**When uncertain, default to "No" rather than "Yes".**

Respond with a single JSON object (no markdown, no extra text):

{{"constitution": "Yes or No", "document_name": "name(s) or N/A (semicolon-separated if multiple)", "constitution_year": "exact integer year(s) or N/A (semicolon-separated if multiple, no 'c.' or 'circa')", "reasoning": "step-by-step reasoning", "confidence_score": 1-100}}

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
        end_year: End year of the leader's reign (None if unknown/unavailable in data)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Format reign period (show "unknown" if end year is missing in data)
    reign_period = f"{start_year}-{end_year if end_year is not None else 'unknown'}"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        polity=polity,
        name=name,
        reign_period=reign_period
    )

    return SYSTEM_PROMPT, user_prompt


# Backwards compatibility alias
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
    return ['Yes', 'No']


# Backwards compatibility alias
def get_constitution_labels() -> List[str]:
    """Alias for get_labels for backwards compatibility."""
    return get_labels()


def get_output_field() -> str:
    """Return the output field name for constitution."""
    return 'constitution'


# Backwards compatibility alias
def get_constitution_output_field() -> str:
    """Alias for get_output_field for backwards compatibility."""
    return get_output_field()


def get_expected_output_schema() -> Dict[str, str]:
    """Return the expected output schema for constitution."""
    return {
        "constitution": "string, 'Yes' or 'No'",
        "document_name": "string, document name(s) or 'N/A' (semicolon-separated if multiple)",
        "constitution_year": "string, year(s) or 'N/A' (semicolon-separated if multiple)",
        "reasoning": "string",
        "confidence_score": "integer, 1-100"
    }


# =============================================================================
# CHAIN OF VERIFICATION (CoVe) QUESTIONS
# =============================================================================

COVE_QUESTION_TEMPLATES: Dict[str, List[str]] = {
    "element_1_written": [
        "What written legal documents governed {polity}'s political structure during {name}'s reign ({start_year}-{end_year})?",
        "Can you identify specific written constitutions, charters, or fundamental laws for {polity} under {name}?"
    ],
    "element_2_legal_code": [
        "Did these documents have binding legal force in {polity} during {name}'s reign?",
        "Were the governance documents in {polity} legally enforceable under {name}?"
    ],
    "element_3_governance": [
        "How was succession determined in {polity} during {name}'s reign ({start_year}-{end_year})?",
        "What institution was responsible for legislation in {polity} under {name}?"
    ],
    "element_4_limitations": [
        "What constraints existed on {name}'s power in {polity}?",
        "Could {name} legislate without institutional approval in {polity}?"
    ],
    "anchor_leader": [
        "Who was {name} and when did they rule {polity}?",
        "What was {name}'s role/title in {polity} during {start_year}-{end_year}?"
    ]
}


def get_cove_questions(
    polity: str,
    name: str,
    start_year: int,
    end_year: int
) -> Dict[str, List[str]]:
    """
    Get Chain of Verification questions for constitution indicator (leader-level).

    Args:
        polity: Name of the polity
        name: Name of the leader
        start_year: Start year of the leader's reign
        end_year: End year of the leader's reign

    Returns:
        Dictionary of questions organized by element
    """
    formatted_questions = {}

    for element, questions in COVE_QUESTION_TEMPLATES.items():
        formatted_questions[element] = [
            q.format(polity=polity, name=name, start_year=start_year, end_year=end_year)
            for q in questions
        ]

    return formatted_questions


def get_cove_questions_flat(
    polity: str,
    name: str,
    start_year: int,
    end_year: int
) -> List[str]:
    """
    Get Chain of Verification questions as a flat list.

    Args:
        polity: Name of the polity
        name: Name of the leader
        start_year: Start year of the leader's reign
        end_year: End year of the leader's reign

    Returns:
        Flat list of all questions
    """
    questions_dict = get_cove_questions(polity, name, start_year, end_year)
    return [q for questions in questions_dict.values() for q in questions]