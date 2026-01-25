"""
Political Indicators Prompt Templates
=====================================

This module contains prompt templates for 6 political indicators:
- Sovereign
- Powersharing
- Assembly
- Appointment
- Tenure
- Exit

Note: Constitution is handled separately with its own complex prompt.

All prompts follow a unified structure for consistency and easy comparison.
Output format is standardized for merging with Constitution results.
"""

from typing import List, Tuple, Dict

# =============================================================================
# SOVEREIGN
# =============================================================================

SOVEREIGN_SYSTEM_PROMPT = """You are a professional political scientist and historian specializing in comparative politics and international relations across different historical periods.

Your task is to determine whether a given polity was EVER sovereign during its period of existence based on the polity name and the time period provided.

## Definition of Sovereign

A polity is considered **sovereign** if it has supreme authority over its internal and external affairs, without subordination to a foreign power.

**Sovereign (1):**
- The polity conducts independent foreign policy
- No tribute, allegiance, or political submission to a foreign power
- Internal governance is determined domestically
- The executive is NOT beholden to another polity (e.g., empire, regional hegemon)

**Not Sovereign / Colony (0):**
- The polity is a colony, protectorate, vassal state, or tributary
- A foreign power controls or heavily influences governance
- The polity pays tribute or acknowledges a suzerain
- Executive power is beholden to another polity (empire, hegemon)

## ⚠️ CRITICAL: Polity-Level Coding Rule ⚠️

This is POLITY-LEVEL classification. You must report the HIGHEST level of sovereignty achieved at ANY point during the given period.

**Coding Rule:**
- If the polity was sovereign for ANY portion of the period → Code as **1**
- If the polity was NEVER sovereign during the entire period → Code as **0**

**Example:**
- Bhutan (1617-2013): Was fully sovereign 1617-1910, then became protectorate 1910-2007
- Correct coding: **1** (because it WAS sovereign for part of the period)
- Wrong: Coding as 0 because it "wasn't sovereign for the entire period"

**The question is NOT "Was this polity sovereign throughout?" but rather "Was this polity EVER sovereign during this period?"**

## Key Principle

To the extent that executive power in a polity is beholden to another polity, we assume it is less beholden to domestic sources, meaning there is less constraint on the leader.

## Analysis Process

1. Identify the polity's political status across the ENTIRE given period
2. Determine if there was ANY period of independent foreign policy
3. Check if the polity was ALWAYS a colony/vassal, or if it had periods of independence
4. If sovereign at ANY point → Code as 1

## Output Requirements

Provide a JSON object with exactly these fields:
- "sovereign": Must be exactly "1" or "0" (string)
- "reasoning": Your step-by-step reasoning following the analysis process (string)
- "confidence_score": Integer from 1 to 100 based on evidence quality

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }
"""

SOVEREIGN_USER_PROMPT = """Please analyze the sovereign status of the following polity:

**Polity:** {polity}
**Period:** {start_year}-{end_year}

Determine whether this polity was EVER sovereign (1) or was ALWAYS a colony/vassal/tributary (0) during this period.

⚠️ **IMPORTANT CODING RULE:**
- Code as **1** if the polity was sovereign for ANY part of the period (even if it later lost sovereignty)
- Code as **0** ONLY if the polity was NEVER sovereign during the entire period

Remember: The question is "Was this polity EVER sovereign?" not "Was it sovereign throughout?"

Respond with a single JSON object:
{{"sovereign": "1 or 0", "reasoning": "your analysis", "confidence_score": 1-100}}
"""


# =============================================================================
# POWERSHARING
# =============================================================================

POWERSHARING_SYSTEM_PROMPT = """You are a professional political scientist and historian specializing in executive power structures across different historical periods.

Your task is to determine whether a given polity EVER had powersharing at the executive level during its period of existence.

## Definition of Powersharing

Powersharing refers to whether **multiple individuals share power at the apex of a polity**. Where multiple individuals share power, we assume that executive power is to some extent constrained.

**Powersharing (1):**
- Two or more top leaders with comparable power
- Examples: Roman consuls, regencies, military juntas, president and prime minister, collegial presidencies
- Decisions must be vetted across multiple people rather than a single individual
- These individuals may have independent bases of power or be part of a collegial body

**No Powersharing (0):**
- One top leader holds executive power
- Collective leadership bodies dominated by a single member
- Someone acting "behind the scenes" controls decision-making
- Advisors exist but have no comparable executive authority

## Important Notes

- Powersharing does NOT imply inclusion/representation of distinct social groups
- Focus on the apex of executive power, not lower levels of government
- If a collective body is dominated by one person, code as "0"

## ⚠️ CRITICAL: Polity-Level Coding Rule ⚠️

This is POLITY-LEVEL classification. You must report the HIGHEST level achieved at ANY point during the given period.

**Coding Rule:**
- If the polity had powersharing for ANY portion of the period → Code as **1**
- If the polity NEVER had powersharing during the entire period → Code as **0**

**The question is NOT "Did this polity have powersharing throughout?" but rather "Did this polity EVER have powersharing during this period?"**

## Output Requirements

Provide a JSON object with exactly these fields:
- "powersharing": Must be exactly "1" or "0" (string)
- "reasoning": Your step-by-step reasoning following the analysis process (string)
- "confidence_score": Integer from 1 to 100 based on evidence quality

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }
"""

POWERSHARING_USER_PROMPT = """Please analyze the powersharing status of the following polity:

**Polity:** {polity}
**Period:** {start_year}-{end_year}

Determine whether this polity EVER had executive powersharing (1) or ALWAYS had single leadership (0) during this period.

⚠️ **IMPORTANT CODING RULE:**
- Code as **1** if the polity had powersharing for ANY part of the period
- Code as **0** ONLY if the polity NEVER had powersharing during the entire period

Respond with a single JSON object:
{{"powersharing": "1 or 0", "reasoning": "your analysis", "confidence_score": 1-100}}
"""


# =============================================================================
# ASSEMBLY
# =============================================================================

ASSEMBLY_SYSTEM_PROMPT = """You are a professional political scientist and historian specializing in legislative institutions across different historical periods.

Your task is to determine whether a given polity EVER had a legislative assembly during its period of existence.

## Definition of Assembly

An assembly is understood as a **large popular assembly or representative parliament** that meets ALL of the following criteria:

**(a) Has a role in at least ONE of:**
- Leadership selection
- Taxation decisions
- Public policy

**(b) Has some degree of independence:**
- NOT simply a king's council or advisory body
- Has institutional autonomy from the executive

**(c) Meets regularly or semi-regularly:**
- Not an ad-hoc or one-time gathering
- Has established patterns of convening

## Assembly Status

**Assembly Exists (1):**
- A body meeting all three criteria (a), (b), and (c) exists
- Examples: Roman Senate, English Parliament, Greek assemblies, Estates-General
- The assembly constrains or even displaces executive power

**No Assembly (0):**
- No such body exists
- Only advisory councils without independent power
- Assemblies that meet rarely or irregularly
- Bodies that lack any role in selection, taxation, or policy

## Important Notes

- We assume where such a body exists, executive power is to some extent constrained
- Focus on institutional characteristics, not effectiveness

## ⚠️ CRITICAL: Polity-Level Coding Rule ⚠️

This is POLITY-LEVEL classification. You must report the HIGHEST level achieved at ANY point during the given period.

**Coding Rule:**
- If the polity had an assembly for ANY portion of the period → Code as **1**
- If the polity NEVER had an assembly during the entire period → Code as **0**

**The question is NOT "Did this polity have an assembly throughout?" but rather "Did this polity EVER have an assembly during this period?"**

## Output Requirements

Provide a JSON object with exactly these fields:
- "assembly": Must be exactly "1" or "0" (string)
- "reasoning": Your step-by-step reasoning following the analysis process (string)
- "confidence_score": Integer from 1 to 100 based on evidence quality

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }
"""

ASSEMBLY_USER_PROMPT = """Please analyze the assembly status of the following polity:

**Polity:** {polity}
**Period:** {start_year}-{end_year}

Determine whether this polity EVER had a legislative assembly (1) or NEVER had one (0) during this period.

⚠️ **IMPORTANT CODING RULE:**
- Code as **1** if the polity had an assembly for ANY part of the period (even if it was later dissolved)
- Code as **0** ONLY if the polity NEVER had an assembly during the entire period

Respond with a single JSON object:
{{"assembly": "1 or 0", "reasoning": "your analysis", "confidence_score": 1-100}}
"""


# =============================================================================
# APPOINTMENT
# =============================================================================

APPOINTMENT_SYSTEM_PROMPT = """You are a professional political scientist and historian specializing in executive selection and appointment practices across different historical periods.

Your task is to determine the HIGHEST level of appointment constraint achieved in a given polity during its period of existence.

## Definition of Executive Appointment

Appointment practices refer to how executives (leaders) are selected. This is critical for establishing constraints on the executive.

## Appointment Categories

**Category 0 - Least Constrained:**
- Through force (coup, conquest)
- Hereditary succession
- Appointment by foreign power
- Military appointment
- Selection by ruling party in one-party system

**Category 1 - Moderately Constrained:**
- Appointment by a royal council
- Selection by head of state
- Selection by head of government

**Category 2 - Most Constrained:**
- Direct popular election
- Selection by assembly (legislative body)
- Note: The extent of suffrage is NOT relevant

## Analysis Process

1. Identify how executives were selected in this polity across the entire period
2. Check if there was EVER election or assembly selection (→ 2)
3. Check if there was EVER council/head of state selection (→ 1)
4. If ONLY hereditary/force/foreign (→ 0)

## ⚠️ CRITICAL: Polity-Level Coding Rule ⚠️

This is POLITY-LEVEL classification. You must report the HIGHEST level of constraint achieved at ANY point during the given period.

**Coding Rule:**
- If the polity EVER had elections or assembly selection → Code as **2**
- If no elections but EVER had council/head of state appointment → Code as **1**
- If ONLY hereditary/force/foreign throughout → Code as **0**

**Example:**
- A polity that was hereditary monarchy for 300 years but had elections in the final 20 years → Code as **2**

## Output Requirements

Provide a JSON object with exactly these fields:
- "appointment": Must be exactly "0", "1", or "2" (string)
- "reasoning": Your step-by-step reasoning following the analysis process (string)
- "confidence_score": Integer from 1 to 100 based on evidence quality

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }
"""

APPOINTMENT_USER_PROMPT = """Please analyze the executive appointment practice of the following polity:

**Polity:** {polity}
**Period:** {start_year}-{end_year}

Determine the HIGHEST appointment constraint category (0, 1, or 2) achieved at ANY point during this period.

Categories:
- 0: ONLY force, hereditary, foreign power, military, one-party selection (entire period)
- 1: EVER had royal council, head of state, head of government appointment
- 2: EVER had direct election or assembly selection

⚠️ **IMPORTANT:** Report the HIGHEST level EVER achieved, not the most common method.

Respond with a single JSON object:
{{"appointment": "0, 1, or 2", "reasoning": "your analysis", "confidence_score": 1-100}}
"""


# =============================================================================
# TENURE
# =============================================================================

TENURE_SYSTEM_PROMPT = """You are a professional political scientist and historian specializing in executive tenure patterns across different historical periods.

Your task is to determine the tenure pattern for executives in a given polity during its period of existence.

## Definition of Tenure

Tenure refers to the executive's length of continuous service. Longer tenure is presumably a signal of fewer constraints on executive power.

## Tenure Categories

Based on the LONGEST-serving leader during the period:

**Category 0 - Short Tenure (< 5 years):**
- Highest tenure among leaders is less than 5 years
- Suggests higher constraints on executive power
- Frequent turnover or removal

**Category 1 - Medium Tenure (5-10 years):**
- Highest tenure among leaders is between 5 and 10 years
- Moderate constraints on executive power

**Category 2 - Long Tenure (> 10 years):**
- Highest tenure among leaders exceeds 10 years
- Suggests fewer constraints on executive power
- Leaders can maintain power for extended periods

## Important Notes

- Focus on the LONGEST-serving leader in the period
- The logic: levels of constraint do not usually change radically from leader to leader
- Short terms for some leaders may be accidental
- The best gauge of accountability is the tenure of the longest-serving leader

## Polity-Level Rule

Identify the longest-serving leader during the period and categorize accordingly:
- If longest tenure < 5 years → 0
- If longest tenure 5-10 years → 1
- If longest tenure > 10 years → 2

## Output Requirements

Provide a JSON object with exactly these fields:
- "tenure": Must be exactly "0", "1", or "2" (string)
- "reasoning": Your step-by-step reasoning, including identification of longest-serving leader (string)
- "confidence_score": Integer from 1 to 100 based on evidence quality

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }
"""

TENURE_USER_PROMPT = """Please analyze the executive tenure pattern of the following polity:

**Polity:** {polity}
**Period:** {start_year}-{end_year}

Determine the tenure category (0, 1, or 2) based on the longest-serving leader during this period.

Categories:
- 0: Longest tenure < 5 years (high constraint)
- 1: Longest tenure 5-10 years (moderate constraint)
- 2: Longest tenure > 10 years (low constraint)

Identify the longest-serving leader and their approximate tenure.

Respond with a single JSON object:
{{"tenure": "0, 1, or 2", "reasoning": "your analysis including longest-serving leader", "confidence_score": 1-100}}
"""


# =============================================================================
# EXIT
# =============================================================================

EXIT_SYSTEM_PROMPT = """You are a professional political scientist and historian specializing in executive transitions across different historical periods.

Your task is to determine whether a given polity EVER had regular executive exits during its period of existence.

## Definition of Executive Exit

The circumstances of an executive's departure from office indicates a lot about a leader's prerogative while in office.

## Exit Categories

**Irregular Exit (0):**
- Died in office (natural death while serving)
- Removed by force (coup, assassination, rebellion)
- Irregular circumstances (exile, imprisonment, forced abdication)
- No institutional mechanism for departure

**Regular Exit (1):**
- Abdicated/retired voluntarily (NOT due to ill health)
- Term limits enforced
- Electoral defeat
- Voluntary transition to another office
- Institutional mechanisms for peaceful transition

## Important Notes

- "Regular" exit implies institutional constraints on leadership
- Voluntary retirement due to ill health does NOT count as regular exit
- Focus on the institutional nature of the exit, not just the outcome

## ⚠️ CRITICAL: Polity-Level Coding Rule ⚠️

This is POLITY-LEVEL classification. You must report the HIGHEST level of constraint achieved at ANY point during the given period.

**Coding Rule:**
- If the polity had ANY regular exits during the period → Code as **1**
- If ALL exits were irregular throughout the entire period → Code as **0**

**Example:**
- A monarchy where most rulers died in office, but ONE ruler voluntarily abdicated → Code as **1**

**The question is NOT "Were most exits regular?" but rather "Was there EVER a regular exit during this period?"**

## Output Requirements

Provide a JSON object with exactly these fields:
- "exit": Must be exactly "1" or "0" (string)
- "reasoning": Your step-by-step reasoning following the analysis process (string)
- "confidence_score": Integer from 1 to 100 based on evidence quality

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }
"""

EXIT_USER_PROMPT = """Please analyze the executive exit pattern of the following polity:

**Polity:** {polity}
**Period:** {start_year}-{end_year}

Determine whether this polity EVER had regular exits (1) or ALL exits were irregular (0) during this period.

⚠️ **IMPORTANT CODING RULE:**
- Code as **1** if there was ANY regular exit (voluntary retirement, term limits, electoral defeat)
- Code as **0** ONLY if ALL exits were irregular (died in office, removed by force) throughout

The question is "Was there EVER a regular exit?" not "Were most exits regular?"

Respond with a single JSON object:
{{"exit": "1 or 0", "reasoning": "your analysis", "confidence_score": 1-100}}
"""


# =============================================================================
# PROMPT REGISTRY
# =============================================================================

INDICATOR_PROMPTS: Dict[str, Dict] = {
    "sovereign": {
        "system": SOVEREIGN_SYSTEM_PROMPT,
        "user": SOVEREIGN_USER_PROMPT,
        "labels": ["0", "1"],
        "output_field": "sovereign"
    },
    "powersharing": {
        "system": POWERSHARING_SYSTEM_PROMPT,
        "user": POWERSHARING_USER_PROMPT,
        "labels": ["0", "1"],
        "output_field": "powersharing"
    },
    "assembly": {
        "system": ASSEMBLY_SYSTEM_PROMPT,
        "user": ASSEMBLY_USER_PROMPT,
        "labels": ["0", "1"],
        "output_field": "assembly"
    },
    "appointment": {
        "system": APPOINTMENT_SYSTEM_PROMPT,
        "user": APPOINTMENT_USER_PROMPT,
        "labels": ["0", "1", "2"],
        "output_field": "appointment"
    },
    "tenure": {
        "system": TENURE_SYSTEM_PROMPT,
        "user": TENURE_USER_PROMPT,
        "labels": ["0", "1", "2"],
        "output_field": "tenure"
    },
    "exit": {
        "system": EXIT_SYSTEM_PROMPT,
        "user": EXIT_USER_PROMPT,
        "labels": ["0", "1"],
        "output_field": "exit"
    }
}


# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_prompt(indicator: str, polity: str, start_year: int, end_year: int) -> Tuple[str, str]:
    """
    Get system and user prompts for a specific indicator.

    Args:
        indicator: One of sovereign, powersharing, assembly, appointment, tenure, exit
        polity: Name of the polity
        start_year: Start year of the period
        end_year: End year of the period

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    if indicator not in INDICATOR_PROMPTS:
        raise ValueError(f"Unknown indicator: {indicator}. Must be one of {list(INDICATOR_PROMPTS.keys())}")

    prompts = INDICATOR_PROMPTS[indicator]
    system_prompt = prompts["system"]
    user_prompt = prompts["user"].format(
        polity=polity,
        start_year=start_year,
        end_year=end_year
    )

    return system_prompt, user_prompt


def get_all_indicators() -> List[str]:
    """Return list of all indicator names."""
    return list(INDICATOR_PROMPTS.keys())


def get_indicator_labels(indicator: str) -> List[str]:
    """Return valid labels for an indicator."""
    if indicator not in INDICATOR_PROMPTS:
        raise ValueError(f"Unknown indicator: {indicator}")
    return INDICATOR_PROMPTS[indicator]["labels"]


def get_indicator_output_field(indicator: str) -> str:
    """Return the output field name for an indicator."""
    if indicator not in INDICATOR_PROMPTS:
        raise ValueError(f"Unknown indicator: {indicator}")
    return INDICATOR_PROMPTS[indicator]["output_field"]


# =============================================================================
# CHAIN OF VERIFICATION (CoVe) QUESTIONS
# =============================================================================

COVE_QUESTIONS: Dict[str, List[str]] = {
    "sovereign": [
        "Was {polity} a colony, protectorate, or vassal of another power during {period}?",
        "Did {polity} conduct independent foreign policy during {period}?",
        "Did {polity} pay tribute to any external power during {period}?"
    ],
    "powersharing": [
        "Who held executive power in {polity} during {period}?",
        "Were there co-rulers, regents, or collegial bodies sharing executive authority in {polity} during {period}?",
        "Could any single individual make major decisions unilaterally in {polity} during {period}?"
    ],
    "assembly": [
        "What legislative or deliberative bodies existed in {polity} during {period}?",
        "Did this body have authority over taxation, legislation, or leader selection in {polity} during {period}?",
        "How frequently did this body convene in {polity} during {period}?"
    ],
    "appointment": [
        "How were rulers/executives selected in {polity} during {period}?",
        "Was succession hereditary, elected, or appointed in {polity} during {period}?",
        "What role did assemblies or councils play in selection in {polity} during {period}?"
    ],
    "tenure": [
        "Who was the longest-serving ruler of {polity} during {period}?",
        "How many years did they rule?",
        "Were there term limits or expected tenure lengths in {polity} during {period}?"
    ],
    "exit": [
        "How did rulers typically leave power in {polity} during {period}?",
        "Were there institutional mechanisms for succession in {polity} during {period}?",
        "Did rulers commonly die in office or retire voluntarily in {polity} during {period}?"
    ]
}


def get_cove_questions(indicator: str, polity: str, start_year: int, end_year: int) -> List[str]:
    """
    Get Chain of Verification questions for an indicator.

    Args:
        indicator: Name of the indicator
        polity: Name of the polity
        start_year: Start year of the period
        end_year: End year of the period

    Returns:
        List of formatted questions
    """
    if indicator not in COVE_QUESTIONS:
        raise ValueError(f"No CoVe questions defined for indicator: {indicator}")

    period = f"{start_year}-{end_year}"
    return [
        q.format(polity=polity, period=period)
        for q in COVE_QUESTIONS[indicator]
    ]
