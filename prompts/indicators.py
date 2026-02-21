"""
Political Indicators Prompt Templates (Leader-Level)
=====================================================

This module contains prompt templates for 8 political indicators:
- Sovereign
- Assembly
- Appointment
- Tenure
- Exit
- Collegiality
- Separate Powers

Note: Constitution is handled separately with its own complex prompt.

All prompts follow a unified structure for consistency and easy comparison.
Output format is standardized for merging with Constitution results.

Key Change: Now supports LEADER-LEVEL analysis with 'name' parameter.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# =============================================================================
# SHARED OUTPUT FORMAT TEMPLATE
# =============================================================================

OUTPUT_FORMAT_TEMPLATE = """
## Output Requirements

Provide a JSON object with exactly these fields:
- "{indicator}": Must be exactly {valid_labels} (string)
- "reasoning": Your step-by-step reasoning following the analysis process (string)
- "confidence_score": Integer from 1 to 100 based on evidence quality

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with {{ and end with }}
"""

SYSTEM_PROMPT_HEADER = """You are a professional political scientist and historian specializing in {specialization} across different historical periods.

Your task is to determine {task_description} based on the polity name, leader name, and the leader's reign period provided.
"""

USER_PROMPT_TEMPLATE = """Please analyze the {indicator_display} of the following leader's reign:

**Polity:** {polity}
**Leader:** {name}
**Reign Period:** {start_year}-{end_year}

{task_instruction}

{coding_rule_reminder}

Respond with a single JSON object:
{{"{indicator}": "{label_format}", "reasoning": "your analysis", "confidence_score": 1-100}}
"""


# =============================================================================
# INDICATOR CONFIGURATIONS
# =============================================================================

@dataclass
class IndicatorConfig:
    """Configuration for a political indicator."""
    name: str                      # Internal name (e.g., "sovereign")
    display_name: str              # Display name (e.g., "sovereign status")
    specialization: str            # LLM role specialization
    labels: List[str]              # Valid labels
    definition: str                # Full definition text
    task_description: str          # What the LLM needs to determine
    task_instruction: str          # User prompt instruction
    coding_rule_reminder: str      # Reminder about coding rules


INDICATOR_CONFIGS: Dict[str, IndicatorConfig] = {
    
    # =========================================================================
    # SOVEREIGN
    # =========================================================================
    "sovereign": IndicatorConfig(
        name="sovereign",
        display_name="sovereign status",
        specialization="comparative politics and international relations",
        labels=["0", "1"],
        task_description="whether a given polity was sovereign during a specific leader's reign",
        
        definition="""
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

## Key Principle

To the extent that executive power in a polity is beholden to another polity, we assume it is less beholden to domestic sources, meaning there is less constraint on the leader.

## Analysis Process

1. Identify the polity's political status during this leader's reign
2. Determine if the polity had independent foreign policy during this reign
3. Check for any tribute, vassalage, or colonial relationships during this reign
4. Assess whether executive power was controlled externally
""",
        
        task_instruction="""Determine whether this polity was sovereign (1) or a colony/vassal/tributary (0) during this leader's reign.

Remember:
- Sovereign (1): Independent foreign policy, no subordination to foreign power
- Not Sovereign (0): Colony, protectorate, vassal, or tributary state""",
        
        coding_rule_reminder="""⚠️ **IMPORTANT:** Focus on the status during THIS LEADER'S REIGN, not the entire polity history."""
    ),
    
    # =========================================================================
    # ASSEMBLY
    # =========================================================================
    "assembly": IndicatorConfig(
        name="assembly",
        display_name="assembly status",
        specialization="legislative institutions",
        labels=["0", "1"],
        task_description="whether a given polity had a legislative assembly during a specific leader's reign",
        
        definition="""
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

## Analysis Process

1. Identify any legislative or deliberative bodies during this leader's reign
2. Check if the body meets criteria (a), (b), and (c)
3. Determine if the assembly functioned during this reign
""",
        
        task_instruction="""Determine whether this polity had a legislative assembly (1) or not (0) during this leader's reign.

Remember:
- Assembly (1): Popular assembly or parliament with (a) role in selection/taxation/policy, (b) independence from executive, (c) regular meetings
- No Assembly (0): No such body, or only advisory councils without institutional power""",
        
        coding_rule_reminder="""⚠️ **IMPORTANT:** Focus on whether an assembly existed and functioned during THIS LEADER'S REIGN."""
    ),
    
    # =========================================================================
    # APPOINTMENT
    # =========================================================================
    "appointment": IndicatorConfig(
        name="appointment",
        display_name="executive appointment method",
        specialization="executive selection and appointment practices",
        labels=["0", "1", "2"],
        task_description="how the executive was appointed/selected during a specific leader's reign",
        
        definition="""
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

1. Identify how THIS LEADER came to power
2. Determine if selection was hereditary, by force, or by foreign power (→ 0)
3. Determine if selection was by council or head of state/government (→ 1)
4. Determine if selection was by election or assembly (→ 2)
""",
        
        task_instruction="""Determine the appointment method category (0, 1, or 2) for how THIS LEADER came to power.

Categories:
- 0: Force, hereditary, foreign power, military, one-party selection
- 1: Royal council, head of state, head of government appointment
- 2: Direct election or assembly selection""",
        
        coding_rule_reminder="""⚠️ **IMPORTANT:** Focus on how THIS SPECIFIC LEADER was appointed/selected."""
    ),
    
    # =========================================================================
    # TENURE
    # =========================================================================
    "tenure": IndicatorConfig(
        name="tenure",
        display_name="tenure length",
        specialization="executive tenure patterns",
        labels=["0", "1", "2"],
        task_description="the tenure length of a specific leader",
        
        definition="""
## Definition of Tenure

Tenure refers to the executive's length of continuous service. Longer tenure is presumably a signal of fewer constraints on executive power.

## Tenure Categories

**Category 0 - Short Tenure (< 5 years):**
- Leader's reign is less than 5 years
- Suggests higher constraints on executive power or instability

**Category 1 - Medium Tenure (5-10 years):**
- Leader's reign is between 5 and 10 years
- Moderate constraints on executive power

**Category 2 - Long Tenure (> 10 years):**
- Leader's reign exceeds 10 years
- Suggests fewer constraints on executive power
- Leader can maintain power for extended periods

## Analysis Process

1. Calculate the length of this leader's reign (end_year - start_year)
2. Categorize based on tenure length:
   - < 5 years → 0
   - 5-10 years → 1
   - > 10 years → 2
""",
        
        task_instruction="""Determine the tenure category (0, 1, or 2) based on this leader's reign length.

Categories:
- 0: Tenure < 5 years (high constraint)
- 1: Tenure 5-10 years (moderate constraint)
- 2: Tenure > 10 years (low constraint)""",
        
        coding_rule_reminder="""⚠️ **IMPORTANT:** Calculate based on the reign period provided ({start_year} to {end_year})."""
    ),
    
    # =========================================================================
    # EXIT
    # =========================================================================
    "exit": IndicatorConfig(
        name="exit",
        display_name="exit pattern",
        specialization="executive transitions",
        labels=["0", "1"],
        task_description="the circumstances of a specific leader's departure from office",
        
        definition="""
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

## Analysis Process

1. Determine how this leader left power
2. Assess if the exit was through institutional mechanisms (→ 1)
3. Assess if the exit was irregular (death, force, exile) (→ 0)
""",
        
        task_instruction="""Determine the exit pattern (0 or 1) for how THIS LEADER left power.

Categories:
- 0: Irregular exit - died in office, removed by force, no institutional transition
- 1: Regular exit - voluntary retirement, term limits, electoral defeat, peaceful transition""",
        
        coding_rule_reminder="""⚠️ **IMPORTANT:** Focus on how THIS SPECIFIC LEADER left power (or is expected to, if still ruling at end of period)."""
    ),
    
    # =========================================================================
    # COLLEGIALITY
    # =========================================================================
    "collegiality": IndicatorConfig(
        name="collegiality",
        display_name="collegiality status",
        specialization="executive power structures and decision-making processes",
        labels=["0", "1"],
        task_description="whether decision-making within the executive was collegial during a specific leader's reign",
        
        definition="""
## Definition of Collegiality

Collegiality refers to whether **decision-making within the executive is shared by members of a formally constituted body**. Where decision-making is collegial, we assume that executive power is to some extent constrained.

**Collegial (1):**
- Decision-making is shared by members of a formally constituted body
- Examples: cabinets, military juntas, Roman consuls, regencies, Switzerland's presidency (all-party cabinet)
- Decisions require collective deliberation and agreement
- Power is distributed among multiple members of the executive body

**Non-Collegial (0):**
- A single actor dominates decision-making
- Executive body exists but is dominated by one person
- Collective bodies that are formally collegial but actually controlled by a single actor
- Advisory bodies without actual decision-making power

## Critical Distinction: De Facto vs De Jure

**Wherever de facto power differs from de jure power, it is the former (actual practice) that should govern coding decisions.**

- If a body is formally collegial but actually dominated by a single actor → Code as **0**
- If informal collegial practices exist despite formal single leadership → Consider actual power dynamics

## Analysis Process

1. Identify the formal executive structure during this leader's reign
2. Determine if there was a formally constituted collegial body
3. **Critically assess**: Was decision-making actually shared, or did one person dominate?
4. Focus on de facto (actual) power, not de jure (formal) arrangements
""",
        
        task_instruction="""Determine whether decision-making in the executive was collegial (1) or non-collegial (0) during this leader's reign.

Remember:
- Collegial (1): Decisions shared by members of a formally constituted body (e.g., cabinet, junta, consuls)
- Non-Collegial (0): Single actor dominates, OR collegial body is dominated by one person

**CRITICAL:** Code based on de facto (actual) power, not de jure (formal) arrangements. If a body is formally collegial but one person dominates, code as 0.""",
        
        coding_rule_reminder="""⚠️ **IMPORTANT:** Focus on ACTUAL decision-making practice during THIS LEADER'S REIGN, not formal structures."""
    ),
    
    # =========================================================================
    # SEPARATE POWERS
    # =========================================================================
    "separate_powers": IndicatorConfig(
        name="separate_powers",
        display_name="separate powers status",
        specialization="constitutional structures and checks and balances",
        labels=["0", "1"],
        task_description="whether power at the top was divided between multiple independent organizations during a specific leader's reign",
        
        definition="""
## Definition of Separate Powers

Separate powers refers to whether **power at the top is divided between multiple independent organizations**. Where such division exists, we assume that executive power is to some extent constrained. This may also be referred to as horizontal accountability or checks and balances.

**Separate Powers (1):**
- Power is divided between multiple independent organizations
- Examples include:
  * Executive chosen separately from legislature (and not responsible to it)
  * Independent judiciary with capacity to check the executive
  * Separately designated religious authority with checking power over executive
  * Military authority with ultimate or checking power over executive
- **Key requirements:**
  * (a) More than one organization has authority over policymaking
  * (b) These organizations are independent of each other

**Unitary Authority (0):**
- Power is concentrated in one organization
- No effective checks and balances between independent bodies
- System that looks like separate powers on paper but is entirely controlled by one organization
- All branches formally exist but are subordinate to the executive

## Critical Distinction: De Facto vs De Jure

**Wherever de facto power diverges from de jure power, we are concerned with the former (actual practice).**

- A system that looks like separate powers on paper but is in fact entirely controlled by one organization → Code as **0**
- Informal but effective checks on executive power → Consider actual power dynamics

## Analysis Process

1. Identify the formal institutional structure during this leader's reign
2. Determine if multiple organizations had authority over policymaking
3. Assess whether these organizations were truly independent of each other
4. **Critically assess**: Were checks and balances effective in practice, or merely nominal?
5. Focus on de facto (actual) power relationships, not de jure (formal) arrangements
""",
        
        task_instruction="""Determine whether power was divided between independent organizations (1) or concentrated in unitary authority (0) during this leader's reign.

Remember:
- Separate Powers (1): Multiple independent organizations with authority over policymaking (e.g., independent legislature, judiciary, or religious/military authority)
- Unitary Authority (0): Power concentrated in one organization, OR separate branches exist but are controlled by the executive

**CRITICAL:** Code based on de facto (actual) power, not de jure (formal) arrangements. If branches exist on paper but one organization controls everything, code as 0.""",
        
        coding_rule_reminder="""⚠️ **IMPORTANT:** Focus on ACTUAL power relationships during THIS LEADER'S REIGN, not formal constitutional structures."""
    ),
}


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def build_system_prompt(indicator: str) -> str:
    """
    Build system prompt for an indicator.
    
    Args:
        indicator: Name of the indicator
        
    Returns:
        Complete system prompt string
    """
    if indicator not in INDICATOR_CONFIGS:
        raise ValueError(f"Unknown indicator: {indicator}. Must be one of {list(INDICATOR_CONFIGS.keys())}")
    
    config = INDICATOR_CONFIGS[indicator]
    
    # Build header
    header = SYSTEM_PROMPT_HEADER.format(
        specialization=config.specialization,
        task_description=config.task_description
    )
    
    # Build output format
    label_display = f'"{config.labels[0]}"' if len(config.labels) == 2 else f'one of {config.labels}'
    output_format = OUTPUT_FORMAT_TEMPLATE.format(
        indicator=config.name,
        valid_labels=label_display
    )
    
    return header + config.definition + output_format


def build_user_prompt(
    indicator: str,
    polity: str,
    name: str,
    start_year: int,
    end_year: int
) -> str:
    """
    Build user prompt for an indicator with leader-level information.
    
    Args:
        indicator: Name of the indicator
        polity: Name of the polity
        name: Name of the leader
        start_year: Start year of the leader's reign
        end_year: End year of the leader's reign
        
    Returns:
        Complete user prompt string
    """
    if indicator not in INDICATOR_CONFIGS:
        raise ValueError(f"Unknown indicator: {indicator}. Must be one of {list(INDICATOR_CONFIGS.keys())}")
    
    config = INDICATOR_CONFIGS[indicator]
    
    # Format label display for JSON example
    if len(config.labels) == 2:
        label_format = f"{config.labels[0]} or {config.labels[1]}"
    else:
        label_format = ", ".join(config.labels[:-1]) + f", or {config.labels[-1]}"
    
    # Handle tenure's special coding rule reminder with years
    coding_rule = config.coding_rule_reminder
    if indicator == "tenure":
        coding_rule = coding_rule.format(start_year=start_year, end_year=end_year)
    
    return USER_PROMPT_TEMPLATE.format(
        indicator_display=config.display_name,
        polity=polity,
        name=name,
        start_year=start_year,
        end_year=end_year,
        task_instruction=config.task_instruction,
        coding_rule_reminder=coding_rule,
        indicator=config.name,
        label_format=label_format
    )


# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_prompt(
    indicator: str,
    polity: str,
    name: str,
    start_year: int,
    end_year: int
) -> Tuple[str, str]:
    """
    Get system and user prompts for a specific indicator (leader-level).
    
    Args:
        indicator: One of sovereign, assembly, appointment, tenure, exit, collegiality, separate_powers
        polity: Name of the polity
        name: Name of the leader
        start_year: Start year of the leader's reign
        end_year: End year of the leader's reign
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = build_system_prompt(indicator)
    user_prompt = build_user_prompt(indicator, polity, name, start_year, end_year)
    return system_prompt, user_prompt


def get_all_indicators() -> List[str]:
    """Return list of all indicator names."""
    return list(INDICATOR_CONFIGS.keys())


def get_indicator_labels(indicator: str) -> List[str]:
    """Return valid labels for an indicator."""
    if indicator not in INDICATOR_CONFIGS:
        raise ValueError(f"Unknown indicator: {indicator}")
    return INDICATOR_CONFIGS[indicator].labels


def get_indicator_config(indicator: str) -> IndicatorConfig:
    """Return the full configuration for an indicator."""
    if indicator not in INDICATOR_CONFIGS:
        raise ValueError(f"Unknown indicator: {indicator}")
    return INDICATOR_CONFIGS[indicator]


# =============================================================================
# CHAIN OF VERIFICATION (CoVe) QUESTIONS
# =============================================================================

COVE_QUESTION_TEMPLATES: Dict[str, List[str]] = {
    "sovereign": [
        "Was {polity} a colony, protectorate, or vassal of another power during {name}'s reign ({start_year}-{end_year})?",
        "Did {polity} conduct independent foreign policy under {name}?",
        "Did {polity} pay tribute to any external power during {name}'s rule?"
    ],
    "assembly": [
        "What legislative or deliberative bodies existed in {polity} during {name}'s reign ({start_year}-{end_year})?",
        "Did any assembly have authority over taxation, legislation, or leader selection under {name}?",
        "How frequently did legislative bodies convene during {name}'s rule?"
    ],
    "appointment": [
        "How did {name} come to power in {polity}?",
        "Was {name}'s succession hereditary, elected, or appointed?",
        "What role did assemblies or councils play in {name}'s selection?"
    ],
    "tenure": [
        "How long did {name} rule {polity}?",
        "What was the typical tenure length for rulers in {polity} during this era?",
        "Were there term limits or expected tenure lengths in {polity} during {name}'s time?"
    ],
    "exit": [
        "How did {name} leave power in {polity}?",
        "Did {name} die in office, abdicate voluntarily, or face removal?",
        "Were there institutional mechanisms for succession when {name}'s rule ended?"
    ],
    "collegiality": [
        "What was the formal executive structure in {polity} during {name}'s reign ({start_year}-{end_year})?",
        "Did {name} dominate decision-making, or were decisions genuinely shared among multiple actors?",
        "Were there co-rulers, regents, or formally constituted bodies sharing executive power with {name}?"
    ],
    "separate_powers": [
        "What independent institutions existed in {polity} during {name}'s reign ({start_year}-{end_year})?",
        "Did an independent judiciary have the capacity to check {name}'s power?",
        "Did multiple independent organizations have authority over policymaking under {name}?"
    ]
}


def get_cove_questions(
    indicator: str,
    polity: str,
    name: str,
    start_year: int,
    end_year: int
) -> List[str]:
    """
    Get Chain of Verification questions for an indicator (leader-level).
    
    Args:
        indicator: Name of the indicator
        polity: Name of the polity
        name: Name of the leader
        start_year: Start year of the leader's reign
        end_year: End year of the leader's reign
        
    Returns:
        List of formatted questions
    """
    if indicator not in COVE_QUESTION_TEMPLATES:
        raise ValueError(f"No CoVe questions defined for indicator: {indicator}")
    
    return [
        q.format(polity=polity, name=name, start_year=start_year, end_year=end_year)
        for q in COVE_QUESTION_TEMPLATES[indicator]
    ]


# =============================================================================
# BATCH PROCESSING HELPERS
# =============================================================================

def get_all_prompts_for_leader(
    polity: str,
    name: str,
    start_year: int,
    end_year: int,
    indicators: Optional[List[str]] = None
) -> Dict[str, Tuple[str, str]]:
    """
    Get prompts for all (or specified) indicators for a single leader.
    
    Args:
        polity: Name of the polity
        name: Name of the leader
        start_year: Start year of the leader's reign
        end_year: End year of the leader's reign
        indicators: Optional list of indicators to include (default: all)
        
    Returns:
        Dictionary mapping indicator name to (system_prompt, user_prompt) tuple
    """
    if indicators is None:
        indicators = get_all_indicators()
    
    return {
        ind: get_prompt(ind, polity, name, start_year, end_year)
        for ind in indicators
    }


def get_expected_output_schema(indicators: Optional[List[str]] = None) -> Dict[str, dict]:
    """
    Get the expected output schema for specified indicators.
    
    Args:
        indicators: Optional list of indicators (default: all)
        
    Returns:
        Dictionary describing expected output fields per indicator
    """
    if indicators is None:
        indicators = get_all_indicators()
    
    return {
        ind: {
            "fields": {
                ind: f"string, one of {get_indicator_labels(ind)}",
                "reasoning": "string",
                "confidence_score": "integer, 1-100"
            }
        }
        for ind in indicators
    }