"""
Single Prompt Builder

Combines multiple indicators into a single prompt.
Efficient (fewer API calls) but may cause cross-indicator contamination.

Indicators (based on "Growth of Executive Constraints" paper):
- sovereign (0/1)
- federalism (0/1)
- checks (0/1/2)
- collegiality (0/1)
- assembly (0/1/2/3)
- entry (0-10) — fine-grained, 11 categories
- entry_4 (0-3) — coarse, independent query for robustness check
- exit (0-15) — fine-grained, 16 categories
- exit_4 (0-3) — coarse, independent query for robustness check

NOTE: elections is a downstream indicator derived via post_processing.py.
      It is NOT included in this prompt (hard-coded 0 when assembly != 2).
NOTE: tenure is excluded — it is a continuous variable, not a categorical label.
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from prompts.base_builder import PromptOutput


# =============================================================================
# INDICATOR CONFIGURATIONS
# =============================================================================

@dataclass
class IndicatorConfig:
    """Configuration for a political indicator."""
    name: str
    display_name: str
    labels: List[str]
    summary: str


INDICATOR_CONFIGS: Dict[str, IndicatorConfig] = {

    # =========================================================================
    # SOVEREIGN
    # =========================================================================
    "sovereign": IndicatorConfig(
        name="sovereign",
        display_name="Sovereignty",
        labels=["0", "1"],
        summary=(
            "Sovereignty refers to a polity's ability to conduct domestic affairs without foreign interference. "
            "Because we are concerned with executive constraints that arise from within a polity, foreign influences are usually corrupting. "
            "An executive obeisant to foreign diktats must be (ceteris paribus) less sensitive to domestic actors.\n\n"
            "Coding:\n"
            "• 0 = Semi-sovereign. Examples: colony, protectorate, distant or overseas territory (not fully incorporated into the metropole).\n"
            "• 1 = Sovereign. Examples: city-states, nation-states, empires, republics, monarchies, tributary states "
            "(so long as they held primary responsibility for domestic affairs), states within the Peloponnesian League, "
            "the Hanseatic League, the Holy Roman Empire, and the European Union, any state recognized by European powers "
            "or international law in the modern era, as well as a few that enjoy de facto sovereignty such as Somaliland and Taiwan."
        )
    ),

    # =========================================================================
    # FEDERALISM
    # =========================================================================
    "federalism": IndicatorConfig(
        name="federalism",
        display_name="Federalism",
        labels=["0", "1"],
        summary=(
            "Federalism refers generally to a division of sovereignty between central and local units, "
            "reserving some important powers to the latter and promising a relatively decentralized mode of governance. "
            "Typically, rights enjoyed by local governance units are enshrined in a constitution. "
            "Commonly, localities are represented in a legislative chamber at the polity level. "
            "However constituted, a federal polity may be expected to impose constraints on the executive.\n\n"
            "Coding:\n"
            "• 0 = Non-federal.\n"
            "• 1 = Federal. Includes confederations, leagues, and composite monarchies. "
            "Historical examples: Achaean League, Aetolian League, Lycian League, Boeotian League, Old Swiss Confederacy, "
            "Dutch Republic, Holy Roman Empire, Iroquois Confederacy, Hanseatic League, Polish-Lithuanian Commonwealth, Tokugawa Japan. "
            "Contemporary examples: Canada, Germany, India, United States, and the European Union."
        )
    ),

    # =========================================================================
    # CHECKS
    # =========================================================================
    "checks": IndicatorConfig(
        name="checks",
        display_name="Checks",
        labels=["0", "1", "2"],
        summary=(
            "Effective checks exist when independent bodies adjacent to the executive have the capacity to resist actions taken by the executive. "
            "Countervailing powers might be exercised by legislative, judicial, religious, military, caste, aristocratic, or bureaucratic organizations, "
            "by a privy council or council of elders, by a head of state (if not part of the executive), or by constituent units "
            "(regional and local governments, tribes, clans et al.). "
            "Actors such as the media, civil society groups, or ordinary citizens are not considered here as their influence is apt to be sporadic.\n\n"
            "Coding:\n"
            "• 0 = None. There are no independent organizations with the capacity to resist the executive.\n"
            "• 1 = Partial. Independent organizations may, on occasion, resist the executive. But their power is informal and/or their interventions are rare.\n"
            "• 2 = Full. Independent organizations have the capacity to regularly and effectively resist the executive. "
            "This includes settings where the checking body must approve legislation or has veto rights (e.g., judicial review)."
        )
    ),

    # =========================================================================
    # COLLEGIALITY
    # =========================================================================
    "collegiality": IndicatorConfig(
        name="collegiality",
        display_name="Collegiality",
        labels=["0", "1"],
        summary=(
            "Where power at the apex of a polity is exercised in a collegial manner, decisionmaking power is shared among a number of actors. "
            "There may be a titular head (e.g., director or chair) but the other members of the group are regarded as co-equals, partners, collaborators. "
            "Collegial decisionmaking is consultative – lateral rather than vertical. "
            "Wherever de facto practices differs from de jure rules, it is the former that governs coding decisions. "
            "That is, if a body is formally collegial but actually dominated by a single actor it should not be coded as collegial.\n\n"
            "Coding:\n"
            "• 0 = Non-collegial. Examples: most presidencies, monarchies, and dictatorships.\n"
            "• 1 = Collegial. Examples: most cabinets, some military juntas, many regencies, Roman consuls, Switzerland's modern presidency."
        )
    ),

    # =========================================================================
    # ASSEMBLY
    # =========================================================================
    "assembly": IndicatorConfig(
        name="assembly",
        display_name="Assembly",
        labels=["0", "1", "2", "3"],
        summary=(
            "An assembly is a body designed to govern (directly), to select leaders, or to assist in governing. "
            "It may be advisory (a council of selected elites), representative (a legislature), or inclusive of all citizens (a popular assembly).\n\n"
            "Coding:\n"
            "• 0 = None.\n"
            "• 1 = Council. A small advisory council appointed by the ruler, which may not enjoy much autonomy but is nonetheless institutionalized "
            "(has a recognized name, meets regularly, and has a fairly stable membership). "
            "Examples: noble or aristocratic councils, privy councils, dynastic councils, Ottoman divan.\n"
            "• 2 = Legislature. A large representative body that plays some role in policymaking or leadership selection, de jure or de facto. "
            "Examples: estates assemblies in premodern Europe, the Hwabaek Council in Korea during the Silla Dynasty, legislatures in modern governments.\n"
            "• 3 = Popular assembly. An assembly that includes most citizens of the polity or a representative sample chosen by lot. "
            "Examples: Ecclesia in ancient Athens, Landsgemeinden in modern Swiss cantons."
        )
    ),

    # =========================================================================
    # ENTRY (Fine-grained, 11 categories)
    # =========================================================================
    "entry": IndicatorConfig(
        name="entry",
        display_name="Entry",
        labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        summary=(
            "The manner in which executives enter office is widely regarded as a key to their power while in office. "
            "Leader selection indicates the sorts of constraints executives are likely to face.\n\n"
            "Coding:\n"
            "• 0 = Through the threat or application of force, such as a coup or rebellion\n"
            "• 1 = Appointed by a foreign power\n"
            "• 2 = Appointed by the ruling party (in a one-party system)\n"
            "• 3 = Appointed by a royal council (either members of the royal family or conclave of aristocrats)\n"
            "• 4 = Through hereditary succession\n"
            "• 5 = Appointed by the military\n"
            "• 6 = Appointed by the legislature\n"
            "• 7 = Appointed by the head of state\n"
            "• 8 = Appointed by the head of government\n"
            "• 9 = Directly through a popular election (regardless of the extension of the suffrage)\n"
            "• 10 = Other (including clerical bodies such as the College of Cardinals)"
        )
    ),

    # =========================================================================
    # ENTRY_4 (Coarse, 4 categories - independent robustness query)
    # =========================================================================
    "entry_4": IndicatorConfig(
        name="entry_4",
        display_name="Entry (4-category)",
        labels=["0", "1", "2", "3"],
        summary=(
            "Coarse classification of executive entry — an independent query for robustness check.\n\n"
            "Coding:\n"
            "• 0 = Irregular. By force, foreign actor, military junta.\n"
            "• 1 = Hereditary. Institutionalized process by which a designated family heir inherits office.\n"
            "• 2 = Appointment. Institutionalized process of appointment by a domestic body that is not democratically elected "
            "such as a royal council, monarch, or ruling party in a one-party state.\n"
            "• 3 = Election. Direct popular election, selection by elective body (e.g., legislature or electoral college), "
            "by local governments, or by lot. The extent of suffrage or eligibility is irrelevant."
        )
    ),

    # =========================================================================
    # EXIT (Fine-grained, 16 categories)
    # =========================================================================
    "exit": IndicatorConfig(
        name="exit",
        display_name="Exit",
        labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
        summary=(
            "The circumstances of an executive's departure from office says a lot about a leader's prerogative while in office.\n\n"
            "Coding:\n"
            "• 0 = Abdicated or retired voluntarily but NOT due to ill health\n"
            "• 1 = Other regular exit (e.g., term limits or defeat in election)\n"
            "• 2 = Transition to another office type/typology (by regular procedures)\n"
            "• 3 = Died (of disease or accident) on campaign in civil war\n"
            "• 4 = Died (of disease or accident) on campaign in foreign war\n"
            "• 5 = Died of other natural causes\n"
            "• 6 = Retired due to ill health\n"
            "• 7 = Suicide\n"
            "• 8 = Deposed by domestic actors\n"
            "• 9 = Assassinated or forced suicide\n"
            "• 10 = Died in battle in civil war\n"
            "• 11 = Died in battle in foreign war\n"
            "• 12 = Transition to another office by irregular procedures\n"
            "• 13 = Through deposition by a foreign state\n"
            "• 14 = Unknown\n"
            "• 15 = Still in office"
        )
    ),

    # =========================================================================
    # EXIT_4 (Coarse, 4 categories - independent robustness query)
    # =========================================================================
    "exit_4": IndicatorConfig(
        name="exit_4",
        display_name="Exit (4-category)",
        labels=["0", "1", "2", "3"],
        summary=(
            "Coarse classification of executive exit — an independent query for robustness check.\n\n"
            "Coding:\n"
            "• 0 = Irregular. Executive is forcibly removed or retires under duress.\n"
            "• 1 = Natural. Executive retires due to ill health or dies in office.\n"
            "• 2 = Voluntary. Executive voluntarily retires or abdicates (not due to ill health).\n"
            "• 3 = Institutionalized. Executive exits at (or near) the expiration of a term, after electoral defeat, "
            "or as part of a transition to another government office."
        )
    ),
}


# =============================================================================
# MAPPING TABLES (reference only — entry_4 and exit_4 are queried independently)
# These tables are retained for post-hoc consistency checks and future use.
# The pipeline does NOT use these mappings; it queries entry_4 and exit_4 directly.
# =============================================================================

ENTRY_TO_ENTRY_4: Dict[str, str] = {
    "0": "0",   # Force → Irregular
    "1": "0",   # Foreign power → Irregular
    "2": "2",   # Ruling party → Appointment
    "3": "2",   # Royal council → Appointment
    "4": "1",   # Hereditary → Hereditary
    "5": "0",   # Military → Irregular
    "6": "3",   # Legislature → Election
    "7": "2",   # Head of state → Appointment
    "8": "2",   # Head of government → Appointment
    "9": "3",   # Popular election → Election
    "10": "3",  # Other (clerical) → Election
}

EXIT_TO_EXIT_4: Dict[str, str | None] = {
    "0": "2",   # Abdicated/retired voluntarily (not ill health) → Voluntary
    "1": "3",   # Other regular exit (term limits, defeat) → Institutionalized
    "2": "3",   # Transition to another office (regular) → Institutionalized
    "3": "1",   # Died on campaign, civil war (disease/accident) → Natural
    "4": "1",   # Died on campaign, foreign war (disease/accident) → Natural
    "5": "1",   # Died of natural causes → Natural
    "6": "1",   # Retired due to ill health → Natural
    "7": "1",   # Suicide → Natural
    "8": "0",   # Deposed by domestic actors → Irregular
    "9": "0",   # Assassinated/forced suicide → Irregular
    "10": "0",  # Died in battle, civil war → Irregular
    "11": "0",  # Died in battle, foreign war → Irregular
    "12": "0",  # Transition to another office (irregular) → Irregular
    "13": "0",  # Deposed by foreign state → Irregular
    "14": None, # Unknown → N/A
    "15": None, # Still in office → N/A
}


# =============================================================================
# COMPACT SUMMARIES FOR COMBINED PROMPTS
# =============================================================================

INDICATOR_SUMMARIES: Dict[str, str] = {
    "sovereign": (
        "Sovereignty: ability to conduct domestic affairs without foreign interference. "
        "(0) Semi-sovereign — colony, protectorate, overseas territory; "
        "(1) Sovereign — city-states, nation-states, empires, tributary states with primary domestic responsibility."
    ),

    "federalism": (
        "Federalism: division of sovereignty between central and local units. "
        "(0) Non-federal; "
        "(1) Federal — includes confederations, leagues, composite monarchies. "
        "Examples: Achaean League, Dutch Republic, Holy Roman Empire, Iroquois Confederacy, US, EU."
    ),

    "checks": (
        "Checks: capacity of independent bodies to resist executive actions. "
        "(0) None — no independent organizations can resist the executive; "
        "(1) Partial — independent organizations may occasionally resist, but informally or rarely; "
        "(2) Full — independent organizations regularly and effectively resist, including veto rights or judicial review."
    ),

    "collegiality": (
        "Collegiality: whether decisionmaking power is shared among multiple actors at the apex. "
        "Code based on de facto practice, not de jure rules. "
        "(0) Non-collegial — single actor dominates (most presidencies, monarchies, dictatorships); "
        "(1) Collegial — power shared among co-equals (cabinets, juntas, regencies, Roman consuls, Swiss presidency)."
    ),

    "assembly": (
        "Assembly: body designed to govern, select leaders, or assist in governing. "
        "(0) None; "
        "(1) Council — small advisory council, institutionalized (privy councils, dynastic councils, Ottoman divan); "
        "(2) Legislature — large representative body with policymaking role (estates assemblies, modern legislatures); "
        "(3) Popular assembly — includes most citizens or chosen by lot (Athenian Ecclesia, Swiss Landsgemeinden)."
    ),

    "entry": (
        "Entry: manner of executive entry into office. "
        "(0) Force (coup, rebellion); (1) Foreign power; (2) Ruling party (one-party system); "
        "(3) Royal council; (4) Hereditary succession; (5) Military; "
        "(6) Legislature; (7) Head of state; (8) Head of government; "
        "(9) Direct popular election; (10) Other (e.g., clerical bodies)."
    ),

    "entry_4": (
        "Entry (4-category): coarse classification, queried independently for robustness. "
        "(0) Irregular — force, foreign power, military; "
        "(1) Hereditary — designated family heir; "
        "(2) Appointment — royal council, head of state/government, ruling party; "
        "(3) Election — popular election, legislature, lot."
    ),

    "exit": (
        "Exit: circumstances of executive departure from office. "
        "(0) Abdicated/retired voluntarily (not ill health); (1) Other regular exit (term limits, defeat); "
        "(2) Transition to another office (regular); (3) Died on campaign, civil war (disease/accident); "
        "(4) Died on campaign, foreign war (disease/accident); (5) Died of natural causes; "
        "(6) Retired due to ill health; (7) Suicide; "
        "(8) Deposed by domestic actors; (9) Assassinated/forced suicide; "
        "(10) Died in battle, civil war; (11) Died in battle, foreign war; "
        "(12) Transition to another office (irregular); (13) Deposed by foreign state; "
        "(14) Unknown; (15) Still in office."
    ),

    "exit_4": (
        "Exit (4-category): coarse classification, queried independently for robustness. "
        "(0) Irregular — forcibly removed or under duress; "
        "(1) Natural — died in office or retired due to ill health; "
        "(2) Voluntary — voluntarily retired/abdicated (not ill health); "
        "(3) Institutionalized — term expiration, electoral defeat, regular transition."
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_indicators(include_robustness: bool = True) -> List[str]:
    """
    Return the default indicator list.

    Args:
        include_robustness: Whether to include entry_4 and exit_4 (default True —
                            both are queried independently as robustness checks).
    """
    base = ["sovereign", "federalism", "checks", "collegiality", "assembly", "entry", "exit"]

    if include_robustness:
        base.extend(["entry_4", "exit_4"])

    return base


def get_categorical_indicators() -> List[str]:
    """Return all categorical (non-continuous) indicator names."""
    return list(INDICATOR_CONFIGS.keys())


def map_entry_to_4(entry_value: str) -> Optional[str]:
    """Map fine-grained entry value to 4-category (reference only, not used by pipeline)."""
    return ENTRY_TO_ENTRY_4.get(entry_value)


def map_exit_to_4(exit_value: str) -> Optional[str]:
    """Map fine-grained exit value to 4-category (reference only, not used by pipeline)."""
    return EXIT_TO_EXIT_4.get(exit_value)


# =============================================================================
# VERSION 1 — SinglePromptBuilder (full definitions, concise framing)
# =============================================================================

class SinglePromptBuilder:
    """
    Combines all selected indicators into a single prompt.

    Builds one comprehensive prompt asking the LLM to predict multiple
    indicators simultaneously. More efficient (fewer API calls) but may
    allow indicators to influence each other's predictions.
    """

    def __init__(
        self,
        indicators: Optional[List[str]] = None,
        reasoning: bool = True,
        include_robustness: bool = True
    ):
        if indicators is None:
            indicators = get_all_indicators(include_robustness=include_robustness)
        self.indicators = indicators
        self.reasoning = reasoning

    def build(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> List[PromptOutput]:
        return [PromptOutput(
            system_prompt=self._build_system_prompt(),
            user_prompt=self._build_user_prompt(polity, name, start_year, end_year),
            indicators=self.indicators,
            metadata={'mode': 'single', 'version': 'v1'},
        )]

    def _build_system_prompt(self) -> str:
        prompt = (
            "You are a political scientist coding executive constraints for historical leaders.\n\n"
            "**Core rule:** Code de facto (actual) practice, not de jure (formal) arrangements. "
            "Focus on THIS specific leader's reign. When evidence is uncertain, prefer the more conservative (lower) code.\n\n"
            "## Indicator Definitions\n\n"
        )

        for ind in self.indicators:
            if ind in INDICATOR_CONFIGS:
                config = INDICATOR_CONFIGS[ind]
                prompt += f"### {config.display_name}\n\n{config.summary}\n\n"

        prompt += "## Output Format\n\nRespond with ONLY a valid JSON object — no markdown fences, no extra text.\n\nFields required:\n"

        for ind in self.indicators:
            if ind in INDICATOR_CONFIGS:
                config = INDICATOR_CONFIGS[ind]
                labels_str = ", ".join([f'"{l}"' for l in config.labels])
                prompt += f'- "{ind}": one of {labels_str}\n'
                if self.reasoning:
                    prompt += f'- "{ind}_reasoning": brief justification (string)\n'
                prompt += f'- "{ind}_confidence": integer 1–100\n'

        return prompt

    def _build_user_prompt(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> str:
        reign = f"{start_year}-{end_year if end_year is not None else 'present'}"
        return (
            f"Classify the following leader on all indicators:\n\n"
            f"**Polity:** {polity}\n"
            f"**Leader:** {name}\n"
            f"**Reign:** {reign}\n\n"
            f"Return a single JSON object with all required fields."
        )


# =============================================================================
# VERSION 2 — SinglePromptBuilderV2 (directive framing, tabular definitions)
# =============================================================================

class SinglePromptBuilderV2:
    """
    Alternative prompt framing: authoritative expert annotator role with
    tabular indicator reference and explicit step-by-step instructions.
    Keeps all definitions; varies structure and persona.
    """

    def __init__(
        self,
        indicators: Optional[List[str]] = None,
        reasoning: bool = True,
        include_robustness: bool = True
    ):
        if indicators is None:
            indicators = get_all_indicators(include_robustness=include_robustness)
        self.indicators = indicators
        self.reasoning = reasoning

    def build(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> List[PromptOutput]:
        return [PromptOutput(
            system_prompt=self._build_system_prompt(),
            user_prompt=self._build_user_prompt(polity, name, start_year, end_year),
            indicators=self.indicators,
            metadata={'mode': 'single', 'version': 'v2'},
        )]

    def _build_system_prompt(self) -> str:
        prompt = (
            "You are an expert annotator for the \"Growth of Executive Constraints\" dataset. "
            "Your job is to assign precise numerical codes to historical leaders based on the indicator "
            "definitions below. These codes will be used in academic research, so accuracy is paramount.\n\n"
            "**Annotation Rules:**\n"
            "1. Always code actual (de facto) behavior, never formal (de jure) arrangements.\n"
            "2. Evaluate conditions as they existed during THIS leader's specific reign.\n"
            "3. When evidence is ambiguous or insufficient, assign the more conservative (lower) code.\n"
            "4. Each indicator is independent — do not let your assessment of one influence another.\n\n"
            "## Indicator Reference\n\n"
        )

        for ind in self.indicators:
            if ind in INDICATOR_CONFIGS:
                config = INDICATOR_CONFIGS[ind]
                prompt += f"**{config.display_name}** (`{ind}`)\n"
                prompt += f"{config.summary}\n\n"

        prompt += (
            "## Step-by-Step Instructions\n\n"
            "For each indicator:\n"
            "1. Recall relevant historical facts about this leader's reign.\n"
            "2. Match those facts to the indicator definitions above.\n"
            "3. Assign the appropriate code.\n"
            "4. Record your confidence (1–100) based on quality of evidence.\n\n"
            "## Required JSON Output\n\n"
            "Output ONLY a JSON object with these fields (no markdown, no preamble):\n"
        )

        for ind in self.indicators:
            if ind in INDICATOR_CONFIGS:
                config = INDICATOR_CONFIGS[ind]
                labels_str = " | ".join(config.labels)
                prompt += f'- "{ind}": "{labels_str}" (string)\n'
                if self.reasoning:
                    prompt += f'- "{ind}_reasoning": concise evidence-based justification\n'
                prompt += f'- "{ind}_confidence": 1–100 (integer)\n'

        return prompt

    def _build_user_prompt(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> str:
        reign = f"{start_year}–{end_year if end_year is not None else 'present'}"
        return (
            f"Annotate the following leader:\n\n"
            f"Polity: {polity}\n"
            f"Leader: {name}\n"
            f"Reign: {reign}\n\n"
            f"Apply the indicator definitions strictly. Return your annotation as a single JSON object."
        )


# =============================================================================
# VERSION 3 — SinglePromptBuilderV3 (compact/minimal, uses summary strings)
# =============================================================================

class SinglePromptBuilderV3:
    """
    Token-efficient prompt using compact one-line indicator summaries.
    Trades verbosity for brevity — useful for cost/token sensitivity testing.
    All definitions are preserved but condensed to single lines per indicator.
    """

    def __init__(
        self,
        indicators: Optional[List[str]] = None,
        reasoning: bool = True,
        include_robustness: bool = True
    ):
        if indicators is None:
            indicators = get_all_indicators(include_robustness=include_robustness)
        self.indicators = indicators
        self.reasoning = reasoning

    def build(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> List[PromptOutput]:
        return [PromptOutput(
            system_prompt=self._build_system_prompt(),
            user_prompt=self._build_user_prompt(polity, name, start_year, end_year),
            indicators=self.indicators,
            metadata={'mode': 'single', 'version': 'v3'},
        )]

    def _build_system_prompt(self) -> str:
        prompt = (
            "You are a political historian classifying executive constraints for historical leaders. "
            "Code based on de facto (actual) practice, not de jure arrangements. "
            "Focus on this specific leader's reign. Prefer conservative codes when uncertain.\n\n"
            "## Indicator Definitions\n\n"
        )

        for ind in self.indicators:
            if ind in INDICATOR_SUMMARIES:
                prompt += f"**{ind}**: {INDICATOR_SUMMARIES[ind]}\n\n"
            elif ind in INDICATOR_CONFIGS:
                config = INDICATOR_CONFIGS[ind]
                labels_str = "/".join(config.labels)
                prompt += f"**{ind}** ({labels_str}): {config.summary[:200]}...\n\n"

        prompt += "## Output\n\nRespond with ONLY a valid JSON object (no markdown fences).\n"

        return prompt

    def _build_user_prompt(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> str:
        reign = f"{start_year}-{end_year if end_year is not None else 'present'}"

        fields = []
        for ind in self.indicators:
            if ind in INDICATOR_CONFIGS:
                config = INDICATOR_CONFIGS[ind]
                labels_str = "/".join(config.labels)
                fields.append(f'"{ind}" ({labels_str})')
                if self.reasoning:
                    fields.append(f'"{ind}_reasoning"')
                fields.append(f'"{ind}_confidence" (1-100)')

        fields_str = ", ".join(fields)

        return (
            f"Classify: **{polity}** | **{name}** | **{reign}**\n\n"
            f"Return JSON with: {fields_str}"
        )


# =============================================================================
# COMPACT BUILDER FUNCTION (convenience wrapper around V3)
# =============================================================================

def build_compact_prompt(
    polity: str,
    name: str,
    start_year: int,
    end_year: Optional[int],
    indicators: Optional[List[str]] = None,
    include_reasoning: bool = False
) -> Tuple[str, str]:
    """
    Build a compact combined prompt (token-efficient, uses summary strings).

    Args:
        polity: Name of the polity
        name: Name of the leader
        start_year: Start year of reign
        end_year: End year of reign
        indicators: List of indicators (default: all)
        include_reasoning: Whether to request reasoning fields

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    builder = SinglePromptBuilderV3(
        indicators=indicators,
        reasoning=include_reasoning
    )
    return builder.build(polity, name, start_year, end_year)
