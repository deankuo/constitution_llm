"""
Single Prompt Builder

Combines multiple indicators into a single prompt.
Efficient (fewer API calls) but may cause cross-indicator contamination.

Indicators:
- sovereign (0/1)
- federalism (0/1)
- checks (0-9, multi-select) — output as JSON array of selected values
- collegiality (0/1)
- petition (0/1)
- assembly (0/1/2/3)
- entry (0-10) — fine-grained, 11 categories
- exit (0-15) — fine-grained, 16 categories
- symbolism (0/1/2/3, non-monotonic)

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
    multi_select: bool = False


INDICATOR_CONFIGS: Dict[str, IndicatorConfig] = {

    # =========================================================================
    # SOVEREIGN
    # =========================================================================
    "sovereign": IndicatorConfig(
        name="sovereign",
        display_name="Sovereignty",
        labels=["0", "1"],
        summary=(
            "IMPORTANT: This scale is NON-MONOTONIC with respect to leader power — a higher code does "
            "NOT mean more actual power. Codes 0-2 rise with the grandeur of the office, but code 3 "
            "marks largely ceremonial figureheads with little real power. Code the trappings and "
            "self-presentation of the office, NOT the leader's actual power. (Purely ceremonial heads "
            "who are not paramount leaders are excluded from the sample.)\n\n"
            "Symbolic power is reflected in the trappings of office and the presentation of self — "
            "e.g., a grandly appointed palace; regalia such as scepters, seals, thrones, and special "
            "garments; an aristocratic court and retinue; performance of spiritual rituals central to "
            "the polity; powers normally reserved for deities; special forms of address marking the "
            "ruler's apartness; and protections of the ruler's status such as lèse-majesté.\n\n"
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
            "Federalism refers to a division of sovereignty between central and local units, reserving "
            "some important powers to the latter and providing a relatively decentralized mode of "
            "governance. Such powers are typically enshrined in a constitution, and localities are "
            "often represented in a chamber at the polity level.\n\n"
            "Coding:\n"
            "• 0 = Non-federal.\n"
            "• 1 = Federal. Includes confederations, leagues, and composite monarchies. "
            "Historical examples: Achaean League, Aetolian League, Lycian League, Boeotian League, Old Swiss Confederacy, "
            "Dutch Republic, Holy Roman Empire, Iroquois Confederacy, Hanseatic League, Polish-Lithuanian Commonwealth, Tokugawa Japan. "
            "Contemporary examples: Canada, Germany, India, United States, and the European Union."
        )
    ),

    # =========================================================================
    # CHECKS (Multi-select, 10 categories — formerly checks_actors)
    # =========================================================================
    "checks": IndicatorConfig(
        name="checks",
        display_name="Checks (Actors)",
        labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        multi_select=True,
        summary=(
            "Effective checks exist when independent groups or bodies have the capacity to resist actions "
            "taken by the executive. Select all actors that provide a check on the executive's actions. "
            "The number of categories transfers into more effective checks.\n\n"
            "Coding (select all that apply):\n"
            "• 0 = None. There are no actors with this capacity or proclivity.\n"
            "• 1 = Local. Examples: clans, tribes, ethnic groups, local governance units, civil society groups, "
            "newspapers and other media.\n"
            "• 2 = Military. Examples: officers, specific branches of the military, a warrior caste (e.g., Samurai).\n"
            "• 3 = Clergy. Power is partly derived from their role as arbiters of a widely espoused religion or "
            "set of beliefs. Examples: established church, priests, or caste (of any religion or denomination).\n"
            "• 4 = Aristocracy. Examples: landed class, upper caste, nobility, hereditary elite, titled class, patriciate.\n"
            "• 5 = Bourgeoisie. Examples: middle class, urbanites, artisans, traders, merchants, commercial classes, "
            "capitalist class, business class, entrepreneurs, financiers, creditors.\n"
            "• 6 = Bureaucracy. Examples: civil servants, Confucian scholars who serve as top-level advisors and bureaucrats.\n"
            "• 7 = Judiciary. Examples: courts of law, tribunals, judicial bodies, adjudicative bodies, legal institutions.\n"
            "• 8 = Assembly. Examples: popular assembly, legislature, parliament.\n"
            "• 9 = Advisory council. Examples: royal council, council of state, regency council, privy council, "
            "council of elders."
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
            "Collegiality refers to power at the apex of a polity being exercised in a shared manner: "
            "decisionmaking power is divided among co-equal actors. There may be a titular head "
            "(e.g., a director or chair), but the other members are regarded as partners and "
            "collaborators; decisionmaking is lateral rather than vertical. Code de facto practice — "
            "if a body is formally collegial but actually dominated by a single actor, it is NOT collegial.\n\n"
            "Coding:\n"
            "• 0 = Non-collegial. Examples: most presidencies, monarchies, and dictatorships.\n"
            "• 1 = Collegial. Examples: most cabinets, some military juntas, many regencies, Roman consuls, Switzerland's modern presidency."
        )
    ),
    
    # =========================================================================
    # PETITIONS
    # =========================================================================
    "petition": IndicatorConfig(
        name="petition",
        display_name="Petition",
        labels=["0", "1"],
        summary=(
            "A petition is a regularized, institutionalized process by which a subject or citizen may "
            "lodge a complaint or request for redress with a high official (e.g., a head of state, "
            "legislature, court, or ombudsman). It may take the form of a face-to-face meeting "
            "(the \"bell of justice\" tradition), a letter, or an electronic communication, with one "
            "or many signatories.\n\n"
            "Coding:\n"
            "• 0 = No. Use of petition is extremely rare and probably ineffective, or there is no record "
            "of its existence.\n"
            "• 1 = Yes. Petitions are a fairly regular feature of political life."
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
            "An assembly is a body designed to govern directly, to select leaders, or to assist in "
            "governing. It may be advisory (a council of selected elites), representative (a "
            "legislature), or inclusive of all citizens (a popular assembly).\n\n"
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
        labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "99"],
        summary=(
            "IMPORTANT: Code the manner of entry at the START of this tenure, not later events. If it cannot be determined, use 99 (do NOT guess to avoid it).\n\n"
            "The manner in which the executive entered office. Code the manner of entry at the START "
            "of this leader's tenure (the accession in {start_year}).\n\n"
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
            "• 10 = Other (including clerical bodies such as the College of Cardinals)\n"
            "• 99 = Unknown (circumstances of entry are unknown)"
        )
    ),

    # =========================================================================
    # EXIT (Fine-grained, 16 categories)
    # =========================================================================
    "exit": IndicatorConfig(
        name="exit",
        display_name="Exit",
        labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "99"],
        summary=(
            "IMPORTANT: Code the manner of entry at the START of this tenure, not later events. If it cannot be determined, use 99 (do NOT guess to avoid it).\n\n"
            "The circumstances of the executive's departure from office. Code the manner of exit at "
            "the END of this leader's tenure (the departure in {end_year}).\n\n"
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
            "• 14 = Still in office\n"
            "• 99 = Unknown"
        )
    ),

    # =========================================================================
    # SYMBOLISM (formerly symbolic_power)
    # =========================================================================
    "symbolism": IndicatorConfig(
        name="symbolism",
        display_name="Symbolism",
        labels=["0", "1", "2", "3"],
        summary=(
            "IMPORTANT: This scale is NON-MONOTONIC with respect to leader power — a higher code does "
            "NOT mean more actual power. Codes 0-2 rise with the grandeur of the office, but code 3 "
            "marks largely ceremonial figureheads with little real power. Code the trappings and "
            "self-presentation of the office, NOT the leader's actual power. (Purely ceremonial heads "
            "who are not paramount leaders are excluded from the sample.)\n\n"
            "Coding:\n"
            "• 0 = Plain. The trappings of the office are plain and simple. Little distinguishes the personage "
            "of the ruler from others in the realm. Example: British prime minister (10 Downing Street).\n"
            "• 1 = Decorated. The trappings of the office are impressive but connected to the office rather "
            "than the officeholder, who is understood as mortal. Example: American president.\n"
            "• 2 = Deified. The trappings of the office are impressive and the holder is regarded as having "
            "divine or quasi-divine status, separate and apart from mere mortals. "
            "Examples: many kings in the premodern era.\n"
            "• 3 = Ceremonial. The trappings of office are so extensive, and so demanding, that they serve as "
            "a constraint on the exercise of power, separating the leader from the springs of policymaking. "
            "The executive's role is as much ceremonial as executive; approval of initiatives is formal. "
            "Example: Japanese emperor during most periods."
        )
    ),
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
        "Checks (Actors): which actors provide a check on the executive (select all that apply, output as JSON array). "
        "(0) None; (1) Local — clans, tribes, civil society, media; "
        "(2) Military — officers, military branches, warrior caste; "
        "(3) Clergy — established church, priests, religious caste; "
        "(4) Aristocracy — nobility, hereditary elite, upper caste; "
        "(5) Bourgeoisie — merchants, commercial classes, financiers; "
        "(6) Bureaucracy — civil servants, Confucian scholars; "
        "(7) Judiciary — courts, tribunals, legal institutions; "
        "(8) Assembly — popular assembly, legislature, parliament; "
        "(9) Advisory council — royal council, privy council, council of elders."
    ),

    "collegiality": (
        "Collegiality: whether decisionmaking power is shared among multiple actors at the apex. "
        "Code based on de facto practice, not de jure rules. "
        "(0) Non-collegial — single actor dominates (most presidencies, monarchies, dictatorships); "
        "(1) Collegial — power shared among co-equals (cabinets, juntas, regencies, Roman consuls, Swiss presidency)."
    ),

    "petition": (
        "Petition: whether citizens or subjects could regularly lodge complaints or requests with high officials. "
        "(0) No — petition use is extremely rare, probably ineffective, or unrecorded; "
        "(1) Yes — petitions are a fairly regular feature of political life."
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
        "(9) Direct popular election; (10) Other (e.g., clerical bodies); (99) Unknown."
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
        "(14) Still in office; (99) Unknown."
    ),

    "symbolism": (
        "Symbolism: trappings of executive office (non-monotonic with leader power; excludes purely ceremonial leaders). "
        "(0) Plain — plain and simple, ruler little distinguished from others in the realm; "
        "(1) Decorated — impressive but office-connected, officeholder understood as mortal; "
        "(2) Deified — holder regarded as divine or quasi-divine; "
        "(3) Ceremonial — trappings so extensive they constrain power, separating leader from policymaking."
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_indicators() -> List[str]:
    """Return the default indicator list."""
    return ["sovereign", "federalism", "checks", "collegiality", "petition", "assembly",
            "entry", "exit", "symbolism"]


def get_categorical_indicators() -> List[str]:
    """Return all categorical (non-continuous) indicator names."""
    return list(INDICATOR_CONFIGS.keys())


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
        include_robustness: bool = False  # deprecated, kept for backward compatibility
    ):
        if indicators is None:
            indicators = get_all_indicators()
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
            "Focus on THIS specific leader's tenure.\n"
            "When evidence is uncertain, apply the indicator-appropriate default:\n"
            "• Institutional-presence indicators (federalism, checks, collegiality, petition, assembly): default to 0 / None / No — if such an institution existed, the historical record would usually mention it, so silence indicates absence.\n"
            "• Sovereignty: default to 1 / Sovereign — overlordship or loss of domestic control would normally be recorded, so silence indicates the polity governed its own domestic affairs. (Be more cautious for premodern and non-Western polities, where semi-sovereign status may go unrecorded.)\n"
            "• Assembly is ordinal (0<1<2<3): when choosing among present-but-ambiguous levels, prefer the lower level.\n"
            "• Nominal indicators (entry, exit) and the non-monotonic symbolism scale: do NOT default to a lower label. If the evidence genuinely does not support any category, output \"N/A\" (entry/exit only) or your best estimate, and lower the confidence_score rather than forcing a code.\n\n"
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
                if config.multi_select:
                    labels_str = str(config.labels)
                    prompt += f'- "{ind}": JSON array of selected values from {labels_str} (select all that apply; use ["0"] if none apply)\n'
                else:
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
        tenure = f"{start_year}-{end_year if end_year is not None else 'present'}"
        return (
            f"Classify the following leader on all indicators:\n\n"
            f"**Polity:** {polity}\n"
            f"**Leader:** {name}\n"
            f"**Tenure:** {tenure}\n\n"
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
        include_robustness: bool = False  # deprecated, kept for backward compatibility
    ):
        if indicators is None:
            indicators = get_all_indicators()
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
            "2. Evaluate conditions as they existed during THIS leader's specific tenure.\n"
            "3. When evidence is uncertain, apply the indicator-appropriate default:\n"
            "   • Institutional-presence indicators (federalism, checks, collegiality, petition, assembly): default to 0 / None / No — if such an institution existed, the historical record would usually mention it, so silence indicates absence.\n"
            "   • Sovereignty: default to 1 / Sovereign — silence indicates the polity governed its own domestic affairs. (Be more cautious for premodern and non-Western polities.)\n"
            "   • Assembly is ordinal (0<1<2<3): when choosing among present-but-ambiguous levels, prefer the lower level.\n"
            "   • Nominal indicators (entry, exit) and the non-monotonic symbolism scale: do NOT default to a lower label — lower the confidence_score instead.\n"
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
            "1. Recall relevant historical facts about this leader's tenure.\n"
            "2. Match those facts to the indicator definitions above.\n"
            "3. Assign the appropriate code.\n"
            "4. Record your confidence (1–100) based on quality of evidence.\n\n"
            "## Required JSON Output\n\n"
            "Output ONLY a JSON object with these fields (no markdown, no preamble):\n"
        )

        for ind in self.indicators:
            if ind in INDICATOR_CONFIGS:
                config = INDICATOR_CONFIGS[ind]
                if config.multi_select:
                    labels_str = str(config.labels)
                    prompt += f'- "{ind}": JSON array of selected values from {labels_str} (select all that apply)\n'
                else:
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
        tenure = f"{start_year}–{end_year if end_year is not None else 'present'}"
        return (
            f"Annotate the following leader:\n\n"
            f"Polity: {polity}\n"
            f"Leader: {name}\n"
            f"Tenure: {tenure}\n\n"
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
        include_robustness: bool = False  # deprecated, kept for backward compatibility
    ):
        if indicators is None:
            indicators = get_all_indicators()
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
            "Focus on this specific leader's tenure. When uncertain: for institutional-presence indicators "
            "(federalism, checks, collegiality, petition, assembly) default to 0; "
            "for sovereignty default to 1; "
            "for assembly prefer the lower ordinal level when ambiguous; "
            "for nominal (entry, exit) or non-monotonic (symbolism) indicators, lower the confidence_score — "
            "do not default to the lower label.\n\n"
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
        tenure = f"{start_year}-{end_year if end_year is not None else 'present'}"

        fields = []
        for ind in self.indicators:
            if ind in INDICATOR_CONFIGS:
                config = INDICATOR_CONFIGS[ind]
                if config.multi_select:
                    labels_str = str(config.labels)
                    fields.append(f'"{ind}" (array, select all from {labels_str})')
                else:
                    labels_str = "/".join(config.labels)
                    fields.append(f'"{ind}" ({labels_str})')
                if self.reasoning:
                    fields.append(f'"{ind}_reasoning"')
                fields.append(f'"{ind}_confidence" (1-100)')

        fields_str = ", ".join(fields)

        return (
            f"Classify: **{polity}** | **{name}** | **{tenure}**\n\n"
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
        start_year: Start year of tenure
        end_year: End year of tenure
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
