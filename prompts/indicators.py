"""
Political Indicators Prompt Templates (Leader-Level)
=====================================================

Current indicator schema (aligned with single_builder.py):
- Sovereign         (0/1)
- Federalism        (0/1)
- Checks            (0-9, multi-select; formerly checks_actors)
- Collegiality      (0/1)
- Petition          (0/1)
- Assembly          (0/1/2/3)
- Entry             (0-10, fine-grained)
- Exit              (0-15, fine-grained)
- Symbolism         (0/1/2/3, non-monotonic; formerly symbolic_power)
- Elections         (0/1/2, downstream — depends on Assembly = 2)

Note: Constitution is handled separately in constitution.py.
Note: Tenure is excluded — it is a continuous variable, not a categorical label.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


# =============================================================================
# SHARED OUTPUT FORMAT TEMPLATES
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
- The "{indicator}" field MUST be a valid label ({valid_labels}). NEVER use null, empty string, or any other value. If uncertain, give your best estimate and lower the confidence_score.
"""

OUTPUT_FORMAT_TEMPLATE_NO_REASONING = """
## Output Requirements

Provide a JSON object with exactly these fields:
- "{indicator}": Must be exactly {valid_labels} (string)
- "confidence_score": Integer from 1 to 100 based on evidence quality

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with {{ and end with }}
- The "{indicator}" field MUST be a valid label ({valid_labels}). NEVER use null, empty string, or any other value.
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
{response_example}
"""


# =============================================================================
# INDICATOR CONFIGURATIONS
# =============================================================================

@dataclass
class IndicatorConfig:
    """Configuration for a political indicator."""
    name: str
    display_name: str
    specialization: str
    labels: List[str]
    definition: str
    task_description: str
    task_instruction: str
    coding_rule_reminder: str
    compact_definition: str = ""
    depends_on: Optional[Dict[str, str]] = None
    multi_select: bool = False


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

A polity is considered **sovereign** if it has supreme authority over its internal affairs without subordination to a foreign power. Sovereignty is concerned with executive constraints that arise from *within* the polity; foreign influences are usually corrupting because an executive beholden to a foreign power is (ceteris paribus) less sensitive to domestic actors.

**Sovereign (1):**
- The polity conducts independent foreign policy
- No tribute, allegiance, or political submission to a foreign power
- Internal governance is determined domestically
- The executive is NOT beholden to another polity
- Examples: city-states, nation-states, empires, republics, monarchies, tributary states (so long as they held primary responsibility for domestic affairs), states within leagues/unions that retained domestic governance

**Not Sovereign (0):**
- The polity is a colony, protectorate, vassal state, or tributary
- A foreign power controls or heavily influences governance
- Executive power is beholden to another polity (empire, hegemon, metropole)
- Examples: colonial territories, occupied states, protectorates, distant overseas territories not fully incorporated into the metropole

## Key Principle

To the extent that executive power in a polity is beholden to another polity, we assume it is less beholden to domestic sources, meaning there is less domestic constraint on the leader.

## Analysis Process

1. Identify the polity's political status during this leader's reign
2. Determine if the polity had independent control over domestic affairs
3. Check for any tribute, vassalage, or colonial relationships during this reign
4. Assess whether executive power was controlled externally
""",

        task_instruction="""Determine whether this polity was sovereign (1) or semi-sovereign (0) during this leader's reign.

- Sovereign (1): Independent domestic governance, no subordination to foreign power; includes city-states, nation-states, empires, tributary states with primary domestic responsibility
- Not Sovereign (0): Colony, protectorate, vassal, distant overseas territory not fully incorporated into the metropole""",

        coding_rule_reminder="⚠️ **IMPORTANT:** Focus on the status during THIS LEADER'S REIGN, not the entire polity history. Default to 1 (Sovereign) when evidence is ambiguous — overlordship or loss of domestic control would normally be recorded, so silence indicates the polity governed its own domestic affairs. Be more cautious for premodern and non-Western polities where semi-sovereign status may go unrecorded.",

        compact_definition=(
            "Whether the polity conducts domestic affairs without foreign subordination: "
            "(0) Semi-sovereign — colony, protectorate, vassal, or distant overseas territory not fully incorporated; "
            "(1) Sovereign — independent domestic governance; includes city-states, nation-states, empires, tributary states "
            "that held primary responsibility for domestic affairs, states in leagues/unions that retained self-governance."
        )
    ),

    # =========================================================================
    # FEDERALISM
    # =========================================================================
    "federalism": IndicatorConfig(
        name="federalism",
        display_name="federalism status",
        specialization="federal systems and territorial politics",
        labels=["0", "1"],
        task_description="whether a given polity had a federal or decentralized territorial structure during a specific leader's reign",

        definition="""
## Definition of Federalism

Federalism refers to a **division of sovereignty between central and local units**, reserving some important powers to local units and promising a relatively decentralized mode of governance. Where federalism exists, we assume that executive power is to some extent constrained by sub-national units.

**Federal (1):**
- Division of sovereignty between central and local units, with local units retaining constitutionally or compactly protected powers
- Local governance units cannot simply be abolished or overridden by the center at will
- Commonly, localities are represented in a legislative chamber at the polity level
- Includes confederations, leagues, composite monarchies, and federal states
- Historical examples: Achaean League, Aetolian League, Lycian League, Boeotian League, Old Swiss Confederacy, Dutch Republic, Holy Roman Empire, Iroquois Confederacy, Hanseatic League, Polish-Lithuanian Commonwealth, Tokugawa Japan
- Contemporary examples: Canada, Germany, India, United States, European Union

**Non-Federal (0):**
- Unitary state: local units exist but derive authority from the center and can be overridden or abolished
- No meaningful autonomous powers reserved to sub-national units by constitution or compact
- Examples: most centralized monarchies and republics, unitary states

## Analysis Process

1. Identify the territorial structure of the polity during this leader's reign
2. Determine if sub-national units have constitutionally or compactly protected powers
3. Assess whether localities are represented in central governance
4. Code based on de facto functioning, not formal constitutional texts
""",

        task_instruction="""Determine whether this polity was federal (1) or non-federal (0) during this leader's reign.

- Federal (1): Division of sovereignty between central and local units; local units have protected powers; includes confederations, leagues, composite monarchies
- Non-Federal (0): Unitary state; local units derive authority from the center with no protected autonomy""",

        coding_rule_reminder="⚠️ **IMPORTANT:** Focus on the territorial structure during THIS LEADER'S REIGN. Code de facto functioning, not formal arrangements.",

        compact_definition=(
            "Whether sovereignty is divided between central and local units: "
            "(0) Non-federal — unitary state, local units derive authority from center with no protected autonomy; "
            "(1) Federal — local units have constitutionally or compactly protected powers; includes confederations, "
            "leagues, composite monarchies. Examples: Achaean League, Dutch Republic, Holy Roman Empire, "
            "Iroquois Confederacy, Hanseatic League, Polish-Lithuanian Commonwealth, US, EU."
        )
    ),

    # =========================================================================
    # CHECKS (Multi-select, 10 categories — formerly checks_actors)
    # =========================================================================
    "checks": IndicatorConfig(
        name="checks",
        display_name="checks",
        specialization="comparative politics and executive constraints",
        labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        multi_select=True,
        task_description="which actors, if any, provide effective checks on executive power during a specific leader's reign",

        definition="""
## Definition of Checks (Actors)

Effective checks exist when independent groups or bodies have the capacity to resist actions taken by the executive. We ask about the identity of these groups or bodies, inferring that the number of categories transfers into more effective checks.

**Q: Which of the following actors provide a check on the actions of the executive? (You may choose more than one.)**

**Categories:**
- **0 = None.** There are no actors with this capacity or proclivity.
- **1 = Local.** Examples: clans, tribes, ethnic groups, local governance units, civil society groups, newspapers and other media.
- **2 = Military.** Examples: officers, specific branches of the military, a warrior caste (e.g., Samurai).
- **3 = Clergy.** Power partly derived from their role as arbiters of a widely espoused religion or set of beliefs. Examples: established church, priests, or caste (of any religion or denomination).
- **4 = Aristocracy.** Examples: landed class, upper caste, nobility, hereditary elite, titled class, patriciate.
- **5 = Bourgeoisie.** Examples: middle class, urbanites, artisans, traders, merchants, commercial classes, capitalist class, business class, entrepreneurs, financiers, creditors.
- **6 = Bureaucracy.** Examples: civil servants, Confucian scholars who serve as top-level advisors and bureaucrats.
- **7 = Judiciary.** Examples: courts of law, tribunals, judicial bodies, adjudicative bodies, legal institutions.
- **8 = Assembly.** Examples: popular assembly, legislature, parliament.
- **9 = Advisory council.** Examples: royal council, council of state, regency council, privy council, council of elders.

## Analysis Process

1. Identify which groups or bodies existed and had independent standing during this leader's reign
2. For each category, assess whether that actor had the capacity AND proclivity to resist executive actions
3. Select all categories that apply — more categories indicate more effective checking of executive power
4. If no actors had this capacity, select only 0 (None)
""",

        task_instruction="""Determine which actors provided effective checks on the executive during this leader's reign. Select all that apply.

Categories (0–9):
- 0: None — no actors with capacity to resist the executive
- 1: Local — clans, tribes, civil society, media
- 2: Military — officers, military branches, warrior caste
- 3: Clergy — established church, priests, religious caste
- 4: Aristocracy — nobility, hereditary elite, upper caste
- 5: Bourgeoisie — merchants, commercial classes, financiers
- 6: Bureaucracy — civil servants, Confucian scholars
- 7: Judiciary — courts, tribunals, legal institutions
- 8: Assembly — popular assembly, legislature, parliament
- 9: Advisory council — royal council, privy council, council of elders

**Output as a JSON array of selected values, e.g. ["1", "4", "7"]. If none apply, output ["0"].**""",

        coding_rule_reminder="⚠️ **IMPORTANT:** Select all actors that actually had the capacity and proclivity to resist the executive during THIS LEADER'S REIGN. Output as a JSON array.",

        compact_definition=(
            "Which actors provide a check on the executive (select all that apply, output as JSON array): "
            "(0) None; (1) Local — clans, tribes, civil society, media; "
            "(2) Military — officers, branches, warrior caste; (3) Clergy — church, priests, religious caste; "
            "(4) Aristocracy — nobility, hereditary elite; (5) Bourgeoisie — merchants, commercial classes; "
            "(6) Bureaucracy — civil servants, Confucian scholars; (7) Judiciary — courts, tribunals; "
            "(8) Assembly — popular assembly, legislature; (9) Advisory council — royal council, privy council."
        )
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

Collegiality refers to whether **power at the apex of a polity is exercised in a collegial manner** — decisionmaking power is shared among a number of actors. There may be a titular head (e.g., director or chair) but the other members of the group are regarded as co-equals, partners, collaborators. Wherever de facto practices differ from de jure rules, it is the former that governs coding decisions.

**Collegial (1):**
- Decision-making is genuinely shared by members of a formally constituted body
- Other members are regarded as co-equals with lateral (not vertical) power relationships
- Examples: cabinets where ministers hold independent authority, military juntas, Roman consuls (each with mutual veto), regent councils, Switzerland's Federal Council (all-party cabinet)

**Non-Collegial (0):**
- A single actor dominates decision-making
- Formally collegial bodies that are actually controlled by one person → Code as **0**
- Examples: most presidencies, monarchies, dictatorships; Stalin dominating the Politburo; Hitler's cabinet (ministers as executors); a sultan with a nominal advisory council

## Critical Distinction: De Facto vs De Jure

If a body is formally collegial but actually dominated by a single actor → Code as **0**

## Analysis Process

1. Identify the formal executive structure during this leader's reign
2. Determine if there was a formally constituted collegial body
3. **Critically assess**: Was decision-making actually shared, or did one person dominate?
4. Focus on de facto (actual) power, not de jure (formal) arrangements
5. **Default to 0 when evidence of genuine power-sharing is absent**
""",

        task_instruction="""Determine whether decision-making in the executive was collegial (1) or non-collegial (0) during this leader's reign.

- Collegial (1): Decisions genuinely shared by members of a formally constituted body — cabinets with independent ministers, military juntas, Roman consuls, regent councils, Swiss Federal Council
- Non-Collegial (0): Single actor dominates, OR formally collegial body controlled by one person in practice

**CRITICAL:** Code based on de facto (actual) power, not de jure (formal) arrangements.""",

        coding_rule_reminder="⚠️ **IMPORTANT:** Focus on ACTUAL decision-making practice during THIS LEADER'S REIGN. Default to 0 when evidence of genuine power-sharing is absent.",

        compact_definition=(
            "Whether decisionmaking power is shared among co-equals at the apex (de facto, not de jure): "
            "(0) Non-collegial — single actor dominates, or formally collegial body controlled by one person in practice; "
            "examples: most presidencies, monarchies, dictatorships. "
            "(1) Collegial — decisions genuinely shared among co-equals; "
            "examples: cabinets with independent ministers, military juntas, Roman consuls, regencies, Swiss presidency."
        )
    ),

    # =========================================================================
    # PETITION
    # =========================================================================
    "petition": IndicatorConfig(
        name="petition",
        display_name="petition",
        specialization="governance institutions and political access",
        labels=["0", "1"],
        task_description="whether petitioning was a regular and institutionalized feature of political life during a specific leader's reign",

        definition="""
## Definition of Petition

A petition is a formal process by which a citizen or subject may lodge a complaint or request for redress with a high official, e.g., a head of state, legislature, court, or ombudsman. The petition may take the form of a face-to-face meeting (the "bell of justice" tradition in Asia), a letter, or an electronic communication. It may have individual or multiple signatories. Whatever the particulars, the process is regularized and to some extent institutionalized — it is part of the governance structure, and as such may influence decisionmaking at the top.

## Petition Categories

- **0 = No.** Use of petition is extremely rare and probably ineffective, or there is no record of its existence.
- **1 = Yes.** Petitions are a fairly regular feature of political life.

## Analysis Process

1. Identify any institutionalized petition mechanisms during this leader's reign
2. Assess whether petitioning was regularized and effective, or rare and ineffective
3. Focus on de facto practice, not merely formal existence of petition mechanisms
""",

        task_instruction="""Determine whether petitioning was a regular and institutionalized feature of political life (0 or 1) during this leader's reign.

- 0 (No): Use of petition is extremely rare and probably ineffective, or there is no record of its existence.
- 1 (Yes): Petitions are a fairly regular feature of political life — citizens or subjects regularly lodge complaints or requests with high officials.""",

        coding_rule_reminder="⚠️ **IMPORTANT:** Focus on whether petitioning was actually practiced and effective during THIS LEADER'S REIGN. Code based on de facto use, not formal existence of mechanisms.",

        compact_definition=(
            "Whether petitioning was a regularized feature of governance: "
            "(0) No — petition use is extremely rare, probably ineffective, or unrecorded; "
            "(1) Yes — petitions are a fairly regular feature of political life, citizens or subjects regularly lodge complaints or requests."
        )
    ),

    # =========================================================================
    # ASSEMBLY
    # =========================================================================
    "assembly": IndicatorConfig(
        name="assembly",
        display_name="assembly status",
        specialization="legislative institutions",
        labels=["0", "1", "2", "3"],
        task_description="the type of assembly or council that existed during a specific leader's reign",

        definition="""
## Definition of Assembly

An assembly is a body designed to govern (directly), to select leaders, or to assist in governing. Where a council or assembly exists, we assume that executive power is to some extent constrained or perhaps entirely displaced.

## Assembly Types

**Type 0 — None:**
- No deliberative or advisory body of any kind
- Purely autocratic rule with no institutionalized council structure

**Type 1 — Council:**
- A small advisory council appointed by the ruler
- May not enjoy much autonomy but is nonetheless **institutionalized**: meets regularly, has a designated name, has a fairly stable membership
- Examples: noble councils, aristocratic councils, royal councils, privy councils, dynastic councils, Ottoman divan

**Type 2 — Legislature:**
- A large representative body that plays some role in **policymaking or leadership selection** — de jure or de facto
- Membership is not limited to most citizens; it is a representative body with a defined constituency
- Examples: estates assemblies in premodern Europe, the Hwabaek Council in Korea during the Silla Dynasty, legislatures in modern governments

**Type 3 — Popular Assembly:**
- An assembly that **includes most citizens of the polity**, or a representative sample chosen by lot
- More inclusive than a legislature; the demos or a cross-section of it participates directly
- Examples: Ecclesia in ancient Athens, Landsgemeinden in modern Swiss cantons

## Coding Rules

- Code based on **de facto** (actual) practice, not de jure (formal) arrangements
- A body that nominally exists but never meets or has no stable membership → Type 0
- Focus on the **highest type** of assembly that existed and actually functioned during THIS LEADER'S REIGN
- **Default to Type 0 when evidence is absent.** Regional or civilizational generalizations alone are NOT sufficient.

## Analysis Process

1. Identify any deliberative or advisory bodies during this leader's reign
2. Determine if the body is institutionalized (regular meetings, designated name, stable membership)
3. Assess the body's scope: small appointed council (Type 1), large representative body (Type 2), or most-citizens assembly (Type 3)
4. Code based on actual (de facto) functioning, not formal arrangements
""",

        task_instruction="""Determine the assembly type (0, 1, 2, or 3) for this leader's reign.

Types:
- 0: None — no assembly or council; purely autocratic rule
- 1: Council — small advisory council appointed by ruler, institutionalized (regular meetings, designated name, stable membership); examples: privy councils, aristocratic councils, Ottoman divan
- 2: Legislature — large representative body with policymaking or leadership selection role (de jure or de facto); examples: estates assemblies, premodern European legislatures, modern parliaments
- 3: Popular Assembly — includes most citizens or a representative sample chosen by lot; examples: Athenian Ecclesia, Swiss Landsgemeinden

**Code based on de facto functioning, not formal structures.**""",

        coding_rule_reminder=(
            "⚠️ **IMPORTANT:** Focus on the HIGHEST type of assembly that actually functioned during THIS LEADER'S REIGN. "
            "Code de facto, not de jure. Default to Type 0 when evidence is absent — do NOT infer a council from regional "
            "or civilizational patterns alone."
        ),

        compact_definition=(
            "Type of assembly/council (de facto): "
            "(0) None — purely autocratic rule; "
            "(1) Council — small advisory council appointed by ruler, institutionalized (regular meetings, designated name, stable membership); "
            "examples: privy councils, aristocratic councils, Ottoman divan; "
            "(2) Legislature — large representative body with policymaking role (de jure or de facto); "
            "examples: estates assemblies, premodern European legislatures, modern parliaments; "
            "(3) Popular Assembly — includes most citizens or chosen by lot; examples: Athenian Ecclesia, Swiss Landsgemeinden. "
            "Default to 0 when evidence is absent."
        )
    ),

    # =========================================================================
    # ENTRY (Fine-grained, 11 categories)
    # =========================================================================
    "entry": IndicatorConfig(
        name="entry",
        display_name="executive entry mode",
        specialization="executive selection and leadership transitions",
        labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "99"],
        task_description="the precise mode by which the executive came to power during a specific leader's reign",

        definition="""
## Definition of Executive Entry

The manner in which executives enter office is widely regarded as a key indicator of their power while in office. Leader selection indicates the sorts of constraints executives are likely to face.

## Entry Categories

- **0 = Force.** Through the threat or application of force, such as a coup or rebellion.
- **1 = Foreign power.** Appointed by a foreign power.
- **2 = Ruling party.** Appointed by the ruling party in a one-party system.
- **3 = Royal council.** Appointed by a royal council (members of the royal family or a conclave of aristocrats).
- **4 = Hereditary.** Through hereditary succession (designated family heir inherits office).
- **5 = Military.** Appointed by the military.
- **6 = Legislature.** Appointed by the legislature or legislative body.
- **7 = Head of state.** Appointed by the head of state.
- **8 = Head of government.** Appointed by the head of government.
- **9 = Popular election.** Directly through a popular election (regardless of the extension of the suffrage).
- **10 = Other.** Other means, including clerical bodies such as the College of Cardinals.
- **99 = Unknown.** The circumstances of executive entry are unknown.

## Analysis Process

1. Identify how THIS LEADER specifically came to power in this polity
2. Distinguish the immediate mechanism (who/what selected them) from background conditions
3. Match to the most specific applicable category above
4. If multiple mechanisms apply, choose the primary/proximate one
""",

        task_instruction="""Determine the entry category (0–10 or 99) for how THIS LEADER came to power.

- 0: Through force — coup, rebellion, conquest
- 1: Appointed by foreign power
- 2: Appointed by ruling party (one-party system)
- 3: Appointed by royal council (royal family members or aristocratic conclave)
- 4: Through hereditary succession
- 5: Appointed by the military
- 6: Appointed by the legislature
- 7: Appointed by the head of state
- 8: Appointed by the head of government
- 9: Through direct popular election (extent of suffrage irrelevant)
- 10: Other (e.g., clerical bodies such as College of Cardinals)
- 99: Unknown (circumstances of entry are unknown)""",

        coding_rule_reminder="⚠️ **IMPORTANT:** Focus on how THIS SPECIFIC LEADER came to power. Choose the most specific applicable category.",

        compact_definition=(
            "How the executive came to power (choose one): "
            "(0) Force — coup, rebellion, conquest; (1) Foreign power appointment; "
            "(2) Ruling party appointment (one-party system); (3) Royal council appointment; "
            "(4) Hereditary succession; (5) Military appointment; "
            "(6) Legislature appointment; (7) Head of state appointment; "
            "(8) Head of government appointment; (9) Direct popular election; "
            "(10) Other (e.g., clerical bodies); (99) Unknown."
        )
    ),

    # =========================================================================
    # EXIT (Fine-grained, 16 categories)
    # =========================================================================
    "exit": IndicatorConfig(
        name="exit",
        display_name="executive exit mode",
        specialization="executive transitions and leadership succession",
        labels=["0", "1", "2", "3", "4", "5", "6", "7", "8",
                "9", "10", "11", "12", "13", "14", "99"],
        task_description="the precise circumstances of the executive's departure from office",

        definition="""
## Definition of Executive Exit

The circumstances of an executive's departure from office says a lot about a leader's prerogative while in office. We distinguish regular from irregular exits, and deaths from voluntary departures.

## Exit Categories

- **0 = Voluntary retirement/abdication.** Abdicated or retired voluntarily, but NOT due to ill health.
- **1 = Other regular exit.** Other regular institutional exit such as term limits or defeat in election.
- **2 = Regular transition to another office.** Transition to another office type/typology by regular procedures.
- **3 = Died on campaign (civil war, disease/accident).** Died of disease or accident on campaign in civil war.
- **4 = Died on campaign (foreign war, disease/accident).** Died of disease or accident on campaign in a foreign war.
- **5 = Died of natural causes.** Died of natural causes (not on campaign, not violent death).
- **6 = Retired due to ill health.** Retired voluntarily but due to ill health.
- **7 = Suicide.** Died by suicide.
- **8 = Deposed by domestic actors.** Removed from office by domestic actors (not assassination).
- **9 = Assassinated or forced suicide.** Killed or forced to commit suicide by rivals/opponents.
- **10 = Died in battle (civil war).** Killed in battle during a civil war.
- **11 = Died in battle (foreign war).** Killed in battle during a foreign war.
- **12 = Irregular transition to another office.** Transition to another office by irregular procedures.
- **13 = Deposed by foreign state.** Removed from office by a foreign state (occupation, intervention).
- **14 = Still in office.** The leader is still in office at the end of the observation period.
- **99 = Unknown.** Circumstances of exit are unknown.

## Analysis Process

1. Determine how this leader left power (or note if still in office)
2. Distinguish between voluntary departures, institutional exits, natural deaths, and violent/forced removals
3. Match to the most specific applicable category
""",

        task_instruction="""Determine the exit category (0–14 or 99) for how THIS LEADER left power.

- 0: Voluntarily retired/abdicated (NOT due to ill health)
- 1: Other regular exit (term limits, electoral defeat)
- 2: Regular transition to another office
- 3: Died on campaign in civil war (disease/accident)
- 4: Died on campaign in foreign war (disease/accident)
- 5: Died of natural causes
- 6: Retired due to ill health
- 7: Suicide
- 8: Deposed by domestic actors
- 9: Assassinated or forced suicide
- 10: Died in battle in civil war
- 11: Died in battle in foreign war
- 12: Irregular transition to another office
- 13: Deposed by foreign state
- 14: Still in office
- 99: Unknown""",

        coding_rule_reminder="⚠️ **IMPORTANT:** Focus on how THIS SPECIFIC LEADER left power. If still ruling at end of observation period, use 14.",

        compact_definition=(
            "Circumstances of executive departure (choose one): "
            "(0) Voluntary retirement/abdication (not ill health); (1) Other regular exit (term limits, electoral defeat); "
            "(2) Regular transition to another office; (3) Died on campaign, civil war (disease/accident); "
            "(4) Died on campaign, foreign war (disease/accident); (5) Died of natural causes; "
            "(6) Retired due to ill health; (7) Suicide; (8) Deposed by domestic actors; "
            "(9) Assassinated or forced suicide; (10) Died in battle, civil war; (11) Died in battle, foreign war; "
            "(12) Irregular transition to another office; (13) Deposed by foreign state; "
            "(14) Still in office; (99) Unknown."
        )
    ),

    # =========================================================================
    # SYMBOLISM (formerly symbolic_power)
    # =========================================================================
    "symbolism": IndicatorConfig(
        name="symbolism",
        display_name="symbolism",
        specialization="executive power and political symbolism",
        labels=["0", "1", "2", "3"],
        task_description="the degree of symbolic power reflected in the trappings of the executive office during a specific leader's reign",

        definition="""
## Definition of Symbolic Power

The power of the executive is to some extent reflected in the trappings of the office. Monarchs typically inhabit grandly appointed palaces and courts, with courtesans, servants, special rituals, forms of address, and physical objects symbolizing special power. Sometimes leaders are regarded as godlike. Note: **this scale is non-monotonic with respect to leader power** — higher codes do not always mean more actual power. Purely ceremonial leaders are excluded.

## Symbolic Power Categories

- **0 = Plain.** The trappings of the office are plain and simple. Little distinguishes the personage of the ruler from others in the realm. Example: British prime minister (10 Downing Street).
- **1 = Decorated.** The trappings of the office are impressive but connected to the office rather than the officeholder, who is understood as mortal. Example: American president.
- **2 = Deified.** The trappings of the office are impressive and the holder is regarded as having divine or quasi-divine status, separate and apart from mere mortals. Examples: many kings in the premodern era.
- **3 = Ceremonial.** The trappings of office are so extensive, and so demanding, that they serve as a constraint on the exercise of power, separating the leader from the springs of policymaking. The executive's role is as much ceremonial as executive; approval of initiatives is formal. Example: Japanese emperor during most periods.

## Analysis Process

1. Identify the ceremonial and symbolic elements associated with the executive office during this leader's reign
2. Assess whether the officeholder is regarded as mortal (0/1) or divine (2) or ceremonially constrained (3)
3. Determine if the trappings of office enhance or constrain actual executive power
""",

        task_instruction="""Determine the symbolic power category (0, 1, 2, or 3) for this leader's office.

- 0: Plain — plain and simple trappings; ruler little distinguished from others in the realm
- 1: Decorated — impressive trappings but connected to the office, officeholder understood as mortal
- 2: Deified — holder regarded as divine or quasi-divine
- 3: Ceremonial — trappings so extensive they constrain power, separating leader from policymaking

**Note:** This scale is non-monotonic — code 3 (Ceremonial) does NOT mean most powerful.""",

        coding_rule_reminder="⚠️ **IMPORTANT:** This scale is non-monotonic with respect to leader power. Focus on the trappings during THIS LEADER'S REIGN.",

        compact_definition=(
            "Trappings of executive office (non-monotonic with leader power; excludes purely ceremonial leaders): "
            "(0) Plain — plain and simple, ruler little distinguished from others; "
            "(1) Decorated — impressive but office-connected, officeholder understood as mortal; example: US president; "
            "(2) Deified — holder regarded as divine or quasi-divine; examples: many premodern kings; "
            "(3) Ceremonial — trappings so extensive they constrain power, separating leader from policymaking; example: Japanese emperor."
        )
    ),

    # =========================================================================
    # ELECTIONS (DEPENDS ON ASSEMBLY = 2)
    # =========================================================================
    "elections": IndicatorConfig(
        name="elections",
        display_name="assembly elections status",
        specialization="electoral systems and representative institutions",
        labels=["0", "1", "2"],
        task_description="whether members of the large legislature (assembly = 2) were elected and whether elections were contested by organized factions",
        depends_on={"assembly": "2"},

        definition="""
## Definition of Elections

This indicator codes whether members of an existing **Legislature (assembly = 2)** are **elected** to their positions. It is only applicable when a large representative legislature exists (NOT for popular assemblies, assembly = 3).

**An election** refers to a selection procedure in which:
- Members are chosen by an electorate through defined rules (e.g., majority, proportionality)
- These rules translate votes into seats
- The electorate must be considerably larger than the body itself (though far short of universal suffrage is acceptable)

## Election Categories

**No Elections (0):**
- Assembly members are NOT elected
- Members are appointed, hereditary, or selected by non-electoral means
- Also: pass-through for rows where assembly ≠ 2

**Elections Exist (1):**
- Most members of the assembly are elected by defined rules
- Electorate is larger than the body itself
- Elections exist but are NOT organized by factions or parties

**Competitive Elections (2):**
- Elections exist AND are contested by organized factions or parties
- Distinct blocs, factions, or parties compete for seats
- Examples: Roman Optimates vs Populares, English Whigs vs Tories, modern multi-party elections

## Important Notes

- This indicator assumes assembly = 2 (Legislature). Rows where assembly ≠ 2 pass through with elections = 0.
- The extent of suffrage is NOT relevant for coding
- The key distinction for code 2 is organized factions/parties, not just informal competition

## Analysis Process

1. Confirm that a Legislature (assembly = 2) exists during this leader's reign
2. Determine how members of the assembly obtain their positions
3. If elected: are elections contested by organized factions or parties?
""",

        task_instruction="""Determine the elections category (0, 1, or 2) for this leader's reign.

**This indicator only applies when a Legislature (assembly = 2) exists.**

- 0: No elections — members appointed, hereditary, or non-electoral
- 1: Elections exist — most members elected by defined rules, NOT organized by factions/parties
- 2: Competitive elections — contested by organized factions or parties""",

        coding_rule_reminder="⚠️ **IMPORTANT:** This indicator only applies when assembly = 2 (Legislature). Focus on the selection method for assembly members during THIS LEADER'S REIGN.",

        compact_definition=(
            "Whether Legislature (assembly = 2) members are elected (not applicable for assembly ≠ 2): "
            "(0) No elections — members appointed, hereditary, or non-electoral; "
            "(1) Elections exist — most members elected by electorate through defined rules, no organized factions; "
            "(2) Competitive elections — contested by organized factions or parties."
        )
    ),
}


# =============================================================================
# COMPACT DEFINITIONS DICTIONARY (for combined prompts)
# =============================================================================

COMPACT_DEFINITIONS: Dict[str, str] = {
    name: config.compact_definition
    for name, config in INDICATOR_CONFIGS.items()
}


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def build_system_prompt(indicator: str, reasoning: bool = True) -> str:
    """Build system prompt for an indicator (full version)."""
    if indicator not in INDICATOR_CONFIGS:
        raise ValueError(f"Unknown indicator: {indicator}. Must be one of {list(INDICATOR_CONFIGS.keys())}")

    config = INDICATOR_CONFIGS[indicator]

    header = SYSTEM_PROMPT_HEADER.format(
        specialization=config.specialization,
        task_description=config.task_description
    )

    if config.multi_select:
        output_format = f"""
## Output Requirements

Provide a JSON object with exactly these fields:
- "{config.name}": Must be a JSON array of selected values from {config.labels}, e.g. ["1", "4"]. Use ["0"] if none apply.
- {"'reasoning': Your step-by-step reasoning following the analysis process (string)" + chr(10) + "- " if reasoning else ""}"confidence_score": Integer from 1 to 100 based on evidence quality

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with {{ and end with }}
- The "{config.name}" field MUST be a JSON array of valid values from {config.labels}. NEVER use null, empty string, or a plain string. If none apply, use ["0"].
"""
    else:
        label_display = f'"{config.labels[0]}"' if len(config.labels) == 2 else f'one of {config.labels}'
        template = OUTPUT_FORMAT_TEMPLATE if reasoning else OUTPUT_FORMAT_TEMPLATE_NO_REASONING
        output_format = template.format(
            indicator=config.name,
            valid_labels=label_display
        )

    return header + config.definition + output_format


def build_user_prompt(
    indicator: str,
    polity: str,
    name: str,
    start_year: int,
    end_year: int,
    reasoning: bool = True
) -> str:
    """Build user prompt for an indicator with leader-level information (full version)."""
    if indicator not in INDICATOR_CONFIGS:
        raise ValueError(f"Unknown indicator: {indicator}. Must be one of {list(INDICATOR_CONFIGS.keys())}")

    config = INDICATOR_CONFIGS[indicator]

    reasoning_field = '"reasoning": "your analysis", ' if reasoning else ""

    if config.multi_select:
        # Show an actual JSON array in the example so the model returns an array, not a string
        example_array = f'["{config.labels[1]}", "{config.labels[2]}"]'
        response_example = f'{{"{config.name}": {example_array}, {reasoning_field}"confidence_score": 1-100}}'
    else:
        if len(config.labels) == 2:
            label_format = f"{config.labels[0]} or {config.labels[1]}"
        else:
            label_format = ", ".join(config.labels[:-1]) + f", or {config.labels[-1]}"
        response_example = f'{{"{config.name}": "{label_format}", {reasoning_field}"confidence_score": 1-100}}'

    return USER_PROMPT_TEMPLATE.format(
        indicator_display=config.display_name,
        polity=polity,
        name=name,
        start_year=start_year,
        end_year=end_year,
        task_instruction=config.task_instruction,
        coding_rule_reminder=config.coding_rule_reminder,
        response_example=response_example,
    )


# =============================================================================
# COMBINED PROMPT BUILDER (COMPACT VERSION)
# =============================================================================

COMBINED_SYSTEM_PROMPT = """You are a professional political scientist and historian specializing in comparative politics across different historical periods.

Your task is to classify multiple political indicators for a specific leader's reign based on the polity name, leader name, and reign period provided.

## General Guidelines

1. **Independence**: Evaluate each indicator independently based on its specific definition
2. **Evidence-based**: Base judgments on verifiable historical facts
3. **De facto over de jure**: When actual power differs from formal arrangements, code based on actual practice
4. **Leader-specific**: Focus on conditions during THIS SPECIFIC LEADER'S reign

## Output Format

Respond with ONLY a valid JSON object containing all requested indicators.
Do NOT include markdown code fences or any text outside the JSON.
"""


def build_combined_prompt(
    polity: str,
    name: str,
    start_year: int,
    end_year: int,
    indicators: List[str],
    include_elections: bool = False,
    assembly_value: Optional[str] = None
) -> Tuple[str, str]:
    """Build a combined prompt for multiple indicators (compact version)."""
    valid_indicators = []
    for ind in indicators:
        if ind not in INDICATOR_CONFIGS:
            raise ValueError(f"Unknown indicator: {ind}")
        if ind == "elections":
            if include_elections and assembly_value == "2":
                valid_indicators.append(ind)
        else:
            valid_indicators.append(ind)

    definitions_text = "## Indicator Definitions\n\n"
    for ind in valid_indicators:
        config = INDICATOR_CONFIGS[ind]
        definitions_text += f"**{ind}**: {config.compact_definition}\n\n"

    output_fields = []
    for ind in valid_indicators:
        config = INDICATOR_CONFIGS[ind]
        if config.multi_select:
            output_fields.append(f'"{ind}": ["select all from {config.labels}"]')
        else:
            labels_str = "/".join(config.labels)
            output_fields.append(f'"{ind}": "{labels_str}"')

    output_schema = "{\n  " + ",\n  ".join(output_fields)
    output_schema += ',\n  "reasoning": "brief reasoning for each indicator",\n  "confidence_score": 1-100\n}'

    user_prompt = f"""Please analyze the following leader's reign and classify each indicator:

**Polity:** {polity}
**Leader:** {name}
**Reign Period:** {start_year}-{end_year}

{definitions_text}

## Required Output

Provide a JSON object with classifications for all indicators:

{output_schema}

**Remember:**
- Evaluate each indicator independently
- Focus on THIS LEADER'S reign specifically
- Code based on de facto (actual) practice, not de jure (formal) arrangements

Your JSON response:"""

    return COMBINED_SYSTEM_PROMPT, user_prompt


# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_prompt(
    indicator: str,
    polity: str,
    name: str,
    start_year: int,
    end_year: int,
    reasoning: bool = True
) -> Tuple[str, str]:
    """Get system and user prompts for a specific indicator (full version, leader-level)."""
    system_prompt = build_system_prompt(indicator, reasoning=reasoning)
    user_prompt = build_user_prompt(indicator, polity, name, start_year, end_year, reasoning=reasoning)
    return system_prompt, user_prompt


def get_compact_definition(indicator: str) -> str:
    """Get the compact definition for an indicator (for combined prompts)."""
    if indicator not in COMPACT_DEFINITIONS:
        raise ValueError(f"Unknown indicator: {indicator}")
    return COMPACT_DEFINITIONS[indicator]


def get_all_indicators(include_dependent: bool = True) -> List[str]:
    """Return list of all indicator names."""
    if include_dependent:
        return list(INDICATOR_CONFIGS.keys())
    else:
        return [name for name, config in INDICATOR_CONFIGS.items()
                if config.depends_on is None]


def get_independent_indicators() -> List[str]:
    """Return list of indicators without dependencies (can be run in first pass)."""
    return get_all_indicators(include_dependent=False)


def get_dependent_indicators() -> Dict[str, Dict[str, str]]:
    """Return dictionary of dependent indicators and their dependencies."""
    return {
        name: config.depends_on
        for name, config in INDICATOR_CONFIGS.items()
        if config.depends_on is not None
    }


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


def check_dependency(indicator: str, previous_results: Dict[str, str]) -> bool:
    """Check if an indicator's dependencies are satisfied."""
    config = INDICATOR_CONFIGS.get(indicator)
    if config is None:
        raise ValueError(f"Unknown indicator: {indicator}")

    if config.depends_on is None:
        return True

    for dep_indicator, required_value in config.depends_on.items():
        actual_value = previous_results.get(dep_indicator)
        if actual_value != required_value:
            return False

    return True


# =============================================================================
# CHAIN OF VERIFICATION (CoVe) QUESTIONS
# =============================================================================

COVE_QUESTION_TEMPLATES: Dict[str, List[str]] = {
    "sovereign": [
        "Was {polity} a colony, protectorate, or vassal of another power during {name}'s reign ({start_year}-{end_year})?",
        "Did {polity} conduct independent domestic governance under {name}?",
        "Did {polity} pay tribute or acknowledge suzerainty to any external power during {name}'s rule?"
    ],
    "federalism": [
        "Did sub-national units in {polity} have constitutionally or compactly protected powers during {name}'s reign ({start_year}-{end_year})?",
        "Were there representative bodies for local or regional units at the central level in {polity} under {name}?",
        "Could the central government in {polity} freely abolish or override local units during {name}'s reign?"
    ],
    "checks": [
        "Which independent groups or bodies had the capacity to resist the executive in {polity} during {name}'s reign ({start_year}-{end_year})?",
        "Did local actors (clans, tribes, civil society, media) have the capacity or proclivity to resist {name}'s actions?",
        "Did military actors, clergy, or aristocracy constrain {name}'s executive power in {polity}?",
        "Did bureaucracy, judiciary, assembly, or advisory council have the capacity to resist {name}'s actions?"
    ],
    "collegiality": [
        "What was the formal executive structure in {polity} during {name}'s reign ({start_year}-{end_year})?",
        "Was there a cabinet, council, or collegial body that genuinely shared decision-making with {name}?",
        "Did {name} dominate decision-making, or were decisions genuinely shared among co-equals?",
        "Were there co-rulers, regents, or formally constituted bodies sharing executive power with {name}?"
    ],
    "assembly": [
        "What deliberative or advisory bodies existed in {polity} during {name}'s reign ({start_year}-{end_year})?",
        "Was any such body a small advisory council (institutionalized, regular meetings, designated name) appointed by the ruler?",
        "Was there a large representative legislature that played a role in policymaking or leadership selection under {name}?",
        "Was there a popular assembly that included most citizens of {polity} or a sample chosen by lot under {name}?"
    ],
    "petition": [
        "Was there a formal or institutionalized process by which citizens or subjects could lodge complaints or requests with {name} or other high officials in {polity} ({start_year}-{end_year})?",
        "Was petitioning a regular and effective feature of political life in {polity} during {name}'s reign?",
        "What forms did petitioning take in {polity} during {name}'s reign — formal hearings, written requests, or other mechanisms?"
    ],
    "entry": [
        "How did {name} come to power in {polity}?",
        "Was {name}'s assumption of leadership through force, hereditary succession, appointment, or election?",
        "Who or what specifically selected or installed {name} as leader of {polity}?"
    ],
    "exit": [
        "How did {name} leave power in {polity}?",
        "Did {name} die in office, retire voluntarily, face forced removal, or transition to another office?",
        "Was {name}'s departure the result of institutional processes, personal choice, health, or external force?"
    ],
    "symbolism": [
        "What ceremonial or symbolic elements were associated with the executive office in {polity} during {name}'s reign ({start_year}-{end_year})?",
        "Was {name} regarded as divine or quasi-divine, or as an ordinary mortal?",
        "Did the trappings of {name}'s office enhance or constrain actual executive power?"
    ],
    "elections": [
        "How were members of the large legislature selected in {polity} during {name}'s reign ({start_year}-{end_year})?",
        "Were assembly members elected by an electorate considerably larger than the body itself?",
        "If elections existed, were they contested by organized factions or parties in {polity} under {name}?"
    ],
}


def get_cove_questions(
    indicator: str,
    polity: str,
    name: str,
    start_year: int,
    end_year: int
) -> List[str]:
    """Get Chain of Verification questions for an indicator (leader-level)."""
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
    indicators: Optional[List[str]] = None,
    exclude_dependent: bool = False
) -> Dict[str, Tuple[str, str]]:
    """Get prompts for all (or specified) indicators for a single leader."""
    if indicators is None:
        indicators = get_all_indicators(include_dependent=not exclude_dependent)

    return {
        ind: get_prompt(ind, polity, name, start_year, end_year)
        for ind in indicators
        if ind in INDICATOR_CONFIGS
    }


def get_expected_output_schema(indicators: Optional[List[str]] = None) -> Dict[str, dict]:
    """Get the expected output schema for specified indicators."""
    if indicators is None:
        indicators = get_all_indicators()

    return {
        ind: {
            "fields": {
                ind: f"string, one of {get_indicator_labels(ind)}",
                "reasoning": "string",
                "confidence_score": "integer, 1-100"
            },
            "depends_on": INDICATOR_CONFIGS[ind].depends_on
        }
        for ind in indicators
    }
