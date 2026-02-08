"""
Single Prompt Builder

This module provides a prompt builder that combines multiple indicators
into a single prompt for efficiency. This approach may reduce API costs
but could potentially cause cross-indicator contamination.
"""

from typing import List, Optional, Dict

from prompts.base_builder import BasePromptBuilder, PromptOutput
from prompts.constitution import get_prompt as get_constitution_prompt
from prompts.indicators import INDICATOR_CONFIGS


# =============================================================================
# INDICATOR DEFINITION SUMMARIES
# =============================================================================
# Concise summaries of each indicator's definition for combined prompts

INDICATOR_SUMMARIES: Dict[str, str] = {
    "sovereign": (
        "A polity is sovereign (1) if it has supreme authority over internal and external affairs "
        "without subordination to a foreign power. Not sovereign (0) if it's a colony, protectorate, "
        "vassal, or tributary state where executive power is beholden to another polity."
    ),

    "powersharing": (
        "Powersharing (1) exists when two or more individuals share executive power at the apex of "
        "the polity with comparable authority (e.g., Roman consuls, regencies, military juntas, "
        "co-presidents). No powersharing (0) means one dominant leader controls executive power."
    ),

    "assembly": (
        "An assembly (1) is a large popular assembly or representative parliament that: (a) has a "
        "role in leadership selection, taxation, or policy; (b) has independence from the executive; "
        "(c) meets regularly. No assembly (0) if no such body exists or only advisory councils "
        "without institutional power."
    ),

    "appointment": (
        "How executives are selected: (0) through force, hereditary succession, foreign power, "
        "military, or one-party selection (least constrained); (1) by royal council, head of state, "
        "or head of government (moderately constrained); (2) through direct popular election or "
        "assembly selection (most constrained)."
    ),

    "tenure": (
        "Length of leader's reign indicating constraint level: (0) less than 5 years (high constraint), "
        "(1) 5-10 years (moderate constraint), (2) more than 10 years (low constraint)."
    ),

    "exit": (
        "How leaders leave power: (0) irregular exit - died in office or removed by force; "
        "(1) regular exit - voluntary retirement, term limits, electoral defeat, or peaceful "
        "institutional transition."
    )
}


class SinglePromptBuilder(BasePromptBuilder):
    """
    Combines all selected indicators into a single prompt.

    This builder creates one comprehensive prompt that asks the LLM
    to predict multiple indicators simultaneously. This is more
    efficient (fewer API calls) but may cause indicators to influence
    each other's predictions.

    Example:
        builder = SinglePromptBuilder(indicators=['sovereign', 'assembly', 'powersharing'])
        prompts = builder.build("Roman Republic", -509, -27)
        # Returns single PromptOutput covering all 3 indicators
    """

    def __init__(self, indicators: Optional[List[str]] = None, reasoning: bool = True):
        """
        Initialize the single prompt builder.

        Args:
            indicators: List of indicators to include. If None, uses all non-constitution indicators.
            reasoning: Whether to include reasoning in prompts for non-constitution indicators (default True).
        """
        # Default to all indicators except constitution (which has special handling)
        if indicators is None:
            indicators = ['sovereign', 'powersharing', 'assembly', 'appointment', 'tenure', 'exit']
        super().__init__(indicators, reasoning)

    def build(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> List[PromptOutput]:
        """
        Build a single combined prompt for all indicators.

        Args:
            polity: Name of the polity
            name: Name of the leader
            start_year: Start year of the leader's reign
            end_year: End year of the leader's reign (None if unknown/unavailable)

        Returns:
            List containing a single PromptOutput (or two if constitution is included)
        """
        prompts = []

        # Handle constitution separately (always its own prompt due to complexity)
        if 'constitution' in self.indicators:
            system_prompt, user_prompt = get_constitution_prompt(
                polity=polity,
                name=name,
                start_year=start_year,
                end_year=end_year
            )
            prompts.append(PromptOutput(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                indicators=['constitution'],
                metadata={'mode': 'single', 'type': 'constitution'}
            ))

        # Combine other indicators
        other_indicators = [ind for ind in self.indicators if ind != 'constitution']
        if other_indicators:
            system_prompt = self._build_combined_system_prompt(other_indicators)
            user_prompt = self._build_combined_user_prompt(other_indicators, polity, name, start_year, end_year)
            prompts.append(PromptOutput(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                indicators=other_indicators,
                metadata={'mode': 'single', 'type': 'combined'}
            ))

        return prompts

    def _build_combined_system_prompt(self, indicators: List[str]) -> str:
        """Build a combined system prompt for multiple indicators."""
        prompt = """You are a professional political scientist and historian specializing in comparative politics across different historical periods.

Your task is to analyze a given leader's reign and determine values for multiple political indicators.

## Indicators to Analyze

"""

        # Add indicator definitions using pre-written summaries
        for ind in indicators:
            if ind in INDICATOR_CONFIGS and ind in INDICATOR_SUMMARIES:
                config = INDICATOR_CONFIGS[ind]
                prompt += f"**{config.display_name.upper()}**\n"
                labels_str = " / ".join(config.labels)
                prompt += f"Labels: {labels_str}\n"
                # Use concise, well-written summary
                prompt += f"{INDICATOR_SUMMARIES[ind]}\n\n"

        prompt += """## Output Requirements

Provide a JSON object with fields for EACH indicator:
"""

        for ind in indicators:
            if ind in INDICATOR_CONFIGS:
                labels = INDICATOR_CONFIGS[ind].labels
                labels_str = " or ".join([f'"{l}"' for l in labels])
                prompt += f'- "{ind}": Must be exactly {labels_str} (string)\n'
                if self.reasoning:
                    prompt += f'- "{ind}_reasoning": Your analysis for this indicator (string)\n'
                prompt += f'- "{ind}_confidence_score": Integer from 1 to 100 (integer)\n'

        prompt += """
**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }
"""
        return prompt

    def _build_combined_user_prompt(
        self,
        indicators: List[str],
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> str:
        """Build a combined user prompt for multiple indicators."""
        # Format reign period (show "unknown" if end year is missing)
        reign_period = f"{start_year}-{end_year if end_year is not None else 'unknown'}"

        prompt = f"""Analyze these political indicators for the following leader's reign:

**Polity:** {polity}
**Leader:** {name}
**Reign Period:** {reign_period}

Respond with ONLY a valid JSON object (no markdown, no extra text):

{{"""

        # Build example structure
        for i, ind in enumerate(indicators):
            if ind in INDICATOR_CONFIGS:
                labels = INDICATOR_CONFIGS[ind].labels
                example_label = labels[0]  # Use first valid label as example

                prompt += f'\n  "{ind}": "{example_label}",'
                if self.reasoning:
                    prompt += f'\n  "{ind}_reasoning": "Your step-by-step analysis for {ind}",'
                prompt += f'\n  "{ind}_confidence_score": 30'  # Example confidence score

                if i < len(indicators) - 1:
                    prompt += ','

        prompt += "\n}"

        return prompt
