"""
Single Prompt Builder

This module provides a prompt builder that combines multiple indicators
into a single prompt for efficiency. This approach may reduce API costs
but could potentially cause cross-indicator contamination.
"""

from typing import List, Optional

from prompts.base_builder import BasePromptBuilder, PromptOutput
from prompts.constitution import get_prompt as get_constitution_prompt
from prompts.indicators import INDICATOR_CONFIGS


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

    def __init__(self, indicators: Optional[List[str]] = None):
        """
        Initialize the single prompt builder.

        Args:
            indicators: List of indicators to include. If None, uses all non-constitution indicators.
        """
        # Default to all indicators except constitution (which has special handling)
        if indicators is None:
            indicators = ['sovereign', 'powersharing', 'assembly', 'appointment', 'tenure', 'exit']
        super().__init__(indicators)

    def build(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: int
    ) -> List[PromptOutput]:
        """
        Build a single combined prompt for all indicators.

        Args:
            polity: Name of the polity
            name: Name of the leader
            start_year: Start year of the leader's reign
            end_year: End year of the leader's reign

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

        # Import definitions from INDICATOR_CONFIGS (no duplication)
        for ind in indicators:
            if ind in INDICATOR_CONFIGS:
                config = INDICATOR_CONFIGS[ind]
                prompt += f"**{config.display_name.upper()}**\n"
                labels_str = " / ".join(config.labels)
                prompt += f"Labels: {labels_str}\n"
                # Add brief summary of definition (first 200 chars to reduce tokens)
                definition_summary = config.definition.strip()[:200].replace('\n', ' ')
                prompt += f"{definition_summary}...\n\n"

        prompt += """## Output Requirements

Provide a JSON object with fields for EACH indicator:
"""

        for ind in indicators:
            if ind in INDICATOR_CONFIGS:
                labels = INDICATOR_CONFIGS[ind].labels
                labels_str = " or ".join([f'"{l}"' for l in labels])
                prompt += f'- "{ind}": Must be exactly {labels_str} (string)\n'
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
        end_year: int
    ) -> str:
        """Build a combined user prompt for multiple indicators."""
        prompt = f"""Analyze these political indicators for the following leader's reign:

**Polity:** {polity}
**Leader:** {name}
**Reign Period:** {start_year}-{end_year}

Respond with ONLY a valid JSON object (no markdown, no extra text):

{{"""

        # Build example structure
        for i, ind in enumerate(indicators):
            if ind in INDICATOR_CONFIGS:
                labels = INDICATOR_CONFIGS[ind].labels
                example_label = labels[0]  # Use first valid label as example

                prompt += f'\n  "{ind}": "{example_label}",'
                prompt += f'\n  "{ind}_reasoning": "Your step-by-step analysis for {ind}",'
                prompt += f'\n  "{ind}_confidence_score": 30'  # Example confidence score

                if i < len(indicators) - 1:
                    prompt += ','

        prompt += "\n}"

        return prompt
