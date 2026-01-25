"""
Single Prompt Builder

This module provides a prompt builder that combines multiple indicators
into a single prompt for efficiency. This approach may reduce API costs
but could potentially cause cross-indicator contamination.
"""

from typing import List, Optional

from prompts.base_builder import BasePromptBuilder, PromptOutput
from prompts.constitution import SYSTEM_PROMPT as CONSTITUTION_SYSTEM, USER_PROMPT_TEMPLATE as CONSTITUTION_USER
from prompts.indicators import INDICATOR_PROMPTS


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
        start_year: int,
        end_year: int
    ) -> List[PromptOutput]:
        """
        Build a single combined prompt for all indicators.

        Args:
            polity: Name of the polity
            start_year: Start year of the period
            end_year: End year of the period

        Returns:
            List containing a single PromptOutput (or two if constitution is included)
        """
        prompts = []

        # Handle constitution separately (always its own prompt due to complexity)
        if 'constitution' in self.indicators:
            constitution_user = CONSTITUTION_USER.format(
                country=polity,
                start_year=start_year,
                end_year=end_year
            )
            prompts.append(PromptOutput(
                system_prompt=CONSTITUTION_SYSTEM,
                user_prompt=constitution_user,
                indicators=['constitution'],
                metadata={'mode': 'single', 'type': 'constitution'}
            ))

        # Combine other indicators
        other_indicators = [ind for ind in self.indicators if ind != 'constitution']
        if other_indicators:
            system_prompt = self._build_combined_system_prompt(other_indicators)
            user_prompt = self._build_combined_user_prompt(other_indicators, polity, start_year, end_year)
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

                Your task is to analyze a given polity and determine values for multiple political indicators during its period of existence.

                ## CRITICAL CODING RULE FOR ALL INDICATORS

                This is POLITY-LEVEL classification. You must report the HIGHEST level achieved at ANY point during the given period.

                - For binary indicators (sovereign, powersharing, assembly, exit): Code as "1" if the condition was met at ANY point, "0" only if NEVER
                - For multi-class indicators (appointment, tenure): Report the HIGHEST level achieved during the entire period

                ## Indicators to Analyze

                """
                
        # Simplified definitions - just the core categories
        definitions = {
            'sovereign': '**Sovereign**: Whether the polity had independent foreign policy without subordination to foreign power\n- 0: Colony/vassal/tributary\n- 1: Independent sovereign state',
            'powersharing': '**Powersharing**: Whether multiple individuals shared executive power\n- 0: Single top leader\n- 1: Two+ leaders with comparable power',
            'assembly': '**Assembly**: Whether a legislative assembly/parliament existed\n- 0: No assembly\n- 1: Assembly with role in selection/taxation/policy',
            'appointment': '**Appointment**: How executives were selected\n- 0: Force/hereditary/foreign/military/one-party\n- 1: Royal council/head of state appointment\n- 2: Direct election or assembly selection',
            'tenure': '**Tenure**: Longest executive tenure during period\n- 0: <5 years\n- 1: 5-10 years\n- 2: >10 years',
            'exit': '**Exit**: How executives left power\n- 0: Irregular (died/forced)\n- 1: Regular (voluntary/term limits/electoral defeat)'
        }

        for ind in indicators:
            if ind in definitions:
                prompt += f"{definitions[ind]}\n\n"

        prompt += """## Analysis Rules
            1. **Analyze the ENTIRE period**: Don't just look at the final state
            2. **Report HIGHEST level**: If any part of the period qualifies, code accordingly
            3. **Evidence-based**: Base judgments on verifiable historical facts
            4. **Independent analysis**: Each indicator stands alone

            ## Output Requirements

            Provide a JSON object with fields for EACH indicator:
            """
            
        for ind in indicators:
            labels = INDICATOR_PROMPTS[ind]["labels"]
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
        start_year: int,
        end_year: int
    ) -> str:
        """Build a combined user prompt for multiple indicators."""
        prompt = f"""Analyze these political indicators for the ENTIRE period:

        **Polity:** {polity}
        **Period:** {start_year}-{end_year}

        **CRITICAL**: Report the HIGHEST level achieved at ANY point during {start_year}-{end_year}, not just the final state.

        Respond with ONLY a valid JSON object (no markdown, no extra text):

{{"""

        # Build example structure
        for i, ind in enumerate(indicators):
            labels = INDICATOR_PROMPTS[ind]["labels"]
            example_label = labels[0]  # Use first valid label as example

            prompt += f'\n  "{ind}": "{example_label}",'
            prompt += f'\n  "{ind}_reasoning": "Your step-by-step analysis for {ind}",'
            prompt += f'\n  "{ind}_confidence_score": 85'

            if i < len(indicators) - 1:
                prompt += ','

        prompt += "\n}"

        return prompt
