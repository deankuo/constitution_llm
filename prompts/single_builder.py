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

## Indicators to Analyze

"""
        for ind in indicators:
            if ind in INDICATOR_PROMPTS:
                # Extract the definition section from each indicator's system prompt
                full_prompt = INDICATOR_PROMPTS[ind]["system"]
                # Find the definition section
                if "## Definition" in full_prompt:
                    start = full_prompt.find("## Definition")
                    end = full_prompt.find("## ", start + 1)
                    if end == -1:
                        end = full_prompt.find("## Output Requirements")
                    definition = full_prompt[start:end].strip() if end != -1 else full_prompt[start:].strip()
                    prompt += f"### {ind.upper()}\n\n{definition}\n\n"

        prompt += """## General Rules

1. **Polity-Level Aggregation**: For all binary indicators, if the condition was met at ANY point during the period, code as "1".
2. **Multi-class Indicators**: For appointment and tenure, report the HIGHEST level achieved during the period.
3. **Independence**: Analyze each indicator independently based on its specific criteria.
4. **Evidence-Based**: Base all judgments on verifiable historical facts.

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
        indicators_list = ", ".join(indicators)

        prompt = f"""Please analyze the following political indicators for this polity:

**Polity:** {polity}
**Period:** {start_year}-{end_year}

## Indicators to Analyze: {indicators_list}

For each indicator, provide:
1. The classification value
2. Your reasoning
3. A confidence score (1-100)

Respond with a single JSON object containing all indicators:

{{
"""
        for ind in indicators:
            labels = INDICATOR_PROMPTS[ind]["labels"]
            labels_str = " or ".join(labels)
            prompt += f'  "{ind}": "{labels_str}",\n'
            prompt += f'  "{ind}_reasoning": "your analysis",\n'
            prompt += f'  "{ind}_confidence_score": 1-100,\n'

        prompt = prompt.rstrip(",\n") + "\n}"

        return prompt
