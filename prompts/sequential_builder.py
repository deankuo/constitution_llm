"""
Sequential Prompt Builder

This module provides a prompt builder that combines all 7 indicators
(constitution + 6 others) sequentially in a single prompt. The order
of indicators can be specified by the user or randomized.

This approach differs from:
- SinglePromptBuilder: Merges indicators into unified prompt
- MultiplePromptBuilder: Separate LLM calls per indicator
- SequentialPromptBuilder: One LLM call with 7 sequential sections

The key feature is that each indicator maintains its EXACT original prompt
structure from indicators.py and constitution.py.
"""

import random
from typing import List, Optional

from prompts.base_builder import BasePromptBuilder, PromptOutput
from prompts.constitution import get_prompt as get_constitution_prompt
from prompts.indicators import get_prompt as get_indicator_prompt, INDICATOR_CONFIGS


class SequentialPromptBuilder(BasePromptBuilder):
    """
    Combines all 7 indicator prompts sequentially in a single prompt.

    This builder creates one comprehensive prompt by placing all 7 indicator
    sections in sequence. Each indicator uses its exact original prompt from
    indicators.py or constitution.py.

    The order can be:
    - User-specified via sequence parameter
    - Randomized via random_order parameter
    - Default order (constitution first, then others in standard order)

    Example:
        # User-specified order
        builder = SequentialPromptBuilder(
            indicators=['constitution', 'assembly', 'sovereign'],
            sequence=['assembly', 'constitution', 'sovereign']
        )

        # Random order
        builder = SequentialPromptBuilder(
            indicators=['constitution', 'sovereign', 'assembly'],
            random_order=True
        )

        prompts = builder.build("Roman Republic", -509, -27)
        # Returns single PromptOutput covering all 3 indicators in specified order
    """

    def __init__(
        self,
        indicators: Optional[List[str]] = None,
        sequence: Optional[List[str]] = None,
        random_order: bool = False,
        reasoning: bool = True
    ):
        """
        Initialize the sequential prompt builder.

        Args:
            indicators: List of indicators to include. If None, uses all 7 indicators.
            sequence: Specific order of indicators. If provided, must include all indicators.
            random_order: If True, randomize the order of indicators.
            reasoning: Whether to include reasoning for non-constitution indicators (default True).

        Raises:
            ValueError: If sequence doesn't match indicators or contains invalid indicators.
        """
        # Default to all 7 indicators if not specified
        if indicators is None:
            indicators = ['constitution', 'sovereign', 'powersharing', 'assembly', 'appointment', 'tenure', 'exit']

        super().__init__(indicators, reasoning)

        # Validate and set sequence
        if sequence is not None and random_order:
            raise ValueError("Cannot specify both 'sequence' and 'random_order=True'")

        if sequence is not None:
            # Validate that sequence contains exactly the same indicators
            if set(sequence) != set(self.indicators):
                raise ValueError(
                    f"Sequence {sequence} must contain exactly the same indicators as {self.indicators}"
                )
            self.sequence = sequence
        elif random_order:
            self.sequence = self.indicators.copy()
            random.shuffle(self.sequence)
        else:
            # Default order: constitution first, then others
            self.sequence = self._default_order()

    def _default_order(self) -> List[str]:
        """
        Get default ordering for indicators.

        Returns constitution first, then others in standard order.
        """
        default = ['constitution', 'sovereign', 'powersharing', 'assembly', 'appointment', 'tenure', 'exit']
        return [ind for ind in default if ind in self.indicators]

    def build(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> List[PromptOutput]:
        """
        Build a single sequential prompt with all indicators.

        Args:
            polity: Name of the polity
            name: Name of the leader
            start_year: Start year of the leader's reign
            end_year: End year of the leader's reign (None if unknown/unavailable)

        Returns:
            List containing a single PromptOutput with all indicators in sequence
        """
        system_prompt = self._build_sequential_system_prompt(self.sequence, polity, name, start_year, end_year)
        user_prompt = self._build_sequential_user_prompt(self.sequence, polity, name, start_year, end_year)

        return [PromptOutput(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            indicators=self.sequence,  # Preserve order in metadata
            metadata={
                'mode': 'sequential',
                'sequence': self.sequence,
                'num_indicators': len(self.sequence)
            }
        )]

    def _build_sequential_system_prompt(
        self,
        sequence: List[str],
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> str:
        """
        Build a sequential system prompt by concatenating existing prompts.

        Args:
            sequence: Ordered list of indicators
            polity: Polity name
            name: Leader name
            start_year: Start year of the leader's reign
            end_year: End year of the leader's reign (None if unknown/unavailable)

        Returns:
            Combined system prompt with all indicators in sequence
        """
        prompt = f"""You are a professional political scientist and historian specializing in comparative politics across different historical periods.

Your task is to analyze the leader "{name}" of "{polity}" ({start_year} to {end_year}) across {len(sequence)} political indicators.

You will analyze each indicator sequentially. Each indicator has its own definition and coding rules provided below.

⚠️ **CRITICAL OUTPUT REQUIREMENT:**
You MUST provide predictions for ALL {len(sequence)} indicators in a SINGLE JSON object at the end.

═══════════════════════════════════════════════════════════════════════

"""

        # Add each indicator's system prompt in sequence
        for i, indicator in enumerate(sequence, 1):
            prompt += f"## INDICATOR {i}/{len(sequence)}: {indicator.upper()}\n\n"

            if indicator == 'constitution':
                # Use constitution's system prompt (always includes reasoning)
                system_prompt, _ = get_constitution_prompt(polity, name, start_year, end_year)
                prompt += system_prompt
            else:
                # Use indicator's system prompt from indicators.py
                system_prompt, _ = get_indicator_prompt(indicator, polity, name, start_year, end_year, reasoning=self.reasoning)
                prompt += system_prompt

            prompt += "\n\n═══════════════════════════════════════════════════════════════════════\n\n"

        # Add combined output instructions
        prompt += self._build_output_instructions(sequence)

        return prompt

    def _build_output_instructions(self, sequence: List[str]) -> str:
        """
        Build output format instructions for all indicators.

        Args:
            sequence: Ordered list of indicators

        Returns:
            Output format instructions
        """
        prompt = f"""## ⚠️ COMBINED OUTPUT FORMAT

You have analyzed {len(sequence)} indicators above. Now provide a SINGLE JSON object containing predictions for ALL indicators.

**Required structure:**

{{"""

        # Add expected fields for each indicator
        for i, indicator in enumerate(sequence):
            prompt += "\n  "

            if indicator == 'constitution':
                prompt += f'"constitution": "Yes or No",\n  '
                prompt += f'"document_name": "name or N/A",\n  '
                prompt += f'"constitution_year": "exact integer year(s) or N/A (no circa/c.)",\n  '
                prompt += f'"constitution_reasoning": "your constitutional analysis",\n  '
                prompt += f'"constitution_confidence_score": 1-100'
            else:
                labels = INDICATOR_CONFIGS[indicator].labels
                labels_str = " or ".join([f'"{l}"' for l in labels])
                prompt += f'"{indicator}": {labels_str},\n  '
                if self.reasoning:
                    prompt += f'"{indicator}_reasoning": "your {indicator} analysis",\n  '
                prompt += f'"{indicator}_confidence_score": 1-100'

            if i < len(sequence) - 1:
                prompt += ','

        prompt += "\n}"

        if not self.reasoning:
            prompt += "\n**DO NOT include any reasoning or analysis fields for the 6 non-constitution indicators. Only include prediction and confidence_score fields.**\n"

        prompt += """
**CRITICAL REQUIREMENTS:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }
- Include ALL indicators in the single JSON object
- Use the exact field names shown above
"""

        return prompt

    def _build_sequential_user_prompt(
        self,
        sequence: List[str],
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> str:
        """
        Build user prompt for sequential analysis.

        Args:
            sequence: Ordered list of indicators
            polity: Name of the polity
            name: Name of the leader
            start_year: Start year of the leader's reign
            end_year: End year of the leader's reign

        Returns:
            User prompt text
        """
        prompt = f"""Please analyze the following leader's reign across {len(sequence)} political indicators:

**Polity:** {polity}
**Leader:** {name}
**Reign Period:** {start_year} to {end_year}

You will analyze these indicators in sequence:
"""

        for i, indicator in enumerate(sequence, 1):
            prompt += f"{i}. {indicator}\n"

        prompt += f"\n{'='*70}\n\n"

        # Add each indicator's user prompt
        for i, indicator in enumerate(sequence, 1):
            prompt += f"### INDICATOR {i}/{len(sequence)}: {indicator.upper()}\n\n"

            if indicator == 'constitution':
                # Use constitution's user prompt (always includes reasoning)
                _, user_prompt = get_constitution_prompt(polity, name, start_year, end_year)
                prompt += user_prompt
            else:
                # Use indicator's user prompt from indicators.py
                _, user_prompt = get_indicator_prompt(indicator, polity, name, start_year, end_year, reasoning=self.reasoning)
                prompt += user_prompt

            prompt += f"\n\n{'='*70}\n\n"

        # Final reminder for combined output
        prompt += f"""## ⚠️ NOW PROVIDE YOUR COMBINED ANALYSIS

You have been asked to analyze {len(sequence)} indicators above.

Provide a SINGLE JSON object with predictions for ALL {len(sequence)} indicators.

Start your response with {{ and end with }}. No markdown, no extra text.

Your JSON object should include all fields for all {len(sequence)} indicators as specified in the output format above.
"""

        return prompt

    def __repr__(self) -> str:
        return f"SequentialPromptBuilder(indicators={self.indicators}, sequence={self.sequence})"
