"""
Multiple Prompt Builder

This module provides a prompt builder that generates separate prompts
for each indicator. This approach prevents cross-indicator contamination
but requires more API calls.
"""

from typing import List, Optional

from prompts.base_builder import BasePromptBuilder, PromptOutput
from prompts.constitution import get_prompt as get_constitution_prompt
from prompts.indicators import get_prompt as get_indicator_prompt


class MultiplePromptBuilder(BasePromptBuilder):
    """
    Generates a separate prompt for each indicator.

    This builder creates independent prompts for each indicator,
    ensuring that predictions are not influenced by other indicators.
    This is more expensive (more API calls) but may provide more
    accurate predictions.

    Example:
        builder = MultiplePromptBuilder(indicators=['sovereign', 'assembly', 'collegiality'])
        prompts = builder.build("Roman Republic", -509, -27)
        # Returns 3 PromptOutput objects, one per indicator
    """

    def __init__(self, indicators: Optional[List[str]] = None, reasoning: bool = True):
        """
        Initialize the multiple prompt builder.

        Args:
            indicators: List of indicators to include. If None, uses all available.
            reasoning: Whether to include reasoning in prompts for non-constitution indicators (default True).
        """
        super().__init__(indicators, reasoning)

    def build(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> List[PromptOutput]:
        """
        Build separate prompts for each indicator.

        Args:
            polity: Name of the polity
            name: Name of the leader
            start_year: Start year of the leader's reign
            end_year: End year of the leader's reign (None if unknown/unavailable)

        Returns:
            List of PromptOutput objects, one per indicator
        """
        prompts = []

        for indicator in self.indicators:
            if indicator == 'constitution':
                # Use the special constitution prompt
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
                    metadata={'mode': 'multiple', 'type': 'constitution'}
                ))
            else:
                # Use the standard indicator prompts
                system_prompt, user_prompt = get_indicator_prompt(
                    indicator=indicator,
                    polity=polity,
                    name=name,
                    start_year=start_year,
                    end_year=end_year,
                    reasoning=self.reasoning
                )
                prompts.append(PromptOutput(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    indicators=[indicator],
                    metadata={'mode': 'multiple', 'type': 'standard'}
                ))

        return prompts


def create_prompt_builder(
    mode: str,
    indicators: Optional[List[str]] = None,
    sequence: Optional[List[str]] = None,
    random_order: bool = False,
    reasoning: bool = True
) -> BasePromptBuilder:
    """
    Factory function to create a prompt builder.

    Args:
        mode: Either 'single', 'multiple', or 'sequential'
        indicators: List of indicators to include
        sequence: Specific order for sequential mode (optional)
        random_order: Randomize order in sequential mode (optional)
        reasoning: Whether to include reasoning for non-constitution indicators (default True)

    Returns:
        Appropriate PromptBuilder instance
    """
    from prompts.single_builder import SinglePromptBuilder
    from prompts.sequential_builder import SequentialPromptBuilder

    if mode.lower() == 'single':
        return SinglePromptBuilder(indicators=indicators, reasoning=reasoning)
    elif mode.lower() == 'multiple':
        return MultiplePromptBuilder(indicators=indicators, reasoning=reasoning)
    elif mode.lower() == 'sequential':
        return SequentialPromptBuilder(
            indicators=indicators,
            sequence=sequence,
            random_order=random_order,
            reasoning=reasoning
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'single', 'multiple', or 'sequential'")
