"""
Abstract base class for prompt builders.

This module defines the interface for prompt builders that can combine
indicators into single or multiple prompts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PromptOutput:
    """
    Output structure from prompt builders.

    Attributes:
        system_prompt: The system prompt/instruction
        user_prompt: The user message/query
        indicators: List of indicators covered by this prompt
        metadata: Additional metadata about the prompt
    """
    system_prompt: str
    user_prompt: str
    indicators: List[str]
    metadata: Dict = field(default_factory=dict)


class BasePromptBuilder(ABC):
    """
    Abstract base class for prompt builders.

    Prompt builders are responsible for constructing prompts for
    political indicator analysis. They can combine multiple indicators
    into a single prompt (SinglePromptBuilder) or generate separate
    prompts per indicator (MultiplePromptBuilder).

    Example:
        builder = SinglePromptBuilder(indicators=['sovereign', 'assembly'])
        prompts = builder.build(
            polity="Roman Republic",
            start_year=-509,
            end_year=-27
        )
        for prompt in prompts:
            response = llm.call(prompt.system_prompt, prompt.user_prompt)
    """

    def __init__(self, indicators: Optional[List[str]] = None):
        """
        Initialize the prompt builder.

        Args:
            indicators: List of indicators to include. If None, uses all available.
        """
        self.indicators = indicators or self.get_available_indicators()
        self._validate_indicators()

    def _validate_indicators(self) -> None:
        """Validate that all specified indicators are available."""
        available = set(self.get_available_indicators())
        for ind in self.indicators:
            if ind not in available:
                raise ValueError(
                    f"Unknown indicator: {ind}. "
                    f"Available indicators: {sorted(available)}"
                )

    @abstractmethod
    def build(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> List[PromptOutput]:
        """
        Build prompts for the specified leader.

        Args:
            polity: Name of the polity
            name: Name of the leader
            start_year: Start year of the leader's reign
            end_year: End year of the leader's reign (None if unknown/unavailable in data)

        Returns:
            List of PromptOutput objects
        """
        pass

    @staticmethod
    def format_year_range(start_year: int, end_year: Optional[int]) -> str:
        """
        Format year range for prompts, handling missing end_year.

        Args:
            start_year: Start year
            end_year: End year (None if unknown/unavailable)

        Returns:
            Formatted string like "1990-2000" or "2020-unknown"
        """
        if end_year is None:
            return f"{start_year}-unknown"
        return f"{start_year}-{end_year}"

    @staticmethod
    def get_available_indicators() -> List[str]:
        """
        Get list of all available indicators.

        Returns:
            List of indicator names
        """
        return [
            'constitution',
            'sovereign',
            'powersharing',
            'assembly',
            'appointment',
            'tenure',
            'exit'
        ]

    def get_indicator_labels(self, indicator: str) -> List[str]:
        """
        Get valid labels for an indicator.

        Args:
            indicator: Name of the indicator

        Returns:
            List of valid label values
        """
        from config import INDICATOR_LABELS
        if indicator in INDICATOR_LABELS:
            return INDICATOR_LABELS[indicator]
        raise ValueError(f"Unknown indicator: {indicator}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(indicators={self.indicators})"
