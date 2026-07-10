"""
Sequential Prompt Builder

Builds ONE combined prompt (same format as single mode) in which the indicator
sections appear in a controllable order. Use it to test order effects within
the single prompt:

- SinglePromptBuilder:     all indicators in one prompt, fixed default order
- MultiplePromptBuilder:   separate LLM call per indicator
- SequentialPromptBuilder: single-mode prompt with user-defined or shuffled
                           section order (this module)

The prompt content is identical to the selected single-prompt version
(v1/v2/v3 from single_builder.py) — only the ORDER of the indicator sections
differs. This isolates section order as the experimental variable.

NOTE: constitution is NOT supported here. It uses its own dedicated prompt
(prompts/constitution.py) and runs as a separate task (--mode multiple or the
batch constitution task).
"""

import random
from typing import List, Optional

from prompts.base_builder import BasePromptBuilder, PromptOutput
from prompts.single_builder import (
    SinglePromptBuilder,
    SinglePromptBuilderV2,
    SinglePromptBuilderV3,
    get_all_indicators as _single_default_indicators,
)

_BUILDER_CLS = {
    "v1": SinglePromptBuilder,
    "v2": SinglePromptBuilderV2,
    "v3": SinglePromptBuilderV3,
}


class SequentialPromptBuilder(BasePromptBuilder):
    """
    Single-mode prompt with a controllable indicator section order.

    The order can be:
    - User-specified via the ``sequence`` parameter
    - Randomized via ``random_order=True`` (shuffled once at construction, so
      every row in a run sees the same order — comparable across rows)
    - Default single-mode order otherwise

    Example:
        # User-specified order
        builder = SequentialPromptBuilder(
            indicators=['sovereign', 'assembly', 'collegiality'],
            sequence=['assembly', 'sovereign', 'collegiality']
        )

        # Random order, token-efficient v3 prompt
        builder = SequentialPromptBuilder(
            indicators=['sovereign', 'assembly'],
            random_order=True,
            prompt_version='v3'
        )

        prompts = builder.build("Roman Republic", "Julius Caesar", -49, -44)
        # Returns a single PromptOutput covering all indicators in the chosen order
    """

    def __init__(
        self,
        indicators: Optional[List[str]] = None,
        sequence: Optional[List[str]] = None,
        random_order: bool = False,
        reasoning: bool = True,
        prompt_version: str = "v1",
    ):
        """
        Args:
            indicators: Indicators to include. Defaults to all single-mode indicators.
            sequence: Explicit section order; must contain exactly the same indicators.
            random_order: Shuffle the section order (once, at construction).
            reasoning: Include per-indicator reasoning fields (default True).
            prompt_version: Which single-prompt variant to use: 'v1', 'v2', or 'v3'.

        Raises:
            ValueError: If constitution is requested, sequence mismatches indicators,
                        both sequence and random_order are given, or prompt_version
                        is unknown.
        """
        if indicators is None:
            indicators = _single_default_indicators()

        if "constitution" in indicators:
            raise ValueError(
                "constitution is not supported in sequential mode: it uses its own "
                "dedicated prompt (prompts/constitution.py). Run it as a separate "
                "task instead (e.g. --indicators constitution --mode multiple, or "
                "the batch constitution task)."
            )

        super().__init__(indicators, reasoning)

        if sequence is not None and random_order:
            raise ValueError("Cannot specify both 'sequence' and 'random_order=True'")

        if prompt_version not in _BUILDER_CLS:
            raise ValueError(
                f"Unknown prompt_version: {prompt_version!r}. "
                f"Must be one of {sorted(_BUILDER_CLS)}."
            )
        self.prompt_version = prompt_version

        if sequence is not None:
            if set(sequence) != set(self.indicators):
                raise ValueError(
                    f"Sequence {sequence} must contain exactly the same indicators "
                    f"as {self.indicators}"
                )
            self.sequence = list(sequence)
        elif random_order:
            self.sequence = self.indicators.copy()
            random.shuffle(self.sequence)
        else:
            self.sequence = self._default_order()

        # The inner single-prompt builder renders sections in list order, so the
        # sequential prompt is exactly the single prompt with reordered sections.
        self._inner = _BUILDER_CLS[prompt_version](
            indicators=self.sequence, reasoning=reasoning
        )

    def _default_order(self) -> List[str]:
        """Default ordering: single-mode default order filtered to selected indicators."""
        default = _single_default_indicators()
        ordered = [ind for ind in default if ind in self.indicators]
        # Preserve any indicators not covered by the default list (defensive)
        ordered += [ind for ind in self.indicators if ind not in ordered]
        return ordered

    def build(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> List[PromptOutput]:
        """
        Build a single combined prompt with sections in ``self.sequence`` order.

        Returns:
            List containing one PromptOutput covering all indicators.
        """
        inner = self._inner.build(polity, name, start_year, end_year)[0]
        return [PromptOutput(
            system_prompt=inner.system_prompt,
            user_prompt=inner.user_prompt,
            indicators=list(self.sequence),
            metadata={
                "mode": "sequential",
                "version": self.prompt_version,
                "sequence": list(self.sequence),
                "num_indicators": len(self.sequence),
            },
        )]

    def __repr__(self) -> str:
        return (
            f"SequentialPromptBuilder(sequence={self.sequence}, "
            f"prompt_version={self.prompt_version!r})"
        )
