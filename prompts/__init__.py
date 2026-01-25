"""
Prompt templates and builders for political indicator analysis.

This package contains:
- constitution: Constitution indicator prompts (complex)
- indicators: Other 6 indicators with unified templates
- base_builder: Abstract prompt builder interface
- single_builder: Combines all indicators into one prompt
- multiple_builder: Generates separate prompt per indicator
"""

__version__ = '1.0.0'

# Re-export key functions for convenience
from prompts.constitution import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    get_constitution_prompt,
    get_constitution_labels,
    get_cove_questions as get_constitution_cove_questions,
)

from prompts.indicators import (
    INDICATOR_PROMPTS,
    get_prompt,
    get_all_indicators,
    get_indicator_labels,
    get_cove_questions,
)

from prompts.base_builder import BasePromptBuilder, PromptOutput
from prompts.single_builder import SinglePromptBuilder
from prompts.multiple_builder import MultiplePromptBuilder, create_prompt_builder
