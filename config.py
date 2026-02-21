"""
Configuration module for Constitution Analysis Pipeline.

This module centralizes all configuration constants and default values
used throughout the application.

IMPORTANT: This module should ONLY import from the standard library and typing.
Do not import from project modules to avoid circular dependencies.
All project modules can safely import from this config.
"""

from enum import Enum
from typing import List

# =============================================================================
# API Configuration
# =============================================================================

DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 32768  # Increased from 2048 to handle single prompt mode with multiple indicators
                            # NOTE: Actual model limits vary by provider:
                            # - GPT-4o: 16,384 max_tokens
                            # - Gemini 2.5 Pro: 8,192 output tokens (2M input context)
                            # - Claude Sonnet 4.5: 8,192 max_tokens
                            # LLM clients should validate/cap this value before API calls
DEFAULT_TOP_P = 1.0

# =============================================================================
# Processing Configuration
# =============================================================================

DEFAULT_BATCH_SIZE = 100
DEFAULT_DELAY = 1.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5

# =============================================================================
# AWS Configuration
# =============================================================================

AWS_REGION = "us-east-1"

# =============================================================================
# Search Configuration
# =============================================================================

SERPER_API_URL = "https://google.serper.dev/search"
TOP_SEARCH_RESULTS = 5

# =============================================================================
# Data Column Names
# =============================================================================

COL_TERRITORY_NAME = 'territorynamehistorical'
COL_LEADER_NAME = 'name'
COL_START_YEAR = 'start_year'
COL_END_YEAR = 'end_year'
REQUIRED_COLUMNS = [COL_TERRITORY_NAME, COL_LEADER_NAME, COL_START_YEAR, COL_END_YEAR]

# =============================================================================
# Model Provider Identifiers
# =============================================================================

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_BEDROCK = "bedrock"
PROVIDER_ANTHROPIC = "anthropic"

# Model Name Patterns
OPENAI_MODELS = ["gpt-", "o1-", "o3-"]
GEMINI_MODELS = ["gemini-"]
ANTHROPIC_MODELS = ["claude-"]
BEDROCK_ARN_PREFIX = "arn:aws:bedrock"

# =============================================================================
# Indicator Configuration
# =============================================================================

# All available indicators
ALL_INDICATORS = [
    'constitution',
    'sovereign',
    'assembly',
    'appointment',
    'tenure',
    'exit',
    'collegiality',
    'separate_powers',
]

# Indicators with ground truth for evaluation
INDICATORS_WITH_GROUND_TRUTH = [
    'sovereign',
    'assembly',
    'appointment',
    'tenure',
    'exit',
    'collegiality',
    'separate_powers',
]

# Indicator valid labels
INDICATOR_LABELS = {
    'constitution': ['1', '0'],
    'sovereign': ['0', '1'],
    'assembly': ['0', '1'],
    'appointment': ['0', '1', '2'],
    'tenure': ['0', '1', '2'],
    'exit': ['0', '1'],
    'collegiality': ['0', '1'],
    'separate_powers': ['0', '1'],
}

# =============================================================================
# Verification Configuration
# =============================================================================

class VerificationType(Enum):
    """Types of verification methods."""
    NONE = "none"
    SELF_CONSISTENCY = "self_consistency"
    COVE = "cove"
    BOTH = "both"


class PromptMode(Enum):
    """Prompt modes for indicator prediction."""
    SINGLE = "single"          # All indicators in one unified prompt
    MULTIPLE = "multiple"      # Separate prompt per indicator
    SEQUENTIAL = "sequential"  # All 7 indicators in sequence with distinct sections


# Default verification configuration
DEFAULT_VERIFICATION_CONFIG = {
    "self_consistency": {
        "n_samples": 3,
        "temperatures": [0.0, 0.5, 1.0],
        "min_agreement": 0.6
    },
    "cove": {
        "enabled_for": ["constitution"],
        "questions_per_element": 2
    }
}

# =============================================================================
# Output Configuration
# =============================================================================

OUTPUT_FORMATS = ["json", "csv"]
DEFAULT_OUTPUT_DIR = "data/results"

# =============================================================================
# Default Models
# =============================================================================

import os

DEFAULT_PRIMARY_MODEL = "gemini-2.5-pro"

# Bedrock Verifier Model Configuration
# Users can customize by setting BEDROCK_VERIFIER_MODEL in .env
# Format options:
#   1. Model ID: "anthropic.claude-sonnet-4-5-20250929-v1:0"
#   2. Full ARN: "arn:aws:bedrock:region:account:inference-profile/model-id"
#
# If not set, defaults to model ID (will work with your AWS credentials)
DEFAULT_VERIFIER_MODEL = os.getenv(
    'BEDROCK_VERIFIER_MODEL',
    'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
)

# =============================================================================
# Backward Compatibility - Re-export everything from utils.config
# =============================================================================

# These exports maintain backward compatibility with code that imports from utils.config
__all__ = [
    # API Configuration
    'DEFAULT_TEMPERATURE',
    'DEFAULT_MAX_TOKENS',
    'DEFAULT_TOP_P',

    # Processing Configuration
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_DELAY',
    'DEFAULT_MAX_RETRIES',
    'DEFAULT_RETRY_DELAY',

    # AWS Configuration
    'AWS_REGION',

    # Search Configuration
    'SERPER_API_URL',
    'TOP_SEARCH_RESULTS',

    # Data Column Names
    'COL_TERRITORY_NAME',
    'COL_LEADER_NAME',
    'COL_START_YEAR',
    'COL_END_YEAR',
    'REQUIRED_COLUMNS',

    # Model Provider Identifiers
    'PROVIDER_OPENAI',
    'PROVIDER_GEMINI',
    'PROVIDER_BEDROCK',
    'PROVIDER_ANTHROPIC',
    'OPENAI_MODELS',
    'GEMINI_MODELS',
    'ANTHROPIC_MODELS',
    'BEDROCK_ARN_PREFIX',

    # Indicator Configuration
    'ALL_INDICATORS',
    'INDICATORS_WITH_GROUND_TRUTH',
    'INDICATOR_LABELS',

    # Verification Configuration
    'VerificationType',
    'PromptMode',
    'DEFAULT_VERIFICATION_CONFIG',

    # Output Configuration
    'OUTPUT_FORMATS',
    'DEFAULT_OUTPUT_DIR',

    # Default Models
    'DEFAULT_PRIMARY_MODEL',
    'DEFAULT_VERIFIER_MODEL',
]
