"""
Constitution Analysis Pipeline - Utility Modules

This package contains utility modules for the constitution analysis pipeline:
- json_parser: JSON parsing utilities
- cost_tracker: API cost tracking
- logger: Logging utilities
- encoding_fix: CSV encoding utilities
- sanity_check: Data validation utilities
- data_cleaner: Data cleaning and aggregation utilities

For backward compatibility, config is re-exported from root config.py.
LLM clients, prompts, and other modules should be imported from their
respective packages (models/, prompts/, verification/, pipeline/).
"""

__version__ = '2.0.0'

# =============================================================================
# CONFIG RE-EXPORTS (for backward compatibility)
# =============================================================================

from config import (
    # API Configuration
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    # Processing Configuration
    DEFAULT_BATCH_SIZE,
    DEFAULT_DELAY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    # AWS Configuration
    AWS_REGION,
    # Search Configuration
    SERPER_API_URL,
    TOP_SEARCH_RESULTS,
    # Data Column Names
    COL_TERRITORY_NAME,
    COL_START_YEAR,
    COL_END_YEAR,
    REQUIRED_COLUMNS,
    # Model Provider Identifiers
    PROVIDER_OPENAI,
    PROVIDER_GEMINI,
    PROVIDER_BEDROCK,
    PROVIDER_ANTHROPIC,
    OPENAI_MODELS,
    GEMINI_MODELS,
    ANTHROPIC_MODELS,
    BEDROCK_ARN_PREFIX,
    # Indicator Configuration
    ALL_INDICATORS,
    INDICATORS_WITH_GROUND_TRUTH,
    INDICATOR_LABELS,
    # Verification Configuration
    VerificationType,
    PromptMode,
    DEFAULT_VERIFICATION_CONFIG,
    # Output Configuration
    OUTPUT_FORMATS,
    DEFAULT_OUTPUT_DIR,
    # Default Models
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_VERIFIER_MODEL,
)

# =============================================================================
# LOCAL UTILITY MODULE EXPORTS
# =============================================================================

from utils.json_parser import (
    extract_json_from_response,
    parse_json_response,
    validate_indicator_response,
    validate_constitution_response,
)

from utils.cost_tracker import (
    CostTracker,
    UsageRecord,
    ModelUsage,
    PRICING,
    estimate_batch_cost,
)

from utils.logger import (
    setup_logger,
    get_logger,
    ExperimentLogger,
    default_logger,
)

# =============================================================================
# MODULE LIST
# =============================================================================

__all__ = [
    # Config exports
    'DEFAULT_TEMPERATURE',
    'DEFAULT_MAX_TOKENS',
    'DEFAULT_TOP_P',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_DELAY',
    'DEFAULT_MAX_RETRIES',
    'DEFAULT_RETRY_DELAY',
    'AWS_REGION',
    'SERPER_API_URL',
    'TOP_SEARCH_RESULTS',
    'COL_TERRITORY_NAME',
    'COL_START_YEAR',
    'COL_END_YEAR',
    'REQUIRED_COLUMNS',
    'PROVIDER_OPENAI',
    'PROVIDER_GEMINI',
    'PROVIDER_BEDROCK',
    'PROVIDER_ANTHROPIC',
    'OPENAI_MODELS',
    'GEMINI_MODELS',
    'ANTHROPIC_MODELS',
    'BEDROCK_ARN_PREFIX',
    'ALL_INDICATORS',
    'INDICATORS_WITH_GROUND_TRUTH',
    'INDICATOR_LABELS',
    'VerificationType',
    'PromptMode',
    'DEFAULT_VERIFICATION_CONFIG',
    'OUTPUT_FORMATS',
    'DEFAULT_OUTPUT_DIR',
    'DEFAULT_PRIMARY_MODEL',
    'DEFAULT_VERIFIER_MODEL',
    # JSON parser
    'extract_json_from_response',
    'parse_json_response',
    'validate_indicator_response',
    'validate_constitution_response',
    # Cost tracker
    'CostTracker',
    'UsageRecord',
    'ModelUsage',
    'PRICING',
    'estimate_batch_cost',
    # Logger
    'setup_logger',
    'get_logger',
    'ExperimentLogger',
    'default_logger',
]
