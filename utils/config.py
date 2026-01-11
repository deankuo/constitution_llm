"""
Configuration module for Constitution Analysis Pipeline.

This module centralizes all configuration constants and default values
used throughout the application.
"""

# API Configuration
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 1.0

# Processing Configuration
DEFAULT_BATCH_SIZE = 100
DEFAULT_DELAY = 1.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5

# AWS Configuration
AWS_REGION = "us-east-1"

# Search Configuration
SERPER_API_URL = "https://google.serper.dev/search"
TOP_SEARCH_RESULTS = 5

# Data Column Names
COL_TERRITORY_NAME = 'territorynamehistorical'
COL_START_YEAR = 'start_year'
COL_END_YEAR = 'end_year'
REQUIRED_COLUMNS = [COL_TERRITORY_NAME, COL_START_YEAR, COL_END_YEAR]

# Model Provider Identifiers
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_BEDROCK = "bedrock"
PROVIDER_ANTHROPIC = "anthropic"

# Model Name Patterns
OPENAI_MODELS = ["gpt-", "o1-", "o3-"]
GEMINI_MODELS = ["gemini-"]
ANTHROPIC_MODELS = ["claude-"]
BEDROCK_ARN_PREFIX = "arn:aws:bedrock"
