"""
LLM client implementations for various providers.

This module provides class-based LLM implementations:
- OpenAILLM: OpenAI GPT models
- GeminiLLM: Google Gemini models
- AnthropicLLM: Anthropic Claude models (direct API)
- BedrockLLM: AWS Bedrock models

Also provides backward-compatible function interfaces and a factory function.
"""

import time
import traceback
from typing import Any, Dict, Optional, Union

import boto3
import google.generativeai as genai
from anthropic import Anthropic
from botocore.exceptions import ClientError
from openai import OpenAI

from models.base import BaseLLM, ModelResponse, APIKeyError, RateLimitError, ModelError
from config import (
    AWS_REGION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    OPENAI_MODELS,
    GEMINI_MODELS,
    ANTHROPIC_MODELS,
    BEDROCK_ARN_PREFIX
)


# =============================================================================
# OpenAI LLM Implementation
# =============================================================================

class OpenAILLM(BaseLLM):
    """OpenAI GPT model implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        default_temperature: float = DEFAULT_TEMPERATURE,
        default_max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        super().__init__(api_key, model, default_temperature, default_max_tokens)
        self.client = None

    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self.client is None:
            if not self.api_key:
                raise APIKeyError("OpenAI API key not provided")
            self.client = OpenAI(api_key=self.api_key)
        return self.client

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Call OpenAI model."""
        try:
            client = self._get_client()
            model_to_use = model or self.model

            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature if temperature is not None else self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                top_p=top_p if top_p is not None else DEFAULT_TOP_P
            )

            content = response.choices[0].message.content or ""
            usage = {
                'input_tokens': response.usage.prompt_tokens if response.usage else 0,
                'output_tokens': response.usage.completion_tokens if response.usage else 0,
                'cached_tokens': getattr(response.usage, 'prompt_tokens_details', {}).get('cached_tokens', 0) if response.usage else 0,
                'thinking_tokens': 0,
            }

            return ModelResponse(
                content=content.strip(),
                model=model_to_use,
                usage=usage,
                raw_response=response
            )

        except Exception as e:
            raise ModelError(f"Error querying OpenAI model '{self.model}': {e}")


# =============================================================================
# Gemini LLM Implementation
# =============================================================================

class GeminiLLM(BaseLLM):
    """Google Gemini model implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-pro",
        default_temperature: float = DEFAULT_TEMPERATURE,
        default_max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        super().__init__(api_key, model, default_temperature, default_max_tokens)
        self._configured = False

    def _configure(self) -> None:
        """Configure Gemini API."""
        if not self._configured:
            if not self.api_key:
                raise APIKeyError("Google Gemini API key not provided")
            genai.configure(api_key=self.api_key)
            self._configured = True

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Call Gemini model."""
        try:
            self._configure()
            model_to_use = model or self.model

            generation_config = genai.GenerationConfig(
                temperature=temperature if temperature is not None else self.default_temperature,
                top_p=top_p if top_p is not None else DEFAULT_TOP_P,
                max_output_tokens=max_tokens or self.default_max_tokens
            )

            # Safety settings - set to BLOCK_NONE for academic/historical analysis
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            gemini_model = genai.GenerativeModel(
                model_name=model_to_use,
                system_instruction=system_prompt,
                safety_settings=safety_settings
            )

            tool_config = {
                "function_calling_config": {
                    "mode": "NONE"
                }
            }

            response = gemini_model.generate_content(
                user_prompt,
                generation_config=generation_config,
                tool_config=tool_config
            )

            if not response.parts:
                # Get detailed feedback
                feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'No feedback available'
                block_reason = getattr(feedback, 'block_reason', 'UNKNOWN') if feedback != 'No feedback available' else 'UNKNOWN'

                raise ModelError(
                    f"Gemini response was empty or blocked.\n"
                    f"Block reason: {block_reason}\n"
                    f"Full feedback: {feedback}"
                )

            # Extract usage if available
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                metadata = response.usage_metadata
                # CRITICAL: For thinking models (e.g., Gemini 2.5 Pro), thinking tokens
                # are billed as output but NOT included in candidates_token_count.
                # They must be added together to get the true billable output token count.
                # Cost calculation happens in CostTracker.calculate_cost() using this sum.
                thinking_tokens = getattr(metadata, 'thoughts_token_count', 0) or 0
                usage = {
                    'input_tokens': metadata.prompt_token_count or 0,
                    'output_tokens': (metadata.candidates_token_count or 0) + thinking_tokens,
                    'cached_tokens': metadata.cached_content_token_count or 0,
                    'total_tokens': metadata.total_token_count or 0,
                    'thinking_tokens': thinking_tokens,
                }

            return ModelResponse(
                content=response.text.strip(),
                model=model_to_use,
                usage=usage,
                raw_response=response
            )

        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f"Error querying Gemini model '{self.model}': {e}")


# =============================================================================
# Anthropic LLM Implementation
# =============================================================================

class AnthropicLLM(BaseLLM):
    """Anthropic Claude model implementation (direct API)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        default_temperature: float = DEFAULT_TEMPERATURE,
        default_max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        super().__init__(api_key, model, default_temperature, default_max_tokens)
        self.client = None

    def _get_client(self) -> Anthropic:
        """Get or create Anthropic client."""
        if self.client is None:
            if not self.api_key:
                raise APIKeyError("Anthropic API key not provided")
            self.client = Anthropic(api_key=self.api_key)
        return self.client

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Call Anthropic model."""
        try:
            client = self._get_client()
            model_to_use = model or self.model

            response = client.messages.create(
                model=model_to_use,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature if temperature is not None else self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                top_p=top_p if top_p is not None else DEFAULT_TOP_P
            )

            if response.content and len(response.content) > 0:
                content = response.content[0].text
            else:
                content = ""

            # Anthropic returns cache_read_input_tokens when using prompt caching
            cached_tokens = 0
            if response.usage:
                cached_tokens = getattr(response.usage, 'cache_read_input_tokens', 0)

            usage = {
                'input_tokens': response.usage.input_tokens if response.usage else 0,
                'output_tokens': response.usage.output_tokens if response.usage else 0,
                'cached_tokens': cached_tokens,
                'thinking_tokens': 0,
            }

            return ModelResponse(
                content=content.strip(),
                model=model_to_use,
                usage=usage,
                raw_response=response
            )

        except Exception as e:
            raise ModelError(f"Error querying Anthropic model '{self.model}': {e}")


# =============================================================================
# AWS Bedrock LLM Implementation
# =============================================================================

class BedrockLLM(BaseLLM):
    """AWS Bedrock model implementation."""

    def __init__(
        self,
        model: str = "anthropic.claude-sonnet-4-5-20250929-v1:0",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region: str = AWS_REGION,
        default_temperature: float = DEFAULT_TEMPERATURE,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY
    ):
        super().__init__(None, model, default_temperature, default_max_tokens)
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.region = region
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None

    def _get_client(self) -> Any:
        """Get or create Bedrock client."""
        if self.client is None:
            client_args = {"region_name": self.region}

            if self.aws_access_key_id and self.aws_secret_access_key:
                client_args['aws_access_key_id'] = self.aws_access_key_id
                client_args['aws_secret_access_key'] = self.aws_secret_access_key
                if self.aws_session_token:
                    client_args['aws_session_token'] = self.aws_session_token

            self.client = boto3.client("bedrock-runtime", **client_args)

        return self.client

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Call Bedrock model with retry logic."""
        model_to_use = model or self.model

        for attempt in range(1, self.max_retries + 1):
            try:
                client = self._get_client()

                messages = [{"role": "user", "content": [{"text": user_prompt}]}]
                system_messages = [{"text": system_prompt}]
                inference_config = {
                    'maxTokens': max_tokens or self.default_max_tokens,
                    'temperature': temperature if temperature is not None else self.default_temperature,
                }

                response = client.converse(
                    modelId=model_to_use,
                    messages=messages,
                    system=system_messages,
                    inferenceConfig=inference_config
                )

                content = response["output"]["message"]["content"][0]["text"]

                # Extract usage from response
                # Bedrock returns cacheReadInputTokens for cached content
                usage = {}
                if "usage" in response:
                    usage = {
                        'input_tokens': response["usage"].get("inputTokens", 0),
                        'output_tokens': response["usage"].get("outputTokens", 0),
                        'cached_tokens': response["usage"].get("cacheReadInputTokens", 0),
                        'thinking_tokens': 0,
                    }

                return ModelResponse(
                    content=content.strip(),
                    model=model_to_use,
                    usage=usage,
                    raw_response=response
                )

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if 'ThrottlingException' in error_code:
                    if attempt < self.max_retries:
                        print(f"WARN: Bedrock API throttling. Retrying in {self.retry_delay}s... "
                              f"(Attempt {attempt}/{self.max_retries})")
                        time.sleep(self.retry_delay)
                        continue
                    raise RateLimitError(f"Bedrock rate limit exceeded after {self.max_retries} retries")
                raise ModelError(f"Bedrock API error: {e}")

            except Exception as e:
                if attempt < self.max_retries:
                    print(f"ERROR: Bedrock error. Retrying in {self.retry_delay}s... "
                          f"(Attempt {attempt}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                    continue
                raise ModelError(f"Error querying Bedrock model '{model_to_use}': {e}")

        raise ModelError(f"Failed after {self.max_retries} attempts")


# =============================================================================
# Factory Function
# =============================================================================

def create_llm(
    model: str,
    api_keys: Dict[str, str],
    **kwargs
) -> BaseLLM:
    """
    Factory function to create an LLM instance based on model name.

    Args:
        model: Model name or identifier
        api_keys: Dictionary of API keys
        **kwargs: Additional arguments passed to the LLM constructor

    Returns:
        Appropriate LLM instance
    """
    provider = detect_provider(model)

    if provider == 'openai':
        return OpenAILLM(
            api_key=api_keys.get('openai'),
            model=model,
            **kwargs
        )
    elif provider == 'gemini':
        return GeminiLLM(
            api_key=api_keys.get('gemini'),
            model=model,
            **kwargs
        )
    elif provider == 'anthropic':
        return AnthropicLLM(
            api_key=api_keys.get('anthropic'),
            model=model,
            **kwargs
        )
    elif provider == 'bedrock':
        return BedrockLLM(
            model=model,
            aws_access_key_id=api_keys.get('aws_access_key_id'),
            aws_secret_access_key=api_keys.get('aws_secret_access_key'),
            aws_session_token=api_keys.get('aws_session_token'),
            **kwargs
        )
    else:
        # Default to OpenAI
        return OpenAILLM(
            api_key=api_keys.get('openai'),
            model=model,
            **kwargs
        )


def detect_provider(model_identifier: str) -> str:
    """
    Detect the LLM provider based on model identifier.

    Args:
        model_identifier: Model name or ARN

    Returns:
        Provider name: 'openai', 'gemini', 'anthropic', or 'bedrock'
    """
    # Check for Bedrock ARN format first
    if model_identifier.startswith(BEDROCK_ARN_PREFIX):
        return 'bedrock'

    model_lower = model_identifier.lower()

    # Check for Bedrock model ID format (anthropic.*, amazon.*, etc.)
    # Bedrock models have the format: provider.model-name-version
    # Also handle regional prefixes: us.anthropic.*, global.anthropic.*, etc.
    bedrock_prefixes = (
        'anthropic.', 'amazon.', 'meta.', 'cohere.', 'ai21.', 'mistral.',
        'us.anthropic.', 'us.amazon.', 'us.meta.',
        'global.anthropic.', 'global.amazon.', 'global.meta.',
        'eu.anthropic.', 'eu.amazon.', 'eu.meta.',
        'ap.anthropic.', 'ap.amazon.', 'ap.meta.',
    )
    if model_lower.startswith(bedrock_prefixes):
        return 'bedrock'

    # Check for direct Anthropic API (claude- without the anthropic. prefix)
    if any(pattern in model_lower for pattern in ANTHROPIC_MODELS):
        return 'anthropic'

    if any(pattern in model_lower for pattern in GEMINI_MODELS):
        return 'gemini'

    # Default to OpenAI for gpt-, o1-, o3-, and other models
    return 'openai'


# =============================================================================
# Backward Compatible Function Interfaces
# =============================================================================

def query_openai_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: Optional[str] = None,
    llm_params: Optional[Dict] = None,
    max_retries: int = 3,
    retry_delay: int = 5
) -> Optional[str]:
    """
    Query OpenAI model (backward compatible function interface).

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        model: Model name (e.g., 'gpt-4o')
        api_key: OpenAI API key
        llm_params: Dictionary with parameters like temperature, max_tokens, etc.
        max_retries: Maximum number of retries (unused)
        retry_delay: Delay between retries (unused)

    Returns:
        Model response as string, or None if request fails
    """
    try:
        params = llm_params or {}
        llm = OpenAILLM(api_key=api_key, model=model)
        response = llm.call(
            system_prompt,
            user_prompt,
            temperature=params.get('temperature'),
            max_tokens=params.get('max_tokens'),
            top_p=params.get('top_p')
        )
        return response.content
    except Exception as e:
        print(f"Error querying OpenAI model '{model}': {e}")
        return None


def query_gemini_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: Optional[str] = None,
    llm_params: Optional[Dict] = None,
    max_retries: int = 3,
    retry_delay: int = 5
) -> Optional[str]:
    """
    Query a Google Gemini model (backward compatible function interface).

    Args:
        system_prompt: System instruction for the model
        user_prompt: User prompt for the model
        model: Model name (e.g., 'gemini-2.5-pro')
        api_key: Google Gemini API key
        llm_params: Dictionary with parameters like temperature, max_tokens, etc.
        max_retries: Maximum number of retries (unused)
        retry_delay: Delay between retries (unused)

    Returns:
        Model response as string, or None if request fails
    """
    try:
        params = llm_params or {}
        llm = GeminiLLM(api_key=api_key, model=model)
        response = llm.call(
            system_prompt,
            user_prompt,
            temperature=params.get('temperature'),
            max_tokens=params.get('max_tokens'),
            top_p=params.get('top_p')
        )
        return response.content
    except Exception as e:
        print(f"ERROR: Exception in query_gemini_model for model '{model}'. "
              f"Type: {type(e).__name__}, Details: {e}")
        traceback.print_exc()
        return None


def query_anthropic_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: Optional[str] = None,
    llm_params: Optional[Dict] = None,
    max_retries: int = 3,
    retry_delay: int = 5
) -> Optional[str]:
    """
    Query Anthropic Claude model (backward compatible function interface).

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        model: Model name (e.g., 'claude-3-5-sonnet-20241022')
        api_key: Anthropic API key
        llm_params: Dictionary with parameters like temperature, max_tokens, etc.
        max_retries: Maximum number of retries (unused)
        retry_delay: Delay between retries (unused)

    Returns:
        Model response as string, or None if request fails
    """
    try:
        params = llm_params or {}
        llm = AnthropicLLM(api_key=api_key, model=model)
        response = llm.call(
            system_prompt,
            user_prompt,
            temperature=params.get('temperature'),
            max_tokens=params.get('max_tokens'),
            top_p=params.get('top_p')
        )
        return response.content
    except Exception as e:
        print(f"Error querying Anthropic model '{model}': {e}")
        traceback.print_exc()
        return None


def query_aws_bedrock_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_keys: Optional[Dict] = None,
    llm_params: Optional[Dict] = None,
    max_retries: int = 3,
    retry_delay: int = 5
) -> Optional[str]:
    """
    Query AWS Bedrock models (backward compatible function interface).

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        model: Model ARN or ID
        api_keys: Dictionary containing AWS credentials
        llm_params: Dictionary with parameters like temperature, max_tokens, etc.
        max_retries: Maximum number of retries
        retry_delay: Delay in seconds between retries

    Returns:
        Model response as string, or None if all retries fail
    """
    try:
        api_keys = api_keys or {}
        params = llm_params or {}

        llm = BedrockLLM(
            model=model,
            aws_access_key_id=api_keys.get('aws_access_key_id'),
            aws_secret_access_key=api_keys.get('aws_secret_access_key'),
            aws_session_token=api_keys.get('aws_session_token'),
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        response = llm.call(
            system_prompt,
            user_prompt,
            temperature=params.get('temperature'),
            max_tokens=params.get('max_tokens'),
            top_p=params.get('top_p')
        )
        return response.content
    except Exception as e:
        print(f"Failed to get response from Bedrock model '{model}': {e}")
        return None


# Helper function for backward compatibility
def _create_bedrock_client(api_keys: Optional[Dict] = None):
    """Create AWS Bedrock client with optional credentials."""
    client_args = {"region_name": AWS_REGION}

    if api_keys and api_keys.get('aws_access_key_id') and api_keys.get('aws_secret_access_key'):
        client_args['aws_access_key_id'] = api_keys['aws_access_key_id']
        client_args['aws_secret_access_key'] = api_keys['aws_secret_access_key']

    return boto3.client("bedrock-runtime", **client_args)
