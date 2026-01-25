"""
Base classes and data structures for LLM models.

This module provides:
- ModelResponse: Standard response structure from LLM calls
- BaseLLM: Abstract base class for LLM implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelResponse:
    """
    Standard response structure from LLM calls.

    Attributes:
        content: The text content of the response
        model: Model identifier that generated the response
        usage: Token usage and cost information
        raw_response: Optional raw response object from the provider
    """
    content: str
    model: str
    usage: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[Any] = None

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get('input_tokens', 0)

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get('output_tokens', 0)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Get cost in USD."""
        return self.usage.get('cost_usd', 0.0)


class BaseLLM(ABC):
    """
    Abstract base class for LLM implementations.

    All LLM provider classes should inherit from this class and implement
    the call() method.

    Example:
        class GeminiLLM(BaseLLM):
            def call(self, system_prompt, user_prompt, **kwargs):
                # Implementation
                return ModelResponse(...)

        llm = GeminiLLM(api_key="...")
        response = llm.call("You are a historian.", "Analyze Rome.")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048
    ):
        """
        Initialize the LLM.

        Args:
            api_key: API key for the provider
            model: Default model to use
            default_temperature: Default temperature for generation
            default_max_tokens: Default max tokens for generation
        """
        self.api_key = api_key
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Make a call to the LLM.

        Args:
            system_prompt: System prompt/instruction
            user_prompt: User message/query
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse containing the response and metadata
        """
        pass

    def get_provider_name(self) -> str:
        """Get the provider name for this LLM."""
        return self.__class__.__name__.replace('LLM', '').lower()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class APIKeyError(LLMError):
    """Raised when API key is missing or invalid."""
    pass


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    pass


class ModelError(LLMError):
    """Raised when there's an error with the model."""
    pass
