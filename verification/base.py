"""
Base classes for verification mechanisms.

This module provides:
- VerificationResult: Standard result structure from verification
- BaseVerification: Abstract base class for verification implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.base import BaseLLM, ModelResponse
from utils.cost_tracker import CostTracker


@dataclass
class VerificationResult:
    """
    Result structure from verification mechanisms.

    Attributes:
        original_prediction: The initial prediction before verification
        verified_prediction: The final prediction after verification
        confidence: Confidence score (0.0 to 1.0)
        agreement_ratio: Ratio of agreeing samples (for self-consistency)
        verification_details: Additional details about the verification process
        was_revised: Whether the prediction was changed during verification
    """
    original_prediction: str
    verified_prediction: str
    confidence: float
    agreement_ratio: Optional[float] = None
    verification_details: Dict[str, Any] = field(default_factory=dict)
    was_revised: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'original_prediction': self.original_prediction,
            'verified_prediction': self.verified_prediction,
            'confidence': self.confidence,
            'agreement_ratio': self.agreement_ratio,
            'verification_details': self.verification_details,
            'was_revised': self.was_revised
        }


@dataclass
class PredictionSample:
    """
    A single prediction sample for aggregation.

    Attributes:
        prediction: The predicted value
        reasoning: The reasoning provided
        confidence: The confidence score
        temperature: Temperature used for this sample
        response: The full model response
    """
    prediction: str
    reasoning: str
    confidence: Optional[int] = None
    temperature: float = 0.0
    response: Optional[ModelResponse] = None


class BaseVerification(ABC):
    """
    Abstract base class for verification mechanisms.

    Verification mechanisms are used to improve prediction quality through
    techniques like self-consistency (multiple samples) or chain-of-verification
    (cross-model validation).

    Example:
        verifier = SelfConsistencyVerification(llm, n_samples=3)
        result = verifier.verify(
            system_prompt="...",
            user_prompt="...",
            indicator="sovereign",
            valid_labels=["0", "1"]
        )
        print(f"Verified: {result.verified_prediction}, confidence: {result.confidence}")
    """

    def __init__(self, llm: BaseLLM, cost_tracker: Optional[CostTracker] = None):
        """
        Initialize the verification mechanism.

        Args:
            llm: The LLM to use for verification
            cost_tracker: Optional cost tracker for tracking API usage
        """
        self.llm = llm
        self.cost_tracker = cost_tracker or CostTracker()

    @abstractmethod
    def verify(
        self,
        system_prompt: str,
        user_prompt: str,
        indicator: str,
        valid_labels: List[str],
        polity: str = None,
        name: str = None,
        start_year: int = None,
        end_year: int = None,
        **kwargs
    ) -> VerificationResult:
        """
        Verify a prediction.

        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt for the model
            indicator: Name of the indicator being predicted
            valid_labels: List of valid label values
            polity: Name of the polity (for CoVe questions)
            name: Name of the leader (for CoVe questions)
            start_year: Start year (for CoVe questions)
            end_year: End year (for CoVe questions)
            **kwargs: Additional verification-specific parameters

        Returns:
            VerificationResult with verified prediction and confidence
        """
        pass

    def get_verification_name(self) -> str:
        """Get the name of this verification method."""
        return self.__class__.__name__.replace('Verification', '')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(llm={self.llm})"


class NoVerification(BaseVerification):
    """
    Passthrough verification that performs no actual verification.

    Used when verification is disabled but the interface is still needed.
    """

    def __init__(self, llm: BaseLLM, cost_tracker: Optional[CostTracker] = None):
        super().__init__(llm, cost_tracker)

    def verify(
        self,
        system_prompt: str,
        user_prompt: str,
        indicator: str,
        valid_labels: List[str],
        initial_prediction: Optional[str] = None,
        initial_reasoning: Optional[str] = None,
        initial_confidence: Optional[int] = None,
        polity: str = None,
        name: str = None,
        start_year: int = None,
        end_year: int = None,
        **kwargs
    ) -> VerificationResult:
        """
        Return the initial prediction without verification.

        Args:
            system_prompt: System prompt (unused)
            user_prompt: User prompt (unused)
            indicator: Indicator name
            valid_labels: Valid labels (unused)
            initial_prediction: The prediction to pass through
            initial_reasoning: The reasoning to pass through
            initial_confidence: The confidence to pass through
            polity: Polity name (unused)
            name: Leader name (unused)
            start_year: Start year (unused)
            end_year: End year (unused)

        Returns:
            VerificationResult with the original prediction unchanged
        """
        return VerificationResult(
            original_prediction=initial_prediction or '',
            verified_prediction=initial_prediction or '',
            confidence=initial_confidence / 100.0 if initial_confidence else 0.5,
            agreement_ratio=None,
            verification_details={
                'method': 'none',
                'reasoning': initial_reasoning
            },
            was_revised=False
        )
