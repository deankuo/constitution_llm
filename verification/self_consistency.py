"""
Self-Consistency Verification

This module implements self-consistency verification, which samples
multiple predictions at different temperatures and uses majority voting
to determine the final prediction.

The key insight is that consistent predictions across temperature variations
are more likely to be correct.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.base import BaseLLM, ModelResponse
from verification.base import BaseVerification, VerificationResult, PredictionSample
from utils.json_parser import parse_json_response, validate_indicator_response
from utils.cost_tracker import CostTracker


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency verification."""
    n_samples: int = 3
    temperatures: List[float] = None
    min_agreement: float = 0.6

    def __post_init__(self):
        if self.temperatures is None:
            self.temperatures = [0.0, 0.5, 1.0]
        # Ensure we have enough temperatures for n_samples
        if len(self.temperatures) < self.n_samples:
            # Interpolate additional temperatures
            step = 1.0 / (self.n_samples - 1) if self.n_samples > 1 else 0
            self.temperatures = [step * i for i in range(self.n_samples)]


class SelfConsistencyVerification(BaseVerification):
    """
    Self-Consistency verification using temperature sampling.

    This verification method:
    1. Samples N predictions at different temperatures
    2. Counts the frequency of each prediction
    3. Returns the majority prediction with confidence based on agreement ratio

    The agreement ratio serves as a confidence indicator - higher agreement
    suggests more reliable predictions.

    Example:
        config = SelfConsistencyConfig(n_samples=5, temperatures=[0.0, 0.3, 0.5, 0.7, 1.0])
        verifier = SelfConsistencyVerification(llm, config)
        result = verifier.verify(
            system_prompt="...",
            user_prompt="...",
            indicator="sovereign",
            valid_labels=["0", "1"]
        )
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: Optional[SelfConsistencyConfig] = None,
        cost_tracker: Optional[CostTracker] = None
    ):
        """
        Initialize self-consistency verification.

        Args:
            llm: The LLM to use for sampling
            config: Configuration for self-consistency (uses defaults if None)
            cost_tracker: Optional cost tracker for tracking API usage
        """
        super().__init__(llm, cost_tracker)
        self.config = config or SelfConsistencyConfig()

    def verify(
        self,
        system_prompt: str,
        user_prompt: str,
        indicator: str,
        valid_labels: List[str],
        initial_prediction: Optional[str] = None,
        polity: str = None,
        name: str = None,
        start_year: int = None,
        end_year: int = None,
        **kwargs
    ) -> VerificationResult:
        """
        Verify a prediction using self-consistency.

        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt for the model
            indicator: Name of the indicator being predicted
            valid_labels: List of valid label values
            initial_prediction: Optional initial prediction (will be included in sampling)
            polity: Polity name (for consistency with base class)
            name: Leader name (for consistency with base class)
            start_year: Start year (for consistency with base class)
            end_year: End year (for consistency with base class)

        Returns:
            VerificationResult with majority prediction and agreement ratio
        """
        samples = self._collect_samples(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            indicator=indicator,
            valid_labels=valid_labels
        )

        # If we have an initial prediction, include it as a sample
        if initial_prediction is not None and initial_prediction in valid_labels:
            samples.insert(0, PredictionSample(
                prediction=initial_prediction,
                reasoning="Initial prediction",
                temperature=0.0
            ))

        # Aggregate predictions
        aggregated = self._aggregate_predictions(samples, valid_labels)

        return aggregated

    def _collect_samples(
        self,
        system_prompt: str,
        user_prompt: str,
        indicator: str,
        valid_labels: List[str]
    ) -> List[PredictionSample]:
        """Collect prediction samples at different temperatures."""
        samples = []

        for i in range(self.config.n_samples):
            temperature = self.config.temperatures[i % len(self.config.temperatures)]

            try:
                response = self.llm.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature
                )

                # Track self-consistency sampling cost
                self.cost_tracker.add_usage(
                    model=self.llm.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cached_tokens=response.cached_tokens,
                    thinking_tokens=response.thinking_tokens,
                    indicator=f'self_consistency_sample_{i+1}'
                )

                # Parse the response
                parsed = parse_json_response(response.content, verbose=False)
                validated = validate_indicator_response(parsed, indicator, valid_labels)

                sample = PredictionSample(
                    prediction=validated.get(indicator, ''),
                    reasoning=validated.get('reasoning', ''),
                    confidence=validated.get('confidence_score'),
                    temperature=temperature,
                    response=response
                )
                samples.append(sample)

            except Exception as e:
                print(f"Warning: Sample {i+1} failed: {e}")
                continue

        return samples

    def _aggregate_predictions(
        self,
        samples: List[PredictionSample],
        valid_labels: List[str]
    ) -> VerificationResult:
        """Aggregate samples using majority voting."""
        if not samples:
            # No valid samples
            return VerificationResult(
                original_prediction='',
                verified_prediction='',
                confidence=0.0,
                agreement_ratio=0.0,
                verification_details={
                    'method': 'self_consistency',
                    'error': 'No valid samples collected',
                    'n_samples': 0
                },
                was_revised=False
            )

        # Count predictions
        predictions = [s.prediction for s in samples if s.prediction in valid_labels]

        if not predictions:
            # No valid predictions
            return VerificationResult(
                original_prediction=samples[0].prediction if samples else '',
                verified_prediction='',
                confidence=0.0,
                agreement_ratio=0.0,
                verification_details={
                    'method': 'self_consistency',
                    'error': 'No valid predictions in samples',
                    'n_samples': len(samples)
                },
                was_revised=False
            )

        # Majority vote
        counter = Counter(predictions)
        majority_prediction, majority_count = counter.most_common(1)[0]
        agreement_ratio = majority_count / len(predictions)

        # Determine confidence based on agreement
        confidence = agreement_ratio

        # Collect reasoning from agreeing samples
        agreeing_samples = [s for s in samples if s.prediction == majority_prediction]
        reasoning_samples = [s.reasoning for s in agreeing_samples if s.reasoning]

        # Get original prediction (first sample)
        original_prediction = samples[0].prediction

        return VerificationResult(
            original_prediction=original_prediction,
            verified_prediction=majority_prediction,
            confidence=confidence,
            agreement_ratio=agreement_ratio,
            verification_details={
                'method': 'self_consistency',
                'n_samples': len(samples),
                'n_valid': len(predictions),
                'vote_distribution': dict(counter),
                'temperatures_used': [s.temperature for s in samples],
                'high_confidence': agreement_ratio >= self.config.min_agreement,
                'sample_reasonings': reasoning_samples[:3]  # Include up to 3 reasonings
            },
            was_revised=(original_prediction != majority_prediction)
        )


def create_self_consistency_verifier(
    llm: BaseLLM,
    n_samples: int = 3,
    temperatures: Optional[List[float]] = None,
    min_agreement: float = 0.6
) -> SelfConsistencyVerification:
    """
    Factory function to create a self-consistency verifier.

    Args:
        llm: The LLM to use
        n_samples: Number of samples to collect
        temperatures: List of temperatures to use
        min_agreement: Minimum agreement ratio for high confidence

    Returns:
        Configured SelfConsistencyVerification instance
    """
    config = SelfConsistencyConfig(
        n_samples=n_samples,
        temperatures=temperatures,
        min_agreement=min_agreement
    )
    return SelfConsistencyVerification(llm, config)
