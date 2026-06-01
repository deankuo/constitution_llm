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
            self.temperatures = [1.0, 1.0, 1.0]  # Default to 3 samples at temperature 1.0
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
        # Normalize valid_labels to strings so comparisons work regardless of
        # whether predictions come back as float (validate_indicator_response)
        # or as int/str (constitution uses int labels, others use string labels).
        str_valid = [str(v) for v in valid_labels]

        samples = self._collect_samples(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            indicator=indicator,
            valid_labels=str_valid
        )

        # If we have an initial prediction, normalize and include it as a sample.
        # validate_indicator_response returns floats (e.g. 1.0); convert to "1".
        if initial_prediction is not None:
            if isinstance(initial_prediction, (int, float)):
                try:
                    init_str = str(int(float(initial_prediction)))
                except (ValueError, TypeError):
                    init_str = str(initial_prediction)
            else:
                init_str = str(initial_prediction)

            if init_str in str_valid:
                samples.insert(0, PredictionSample(
                    prediction=init_str,
                    reasoning="Initial prediction",
                    temperature=0.0
                ))

        # Aggregate predictions
        aggregated = self._aggregate_predictions(samples, str_valid)

        return aggregated

    def _collect_samples(
        self,
        system_prompt: str,
        user_prompt: str,
        indicator: str,
        valid_labels: List[str]
    ) -> List[PredictionSample]:
        """Collect prediction samples at different temperatures.

        valid_labels must already be normalized to strings by the caller.
        """
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

                # validate_indicator_response returns float predictions; normalize to string
                raw_pred = validated.get(indicator)
                if raw_pred is not None and isinstance(raw_pred, (int, float)):
                    try:
                        pred_str = str(int(float(raw_pred)))
                    except (ValueError, TypeError):
                        pred_str = str(raw_pred)
                elif raw_pred is not None:
                    pred_str = str(raw_pred)
                else:
                    pred_str = ''

                sample = PredictionSample(
                    prediction=pred_str,
                    reasoning=validated.get('reasoning', ''),
                    confidence=validated.get('confidence_score'),
                    temperature=temperature,
                    response=response
                )
                samples.append(sample)

            except Exception as e:
                from tqdm import tqdm as _tqdm
                _tqdm.write(f"WARN: SC sample {i+1} failed: {e}")
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
        n_total = len(predictions)

        # Get original prediction (first sample = initial main-call prediction)
        original_prediction = samples[0].prediction

        # Tie-breaking: when no two samples agree (all predictions differ),
        # fall back to the original prediction rather than an arbitrary winner.
        if majority_count == 1:
            majority_prediction = original_prediction

        # Uncertainty label — generalises beyond the n=3 case:
        #   none  = unanimous (all agree)
        #   low   = majority ≥ 2 agrees but not all
        #   high  = no two samples agree (every prediction is unique)
        if majority_count == n_total:
            sc_uncertainty = 'none'
        elif majority_count >= 2:
            sc_uncertainty = 'low'
        else:
            sc_uncertainty = 'high'

        confidence = agreement_ratio

        # Collect reasoning from agreeing samples
        agreeing_samples = [s for s in samples if s.prediction == majority_prediction]
        reasoning_samples = [s.reasoning for s in agreeing_samples if s.reasoning]

        return VerificationResult(
            original_prediction=original_prediction,
            verified_prediction=majority_prediction,
            confidence=confidence,
            agreement_ratio=agreement_ratio,
            verification_details={
                'method': 'self_consistency',
                'n_samples': n_total,
                'n_valid': n_total,
                'vote_distribution': dict(counter),
                'agreement_ratio': round(agreement_ratio, 3),
                'sc_uncertainty': sc_uncertainty,
                'temperatures_used': [s.temperature for s in samples],
                'high_confidence': agreement_ratio >= self.config.min_agreement,
                'sample_reasonings': reasoning_samples[:3]
            },
            was_revised=(original_prediction != majority_prediction)
        )


    def aggregate_from_predictions(
        self,
        indicator: str,
        valid_labels: List[str],
        initial_prediction,
        additional_parsed_responses: List[Dict],
    ) -> VerificationResult:
        """Aggregate pre-collected parsed responses without making new LLM calls.

        Used in single/sequential mode where one prompt covers multiple indicators —
        SC samples are collected once at the prompt level and shared across indicators.
        additional_parsed_responses may be empty (all prompt-level calls failed); in
        that case the initial prediction is the only vote and uncertainty = 'high'.
        """
        str_valid = [str(v) for v in valid_labels]
        samples: List[PredictionSample] = []

        # Normalize and add initial prediction as the first sample
        if initial_prediction is not None:
            if isinstance(initial_prediction, (int, float)):
                try:
                    init_str = str(int(float(initial_prediction)))
                except (ValueError, TypeError):
                    init_str = str(initial_prediction)
            else:
                init_str = str(initial_prediction)
            if init_str in str_valid:
                samples.append(PredictionSample(
                    prediction=init_str,
                    reasoning="Initial prediction",
                    temperature=0.0
                ))

        # Add samples from pre-collected parsed responses (no LLM calls here)
        for i, parsed in enumerate(additional_parsed_responses):
            validated = validate_indicator_response(parsed, indicator, str_valid)
            raw_pred = validated.get(indicator)
            if raw_pred is not None and isinstance(raw_pred, (int, float)):
                try:
                    pred_str = str(int(float(raw_pred)))
                except (ValueError, TypeError):
                    pred_str = str(raw_pred)
            elif raw_pred is not None:
                pred_str = str(raw_pred)
            else:
                pred_str = ''

            temp = self.config.temperatures[i % len(self.config.temperatures)]
            samples.append(PredictionSample(
                prediction=pred_str,
                reasoning=validated.get('reasoning', ''),
                confidence=validated.get('confidence_score'),
                temperature=temp,
            ))

        return self._aggregate_predictions(samples, str_valid)


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
