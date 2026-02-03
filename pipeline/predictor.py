"""
Core Prediction Orchestrator

This module provides the Predictor class that orchestrates the entire
prediction pipeline: prompt building -> model calling -> verification.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import (
    PromptMode,
    VerificationType,
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_VERIFIER_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    INDICATOR_LABELS
)
from models.base import BaseLLM, ModelResponse
from models.llm_clients import create_llm
from prompts.base_builder import BasePromptBuilder, PromptOutput
from prompts.single_builder import SinglePromptBuilder
from prompts.multiple_builder import MultiplePromptBuilder
from prompts.sequential_builder import SequentialPromptBuilder
from verification.base import BaseVerification, VerificationResult, NoVerification
from verification.self_consistency import SelfConsistencyVerification, SelfConsistencyConfig
from verification.cove import ChainOfVerification, CoVeConfig
from utils.json_parser import (
    parse_json_response,
    validate_indicator_response,
    validate_constitution_response
)
from utils.cost_tracker import CostTracker


@dataclass
class PredictionConfig:
    """Configuration for prediction pipeline."""
    mode: PromptMode = PromptMode.MULTIPLE
    indicators: List[str] = field(default_factory=lambda: ['sovereign', 'powersharing', 'assembly', 'appointment', 'tenure', 'exit'])
    verify: VerificationType = VerificationType.NONE
    verify_indicators: List[str] = field(default_factory=list)
    model: str = DEFAULT_PRIMARY_MODEL
    verifier_model: Optional[str] = DEFAULT_VERIFIER_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    top_p: float = DEFAULT_TOP_P
    sc_n_samples: int = 3
    sc_temperatures: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    cove_questions_per_element: int = 1  # Changed from 2 to 1 (4 questions total for constitution)
    # Sequential mode parameters
    sequence: Optional[List[str]] = None  # Specific order for sequential mode
    random_sequence: bool = False  # Randomize order in sequential mode


@dataclass
class IndicatorPrediction:
    """Prediction result for a single indicator."""
    indicator: str
    prediction: Optional[str]
    reasoning: str
    confidence_score: Optional[int]
    verified_prediction: Optional[str] = None
    verification_details: Optional[Dict] = None
    was_verified: bool = False
    model_used: str = ''
    tokens_used: int = 0
    cost_usd: float = 0.0
    # Constitution-specific fields
    document_name: Optional[str] = None
    constitution_year: Optional[str] = None  # String to handle "N/A" or "1789; 1791"


@dataclass
class PolityPrediction:
    """Complete prediction result for a polity."""
    polity: str
    start_year: int
    end_year: int
    predictions: Dict[str, IndicatorPrediction]
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    verification_applied: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame row."""
        # Don't include 'polity' since it's already in the input data as 'territorynamehistorical'
        result = {
            'total_cost_usd': self.total_cost_usd,
            'total_tokens': self.total_tokens
        }

        for ind_name, ind_pred in self.predictions.items():
            result[f'{ind_name}_prediction'] = ind_pred.prediction
            result[f'{ind_name}_reasoning'] = ind_pred.reasoning
            result[f'{ind_name}_confidence'] = ind_pred.confidence_score

            # Constitution-specific fields
            if ind_name == 'constitution':
                result['constitution_document_name'] = ind_pred.document_name
                result['constitution_year'] = ind_pred.constitution_year

            if ind_pred.was_verified:
                result[f'{ind_name}_verified'] = ind_pred.verified_prediction
                result[f'{ind_name}_verification'] = str(ind_pred.verification_details)

        return result


class Predictor:
    """
    Core prediction orchestrator.

    This class manages the entire prediction pipeline:
    1. Build prompts using the appropriate builder
    2. Call LLM to get predictions
    3. Apply verification if configured
    4. Track costs and usage

    Example:
        config = PredictionConfig(
            mode=PromptMode.MULTIPLE,
            indicators=['sovereign', 'assembly'],
            verify=VerificationType.SELF_CONSISTENCY,
            verify_indicators=['assembly']
        )

        api_keys = {'gemini': os.getenv('GEMINI_API_KEY')}
        predictor = Predictor(config, api_keys)

        result = predictor.predict("Roman Republic", -509, -27)
        print(result.predictions['sovereign'].prediction)
    """

    def __init__(
        self,
        config: PredictionConfig,
        api_keys: Dict[str, str],
        cost_tracker: Optional[CostTracker] = None
    ):
        """
        Initialize the predictor.

        Args:
            config: Prediction configuration
            api_keys: Dictionary of API keys for different providers
            cost_tracker: Optional cost tracker instance
        """
        self.config = config
        self.api_keys = api_keys
        self.cost_tracker = cost_tracker or CostTracker()

        # Initialize LLM
        self.llm = create_llm(config.model, api_keys)

        # Initialize verifier LLM if needed
        self.verifier_llm = None
        if config.verify in [VerificationType.COVE, VerificationType.BOTH]:
            verifier_model = config.verifier_model or DEFAULT_VERIFIER_MODEL
            self.verifier_llm = create_llm(verifier_model, api_keys)

        # Initialize prompt builder
        self.prompt_builder = self._create_prompt_builder()

        # Initialize verifiers
        self.verifiers = self._create_verifiers()

    def _create_prompt_builder(self) -> BasePromptBuilder:
        """Create appropriate prompt builder based on config."""
        if self.config.mode == PromptMode.SINGLE:
            return SinglePromptBuilder(indicators=self.config.indicators)
        elif self.config.mode == PromptMode.SEQUENTIAL:
            return SequentialPromptBuilder(
                indicators=self.config.indicators,
                sequence=self.config.sequence,
                random_order=self.config.random_sequence
            )
        else:
            return MultiplePromptBuilder(indicators=self.config.indicators)

    def _create_verifiers(self) -> Dict[str, BaseVerification]:
        """Create verifier instances for configured indicators."""
        verifiers = {}

        for indicator in self.config.verify_indicators:
            if self.config.verify == VerificationType.SELF_CONSISTENCY:
                verifiers[indicator] = SelfConsistencyVerification(
                    self.llm,
                    SelfConsistencyConfig(
                        n_samples=self.config.sc_n_samples,
                        temperatures=self.config.sc_temperatures
                    ),
                    cost_tracker=self.cost_tracker
                )
            elif self.config.verify == VerificationType.COVE:
                verifiers[indicator] = ChainOfVerification(
                    self.llm,
                    self.verifier_llm,
                    CoVeConfig(
                        questions_per_element=self.config.cove_questions_per_element
                    ),
                    cost_tracker=self.cost_tracker
                )
            elif self.config.verify == VerificationType.BOTH:
                # For "both", chain self-consistency THEN CoVe
                # Note: Currently only CoVe is applied (sequential verification not yet implemented)
                # TODO: Implement sequential verification pipeline
                verifiers[indicator] = ChainOfVerification(
                    self.llm,
                    self.verifier_llm,
                    CoVeConfig(
                        questions_per_element=self.config.cove_questions_per_element
                    ),
                    cost_tracker=self.cost_tracker
                )

        return verifiers

    def predict(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int]
    ) -> PolityPrediction:
        """
        Predict political indicators for a leader.

        Args:
            polity: Name of the polity
            name: Name of the leader
            start_year: Start year of the leader's reign
            end_year: End year of the leader's reign (None if unknown/unavailable in data)

        Returns:
            PolityPrediction with all indicator predictions
        """
        # Build prompts
        prompts = self.prompt_builder.build(polity, name, start_year, end_year)

        # Track results
        predictions = {}
        total_cost = 0.0
        total_tokens = 0
        verification_applied = {}

        # Process each prompt
        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n📝 Processing prompt {prompt_idx+1}/{len(prompts)}: {prompt.indicators}")
            try:
                # Call LLM
                response = self.llm.call(
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.user_prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p
                )

                # Track this API call cost (only once per prompt)
                self.cost_tracker.add_usage(
                    model=self.config.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cached_tokens=response.cached_tokens,
                    polity=polity
                )

                # Parse response
                parsed = parse_json_response(response.content, verbose=False)

                # Calculate cost per indicator if multiple indicators in one prompt
                num_indicators = len(prompt.indicators)
                cost_per_indicator = self._calculate_cost(response) / num_indicators
                tokens_per_indicator = response.total_tokens // num_indicators

                # Process each indicator in the prompt
                for indicator in prompt.indicators:
                    ind_prediction = self._process_indicator(
                        indicator=indicator,
                        parsed=parsed,
                        response=response,
                        polity=polity,
                        start_year=start_year,
                        end_year=end_year,
                        prompt=prompt,
                        cost_per_indicator=cost_per_indicator,
                        tokens_per_indicator=tokens_per_indicator
                    )

                    predictions[indicator] = ind_prediction

                    if ind_prediction.was_verified:
                        verification_applied[indicator] = self.verifiers[indicator].get_verification_name()

                    total_cost += ind_prediction.cost_usd
                    total_tokens += ind_prediction.tokens_used

            except Exception as e:
                # Handle errors gracefully - print error for debugging
                error_msg = f"Error: {str(e)}"
                print(f"\n❌ ERROR processing {prompt.indicators}: {error_msg}")
                print(f"   Prompt metadata: {prompt.metadata}")

                for indicator in prompt.indicators:
                    predictions[indicator] = IndicatorPrediction(
                        indicator=indicator,
                        prediction=None,
                        reasoning=error_msg,
                        confidence_score=None,
                        model_used=self.config.model
                    )

        return PolityPrediction(
            polity=polity,
            start_year=start_year,
            end_year=end_year,
            predictions=predictions,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            verification_applied=verification_applied
        )

    def _process_indicator(
        self,
        indicator: str,
        parsed: Dict,
        response: ModelResponse,
        polity: str,
        start_year: int,
        end_year: int,
        prompt: PromptOutput,
        cost_per_indicator: float = 0.0,
        tokens_per_indicator: int = 0
    ) -> IndicatorPrediction:
        """Process a single indicator from parsed response."""
        # Validate response based on indicator type
        document_name = None
        constitution_year = None

        if indicator == 'constitution':
            validated = validate_constitution_response(parsed)
            prediction = validated.get('constitution')
            reasoning = validated.get('reasoning', '')
            confidence = validated.get('confidence_score')
            document_name = validated.get('document_name')
            constitution_year = validated.get('constitution_year')
            valid_labels = [1, 0]
        else:
            valid_labels = INDICATOR_LABELS.get(indicator, ['0', '1'])
            validated = validate_indicator_response(parsed, indicator, valid_labels)
            prediction = validated.get(indicator)
            reasoning = validated.get('reasoning', '')
            confidence = validated.get('confidence_score')

        # Apply verification if configured for this indicator
        verified_prediction = None
        verification_details = None
        was_verified = False
        verification_cost = 0.0
        verification_tokens = 0

        if indicator in self.verifiers:
            verifier = self.verifiers[indicator]
            try:
                verify_result = verifier.verify(
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.user_prompt,
                    indicator=indicator,
                    valid_labels=valid_labels,
                    initial_prediction=prediction,
                    initial_reasoning=reasoning,
                    polity=polity,
                    name=name,
                    start_year=start_year,
                    end_year=end_year
                )

                verified_prediction = verify_result.verified_prediction
                verification_details = verify_result.verification_details
                was_verified = True

                # Track verification cost separately (verification methods handle their own tracking)
                # We don't add it here to avoid double-counting

            except Exception as e:
                verification_details = {'error': str(e)}

        return IndicatorPrediction(
            indicator=indicator,
            prediction=prediction,
            reasoning=reasoning,
            confidence_score=confidence,
            verified_prediction=verified_prediction,
            verification_details=verification_details,
            was_verified=was_verified,
            model_used=self.config.model,
            tokens_used=tokens_per_indicator,
            cost_usd=cost_per_indicator,
            document_name=document_name,
            constitution_year=constitution_year
        )

    def _calculate_cost(self, response: ModelResponse) -> float:
        """Calculate cost from model response."""
        return self.cost_tracker.calculate_cost(
            model=self.config.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens
        )

    def get_cost_summary(self) -> Dict:
        """Get cost tracking summary."""
        return self.cost_tracker.get_summary()


def create_predictor(
    mode: str = 'multiple',
    indicators: Optional[List[str]] = None,
    model: str = DEFAULT_PRIMARY_MODEL,
    verify: str = 'none',
    verify_indicators: Optional[List[str]] = None,
    api_keys: Optional[Dict[str, str]] = None,
    **kwargs
) -> Predictor:
    """
    Factory function to create a Predictor.

    Args:
        mode: 'single' or 'multiple'
        indicators: List of indicators to predict
        model: Model to use for prediction
        verify: 'none', 'self_consistency', 'cove', or 'both'
        verify_indicators: Which indicators to verify
        api_keys: API keys dictionary
        **kwargs: Additional config options

    Returns:
        Configured Predictor instance
    """
    config = PredictionConfig(
        mode=PromptMode(mode),
        indicators=indicators or ['sovereign', 'powersharing', 'assembly', 'appointment', 'tenure', 'exit'],
        verify=VerificationType(verify),
        verify_indicators=verify_indicators or [],
        model=model,
        **kwargs
    )

    return Predictor(config, api_keys or {})
