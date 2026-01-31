"""
Chain of Verification (CoVe) Implementation

This module implements Chain of Verification, which uses a cross-model
approach to verify predictions through factual questions.

The key insight is that a verifier model answering factual questions
INDEPENDENTLY (without seeing the original prediction) can detect errors
through inconsistencies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.base import BaseLLM, ModelResponse
from verification.base import BaseVerification, VerificationResult
from utils.json_parser import parse_json_response
from utils.cost_tracker import CostTracker


@dataclass
class CoVeConfig:
    """Configuration for Chain of Verification."""
    questions_per_element: int = 2
    verifier_temperature: float = 0.0
    synthesizer_temperature: float = 0.0


@dataclass
class CoVeStep:
    """Record of a single CoVe step."""
    step_name: str
    model_used: str
    input_prompt: str
    output: str
    success: bool = True
    error: Optional[str] = None


class ChainOfVerification(BaseVerification):
    """
    Chain of Verification (CoVe) implementation.

    CoVe follows this process:
    1. Initial Prediction: Primary model generates prediction
    2. Question Generation: Generate verification questions based on the prediction
    3. Independent Answering: Verifier answers questions WITHOUT seeing the original prediction
    4. Synthesis: Compare answers with original prediction and decide to maintain or revise

    The critical insight is "Factored Execution" - the verifier never sees the
    original prediction, preventing confirmation bias.

    Example:
        cove = ChainOfVerification(
            primary_llm=gemini_llm,
            verifier_llm=claude_llm
        )
        result = cove.verify(
            system_prompt="...",
            user_prompt="...",
            indicator="constitution",
            valid_labels=["Yes", "No"],
            polity="Roman Republic",
            start_year=-509,
            end_year=-27
        )
    """

    def __init__(
        self,
        primary_llm: BaseLLM,
        verifier_llm: Optional[BaseLLM] = None,
        config: Optional[CoVeConfig] = None,
        cost_tracker: Optional[CostTracker] = None
    ):
        """
        Initialize Chain of Verification.

        Args:
            primary_llm: LLM for initial prediction and synthesis
            verifier_llm: LLM for verification (uses primary if None)
            config: Configuration for CoVe
            cost_tracker: Optional cost tracker for tracking API usage
        """
        super().__init__(primary_llm, cost_tracker)
        self.verifier_llm = verifier_llm or primary_llm
        self.config = config or CoVeConfig()
        self.steps: List[CoVeStep] = []

    def verify(
        self,
        system_prompt: str,
        user_prompt: str,
        indicator: str,
        valid_labels: List[str],
        polity: Optional[str] = None,
        name: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        initial_prediction: Optional[str] = None,
        initial_reasoning: Optional[str] = None,
        **kwargs
    ) -> VerificationResult:
        """
        Verify a prediction using Chain of Verification.

        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt for the model
            indicator: Name of the indicator being predicted
            valid_labels: List of valid label values
            polity: Name of the polity (for question generation)
            name: Name of the leader (for question generation)
            start_year: Start year of the leader's reign
            end_year: End year of the leader's reign
            initial_prediction: Optional initial prediction to verify
            initial_reasoning: Optional initial reasoning to verify

        Returns:
            VerificationResult with verified prediction and details
        """
        self.steps = []

        # Step 1: Get initial prediction if not provided
        if initial_prediction is None:
            initial_prediction, initial_reasoning = self._get_initial_prediction(
                system_prompt, user_prompt, indicator, valid_labels
            )

        if initial_prediction is None:
            return VerificationResult(
                original_prediction='',
                verified_prediction='',
                confidence=0.0,
                verification_details={
                    'method': 'cove',
                    'error': 'Failed to get initial prediction',
                    'steps': [s.__dict__ for s in self.steps]
                },
                was_revised=False
            )

        # Step 2: Generate verification questions
        questions = self._generate_verification_questions(
            indicator=indicator,
            polity=polity or 'the polity',
            name=name or 'the leader',
            start_year=start_year or 0,
            end_year=end_year or 0,
            initial_prediction=initial_prediction,
            initial_reasoning=initial_reasoning
        )

        # Step 3: Answer questions independently (without seeing original prediction)
        answers = self._answer_questions_independently(
            questions=questions,
            polity=polity or 'the polity',
            name=name or 'the leader',
            start_year=start_year or 0,
            end_year=end_year or 0
        )

        # Step 4: Synthesize and decide
        final_prediction, confidence, synthesis_details = self._synthesize(
            indicator=indicator,
            valid_labels=valid_labels,
            initial_prediction=initial_prediction,
            initial_reasoning=initial_reasoning,
            questions=questions,
            answers=answers
        )

        was_revised = final_prediction != initial_prediction

        return VerificationResult(
            original_prediction=initial_prediction,
            verified_prediction=final_prediction,
            confidence=confidence,
            verification_details={
                'method': 'cove',
                'questions': questions,
                'answers': answers,
                'synthesis': synthesis_details,
                'steps': [s.__dict__ for s in self.steps]
            },
            was_revised=was_revised
        )

    def _get_initial_prediction(
        self,
        system_prompt: str,
        user_prompt: str,
        indicator: str,
        valid_labels: List[str]
    ) -> tuple:
        """Get initial prediction from primary model."""
        try:
            response = self.llm.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0
            )

            self.steps.append(CoVeStep(
                step_name='initial_prediction',
                model_used=str(self.llm.model),
                input_prompt=user_prompt[:500] + '...',
                output=response.content[:500] + '...'
            ))

            parsed = parse_json_response(response.content, verbose=False)
            prediction = parsed.get(indicator) or parsed.get('constitution')
            reasoning = parsed.get('reasoning') or parsed.get('explanation', '')

            return str(prediction) if prediction else None, reasoning

        except Exception as e:
            self.steps.append(CoVeStep(
                step_name='initial_prediction',
                model_used=str(self.llm.model),
                input_prompt=user_prompt[:500] + '...',
                output='',
                success=False,
                error=str(e)
            ))
            return None, None

    def _generate_verification_questions(
        self,
        indicator: str,
        polity: str,
        name: str,
        start_year: int,
        end_year: int,
        initial_prediction: str,
        initial_reasoning: str
    ) -> List[str]:
        """Generate verification questions based on the indicator."""
        # Import CoVe questions based on indicator
        try:
            if indicator == 'constitution':
                from prompts.constitution import get_cove_questions
                questions_dict = get_cove_questions(polity, name, start_year, end_year)
                # Flatten all questions
                questions = []
                for element_questions in questions_dict.values():
                    questions.extend(element_questions[:self.config.questions_per_element])
            else:
                from prompts.indicators import get_cove_questions
                questions = get_cove_questions(indicator, polity, name, start_year, end_year)
        except Exception:
            # Fallback to generic questions
            period = f"{start_year}-{end_year}"
            questions = [
                f"What do historical sources say about {indicator} for {name} of {polity} during {period}?",
                f"What evidence exists regarding {indicator} status during {name}'s reign of {polity} ({period})?"
            ]

        return questions

    def _answer_questions_independently(
        self,
        questions: List[str],
        polity: str,
        name: str,
        start_year: int,
        end_year: int
    ) -> List[Dict[str, str]]:
        """
        Answer verification questions independently.

        CRITICAL: The verifier NEVER sees the original prediction.
        This prevents confirmation bias.
        """
        answers = []

        verifier_system = """You are a professional historian answering factual questions about historical polities.
Answer each question based ONLY on verifiable historical facts.
Be specific and cite concrete evidence where possible.
If uncertain, say so explicitly.
Keep answers concise but informative."""

        for i, question in enumerate(questions):
            try:
                response = self.verifier_llm.call(
                    system_prompt=verifier_system,
                    user_prompt=f"Question about {polity} ({start_year}-{end_year}):\n\n{question}",
                    temperature=self.config.verifier_temperature
                )

                # Track verification cost
                self.cost_tracker.add_usage(
                    model=self.verifier_llm.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cached_tokens=response.cached_tokens,
                    polity=polity,
                    indicator='verification_question'
                )

                self.steps.append(CoVeStep(
                    step_name=f'verify_question_{i+1}',
                    model_used=str(self.verifier_llm.model),
                    input_prompt=question,
                    output=response.content[:500] + '...' if len(response.content) > 500 else response.content
                ))

                answers.append({
                    'question': question,
                    'answer': response.content
                })

            except Exception as e:
                self.steps.append(CoVeStep(
                    step_name=f'verify_question_{i+1}',
                    model_used=str(self.verifier_llm.model),
                    input_prompt=question,
                    output='',
                    success=False,
                    error=str(e)
                ))
                answers.append({
                    'question': question,
                    'answer': f'Error: {str(e)}'
                })

        return answers

    def _synthesize(
        self,
        indicator: str,
        valid_labels: List[str],
        initial_prediction: str,
        initial_reasoning: str,
        questions: List[str],
        answers: List[Dict[str, str]]
    ) -> tuple:
        """Synthesize verification answers and decide on final prediction."""
        synthesis_prompt = f"""You are a synthesis expert evaluating evidence for a political indicator prediction.

## Original Prediction
Indicator: {indicator}
Prediction: {initial_prediction}
Reasoning: {initial_reasoning}

## Verification Evidence
"""
        for qa in answers:
            synthesis_prompt += f"\nQ: {qa['question']}\nA: {qa['answer']}\n"

        synthesis_prompt += f"""
## Your Task
Based on the verification evidence above, determine if the original prediction is:
1. SUPPORTED - The evidence confirms the original prediction
2. CONTRADICTED - The evidence contradicts the original prediction
3. INSUFFICIENT - The evidence is not conclusive

If CONTRADICTED, provide the correct prediction.

Valid labels for {indicator}: {valid_labels}

Respond with a JSON object:
{{"decision": "SUPPORTED or CONTRADICTED or INSUFFICIENT", "final_prediction": "the correct prediction", "confidence": 0.0-1.0, "reasoning": "your synthesis"}}
"""

        try:
            response = self.llm.call(
                system_prompt="You are a careful analyst synthesizing evidence.",
                user_prompt=synthesis_prompt,
                temperature=self.config.synthesizer_temperature
            )

            # Track synthesis cost
            self.cost_tracker.add_usage(
                model=self.llm.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cached_tokens=response.cached_tokens,
                indicator='verification_synthesis'
            )

            self.steps.append(CoVeStep(
                step_name='synthesis',
                model_used=str(self.llm.model),
                input_prompt=synthesis_prompt[:500] + '...',
                output=response.content
            ))

            parsed = parse_json_response(response.content, verbose=False)

            decision = parsed.get('decision', 'SUPPORTED')
            final_prediction = parsed.get('final_prediction', initial_prediction)
            confidence = parsed.get('confidence', 0.5)
            reasoning = parsed.get('reasoning', '')

            # Validate final prediction
            if final_prediction not in valid_labels:
                final_prediction = initial_prediction
                confidence = max(0.3, confidence - 0.2)

            return final_prediction, confidence, {
                'decision': decision,
                'reasoning': reasoning,
                'original_prediction': initial_prediction
            }

        except Exception as e:
            self.steps.append(CoVeStep(
                step_name='synthesis',
                model_used=str(self.llm.model),
                input_prompt=synthesis_prompt[:500] + '...',
                output='',
                success=False,
                error=str(e)
            ))
            # Return original prediction on error
            return initial_prediction, 0.3, {
                'decision': 'ERROR',
                'error': str(e),
                'original_prediction': initial_prediction
            }


def create_cove_verifier(
    primary_llm: BaseLLM,
    verifier_llm: Optional[BaseLLM] = None,
    questions_per_element: int = 2
) -> ChainOfVerification:
    """
    Factory function to create a Chain of Verification verifier.

    Args:
        primary_llm: LLM for initial prediction and synthesis
        verifier_llm: LLM for verification (uses primary if None)
        questions_per_element: Number of questions per verification element

    Returns:
        Configured ChainOfVerification instance
    """
    config = CoVeConfig(questions_per_element=questions_per_element)
    return ChainOfVerification(primary_llm, verifier_llm, config)
