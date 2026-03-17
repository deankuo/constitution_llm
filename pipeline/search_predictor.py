"""
Search-Augmented Prediction Pipeline

This module provides SearchPredictor, which reuses the existing prompt builders
(single/multiple/sequential) but routes LLM calls through web-search agents
instead of plain LLM calls. Each prediction records the queries issued and the
URLs returned by the search API.

Output columns (row-level, not per-indicator):
  - search_queries:   pipe-delimited list of queries (with [Source] prefix)
  - urls_used:        pipe-delimited list of URLs (with [Source] prefix)
  - web_information:  actual retrieved text (single/sequential mode only; JSON output only)
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import INDICATOR_LABELS, PromptMode
from models.llm_clients import detect_provider
from models.search_agents import (
    run_openai_search_agent,
    run_gemini_search_agent,
    run_bedrock_search_agent,
    run_anthropic_search_agent,
)
from prompts.base_builder import BasePromptBuilder
from prompts.single_builder import SinglePromptBuilder
from prompts.multiple_builder import MultiplePromptBuilder
from prompts.sequential_builder import SequentialPromptBuilder
from utils.json_parser import (
    parse_json_response,
    validate_indicator_response,
    validate_constitution_response,
)
from utils.langsmith_utils import traceable


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SearchIndicatorResult:
    """Prediction result for a single indicator produced by the search agent."""
    indicator: str
    prediction: Optional[str]
    reasoning: str
    confidence_score: Optional[int]
    model_used: str = ''
    # Constitution-specific fields
    document_name: Optional[str] = None
    constitution_year: Optional[str] = None


@dataclass
class SearchPolityPrediction:
    """Complete search-augmented prediction result for a leader row."""
    polity: str
    name: str
    start_year: int
    end_year: Optional[int]
    predictions: Dict[str, SearchIndicatorResult] = field(default_factory=dict)
    # Row-level search metadata
    search_queries: List[str] = field(default_factory=list)
    urls_used: List[str] = field(default_factory=list)
    web_information: Optional[str] = None  # Only populated for single/sequential mode
    mode: str = 'multiple'

    def to_dict(self) -> Dict:
        """
        Flatten to a dict suitable for a DataFrame row.

        Columns added per indicator:
          {ind}_prediction, {ind}_reasoning, {ind}_confidence
        Constitution also adds constitution_document_name, constitution_year.

        Row-level search columns:
          search_queries, urls_used, web_information (single/sequential only)
        """
        result: Dict = {}
        for ind_name, ind_res in self.predictions.items():
            result[f'{ind_name}_prediction'] = ind_res.prediction
            result[f'{ind_name}_reasoning'] = ind_res.reasoning
            result[f'{ind_name}_confidence'] = ind_res.confidence_score

            if ind_name == 'constitution':
                result['constitution_document_name'] = ind_res.document_name
                result['constitution_year'] = ind_res.constitution_year

        # Row-level search metadata
        if self.search_queries:
            result['search_queries'] = ' | '.join(self.search_queries)
        if self.urls_used:
            result['urls_used'] = ' | '.join(self.urls_used)

        # web_information only for single/sequential modes (too large for CSV,
        # but included here for JSON output)
        if self.mode in ('single', 'sequential') and self.web_information:
            result['web_information'] = self.web_information

        return result


# =============================================================================
# Prompt Builder Factory
# =============================================================================

def _create_prompt_builder(
    mode: str,
    indicators: List[str],
    reasoning: bool = True,
    sequence: Optional[List[str]] = None,
    random_sequence: bool = False,
) -> BasePromptBuilder:
    """Create the appropriate prompt builder for the given mode."""
    if mode == 'single':
        return SinglePromptBuilder(indicators=indicators, reasoning=reasoning)
    elif mode == 'sequential':
        return SequentialPromptBuilder(
            indicators=indicators,
            sequence=sequence,
            random_order=random_sequence,
            reasoning=reasoning,
        )
    else:  # 'multiple' (default)
        return MultiplePromptBuilder(indicators=indicators, reasoning=reasoning)


# =============================================================================
# Search Predictor
# =============================================================================

class SearchPredictor:
    """
    Prediction orchestrator that routes LLM calls through web-search agents.

    Reuses the same prompt builders as the main Predictor but calls
    run_*_search_agent instead of llm.call(). Each indicator result records
    the queries issued and URLs retrieved during the search.

    Example::

        predictor = SearchPredictor(
            model='gemini-2.5-pro',
            api_keys={'gemini': '...', 'serper': '...'},
            mode='multiple',
            indicators=['sovereign', 'assembly'],
        )
        result = predictor.predict('Roman Republic', 'Julius Caesar', -49, -44)
        print(result.predictions['sovereign'].prediction)
        print(result.predictions['sovereign'].urls_used)
    """

    def __init__(
        self,
        model: str,
        api_keys: Dict[str, str],
        mode: str = 'multiple',
        indicators: Optional[List[str]] = None,
        reasoning: bool = True,
        sequence: Optional[List[str]] = None,
        random_sequence: bool = False,
        force_search: bool = False,
    ):
        """
        Args:
            model:           LLM model identifier (e.g. 'gemini-2.5-pro')
            api_keys:        Dict with keys: 'openai', 'gemini', 'anthropic',
                             'aws_access_key_id', 'aws_secret_access_key',
                             'aws_session_token', and 'serper'
            mode:            'single' | 'multiple' | 'sequential'
            indicators:      Which indicators to predict (default: all 7 non-constitution)
            reasoning:       Include reasoning column in output
            sequence:        Explicit order for sequential mode
            random_sequence: Randomize order in sequential mode
            force_search:    If True, force the LLM to use web search (tool_choice=required)
        """
        self.model = model
        self.api_keys = api_keys
        self.serper_api_key = api_keys.get('serper', '')
        self.provider = detect_provider(model)
        self.reasoning = reasoning
        self.force_search = force_search
        self.mode = mode

        default_indicators = [
            'sovereign', 'assembly', 'appointment', 'tenure',
            'exit', 'collegiality', 'separate_powers',
        ]
        self.indicators = indicators or default_indicators

        self.prompt_builder = _create_prompt_builder(
            mode=mode,
            indicators=self.indicators,
            reasoning=reasoning,
            sequence=sequence,
            random_sequence=random_sequence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @traceable(name="SearchPredictor.predict")
    def predict(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int],
    ) -> SearchPolityPrediction:
        """
        Predict political indicators for a leader using search agents.

        Args:
            polity:     Name of the polity
            name:       Leader name
            start_year: Start year of reign
            end_year:   End year of reign (None if unknown)

        Returns:
            SearchPolityPrediction with per-indicator results including URLs
        """
        prompts = self.prompt_builder.build(polity, name, start_year, end_year)
        predictions: Dict[str, SearchIndicatorResult] = {}

        # Row-level trackers (aggregated across all prompts)
        all_queries: List[str] = []
        all_urls: List[str] = []
        all_content: List[str] = []

        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n🔍 Search prompt {prompt_idx+1}/{len(prompts)}: {prompt.indicators}")
            url_tracker: List[str] = []
            query_tracker: List[str] = []
            content_tracker: List[str] = []

            try:
                response_text = self._call_search_agent(
                    prompt.system_prompt,
                    prompt.user_prompt,
                    url_tracker,
                    query_tracker,
                    content_tracker,
                )

                parsed = parse_json_response(response_text or '', verbose=False)

                for indicator in prompt.indicators:
                    ind_result = self._parse_indicator(
                        indicator=indicator,
                        parsed=parsed,
                    )
                    predictions[indicator] = ind_result

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(f"\n❌ ERROR processing {prompt.indicators}: {error_msg}")
                for indicator in prompt.indicators:
                    predictions[indicator] = SearchIndicatorResult(
                        indicator=indicator,
                        prediction=None,
                        reasoning=error_msg,
                        confidence_score=None,
                        model_used=self.model,
                    )

            # Aggregate into row-level trackers
            all_queries.extend(query_tracker)
            all_urls.extend(url_tracker)
            all_content.extend(content_tracker)

        # Build web_information only for single/sequential mode
        web_info = None
        if self.mode in ('single', 'sequential') and all_content:
            web_info = '\n---\n'.join(all_content)

        return SearchPolityPrediction(
            polity=polity,
            name=name,
            start_year=start_year,
            end_year=end_year,
            predictions=predictions,
            search_queries=all_queries,
            urls_used=all_urls,
            web_information=web_info,
            mode=self.mode,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_search_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        url_tracker: List[str],
        query_tracker: List[str],
        content_tracker: List[str],
    ) -> Optional[str]:
        """Route the call to the correct provider's search agent."""
        kwargs = dict(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            serper_api_key=self.serper_api_key,
            url_tracker=url_tracker,
            query_tracker=query_tracker,
            content_tracker=content_tracker,
            force_search=self.force_search,
        )

        if self.provider == 'openai':
            return run_openai_search_agent(
                model=self.model,
                api_key=self.api_keys.get('openai', ''),
                **kwargs,
            )
        elif self.provider == 'gemini':
            return run_gemini_search_agent(
                model=self.model,
                api_key=self.api_keys.get('gemini', ''),
                **kwargs,
            )
        elif self.provider == 'bedrock':
            return run_bedrock_search_agent(
                model=self.model,
                api_keys=self.api_keys,
                **kwargs,
            )
        elif self.provider == 'anthropic':
            return run_anthropic_search_agent(
                model=self.model,
                api_key=self.api_keys.get('anthropic', ''),
                **kwargs,
            )
        else:
            return run_openai_search_agent(
                model=self.model,
                api_key=self.api_keys.get('openai', ''),
                **kwargs,
            )

    def _parse_indicator(
        self,
        indicator: str,
        parsed: Dict,
    ) -> SearchIndicatorResult:
        """Extract and validate a single indicator's fields from parsed JSON."""
        document_name = None
        constitution_year = None

        if indicator == 'constitution':
            validated = validate_constitution_response(parsed)
            prediction = validated.get('constitution')
            reasoning = validated.get('reasoning', '')
            confidence = validated.get('confidence_score')
            document_name = validated.get('document_name')
            constitution_year = validated.get('constitution_year')
        else:
            valid_labels = INDICATOR_LABELS.get(indicator, ['0', '1'])
            validated = validate_indicator_response(parsed, indicator, valid_labels)
            prediction = validated.get(indicator)
            reasoning = validated.get('reasoning', '')
            confidence = validated.get('confidence_score')

        return SearchIndicatorResult(
            indicator=indicator,
            prediction=prediction,
            reasoning=reasoning,
            confidence_score=confidence,
            model_used=self.model,
            document_name=document_name,
            constitution_year=constitution_year,
        )
