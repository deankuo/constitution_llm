"""
Constitution Analysis Pipeline - Polity Level

This script analyzes historical polities to determine political indicators
including constitution, sovereign, federalism, checks, collegiality,
petition, assembly, entry, exit, symbolism, and (downstream) elections.

Supports multiple LLM providers:
- OpenAI (GPT models)
- Google Gemini
- AWS Bedrock
- Anthropic (Claude models)

The script can optionally use web search to enhance the model's knowledge,
and supports verification methods (Self-Consistency, Chain of Verification).
"""

import os
os.environ['GRPC_VERBOSITY'] = 'NONE'

import argparse
import concurrent.futures
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Import from new module structure
from config import (
    COL_TERRITORY_NAME,
    COL_LEADER_NAME,
    COL_START_YEAR,
    COL_END_YEAR,
    REQUIRED_COLUMNS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DELAY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    ANTHROPIC_MODELS,
    BEDROCK_ARN_PREFIX,
    GEMINI_MODELS,
    OPENAI_MODELS,
    PromptMode,
    VerificationType,
    SearchMode,
    ALL_INDICATORS,
    INDICATORS_WITH_GROUND_TRUTH
)

# Backward-compatible imports from models
from models.llm_clients import (
    query_openai_model,
    query_gemini_model,
    query_anthropic_model,
    query_aws_bedrock_model,
    detect_provider,
    create_llm
)
from models.search_agents import (
    run_openai_search_agent,
    run_gemini_search_agent,
    run_bedrock_search_agent,
    run_anthropic_search_agent
)

# Prompts - backward compatible
from prompts.polity_constitution import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from prompts.polity_indicators import get_prompt as get_indicator_prompt

# New pipeline imports
from pipeline.predictor import Predictor, PredictionConfig, create_predictor
from pipeline.batch_runner import BatchRunner, BatchConfig, load_polity_data

# Verification
from verification.self_consistency import SelfConsistencyVerification, SelfConsistencyConfig
from verification.cove import ChainOfVerification, CoVeConfig
from utils.cost_tracker import CostTracker, log_experiment

# Utils
from utils.json_parser import (
    parse_json_response,
    extract_json_from_response,
    validate_constitution_response
)

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# BACKWARD COMPATIBLE FUNCTIONS
# =============================================================================

def create_prompt(country: str, name: str, start_year: int, end_year: int) -> Tuple[str, str]:
    """
    Create system and user prompts for polity-level constitution analysis.

    Args:
        country: Country/polity name
        name: Leader name (use "Unknown" if not available)
        start_year: Start year of the polity period
        end_year: End year of the polity period

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        country=country,
        name=name,
        start_year=start_year,
        end_year=end_year
    )

    return SYSTEM_PROMPT, user_prompt


def _extract_json_from_response(response: str) -> str:
    """Extract JSON object from LLM response (backward compatible wrapper)."""
    return extract_json_from_response(response)


def parse_llm_response(response: str, max_retries: int, retry_delay: float) -> Dict:
    """Parse the LLM's JSON response with retry logic (backward compatible)."""
    return parse_json_response(response, max_retries, retry_delay, verbose=False)


def _detect_provider(model_identifier: str) -> str:
    """Detect the LLM provider (backward compatible wrapper)."""
    return detect_provider(model_identifier)


def _route_to_model(
    system_prompt: str,
    user_prompt: str,
    model_identifier: str,
    api_keys: Dict,
    llm_params: Dict,
    max_retries: int,
    retry_delay: float,
    use_search: bool,
    url_tracker: Optional[List[str]] = None,
    query_tracker: Optional[List[str]] = None,
    force_search: bool = False,
) -> Optional[str]:
    """Route the query to the appropriate model API."""
    provider = detect_provider(model_identifier)

    if use_search:
        print(f"INFO: Using WEB SEARCH for {model_identifier} ({provider})")

        if provider == 'openai':
            return run_openai_search_agent(
                system_prompt, user_prompt, model_identifier,
                api_key=api_keys.get('openai'),
                serper_api_key=api_keys.get('serper'),
                url_tracker=url_tracker,
                query_tracker=query_tracker,
                force_search=force_search,
            )
        elif provider == 'gemini':
            return run_gemini_search_agent(
                system_prompt, user_prompt, model_identifier,
                api_key=api_keys.get('gemini'),
                serper_api_key=api_keys.get('serper'),
                url_tracker=url_tracker,
                query_tracker=query_tracker,
                force_search=force_search,
            )
        elif provider == 'anthropic':
            return run_anthropic_search_agent(
                system_prompt, user_prompt, model_identifier,
                api_key=api_keys.get('anthropic'),
                serper_api_key=api_keys.get('serper'),
                url_tracker=url_tracker,
                query_tracker=query_tracker,
                force_search=force_search,
            )
        elif provider == 'bedrock':
            return run_bedrock_search_agent(
                system_prompt, user_prompt, model_identifier,
                api_keys=api_keys,
                serper_api_key=api_keys.get('serper'),
                url_tracker=url_tracker,
                query_tracker=query_tracker,
                force_search=force_search,
            )
    else:
        if provider == 'openai':
            return query_openai_model(
                system_prompt, user_prompt, model_identifier,
                api_keys.get('openai'), llm_params, max_retries, retry_delay
            )
        elif provider == 'gemini':
            return query_gemini_model(
                system_prompt, user_prompt, model_identifier,
                api_keys.get('gemini'), llm_params, max_retries, retry_delay
            )
        elif provider == 'anthropic':
            return query_anthropic_model(
                system_prompt, user_prompt, model_identifier,
                api_keys.get('anthropic'), llm_params, max_retries, retry_delay
            )
        elif provider == 'bedrock':
            return query_aws_bedrock_model(
                system_prompt, user_prompt, model_identifier,
                api_keys, llm_params, max_retries, retry_delay
            )

    print(f"ERROR: Unknown provider for model {model_identifier}")
    return None


def _apply_polity_verification(
    result: Dict,
    system_prompt: str,
    user_prompt: str,
    country: str,
    name: str,
    start_year: int,
    end_year: int,
    verify_type: str,
    llm: Any,
    verifier_llm: Any,
    sc_n_samples: int,
    sc_temperatures: List[float],
    cove_questions_per_element: int,
    initial_status: str,
    initial_reasoning: str,
) -> None:
    """Apply SC and/or CoVe verification to a polity-level constitution prediction.

    Updates result in-place. verify_type: 'self_consistency' | 'cove' | 'both'.
    For 'both': SC runs first; its majority prediction feeds into CoVe.
    """
    current_status = initial_status

    # ------------------------------------------------------------------
    # Self-Consistency
    # ------------------------------------------------------------------
    if verify_type in ('self_consistency', 'both'):
        # SC1 = initial call (already written to result as constitution_SC1).
        # SC2..SCN+1 = additional calls at varying temperatures.
        sc_preds: List[float] = []
        if initial_status is not None:
            try:
                seed = float(int(float(initial_status)))
                if seed in (0.0, 1.0, 2.0):
                    sc_preds.append(seed)
            except (ValueError, TypeError):
                pass

        temps = (sc_temperatures or [1.0, 1.0, 1.0])[:sc_n_samples]
        for sc_i, temp in enumerate(temps, start=2):  # SC2, SC3, ...
            try:
                resp = llm.call(system_prompt=system_prompt, user_prompt=user_prompt, temperature=temp)
                parsed = parse_json_response(resp.content, verbose=False)
                validated = validate_constitution_response(parsed)
                pred = validated.get('constitution')
                if pred is not None:
                    sc_preds.append(pred)
                    result[f'constitution_SC{sc_i}'] = int(pred)
                    result[f'constitution_year_SC{sc_i}'] = validated.get('constitution_year')
                    result[f'constitution_document_name_SC{sc_i}'] = validated.get('document_name', 'N/A')
                    result[f'constitution_document_types_SC{sc_i}'] = validated.get('document_types')
                else:
                    result[f'constitution_SC{sc_i}'] = None
            except Exception as e:
                print(f"  SC sample {sc_i} (temp={temp}) failed: {e}")
                result[f'constitution_SC{sc_i}'] = None

        if sc_preds:
            counter = Counter(sc_preds)
            majority_pred, majority_count = counter.most_common(1)[0]
            agreement = majority_count / len(sc_preds)
            n = len(sc_preds)
            if majority_count == n:
                uncertainty = 'none'
            elif majority_count >= 2:
                uncertainty = 'low'
            else:
                uncertainty = 'high'
            verified_int = int(majority_pred)
            current_status = verified_int
            result['constitution_prediction'] = verified_int
            result['constitution_agreement'] = round(agreement, 3)
            result['constitution_uncertainty'] = uncertainty

    # ------------------------------------------------------------------
    # Chain of Verification (CoVe)
    # ------------------------------------------------------------------
    if verify_type in ('cove', 'both'):
        if verifier_llm is None:
            print("  Warning: CoVe requires --verifier-model. Skipping CoVe.")
            return
        try:
            cove = ChainOfVerification(
                primary_llm=llm,
                verifier_llm=verifier_llm,
                config=CoVeConfig(questions_per_element=cove_questions_per_element),
                cost_tracker=CostTracker()
            )
            if current_status is not None and str(current_status) in ('0', '1', '2'):
                init_pred = str(int(current_status))
            elif current_status is not None and str(current_status).lower() in ('yes', 'true'):
                init_pred = '1'
            else:
                init_pred = '0'

            verify_result = cove.verify(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                indicator='constitution',
                valid_labels=['0', '1', '2'],
                initial_prediction=init_pred,
                initial_reasoning=initial_reasoning or '',
                polity=country,
                name=name,
                start_year=start_year,
                end_year=end_year,
            )
            verified_pred = verify_result.verified_prediction
            result['constitution_verification'] = str(verify_result.verification_details)
            if verified_pred in ('0', '1', '2'):
                result['constitution_prediction'] = int(verified_pred)
        except Exception as e:
            print(f"  CoVe verification failed: {e}")
            result['constitution_verification'] = f'Error: {e}'


def process_single_polity(
    country: str,
    name: str,
    start_year: int,
    end_year: int,
    model_key: str,
    model_identifier: str,
    api_keys: Dict,
    llm_params: Dict,
    max_retries: int,
    retry_delay: float,
    use_search_flag: bool,
    # Verification params (optional)
    verify_type: str = 'none',
    llm: Any = None,
    verifier_llm: Any = None,
    sc_n_samples: int = 3,
    sc_temperatures: Optional[List[float]] = None,
    cove_questions_per_element: int = 1,
    # Search tracking
    force_search: bool = False,
) -> Optional[Dict]:
    """Process a single polity for constitution analysis (polity pipeline).

    When an LLM instance is supplied and web search is disabled, uses
    llm.call() so that token counts are available for cost tracking.
    Cost data is returned under private keys (_input_tokens, _output_tokens,
    _cached_tokens) for the caller to aggregate; they are stripped before
    the final CSV is written.

    Search metadata (URLs, queries) are returned under search_queries
    and urls_used columns when search mode is active.
    """
    system_prompt, user_prompt = create_prompt(country, name, start_year, end_year)

    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    thinking_tokens = 0

    # Track search URLs and queries
    url_tracker: List[str] = []
    query_tracker: List[str] = []

    # For forced search mode, run pre-search and enrich the prompt
    if use_search_flag and force_search:
        try:
            from pipeline.pre_search import PreSearcher
            pre_searcher = PreSearcher(serper_api_key=api_keys.get('serper', ''))
            search_result = pre_searcher.search(
                polity=country, name=name, start_year=start_year, end_year=end_year
            )
            if search_result.context:
                user_prompt = pre_searcher.enrich_prompt(user_prompt, search_result)
            url_tracker.extend(search_result.urls_used)
            query_tracker.extend(search_result.search_queries)
        except Exception as e:
            print(f"  Pre-search failed for {country}: {e}")

    # Prefer llm.call() (gives token counts) when available and not in search mode
    if llm is not None and not use_search_flag:
        model_response = llm.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=llm_params.get('temperature', DEFAULT_TEMPERATURE),
            max_tokens=llm_params.get('max_tokens', DEFAULT_MAX_TOKENS),
            top_p=llm_params.get('top_p', DEFAULT_TOP_P),
        )
        response_content = model_response.content
        input_tokens = model_response.input_tokens
        output_tokens = model_response.output_tokens
        cached_tokens = model_response.cached_tokens
        thinking_tokens = model_response.thinking_tokens
    elif llm is not None and use_search_flag and force_search:
        # Forced search: prompt already enriched with pre-search context above,
        # use llm.call() to get token counts
        model_response = llm.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=llm_params.get('temperature', DEFAULT_TEMPERATURE),
            max_tokens=llm_params.get('max_tokens', DEFAULT_MAX_TOKENS),
            top_p=llm_params.get('top_p', DEFAULT_TOP_P),
        )
        response_content = model_response.content
        input_tokens = model_response.input_tokens
        output_tokens = model_response.output_tokens
        cached_tokens = model_response.cached_tokens
        thinking_tokens = model_response.thinking_tokens
    else:
        # Agentic search: the LLM decides what to search
        response_content = _route_to_model(
            system_prompt, user_prompt, model_identifier,
            api_keys, llm_params, max_retries, retry_delay, use_search_flag,
            url_tracker=url_tracker,
            query_tracker=query_tracker,
            force_search=force_search,
        )

    if response_content is None:
        return None

    parsed_result = parse_llm_response(response_content, max_retries, retry_delay)
    validated = validate_constitution_response(parsed_result)

    status = validated.get('constitution')  # float 0.0, 1.0, or 2.0, or None
    explanation = validated.get('reasoning') or 'No explanation provided'

    use_sc = verify_type in ('self_consistency', 'both')
    if use_sc:
        # SC will be applied: use SC1 column names for the initial call.
        result = {
            'constitution_SC1': int(status) if status is not None else 0,
            'constitution_year_SC1': validated.get('constitution_year', None),
            'constitution_document_name_SC1': validated.get('document_name', "N/A"),
            'constitution_document_types_SC1': validated.get('document_types', None),
            # Private cost fields — stripped before CSV output
            '_input_tokens': input_tokens,
            '_output_tokens': output_tokens,
            '_cached_tokens': cached_tokens,
            '_thinking_tokens': thinking_tokens,
            '_model_identifier': model_identifier,
        }
    else:
        # No SC (none or cove): use plain column names.
        result = {
            'constitution_prediction': int(status) if status is not None else 0,
            'constitution_year': validated.get('constitution_year', None),
            'constitution_document_name': validated.get('document_name', "N/A"),
            'constitution_document_types': validated.get('document_types', None),
            'constitution_reasoning': explanation,
            'constitution_confidence': validated.get('confidence_score', None),
            # Private cost fields — stripped before CSV output
            '_input_tokens': input_tokens,
            '_output_tokens': output_tokens,
            '_cached_tokens': cached_tokens,
            '_thinking_tokens': thinking_tokens,
            '_model_identifier': model_identifier,
        }

    # Only emit search columns when search was actually used
    if query_tracker:
        result['search_queries'] = ' | '.join(query_tracker)
    if url_tracker:
        result['urls_used'] = ' | '.join(url_tracker)

    # Apply verification if requested
    if verify_type != 'none' and llm is not None:
        _apply_polity_verification(
            result=result,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            country=country,
            name=name,
            start_year=start_year,
            end_year=end_year,
            verify_type=verify_type,
            llm=llm,
            verifier_llm=verifier_llm,
            sc_n_samples=sc_n_samples,
            sc_temperatures=sc_temperatures or [1.0, 1.0, 1.0],
            cove_questions_per_element=cove_questions_per_element,
            initial_status=status if status is not None else 0,
            initial_reasoning=explanation,
        )

    return result


def _process_one_row(
    row: pd.Series,
    models_dict: Dict[str, str],
    api_keys: Dict,
    llm_params: Dict,
    max_retries: int,
    retry_delay: float,
    use_search_flag: bool,
    verify_type: str,
    model_llms: Dict[str, Any],
    verifier_llm: Any,
    sc_n_samples: int,
    sc_temperatures: Optional[List[float]],
    cove_questions_per_element: int,
    force_search: bool = False,
) -> Dict:
    """
    Process all models for a single polity row.

    Returns a merged result dict (original row data + all model predictions).
    Private cost fields (_input_tokens_*, _output_tokens_*, _cached_tokens_*,
    _model_identifier) are included for the caller to aggregate into CostTracker.
    """
    country = row[COL_TERRITORY_NAME]
    name = str(row.get(COL_LEADER_NAME, "Unknown")) if COL_LEADER_NAME in row.index else "Unknown"
    start_year = int(row[COL_START_YEAR])
    end_year = int(row[COL_END_YEAR])

    entry_result = row.to_dict()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models_dict)) as executor:
        future_to_model = {
            executor.submit(
                process_single_polity,
                country, name, start_year, end_year,
                model_key, model_identifier,
                api_keys, llm_params,
                max_retries, retry_delay, use_search_flag,
                verify_type, model_llms.get(model_key), verifier_llm,
                sc_n_samples, sc_temperatures, cove_questions_per_element,
                force_search,
            ): model_key
            for model_key, model_identifier in models_dict.items()
        }

        for future in concurrent.futures.as_completed(future_to_model):
            model_key = future_to_model[future]
            try:
                result = future.result()
                if result:
                    entry_result.update(result)
                else:
                    raise ValueError("Query function returned None")
            except Exception as exc:
                print(f"\nERROR processing {country} with model {model_key}: {exc}")
                error_msg = f"Failed after retries: {exc}"
                entry_result['constitution_prediction'] = -1
                entry_result['constitution_document_name'] = "Query Failed"
                entry_result['constitution_year'] = None
                entry_result['constitution_reasoning'] = error_msg

    return entry_result


def _aggregate_row_costs(
    entry_result: Dict,
    models_dict: Dict[str, str],
    cost_tracker: CostTracker,
) -> None:
    """Read private cost fields from entry_result, record in cost_tracker, then strip them."""
    model_identifier = next(iter(models_dict.values()), '')
    in_tok = entry_result.pop('_input_tokens', 0) or 0
    out_tok = entry_result.pop('_output_tokens', 0) or 0
    ca_tok = entry_result.pop('_cached_tokens', 0) or 0
    think_tok = entry_result.pop('_thinking_tokens', 0) or 0
    if in_tok or out_tok:
        cost_tracker.add_usage(
            model=model_identifier,
            input_tokens=int(in_tok),
            output_tokens=int(out_tok),
            cached_tokens=int(ca_tok),
            thinking_tokens=int(think_tok),
            indicator='constitution',
        )
    entry_result.pop('_model_identifier', None)


def process_batch(
    df: pd.DataFrame,
    models_dict: Dict[str, str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay: float = DEFAULT_DELAY,
    api_keys: Optional[Dict] = None,
    llm_params: Optional[Dict] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    use_search_flag: bool = False,
    # Verification params (polity pipeline)
    verify_type: str = 'none',
    model_llms: Optional[Dict[str, Any]] = None,   # model_key -> BaseLLM
    verifier_llm: Any = None,
    sc_n_samples: int = 3,
    sc_temperatures: Optional[List[float]] = None,
    cove_questions_per_element: int = 1,
    # Parallel row processing
    max_workers: int = 1,
    # Cost tracking
    cost_tracker: Optional[CostTracker] = None,
    output_path: Optional[str] = None,
    # Search mode
    force_search: bool = False,
) -> pd.DataFrame:
    """Process polity data in batches (polity pipeline).

    Args:
        max_workers: Number of polity rows to process concurrently.
                     1 (default) = sequential; >1 = parallel windows.
        cost_tracker: Optional shared CostTracker. A new one is created if
                      not supplied.
        output_path:  If provided, a cost report JSON is saved alongside it.
    """
    _api_keys = api_keys or {}
    _llm_params = llm_params or {}
    _model_llms = model_llms or {}
    cost_tracker = cost_tracker or CostTracker()

    results: List[Dict] = []
    total_polities = len(df)
    checkpoint_files: List[str] = []
    checkpoint_at_50_percent = total_polities // 2

    verify_info = f" | verification: {verify_type}" if verify_type != 'none' else ""
    print(f"Starting to process {total_polities} polities using models: {list(models_dict.keys())}{verify_info}")
    print(f"Parallel row workers: {max_workers}")
    print(f"Checkpoint will be saved at: {checkpoint_at_50_percent} polities (50%)")

    # Closure that captures shared kwargs — avoids **dict unpacking so that
    # Pylance can resolve argument types correctly.
    def _call_row(row: pd.Series) -> Dict:
        return _process_one_row(
            row=row,
            models_dict=models_dict,
            api_keys=_api_keys,
            llm_params=_llm_params,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_search_flag=use_search_flag,
            verify_type=verify_type,
            model_llms=_model_llms,
            verifier_llm=verifier_llm,
            sc_n_samples=sc_n_samples,
            sc_temperatures=sc_temperatures,
            cove_questions_per_element=cove_questions_per_element,
            force_search=force_search,
        )

    def _handle_row_result(entry_result: Dict, processed_count: int) -> None:
        """Aggregate cost from result, append to results, maybe checkpoint."""
        _aggregate_row_costs(entry_result, models_dict, cost_tracker)
        results.append(entry_result)
        if processed_count == checkpoint_at_50_percent:
            temp_df = pd.DataFrame(results)
            fname = f'checkpoint_50percent_{total_polities}polities.csv'
            temp_df.to_csv(fname, index=False)
            checkpoint_files.append(fname)
            print(f"\n  Checkpoint saved at 50%: {fname}")

    if max_workers <= 1:
        # ── Sequential (original behaviour) ────────────────────────────────
        for processed_count, (_, row) in enumerate(
            tqdm(df.iterrows(), total=total_polities, desc="Processing Polities"), start=1
        ):
            try:
                entry_result = _call_row(row)
            except Exception as e:
                print(f"\nERROR on row {processed_count}: {e}")
                entry_result = row.to_dict()
                entry_result['constitution_prediction'] = -1
                entry_result['constitution_reasoning'] = f"Error: {e}"

            _handle_row_result(entry_result, processed_count)
            time.sleep(delay)

    else:
        # ── Parallel window-based ──────────────────────────────────────────
        row_list = [(i, row) for i, row in df.iterrows()]
        windows = [row_list[i:i + max_workers] for i in range(0, len(row_list), max_workers)]
        processed_count = 0

        with tqdm(total=total_polities, desc="Processing Polities") as pbar:
            for window in windows:
                window_results: List[Tuple[int, Dict]] = []

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_orig = {
                        executor.submit(_call_row, row): orig_idx
                        for orig_idx, row in window
                    }
                    for future in as_completed(future_to_orig):
                        orig_idx = future_to_orig[future]
                        row = df.iloc[orig_idx]
                        try:
                            entry_result = future.result()
                        except Exception as e:
                            print(f"\nERROR on row {orig_idx}: {e}")
                            entry_result = row.to_dict()
                            entry_result['constitution_prediction'] = -1
                            entry_result['constitution_reasoning'] = f"Error: {e}"
                        window_results.append((orig_idx, entry_result))

                # Re-sort by original index to preserve row order
                window_results.sort(key=lambda x: x[0])
                for _, entry_result in window_results:
                    processed_count += 1
                    _handle_row_result(entry_result, processed_count)

                pbar.update(len(window))
                if window != windows[-1]:
                    time.sleep(delay)

    # ── Final checkpoint and cleanup ──────────────────────────────────────
    final_df = pd.DataFrame(results)

    fname = f'checkpoint_100percent_{total_polities}polities.csv'
    final_df.to_csv(fname, index=False)
    checkpoint_files.append(fname)
    print(f"\n  Checkpoint saved at 100%: {fname}")

    for checkpoint_file in checkpoint_files:
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print(f"  Deleted checkpoint file: {checkpoint_file}")
        except Exception as e:
            print(f"  Warning: Could not delete checkpoint file {checkpoint_file}: {e}")

    # ── Cost report ──────────────────────────────────────────────────────
    cost_tracker.print_summary()
    if output_path:
        logs_dir = Path('data/logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        cost_report = logs_dir / f'{Path(output_path).stem}_costs.json'
        cost_tracker.save_report(str(cost_report))
        print(f"  Cost report saved: {cost_report}")

    return final_df


def save_results(results_df: pd.DataFrame, output_path: str) -> None:
    """Save results to CSV file."""
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results saved to: {output_path}")
    print("\nPolity-Level Result Statistics:")
    print(f"Total polities processed: {len(results_df)}")
    print("\nSample results:")
    print(results_df[[COL_TERRITORY_NAME, COL_START_YEAR, COL_END_YEAR]].head())


def _parse_test_argument(test_arg: str, df: pd.DataFrame) -> pd.DataFrame:
    """Parse test argument and slice the DataFrame accordingly."""
    try:
        if ':' in test_arg:
            parts = test_arg.split(':')
            start = int(parts[0]) if parts[0] else None
            end = int(parts[1]) if parts[1] else None
            result_df = df.iloc[start:end]
            print(f"Testing mode: Processing polities from index "
                  f"{start if start is not None else 0} to {end if end is not None else 'end'}")
            return result_df
        else:
            num_rows = int(test_arg)
            result_df = df.head(num_rows)
            print(f"Testing mode: Processing only first {num_rows} polities")
            return result_df

    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid format for --test argument: '{test_arg}'. "
            f"Use a number (e.g., '10') or a slice (e.g., '100:200')."
        ) from e


def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description=(
            'Historical Political Indicators Pipeline\n'
            '\n'
            'Two pipelines:\n'
            '  indicators   -- Main pipeline. Predicts any combination of indicators at the\n'
            '                  leader level (one row per leader reign).\n'
            '                  Input: plt_leaders_data.csv\n'
            '                  Supports single / multiple / sequential prompt modes,\n'
            '                  self-consistency, and Chain-of-Verification (CoVe).\n'
            '\n'
            '  constitution -- Legacy pipeline. Predicts constitution (0/1/2) at the polity\n'
            '                  level. Single model only.\n'
            '                  Input: plt_polity_data_v2.csv'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # --- CONSTITUTION pipeline (legacy, polity level) ---

  python main.py --pipeline constitution
  python main.py --pipeline constitution -i data/plt_polity_data_v2.csv -o results.csv
  python main.py --pipeline constitution --models gemini-3.1-pro-preview --search-mode agentic --test 10


  # --- INDICATORS pipeline (main, leader level) ---

  # Predict all indicators (default: single mode)
  python main.py --pipeline indicators --indicators constitution sovereign federalism checks collegiality petition assembly entry exit symbolism

  # With agentic search
  python main.py --pipeline indicators --indicators sovereign assembly checks --search-mode agentic --test 10

  # With forced search
  python main.py --pipeline indicators --indicators sovereign assembly checks --search-mode forced --test 10

  # With Gemini Batch API (50% cost savings, no search)
  python main.py --pipeline indicators --indicators sovereign assembly checks --models gemini-3.1-pro-preview --use-batch --test 20

  # With self-consistency verification on assembly
  python main.py --pipeline indicators --indicators assembly --verify self_consistency --verify-indicators assembly

  # Multiple prompt mode
  python main.py --pipeline indicators --mode multiple --indicators sovereign federalism checks collegiality petition assembly entry exit symbolism

  # Sequential mode with a user-defined indicator order
  python main.py --pipeline indicators --mode sequential --indicators sovereign federalism checks collegiality assembly entry exit symbolism --sequence assembly sovereign checks collegiality federalism entry exit symbolism

  # CoVe verification with a Bedrock verifier model
  python main.py --pipeline indicators --indicators constitution --verify cove --verify-indicators constitution --verifier-model us.anthropic.claude-sonnet-4-5-20250929-v1:0
        """
    )

    # Input/Output arguments
    parser.add_argument(
        '--input', '-i',
        default=None,
        help=(
            'Input data file path (CSV or JSONL). Format is auto-detected by extension.\n'
            'Defaults to the standard file for the chosen pipeline:\n'
            '  constitution → ./data/plt_polity_data_v2.csv  (requires: territorynamehistorical, start_year, end_year)\n'
            '  indicators   → ./data/plt_leaders_data.csv    (requires: territorynamehistorical, name, start_year, end_year)'
        )
    )
    parser.add_argument(
        '--output', '-o',
        default='./data/results/llm_predictions.csv',
        help='Output CSV file path'
    )

    # Prompt customization
    parser.add_argument('--user_prompt', '-u', help='User prompt for custom queries')
    parser.add_argument('--system_prompt', '-s', help='System prompt for custom queries')

    # Model configuration
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=['Gemini=gemini-3.1-pro-preview'],
        help='Space-separated list of models in KEY=IDENTIFIER format'
    )
    parser.add_argument(
        '--use-search',
        action='store_true',
        help='(Deprecated) Use --search-mode agentic instead. Kept for backward compatibility.'
    )
    parser.add_argument(
        '--search-mode',
        choices=['none', 'agentic', 'forced', 'gemini_grounding'],
        default='none',
        help=(
            'Search mode for LLM generation (default: none).\n'
            '  none              — Pure LLM output, no web search.\n'
            '  agentic           — LLM decides whether to search (tool_choice=auto).\n'
            '  forced            — Always perform web search before LLM answers.\n'
            '  gemini_grounding  — Gemini native Google Search grounding (Gemini models only).\n'
            '                      Grounding metadata stored per indicator / per SC slot.'
        )
    )
    parser.add_argument(
        '--use-batch',
        action='store_true',
        help=(
            'Use Gemini Batch API for main predictions (50%% cost savings).\n'
            'Only works with Gemini models. Verification runs synchronously after batch.\n'
            'Compatible with --search-mode forced and --search-mode gemini_grounding.'
        )
    )

    # API configuration
    parser.add_argument(
        '--api-key', '-k',
        help='API key for OpenAI (or use OPENAI_API_KEY env var)'
    )

    # Processing configuration
    parser.add_argument(
        '--checkpoint', '-b',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        dest='batch_size',
        help='Batch size for temporary saves'
    )
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=DEFAULT_DELAY,
        help='Delay between API calls (seconds)'
    )
    parser.add_argument(
        '--test', '-t',
        help='Process subset of data for testing (e.g., "10" or "100:200")'
    )

    # LLM parameters
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help='Max tokens for model response'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=DEFAULT_TEMPERATURE,
        help='Sampling temperature for generation'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=DEFAULT_TOP_P,
        help='Top-p (nucleus) sampling probability'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help='Maximum number of retries for a failed API call'
    )
    parser.add_argument(
        '--retry-delay',
        type=int,
        default=DEFAULT_RETRY_DELAY,
        help='Delay in seconds between retries'
    )

    parser.add_argument(
        '--pipeline',
        choices=['indicators', 'constitution'],
        default='indicators',
        help=(
            'Pipeline to run (default: indicators).\n'
            '  indicators   — Main pipeline: predicts any combination of indicators at the\n'
            '                 leader level. Supports single/multiple/sequential prompt modes,\n'
            '                 self-consistency, and Chain-of-Verification (CoVe).\n'
            '  constitution — Legacy pipeline: predicts constitution (0/1/2) at the polity\n'
            '                 level. Single model only.'
        )
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'multiple', 'sequential'],
        default='single',
        help='Prompt mode: single (unified prompt), multiple (separate prompts), or sequential (all indicators in sequence)'
    )
    parser.add_argument(
        '--indicators',
        nargs='+',
        default=['constitution'],
        help=f'Indicators to predict. Options: {ALL_INDICATORS}'
    )
    parser.add_argument(
        '--verify',
        choices=['none', 'self_consistency', 'cove', 'both'],
        default='self_consistency',
        help='Verification method to use (default: self_consistency)'
    )
    parser.add_argument(
        '--verify-indicators',
        nargs='+',
        default=[],
        help='Which indicators to apply verification to'
    )
    parser.add_argument(
        '--verifier-model',
        help='Model to use for verification (for CoVe)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=2,
        help='Number of samples for self-consistency'
    )
    parser.add_argument(
        '--sc-temperatures',
        nargs='+',
        type=float,
        default=[1.0, 1.0, 1.0],
        help='Temperature values for self-consistency sampling'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=500,
        help='Save checkpoint every N rows (leader pipeline only, default 500)'
    )
    parser.add_argument(
        '--parallel-rows',
        type=int,
        default=1,
        metavar='N',
        help=(
            'Number of input rows (leaders) to process concurrently (default: 1 = sequential). '
            'Works with all prompt modes (single, multiple, sequential). '
            'Each parallel worker makes its own independent API call(s) for one leader row.'
        )
    )
    parser.add_argument(
        '--sequence',
        nargs='+',
        help='Indicator sequence for sequential mode (space-separated, e.g., "constitution assembly sovereign")'
    )
    parser.add_argument(
        '--random-sequence',
        action='store_true',
        help='Randomize indicator order in sequential mode'
    )
    parser.add_argument(
        '--reasoning',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='Include reasoning in predictions (default: True). Set to False for prediction-only output.'
    )
    parser.add_argument(
        '--logprobs',
        action='store_true',
        default=False,
        help=(
            'Request token-level log probabilities from Gemini for uncertainty quantification.\n'
            'Adds {indicator}_logprob columns to output (one per indicator).\n'
            'Supported models: gemini-2.5-flash, gemini-2.5-pro (default: off).'
        )
    )

    args = parser.parse_args()

    # Resolve default input file based on pipeline level if not explicitly provided
    if args.input is None:
        if args.pipeline == 'indicators':
            args.input = './data/plt_leaders_data.csv'
        else:
            args.input = './data/plt_polity_data_v2.csv'

    # Collect API keys
    api_keys = {
        'openai': args.api_key or os.getenv('OPENAI_API_KEY'),
        'gemini': os.getenv('GEMINI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'aws_session_token': os.getenv('AWS_SESSION_TOKEN'),
        'serper': os.getenv('SERPER_API_KEY')
    }

    # Handle backward-compatible --use-search flag
    if args.use_search and args.search_mode == 'none':
        args.search_mode = 'agentic'

    # Resolve search mode enum
    search_mode = SearchMode(args.search_mode)

    # Validate search configuration
    if search_mode == SearchMode.AGENTIC and not api_keys['serper']:
        raise ValueError(
            "Agentic search mode requires a Serper API key. "
            "Set the SERPER_API_KEY environment variable or use --search-mode forced "
            "(which uses free Wikipedia/DuckDuckGo first)."
        )

    if search_mode == SearchMode.GEMINI_GROUNDING:
        model_arg_check = args.models[0]
        model_id_check = model_arg_check.split('=', 1)[-1] if '=' in model_arg_check else model_arg_check
        if not any(model_id_check.startswith(prefix) for prefix in GEMINI_MODELS):
            raise ValueError(
                f"--search-mode gemini_grounding requires a Gemini model, got '{model_id_check}'. "
                "Gemini grounding is a Gemini-specific feature."
            )

    # Validate batch configuration
    if args.use_batch:
        model_arg_check = args.models[0]
        model_id_check = model_arg_check.split('=', 1)[-1] if '=' in model_arg_check else model_arg_check
        if not any(model_id_check.startswith(prefix) for prefix in GEMINI_MODELS):
            raise ValueError(
                f"--use-batch is only supported with Gemini models, got '{model_id_check}'. "
                "Batch API is a Gemini-specific feature."
            )
        if search_mode == SearchMode.AGENTIC:
            raise ValueError(
                "--use-batch is incompatible with --search-mode agentic. "
                "Agentic search requires multi-turn LLM calls which batch API does not support.\n"
                "Use --search-mode forced --use-batch instead (pre-search + batch)."
            )

    # Logprobs support notice
    if args.logprobs:
        model_arg_for_check = args.models[0]
        model_id_for_check = model_arg_for_check.split('=', 1)[-1] if '=' in model_arg_for_check else model_arg_for_check
        if not any(model_id_for_check.startswith(p) for p in GEMINI_MODELS):
            print(f"WARN: --logprobs is only supported by Gemini models (got '{model_id_for_check}'). Logprobs will be skipped.")
        elif not any(kw in model_id_for_check for kw in ('2.5', '3.5', '3.0', '3.1')):
            print(f"INFO: --logprobs: supported models are gemini-2.5-flash and gemini-2.5-pro. Other Gemini models will fall back silently.")

    if args.pipeline == 'indicators':
        print("Pipeline: INDICATORS (modular, all indicators supported)")
        print(f"Input:    {args.input}")
        print(f"Search:   {search_mode.value}")
        if args.use_batch:
            print("Batch:    Gemini Batch API (50% cost savings)")

        # Validate indicator names early to catch typos before any API calls.
        _valid_set = set(ALL_INDICATORS)
        _bad = [i for i in args.indicators if i not in _valid_set]
        if _bad:
            parser.error(
                f"Unknown indicator(s): {_bad}. Valid options: {ALL_INDICATORS}"
            )

        # When --verify is set but --verify-indicators is omitted, verify all
        # predicted indicators rather than silently doing nothing.
        _verify_indicators = args.verify_indicators
        if args.verify != 'none' and not _verify_indicators:
            _verify_indicators = list(args.indicators)
            print(f"INFO: --verify-indicators not set; defaulting to all --indicators: {_verify_indicators}")

        # Parse model from --models argument (use first one for leader pipeline)
        model_arg = args.models[0]
        if '=' in model_arg:
            _, model_identifier = model_arg.split('=', 1)
        else:
            model_identifier = model_arg

        # ── Search mode: agentic ──────────────────────────────────────────
        # Use SearchPredictor (LLM decides whether to search via tool calling)
        if search_mode == SearchMode.AGENTIC:
            from pipeline.search_predictor import SearchPredictor

            df = load_polity_data(args.input)
            if args.test:
                df = _parse_test_argument(args.test, df)

            search_predictor = SearchPredictor(
                model=model_identifier,
                api_keys=api_keys,
                mode=args.mode,
                indicators=args.indicators,
                reasoning=args.reasoning,
                sequence=args.sequence,
                random_sequence=args.random_sequence,
                force_search=False,
            )

            results = []
            for idx in tqdm(range(len(df)), desc="Processing (agentic search)"):
                row = df.iloc[idx]
                polity = str(row.get(COL_TERRITORY_NAME, "Unknown"))
                name = str(row.get(COL_LEADER_NAME, "Unknown"))
                start_year = int(row[COL_START_YEAR])
                end_year = None if pd.isna(row[COL_END_YEAR]) else int(row[COL_END_YEAR])

                pred = search_predictor.predict(polity, name, start_year, end_year)
                result_dict = row.to_dict()
                result_dict.update(pred.to_dict())
                results.append(result_dict)
                time.sleep(args.delay)

            results_df = pd.DataFrame(results)
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

            # Save CSV
            results_df.to_csv(args.output, index=False)

            # Save JSON
            json_path = args.output.replace('.csv', '.json')
            records = results_df.to_dict(orient='records')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2, default=str)

            print(f"\nResults saved to: {args.output}")
            print(f"JSON saved to: {json_path}")
            if not args.test:
                log_experiment(
                    output_path=args.output, pipeline='indicators',
                    prompt_style=args.mode, model=model_identifier,
                    indicators=args.indicators, verify=args.verify,
                    search_mode=args.search_mode, total_entries=len(df),
                    cost_summary={}, n_samples=args.n_samples,
                )
            print("\nLeader-level pipeline (agentic search) completed successfully!")
            return

        # ── Search mode: forced ───────────────────────────────────────────
        # Two sub-paths: forced + batch  OR  forced + synchronous
        if search_mode == SearchMode.FORCED:
            if args.use_batch:
                # Pre-search + Gemini Batch API (SC embedded in batch)
                from pipeline.pre_search import PreSearcher
                from pipeline.jsonl_batch_runner import run_inline_batch

                config = PredictionConfig(
                    mode=PromptMode.MULTIPLE,  # batch requires multiple mode (one indicator per prompt)
                    indicators=args.indicators,
                    verify=VerificationType.NONE,  # SC handled inside batch via n_samples
                    verify_indicators=[],
                    model=model_identifier,
                    verifier_model=args.verifier_model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    sc_n_samples=0,
                    sc_temperatures=args.sc_temperatures,
                    sequence=args.sequence,
                    random_sequence=args.random_sequence,
                    reasoning=args.reasoning,
                    use_logprobs=args.logprobs,
                )
                predictor = Predictor(config, api_keys)

                df = load_polity_data(args.input)
                if args.test:
                    df = _parse_test_argument(args.test, df)

                # Phase 1: Pre-search and enrich prompts via monkey-patch
                print("\n[Pre-Search] Running deterministic search for all rows...")
                pre_searcher = PreSearcher(serper_api_key=api_keys.get('serper', ''))
                original_build = predictor.prompt_builder.build

                def _build_with_search(polity, name, start_year, end_year):
                    search_result = pre_searcher.search(polity, name, start_year, end_year)
                    prompts = original_build(polity, name, start_year, end_year)
                    enriched = []
                    for p in prompts:
                        enriched_user = pre_searcher.enrich_prompt(p.user_prompt, search_result)
                        from prompts.base_builder import PromptOutput
                        enriched.append(PromptOutput(
                            system_prompt=p.system_prompt,
                            user_prompt=enriched_user,
                            indicators=p.indicators,
                            metadata={**p.metadata, 'search_queries': search_result.search_queries,
                                      'urls_used': search_result.urls_used,
                                      'sources_used': search_result.sources_used,
                                      'search_context': search_result.context},
                        ))
                    return enriched

                predictor.prompt_builder.build = _build_with_search

                # Phase 2: Run batch (SC embedded via n_samples)
                results_df = run_inline_batch(
                    df=df,
                    indicators=args.indicators,
                    model=model_identifier,
                    api_key=api_keys.get('gemini', ''),
                    n_samples=args.n_samples,
                    output_path=args.output,
                    prompt_builder=predictor.prompt_builder,
                    sc_temperatures=args.sc_temperatures,
                    max_tokens=args.max_tokens,
                    reasoning=args.reasoning,
                )
                if not args.test:
                    log_experiment(
                        output_path=args.output, pipeline='indicators',
                        prompt_style=args.mode, model=model_identifier,
                        indicators=args.indicators, verify=args.verify,
                        search_mode=args.search_mode, total_entries=len(df),
                        cost_summary={}, n_samples=args.n_samples,
                    )
                print("\nLeader-level pipeline (forced search + batch) completed successfully!")
                return

            else:
                # Forced search without batch: use PreSearcher (tiered) + regular Predictor
                # Same tiered search as batch path: Wikipedia → DuckDuckGo → Serper
                from pipeline.pre_search import PreSearcher
                from prompts.base_builder import PromptOutput

                config = PredictionConfig(
                    mode=PromptMode(args.mode),
                    indicators=args.indicators,
                    verify=VerificationType(args.verify),
                    verify_indicators=_verify_indicators,
                    model=model_identifier,
                    verifier_model=args.verifier_model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    sc_n_samples=args.n_samples,
                    sc_temperatures=args.sc_temperatures,
                    sequence=args.sequence,
                    random_sequence=args.random_sequence,
                    reasoning=args.reasoning,
                    use_logprobs=args.logprobs,
                )
                predictor = Predictor(config, api_keys)

                df = load_polity_data(args.input)
                if args.test:
                    df = _parse_test_argument(args.test, df)

                pre_searcher = PreSearcher(serper_api_key=api_keys.get('serper', ''))

                # Track last search result via closure for metadata extraction
                _last_search = [None]
                original_build = predictor.prompt_builder.build

                def _build_with_search_sync(polity, name, start_year, end_year):
                    search_result = pre_searcher.search(polity, name, start_year, end_year)
                    _last_search[0] = search_result
                    prompts = original_build(polity, name, start_year, end_year)
                    enriched = []
                    for p in prompts:
                        enriched_user = pre_searcher.enrich_prompt(p.user_prompt, search_result)
                        enriched.append(PromptOutput(
                            system_prompt=p.system_prompt,
                            user_prompt=enriched_user,
                            indicators=p.indicators,
                            metadata={**p.metadata,
                                      'search_queries': search_result.search_queries,
                                      'urls_used': search_result.urls_used,
                                      'sources_used': search_result.sources_used,
                                      'search_context': search_result.context},
                        ))
                    return enriched

                predictor.prompt_builder.build = _build_with_search_sync

                is_single_or_seq = args.mode in ('single', 'sequential')

                results = []
                for idx in tqdm(range(len(df)), desc="Processing (forced search)"):
                    row = df.iloc[idx]
                    polity = str(row.get(COL_TERRITORY_NAME, "Unknown"))
                    name = str(row.get(COL_LEADER_NAME, "Unknown"))
                    start_year = int(row[COL_START_YEAR])
                    end_year = None if pd.isna(row[COL_END_YEAR]) else int(row[COL_END_YEAR])

                    prediction = predictor.predict(polity, name, start_year, end_year)
                    result_dict = row.to_dict()
                    result_dict.update(prediction.to_dict())

                    # Add search metadata from the tiered pre-search
                    sr = _last_search[0]
                    if sr:
                        if sr.search_queries:
                            result_dict['search_queries'] = ' | '.join(sr.search_queries)
                        if sr.urls_used:
                            result_dict['urls_used'] = ' | '.join(sr.urls_used)
                        if is_single_or_seq and sr.context:
                            result_dict['web_information'] = sr.context

                    results.append(result_dict)
                    time.sleep(args.delay)

                results_df = pd.DataFrame(results)
                os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

                # Save CSV
                results_df.to_csv(args.output, index=False)

                # Save JSON (includes web_information for single/sequential mode)
                json_path = args.output.replace('.csv', '.json')
                records = results_df.to_dict(orient='records')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(records, f, ensure_ascii=False, indent=2, default=str)

                # Save cost report
                logs_dir = Path('data/logs')
                logs_dir.mkdir(parents=True, exist_ok=True)
                cost_path = logs_dir / f'{Path(args.output).stem}_costs.json'
                predictor.cost_tracker.save_report(str(cost_path))

                print(f"\nResults saved to: {args.output}")
                print(f"JSON saved to: {json_path}")
                predictor.cost_tracker.print_summary()
                if not args.test:
                    log_experiment(
                        output_path=args.output, pipeline='indicators',
                        prompt_style=args.mode, model=model_identifier,
                        indicators=args.indicators, verify=args.verify,
                        search_mode=args.search_mode, total_entries=len(df),
                        cost_summary=predictor.cost_tracker.get_summary(),
                        n_samples=args.n_samples,
                    )
                print("\nLeader-level pipeline (forced search) completed successfully!")
                return

        # ── Search mode: gemini_grounding ─────────────────────────────────
        # Gemini native Google Search grounding; grounding metadata stored per
        # indicator / per SC slot.  Batch path: tools config embedded at build
        # time and auto-detected by the runner.  Sync path: use_grounding=True
        # on PredictionConfig so GeminiLLM always adds the search tool.
        if search_mode == SearchMode.GEMINI_GROUNDING:
            df = load_polity_data(args.input)
            if args.test:
                df = _parse_test_argument(args.test, df)

            if args.use_batch:
                from pipeline.jsonl_batch_runner import run_inline_batch
                results_df = run_inline_batch(
                    df=df,
                    indicators=args.indicators,
                    model=model_identifier,
                    api_key=api_keys.get('gemini', ''),
                    n_samples=args.n_samples,
                    output_path=args.output,
                    sc_temperatures=args.sc_temperatures,
                    max_tokens=args.max_tokens,
                    reasoning=args.reasoning,
                    use_grounding=True,
                )
                print("\n[INFO] Cost tracking not available in Gemini Batch API mode.")
            else:
                config = PredictionConfig(
                    mode=PromptMode(args.mode),
                    indicators=args.indicators,
                    verify=VerificationType(args.verify),
                    verify_indicators=_verify_indicators,
                    model=model_identifier,
                    verifier_model=args.verifier_model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    sc_n_samples=args.n_samples,
                    sc_temperatures=args.sc_temperatures,
                    sequence=args.sequence,
                    random_sequence=args.random_sequence,
                    reasoning=args.reasoning,
                    use_logprobs=args.logprobs,
                    use_grounding=True,
                )
                predictor = Predictor(config, api_keys)
                batch_config = BatchConfig(
                    checkpoint_interval=args.checkpoint_interval,
                    delay_between_calls=args.delay,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                    max_workers=args.parallel_rows,
                )
                runner = BatchRunner(predictor=predictor, config=batch_config, output_path=args.output)
                results_df = runner.run(df)
                predictor.cost_tracker.print_summary()
                if not args.test:
                    log_experiment(
                        output_path=args.output, pipeline='indicators',
                        prompt_style=args.mode, model=model_identifier,
                        indicators=args.indicators, verify=args.verify,
                        search_mode=args.search_mode, total_entries=len(df),
                        cost_summary=predictor.cost_tracker.get_summary(),
                        n_samples=args.n_samples,
                    )
            print("\nLeader-level pipeline (Gemini grounding) completed successfully!")
            return

        # ── Search mode: none (default) ───────────────────────────────────
        # Two sub-paths: batch (Gemini Batch API) OR synchronous (BatchRunner)

        # Create prediction config
        config = PredictionConfig(
            mode=PromptMode(args.mode),
            indicators=args.indicators,
            verify=VerificationType(args.verify),
            verify_indicators=_verify_indicators,
            model=model_identifier,
            verifier_model=args.verifier_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            sc_n_samples=args.n_samples,
            sc_temperatures=args.sc_temperatures,
            sequence=args.sequence,
            random_sequence=args.random_sequence,
            reasoning=args.reasoning,
            use_logprobs=args.logprobs,
        )

        # Create predictor
        predictor = Predictor(config, api_keys)

        # Create batch config
        batch_config = BatchConfig(
            checkpoint_interval=args.checkpoint_interval,
            delay_between_calls=args.delay,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            max_workers=args.parallel_rows
        )

        # Load data
        df = load_polity_data(args.input)

        # Apply test mode if specified
        if args.test:
            df = _parse_test_argument(args.test, df)

        if args.use_batch:
            # Gemini Batch API path (no search); SC embedded in batch via n_samples
            from pipeline.jsonl_batch_runner import run_inline_batch
            results_df = run_inline_batch(
                df=df,
                indicators=args.indicators,
                model=model_identifier,
                api_key=api_keys.get('gemini', ''),
                n_samples=args.n_samples,
                output_path=args.output,
                sc_temperatures=args.sc_temperatures,
                max_tokens=args.max_tokens,
                reasoning=args.reasoning,
            )
        else:
            # Standard synchronous BatchRunner
            runner = BatchRunner(
                predictor=predictor,
                config=batch_config,
                output_path=args.output
            )
            results_df = runner.run(df)

        # Print cost summary (batch mode bypasses predictor so tracker shows $0)
        if args.use_batch:
            print("\n[INFO] Cost tracking not available in Gemini Batch API mode.")
        else:
            predictor.cost_tracker.print_summary()

        if not args.test:
            log_experiment(
                output_path=args.output, pipeline='indicators',
                prompt_style=args.mode, model=model_identifier,
                indicators=args.indicators, verify=args.verify,
                search_mode=args.search_mode, total_entries=len(df),
                cost_summary=predictor.cost_tracker.get_summary(),
                n_samples=args.n_samples,
            )
        print("\nLeader-level pipeline completed successfully!")
        return

    # ==========================================================================
    # CONSTITUTION PIPELINE PATH (legacy, constitution only)
    # ==========================================================================
    print("Pipeline: CONSTITUTION (legacy, single-model, polity level)")
    print(f"Input:    {args.input}")

    # Single model only — take the first --models argument.
    _model_arg = args.models[0]
    if '=' in _model_arg:
        _, _model_identifier = _model_arg.split('=', 1)
    else:
        _model_identifier = _model_arg
    models_dict = {"model": _model_identifier}

    # Collect LLM parameters
    llm_params = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
    }

    # Load data (polity-level: only requires territorynamehistorical, start_year, end_year)
    print(f"Loading polity data from {args.input}...")
    from utils.data_loader import load_dataframe
    df = load_dataframe(args.input)
    polity_required = [COL_TERRITORY_NAME, COL_START_YEAR, COL_END_YEAR]
    missing = [c for c in polity_required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for polity pipeline: {missing}")
    print(f"Data loaded successfully! Total polities: {len(df)}")

    # Apply test mode if specified
    if args.test:
        df = _parse_test_argument(args.test, df)

    # Apply prompt customization if specified
    global SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    if args.system_prompt:
        SYSTEM_PROMPT = args.system_prompt
    if args.user_prompt:
        USER_PROMPT_TEMPLATE = args.user_prompt

    # Create single LLM instance and verifier for verification (if requested)
    model_llms: Dict[str, Any] = {}
    verifier_llm_instance: Any = None
    print("Creating LLM instance...")
    model_llms["model"] = create_llm(_model_identifier, api_keys, use_logprobs=args.logprobs)

    if args.verify in ('cove', 'both'):
        if args.verifier_model:
            verifier_llm_instance = create_llm(args.verifier_model, api_keys)
        else:
            print("Warning: CoVe requires --verifier-model. CoVe will be skipped.")

    # Process data
    polity_cost_tracker = CostTracker()
    results_df = process_batch(
        df,
        models_dict=models_dict,
        batch_size=args.batch_size,
        delay=args.delay,
        api_keys=api_keys,
        llm_params=llm_params,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        use_search_flag=(search_mode != SearchMode.NONE),
        verify_type=args.verify,
        model_llms=model_llms,
        verifier_llm=verifier_llm_instance,
        sc_n_samples=args.n_samples,
        sc_temperatures=args.sc_temperatures,
        cove_questions_per_element=1,
        max_workers=args.parallel_rows,
        cost_tracker=polity_cost_tracker,
        output_path=args.output,
        force_search=(search_mode == SearchMode.FORCED),
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Save results
    save_results(results_df, args.output)

    if not args.test:
        log_experiment(
            output_path=args.output, pipeline='constitution',
            prompt_style='legacy', model=_model_identifier,
            indicators=['constitution'], verify=args.verify,
            search_mode=args.search_mode, total_entries=len(df),
            cost_summary=polity_cost_tracker.get_summary(),
            n_samples=args.n_samples,
        )
    print("\nConstitution pipeline completed successfully!")


if __name__ == "__main__":
    main()
