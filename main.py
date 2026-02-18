"""
Constitution Analysis Pipeline - Polity Level

This script analyzes historical polities to determine political indicators
including constitution status, sovereignty, powersharing, assembly, appointment,
tenure, and exit patterns.

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
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Import from new module structure
from config import (
    COL_TERRITORY_NAME,
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
from utils.cost_tracker import CostTracker

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

def create_prompt(country: str, start_year: int, end_year: int) -> Tuple[str, str]:
    """
    Create system and user prompts for polity-level constitution analysis.

    Args:
        country: Country/polity name
        start_year: Start year of the polity period
        end_year: End year of the polity period

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        country=country,
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
    use_search: bool
) -> Optional[str]:
    """Route the query to the appropriate model API."""
    provider = detect_provider(model_identifier)

    if use_search:
        print(f"INFO: Using WEB SEARCH for {model_identifier} ({provider})")

        if provider == 'openai':
            return run_openai_search_agent(
                system_prompt, user_prompt, model_identifier,
                api_key=api_keys.get('openai'),
                serper_api_key=api_keys.get('serper')
            )
        elif provider == 'gemini':
            return run_gemini_search_agent(
                system_prompt, user_prompt, model_identifier,
                api_key=api_keys.get('gemini'),
                serper_api_key=api_keys.get('serper')
            )
        elif provider == 'anthropic':
            return run_anthropic_search_agent(
                system_prompt, user_prompt, model_identifier,
                api_key=api_keys.get('anthropic'),
                serper_api_key=api_keys.get('serper')
            )
        elif provider == 'bedrock':
            return run_bedrock_search_agent(
                system_prompt, user_prompt, model_identifier,
                api_keys=api_keys,
                serper_api_key=api_keys.get('serper')
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
    start_year: int,
    end_year: int,
    model_suffix: str,
    verify_type: str,
    llm: Any,
    verifier_llm: Any,
    sc_n_samples: int,
    sc_temperatures: List[float],
    cove_questions_per_element: int,
    initial_status: str,
    initial_reasoning: str,
) -> None:
    """
    Apply self-consistency and/or CoVe verification to a polity-level constitution
    prediction.  Updates `result` in-place.

    verify_type: 'self_consistency' | 'cove' | 'both'
    For 'both': SC runs first; its majority prediction feeds into CoVe as the
    initial prediction.
    """
    current_status = initial_status  # may be updated by SC before CoVe

    # ------------------------------------------------------------------
    # Self-Consistency
    # ------------------------------------------------------------------
    if verify_type in ('self_consistency', 'both'):
        sc_preds: List[float] = []
        temps = (sc_temperatures or [0.0, 0.5, 1.0])[:sc_n_samples]
        for temp in temps:
            try:
                resp = llm.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temp
                )
                parsed = parse_json_response(resp.content, verbose=False)
                validated = validate_constitution_response(parsed)
                pred = validated.get('constitution')   # 1.0 or 0.0
                if pred is not None:
                    sc_preds.append(pred)
            except Exception as e:
                print(f"  SC sample (temp={temp}) failed: {e}")

        if sc_preds:
            counter = Counter(sc_preds)
            majority_pred, majority_count = counter.most_common(1)[0]
            agreement = majority_count / len(sc_preds)
            verified_int = int(majority_pred)
            current_status = 'Yes' if verified_int else 'No'
            result[f'constitution_{model_suffix}'] = verified_int
            result[f'constitution_sc_agreement_{model_suffix}'] = round(agreement, 3)
            result[f'constitution_sc_verification_{model_suffix}'] = str({
                'method': 'self_consistency',
                'n_samples': len(sc_preds),
                'vote_distribution': {int(k): v for k, v in dict(counter).items()},
                'agreement_ratio': round(agreement, 3),
            })

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
            # Normalise initial prediction to 'Yes' / 'No'
            if current_status is not None:
                init_pred = 'Yes' if str(current_status).lower() in ('yes', '1', 'true') else 'No'
            else:
                init_pred = 'No'

            verify_result = cove.verify(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                indicator='constitution',
                valid_labels=['Yes', 'No'],
                initial_prediction=init_pred,
                initial_reasoning=initial_reasoning or '',
                polity=country,
                name='N/A',          # no single leader at polity level
                start_year=start_year,
                end_year=end_year,
            )
            verified_pred = verify_result.verified_prediction
            result[f'constitution_cove_verification_{model_suffix}'] = str(
                verify_result.verification_details
            )
            if verified_pred in ('Yes', 'No'):
                result[f'constitution_{model_suffix}'] = 1 if verified_pred == 'Yes' else 0
        except Exception as e:
            print(f"  CoVe verification failed: {e}")
            result[f'constitution_cove_verification_{model_suffix}'] = f'Error: {e}'


def process_single_polity(
    country: str,
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
) -> Optional[Dict]:
    """Process a single polity for constitution analysis (polity pipeline)."""
    system_prompt, user_prompt = create_prompt(country, start_year, end_year)

    response = _route_to_model(
        system_prompt, user_prompt, model_identifier,
        api_keys, llm_params, max_retries, retry_delay, use_search_flag
    )

    if response is None:
        return None

    parsed_result = parse_llm_response(response, max_retries, retry_delay)

    model_suffix = model_key.lower().replace("-", "_")
    status = parsed_result.get('constitution_status') or parsed_result.get('constitution')
    explanation = parsed_result.get('explanation') or parsed_result.get('reasoning', "No explanation provided")

    result = {
        f'constitution_{model_suffix}': 1 if status and str(status).lower() == 'yes' else 0,
        'constitution_year': parsed_result.get('constitution_year', None),
        f'constitution_name_{model_suffix}': parsed_result.get('document_name', "N/A"),
        f'explanation_{model_suffix}': explanation,
        f'explanation_length_{model_suffix}': len(explanation) if explanation else 0,
        f'confidence_score_{model_suffix}': parsed_result.get('confidence_score', None),
    }

    # Apply verification if requested
    if verify_type != 'none' and llm is not None:
        _apply_polity_verification(
            result=result,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            country=country,
            start_year=start_year,
            end_year=end_year,
            model_suffix=model_suffix,
            verify_type=verify_type,
            llm=llm,
            verifier_llm=verifier_llm,
            sc_n_samples=sc_n_samples,
            sc_temperatures=sc_temperatures or [0.0, 0.5, 1.0],
            cove_questions_per_element=cove_questions_per_element,
            initial_status=status,
            initial_reasoning=explanation,
        )

    return result


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
) -> pd.DataFrame:
    """Process polity data in batches (polity pipeline)."""
    results = []
    total_polities = len(df)
    checkpoint_files = []
    checkpoint_at_50_percent = total_polities // 2

    verify_info = f" | verification: {verify_type}" if verify_type != 'none' else ""
    print(f"Starting to process {total_polities} polities using models: {list(models_dict.keys())}{verify_info}")
    print(f"Checkpoint will be saved at: {checkpoint_at_50_percent} polities (50%)")

    for idx, row in tqdm(df.iterrows(), total=total_polities, desc="Processing Polities"):
        country = row[COL_TERRITORY_NAME]
        start_year = int(row[COL_START_YEAR])
        end_year = int(row[COL_END_YEAR])

        entry_result = row.to_dict()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models_dict)) as executor:
            future_to_model = {}

            for model_key, model_identifier in models_dict.items():
                llm_for_verify = (model_llms or {}).get(model_key)
                future = executor.submit(
                    process_single_polity,
                    country, start_year, end_year,
                    model_key, model_identifier,
                    api_keys, llm_params,
                    max_retries, retry_delay, use_search_flag,
                    verify_type, llm_for_verify, verifier_llm,
                    sc_n_samples, sc_temperatures, cove_questions_per_element,
                )
                future_to_model[future] = model_key

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
                    model_suffix = model_key.lower().replace("-", "_")
                    error_msg = f"Failed after retries: {exc}"
                    entry_result[f'constitution_{model_suffix}'] = -1
                    entry_result[f'constitution_name_{model_suffix}'] = "Query Failed"
                    entry_result['constitution_year'] = None
                    entry_result[f'explanation_{model_suffix}'] = error_msg
                    entry_result[f'explanation_length_{model_suffix}'] = len(error_msg)

        results.append(entry_result)

        if (idx + 1) == checkpoint_at_50_percent:
            temp_df = pd.DataFrame(results)
            temp_filename = f'checkpoint_50percent_{total_polities}polities.csv'
            temp_df.to_csv(temp_filename, index=False)
            checkpoint_files.append(temp_filename)
            print(f"\n  Checkpoint saved at 50%: {temp_filename}")

        time.sleep(delay)

    final_df = pd.DataFrame(results)

    temp_filename = f'checkpoint_100percent_{total_polities}polities.csv'
    final_df.to_csv(temp_filename, index=False)
    checkpoint_files.append(temp_filename)
    print(f"\n  Checkpoint saved at 100%: {temp_filename}")

    for checkpoint_file in checkpoint_files:
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print(f"  Deleted checkpoint file: {checkpoint_file}")
        except Exception as e:
            print(f"  Warning: Could not delete checkpoint file {checkpoint_file}: {e}")

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
            'Two pipeline levels:\n'
            '  polity  -- Legacy pipeline. Predicts constitution (Yes/No) at the polity level.\n'
            '             Input: plt_polity_data_v2.csv  (columns: territorynamehistorical, start_year, end_year)\n'
            '             Supports multiple models in parallel and web search.\n'
            '\n'
            '  leader  -- New modular pipeline. Predicts any combination of 7 indicators at the\n'
            '             leader level (one row per leader reign).\n'
            '             Input: plt_leaders_data.csv  (columns: territorynamehistorical, name, start_year, end_year)\n'
            '             Supports single / multiple / sequential prompt modes, self-consistency,\n'
            '             and Chain-of-Verification (CoVe).'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # --- POLITY level (legacy, constitution only) ---

  # Basic run with default Gemini model
  python main.py --pipeline polity

  # Multiple models, custom input/output
  python main.py --pipeline polity -i data/plt_polity_data_v2.csv -o results.csv -m GPT=gpt-4o Claude=claude-3-5-sonnet-20241022

  # Enable web search, test on first 10 rows
  python main.py --pipeline polity --models GPT=gpt-4o --use-search --test 10


  # --- LEADER level (new modular pipeline, all 7 indicators) ---

  # Predict sovereign + assembly with multiple prompts (default mode)
  python main.py --pipeline leader --indicators sovereign assembly powersharing

  # With self-consistency verification on assembly
  python main.py --pipeline leader --indicators assembly --verify self_consistency --verify-indicators assembly

  # Single prompt mode — all indicators in one LLM call
  python main.py --pipeline leader --mode single --indicators sovereign assembly exit

  # Sequential mode with a user-defined indicator order
  python main.py --pipeline leader --mode sequential --indicators constitution sovereign assembly powersharing appointment tenure exit --sequence assembly constitution sovereign exit powersharing tenure appointment

  # Sequential mode with randomised indicator order
  python main.py --pipeline leader --mode sequential --indicators constitution sovereign assembly powersharing appointment tenure exit --random-sequence

  # CoVe verification with a Bedrock verifier model
  python main.py --pipeline leader --indicators constitution --verify cove --verify-indicators constitution --verifier-model us.anthropic.claude-sonnet-4-5-20250929-v1:0
        """
    )

    # Input/Output arguments
    parser.add_argument(
        '--input', '-i',
        default=None,
        help=(
            'Input CSV file path. Defaults to the standard file for the chosen pipeline level:\n'
            '  polity → ./data/plt_polity_data_v2.csv  (requires: territorynamehistorical, start_year, end_year)\n'
            '  leader → ./data/plt_leaders_data.csv    (requires: territorynamehistorical, name, start_year, end_year)'
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
        default=['Gemini=gemini-2.5-pro'],
        help='Space-separated list of models in KEY=IDENTIFIER format'
    )
    parser.add_argument(
        '--use-search',
        action='store_true',
        help='Enable web search functionality for models'
    )

    # API configuration
    parser.add_argument(
        '--api-key', '-k',
        help='API key for OpenAI (or use OPENAI_API_KEY env var)'
    )

    # Processing configuration
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=DEFAULT_BATCH_SIZE,
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
        choices=['leader', 'polity'],
        default='polity',
        help=(
            'Pipeline level to run (default: polity).\n'
            '  polity  — Legacy pipeline: predicts constitution (Yes/No) at the polity level.\n'
            '            One row per polity period; supports multiple models in parallel.\n'
            '  leader  — New modular pipeline: predicts any of 7 indicators at the leader level.\n'
            '            One row per leader reign; supports single/multiple/sequential prompt\n'
            '            modes, self-consistency, and Chain-of-Verification (CoVe).'
        )
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'multiple', 'sequential'],
        default='multiple',
        help='Prompt mode: single (unified prompt), multiple (separate prompts), or sequential (7 indicators in sequence)'
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
        default='none',
        help='Verification method to use'
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
        default=3,
        help='Number of samples for self-consistency'
    )
    parser.add_argument(
        '--sc-temperatures',
        nargs='+',
        type=float,
        default=[0.0, 0.5, 1.0],
        help='Temperature values for self-consistency sampling'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=50,
        help='Save checkpoint every N rows (leader pipeline only)'
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
        help='Include reasoning in predictions for 6 indicators (default: True). Set to False for prediction-only output. Constitution always includes reasoning.'
    )

    args = parser.parse_args()

    # Resolve default input file based on pipeline level if not explicitly provided
    if args.input is None:
        if args.pipeline == 'leader':
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

    # Validate search configuration
    if args.use_search and not api_keys['serper']:
        raise ValueError(
            "Web search is enabled (--use-search), but no Serper API key was provided. "
            "Set the SERPER_API_KEY environment variable."
        )

    if args.pipeline == 'leader':
        print("Pipeline: LEADER level (modular, all 7 indicators supported)")
        print(f"Input:    {args.input}")

        # Parse model from --models argument (use first one for leader pipeline)
        model_arg = args.models[0]
        if '=' in model_arg:
            _, model_identifier = model_arg.split('=', 1)
        else:
            model_identifier = model_arg

        # Create prediction config
        config = PredictionConfig(
            mode=PromptMode(args.mode),
            indicators=args.indicators,
            verify=VerificationType(args.verify),
            verify_indicators=args.verify_indicators,
            model=model_identifier,
            verifier_model=args.verifier_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            sc_n_samples=args.n_samples,
            sc_temperatures=args.sc_temperatures,
            sequence=args.sequence,
            random_sequence=args.random_sequence,
            reasoning=args.reasoning
        )

        # Create predictor
        predictor = Predictor(config, api_keys)

        # Create batch config
        batch_config = BatchConfig(
            checkpoint_interval=args.checkpoint_interval,
            delay_between_calls=args.delay,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )

        # Load data
        df = load_polity_data(args.input)

        # Apply test mode if specified
        if args.test:
            df = _parse_test_argument(args.test, df)

        # Create runner and run
        runner = BatchRunner(
            predictor=predictor,
            config=batch_config,
            output_path=args.output
        )

        results_df = runner.run(df)

        # Print cost summary
        predictor.cost_tracker.print_summary()

        print("\nLeader-level pipeline completed successfully!")
        return

    # ==========================================================================
    # POLITY PIPELINE PATH (legacy, constitution only)
    # ==========================================================================
    print("Pipeline: POLITY level (legacy, constitution only)")
    print(f"Input:    {args.input}")

    # Parse model specifications
    models_dict = {}
    for model_arg in args.models:
        if '=' not in model_arg:
            raise ValueError(
                f"Invalid model format: '{model_arg}'. "
                f"Must be in KEY=IDENTIFIER format (e.g., 'GPT=gpt-4o')."
            )
        key, value = model_arg.split('=', 1)
        models_dict[key] = value

    # Collect LLM parameters
    llm_params = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
    }

    # Load data (polity-level: only requires territorynamehistorical, start_year, end_year)
    print(f"Loading polity data from {args.input}...")
    df = pd.read_csv(args.input)
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

    # Create BaseLLM instances and verifier for verification (if requested)
    model_llms: Dict[str, Any] = {}
    verifier_llm_instance: Any = None
    if args.verify != 'none':
        print(f"Verification: {args.verify} — creating LLM instances...")
        for model_key, model_identifier in models_dict.items():
            model_llms[model_key] = create_llm(model_identifier, api_keys)
        if args.verify in ('cove', 'both'):
            if args.verifier_model:
                verifier_llm_instance = create_llm(args.verifier_model, api_keys)
            else:
                print("Warning: CoVe requires --verifier-model. CoVe will be skipped.")

    # Process data
    results_df = process_batch(
        df,
        models_dict=models_dict,
        batch_size=args.batch_size,
        delay=args.delay,
        api_keys=api_keys,
        llm_params=llm_params,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        use_search_flag=args.use_search,
        verify_type=args.verify,
        model_llms=model_llms if model_llms else None,
        verifier_llm=verifier_llm_instance,
        sc_n_samples=args.n_samples,
        sc_temperatures=args.sc_temperatures,
        cove_questions_per_element=1,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Save results
    save_results(results_df, args.output)
    print("\nPolity-level pipeline completed successfully!")


if __name__ == "__main__":
    main()
