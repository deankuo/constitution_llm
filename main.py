"""
Constitution Analysis Pipeline - Polity Level

This script analyzes historical polities to determine whether they had constitutions
during their periods of existence. It supports multiple LLM providers:
- OpenAI (GPT models)
- Google Gemini
- AWS Bedrock
- Anthropic (Claude models)

The script can optionally use web search to enhance the model's knowledge.
"""

import os
os.environ['GRPC_VERBOSITY'] = 'NONE'

import argparse
import concurrent.futures
import json
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from utils.config import (
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
    OPENAI_MODELS
)
from utils.llm_clients import (
    query_openai_model,
    query_gemini_model,
    query_anthropic_model,
    query_aws_bedrock_model
)
from utils.search_agents import (
    run_openai_search_agent,
    run_gemini_search_agent,
    run_bedrock_search_agent,
    run_anthropic_search_agent
)
from utils.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# Load environment variables from .env file
load_dotenv()


def load_polity_data(file_path: str) -> pd.DataFrame:
    """
    Load preprocessed polity data from CSV file.

    Args:
        file_path: Path to the CSV file containing polity data

    Returns:
        DataFrame with polity information

    Raises:
        ValueError: If required columns are missing from the CSV
        FileNotFoundError: If the file doesn't exist
    """
    print(f"Loading polity data from {file_path}...")

    try:
        df = pd.read_csv(file_path)

        # Validate required columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print(f"Data loaded successfully! Total polities: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data:")
        print(df.head())

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


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
    """
    Extract JSON object from LLM response that may contain additional text.

    Args:
        response: Raw LLM response

    Returns:
        Extracted JSON string or original response if no JSON found
    """
    try:
        start_index = response.find('{')
        end_index = response.rfind('}')

        if start_index != -1 and end_index != -1:
            return response[start_index:end_index + 1]
        return response
    except Exception:
        return response


def parse_llm_response(response: str, max_retries: int, retry_delay: float) -> Dict:
    """
    Parse the LLM's JSON response with retry logic.

    Args:
        response: Raw LLM response, expected to be a JSON string
        max_retries: Maximum number of retries for parsing
        retry_delay: Delay in seconds between retries

    Returns:
        Dictionary containing parsed results or error structure if parsing fails
    """
    clean_response = _extract_json_from_response(response)

    print("-" * 60)
    print(clean_response)
    print("-" * 60)

    for attempt in range(1, max_retries + 1):
        try:
            return json.loads(clean_response)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}. Attempt {attempt} of {max_retries}.")

            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds")
                time.sleep(retry_delay)

    # All retries failed
    print(f"All {max_retries} retries failed. Returning an error structure.")
    print(f"Final raw response that failed: {response}")

    return {
        'constitution_status': None,
        'constitution_name': 'Parsing Error after multiple retries',
        'constitution_year': None,
        'explanation': response
    }


def _detect_provider(model_identifier: str) -> str:
    """
    Detect the LLM provider based on model identifier.

    Args:
        model_identifier: Model name or ARN

    Returns:
        Provider name: 'openai', 'gemini', 'anthropic', or 'bedrock'
    """
    if model_identifier.startswith(BEDROCK_ARN_PREFIX):
        return 'bedrock'

    model_lower = model_identifier.lower()

    if any(pattern in model_lower for pattern in ANTHROPIC_MODELS):
        return 'anthropic'

    if any(pattern in model_lower for pattern in GEMINI_MODELS):
        return 'gemini'

    # Default to OpenAI for gpt-, o1-, o3-, and other models
    return 'openai'


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
    """
    Route the query to the appropriate model API.

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        model_identifier: The full model name or ARN
        api_keys: Dictionary of API keys
        llm_params: Dictionary of LLM inference parameters
        max_retries: Maximum number of retries
        retry_delay: Delay in seconds between retries
        use_search: Whether to use web search functionality

    Returns:
        Model response as string, or None if request fails
    """
    provider = _detect_provider(model_identifier)

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
    use_search_flag: bool
) -> Optional[Dict]:
    """
    Process a single polity by querying a model for constitution information.

    Args:
        country: Country/polity name
        start_year: Start year of the polity period
        end_year: End year of the polity period
        model_key: Short name for the model (e.g., "GPT", "Claude")
        model_identifier: Full model name or ARN
        api_keys: Dictionary of API keys
        llm_params: Dictionary of LLM inference parameters
        max_retries: Maximum number of retries for a failed API call
        retry_delay: Delay in seconds between retries
        use_search_flag: Whether to enable web search functionality

    Returns:
        Dictionary with model-specific results only, or None if query fails
    """
    system_prompt, user_prompt = create_prompt(country, start_year, end_year)

    response = _route_to_model(
        system_prompt, user_prompt, model_identifier,
        api_keys, llm_params, max_retries, retry_delay, use_search_flag
    )

    if response is None:
        return None

    parsed_result = parse_llm_response(response, max_retries, retry_delay)

    # Create column name suffix from model key
    model_suffix = model_key.lower().replace("-", "_")
    status = parsed_result.get('constitution_status')
    explanation = parsed_result.get('explanation', "No explanation provided")

    # Return only model-specific columns
    result = {
        f'constitution_{model_suffix}': 1 if status and status.lower() == 'yes' else 0,
        'constitution_year': parsed_result.get('constitution_year', None),
        f'constitution_name_{model_suffix}': parsed_result.get('document_name', "N/A"),
        f'explanation_{model_suffix}': explanation,
        f'explanation_length_{model_suffix}': len(explanation) if explanation else 0,
        f'confidence_score_{model_suffix}': parsed_result.get('confidence_score', None),
    }

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
    use_search_flag: bool = False
) -> pd.DataFrame:
    """
    Process polity data in batches using concurrent requests.

    Args:
        df: Input DataFrame with polity data
        models_dict: Dictionary mapping model keys to their identifiers
        batch_size: Number of polities to process before temporary save (deprecated, using 2 checkpoints)
        delay: Delay between processing each polity (seconds)
        api_keys: Dictionary of API keys
        llm_params: Dictionary of LLM inference parameters
        max_retries: Maximum number of retries for a failed API call
        retry_delay: Delay in seconds between retries
        use_search_flag: Whether to enable web search functionality

    Returns:
        DataFrame containing results from all models
    """
    results = []
    total_polities = len(df)
    checkpoint_files = []

    # Calculate checkpoint at 50% mark
    checkpoint_at_50_percent = total_polities // 2

    print(f"Starting to process {total_polities} polities using models: {list(models_dict.keys())}")
    print(f"Checkpoint will be saved at: {checkpoint_at_50_percent} polities (50%)")

    for idx, row in tqdm(df.iterrows(), total=total_polities, desc="Processing Polities"):
        country = row[COL_TERRITORY_NAME]
        start_year = int(row[COL_START_YEAR])
        end_year = int(row[COL_END_YEAR])

        # Initialize result entry with ALL columns from original row
        entry_result = row.to_dict()

        # Process all models concurrently for this polity
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models_dict)) as executor:
            future_to_model = {}

            # Submit jobs for each model
            for model_key, model_identifier in models_dict.items():
                future = executor.submit(
                    process_single_polity,
                    country, start_year, end_year,
                    model_key, model_identifier,
                    api_keys, llm_params,
                    max_retries, retry_delay, use_search_flag
                )
                future_to_model[future] = model_key

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                model_key = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        # Merge model-specific results
                        entry_result.update(result)
                    else:
                        raise ValueError("Query function returned None")
                except Exception as exc:
                    print(f"\nERROR processing {country} with model {model_key}: {exc}")
                    # Populate error information
                    model_suffix = model_key.lower().replace("-", "_")
                    error_msg = f"Failed after retries: {exc}"
                    entry_result[f'constitution_{model_suffix}'] = -1
                    entry_result[f'constitution_name_{model_suffix}'] = "Query Failed"
                    entry_result['constitution_year'] = None
                    entry_result[f'explanation_{model_suffix}'] = error_msg
                    entry_result[f'explanation_length_{model_suffix}'] = len(error_msg)

        results.append(entry_result)

        # Save checkpoint at 50% mark
        if (idx + 1) == checkpoint_at_50_percent:
            temp_df = pd.DataFrame(results)
            temp_filename = f'checkpoint_50percent_{total_polities}polities.csv'
            temp_df.to_csv(temp_filename, index=False)
            checkpoint_files.append(temp_filename)
            print(f"\n  Checkpoint saved at 50%: {temp_filename}")

        # Rate limiting delay between polities
        time.sleep(delay)

    # Create final DataFrame
    final_df = pd.DataFrame(results)

    # Save final checkpoint at 100%
    temp_filename = f'checkpoint_100percent_{total_polities}polities.csv'
    final_df.to_csv(temp_filename, index=False)
    checkpoint_files.append(temp_filename)
    print(f"\n  Checkpoint saved at 100%: {temp_filename}")

    # Delete all checkpoint files
    for checkpoint_file in checkpoint_files:
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print(f"  Deleted checkpoint file: {checkpoint_file}")
        except Exception as e:
            print(f"  Warning: Could not delete checkpoint file {checkpoint_file}: {e}")

    return final_df


def save_results(results_df: pd.DataFrame, output_path: str) -> None:
    """
    Save results to CSV file with summary statistics.

    Args:
        results_df: Results DataFrame
        output_path: Output file path
    """
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results saved to: {output_path}")

    # Display statistics
    print("\nPolity-Level Result Statistics:")
    print(f"Total polities processed: {len(results_df)}")

    # Show sample results
    print("\nSample results:")
    print(results_df[[COL_TERRITORY_NAME, COL_START_YEAR, COL_END_YEAR]].head())


def _parse_test_argument(test_arg: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse test argument and slice the DataFrame accordingly.

    Args:
        test_arg: Test argument string (e.g., '10' or '100:200')
        df: Input DataFrame

    Returns:
        Sliced DataFrame for testing

    Raises:
        ValueError: If the test argument format is invalid
    """
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
        description='Constitution Analysis Pipeline - Polity Level',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default Gemini model
  python main.py

  # Use multiple models with custom input/output
  python main.py -i data.csv -o results.csv -m GPT=gpt-4o Claude=claude-3-5-sonnet-20241022

  # Enable web search with GPT-4
  python main.py --models GPT=gpt-4o --use-search

  # Test mode with first 10 polities
  python main.py --test 10

  # Custom temperature and max tokens
  python main.py --temperature 0.7 --max_tokens 4096
        """
    )

    # Input/Output arguments
    parser.add_argument(
        '--input', '-i',
        default='./Dataset/polity_level_data.csv',
        help='Input CSV file path (preprocessed polity data)'
    )
    parser.add_argument(
        '--output', '-o',
        default='./Dataset/llm_predictions.csv',
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
        help='Space-separated list of models in KEY=IDENTIFIER format (e.g., "GPT=gpt-4o Claude=claude-3-5-sonnet-20241022")'
    )
    parser.add_argument(
        '--use-search',
        action='store_true',
        help='Enable web search functionality for models. Requires a Serper API key.'
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
        default=4096,
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

    args = parser.parse_args()

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

    # Load data
    df = load_polity_data(args.input)

    # Apply test mode if specified
    if args.test:
        df = _parse_test_argument(args.test, df)

    # Apply prompt customization if specified
    if args.system_prompt:
        global SYSTEM_PROMPT
        SYSTEM_PROMPT = args.system_prompt
    if args.user_prompt:
        global USER_PROMPT_TEMPLATE
        USER_PROMPT_TEMPLATE = args.user_prompt

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
        use_search_flag=args.use_search
    )

    # Save results
    save_results(results_df, args.output)
    print("\nPolity-level processing completed successfully!")


if __name__ == "__main__":
    main()
