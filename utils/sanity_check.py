"""
Sanity check and reprocessing utilities for LLM predictions.

This module provides functions to:
1. Identify rows in a dataset that failed sanity checks
2. Re-run failed rows through the LLM pipeline
3. Merge reprocessed results back into the original dataset

Supports both legacy column names and new naming convention:
- Legacy: constitution_gemini, confidence_score_gemini, constitution_name_gemini, explanation_gemini
- New: {indicator}_prediction, {indicator}_confidence, {indicator}_reasoning
"""

import os
import sys
import pandas as pd
from typing import List, Optional, Callable
import warnings

# Import configuration from root config.py
# Use try/except to handle both package import and direct script execution
try:
    from config import DEFAULT_MAX_TOKENS
except ImportError:
    # When run as script, add parent directory to path
    _parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from config import DEFAULT_MAX_TOKENS

# All available indicators
ALL_INDICATORS = ['constitution', 'sovereign', 'powersharing', 'assembly', 'appointment', 'tenure', 'exit']


def fix_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix column types to ensure proper data types.

    Pandas reads NA values as float. Convert prediction/confidence columns to
    nullable integers, keep string columns as strings.

    Args:
        df: DataFrame with LLM predictions

    Returns:
        DataFrame with corrected column types
    """
    # Columns that should be strings
    string_cols = ['territorynamehistorical', 'id', 'region']

    # Reasoning and document name columns should be strings
    for col in df.columns:
        if 'reasoning' in col or 'document_name' in col:
            string_cols.append(col)

    # Convert string columns
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Convert prediction columns to nullable integers (Int64 handles NA properly)
    for indicator in ALL_INDICATORS:
        pred_col = f'{indicator}_prediction'
        if pred_col in df.columns:
            # Convert to nullable Int64 (capital I)
            df[pred_col] = pd.to_numeric(df[pred_col], errors='coerce')
            df[pred_col] = df[pred_col].astype('Int64')  # Nullable integer type

    # Convert confidence columns to nullable integers
    for indicator in ALL_INDICATORS:
        conf_col = f'{indicator}_confidence'
        if conf_col in df.columns:
            df[conf_col] = pd.to_numeric(df[conf_col], errors='coerce')
            df[conf_col] = df[conf_col].astype('Int64')  # Nullable integer type

    # Convert year columns to nullable integers
    year_cols = ['start_year', 'end_year', 'constitution_year']
    for col in year_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].astype('Int64')  # Nullable integer type

    return df


# Default sanity check configuration (new format)
DEFAULT_INDICATOR = 'constitution'
DEFAULT_CONFIDENCE_COL = 'constitution_confidence'
DEFAULT_PREDICTION_COL = 'constitution_prediction'
DEFAULT_DOCUMENT_NAME_COL = 'constitution_document_name'
DEFAULT_REASONING_COL = 'constitution_reasoning'

# Legacy column names for backward compatibility
LEGACY_CONFIDENCE_COL = 'confidence_score_gemini'
LEGACY_CONSTITUTION_COL = 'constitution_gemini'
LEGACY_CONSTITUTION_NAME_COL = 'constitution_name_gemini'
LEGACY_REASONING_COL = 'explanation_gemini'


def identify_failed_rows(
    df: pd.DataFrame,
    indicator: str = DEFAULT_INDICATOR,
    confidence_col: Optional[str] = None,
    prediction_col: Optional[str] = None,
    reasoning_col: Optional[str] = None,
    document_name_col: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_reasoning_length: Optional[int] = 100,
    check_na: bool = True,
    check_negative: bool = True,
    check_uncodified: bool = False,
    check_null_prediction: bool = True,
    custom_conditions: Optional[List[Callable]] = None,
    auto_detect_legacy: bool = True
) -> pd.DataFrame:
    """
    Identify rows that failed sanity checks based on various criteria.

    Supports both new and legacy column naming conventions.

    Args:
        df: Input DataFrame containing LLM predictions
        indicator: Indicator name (e.g., 'constitution', 'sovereign', 'assembly')
        confidence_col: Column name for confidence scores (auto-detected if None)
        prediction_col: Column name for predictions (auto-detected if None)
        reasoning_col: Column name for reasoning/explanation (auto-detected if None)
        document_name_col: Column name for constitution document name (auto-detected if None)
        min_confidence: Minimum acceptable confidence score (optional)
        min_reasoning_length: Minimum acceptable reasoning length (optional)
        check_na: Check for missing/NA confidence scores
        check_negative: Check for negative confidence scores (-1.0)
        check_uncodified: Check for uncodified/customary constitutions (constitution only)
        check_null_prediction: Check for None/null predictions
        custom_conditions: List of custom condition functions (optional)
        auto_detect_legacy: Auto-detect legacy column names if new format not found

    Returns:
        DataFrame containing only the rows that failed sanity checks
    """
    # Auto-detect column names
    if confidence_col is None:
        # Try new format first
        new_col = f'{indicator}_confidence'
        if new_col in df.columns:
            confidence_col = new_col
        elif auto_detect_legacy and LEGACY_CONFIDENCE_COL in df.columns:
            confidence_col = LEGACY_CONFIDENCE_COL
            print(f"Using legacy column name: {LEGACY_CONFIDENCE_COL}")
        else:
            warnings.warn(f"Confidence column not found for indicator '{indicator}'")

    if prediction_col is None:
        new_col = f'{indicator}_prediction'
        if new_col in df.columns:
            prediction_col = new_col
        elif auto_detect_legacy and indicator == 'constitution' and LEGACY_CONSTITUTION_COL in df.columns:
            prediction_col = LEGACY_CONSTITUTION_COL
            print(f"Using legacy column name: {LEGACY_CONSTITUTION_COL}")
        else:
            warnings.warn(f"Prediction column not found for indicator '{indicator}'")

    if reasoning_col is None:
        new_col = f'{indicator}_reasoning'
        if new_col in df.columns:
            reasoning_col = new_col
        elif auto_detect_legacy and LEGACY_REASONING_COL in df.columns:
            reasoning_col = LEGACY_REASONING_COL
            print(f"Using legacy column name: {LEGACY_REASONING_COL}")

    if document_name_col is None and indicator == 'constitution':
        new_col = 'constitution_document_name'
        if new_col in df.columns:
            document_name_col = new_col
        elif auto_detect_legacy and LEGACY_CONSTITUTION_NAME_COL in df.columns:
            document_name_col = LEGACY_CONSTITUTION_NAME_COL
            print(f"Using legacy column name: {LEGACY_CONSTITUTION_NAME_COL}")

    print(f"\nSanity checking indicator: {indicator}")
    print(f"Using columns:")
    print(f"  - Confidence: {confidence_col}")
    print(f"  - Prediction: {prediction_col}")
    print(f"  - Reasoning: {reasoning_col}")
    if document_name_col:
        print(f"  - Document name: {document_name_col}")
    print()

    conditions = []

    # Condition 1: Missing confidence scores
    if check_na and confidence_col and confidence_col in df.columns:
        condition_na = df[confidence_col].isna()
        conditions.append(condition_na)
        print(f"Found {condition_na.sum()} rows with missing confidence scores")

    # Condition 2: Negative confidence scores (indicates error)
    if check_negative and confidence_col and confidence_col in df.columns:
        condition_negative = (df[confidence_col] == -1.0)
        conditions.append(condition_negative)
        print(f"Found {condition_negative.sum()} rows with negative confidence scores")

    # Condition 3: Low confidence scores
    if min_confidence is not None and confidence_col and confidence_col in df.columns:
        condition_low_conf = (df[confidence_col] < min_confidence) & (df[confidence_col] >= 0)
        conditions.append(condition_low_conf)
        print(f"Found {condition_low_conf.sum()} rows with confidence < {min_confidence}")

    # Condition 4: Null/None predictions
    if check_null_prediction and prediction_col and prediction_col in df.columns:
        condition_null_pred = df[prediction_col].isna()
        conditions.append(condition_null_pred)
        print(f"Found {condition_null_pred.sum()} rows with null predictions")

    # Condition 5: Short reasoning (may indicate incomplete processing)
    if min_reasoning_length is not None and reasoning_col and reasoning_col in df.columns:
        # Calculate length if not already a column
        reasoning_lengths = df[reasoning_col].fillna('').astype(str).str.len()
        condition_short = reasoning_lengths < min_reasoning_length
        conditions.append(condition_short)
        print(f"Found {condition_short.sum()} rows with reasoning length < {min_reasoning_length}")

    # Condition 6: Uncodified/customary constitutions (optional, constitution only)
    if check_uncodified and indicator == 'constitution' and document_name_col and document_name_col in df.columns:
        target_names = ["customary", "uncodified", "Uncodified customary"]
        condition_uncodified = df[document_name_col].fillna('').astype(str).str.contains(
            '|'.join(target_names),
            case=False,
            na=False
        )
        conditions.append(condition_uncodified)
        print(f"Found {condition_uncodified.sum()} rows with uncodified/customary constitutions")

    # Add custom conditions
    if custom_conditions:
        for i, custom_cond in enumerate(custom_conditions):
            try:
                cond_result = custom_cond(df)
                conditions.append(cond_result)
                print(f"Custom condition {i+1}: Found {cond_result.sum()} rows")
            except Exception as e:
                warnings.warn(f"Custom condition {i+1} failed: {e}")

    # Combine all conditions with OR logic
    if not conditions:
        warnings.warn("No conditions specified. Returning empty DataFrame.")
        return df.iloc[0:0]  # Return empty DataFrame with same columns

    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition = combined_condition | condition

    failed_df = df[combined_condition].copy()

    print(f"\n{'='*60}")
    print(f"Total rows in original dataset: {len(df)}")
    print(f"Total rows failing sanity check: {len(failed_df)} ({len(failed_df)/len(df)*100:.2f}%)")
    print(f"{'='*60}\n")

    return failed_df


def save_for_reprocessing(
    failed_df: pd.DataFrame,
    output_path: str,
    required_columns: Optional[List[str]] = None
) -> None:
    """
    Save failed rows to CSV for reprocessing.

    Args:
        failed_df: DataFrame containing failed rows
        output_path: Path to save the CSV file
        required_columns: List of column names to include (optional, defaults to all)
    """
    if required_columns:
        # Ensure required columns exist
        missing_cols = [col for col in required_columns if col not in failed_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        save_df = failed_df[required_columns].copy()
    else:
        save_df = failed_df.copy()

    save_df.to_csv(output_path, index=False)
    print(f"Saved {len(save_df)} rows to {output_path} for reprocessing")


def reprocess_with_main(
    input_csv: str,
    output_csv: str,
    models: List[str] = ['Gemini=gemini-2.5-pro'],
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    delay: float = 2.0,
    use_search: bool = False,
    main_script_path: str = './main.py',
    additional_args: Optional[List[str]] = None
) -> bool:
    """
    Reprocess failed rows using the main.py script.

    Args:
        input_csv: Path to CSV file containing rows to reprocess
        output_csv: Path to save reprocessed results
        models: List of model specifications (e.g., ['Gemini=gemini-2.5-pro'])
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens for LLM response
        delay: Delay between API calls
        use_search: Whether to use web search
        main_script_path: Path to main.py script
        additional_args: Additional command-line arguments (optional)

    Returns:
        True if reprocessing succeeded, False otherwise
    """
    import subprocess

    # Build command
    cmd = [
        'python3', main_script_path,
        '--input', input_csv,
        '--output', output_csv,
        '--temperature', str(temperature),
        '--max_tokens', str(max_tokens),
        '--delay', str(delay),
        '--models'
    ] + models

    if use_search:
        cmd.append('--use-search')

    if additional_args:
        cmd.extend(additional_args)

    print(f"Running reprocessing command:")
    print(' '.join(cmd))
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)

        # Check if output file was created
        if os.path.exists(output_csv):
            print(f"\nReprocessing completed successfully!")
            print(f"Results saved to: {output_csv}")
            return True
        else:
            print(f"\nWarning: Output file {output_csv} was not created")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Error during reprocessing: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def merge_reprocessed_results(
    original_df: pd.DataFrame,
    reprocessed_df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
    update_columns: Optional[List[str]] = None,
    keep_failed_if_reprocess_fails: bool = True,
    indicator: Optional[str] = None
) -> pd.DataFrame:
    """
    Merge reprocessed results back into the original dataset.

    Args:
        original_df: Original DataFrame with all rows
        reprocessed_df: DataFrame with reprocessed results
        key_columns: Columns to use as merge keys (defaults to ['unique_id'] if available,
                    otherwise ['territorynamehistorical', 'start_year', 'end_year'])
        update_columns: Columns to update from reprocessed data (None = all non-key columns)
        keep_failed_if_reprocess_fails: If True, keep original values when reprocessing fails
        indicator: Indicator name for checking success (optional)

    Returns:
        Merged DataFrame with updated values
    """
    # Auto-detect key columns if not specified
    if key_columns is None:
        if 'unique_id' in original_df.columns and 'unique_id' in reprocessed_df.columns:
            key_columns = ['unique_id']
            print("Auto-detected 'unique_id' column for merging")
        else:
            key_columns = ['territorynamehistorical', 'start_year', 'end_year']
            print("Using fallback key columns: territorynamehistorical, start_year, end_year")

    # Validate key columns exist in both dataframes
    missing_keys_orig = [col for col in key_columns if col not in original_df.columns]
    missing_keys_reproc = [col for col in key_columns if col not in reprocessed_df.columns]

    if missing_keys_orig:
        raise ValueError(f"Key columns missing in original_df: {missing_keys_orig}")
    if missing_keys_reproc:
        raise ValueError(f"Key columns missing in reprocessed_df: {missing_keys_reproc}")

    # Determine which columns to update
    if update_columns is None:
        # Update all columns that exist in reprocessed_df but are not key columns
        update_columns = [col for col in reprocessed_df.columns if col not in key_columns]

    print(f"Merging {len(reprocessed_df)} reprocessed rows into dataset of {len(original_df)} rows")
    print(f"Using key columns: {key_columns}")
    print(f"Updating columns: {update_columns}")

    # Create copies to avoid modifying inputs
    result_df = original_df.copy()
    reprocessed_df = reprocessed_df.copy()

    # Create a unique identifier for matching
    try:
        result_df['_merge_key'] = result_df[key_columns].apply(
            lambda row: '||'.join(row.astype(str)), axis=1
        )
    except KeyError as e:
        raise ValueError(f"Key columns {key_columns} not found in original_df. Available columns: {list(original_df.columns)}") from e

    try:
        reprocessed_df['_merge_key'] = reprocessed_df[key_columns].apply(
            lambda row: '||'.join(row.astype(str)), axis=1
        )
    except KeyError as e:
        raise ValueError(f"Key columns {key_columns} not found in reprocessed_df. Available columns: {list(reprocessed_df.columns)}") from e

    # Track updates
    updated_count = 0
    failed_count = 0

    # Determine confidence column for checking success
    confidence_check_col = None
    if indicator:
        # Try new format
        new_conf_col = f'{indicator}_confidence'
        if new_conf_col in reprocessed_df.columns:
            confidence_check_col = new_conf_col
    # Fallback to legacy
    if confidence_check_col is None and LEGACY_CONFIDENCE_COL in reprocessed_df.columns:
        confidence_check_col = LEGACY_CONFIDENCE_COL

    # Update rows that were reprocessed
    for _, reproc_row in reprocessed_df.iterrows():
        merge_key = reproc_row['_merge_key']
        mask = result_df['_merge_key'] == merge_key

        if mask.any():
            # Check if reprocessing was successful
            reprocess_success = True
            if confidence_check_col and confidence_check_col in reproc_row.index:
                if pd.isna(reproc_row[confidence_check_col]) or reproc_row[confidence_check_col] == -1.0:
                    reprocess_success = False

            if reprocess_success or not keep_failed_if_reprocess_fails:
                # Update the columns
                for col in update_columns:
                    if col in reproc_row.index:
                        result_df.loc[mask, col] = reproc_row[col]
                updated_count += 1
            else:
                failed_count += 1
                print(f"Warning: Reprocessing failed for key {merge_key}, keeping original values")

    # Remove temporary merge key
    result_df = result_df.drop('_merge_key', axis=1)

    print(f"\n{'='*60}")
    print(f"Successfully updated: {updated_count} rows")
    print(f"Failed reprocessing: {failed_count} rows")
    print(f"{'='*60}\n")

    return result_df


def sanity_check_and_reprocess(
    input_csv: str,
    output_csv: str,
    indicator: str = DEFAULT_INDICATOR,
    temp_failed_csv: str = './data/temp/temp_failed_rows.csv',
    temp_reprocessed_csv: str = './data/temp/temp_reprocessed_rows.csv',
    min_confidence: Optional[float] = None,
    min_reasoning_length: Optional[int] = 100,
    mode: str = 'multiple',
    indicators: Optional[List[str]] = None,
    model: str = 'gemini-2.5-pro',
    verify: str = 'none',
    temperature: float = 0.0,
    sequence: Optional[List[str]] = None,
    random_sequence: bool = False,
    cleanup_temp_files: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Complete workflow: sanity check, reprocess, and merge results.

    Supports both legacy and new pipeline configurations.

    Args:
        input_csv: Path to input CSV file with predictions
        output_csv: Path to save final merged results
        indicator: Indicator to check (e.g., 'constitution', 'sovereign')
        temp_failed_csv: Temporary file for failed rows
        temp_reprocessed_csv: Temporary file for reprocessed results
        min_confidence: Minimum acceptable confidence score
        min_reasoning_length: Minimum acceptable reasoning length
        mode: Prompt mode ('single', 'multiple', or 'sequential')
        indicators: List of indicators to reprocess (defaults to the specified indicator)
        model: Model to use for reprocessing
        verify: Verification method ('none', 'self_consistency', 'cove', 'both')
        temperature: Temperature for LLM generation
        sequence: Indicator sequence for sequential mode (space-separated list)
        random_sequence: Randomize indicator order in sequential mode
        cleanup_temp_files: Whether to delete temporary files after completion
        **kwargs: Additional arguments passed to identify_failed_rows

    Returns:
        Final DataFrame with reprocessed results merged
    """
    print("="*80)
    print(f"STARTING SANITY CHECK AND REPROCESSING WORKFLOW (MODE: {mode.upper()})")
    print("="*80)
    print()

    # Step 1: Load original data
    print("Step 1: Loading original dataset...")
    df = pd.read_csv(input_csv)
    df = fix_column_types(df)
    print(f"Loaded {len(df)} rows from {input_csv}\n")

    # Step 2: Identify failed rows
    print("Step 2: Identifying failed rows...")

    # For SINGLE mode: Check ALL indicators and collect unique failed rows
    if mode == 'single':
        print("MODE: SINGLE - Checking all indicators for failures...")

        # Determine which indicators to check
        if indicators is None:
            # Auto-detect: check all 6 standard indicators
            check_indicators = ['sovereign', 'powersharing', 'assembly', 'appointment', 'tenure', 'exit']
            # Also check constitution if columns exist
            if 'constitution_prediction' in df.columns or 'constitution_confidence' in df.columns:
                check_indicators.insert(0, 'constitution')
        else:
            check_indicators = indicators

        print(f"Checking indicators: {check_indicators}\n")

        # Collect all failed rows across all indicators
        all_failed_indices = set()
        indicator_failures = {}

        for ind in check_indicators:
            # Special handling for constitution - only check prediction and confidence
            if ind == 'constitution':
                check_uncodified = False
            else:
                check_uncodified = kwargs.get('check_uncodified', False)

            # Identify failed rows for this indicator
            ind_failed = identify_failed_rows(
                df,
                indicator=ind,
                min_confidence=min_confidence,
                min_reasoning_length=min_reasoning_length,
                check_uncodified=check_uncodified,
                **{k: v for k, v in kwargs.items() if k != 'check_uncodified'}
            )

            if len(ind_failed) > 0:
                failed_indices = set(ind_failed.index.tolist())
                all_failed_indices.update(failed_indices)
                indicator_failures[ind] = len(failed_indices)

        print(f"\n{'='*80}")
        print("SINGLE MODE: Summary of failures by indicator")
        print(f"{'='*80}")
        for ind, count in indicator_failures.items():
            print(f"  {ind}: {count} rows")
        print(f"\nTotal unique rows with at least one failed indicator: {len(all_failed_indices)}")
        print(f"{'='*80}\n")

        # Get the combined failed dataframe
        if len(all_failed_indices) > 0:
            failed_df = df.loc[list(all_failed_indices)].copy()
        else:
            failed_df = df.iloc[0:0]  # Empty dataframe

    else:
        # For MULTIPLE mode: Check only the specified indicator
        print(f"MODE: MULTIPLE - Checking indicator: {indicator}")
        failed_df = identify_failed_rows(
            df,
            indicator=indicator,
            min_confidence=min_confidence,
            min_reasoning_length=min_reasoning_length,
            **kwargs
        )

    if len(failed_df) == 0:
        print("No failed rows found! Dataset passes all sanity checks.")
        print(f"Saving original dataset to {output_csv}")
        df.to_csv(output_csv, index=False)
        return df

    # Step 3: Save failed rows for reprocessing
    print("\nStep 3: Saving failed rows for reprocessing...")
    # Create temp directory if it doesn't exist
    temp_dir = os.path.dirname(temp_failed_csv)
    if temp_dir and not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    save_for_reprocessing(failed_df, temp_failed_csv)

    # Step 4: Reprocess failed rows using new pipeline
    print("\nStep 4: Reprocessing failed rows...")
    import subprocess

    # Build command for new pipeline
    # For single mode: reprocess ALL indicators together
    # For multiple mode: reprocess only specified indicators
    if indicators is None:
        if mode == 'single':
            # Reprocess all 6 indicators together (constitution handled separately if present)
            indicators = ['sovereign', 'powersharing', 'assembly', 'appointment', 'tenure', 'exit']
            print(f"Single mode: Will reprocess all 6 indicators together: {indicators}")
        else:
            # Multiple mode: just the checked indicator
            indicators = [indicator]
            print(f"Multiple mode: Will reprocess indicator: {indicator}")
    else:
        print(f"Using custom indicators: {indicators}")

    cmd = [
        'python3', 'main.py',
        '--new-pipeline',
        '--input', temp_failed_csv,
        '--output', temp_reprocessed_csv,
        '--mode', mode,
        '--indicators'] + indicators + [
        '--models', model,
        '--verify', verify,
        '--temperature', str(temperature),
    ]

    # Add sequential mode specific arguments
    if mode == 'sequential':
        if sequence:
            cmd.extend(['--sequence'] + sequence)
        if random_sequence:
            cmd.append('--random-sequence')

    print(f"Running reprocessing command:")
    print(' '.join(cmd))
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)

        # Check if output file was created
        if not os.path.exists(temp_reprocessed_csv):
            print(f"\nWarning: Output file {temp_reprocessed_csv} was not created")
            return df

    except subprocess.CalledProcessError as e:
        print(f"Error during reprocessing: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return df

    # Step 5: Load reprocessed results
    print("\nStep 5: Loading reprocessed results...")
    reprocessed_df = pd.read_csv(temp_reprocessed_csv)
    reprocessed_df = fix_column_types(reprocessed_df)
    print(f"Loaded {len(reprocessed_df)} reprocessed rows\n")

    # Step 6: Merge results
    print("Step 6: Merging reprocessed results with original dataset...")

    # Auto-detect which columns to update based on indicators
    update_cols = []
    for ind in indicators:
        # Add prediction, reasoning, confidence columns
        for suffix in ['_prediction', '_reasoning', '_confidence']:
            col = f'{ind}{suffix}'
            if col in reprocessed_df.columns:
                update_cols.append(col)

        # Add constitution-specific columns
        if ind == 'constitution':
            if 'constitution_document_name' in reprocessed_df.columns:
                update_cols.append('constitution_document_name')
            if 'constitution_year' in reprocessed_df.columns:
                update_cols.append('constitution_year')

    # Add cost/token columns if present
    for col in ['total_cost_usd', 'total_tokens']:
        if col in reprocessed_df.columns and col not in update_cols:
            update_cols.append(col)

    # Fallback to legacy columns if no new columns found
    if not update_cols:
        legacy_cols = [
            'constitution_gemini',
            'constitution_year',
            'constitution_name_gemini',
            'explanation_gemini',
            'confidence_score_gemini'
        ]
        update_cols = [col for col in legacy_cols if col in reprocessed_df.columns]

    print(f"Columns to update: {update_cols}")

    final_df = merge_reprocessed_results(
        df,
        reprocessed_df,
        update_columns=update_cols,
        indicator=indicator
    )

    # Fix column types after merging
    final_df = fix_column_types(final_df)

    # Step 7: Save final results
    print(f"\nStep 7: Saving final results to {output_csv}...")
    final_df.to_csv(output_csv, index=False)
    print(f"Saved {len(final_df)} rows to {output_csv}")

    # Step 8: Cleanup temporary files
    if cleanup_temp_files:
        print("\nStep 8: Cleaning up temporary files...")
        for temp_file in [temp_failed_csv, temp_reprocessed_csv]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Deleted {temp_file}")

    print("\n" + "="*80)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)

    return final_df


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description='Sanity check and reprocess LLM predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check constitution predictions
  python sanity_check.py -i data/results/predictions.csv -o data/results/fixed.csv --indicator constitution

  # Check sovereign with minimum confidence threshold
  python sanity_check.py -i data/results/predictions.csv -o data/results/fixed.csv \\
      --indicator sovereign --min-confidence 50

  # Check multiple indicators
  python sanity_check.py -i data/results/predictions.csv -o data/results/fixed.csv \\
      --indicator constitution --indicators constitution sovereign assembly
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file')
    parser.add_argument('--indicator', default='constitution',
                       choices=ALL_INDICATORS,
                       help='Primary indicator to check (default: constitution)')
    parser.add_argument('--indicators', nargs='+',
                       help='List of indicators to reprocess (default: same as --indicator)')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence score (1-100)')
    parser.add_argument('--min-reasoning-length', type=int, default=100,
                       help='Minimum reasoning length (default: 100)')
    parser.add_argument('--mode', choices=['single', 'multiple', 'sequential'], default='multiple',
                       help='Prompt mode for reprocessing: single (unified), multiple (separate), sequential (sequence)')
    parser.add_argument('--model', default='Gemini=gemini-2.5-pro', help='Model to use (default: gemini-2.5-pro)')
    parser.add_argument('--verify', choices=['none', 'self_consistency', 'cove', 'both'],
                       default='none', help='Verification method (default: none)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature (default: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS, help=f'Max tokens (default: {DEFAULT_MAX_TOKENS})')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p sampling (default: 0.95)')
    parser.add_argument('--check-null-prediction', action='store_true', default=True,
                       help='Check for null predictions (default: True)')
    parser.add_argument('--sequence', nargs='+',
                       help='Indicator sequence for sequential mode (space-separated)')
    parser.add_argument('--random-sequence', action='store_true',
                       help='Randomize indicator order in sequential mode')
    parser.add_argument('--no-cleanup', action='store_true', help='Keep temporary files')

    args = parser.parse_args()

    sanity_check_and_reprocess(
        input_csv=args.input,
        output_csv=args.output,
        indicator=args.indicator,
        indicators=args.indicators,
        min_confidence=args.min_confidence,
        min_reasoning_length=args.min_reasoning_length,
        mode=args.mode,
        model=args.model,
        verify=args.verify,
        temperature=args.temperature,
        sequence=args.sequence,
        random_sequence=args.random_sequence,
        check_null_prediction=args.check_null_prediction,
        cleanup_temp_files=not args.no_cleanup
    )
