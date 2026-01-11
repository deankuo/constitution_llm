"""
Sanity check and reprocessing utilities for Gemini LLM predictions.

This module provides functions to:
1. Identify rows in a dataset that failed sanity checks
2. Re-run failed rows through the LLM pipeline
3. Merge reprocessed results back into the original dataset
"""

import os
import pandas as pd
from typing import List, Optional, Callable
import warnings


# Default sanity check configuration
DEFAULT_CONFIDENCE_COL = 'confidence_score_gemini'
DEFAULT_CONSTITUTION_COL = 'constitution_gemini'
DEFAULT_CONSTITUTION_NAME_COL = 'constitution_name_gemini'
DEFAULT_LENGTH_COL = 'Length'


def identify_failed_rows(
    df: pd.DataFrame,
    confidence_col: str = DEFAULT_CONFIDENCE_COL,
    length_col: Optional[str] = DEFAULT_LENGTH_COL,
    constitution_name_col: Optional[str] = DEFAULT_CONSTITUTION_NAME_COL,
    constitution_col: Optional[str] = DEFAULT_CONSTITUTION_COL,
    min_confidence: Optional[float] = None,
    min_length: Optional[int] = 100,
    check_na: bool = True,
    check_negative: bool = True,
    check_uncodified: bool = False,
    custom_conditions: Optional[List[Callable]] = None
) -> pd.DataFrame:
    """
    Identify rows that failed sanity checks based on various criteria.

    Args:
        df: Input DataFrame containing LLM predictions
        confidence_col: Column name for confidence scores
        length_col: Column name for explanation length (optional)
        constitution_name_col: Column name for constitution names (optional)
        constitution_col: Column name for constitution status (optional)
        min_confidence: Minimum acceptable confidence score (optional)
        min_length: Minimum acceptable explanation length (optional)
        check_na: Check for missing/NA confidence scores
        check_negative: Check for negative confidence scores (-1.0)
        check_uncodified: Check for uncodified/customary constitutions
        custom_conditions: List of custom condition functions (optional)

    Returns:
        DataFrame containing only the rows that failed sanity checks
    """
    conditions = []

    # Condition 1: Missing confidence scores
    if check_na and confidence_col in df.columns:
        condition_na = df[confidence_col].isna()
        conditions.append(condition_na)
        print(f"Found {condition_na.sum()} rows with missing confidence scores")

    # Condition 2: Negative confidence scores (indicates error)
    if check_negative and confidence_col in df.columns:
        condition_negative = (df[confidence_col] == -1.0)
        conditions.append(condition_negative)
        print(f"Found {condition_negative.sum()} rows with negative confidence scores")

    # Condition 3: Low confidence scores
    if min_confidence is not None and confidence_col in df.columns:
        condition_low_conf = (df[confidence_col] < min_confidence) & (df[confidence_col] >= 0)
        conditions.append(condition_low_conf)
        print(f"Found {condition_low_conf.sum()} rows with confidence < {min_confidence}")

    # Condition 4: Short explanations (may indicate incomplete processing)
    if min_length is not None and length_col and length_col in df.columns:
        condition_short = df[length_col] < min_length
        conditions.append(condition_short)
        print(f"Found {condition_short.sum()} rows with explanation length < {min_length}")

    # Condition 5: Uncodified/customary constitutions (optional)
    if check_uncodified and constitution_name_col and constitution_name_col in df.columns:
        target_names = ["customary", "uncodified", "Uncodified customary"]
        condition_uncodified = df[constitution_name_col].str.contains(
            '|'.join(target_names),
            case=False,
            na=False
        )
        conditions.append(condition_uncodified)
        print(f"Found {condition_uncodified.sum()} rows with uncodified/customary constitutions")

    # Condition 6: Constitution status is -1 (error)
    if check_negative and constitution_col and constitution_col in df.columns:
        condition_const_error = (df[constitution_col] == -1.0)
        conditions.append(condition_const_error)
        print(f"Found {condition_const_error.sum()} rows with constitution status = -1")

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
    max_tokens: int = 8192,
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
    keep_failed_if_reprocess_fails: bool = True
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

    # Update rows that were reprocessed
    for _, reproc_row in reprocessed_df.iterrows():
        merge_key = reproc_row['_merge_key']
        mask = result_df['_merge_key'] == merge_key

        if mask.any():
            # Check if reprocessing was successful
            reprocess_success = True
            if 'confidence_score_gemini' in reproc_row:
                if pd.isna(reproc_row['confidence_score_gemini']) or reproc_row['confidence_score_gemini'] == -1.0:
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
    temp_failed_csv: str = './Dataset/temp_failed_rows.csv',
    temp_reprocessed_csv: str = './Dataset/temp_reprocessed_rows.csv',
    confidence_col: str = DEFAULT_CONFIDENCE_COL,
    min_confidence: Optional[float] = None,
    min_length: Optional[int] = 100,
    models: List[str] = ['Gemini=gemini-2.5-pro'],
    temperature: float = 0.0,
    max_tokens: int = 8192,
    delay: float = 2.0,
    use_search: bool = False,
    cleanup_temp_files: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Complete workflow: sanity check, reprocess, and merge results.

    Args:
        input_csv: Path to input CSV file with predictions
        output_csv: Path to save final merged results
        temp_failed_csv: Temporary file for failed rows
        temp_reprocessed_csv: Temporary file for reprocessed results
        confidence_col: Column name for confidence scores
        min_confidence: Minimum acceptable confidence score
        min_length: Minimum acceptable explanation length
        models: List of model specifications
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens for LLM response
        delay: Delay between API calls
        use_search: Whether to use web search
        cleanup_temp_files: Whether to delete temporary files after completion
        **kwargs: Additional arguments passed to identify_failed_rows

    Returns:
        Final DataFrame with reprocessed results merged
    """
    print("="*80)
    print("STARTING SANITY CHECK AND REPROCESSING WORKFLOW")
    print("="*80)
    print()

    # Step 1: Load original data
    print("Step 1: Loading original dataset...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}\n")

    # Step 2: Identify failed rows
    print("Step 2: Identifying failed rows...")
    failed_df = identify_failed_rows(
        df,
        confidence_col=confidence_col,
        min_confidence=min_confidence,
        min_length=min_length,
        **kwargs
    )

    if len(failed_df) == 0:
        print("No failed rows found! Dataset passes all sanity checks.")
        print(f"Saving original dataset to {output_csv}")
        df.to_csv(output_csv, index=False)
        return df

    # Step 3: Save failed rows for reprocessing
    print("\nStep 3: Saving failed rows for reprocessing...")
    save_for_reprocessing(failed_df, temp_failed_csv)

    # Step 4: Reprocess failed rows
    print("\nStep 4: Reprocessing failed rows...")
    success = reprocess_with_main(
        input_csv=temp_failed_csv,
        output_csv=temp_reprocessed_csv,
        models=models,
        temperature=temperature,
        max_tokens=max_tokens,
        delay=delay,
        use_search=use_search
    )

    if not success:
        print("\nWarning: Reprocessing encountered errors.")
        print(f"You may need to manually check {temp_failed_csv} and {temp_reprocessed_csv}")
        return df

    # Step 5: Load reprocessed results
    print("\nStep 5: Loading reprocessed results...")
    reprocessed_df = pd.read_csv(temp_reprocessed_csv)
    print(f"Loaded {len(reprocessed_df)} reprocessed rows\n")

    # Step 6: Merge results
    print("Step 6: Merging reprocessed results with original dataset...")

    # Determine which columns to update (only LLM output columns)
    update_cols = [
        'constitution_gemini',
        'constitution_year',
        'constitution_name_gemini',
        'explanation_gemini',
        'explanation_length_gemini',
        'confidence_score_gemini'
    ]
    # Filter to only columns that exist in reprocessed data
    update_cols = [col for col in update_cols if col in reprocessed_df.columns]

    final_df = merge_reprocessed_results(
        df,
        reprocessed_df,
        update_columns=update_cols
    )

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

    parser = argparse.ArgumentParser(description='Sanity check and reprocess LLM predictions')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence score')
    parser.add_argument('--min-length', type=int, default=100, help='Minimum explanation length')
    parser.add_argument('--models', nargs='+', default=['Gemini=gemini-2.5-pro'], help='Models to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature')
    parser.add_argument('--max-tokens', type=int, default=8192, help='Max tokens')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between calls')
    parser.add_argument('--use-search', action='store_true', help='Use web search')
    parser.add_argument('--no-cleanup', action='store_true', help='Keep temporary files')

    args = parser.parse_args()

    sanity_check_and_reprocess(
        input_csv=args.input,
        output_csv=args.output,
        min_confidence=args.min_confidence,
        min_length=args.min_length,
        models=args.models,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        delay=args.delay,
        use_search=args.use_search,
        cleanup_temp_files=not args.no_cleanup
    )
