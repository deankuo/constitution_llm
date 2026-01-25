"""
Data Cleaning Script for PLT Leaders Dataset

This script converts leader-level data to polity-level data by aggregating
6 indicators: Assembly, Exit, Powersharing, Appointment, Tenure, and Sovereignty.

The cleaning logic:
1. Capitalize the first character of territorynamehistorical (polity name)
2. Group data by polity (territorynamehistorical only)
3. Aggregate region and 6 indicators at the polity level with specific rules:

   - region: Take the first non-missing region value, or NA if all are missing

   - assembly_binary: If ANY leader has 1, code polity as 1, else 0
   - exit: If ANY leader has 1, code polity as 1, else 0
   - powersharing: If ANY leader has 1, code polity as 1, else 0
   - sovereign: If ANY leader has 1, code polity as 1, else 0

   - appointment: Take the HIGHEST value among leaders
     * If any leader has 2, code polity as 2
     * If highest is 1, code polity as 1
     * Else code as 0

   - tenure: If the HIGHEST value among leaders is < 5, code polity as 0, if 5-10, code as 1, if > 10, code as 2

4. Calculate additional statistics:
   - Count of leaders per polity
   - Year range (earliest entry year to latest exit year)
5. Add UUID as unique identifier for each polity

Usage:
    Command line:
        python utils/data_cleaner.py --input Dataset/plt_leaders_for_llm_20260120.csv --output Dataset/cleaned_data.csv

    In Jupyter notebook:
        from utils.data_cleaner import clean_data, read_cleaned_data

        # Clean the data
        clean_data(input_path="Dataset/plt_leaders_for_llm_20260120.csv",
                   output_path="Dataset/cleaned_data.csv")

        # Read the cleaned data back with correct types
        df = read_cleaned_data("Dataset/cleaned_data.csv")

Note:
    CSV format cannot preserve nullable integer types (Int64). When reading a cleaned
    CSV file, use read_cleaned_data() instead of pd.read_csv() to restore correct types.
"""

import pandas as pd
import argparse
import os
import uuid
from typing import Optional


def capitalize_polity_name(name: str) -> str:
    """
    Capitalize the first character of the polity name.

    Args:
        name: The polity name (territorynamehistorical)

    Returns:
        Polity name with first character capitalized
    """
    if isinstance(name, str) and len(name) > 0:
        return name[0].upper() + name[1:]
    return name


def aggregate_tenure(series):
    """
    Aggregate tenure indicator for a polity.
    If the highest tenure value among leaders is < 5, code as 0, if 5-10, code as 1, if > 10, code as 2.
    If all values are missing, return NaN (not 0).

    Args:
        series: pandas Series containing tenure values

    Returns:
        2 if max tenure > 10, 1 if max tenure is between 5 and 10, 0 if max <= 5, NaN if all values are missing
    """
    # Filter out NaN values
    valid_values = series.dropna()

    # If all values are NaN, return NaN (not 0)
    if len(valid_values) == 0:
        return pd.NA

    max_tenure = valid_values.max()
    if max_tenure > 10:
        return 2
    elif max_tenure >= 5:
        return 1
    else:
        return 0


def aggregate_binary_indicator(series):
    """
    Aggregate binary indicators (assembly_binary, exit, powersharing, sovereign).
    If ANY leader has a value of 1, code polity as 1, else 0.
    If all values are missing, return NaN (not 0).

    Args:
        series: pandas Series containing binary indicator values

    Returns:
        1 if any value >= 1, 0 if all values are 0, NaN if all values are missing
    """
    # Filter out NaN values
    valid_values = series.dropna()

    # If all values are NaN, return NaN (not 0)
    if len(valid_values) == 0:
        return pd.NA

    max_val = valid_values.max()
    return 1 if max_val >= 1 else 0


def aggregate_appointment(series):
    """
    Aggregate appointment indicator for a polity.
    Take the highest value: if any leader has 2, code as 2;
    if highest is 1, code as 1; else 0.
    If all values are missing, return NaN (not 0).

    Args:
        series: pandas Series containing appointment values

    Returns:
        The maximum appointment value (0, 1, or 2), or NaN if all values are missing
    """
    # Filter out NaN values
    valid_values = series.dropna()

    # If all values are NaN, return NaN (not 0)
    if len(valid_values) == 0:
        return pd.NA

    max_val = valid_values.max()
    return int(max_val)


def aggregate_region(series):
    """
    Aggregate region for a polity by taking the first non-missing value.
    If all values are missing, return NaN.

    Args:
        series: pandas Series containing region values

    Returns:
        The first non-missing region value, or NaN if all values are missing
    """
    # Filter out NaN values
    valid_values = series.dropna()

    # If all values are NaN, return NaN
    if len(valid_values) == 0:
        return pd.NA

    # Return the first non-missing value
    return valid_values.iloc[0]


def read_cleaned_data(file_path: str) -> pd.DataFrame:
    """
    Read cleaned polity-level data from CSV with correct data types.

    This function reads a CSV file saved by clean_data() and ensures all
    indicator columns are converted back to nullable integer type (Int64).
    This is necessary because CSV format doesn't preserve nullable integer types.

    Args:
        file_path: Path to the cleaned CSV file

    Returns:
        DataFrame with correct data types (Int64 for indicators)
    """
    # Read the CSV
    df = pd.read_csv(file_path)

    # Convert indicator columns back to Int64 (nullable integer)
    int_columns = [
        'start_year', 'end_year', 'leader_count',
        'assembly_binary', 'exit', 'powersharing',
        'sovereign', 'appointment', 'tenure', 'year_range'
    ]

    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype('Int64')

    return df


def clean_data(input_path: str, output_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Clean and aggregate leader-level data to polity-level data.

    This function reads the leader-level dataset, processes it by capitalizing
    polity names, and aggregates the region and 6 indicators (assembly_binary, exit,
    powersharing, appointment, tenure, sovereign) at the polity level using
    specific aggregation rules for each field.

    Aggregation Rules:
    - region: Take the first non-missing value, or NA if all are missing
    - assembly_binary, exit, powersharing, sovereign: If ANY leader has 1, polity = 1, else 0
    - appointment: Take the highest value (2 if any leader has 2, 1 if highest is 1, else 0)
    - tenure: If max value < 5, polity = 0, if 5-10, polity = 1, if > 10, polity = 2

    Args:
        input_path: Path to the input CSV file containing leader-level data
        output_path: Path where the cleaned polity-level data will be saved
        verbose: If True, print progress and statistics

    Returns:
        DataFrame containing the cleaned polity-level data

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required columns are missing from the input data
    """

    # Step 1: Load the data
    if verbose:
        print(f"Loading data from: {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if verbose:
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        print(f"\nOriginal columns: {df.columns.tolist()}")

    # Step 2: Verify required columns exist
    required_columns = [
        'territorynamehistorical', 'region', 'name_clean',
        'entrydateyear', 'exitdateyear', 'powersharing',
        'sovereign', 'assembly_binary', 'appointment', 'tenure', 'exit'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Step 3: Capitalize the first character of territorynamehistorical (polity name)
    if verbose:
        print("\nCapitalizing polity names")

    df['territorynamehistorical'] = df['territorynamehistorical'].apply(capitalize_polity_name)

    # Step 4: Prepare aggregation with specific rules for each indicator
    if verbose:
        print("\nAggregating 6 indicators at polity level with custom rules")
        print("Rules:")
        print("  - region: take first non-missing value")
        print("  - assembly_binary, exit, powersharing, sovereign: if any leader = 1, polity = 1")
        print("  - appointment: take highest value (0, 1, or 2)")
        print("  - tenure: if max < 5, polity = 0; if 5-10, polity = 1; if > 10, polity = 2")

    # Step 5: Group by polity, then aggregate with specific rules
    aggregation_dict = {
        'name_clean': 'count',  # Count of leaders
        'region': aggregate_region,  # First non-missing region value
        'entrydateyear': 'min',  # Earliest entry year
        'exitdateyear': 'max',   # Latest exit year
        # Binary indicators: if any leader has 1, polity = 1
        'assembly_binary': aggregate_binary_indicator,
        'exit': aggregate_binary_indicator,
        'powersharing': aggregate_binary_indicator,
        'sovereign': aggregate_binary_indicator,
        # Appointment: take highest value
        'appointment': aggregate_appointment,
        # Tenure: if max > 5, polity = 1
        'tenure': aggregate_tenure
    }

    # Group by territorynamehistorical only (not by region to avoid aggregation errors)
    polity_data = df.groupby(['territorynamehistorical'], dropna=False).agg(
        aggregation_dict
    ).reset_index()

    # Step 6: Rename columns for clarity
    polity_data.rename(columns={
        'name_clean': 'leader_count',
        'entrydateyear': 'start_year',
        'exitdateyear': 'end_year'
    }, inplace=True)

    # Step 7: Calculate year range
    polity_data['year_range'] = polity_data['end_year'] - polity_data['start_year']
    
    # Step 7.5: Convert indicator columns to nullable integer type (Int64)
    # This ensures all indicators are integers or NA (not float)
    if verbose:
        print("\nConverting indicator columns to nullable integer type")

    indicator_columns = ['start_year', 'end_year', 'leader_count', 'assembly_binary', 'exit', 'powersharing', 'sovereign', 'appointment', 'tenure', 'year_range']
    for col in indicator_columns:
        polity_data[col] = polity_data[col].astype('Int64')

    # Step 8: Add UUID as unique identifier for each polity
    if verbose:
        print("\nGenerating UUID for each polity")

    polity_data.insert(0, 'id', [str(uuid.uuid4()) for _ in range(len(polity_data))])

    # Step 9: Save the cleaned data
    if verbose:
        print(f"\nSaving cleaned data to: {output_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    polity_data.to_csv(output_path, index=False)

    if verbose:
        print(f"\nCleaning complete!")
        print(f"Original records (leader-level): {len(df)}")
        print(f"Cleaned records (polity-level): {len(polity_data)}")
        print(f"\nCleaned data columns: {polity_data.columns.tolist()}")
        print(f"\nSample of cleaned data:")
        print(polity_data.head())
        print(f"\nIndicator value distributions:")
        for indicator in ['assembly_binary', 'exit', 'powersharing', 'sovereign', 'appointment', 'tenure']:
            print(f"  {indicator}: {polity_data[indicator].value_counts().to_dict()}")
        print(f"\n{'='*60}")
        print("IMPORTANT: CSV format cannot preserve Int64 types.")
        print("When reading this CSV file, use:")
        print("  from utils.data_cleaner import read_cleaned_data")
        print(f"  df = read_cleaned_data('{output_path}')")
        print("This will restore correct integer types for all indicators.")
        print(f"{'='*60}")

    return polity_data


def main():
    """
    Main function for command-line usage.

    Parses command-line arguments and executes the data cleaning process.
    """
    parser = argparse.ArgumentParser(
        description='Clean and aggregate leader-level data to polity-level data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python utils/data_cleaner.py --input Dataset/plt_leaders_for_llm_20260120.csv --output Dataset/cleaned_data.csv
    python utils/data_cleaner.py -i data.csv -o output.csv --quiet
    """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the input CSV file containing leader-level data'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path where the cleaned polity-level data will be saved'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages (default: False)'
    )

    args = parser.parse_args()

    # Execute cleaning
    try:
        clean_data(
            input_path=args.input,
            output_path=args.output,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        raise


if __name__ == '__main__':
    main()
