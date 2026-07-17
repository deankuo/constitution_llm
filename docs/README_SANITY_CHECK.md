# Sanity Check and Reprocessing Utilities

This module provides automated tools for identifying, reprocessing, and merging failed LLM predictions in your constitution dataset.

## Overview

When running Gemini queries on large datasets, some queries may fail or return incomplete results. The `sanity_check.py` module helps you:

1. **Identify** rows that failed sanity checks (missing confidence scores, low confidence, etc.)
2. **Reprocess** those failed rows through your LLM pipeline
3. **Merge** the reprocessed results back into your original dataset

## Quick Start

### Option 1: Complete Automated Workflow (Recommended)

```python
from utils.sanity_check import sanity_check_and_reprocess

final_df = sanity_check_and_reprocess(
    input_csv='./Dataset/llm_predictions.csv',
    output_csv='./Dataset/llm_predictions_clean.csv',
    min_length=100,  # Minimum explanation length
    models=['Gemini=gemini-2.5-pro'],
    temperature=0.0,
    # max_tokens uses DEFAULT_MAX_TOKENS (32768) from config.py if not specified
    delay=2.0
)
```

This single function call will:
- Load your dataset
- Identify all failed rows
- Reprocess them through main.py
- Merge results back
- Save the final cleaned dataset

### Option 2: Step-by-Step Control

```python
from utils.sanity_check import (
    identify_failed_rows,
    save_for_reprocessing,
    reprocess_with_main,
    merge_reprocessed_results
)

# Step 1: Identify failed rows
df = pd.read_csv('./Dataset/llm_predictions.csv')
failed_df = identify_failed_rows(df, min_length=100)

# Step 2: Save for reprocessing
save_for_reprocessing(failed_df, './Dataset/failed_rows.csv')

# Step 3: Reprocess
reprocess_with_main(
    input_csv='./Dataset/failed_rows.csv',
    output_csv='./Dataset/reprocessed.csv',
    models=['Gemini=gemini-2.5-pro']
)

# Step 4: Merge back
reprocessed_df = pd.read_csv('./Dataset/reprocessed.csv')
final_df = merge_reprocessed_results(df, reprocessed_df)
final_df.to_csv('./Dataset/final.csv', index=False)
```

### Option 3: Identification Only

```python
from utils.sanity_check import identify_failed_rows

df = pd.read_csv('./Dataset/llm_predictions.csv')
failed_df = identify_failed_rows(
    df,
    min_confidence=3.0,  # Minimum confidence score
    min_length=100,      # Minimum explanation length
    check_na=True,       # Check for missing scores
    check_negative=True  # Check for -1.0 scores
)

print(f"Found {len(failed_df)} failed rows")
```

## Sanity Check Criteria

The `identify_failed_rows()` function supports multiple criteria:

### Built-in Checks

- **`check_na`**: Identifies rows with missing `confidence_score_gemini` (NaN)
- **`check_negative`**: Identifies rows with `confidence_score_gemini == -1.0` (error indicator)
- **`min_confidence`**: Identifies rows below a confidence threshold (e.g., < 3.0)
- **`min_length`**: Identifies rows with short explanations (e.g., < 100 characters)
- **`check_uncodified`**: Identifies rows mentioning "uncodified" or "customary" constitutions

### Custom Conditions

You can add your own sanity check logic:

```python
def check_inconsistent_data(df):
    """Flag rows where constitution=1 but year is missing"""
    return (df['constitution_gemini'] == 1) & (df['constitution_year'].isna())

failed_df = identify_failed_rows(
    df,
    custom_conditions=[check_inconsistent_data]
)
```

## Configuration Options

### Sanity Check Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence_col` | str | `'confidence_score_gemini'` | Column containing confidence scores |
| `length_col` | str | `'Length'` | Column containing explanation lengths |
| `min_confidence` | float | `None` | Minimum acceptable confidence score |
| `min_length` | int | `100` | Minimum acceptable explanation length |
| `check_na` | bool | `True` | Check for missing confidence scores |
| `check_negative` | bool | `True` | Check for -1.0 confidence scores |
| `check_uncodified` | bool | `False` | Check for uncodified constitutions |

### Reprocessing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | list | `['Gemini=gemini-2.5-pro']` | LLM models to use |
| `temperature` | float | `0.0` | Temperature for generation |
| `max_tokens` | int | `8192` | Maximum tokens in response |
| `delay` | float | `2.0` | Delay between API calls (seconds) |
| `use_search` | bool | `False` | Enable web search functionality |

## Command Line Usage

You can also run the sanity check from the command line:

```bash
python -m utils.sanity_check \
  --input ./Dataset/llm_predictions.csv \
  --output ./Dataset/llm_predictions_clean.csv \
  --min-length 100 \
  --models Gemini=gemini-2.5-pro \
  --temperature 0.0 \
  --max-tokens 8192 \
  --delay 2.0
```

### Command Line Options

- `--input, -i`: Input CSV file (required)
- `--output, -o`: Output CSV file (required)
- `--min-confidence`: Minimum confidence score threshold
- `--min-length`: Minimum explanation length (default: 100)
- `--models`: Models to use (default: Gemini=gemini-2.5-pro)
- `--temperature`: Temperature for generation (default: 0.0)
- `--max-tokens`: Max tokens (default: 8192)
- `--delay`: Delay between calls (default: 2.0)
- `--use-search`: Enable web search
- `--no-cleanup`: Keep temporary files for debugging

## Example Workflows

### Example 1: Find and Review Failed Rows

```python
from utils.sanity_check import identify_failed_rows

df = pd.read_csv('./Dataset/llm_predictions.csv')
failed_df = identify_failed_rows(df)

# Review the failures
print(failed_df[['territorynamehistorical', 'confidence_score_gemini', 'Length']])

# Optionally save for manual inspection
failed_df.to_csv('./Dataset/failed_for_review.csv', index=False)
```

### Example 2: Reprocess Only High-Priority Failures

```python
from utils.sanity_check import identify_failed_rows, sanity_check_and_reprocess

# First, identify critical failures (missing data)
df = pd.read_csv('./Dataset/llm_predictions.csv')
critical_failures = identify_failed_rows(
    df,
    min_confidence=None,  # Don't filter by confidence
    min_length=None,      # Don't filter by length
    check_na=True,        # Only check for missing data
    check_negative=True   # And errors
)

# Save critical failures
critical_failures.to_csv('./Dataset/critical_failures.csv', index=False)

# Reprocess only critical failures
sanity_check_and_reprocess(
    input_csv='./Dataset/critical_failures.csv',
    output_csv='./Dataset/critical_fixed.csv'
)
```

### Example 3: Iterative Quality Improvement

```python
from utils.sanity_check import sanity_check_and_reprocess

# First pass: Fix critical errors
round1_df = sanity_check_and_reprocess(
    input_csv='./Dataset/llm_predictions.csv',
    output_csv='./Dataset/round1_clean.csv',
    min_confidence=None,
    min_length=100,
    check_na=True,
    check_negative=True
)

# Second pass: Improve low-confidence predictions
round2_df = sanity_check_and_reprocess(
    input_csv='./Dataset/round1_clean.csv',
    output_csv='./Dataset/round2_clean.csv',
    min_confidence=3.0,  # Now target low confidence scores
    check_na=False,      # Already fixed
    check_negative=False # Already fixed
)
```

## Merge Key Selection

The system intelligently selects merge keys:

1. **Primary (Recommended)**: Uses `unique_id` column if available in both datasets
   - Most reliable - guarantees unique matching
   - Faster merging
   - No issues with special characters

2. **Fallback**: Uses `['territorynamehistorical', 'start_year', 'end_year']` if `unique_id` not available
   - Works with older datasets
   - Still reliable but slower

You can override this by specifying `key_columns` explicitly:
```python
final_df = merge_reprocessed_results(
    original_df=df,
    reprocessed_df=reprocessed_df,
    key_columns=['custom_id']  # Use your own key
)
```

## How It Works

### 1. Identification Phase

The system checks each row against your specified criteria:
- Missing or NaN confidence scores
- Negative confidence scores (-1.0 indicates an error)
- Low confidence scores (below your threshold)
- Short explanations (below minimum length)
- Custom conditions you define

### 2. Reprocessing Phase

Failed rows are extracted and run through your main.py pipeline:
- Creates a temporary CSV with failed rows
- Calls `python3 main.py --input <failed_rows> --output <reprocessed>`
- Uses the same model configuration you specify

### 3. Merging Phase

Reprocessed results are intelligently merged back:
- Matches rows by key columns (territory name, start year, end year)
- Updates only the columns that were reprocessed
- Preserves original values if reprocessing fails again
- Maintains data integrity throughout

## Troubleshooting

### "No conditions specified" warning

Make sure you enable at least one check:
```python
failed_df = identify_failed_rows(
    df,
    check_na=True  # Enable at least one check!
)
```

### Reprocessing fails

1. Check that your API keys are set in `.env`
2. Try setting `cleanup_temp_files=False` to inspect intermediate files
3. Verify the failed rows CSV is valid
4. Check the console output for specific error messages

### Too many rows flagged as failed

Adjust your thresholds:
```python
failed_df = identify_failed_rows(
    df,
    min_confidence=2.0,  # More lenient (was 3.0)
    min_length=50        # More lenient (was 100)
)
```

## Best Practices

1. **Start with strict criteria**: Better to reprocess more rows than miss failures
2. **Keep temporary files initially**: Use `cleanup_temp_files=False` while debugging
3. **Review failures first**: Use `identify_failed_rows()` before committing to reprocessing
4. **Use custom conditions**: Add domain-specific checks for your data
5. **Iterate**: Run multiple passes with different thresholds for quality improvement

## Integration with Existing Workflow

This utility integrates seamlessly with your existing code:

```python
# Your existing preprocessing
df = pd.read_csv('./Dataset/polity_level_data.csv')
# ... merge with other data sources ...

# Run LLM predictions
!python3 main.py --input ./Dataset/polity_level_data.csv --output ./Dataset/llm_predictions.csv

# NEW: Sanity check and reprocess
from utils.sanity_check import sanity_check_and_reprocess
clean_df = sanity_check_and_reprocess(
    input_csv='./Dataset/llm_predictions.csv',
    output_csv='./Dataset/llm_predictions_clean.csv'
)

# Continue with your analysis
# ... create visualizations, HTML reports, etc. ...
```

## API Reference

### `identify_failed_rows()`
Identifies rows failing sanity checks.

**Returns**: DataFrame with failed rows

### `save_for_reprocessing()`
Saves failed rows to CSV.

**Returns**: None

### `reprocess_with_main()`
Reprocesses failed rows using main.py.

**Returns**: bool (success/failure)

### `merge_reprocessed_results()`
Merges reprocessed results into original dataset.

**Key Feature**: Auto-detects `unique_id` column for merging if available, otherwise falls back to `['territorynamehistorical', 'start_year', 'end_year']`

**Returns**: DataFrame with merged results

### `sanity_check_and_reprocess()`
Complete workflow in one function.

**Returns**: DataFrame with final results

## Support

For issues or questions:
1. Check the docstrings: `help(identify_failed_rows)`
2. Review the example cells in `main.ipynb`
3. Enable debug mode: `cleanup_temp_files=False`

## License

Same as parent project.
