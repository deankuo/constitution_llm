# Evaluation Guide

Guide for evaluating model predictions in Jupyter notebooks.

## Quick Start

```python
# In your Jupyter notebook
import sys
sys.path.append('/Users/deankuo/Desktop/python/constitution_llm')

from evaluation import quick_eval

# Run complete evaluation
df, summary = quick_eval('data/test_sample_5_prediction.csv')
```

## Basic Usage

### 1. Load and Evaluate All Indicators

```python
from evaluation import (
    load_predictions,
    evaluate_all_indicators,
    calculate_polity_accuracy,
    plot_accuracy_comparison
)

# Load predictions
df = load_predictions('data/results/predictions.csv')

# Add polity-level accuracy
df = calculate_polity_accuracy(df)

# Evaluate all indicators
summary = evaluate_all_indicators(df, show_plots=True)

# Show comparison plot
plot_accuracy_comparison(df)
```

### 2. Evaluate Single Indicator

```python
from evaluation import evaluate_indicator

# Evaluate one indicator with confusion matrix
metrics = evaluate_indicator(
    df,
    indicator='sovereign',
    show_confusion_matrix=True,
    show_report=True
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")
```

### 3. Filter by Time Period

```python
from evaluation import filter_data, evaluate_all_indicators

# Filter to 19th-20th century
df_filtered = filter_data(
    df,
    start_year=(1800, 2000),  # Start year range
    end_year=(1850, 2050)     # End year range
)

# Evaluate filtered data
summary = evaluate_all_indicators(df_filtered)
```

### 4. Filter by Region

```python
# Filter to specific regions
df_asia = filter_data(
    df,
    region_v1=['Asia']  # Filter by region_v1
)

df_europe_africa = filter_data(
    df,
    region_v2=['Western Europe', 'Northern Africa']  # Filter by region_v2
)

# Evaluate
summary_asia = evaluate_all_indicators(df_asia)
```

### 5. Combined Filters

```python
# Complex filtering
df_subset = filter_data(
    df,
    start_year=(1900, 2000),
    region_v1=['Europe', 'Asia']
)

summary = evaluate_all_indicators(df_subset)
```

## Advanced Usage

### Evaluate Specific Indicators Only

```python
# Evaluate only binary indicators
binary_indicators = ['sovereign', 'powersharing', 'assembly', 'exit']
summary = evaluate_all_indicators(df, indicators=binary_indicators)

# Evaluate only multi-class indicators
multiclass_indicators = ['appointment', 'tenure']
summary = evaluate_all_indicators(df, indicators=multiclass_indicators)
```

### Access Polity-Level Accuracy

```python
# Calculate polity accuracy
df = calculate_polity_accuracy(df)

# View polity accuracy
print(df[['territorynamehistorical', 'polity_accuracy',
          'polity_correct_predictions', 'polity_total_indicators']].head(10))

# Get statistics
print(f"Mean polity accuracy: {df['polity_accuracy'].mean():.3f}")
print(f"Median polity accuracy: {df['polity_accuracy'].median():.3f}")

# Find best/worst polities
best_polities = df.nlargest(10, 'polity_accuracy')
worst_polities = df.nsmallest(10, 'polity_accuracy')
```

### Custom Confusion Matrix Size

```python
# Larger confusion matrix
metrics = evaluate_indicator(
    df,
    indicator='appointment',  # 3-class indicator
    show_confusion_matrix=True,
    figsize=(10, 8)  # Larger plot
)
```

### Evaluate Without Plots

```python
# Get metrics without visualization (faster)
summary = evaluate_all_indicators(df, show_plots=False)
print(summary)
```

## Complete Workflow Example

```python
import sys
sys.path.append('/Users/deankuo/Desktop/python/constitution_llm')

from evaluation import *
import pandas as pd

# 1. Load data
df = load_predictions('data/results/predictions.csv')
print(f"Loaded {len(df)} polities")

# 2. Filter to modern era, European polities
df_filtered = filter_data(
    df,
    start_year=(1800, 2000),
    region_v1=['Europe']
)
print(f"Filtered to {len(df_filtered)} polities")

# 3. Add polity-level accuracy
df_filtered = calculate_polity_accuracy(df_filtered)

# 4. Evaluate all indicators
print("\n" + "="*60)
print("EVALUATING ALL INDICATORS")
print("="*60)
summary = evaluate_all_indicators(df_filtered, show_plots=True)

# 5. Show comparison
plot_accuracy_comparison(df_filtered)

# 6. Display best/worst polities
print("\n📈 Top 10 Best Predictions:")
print(df_filtered.nlargest(10, 'polity_accuracy')[
    ['territorynamehistorical', 'start_year', 'end_year', 'polity_accuracy']
])

print("\n📉 Bottom 10 Worst Predictions:")
print(df_filtered.nsmallest(10, 'polity_accuracy')[
    ['territorynamehistorical', 'start_year', 'end_year', 'polity_accuracy']
])

# 7. Save results
df_filtered.to_csv('data/results/evaluated_predictions.csv', index=False)
summary.to_csv('data/results/summary_metrics.csv', index=False)
```

## Understanding the Metrics

### For Binary Indicators (2-class)
- **Accuracy**: Overall correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: 2x2 matrix showing TP, TN, FP, FN

### For Multi-class Indicators (3-class)
- **Accuracy**: Overall correct predictions
- **F1 Macro**: Average F1 across all classes (treats all classes equally)
- **F1 Weighted**: Weighted average F1 by class frequency
- **Confusion Matrix**: 3x3 matrix showing all class combinations

### Polity-Level Metrics
- **polity_correct_predictions**: Number of correct indicators for this polity
- **polity_total_indicators**: Total indicators evaluated (max 6, excluding constitution)
- **polity_accuracy**: Ratio of correct predictions (0.0 to 1.0)

## Output Examples

### Confusion Matrix
Shows actual vs predicted classifications:
```
              Predicted
              0    1
Actual   0   45    5
         1    3   47
```

### Summary Table
```
indicator    accuracy  f1_macro  n_samples  n_classes
sovereign      0.920     0.918        100          2
assembly       0.850     0.845        100          2
appointment    0.780     0.765        100          3
tenure         0.720     0.710        100          3
```

### Polity Accuracy
```
territorynamehistorical  polity_accuracy  polity_correct  polity_total
Roman Republic                    1.000                6             6
Kingdom of France                 0.833                5             6
Ottoman Empire                    0.667                4             6
```

## Tips

1. **Always filter data first** before evaluation to focus on specific subsets
2. **Use `quick_eval()`** for fastest workflow in notebooks
3. **Check `n_samples`** in metrics to ensure sufficient data
4. **3-class indicators** (appointment, tenure) are typically harder - use F1 macro
5. **Polity accuracy** gives you per-polity performance - useful for finding edge cases
6. **Save results** to CSV for further analysis in Excel/R

## Available Indicators

### Binary (2-class)
- `sovereign`: Independent (1) vs Colony/vassal (0)
- `powersharing`: Multiple leaders (1) vs Single leader (0)
- `assembly`: Assembly exists (1) vs No assembly (0)
- `exit`: Regular exit (1) vs Irregular (0)

### Multi-class (3-class)
- `appointment`: Direct election (2), Council appointment (1), Hereditary/force (0)
- `tenure`: >10 years (2), 5-10 years (1), <5 years (0)

### No Ground Truth
- `constitution`: Not evaluated automatically (no ground truth available)

## Multi-Dataset Comparison

Compare multiple experiments (e.g., baseline vs self-consistency vs CoVe) with side-by-side visualizations.

### Quick Start

#### Option 1: Using File Paths

```python
from evaluation import compare_experiments

# Define your datasets (file paths)
datasets = {
    'Baseline': 'data/results/baseline.csv',
    'Self-Consistency': 'data/results/self_consistency.csv',
    'CoVe': 'data/results/cove.csv'
}

# Run comparison
binary_metrics, multiclass_metrics = compare_experiments(datasets)
```

#### Option 2: Using DataFrames (Already Loaded in Notebook)

```python
import pandas as pd
from evaluation import compare_experiments

# Load datasets in notebook (gives you control over filtering/processing)
df_baseline = pd.read_csv('data/results/baseline.csv')
df_sc = pd.read_csv('data/results/self_consistency.csv')
df_cove = pd.read_csv('data/results/cove.csv')

# Pass DataFrames directly
datasets = {
    'Baseline': df_baseline,
    'Self-Consistency': df_sc,
    'CoVe': df_cove
}

# Run comparison
binary_metrics, multiclass_metrics = compare_experiments(datasets)
```

#### Option 3: Mixed (File Paths + DataFrames)

```python
from evaluation import compare_experiments, load_predictions, filter_data

# Load and filter one dataset in notebook
df_baseline = load_predictions('data/results/baseline.csv')
df_europe = filter_data(df_baseline, region_v1=['Europe'])

# Mix with file paths
datasets = {
    'Full Dataset': 'data/results/baseline.csv',
    'Europe Only': df_europe,  # Pre-filtered DataFrame
    'CoVe Full': 'data/results/cove.csv'
}

# Run comparison
binary_metrics, multiclass_metrics = compare_experiments(datasets)
```

**What happens:**
1. Loads/uses all datasets and validates they have the same indicators
2. Computes metrics for each dataset
3. Prints summary tables
4. Generates 2 comparison plots automatically

### Plot 1: Binary Indicators (4 Subplots)

Shows **sovereign, powersharing, assembly, exit** with 4 metrics:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

Each subplot has:
- X-axis: Indicators (sovereign, powersharing, assembly, exit)
- Y-axis: Metric value (0.0 to 1.0)
- Bars: Different colors for each dataset

### Plot 2: Multi-Class Indicators (3 Metrics)

Shows **appointment, tenure** with 3 metrics:
- **Accuracy**: Overall correct predictions
- **F1 (Macro)**: Average F1 across all classes (treats all classes equally)
- **F1 (Weighted)**: Weighted average F1 by class frequency

Each subplot has:
- X-axis: Indicators (appointment, tenure)
- Y-axis: Metric value (0.0 to 1.0)
- Bars: Different colors for each dataset

### Step-by-Step Usage

#### 1. Define Your Datasets

```python
datasets = {
    'Baseline (T=0.0)': 'data/results/baseline_t0.csv',
    'Baseline (T=0.7)': 'data/results/baseline_t0.7.csv',
    'With SC (n=3)': 'data/results/sc_n3.csv',
    'With SC (n=5)': 'data/results/sc_n5.csv',
    'With CoVe': 'data/results/cove.csv'
}
```

#### 2. Get Metrics DataFrames

```python
from evaluation import compare_datasets

# Get raw metrics
binary_df, multiclass_df = compare_datasets(datasets)

# View binary metrics
print(binary_df)
# Output:
#   dataset    indicator  accuracy  precision  recall  f1  n_samples
#   Baseline   sovereign     0.920      0.935   0.910  0.922      100
#   With CoVe  sovereign     0.950      0.960   0.945  0.952      100
#   ...

# View multi-class metrics
print(multiclass_df)
# Output:
#   dataset    indicator  accuracy  f1_macro  f1_weighted  n_samples
#   Baseline   appointment   0.780     0.765        0.775        100
#   With CoVe  appointment   0.820     0.810        0.818        100
#   ...
```

#### 3. Generate Plots Separately

```python
from evaluation import plot_binary_comparison, plot_multiclass_comparison

# Plot binary indicators only
plot_binary_comparison(binary_df)

# Plot multi-class indicators only
plot_multiclass_comparison(multiclass_df)
```

#### 4. Compare Specific Indicators

```python
# Compare only sovereign and assembly
binary_df, multiclass_df = compare_datasets(
    datasets,
    indicators=['sovereign', 'assembly']
)
```

### Working with Filtered Data in Notebooks

#### Method 1: Filter in Notebook, Compare DataFrames Directly

```python
from evaluation import load_predictions, filter_data, compare_experiments

# Load all datasets
df_baseline = load_predictions('data/results/baseline.csv')
df_sc = load_predictions('data/results/sc.csv')
df_cove = load_predictions('data/results/cove.csv')

# Apply the same filter to all
df_baseline_europe = filter_data(df_baseline, region_v1=['Europe'])
df_sc_europe = filter_data(df_sc, region_v1=['Europe'])
df_cove_europe = filter_data(df_cove, region_v1=['Europe'])

# Compare filtered DataFrames
datasets = {
    'Baseline (Europe)': df_baseline_europe,
    'SC (Europe)': df_sc_europe,
    'CoVe (Europe)': df_cove_europe
}

binary_df, multiclass_df = compare_experiments(datasets)
```

#### Method 2: Compare Different Slices of Same Dataset

```python
from evaluation import load_predictions, filter_data, compare_experiments

# Load one dataset
df = load_predictions('data/results/baseline.csv')

# Create different slices
df_europe = filter_data(df, region_v1=['Europe'])
df_asia = filter_data(df, region_v1=['Asia'])
df_modern = filter_data(df, start_year=(1800, 2000))
df_ancient = filter_data(df, start_year=(-5000, 1800))

# Compare slices
datasets = {
    'Europe': df_europe,
    'Asia': df_asia,
    'Modern (1800-2000)': df_modern,
    'Ancient (<1800)': df_ancient
}

binary_df, multiclass_df = compare_experiments(datasets)
```

#### Method 3: Compare Full vs Filtered

```python
from evaluation import load_predictions, filter_data, compare_experiments

# Load dataset
df_full = load_predictions('data/results/baseline.csv')

# Apply various filters
df_high_conf = df_full[df_full['sovereign_confidence'] >= 70]
df_europe = filter_data(df_full, region_v1=['Europe'])
df_modern = filter_data(df_full, start_year=(1800, 2000))

# Compare
datasets = {
    'Full Dataset': df_full,
    'High Confidence (≥70)': df_high_conf,
    'Europe Only': df_europe,
    'Modern Era': df_modern
}

binary_df, multiclass_df = compare_experiments(datasets)
```

#### Method 4: Save Filtered Versions (Traditional Approach)

```python
from evaluation import load_predictions, filter_data, compare_experiments

# Load and filter each dataset before comparison
original_datasets = {
    'Baseline': 'data/results/baseline.csv',
    'SC': 'data/results/sc.csv',
    'CoVe': 'data/results/cove.csv'
}

filtered_paths = {}

for label, path in original_datasets.items():
    df = load_predictions(path)

    # Filter to 19th-20th century Europe
    df_filtered = filter_data(
        df,
        start_year=(1800, 2000),
        region_v1=['Europe']
    )

    # Save filtered version
    filtered_path = f'data/temp/{label}_filtered.csv'
    df_filtered.to_csv(filtered_path, index=False)
    filtered_paths[label] = filtered_path

# Compare using file paths
binary_df, multiclass_df = compare_experiments(filtered_paths)
```

### Export Metrics for Further Analysis

```python
# Save metrics to CSV
binary_df.to_csv('data/results/binary_comparison.csv', index=False)
multiclass_df.to_csv('data/results/multiclass_comparison.csv', index=False)

# Pivot for Excel analysis
pivot_accuracy = binary_df.pivot(
    index='indicator',
    columns='dataset',
    values='accuracy'
)
pivot_accuracy.to_csv('data/results/accuracy_comparison.csv')
```

### Example: Complete Comparison Workflow

```python
import sys
sys.path.append('/Users/deankuo/Desktop/python/constitution_llm')

from evaluation import compare_experiments

# Define experiments
datasets = {
    'Baseline': 'data/results/baseline.csv',
    'Temperature 0.5': 'data/results/temp_0.5.csv',
    'Temperature 0.7': 'data/results/temp_0.7.csv',
    'Self-Consistency': 'data/results/self_consistency.csv',
    'CoVe': 'data/results/cove.csv'
}

# Run complete comparison
binary_metrics, multiclass_metrics = compare_experiments(datasets)

# Results are automatically displayed:
# - Summary tables printed to console
# - Plot 1: Binary indicators (4 subplots)
# - Plot 2: Multi-class indicators (3 metrics)

# Save results
binary_metrics.to_csv('data/analysis/binary_comparison.csv', index=False)
multiclass_metrics.to_csv('data/analysis/multiclass_comparison.csv', index=False)
```

### Understanding the Output

**Binary Metrics DataFrame Columns:**
- `dataset`: Experiment label
- `indicator`: Indicator name (sovereign, powersharing, assembly, exit)
- `accuracy`: Overall accuracy
- `precision`: Precision score
- `recall`: Recall score
- `f1`: F1-score
- `n_samples`: Number of valid predictions

**Multi-Class Metrics DataFrame Columns:**
- `dataset`: Experiment label
- `indicator`: Indicator name (appointment, tenure)
- `accuracy`: Overall accuracy
- `f1_macro`: Macro-averaged F1
- `f1_weighted`: Weighted-averaged F1
- `n_samples`: Number of valid predictions

### Requirements

All datasets must:
1. Have the same indicators (column names must match)
2. Use the naming convention: `{indicator}` for ground truth, `{indicator}_prediction` for predictions
3. Be in CSV format
4. Have valid ground truth values for comparison

### Tips

1. **Label datasets clearly** - Use descriptive names (e.g., "Baseline T=0.0", "SC n=5", "CoVe + SC")
2. **Use consistent filtering** - Apply the same filters to all datasets for fair comparison
3. **Check sample sizes** - Ensure all datasets have sufficient samples (shown in n_samples column)
4. **Save intermediate results** - Export metrics DataFrames for further statistical analysis
5. **Compare apples to apples** - Use the same model, test set, and evaluation period across experiments
