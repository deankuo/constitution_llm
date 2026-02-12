# Constitution Analysis Pipeline

A sophisticated pipeline for analyzing historical polities to predict political indicators. The pipeline leverages multiple Large Language Model (LLM) providers with optional verification mechanisms (Self-Consistency and Chain of Verification).

## Features

- **Multi-LLM Support**: Works with multiple LLM providers
  - OpenAI (GPT-5, GPT-4o, etc.)
  - Anthropic (Claude 4.5 Sonnet, etc.)
  - Google Gemini (Gemini 2.5 Pro, etc.)
  - AWS Bedrock (Claude on Bedrock, etc.)

- **7 Political Indicators**:
  - **Constitution**: Written document with governance rules and limitations
  - **Sovereign**: Independent vs. colony/vassal/tributary
  - **Powersharing**: Single leader vs. multiple leaders with comparable power
  - **Assembly**: Existence of legislative assembly/parliament
  - **Appointment**: How executives are selected (3-class)
  - **Tenure**: Longest leader's tenure: <5y, 5-10y, >10y
  - **Exit**: Irregular (died/forced) vs. regular (voluntary/term limits)

- **Verification Methods**:
  - **Self-Consistency**: Multiple samples at different temperatures with majority vote (3 samples default)
  - **Chain of Verification (CoVe)**: Cross-model verification with factual questions (4 questions for constitution)
  - **Note**: `--verify both` currently runs CoVe only (sequential verification not yet implemented)

- **Prompt Modes**:
  - **Single**: All indicators in one unified prompt
  - **Multiple**: Separate prompt per indicator (recommended)
  - **Sequential**: All 7 indicators in one prompt with distinct sequential sections (user-defined or random order)

- **Robust Processing**: Checkpoint system, automatic retries, cost tracking

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/constitution_llm.git
cd constitution_llm
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
# Edit .env with your keys
```

Key configurations:
```bash
# Required: Choose your LLM provider(s)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional

# AWS Bedrock (for Claude on Bedrock)
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1

# Bedrock Verifier Model (for CoVe)
BEDROCK_VERIFIER_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

**📖 See [docs/BEDROCK_SETUP.md](docs/BEDROCK_SETUP.md) for detailed AWS Bedrock configuration**

## Quick Start

### Using the New Pipeline (Recommended)

```bash
# Run predictions for 6 indicators (excluding constitution)
python main.py --new-pipeline \
  --indicators sovereign powersharing assembly appointment tenure exit \
  --models gemini-2.5-pro \
  --mode multiple \
  --test 5

# Run with Self-Consistency verification
python main.py --new-pipeline \
  --indicators assembly appointment \
  --models gemini-2.5-pro \
  --verify self_consistency \
  --verify-indicators assembly \
  --test 10

# Run with Chain of Verification (CoVe)
# Note: Verifier model can be set in .env as BEDROCK_VERIFIER_MODEL
python main.py --new-pipeline \
  --indicators constitution \
  --models gemini-2.5-pro \
  --verify cove \
  --verify-indicators constitution \
  --test 5

# Or override verifier model via CLI
python main.py --new-pipeline \
  --indicators constitution \
  --models gemini-2.5-pro \
  --verify cove \
  --verify-indicators constitution \
  --verifier-model anthropic.claude-opus-4-5-20250514-v1:0 \
  --test 5
```

### Using Python API

```python
from pipeline.predictor import Predictor, PredictionConfig
from pipeline.batch_runner import BatchRunner, BatchConfig
from config import PromptMode, VerificationType
import pandas as pd
import os

# Configure
config = PredictionConfig(
    mode=PromptMode.MULTIPLE,
    indicators=['sovereign', 'powersharing', 'assembly'],
    verify=VerificationType.NONE,
    model='gemini-2.5-pro',
    temperature=0.0
)

api_keys = {'gemini': os.getenv('GEMINI_API_KEY')}

# Single prediction (leader-level: polity, name, start_year, end_year)
predictor = Predictor(config, api_keys)
result = predictor.predict("Roman Republic", "Julius Caesar", -49, -44)
print(result.predictions['sovereign'].prediction)
print(result.predictions['sovereign'].reasoning)

# Batch processing
df = pd.read_csv('data/plt_leaders_data.csv')
runner = BatchRunner(
    predictor,
    BatchConfig(checkpoint_interval=50),
    'data/results/output.csv'
)
results_df = runner.run(df.head(100))
```

## Architecture

### Project Structure

```
constitution_llm/
├── main.py                        # CLI entry point (legacy + new pipeline)
├── main_test.py                   # Test script for new pipeline
├── config.py                      # Global configuration, enums
│
├── prompts/
│   ├── base_builder.py            # BasePromptBuilder ABC + PromptOutput
│   ├── constitution.py            # Constitution prompt (leader-level, 4 elements)
│   ├── polity_constitution.py     # Legacy polity-level constitution prompt
│   ├── indicators.py              # 6 other indicator prompts (leader-level)
│   ├── polity_indicators.py       # Legacy polity-level indicator prompts
│   ├── single_builder.py          # Combines indicators (unified prompt)
│   ├── multiple_builder.py        # Separate prompt per indicator
│   └── sequential_builder.py      # All 7 indicators in sequence
│
├── models/
│   ├── base.py                    # BaseLLM abstract class + ModelResponse
│   ├── llm_clients.py             # OpenAILLM, GeminiLLM, AnthropicLLM, BedrockLLM
│   └── search_agents.py           # Web search agents
│
├── verification/
│   ├── base.py                    # BaseVerification ABC + VerificationResult
│   ├── self_consistency.py        # Temperature sampling + majority vote
│   └── cove.py                    # Chain of Verification (cross-model)
│
├── pipeline/
│   ├── predictor.py               # Core prediction orchestrator
│   └── batch_runner.py            # Batch processing + checkpoints
│
├── evaluation/
│   ├── metrics.py                 # Accuracy, F1, Cohen's kappa
│   ├── analyzer.py                # ResultAnalyzer class
│   └── notebook_utils.py          # Jupyter notebook helpers
│
├── utils/
│   ├── json_parser.py             # Robust JSON extraction + validation
│   ├── cost_tracker.py            # API cost tracking per model/indicator
│   ├── logger.py                  # Logging utilities
│   ├── data_cleaner.py            # Data cleaning utilities
│   ├── encoding_fix.py            # CSV encoding utilities
│   └── sanity_check.py            # Failed row identification + reprocessing
│
├── tests/
│   └── test_leader_level.py       # Leader-level prompt tests
│
├── docs/
│   ├── BEDROCK_SETUP.md           # AWS Bedrock configuration guide
│   ├── EVALUATION_GUIDE.md        # Evaluation methodology guide
│   └── ...                        # Other documentation
│
└── data/
    ├── Original_Data/             # Raw datasets
    ├── plt_leaders_data.csv       # Leader-level input data
    ├── plt_polity_data_v2.csv     # Polity-level input data
    ├── results/                   # Output directory
    └── logs/                      # Cost tracking logs
```

### System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   Config    │     │              Prompt Layer                       │   │
│  │  (argparse) │     │  ┌─────────────────┐  ┌─────────────────────┐   │   │
│  │             │     │  │  Constitution   │  │  Other 6 Indicators │   │   │
│  │ --mode      │     │  │  (complex)      │  │  (unified template) │   │   │
│  │ --indicators│────▶│  └─────────────────┘  └─────────────────────┘   │   │
│  │ --verify    │     │           │                     │               │   │
│  │ --model     │     │           ▼                     ▼               │   │
│  └─────────────┘     │  ┌─────────────────────────────────────────┐    │   │
│                      │  │         Prompt Builder                  │    │   │
│                      │  │  • SinglePromptBuilder (unified)        │    │   │
│                      │  │  • MultiplePromptBuilder (separate)     │    │   │
│                      │  │  • SequentialPromptBuilder (sequence)   │    │   │
│                      │  └─────────────────────────────────────────┘    │   │
│                      └─────────────────────────────────────────────────┘   │
│                                           │                                │
│                                           ▼                                │
│                      ┌─────────────────────────────────────────────────┐   │
│                      │              Model Layer                        │   │
│                      │  ┌─────────┐ ┌─────────┐ ┌─────────┐            │   │
│                      │  │ Gemini  │ │ Claude  │ │   GPT   │  ...       │   │
│                      │  └─────────┘ └─────────┘ └─────────┘            │   │
│                      │         (Unified BaseLLM Interface)             │   │
│                      └─────────────────────────────────────────────────┘   │
│                                           │                                │
│                                           ▼                                │
│                      ┌─────────────────────────────────────────────────┐   │
│                      │           Verification Layer (Optional)         │   │
│                      │                                                 │   │
│                      │  ┌───────────────────┐  ┌───────────────────┐   │   │
│                      │  │ Self-Consistency  │  │       CoVe        │   │   │
│                      │  │                   │  │                   │   │   │
│                      │  │ • n_samples       │  │ • Question Gen    │   │   │
│                      │  │ • temperature     │  │ • Cross-model     │   │   │
│                      │  │ • majority vote   │  │ • Synthesis       │   │   │
│                      │  └───────────────────┘  └───────────────────┘   │   │
│                      │         (Configurable per indicator)            │   │
│                      └─────────────────────────────────────────────────┘   │
│                                           │                                │
│                                           ▼                                │
│                      ┌─────────────────────────────────────────────────┐   │
│                      │              Output & Evaluation                │   │
│                      │  • JSON parsing (robust)                        │   │
│                      │  • Metrics (F1, accuracy, per-class)            │   │
│                      │  • Cost tracking                                │   │
│                      │  • Experiment logging                           │   │
│                      └─────────────────────────────────────────────────┘   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Usage

### CLI Options

```bash
python main.py --help
```

#### New Pipeline Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--new-pipeline` | Use the new modular pipeline | False |
| `--mode` | Prompt mode: `single`, `multiple`, or `sequential` | `multiple` |
| `--indicators` | Indicators to predict | All 6 (excl. constitution) |
| `--models` | Primary model (first value used) | `Gemini=gemini-2.5-pro` |
| `--verify` | Verification: `none`, `self_consistency`, `cove`, `both` | `none` |
| `--verify-indicators` | Which indicators to verify | None |
| `--verifier-model` | Model for CoVe verification | Bedrock Claude |
| `--n-samples` | Self-consistency samples | 3 |
| `--sc-temperatures` | SC temperature list | `0.0,0.5,1.0` |
| `--sequence` | Indicator order for sequential mode (space-separated) | None |
| `--random-sequence` | Randomize order in sequential mode | False |

#### Legacy Arguments (still supported)

| Argument | Description | Default |
|----------|-------------|---------|
| `--models` | Legacy model spec (KEY=model) | Gemini |
| `--use-search` | Enable web search | False |
| `--temperature` | Generation temperature | 0 |
| `--max_tokens` | Max tokens per response | 2048 |
| `--delay` | Delay between API calls | 1.0 |

### Example Commands

```bash
# Basic prediction with multiple indicators
python main.py --new-pipeline \
  --indicators sovereign assembly appointment \
  --models gemini-2.5-pro \
  --test 10

# Sequential mode with user-defined order
python main.py --new-pipeline \
  --mode sequential \
  --indicators constitution sovereign assembly powersharing appointment tenure exit \
  --sequence assembly constitution sovereign exit powersharing tenure appointment \
  --models gemini-2.5-pro \
  --test 5

# Sequential mode with random order
python main.py --new-pipeline \
  --mode sequential \
  --indicators constitution sovereign assembly powersharing appointment tenure exit \
  --random-sequence \
  --models gemini-2.5-pro \
  --test 5

# Self-Consistency verification
python main.py --new-pipeline \
  --indicators assembly \
  --models gemini-2.5-pro \
  --verify self_consistency \
  --verify-indicators assembly \
  --n-samples 5 \
  --sc-temperatures 0.0,0.3,0.5,0.7,1.0 \
  --test 10

# CoVe verification for constitution
python main.py --new-pipeline \
  --indicators constitution \
  --models gemini-2.5-pro \
  --verify cove \
  --verify-indicators constitution \
  --verifier-model anthropic.claude-sonnet-4-5-20250929-v1:0 \
  --test 5

# Full batch processing
python main.py --new-pipeline \
  --indicators sovereign powersharing assembly appointment tenure exit \
  --models gemini-2.5-pro \
  --mode multiple \
  --input data/plt_polity_data_v2.csv \
  --output data/results/experiment_001.csv
```

### Understanding Prompt Modes

The pipeline supports three distinct prompt modes, each with different characteristics:

#### 1. Multiple Mode (Default, Recommended)
- **How it works**: Separate LLM call for each indicator
- **Pros**: No cross-indicator contamination, independent predictions
- **Cons**: More API calls, higher cost
- **Use when**: You want the most accurate predictions and cost is not a primary concern
- **Example**: 7 indicators = 7 separate LLM calls

#### 2. Single Mode
- **How it works**: All indicators merged into one unified prompt
- **Pros**: Fewer API calls, lower cost
- **Cons**: Potential cross-indicator contamination (predictions may influence each other)
- **Use when**: You want to reduce costs and are willing to accept potential contamination
- **Example**: 7 indicators = 1 LLM call with unified definitions

#### 3. Sequential Mode
- **How it works**: All 7 indicators presented as distinct sequential sections in one prompt
- **Pros**:
  - Single API call (cost-efficient)
  - Each indicator maintains its distinct prompt structure
  - Can control or randomize the order to test sequence effects
- **Cons**:
  - Indicators may still influence each other due to shared context
  - Longer prompt may affect response quality
- **Use when**:
  - You want to study how indicator ordering affects predictions
  - You want cost efficiency while maintaining distinct indicator definitions
- **Example**: 7 indicators = 1 LLM call with 7 sequential sections

**Sequential Mode Order Options:**
```bash
# Default order (constitution first, then others)
--mode sequential

# User-specified order
--mode sequential --sequence assembly sovereign exit constitution powersharing tenure appointment

# Random order (useful for testing sequence effects)
--mode sequential --random-sequence
```

**Note**: The `--sequence` argument must include all indicators specified in `--indicators`. The order will affect how the LLM sees and processes each indicator.


## Configuration

### config.py

```python
# Default models
DEFAULT_PRIMARY_MODEL = "gemini-2.5-pro"
DEFAULT_VERIFIER_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"  # From BEDROCK_VERIFIER_MODEL env

# LLM Parameters
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 32768  # Increased to handle single prompt mode with multiple indicators

# Processing
DEFAULT_BATCH_SIZE = 100
DEFAULT_DELAY = 1.0
DEFAULT_MAX_RETRIES = 3

# Indicators
ALL_INDICATORS = [
    'constitution', 'sovereign', 'powersharing',
    'assembly', 'appointment', 'tenure', 'exit'
]

INDICATOR_LABELS = {
    'constitution': ['1', '0'],
    'sovereign': ['0', '1'],
    'powersharing': ['0', '1'],
    'assembly': ['0', '1'],
    'appointment': ['0', '1', '2'],
    'tenure': ['0', '1', '2'],
    'exit': ['0', '1']
}
```

## Output Format

### CSV Columns

For each indicator `{ind}`:
- `{ind}` - Prediction ("0"/"1"/"2")
- `{ind}_reasoning` - Reasoning text
- `{ind}_confidence` - Score 1-100

With verification:
- `{ind}_verified` - Final prediction after verification
- `{ind}_verification` - Verification details

Additional columns:
- `total_cost_usd` - Total API cost
- `total_tokens` - Total tokens used

### Example Output

```csv
territorynamehistorical,start_year,end_year,sovereign,sovereign_reasoning,sovereign_confidence,assembly,assembly_reasoning,assembly_confidence,total_cost_usd,total_tokens
Roman Republic,-509,-27,1,"The Roman Republic...",85,1,"The Roman Republic had...",90,0.0125,2500
```

## API Reference

### Predictor

```python
from pipeline.predictor import Predictor, PredictionConfig
from config import PromptMode, VerificationType

config = PredictionConfig(
    mode=PromptMode.MULTIPLE,
    indicators=['sovereign', 'assembly'],
    verify=VerificationType.SELF_CONSISTENCY,
    verify_indicators=['assembly'],
    model='gemini-2.5-pro',
    temperature=0.0,
    sc_n_samples=3,
    sc_temperatures=[0.0, 0.5, 1.0]
)

predictor = Predictor(config, api_keys)
result = predictor.predict("Roman Republic", "Julius Caesar", -49, -44)
```

### BatchRunner

```python
from pipeline.batch_runner import BatchRunner, BatchConfig

runner = BatchRunner(
    predictor=predictor,
    config=BatchConfig(
        checkpoint_interval=50,
        delay_between_calls=1.0,
        output_formats=['csv', 'json']
    ),
    output_path='data/results/output.csv'
)

results_df = runner.run(df)
```

### Evaluation

```python
from evaluation.metrics import evaluate_indicator, format_metrics_report

# Evaluate predictions against ground truth
metrics = evaluate_indicator(
    predictions=['1', '0', '1', '1'],
    ground_truth=['1', '0', '0', '1'],
    indicator='assembly'
)

print(format_metrics_report(metrics))
```

**📖 See [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md) for complete evaluation guide with filtering, visualization, and polity-level accuracy.**

**Multi-Dataset Comparison:**
```python
from evaluation import compare_experiments

# Option 1: Compare using file paths
datasets = {
    'Baseline': 'data/results/baseline.csv',
    'Self-Consistency': 'data/results/sc.csv',
    'CoVe': 'data/results/cove.csv'
}

# Option 2: Compare using DataFrames (loaded/filtered in notebook)
# datasets = {
#     'Baseline': df_baseline,
#     'Filtered Europe': df_europe,  # Pre-filtered DataFrame
#     'CoVe': df_cove
# }

binary_metrics, multiclass_metrics = compare_experiments(datasets)
# Generates 2 plots:
# 1. Binary indicators: 4 subplots (accuracy, precision, recall, f1)
# 2. Multi-class indicators: 3 metrics (accuracy, f1_macro, f1_weighted)
```

### Sanity Check and Reprocessing

The `sanity_check.py` utility identifies problematic predictions and automatically reprocesses them.

```bash
# Check constitution predictions
python utils/sanity_check.py \
    -i data/results/predictions.csv \
    -o data/results/predictions_fixed.csv \
    --indicator constitution

# Check with confidence threshold
python utils/sanity_check.py \
    -i data/results/predictions.csv \
    -o data/results/predictions_fixed.csv \
    --indicator sovereign \
    --min-confidence 50
```

**Sanity Check Criteria:**
- Missing confidence scores (NA/null values)
- Negative confidence (-1 indicates error)
- Low confidence (below threshold if specified)
- Null predictions (missing values)
- Short reasoning (< 100 characters default)
- Uncodified constitutions (for constitution indicator only)

**Command Options:**
```bash
--indicator          # Primary indicator to check (default: constitution)
--indicators         # List of indicators to reprocess
--min-confidence     # Minimum confidence threshold (1-100)
--min-reasoning-length  # Minimum reasoning length (default: 100)
--mode               # Prompt mode: single, multiple, or sequential (default: multiple)
--model              # Model in format Provider=model (default: Gemini=gemini-2.5-pro)
--verify             # Verification: none, self_consistency, cove, both
--sequence           # Indicator sequence for sequential mode (space-separated)
--random-sequence    # Randomize indicator order in sequential mode
--no-cleanup         # Keep temporary files for debugging
```

**Important:** Output is saved to the path specified by `--output` (not overwriting the input file unless you specify the same path).

**Example with Verification:**
```bash
python utils/sanity_check.py \
    -i data/results/predictions.csv \
    -o data/results/predictions_fixed.csv \
    --indicator constitution \
    --verify cove \
    --model Gemini=gemini-2.5-pro
```

**Example with Sequential Mode:**
```bash
# Sequential mode with user-defined order
python utils/sanity_check.py \
    -i data/results/predictions.csv \
    -o data/results/predictions_fixed.csv \
    --indicators constitution sovereign assembly \
    --mode sequential \
    --sequence assembly sovereign constitution \
    --model Gemini=gemini-2.5-pro

# Sequential mode with random order
python utils/sanity_check.py \
    -i data/results/predictions.csv \
    -o data/results/predictions_fixed.csv \
    --indicators constitution sovereign assembly \
    --mode sequential \
    --random-sequence \
    --model Gemini=gemini-2.5-pro
```

**Column Naming Support:**
- Automatically detects new format: `{indicator}_prediction`, `{indicator}_confidence`, `{indicator}_reasoning`
- Backward compatible with legacy format: `constitution_gemini`, `confidence_score_gemini`, `explanation_gemini`

## Troubleshooting

### Common Issues

#### 1. API Key Errors

```
ERROR: Anthropic API key not provided.
```

Ensure your `.env` file contains the required API keys.

#### 2. Module Import Errors

```
ModuleNotFoundError: No module named 'anthropic'
```

Install dependencies: `pip install -r requirements.txt`

#### 3. Rate Limiting

```
WARN: Bedrock API throttling detected
```

Increase delay: `--delay 2.0`

#### 4. JSON Parsing Errors

The pipeline includes robust JSON parsing that handles markdown code fences and partial responses. If issues persist, check the raw responses in the logs.

### Debug Mode

Set environment variable for verbose output:
```bash
export CONSTITUTION_DEBUG=1
python main.py --new-pipeline --test 1
```

## Additional Documentation

- **[docs/BEDROCK_SETUP.md](docs/BEDROCK_SETUP.md)** - Complete AWS Bedrock configuration guide
  - How to find Bedrock model ARNs
  - Environment variable setup
  - Troubleshooting Bedrock issues
  - Best practices for public repositories

- **[docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md)** - Evaluation methodology guide
  - Metrics (accuracy, F1, Cohen's kappa)
  - Filtering and visualization
  - Multi-dataset comparison

- **[CLAUDE.md](CLAUDE.md)** - Project design document
  - Architecture overview
  - Indicator definitions
  - Research questions
  - Implementation details

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude models
- Google for Gemini models
- AWS for Bedrock platform
