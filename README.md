# Constitution Analysis Pipeline

A sophisticated pipeline for analyzing historical polities to predict political indicators. The pipeline leverages multiple Large Language Model (LLM) providers with optional verification mechanisms (Self-Consistency and Chain of Verification).

## Features

- **Multi-LLM Support**: Works with multiple LLM providers
  - OpenAI (GPT-5, GPT-4o, etc.)
  - Anthropic (Claude 4.5 Sonnet, etc.)
  - Google Gemini (Gemini 2.5 Pro, etc.)
  - AWS Bedrock (Claude on Bedrock, etc.)

- **8 Political Indicators**:
  - **Constitution**: Written document with governance rules and limitations
  - **Sovereign**: Independent vs. colony/vassal/tributary
  - **Assembly**: Existence of legislative assembly/parliament
  - **Appointment**: How executives are selected (3-class)
  - **Tenure**: Longest leader's tenure: <5y, 5-10y, >10y
  - **Exit**: Irregular (died/forced) vs. regular (voluntary/term limits)
  - **Collegiality**: Whether executive decision-making is shared by a formally constituted body
  - **Separate Powers**: Whether power is divided between multiple independent organizations

- **Verification Methods**:
  - **Self-Consistency**: Multiple samples at different temperatures with majority vote (3 samples default)
  - **Chain of Verification (CoVe)**: Cross-model verification with factual questions (4 questions for constitution)
  - **Note**: `--verify both` currently runs CoVe only (sequential verification not yet implemented)

- **Prompt Modes**:
  - **Single**: All indicators in one unified prompt
  - **Multiple**: Separate prompt per indicator (recommended)
  - **Sequential**: All indicators in one prompt with distinct sequential sections (user-defined or random order)

- **Parallel Row Processing**: Process N leader rows concurrently (`--parallel-rows N`) for faster batch runs. Works with all prompt modes.

- **Downstream Classifiers** (`pipeline/post_processing.py`): Post-processing scripts that run after the main pipeline. Both depend on `assembly_prediction = 1`. Use `--task` to select which to run:
  - `assembly_extended` (default): Upgrades assembly predictions (0/1) to (0/1/2) — label 2 = competitive factions or parties.
  - `elections`: Codes whether assembly members are elected (0/1/2) — label 1 = elected, label 2 = competitive elections (organized factions/parties).
  - `all`: Runs both classifiers in sequence.

- **Search Modes** (`--search-mode`):
  - **None**: Pure LLM output, no web search (default)
  - **Agentic**: LLM decides whether to search via tool calling (Serper API)
  - **Forced**: Always search before LLM answers (Wikipedia/DuckDuckGo/Serper tiered)

- **Gemini Batch API** (`--use-batch`): Submit predictions as a batch job for 50% cost savings. Compatible with `--search-mode forced` (pre-search + batch). Verification runs synchronously after batch completes. Batches are split by `--checkpoint-interval` (default 500 rows) for checkpointing.

- **CSV & JSONL Input**: Accepts both CSV and JSONL input files (auto-detected by extension). Convert between formats with `scripts/csv_to_jsonl.py`.

- **LangSmith Observability** (optional): Trace all LLM calls, prompts, and outputs in the [LangSmith](https://smith.langchain.com/) dashboard. Zero overhead when disabled. Set `LANGCHAIN_TRACING_V2=true` in `.env` to enable.

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

# LangSmith Observability (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=constitution-llm
```

**📖 See [docs/BEDROCK_SETUP.md](docs/BEDROCK_SETUP.md) for detailed AWS Bedrock configuration**

## Quick Start

### Leader Pipeline (All 8 Indicators)

```bash
# Run predictions for all non-constitution indicators
python main.py --pipeline leader \
  --indicators sovereign assembly appointment tenure exit collegiality separate_powers \
  --models gemini-2.5-pro \
  --mode multiple \
  --test 5

# Process 4 leader rows in parallel (faster batch runs)
python main.py --pipeline leader \
  --indicators sovereign assembly \
  --models gemini-2.5-pro \
  --parallel-rows 4 \
  --test 20

# Run with Self-Consistency verification
python main.py --pipeline leader \
  --indicators assembly appointment \
  --models gemini-2.5-pro \
  --verify self_consistency \
  --verify-indicators assembly \
  --test 10

# Run with Chain of Verification (CoVe)
# Note: Verifier model can be set in .env as BEDROCK_VERIFIER_MODEL
python main.py --pipeline leader \
  --indicators constitution \
  --models gemini-2.5-pro \
  --verify cove \
  --verify-indicators constitution \
  --test 5

# Or override verifier model via CLI
python main.py --pipeline leader \
  --indicators constitution \
  --models gemini-2.5-pro \
  --verify cove \
  --verify-indicators constitution \
  --verifier-model anthropic.claude-opus-4-5-20250514-v1:0 \
  --test 5
```

### Polity Pipeline (Legacy, Constitution Only)

```bash
# Basic polity-level constitution prediction
python main.py --pipeline polity --test 5

# With Self-Consistency verification
python main.py --pipeline polity \
  --verify self_consistency \
  --models Gemini=gemini-2.5-pro \
  --test 5

# Multiple models in parallel
python main.py --pipeline polity \
  -m GPT=gpt-4o Gemini=gemini-2.5-pro \
  -i data/plt_polity_data_v2.csv \
  -o data/results/polity_results.csv
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
    indicators=['sovereign', 'assembly', 'collegiality'],
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

# Batch processing (accepts CSV or JSONL)
from utils.data_loader import load_dataframe
df = load_dataframe('data/plt_leaders_data.csv')  # or .jsonl
runner = BatchRunner(
    predictor,
    BatchConfig(checkpoint_interval=500),
    'data/results/output.csv'
)
results_df = runner.run(df.head(100))
```

## Architecture

### Project Structure

```
constitution_llm/
│
├── CLAUDE.md                      # Project design document and specifications
├── README.md                      # Project overview and usage guide
├── main.py                        # CLI entry point (leader + polity pipelines)
├── main_test.py                   # Test script for new pipeline
├── config.py                      # Global configuration, enums, constants
├── requirements.txt               # Python dependencies
├── run.sh                         # Shell script (background exec, env check, etc.)
├── .env.example                   # Environment variable template
│
├── prompts/
│   ├── base_builder.py            # BasePromptBuilder ABC + PromptOutput
│   ├── constitution.py            # Constitution prompt (leader-level, 4 elements)
│   ├── polity_constitution.py     # Legacy polity-level constitution prompt
│   ├── indicators.py              # 7 other indicator prompts (leader-level)
│   ├── polity_indicators.py       # Legacy polity-level indicator prompts
│   ├── single_builder.py          # Combines indicators (unified prompt)
│   ├── multiple_builder.py        # Separate prompt per indicator
│   └── sequential_builder.py      # All 8 indicators in sequential sections
│
├── models/
│   ├── base.py                    # BaseLLM abstract class + ModelResponse
│   ├── llm_clients.py             # OpenAILLM, GeminiLLM, AnthropicLLM, BedrockLLM
│   └── search_agents.py           # Search agents (DuckDuckGo, Wiki, Serper)
│
├── verification/
│   ├── base.py                    # BaseVerification ABC + VerificationResult
│   ├── self_consistency.py        # Temperature sampling + majority vote
│   └── cove.py                    # Chain of Verification (cross-model)
│
├── pipeline/
│   ├── predictor.py               # Core prediction orchestrator
│   ├── batch_runner.py            # Batch processing + checkpoints
│   ├── batch_gemini.py            # Gemini Batch API runner (50% cost savings)
│   ├── search_predictor.py        # Search-augmented predictions (agentic search)
│   ├── pre_search.py              # Deterministic pre-search (Wikipedia/DDG/Serper)
│   └── post_processing.py       # Downstream classifiers: assembly_extended + elections
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
│   ├── data_loader.py             # Unified CSV/JSONL data loading
│   ├── data_cleaner.py            # Data cleaning utilities
│   ├── encoding_fix.py            # CSV encoding utilities
│   ├── langsmith_utils.py         # LangSmith tracing (conditional, zero-overhead)
│   └── sanity_check.py            # Failed row identification + reprocessing
│
├── scripts/
│   └── csv_to_jsonl.py            # CSV to JSONL conversion utility
│
├── tests/
│   └── test_leader_level.py       # Leader-level prompt tests
│
├── docs/
│   ├── BEDROCK_SETUP.md           # AWS Bedrock configuration guide
│   ├── EVALUATION_GUIDE.md        # Evaluation methodology guide
│   ├── IMPLEMENTATION_SUMMARY.md  # Implementation details
│   ├── MIGRATION_GUIDE.md         # Migration from legacy to new pipeline
│   ├── MISSING_VALUES.md          # Handling missing data
│   └── ...                        # Other documentation
│
├── Graph/                         # Visualization outputs (PNG plots)
│
├── *.ipynb                        # Jupyter notebooks (evaluation, data cleaning, etc.)
│
└── data/
    ├── Original_Data/             # Raw datasets
    ├── plt_leaders_data.csv       # Leader-level input data
    ├── plt_polity_data_v2.csv     # Polity-level input data
    ├── temp/                      # Batch API debug files (requests/responses JSONL)
    ├── results/                   # Output directory
    └── logs/                      # Cost tracking logs
```

### System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     SYSTEM ARCHITECTURE                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Config (argparse)                                     │  │
│  │  --mode  --indicators  --verify  --model               │  │
│  │  --search-mode  --use-batch  --parallel-rows           │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Prompt Layer                                          │  │
│  │                                                        │  │
│  │  ┌──────────────┐  ┌───────────────────────────────┐   │  │
│  │  │ Constitution │  │ Other 7 Indicators            │   │  │
│  │  │ (4 elements) │  │ (unified template)            │   │  │
│  │  └──────────────┘  └───────────────────────────────┘   │  │
│  │                                                        │  │
│  │  Builders: Single | Multiple | Sequential              │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Search Layer (--search-mode, optional)                │  │
│  │                                                        │  │
│  │  none:    No search (default, pure LLM)                │  │
│  │  agentic: LLM decides via tool calling (Serper)        │  │
│  │  forced:  Wikipedia → DuckDuckGo → Serper (tiered)     │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │                                   │
│             ┌────────────┴────────────┐                      │
│             ▼                         ▼                      │
│  ┌────────────────────┐  ┌──────────────────────────┐        │
│  │  Model Layer       │  │  Gemini Batch API        │        │
│  │  (sync)            │  │  (--use-batch)           │        │
│  │                    │  │                          │        │
│  │  Gemini | Claude   │  │  • 50% cost savings      │        │
│  │  GPT    | Bedrock  │  │  • Server-side parallel  │        │
│  │                    │  │  • Pre-search compatible │        │
│  │  Unified BaseLLM   │  │  • Sub-batch checkpoint  │        │
│  └─────────┬──────────┘  └────────────┬─────────────┘        │
│             └────────────┬────────────┘                      │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Verification Layer (--verify, optional)               │  │
│  │                                                        │  │
│  │  ┌───────────────────┐  ┌───────────────────┐          │  │
│  │  │ Self-Consistency  │  │      CoVe         │          │  │
│  │  │ • n temperature   │  │ • Question Gen    │          │  │
│  │  │   samples         │  │ • Cross-model     │          │  │
│  │  │ • majority vote   │  │ • Factored exec   │          │  │
│  │  └───────────────────┘  └───────────────────┘          │  │
│  │  Configurable per indicator (--verify-indicators)      │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Output & Evaluation                                   │  │
│  │                                                        │  │
│  │  • CSV + JSON output       • Cost tracking             │  │
│  │  • F1, accuracy, kappa     • Search metadata           │  │
│  │  • Per-class metrics       • Experiment logging        │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │                                   │
│                          ▼  (run separately after main)      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Downstream Classifiers (pipeline/post_processing.py)│  │
│  │                                                        │  │
│  │  Depend on assembly_prediction = 1 from main pipeline  │  │
│  │  assembly_prediction = 0 → pass-through label 0        │  │
│  │                                                        │  │
│  │  --task assembly_extended                              │  │
│  │    0 = no assembly  1 = no factions  2 = factions      │  │
│  │                                                        │  │
│  │  --task elections                                      │  │
│  │    0 = not elected  1 = elected  2 = competitive       │  │
│  │                                                        │  │
│  │  --task all  (runs both in sequence)                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Observability (optional, LANGCHAIN_TRACING_V2=true)   │  │
│  │                                                        │  │
│  │  LangSmith: @traceable on all LLM calls, predict(),    │  │
│  │  search agents. OpenAI/Anthropic clients auto-wrapped. │  │
│  │  Zero overhead when disabled.                          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Usage

### CLI Options

```bash
python main.py --help
```

#### Pipeline Selection

| Argument | Description | Default |
|----------|-------------|---------|
| `--pipeline` | `leader` (new, all 8 indicators) or `polity` (legacy, constitution only) | `polity` |

#### Leader Pipeline Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Prompt mode: `single`, `multiple`, or `sequential` | `multiple` |
| `--indicators` | Indicators to predict | `['constitution']` |
| `--models` | Model identifier (first value used) | `Gemini=gemini-2.5-pro` |
| `--verify` | Verification: `none`, `self_consistency`, `cove`, `both` | `none` |
| `--verify-indicators` | Which indicators to apply verification to | None |
| `--verifier-model` | Model for CoVe verification | Bedrock Claude |
| `--n-samples` | Self-consistency samples | 3 |
| `--sc-temperatures` | SC temperature list | `0.0 0.5 1.0` |
| `--sequence` | Indicator order for sequential mode (space-separated) | None |
| `--random-sequence` | Randomize order in sequential mode | False |
| `--reasoning` | Include reasoning columns (True/False) | `True` |
| `--parallel-rows` | Number of leader rows to process concurrently (no effect with `--use-batch`) | `1` |
| `--checkpoint-interval` | Rows per batch/checkpoint | `500` |
| `--search-mode` | Search mode: `none`, `agentic`, `forced` | `none` |
| `--use-batch` | Use Gemini Batch API (50% cost savings, Gemini only) | `False` |

#### Polity Pipeline Arguments (constitution only, supports multiple models)

| Argument | Description | Default |
|----------|-------------|---------|
| `--models` | One or more models in `KEY=model` format | `Gemini=gemini-2.5-pro` |
| `--verify` | Verification: `none`, `self_consistency`, `cove`, `both` | `none` |
| `--verifier-model` | Model for CoVe verification | None |
| `--search-mode` | Search mode: `none`, `agentic`, `forced` | `none` |
| `--temperature` | Generation temperature | 0 |
| `--max_tokens` | Max tokens per response | 32768 |
| `--delay` | Delay between API calls | 1.0 |

### Example Commands

```bash
# --- LEADER PIPELINE (leader-level, all 8 indicators) ---

# Basic multi-indicator prediction
python main.py --pipeline leader \
  --indicators sovereign assembly appointment \
  --models gemini-2.5-pro \
  --test 10

# Sequential mode with user-defined order
python main.py --pipeline leader \
  --mode sequential \
  --indicators constitution sovereign assembly collegiality separate_powers appointment tenure exit \
  --sequence assembly constitution sovereign exit collegiality separate_powers tenure appointment \
  --models gemini-2.5-pro \
  --test 5

# Sequential mode with random order
python main.py --pipeline leader \
  --mode sequential \
  --indicators constitution sovereign assembly collegiality separate_powers appointment tenure exit \
  --random-sequence \
  --models gemini-2.5-pro \
  --test 5

# Self-Consistency verification
python main.py --pipeline leader \
  --indicators assembly \
  --models gemini-2.5-pro \
  --verify self_consistency \
  --verify-indicators assembly \
  --n-samples 5 \
  --sc-temperatures 0.0 0.3 0.5 0.7 1.0 \
  --test 10

# CoVe verification for constitution
python main.py --pipeline leader \
  --indicators constitution \
  --models gemini-2.5-pro \
  --verify cove \
  --verify-indicators constitution \
  --verifier-model anthropic.claude-sonnet-4-5-20250929-v1:0 \
  --test 5

# Full batch (all non-constitution indicators), 4 rows in parallel
python main.py --pipeline leader \
  --indicators sovereign assembly appointment tenure exit collegiality separate_powers \
  --models gemini-2.5-pro \
  --mode multiple \
  --parallel-rows 4 \
  --input data/plt_leaders_data.csv \
  --output data/results/experiment_001.csv

# --- POLITY PIPELINE (polity-level, constitution only) ---

# Basic polity-level constitution run
python main.py --pipeline polity \
  --models Gemini=gemini-2.5-pro \
  --input data/plt_polity_data_v2.csv \
  --output data/results/polity_001.csv

# With self-consistency verification
python main.py --pipeline polity \
  --models Gemini=gemini-2.5-pro \
  --verify self_consistency \
  --n-samples 3 \
  --test 10

# With CoVe verification
python main.py --pipeline polity \
  --models Gemini=gemini-2.5-pro \
  --verify cove \
  --verifier-model us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  --test 5
```

### Understanding Prompt Modes

The pipeline supports three distinct prompt modes, each with different characteristics:

#### 1. Multiple Mode (Default, Recommended)
- **How it works**: Separate LLM call for each indicator
- **Pros**: No cross-indicator contamination, independent predictions
- **Cons**: More API calls, higher cost
- **Use when**: You want the most accurate predictions and cost is not a primary concern
- **Example**: 8 indicators = 8 separate LLM calls

#### 2. Single Mode
- **How it works**: All indicators merged into one unified prompt
- **Pros**: Fewer API calls, lower cost
- **Cons**: Potential cross-indicator contamination (predictions may influence each other)
- **Use when**: You want to reduce costs and are willing to accept potential contamination
- **Example**: 8 indicators = 1 LLM call with unified definitions

#### 3. Sequential Mode
- **How it works**: All 8 indicators presented as distinct sequential sections in one prompt
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
- **Example**: 8 indicators = 1 LLM call with 8 sequential sections

**Sequential Mode Order Options:**
```bash
# Default order (constitution first, then others)
--mode sequential

# User-specified order
--mode sequential --sequence assembly sovereign exit constitution collegiality separate_powers tenure appointment

# Random order (useful for testing sequence effects)
--mode sequential --random-sequence
```

**Note**: The `--sequence` argument must include all indicators specified in `--indicators`. The order will affect how the LLM sees and processes each indicator.

### Understanding Search Modes

The pipeline supports three search modes for experimental comparison:

#### 1. None (Default)
- **How it works**: Pure LLM output with no web search
- **Use when**: You want baseline predictions using only the model's training knowledge
- **Example**: `--search-mode none`

#### 2. Agentic
- **How it works**: LLM decides whether to search via tool calling (`tool_choice=auto`)
- **Requires**: `SERPER_API_KEY` environment variable
- **Not compatible with**: `--use-batch` (multi-turn interaction required)
- **Example**: `--search-mode agentic`

#### 3. Forced
- **How it works**: Always runs deterministic tiered pre-search before LLM answers (both with and without `--use-batch`)
- **Search order**: Wikipedia API -> DuckDuckGo -> Serper (optional, tiered — lower tiers only run if previous tiers returned < 200 chars)
- **Search query format**:
  - Leader pipeline: `"{leader_name} of {polity} during {start_year}-{end_year}"`
  - Polity pipeline: `"{polity} during {start_year}-{end_year}"`
  - If `end_year` is missing: `"{leader_name} of {polity} reign started in {start_year}"`
  - If `start_year` is missing: `"{leader_name} of {polity} reign ended in {end_year}"`
- **Compatible with**: `--use-batch` (pre-search + batch)
- **Output**: Both CSV and JSON with `search_queries`, `urls_used`, and `web_information` (single/sequential mode)
- **Example**: `--search-mode forced`

**Note on single vs. multiple mode with search:**
- `--mode single`: 1 search call per row (all indicators share the same search results)
- `--mode multiple`: 1 search call per indicator per row (each indicator gets independent search)

```bash
# Compare all three search modes
python main.py --pipeline leader --indicators sovereign assembly --search-mode none --test 10
python main.py --pipeline leader --indicators sovereign assembly --search-mode agentic --test 10
python main.py --pipeline leader --indicators sovereign assembly --search-mode forced --test 10
```

### Using Gemini Batch API

The `--use-batch` flag submits predictions to the Gemini Batch API for **50% cost savings**:

```bash
# Basic batch mode (no search)
python main.py --pipeline leader --indicators sovereign assembly \
    --models gemini-2.5-pro --use-batch --test 20

# Pre-search + batch (forced search compatible)
python main.py --pipeline leader --indicators sovereign assembly \
    --models gemini-2.5-pro --search-mode forced --use-batch --test 20
```

**How batching works:**
- Rows are split into sub-batches based on `--checkpoint-interval` (default 500 rows)
- Each sub-batch is submitted as one Gemini batch job
- Example: 1000 rows, single mode → 2 batch jobs of 500 requests each
- Example: 1000 rows, multiple mode (5 indicators) → 2 batch jobs of 2500 requests each
- `--parallel-rows` has **no effect** with `--use-batch` (Gemini handles parallelism server-side)

**Debug files** (saved to `data/temp/`):
- `{stem}_batch_requests.jsonl` — full Gemini REST API format (model, system_instruction, contents, generationConfig, safetySettings)
- `{stem}_batch_responses.jsonl` — raw LLM responses per request

**Constraints:**
- Only works with Gemini models
- Incompatible with `--search-mode agentic` (use `forced` instead)
- Verification runs synchronously after batch completes
- Batch jobs are polled every 30 seconds until completion


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
    'constitution', 'sovereign', 'assembly',
    'appointment', 'tenure', 'exit', 'collegiality', 'separate_powers'
]

INDICATOR_LABELS = {
    'constitution': ['1', '0'],
    'sovereign': ['0', '1'],
    'assembly': ['0', '1'],
    'appointment': ['0', '1', '2'],
    'tenure': ['0', '1', '2'],
    'exit': ['0', '1'],
    'collegiality': ['0', '1'],
    'separate_powers': ['0', '1'],
}
```

## Output Format

### CSV Columns

For each indicator `{ind}`:
- `{ind}_prediction` - Prediction ("0"/"1"/"2")
- `{ind}_reasoning` - Reasoning text (omitted if `--reasoning False`)
- `{ind}_confidence` - Score 1-100

With verification:
- `{ind}_verified` - Final prediction after verification
- `{ind}_verification` - Verification details

With search mode (leader pipeline, all modes — row-level):
- `search_queries` - Pipe-delimited search queries with source markers (e.g., `[Wikipedia] query | [Serper] query`)
- `urls_used` - Pipe-delimited URLs with source markers (e.g., `[Wikipedia] https://... | [Serper] https://...`)
- `web_information` - Actual retrieved text content (single/sequential mode only; included in both CSV and JSON output)

With search mode (polity pipeline):
- `search_queries_{model}` - Pipe-delimited search queries with source markers
- `urls_used_{model}` - Pipe-delimited URLs with source markers

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
        checkpoint_interval=500,   # Save checkpoint every 500 rows
        delay_between_calls=1.0,
        max_workers=4,             # Process 4 rows concurrently
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
# Generates a unified 2x2 comparison plot:
# Subplots: Accuracy, Precision, Recall, F1-Score
# All indicators shown together with binary vs. multi-class differentiation
# Multi-class indicators use macro-averaged metrics
```

**Unified 2x2 Comparison Plot (standalone):**
```python
from evaluation.notebook_utils import plot_comparison_2x2

datasets = {
    'Baseline': 'data/results/baseline.csv',
    'Self-Consistency': 'data/results/sc.csv',
}
# 2x2 grid: Accuracy, Precision, Recall, F1-Score
# Binary and multi-class indicators differentiated visually
plot_comparison_2x2(datasets)
```

**Per-Class Metrics Visualization:**
```python
from evaluation import plot_per_class_metrics

datasets = {
    'Baseline': 'data/results/baseline.csv',
    'Self-Consistency': 'data/results/sc.csv',
    'CoVe': 'data/results/cove.csv',
}

# Plot per-class precision/recall/F1 for specific indicators
plot_per_class_metrics(datasets, indicators=['assembly', 'appointment', 'tenure'])
# Generates one figure per indicator showing per-class metrics across experiments
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
--pipeline           # leader (default) or polity — which pipeline to use for reprocessing
--indicator          # Primary indicator to check (default: constitution)
--indicators         # List of indicators to reprocess (leader pipeline only)
--min-confidence     # Minimum confidence threshold (1-100)
--min-reasoning-length  # Minimum reasoning length (default: 100)
--mode               # Prompt mode: single, multiple, or sequential (leader only, default: multiple)
--model              # Model identifier (default: gemini-2.5-pro)
--verify             # Verification: none, self_consistency, cove, both
--verifier-model     # Model for CoVe verification
--sequence           # Indicator sequence for sequential mode (space-separated, leader only)
--random-sequence    # Randomize indicator order in sequential mode (leader only)
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

## Downstream Classifiers

`pipeline/post_processing.py` contains two **standalone downstream classifiers** that run
**after** the main pipeline. Both depend on `assembly_prediction` from the main output.
Rows where `assembly = 0` receive a pass-through label `0` (no API call).

Use `--task` to select which classifier(s) to run.

### AssemblyExtended (`--task assembly_extended`, default)

Extends binary assembly predictions (0/1) to a three-label scheme.

| Label | Meaning |
|-------|---------|
| `0` | No assembly (pass-through — no API call) |
| `1` | Assembly exists, no competitive factions or parties |
| `2` | Assembly exists **with** competitive factions or parties |

Output columns added: `assembly_extended_prediction`, `assembly_extended_confidence`, `assembly_extended_reasoning`

### Elections (`--task elections`)

For polities where an assembly exists, codes whether members are elected and whether
elections are contested by organized factions or parties.

| Label | Meaning |
|-------|---------|
| `0` | No assembly / members not elected (pass-through when assembly=0; LLM call when assembly=1 → not elected) |
| `1` | Members elected, no organized factions or parties |
| `2` | Competitive elections — contested by organized factions or parties |

Output columns added: `elections_prediction`, `elections_confidence`, `elections_reasoning`

### Usage

```bash
# Assembly extended only (default)
python pipeline/post_processing.py \
    --input  data/results/predictions.csv \
    --output data/results/predictions_extended.csv

# Elections only
python pipeline/post_processing.py \
    --input  data/results/predictions.csv \
    --output data/results/predictions_extended.csv \
    --task   elections

# Both classifiers in sequence (recommended for full downstream pass)
python pipeline/post_processing.py \
    --input  data/results/predictions.csv \
    --output data/results/predictions_extended.csv \
    --task   all \
    --model  gemini-2.5-pro \
    --parallel-rows 4

# Test on first 10 rows only
python pipeline/post_processing.py \
    --input  data/results/predictions.csv \
    --output data/results/predictions_extended.csv \
    --task   all --test 10
```

### CLI Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Predictions file (CSV or JSONL) from the main pipeline | required |
| `--output` | Output CSV path | required |
| `--task` | `assembly_extended`, `elections`, or `all` | `assembly_extended` |
| `--model` | LLM model identifier | `gemini-2.5-pro` |
| `--assembly-col` | Column name with binary assembly predictions | `assembly_prediction` |
| `--parallel-rows` | Concurrent rows to process | `1` |
| `--delay` | Seconds between calls / windows | `1.0` |
| `--test` | Process only first N rows | None |

---

## Using run.sh

The `run.sh` script is a user-friendly wrapper for running the pipeline. It supports all CLI options, environment validation, and background execution that survives computer sleep.

```bash
# Make it executable (first time only)
chmod +x run.sh

# Check environment setup (Python, packages, API keys)
./run.sh --check-env

# Quick test (5 rows, leader pipeline)
./run.sh --quick-test

# Run a leader pipeline experiment
./run.sh --pipeline leader \
  --indicators sovereign assembly \
  --models gemini-2.5-pro \
  --test 20

# Run in background (survives sleep/logout, uses caffeinate)
./run.sh --background --pipeline leader \
  --indicators sovereign assembly appointment tenure exit collegiality separate_powers \
  --models gemini-2.5-pro \
  --input data/plt_leaders_data.csv \
  --output data/results/full_run.csv

# Preview command without running (dry run)
./run.sh --dry-run --pipeline leader --indicators sovereign --test 5

# Send macOS notification on completion
./run.sh --notify --pipeline leader --indicators sovereign --test 10
```

**Background mode** uses `caffeinate -i` to prevent macOS sleep and `nohup` to survive terminal close. Logs are saved to `data/logs/run_<timestamp>.log`.

---

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
python main.py --pipeline leader --test 1
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
