# Constitution Analysis Pipeline

A pipeline for predicting political indicators for historical pre-modern polities using LLMs, with optional self-consistency verification, Chain of Verification, and web search augmentation.

## Features

- **Multi-LLM Support**: Google Gemini (default: `gemini-3.1-pro-preview`), OpenAI GPT, Anthropic Claude, AWS Bedrock
- **Political Indicators**: Constitution, Sovereign, Federalism, Checks, Collegiality, Petition, Assembly, Entry, Exit, Symbolism, Elections (downstream)
- **Verification**: Self-Consistency (default, n=2 additional samples) and Chain of Verification (CoVe, cross-model)
- **Prompt Modes**: Single (default, all indicators in one call), Multiple (one call per indicator), Sequential
- **Search Modes**: None (default), Agentic (Serper), Forced (Wikipedia → DuckDuckGo → Serper tiered)
- **Gemini Batch API**: 50% cost savings via `--use-batch`
- **CSV & JSONL Input**: Auto-detected by file extension
- **LangSmith Observability**: Optional tracing, zero overhead when disabled

## Installation

```bash
# Clone and enter repo
git clone https://github.com/yourusername/constitution_llm.git
cd constitution_llm

# Install dependencies (uv recommended)
uv sync
# or: pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env with your keys
```

Required `.env` keys:
```bash
GEMINI_API_KEY=your_gemini_api_key

# Optional: other providers
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
BEDROCK_VERIFIER_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=constitution-llm
```

## Quick Start

```bash
# 5-row test (default: single mode, gemini-3.1-pro-preview, self-consistency n=2)
python main.py --pipeline indicators --test 5

# Or use run.sh
./run.sh --quick-test
```

## Pipelines

### Indicators Pipeline (main)

Predicts any combination of indicators at the leader level.

```bash
# Default settings: single mode, self-consistency n=2, gemini-3.1-pro-preview
python main.py --pipeline indicators \
    --indicators constitution sovereign federalism checks collegiality petition assembly entry exit symbolism \
    --input data/plt_leaders_data.csv \
    --output data/results/exp001.csv

# Multiple mode (separate call per indicator)
python main.py --pipeline indicators \
    --mode multiple \
    --indicators sovereign assembly collegiality \
    --models gemini-3.1-pro-preview \
    --output data/results/exp002.csv

# Sequential mode with user-defined order
python main.py --pipeline indicators \
    --mode sequential \
    --indicators constitution sovereign assembly collegiality \
    --sequence assembly constitution sovereign collegiality \
    --output data/results/exp003.csv

# CoVe verification for constitution
python main.py --pipeline indicators \
    --mode multiple \
    --indicators constitution \
    --verify cove \
    --verify-indicators constitution \
    --verifier-model us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --output data/results/exp004.csv

# Forced search
python main.py --pipeline indicators \
    --mode multiple \
    --indicators sovereign assembly \
    --search-mode forced \
    --output data/results/exp005.csv

# Gemini Batch API (50% cost savings)
python main.py --pipeline indicators \
    --indicators sovereign assembly \
    --use-batch \
    --output data/results/exp006.csv

# Parallel row processing (4 rows at once)
python main.py --pipeline indicators \
    --indicators constitution sovereign \
    --parallel-rows 4 \
    --output data/results/exp007.csv
```

### Constitution Pipeline (legacy, polity level)

Single-model, constitution-only, polity-level predictions.

```bash
# Basic run
python main.py --pipeline constitution \
    --models gemini-3.1-pro-preview \
    --input data/plt_polity_data_v2.csv \
    --output data/results/const_exp001.csv

# With self-consistency
python main.py --pipeline constitution \
    --models gemini-3.1-pro-preview \
    --verify self_consistency \
    --n-samples 2 \
    --input data/plt_polity_data_v2.csv \
    --output data/results/const_exp002.csv
```

### Downstream Classifiers (post_processing.py)

Run **after** the main pipeline. Elections depends on `assembly_prediction = 2`; other rows pass through with `elections = 0`.

```bash
python pipeline/post_processing.py \
    --input  data/results/predictions.csv \
    --output data/results/predictions_extended.csv \
    --model  gemini-3.1-pro-preview \
    --parallel-rows 4

# With self-consistency (n=2 additional samples = 3 total votes)
python pipeline/post_processing.py \
    --input  data/results/predictions.csv \
    --output data/results/predictions_extended.csv \
    --n-samples 2
```

## CLI Reference

### Indicators Pipeline Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--pipeline` | `indicators` or `constitution` | `indicators` |
| `--mode` | `single`, `multiple`, or `sequential` | `single` |
| `--indicators` | Space-separated list of indicators | `constitution` |
| `--models` | Model identifier | `gemini-3.1-pro-preview` |
| `--verify` | `none`, `self_consistency`, `cove`, `both` | `self_consistency` |
| `--verify-indicators` | Which indicators to verify (omit = all) | all `--indicators` |
| `--verifier-model` | Model for CoVe | Bedrock Claude (from `BEDROCK_VERIFIER_MODEL`) |
| `--n-samples` | Additional SC samples (total votes = n+1) | `2` |
| `--sc-temperatures` | Temperature list for SC samples | `1.0 1.0 1.0` |
| `--search-mode` | `none`, `agentic`, or `forced` | `none` |
| `--use-batch` | Gemini Batch API (50% savings) | `False` |
| `--parallel-rows` | Concurrent row workers | `1` |
| `--checkpoint-interval` | Rows per checkpoint | `500` |
| `--sequence` | Indicator order for sequential mode | None |
| `--random-sequence` | Randomize sequential order | `False` |
| `--reasoning` | Include reasoning columns | `True` |
| `--logprobs` | Token-level log probabilities (Gemini only) | `False` |
| `--input` | Input CSV or JSONL | `data/plt_leaders_data.csv` |
| `--output` | Output CSV path | `data/results/llm_predictions.csv` |
| `--test` | Process first N rows (or `start:end` range) | None |
| `--delay` | Seconds between API calls | `1.0` |
| `--temperature` | LLM temperature | `1.0` |

## Architecture

### System Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     SYSTEM ARCHITECTURE                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Config (argparse)                                           │
│  --mode  --indicators  --verify  --model                     │
│  --search-mode  --use-batch  --parallel-rows                 │
│              │                                               │
│              ▼                                               │
│  Prompt Layer                                                │
│  ┌──────────────┐  ┌─────────────────────────────────────┐  │
│  │ Constitution │  │ Other Indicators (unified template)  │  │
│  │ (4 elements) │  │                                     │  │
│  └──────────────┘  └─────────────────────────────────────┘  │
│  Builders: Single | Multiple | Sequential                    │
│              │                                               │
│              ▼                                               │
│  Search Layer (--search-mode, optional)                      │
│  none: No search  agentic: tool calling  forced: tiered      │
│              │                                               │
│        ┌─────┴──────┐                                        │
│        ▼            ▼                                        │
│  Model Layer    Gemini Batch API (--use-batch)               │
│  Gemini|Claude  50% cost  · server-side parallel             │
│  GPT|Bedrock    · pre-search compatible                      │
│        └─────┬──────┘                                        │
│              ▼                                               │
│  Verification Layer (--verify, optional)                     │
│  Self-Consistency: n+1 total votes · majority wins           │
│  CoVe: question gen · cross-model · factored execution       │
│              │                                               │
│              ▼                                               │
│  Output: CSV + JSON · cost tracking · experiment log         │
│              │                                               │
│              ▼  (run separately)                             │
│  Downstream Classifiers (pipeline/post_processing.py)        │
│  elections: assembly=2 → LLM call; else → pass-through 0    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Project Structure

```
constitution_llm/
│
├── CLAUDE.md                      # Project design doc and specs
├── README.md                      # This file
├── docs/CODEBOOK.md               # Output dataset column codebook
├── main.py                        # CLI entry point
├── config.py                      # Global settings, enums, constants
├── requirements.txt               # Python dependencies
├── run.sh                         # Shell wrapper (background, env check, etc.)
│
├── prompts/
│   ├── base_builder.py            # BasePromptBuilder ABC + PromptOutput
│   ├── constitution.py            # Constitution prompt (leader-level, 4 elements)
│   ├── polity_constitution.py     # Legacy polity-level constitution prompt
│   ├── indicators.py              # Non-constitution indicator prompts
│   ├── single_builder.py          # Unified prompt (all indicators)
│   ├── multiple_builder.py        # Separate prompt per indicator
│   └── sequential_builder.py     # Sequential sections prompt
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
│   ├── jsonl_batch_runner.py      # Gemini Batch API runner
│   ├── search_predictor.py        # Agentic search-augmented predictions
│   ├── pre_search.py              # Deterministic pre-search
│   └── post_processing.py        # Downstream classifiers (elections)
│
├── evaluation/
│   ├── metrics.py                 # Accuracy, F1, Cohen's kappa
│   ├── analyzer.py                # ResultAnalyzer class
│   └── notebook_utils.py          # Jupyter notebook helpers
│
├── utils/
│   ├── json_parser.py             # Robust JSON extraction + validation
│   ├── cost_tracker.py            # API cost tracking per model/indicator
│   ├── data_loader.py             # Unified CSV/JSONL loading
│   ├── sanity_check.py            # Failed row identification + reprocessing
│   └── langsmith_utils.py         # LangSmith tracing (zero-overhead when off)
│
├── src/
│   └── csv_to_jsonl.py            # CSV to JSONL conversion
│
└── data/
    ├── plt_leaders_data.csv       # Leader-level input data
    ├── plt_polity_data_v2.csv     # Polity-level input data
    ├── temp/                      # Batch API debug files
    ├── results/                   # Output directory
    └── logs/
        ├── experiments.jsonl      # Append-only experiment log
        └── run_*.log              # Background run logs
```

## Output Format

See **[docs/CODEBOOK.md](docs/CODEBOOK.md)** for the full column-by-column codebook.

**Summary:**

Without SC (plain mode): `{ind}_prediction`, `{ind}_reasoning`, `{ind}_confidence`

With SC (`--verify self_consistency --n-samples N`, total N+1 votes):
- `{ind}_prediction` — majority vote from all N+1 SC slots
- `{ind}_SC1` — initial call (SC slot 1)
- `{ind}_SC2`, `{ind}_SC3` — additional samples
- `{ind}_reasoning_SC1/2/3`, `{ind}_confidence_SC1/2/3` — per-slot (non-constitution only)
- Constitution: `constitution_document_name_SC1/2/3`, `constitution_year_SC1/2/3`, `constitution_document_types_SC1/2/3`
- `{ind}_agreement` — ratio of votes on the majority label (0.0–1.0)
- `{ind}_uncertainty` — `none` | `low` | `high`

With CoVe: `{ind}_prediction` = CoVe-revised result; `{ind}_verification` = CoVe details

## Prompt Modes

| Mode | API calls/row | Description |
|------|-------------|-------------|
| `single` (default) | 1 | All indicators in one unified prompt |
| `multiple` | N (one per indicator) | Independent call per indicator |
| `sequential` | 1 | All indicators as distinct sequential sections in one prompt |

**SC interaction**: in `single`/`sequential` mode, each SC sample is 1 extra call for all indicators. In `multiple` mode, each SC sample is 1 call per indicator.

## Sanity Check & Reprocessing

```bash
# Identify and reprocess failed rows (default: self-consistency, indicators pipeline)
python utils/sanity_check.py \
    -i data/results/predictions.csv \
    -o data/results/predictions_fixed.csv \
    --indicator constitution

# With confidence threshold
python utils/sanity_check.py \
    -i data/results/predictions.csv \
    -o data/results/predictions_fixed.csv \
    --indicator sovereign \
    --min-confidence 50
```

| Argument | Default |
|----------|---------|
| `--pipeline` | `indicators` |
| `--mode` | `single` |
| `--verify` | `self_consistency` |
| `--n-samples` | `2` |
| `--model` | `gemini-3.1-pro-preview` |

## Using run.sh

```bash
chmod +x run.sh

./run.sh --check-env                        # Validate environment
./run.sh --quick-test                       # 5-row test
./run.sh --pipeline indicators \
  --indicators sovereign assembly \
  --test 20

# Background mode (survives sleep/logout)
./run.sh --background --pipeline indicators \
  --indicators sovereign assembly entry exit \
  --input data/plt_leaders_data.csv \
  --output data/results/full_run.csv

./run.sh --dry-run --pipeline indicators --indicators sovereign --test 5
./run.sh --notify --pipeline indicators --indicators sovereign --test 10
```

## Evaluation

```python
from evaluation.notebook_utils import quick_eval, compare_experiments

# Single file
df, summary = quick_eval('data/results/predictions.csv')

# Compare multiple experiments
datasets = {
    'Baseline': 'data/results/baseline.csv',
    'Self-Consistency': 'data/results/sc.csv',
}
binary_metrics, multiclass_metrics = compare_experiments(datasets)
```

Indicators with ground truth: `sovereign`, `collegiality`, `assembly`

## API Reference

```python
from pipeline.predictor import Predictor, PredictionConfig
from config import PromptMode, VerificationType

config = PredictionConfig(
    mode=PromptMode.SINGLE,
    indicators=['sovereign', 'assembly'],
    verify=VerificationType.SELF_CONSISTENCY,
    model='gemini-3.1-pro-preview',
    sc_n_samples=2,
    sc_temperatures=[1.0, 1.0]
)

predictor = Predictor(config, api_keys)
result = predictor.predict("Roman Republic", "Julius Caesar", -49, -44)
print(result.predictions['sovereign'].prediction)
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `API key not provided` | Check `.env` has the right key |
| `ModuleNotFoundError` | Run `uv sync` or `pip install -r requirements.txt` |
| Bedrock throttling | Add `--delay 2.0` |
| JSON parsing failures | Check raw responses in `data/logs/` |

Set `CONSTITUTION_DEBUG=1` for verbose output.

## Additional Documentation

- **[docs/CODEBOOK.md](docs/CODEBOOK.md)** — Output dataset column codebook
- **[docs/BEDROCK_SETUP.md](docs/BEDROCK_SETUP.md)** — AWS Bedrock configuration guide
- **[docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md)** — Evaluation methodology
- **[CLAUDE.md](CLAUDE.md)** — Full project design doc and architecture

## License

MIT License — see LICENSE file for details.
