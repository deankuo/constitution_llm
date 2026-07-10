# Constitution Analysis Pipeline

A pipeline for predicting political indicators for historical pre-modern polities using LLMs, with optional self-consistency verification, Chain of Verification, and web search augmentation.

## Features

- **Multi-LLM Support**: Google Gemini (default: `gemini-3.1-pro-preview`), OpenAI GPT, Anthropic Claude, AWS Bedrock
- **Political Indicators**: Constitution, Sovereign, Federalism, nine binary Checks sub-indicators, Collegiality, Petition, Assembly, Entry, Exit, Symbolism, Elections (downstream)
- **Three Tasks**: `constitution`, `indicators`, and `elections` — run separately, merged into one dataset
- **Verification**: Self-Consistency (opt-in via `--verify self_consistency --n-samples N`) and Chain of Verification (CoVe, cross-model)
- **Prompt Modes**: Single (default, all indicators in one call), Multiple (one call per indicator), Sequential (single prompt with shuffled/user-defined section order)
- **Prompt Versions**: `v1` (full definitions), `v2` (expert-annotator persona), `v3` (compact) via `--prompt-version`
- **Search Modes**: None (default), Agentic (Serper tool calling), Forced (Wikipedia → DuckDuckGo → Serper tiered), Gemini Grounding (native Google Search)
- **Gemini Batch API**: 50% cost savings via the standalone `build_batch_jsonl.py` → `jsonl_batch_runner.py` flow, with automatic failed-row detection and retry-merge
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
# 5-row test (default: single mode, gemini-3.1-pro-preview, no verification)
python main.py --pipeline indicators --indicators sovereign assembly --test 5

# Or use run.sh
./run.sh --quick-test
```

## Full-Dataset Runs: Gemini Batch API (recommended)

Batch runs use a standalone two-step flow (NOT `main.py`): build a request
JSONL, then submit it. This gives 50% cost savings, server-side parallelism,
per-row failure detection, and a reproducible retry loop.

```bash
# Step 1 — build the request JSONL (example: full-dataset indicators run,
# native Google Search grounding, pure prediction without reasoning)
python src/build_batch_jsonl.py \
    --task indicators \
    --input data/plt_leaders_data.csv \
    --output data/temp/batch_indicators.jsonl \
    --search-mode gemini_grounding \
    --reasoning false

# Step 2 — submit and collect (writes CSV + JSON + provenance)
python pipeline/jsonl_batch_runner.py \
    --input data/temp/batch_indicators.jsonl \
    --dataset data/plt_leaders_data.csv \
    --output data/results/exp001.csv

# Step 3 — retry failed rows (repeat until no _failed_requests.jsonl is produced).
# IMPORTANT: --dataset must be the PREVIOUS run's output, not the raw input.
python pipeline/jsonl_batch_runner.py \
    --input data/results/exp001_failed_requests.jsonl \
    --dataset data/results/exp001.csv \
    --output data/results/exp001.csv
```

Notes on the batch flow:
- `--task` is one of `constitution`, `indicators`, `elections`. Elections requires
  an input containing `assembly_prediction` (run it on a prior output).
- `--search-mode gemini_grounding` embeds Gemini's native Google Search tool; the
  runner writes the queries/URLs into `{task}_search_queries` / `{task}_urls_used`
  columns (always present; NA when no search occurred for a row).
- `--reasoning false` removes reasoning from both the prompt AND the output columns.
- `--n-samples N` (default 0) embeds self-consistency: N additional calls per row,
  majority vote across N+1 votes, `_SC1..SC{N+1}`/`_agreement`/`_uncertainty` columns.
- The runner writes `{output}_provenance.json` (model, timestamps, failed rows) and,
  when rows fail, `{output}_failed_requests.jsonl` ready to resubmit.
- To chain the three tasks into one dataset, point each run's `--dataset` at the
  previous output: constitution → `exp001.csv`, indicators (`--dataset exp001.csv`)
  → `exp002.csv`, elections built from `exp002.csv` → final. Do not reorder or
  filter the CSV between rounds — merging is positional by row index.

## Pipelines

### Indicators Pipeline (main, synchronous)

Predicts any combination of indicators at the leader level. Used for experiments
and smaller samples; works with any provider (Gemini/GPT/Claude/Bedrock).

```bash
# Default settings: single mode, no verification
python main.py --pipeline indicators \
    --indicators sovereign federalism checks_local checks_military checks_clergy \
                 checks_aristocracy checks_bourgeoisie checks_bureaucracy \
                 checks_judiciary checks_assembly checks_council \
                 collegiality petition assembly entry exit symbolism \
    --input data/plt_leaders_data.csv \
    --output data/results/exp001.csv

# Constitution runs as its own task (multiple mode — it has a dedicated prompt)
python main.py --pipeline indicators \
    --mode multiple \
    --indicators constitution \
    --output data/results/exp002.csv

# Prompt persona variants (single/sequential mode)
python main.py --pipeline indicators \
    --indicators sovereign assembly \
    --prompt-version v2 \
    --output data/results/exp003.csv

# Sequential mode: the SAME single prompt with shuffled section order
python main.py --pipeline indicators \
    --mode sequential \
    --indicators sovereign assembly collegiality entry exit \
    --random-sequence \
    --output data/results/exp004.csv

# Self-consistency (opt-in): 2 extra samples = 3 total votes
python main.py --pipeline indicators \
    --indicators assembly \
    --verify self_consistency --n-samples 2 \
    --verify-indicators assembly \
    --output data/results/exp005.csv

# CoVe verification for constitution
python main.py --pipeline indicators \
    --mode multiple \
    --indicators constitution \
    --verify cove \
    --verify-indicators constitution \
    --verifier-model us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --output data/results/exp006.csv

# Forced search (works with any provider)
python main.py --pipeline indicators \
    --mode multiple \
    --indicators sovereign assembly \
    --search-mode forced \
    --output data/results/exp007.csv

# Parallel row processing (4 rows at once)
python main.py --pipeline indicators \
    --indicators sovereign assembly \
    --parallel-rows 4 \
    --output data/results/exp008.csv
```

### Constitution Pipeline (legacy, polity level)

Single-model, constitution-only, polity-level predictions.

```bash
python main.py --pipeline constitution \
    --models gemini-3.1-pro-preview \
    --input data/plt_polity_data_v2.csv \
    --output data/results/const_exp001.csv
```

### Downstream Classifiers (post_processing.py)

Run **after** the main pipeline. Elections depends on `assembly_prediction = 2`;
other rows pass through with `elections = 0`.

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

### Indicators Pipeline Arguments (main.py)

| Argument | Description | Default |
|----------|-------------|---------|
| `--pipeline` | `indicators` or `constitution` | `indicators` |
| `--mode` | `single`, `multiple`, or `sequential` | `single` |
| `--indicators` | Space-separated list of indicators | `constitution` |
| `--models` | Model identifier | `gemini-3.1-pro-preview` |
| `--prompt-version` | `v1`, `v2`, `v3` (single/sequential only) | `v1` |
| `--verify` | `none`, `self_consistency`, `cove`, `both` | `none` |
| `--verify-indicators` | Which indicators to verify (omit = all) | all `--indicators` |
| `--verifier-model` | Model for CoVe | Bedrock Claude (from `BEDROCK_VERIFIER_MODEL`) |
| `--n-samples` | Additional SC samples (total votes = n+1) | `0` |
| `--sc-temperatures` | Temperature list for SC samples | `1.0 1.0 1.0` |
| `--search-mode` | `none`, `agentic`, `forced`, `gemini_grounding` | `none` |
| `--parallel-rows` | Concurrent row workers | `1` |
| `--checkpoint-interval` | Rows per checkpoint | `500` |
| `--sequence` | Section order for sequential mode | None |
| `--random-sequence` | Randomize sequential order | `False` |
| `--reasoning` | Include reasoning columns | `True` |
| `--logprobs` | Token-level log probabilities (Gemini only) | `False` |
| `--input` | Input CSV or JSONL | `data/plt_leaders_data.csv` |
| `--output` | Output CSV path | `data/results/llm_predictions.csv` |
| `--test` | Process first N rows (or `start:end` range) | None |
| `--delay` | Seconds between API calls | `1.0` |
| `--temperature` | LLM temperature | `1.0` |

Notes:
- `constitution` cannot be combined with `--mode single`/`sequential` (it has its
  own dedicated prompt) — the CLI errors with guidance to use `--mode multiple`.
- Gemini Batch API runs do not go through `main.py`; see the batch flow above.

### Batch Build Arguments (src/build_batch_jsonl.py)

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | `constitution`, `indicators`, `elections` | (required) |
| `--indicators` | Indicators for the indicators task | all non-constitution |
| `--n-samples` | Additional SC samples per row | `0` |
| `--prompt-version` | `v1`, `v2`, `v3` | `v1` |
| `--reasoning` | Include reasoning in prompt AND output | `true` |
| `--search-mode` | `none`, `pre_search`, `gemini_grounding` | `none` |
| `--original-idx-col` | Column holding original row indices (subset re-runs) | None |
| `--test` | Build only first N rows | None |

## Architecture

### System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                       SYSTEM ARCHITECTURE                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Three tasks, merged into one dataset:                           │
│    constitution │ indicators (incl. 9 checks_*) │ elections      │
│                                                                  │
│  ┌───────────────────────────┐  ┌─────────────────────────────┐  │
│  │ SYNCHRONOUS (main.py)     │  │ BATCH (standalone, Gemini)  │  │
│  │ experiments/small samples │  │ full-dataset runs           │  │
│  │ any provider              │  │                             │  │
│  │                           │  │ src/build_batch_jsonl.py    │  │
│  │ Prompt Layer              │  │   --task --search-mode      │  │
│  │  single (v1/v2/v3)        │  │   --n-samples --reasoning   │  │
│  │  multiple (per indicator) │  │          │                  │  │
│  │  sequential (= single     │  │          ▼                  │  │
│  │   with shuffled sections) │  │ pipeline/jsonl_batch_       │  │
│  │          │                │  │   runner.py                 │  │
│  │          ▼                │  │  • 50% cost savings         │  │
│  │ Search Layer (optional)   │  │  • grounding metadata       │  │
│  │  agentic | forced |       │  │  • SC embedded              │  │
│  │  gemini_grounding         │  │  • failed-row detection     │  │
│  │          │                │  │  • retry-merge loop         │  │
│  │          ▼                │  │  • provenance JSON          │  │
│  │ Model Layer (BaseLLM)     │  └──────────┬──────────────────┘  │
│  │  Gemini | GPT | Claude |  │             │                     │
│  │  Bedrock                  │             │                     │
│  │          │                │             │                     │
│  │          ▼                │             │                     │
│  │ Verification (opt-in)     │             │                     │
│  │  Self-Consistency | CoVe  │             │                     │
│  └──────────┬────────────────┘             │                     │
│             └───────────────┬──────────────┘                     │
│                             ▼                                    │
│  Output: CSV + JSON · task-prefixed search columns ·             │
│          cost tracking · experiment log                          │
│                             │                                    │
│                             ▼  (run separately)                  │
│  Downstream: elections (assembly=2 → LLM; else pass-through 0)   │
│  via pipeline/post_processing.py or batch --task elections       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
constitution_llm/
│
├── CLAUDE.md                      # Project design doc and specs
├── README.md                      # This file
├── main.py                        # CLI entry point (synchronous pipelines)
├── config.py                      # Global settings, enums, constants
├── requirements.txt               # Python dependencies
├── run.sh                         # Shell wrapper (background, env check, etc.)
│
├── prompts/
│   ├── base_builder.py            # BasePromptBuilder ABC + PromptOutput
│   ├── constitution.py            # Constitution prompt (leader-level, 4 elements)
│   ├── polity_constitution.py     # Legacy polity-level constitution prompt
│   ├── indicators.py              # Per-indicator prompts (multiple mode)
│   ├── single_builder.py          # Unified prompt, versions v1/v2/v3
│   ├── multiple_builder.py        # Separate prompt per indicator
│   └── sequential_builder.py      # Single prompt with shuffled section order
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
│   ├── batch_runner.py            # Synchronous batch processing + checkpoints
│   ├── jsonl_batch_runner.py      # Gemini Batch API runner (standalone)
│   ├── search_predictor.py        # Agentic search-augmented predictions
│   ├── pre_search.py              # Deterministic tiered pre-search
│   └── post_processing.py         # Downstream classifiers (elections)
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
│   ├── build_batch_jsonl.py       # Build Gemini Batch API request JSONL
│   ├── diagnose_batch.py          # Batch output debugging helper
│   └── csv_to_jsonl.py            # CSV to JSONL conversion
│
├── notebooks/                     # Workflow demonstration notebooks (main.ipynb)
│
├── docs/
│   ├── BEDROCK_SETUP.md           # AWS Bedrock configuration guide
│   ├── EVALUATION_GUIDE.md        # Evaluation methodology
│   ├── MISSING_VALUES.md          # Missing value handling
│   └── codebook/                  # LaTeX codebook
│
└── data/
    ├── plt_leaders_data.csv       # Leader-level input data
    ├── plt_polity_data_v2.csv     # Polity-level input data
    ├── temp/                      # Batch request JSONL + build manifests
    ├── results/                   # Output directory (+ provenance/retry files)
    └── logs/
        ├── experiments.jsonl      # Append-only experiment log
        └── run_*.log              # Background run logs
```

## Output Format

**Plain mode (no SC):** `{ind}_prediction`, `{ind}_confidence`, and — only when
`--reasoning true` — `{ind}_reasoning`. With `--reasoning false`, reasoning
columns are omitted entirely (not written empty).

**With SC** (`--verify self_consistency --n-samples N` sync, or `--n-samples N` batch; total N+1 votes):
- `{ind}_prediction` — majority vote from all N+1 SC slots
- `{ind}_SC1` — initial call; `{ind}_SC2..SC{N+1}` — additional samples
- `{ind}_reasoning_SC{n}` (if reasoning), `{ind}_confidence_SC{n}` — per-slot (non-constitution)
- Constitution: `constitution_document_name_SC{n}`, `constitution_year_SC{n}`, `constitution_document_types_SC{n}`
- `{ind}_agreement` — ratio of votes on the majority label (0.0–1.0)
- `{ind}_uncertainty` — `none` (unanimous) | `low` (majority ≥ 2) | `high` (all differ; falls back to SC1)

**With CoVe:** `{ind}_prediction` = CoVe-revised result; `{ind}_verification` = CoVe details

**Search metadata (task-prefixed, so merged task outputs never collide):**
- `indicators_search_queries` / `indicators_urls_used` — indicators task
- `constitution_search_queries` / `constitution_urls_used` — constitution task
- `elections_search_queries` / `elections_urls_used` — elections task
- With SC + grounding, per-slot: `{task}_search_queries_SC{n}` / `{task}_urls_used_SC{n}`
- When grounding is enabled the columns are always present; rows where Gemini
  did not search read as NA.
- Forced/agentic sync runs also add `{task}_web_information` (single/sequential mode).

## Prompt Modes

| Mode | API calls/row | Description |
|------|-------------|-------------|
| `single` (default) | 1 | All indicators in one unified prompt (`--prompt-version v1/v2/v3`) |
| `multiple` | N (one per indicator) | Independent detailed prompt per indicator |
| `sequential` | 1 | The single prompt with sections in user-defined (`--sequence`) or shuffled (`--random-sequence`) order — isolates order effects |

Notes:
- `constitution` is excluded from `single`/`sequential` (it has its own prompt); run it via `multiple` mode or the batch constitution task.
- **SC interaction**: in `single`/`sequential` mode, each SC sample is 1 extra call shared by all indicators. In `multiple` mode, each SC sample is 1 call per indicator.

## Sanity Check & Reprocessing

```bash
# Identify and reprocess failed rows
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
| `--verify` | `none` |
| `--n-samples` | `0` |
| `--model` | `gemini-3.1-pro-preview` |

For batch runs, prefer the built-in retry loop of `jsonl_batch_runner.py`
(`_failed_requests.jsonl`); use `build_batch_jsonl.py --original-idx-col` for
subset re-runs identified by sanity_check.

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
    sc_temperatures=[1.0, 1.0],
    prompt_version='v1',
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
| Batch rows failed | Resubmit `{output}_failed_requests.jsonl` with `--dataset` = previous output |

Set `CONSTITUTION_DEBUG=1` for verbose output.

## Additional Documentation

- **[docs/BEDROCK_SETUP.md](docs/BEDROCK_SETUP.md)** — AWS Bedrock configuration guide
- **[docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md)** — Evaluation methodology
- **[docs/MISSING_VALUES.md](docs/MISSING_VALUES.md)** — Missing value handling
- **[CLAUDE.md](CLAUDE.md)** — Full project design doc and architecture

## License

MIT License — see LICENSE file for details.
