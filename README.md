# Constitution Analysis Pipeline

A sophisticated pipeline for analyzing historical polities to determine whether they had constitutions during their periods of existence. The pipeline leverages multiple Large Language Model (LLM) providers to ensure accurate and comprehensive analysis.

## Features

- **Multi-LLM Support**: Seamlessly works with multiple LLM providers
  - OpenAI (GPT-4, GPT-4o, etc.)
  - Anthropic (Claude 3.5 Sonnet, etc.)
  - Google Gemini (Gemini 2.5 Pro, etc.)
  - AWS Bedrock (Claude on Bedrock, etc.)

- **Web Search Integration**: Optional web search capability using Serper API for enhanced accuracy

- **Concurrent Processing**: Process multiple models simultaneously for faster results

- **Robust Error Handling**: Automatic retries, checkpoint system, and comprehensive error logging

- **Flexible Configuration**: Customize temperature, max tokens, delays, and more

- **Batch Processing**: Efficient handling of large datasets with automatic checkpoints

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Using Shell Script](#using-shell-script)
  - [Using Python Directly](#using-python-directly)
  - [Advanced Usage](#advanced-usage)
- [Architecture](#architecture)
- [Input Data Format](#input-data-format)
- [Output Format](#output-format)
- [Utilities](#utilities)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

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

Create a `.env` file in the project root directory:

```bash
# OpenAI API Key (for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# AWS Credentials (for Bedrock models)
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_SESSION_TOKEN=your_aws_session_token_here  # Optional

# Serper API Key (for web search)
SERPER_API_KEY=your_serper_api_key_here  # Optional
```

You can use the `.env.example` file as a template:

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

## Configuration

All configuration constants are centralized in `utils/config.py`:

```python
# LLM Parameters
DEFAULT_TEMPERATURE = 0  # Temperature for generation
DEFAULT_MAX_TOKENS = 2048  # Maximum tokens per response
DEFAULT_TOP_P = 1.0  # Top-p sampling parameter

# Processing Parameters
DEFAULT_BATCH_SIZE = 100  # Checkpoint frequency
DEFAULT_DELAY = 1.0  # Delay between API calls (seconds)
DEFAULT_MAX_RETRIES = 3  # Maximum retry attempts
DEFAULT_RETRY_DELAY = 5  # Delay between retries (seconds)
```

## Usage

### Quick Start

The fastest way to test the pipeline:

```bash
# Make the shell script executable
chmod +x run.sh

# Run a quick test with 5 polities
./run.sh --test 5
```

### Using Shell Script

The `run.sh` script provides a user-friendly interface:

```bash
# Check your environment setup
./run.sh --check-env

# Use default settings (Gemini model, all data)
./run.sh

# Test mode with first 10 polities
./run.sh --test 10

# Use specific model
./run.sh -m GPT=gpt-4o

# Use multiple models simultaneously
./run.sh -m GPT=gpt-4o Claude=claude-3-5-sonnet-20241022 Gemini=gemini-2.5-pro

# Enable web search
./run.sh -m GPT=gpt-4o --use-search

# Custom input/output paths
./run.sh -i ./data/input.csv -o ./results/output.csv

# Advanced configuration
./run.sh -m Claude=claude-3-5-sonnet-20241022 --temperature 0.2 --max-tokens 4096 -d 2.0
```

### Using Python Directly

For more control, you can run the Python script directly:

```bash
# Basic usage
python main.py

# Custom input/output
python main.py --input data.csv --output results.csv

# Use specific model
python main.py --models GPT=gpt-4o

# Multiple models
python main.py --models GPT=gpt-4o Claude=claude-3-5-sonnet-20241022

# Test mode (first 10 polities)
python main.py --test 10

# Test mode (polities 100-150)
python main.py --test 100:150

# Enable web search
python main.py --models GPT=gpt-4o --use-search

# Custom LLM parameters
python main.py --temperature 0.7 --max_tokens 4096 --top_p 0.95

# Full configuration
python main.py \
  --input ./Dataset/polity_level_data.csv \
  --output ./Dataset/results.csv \
  --models GPT=gpt-4o Claude=claude-3-5-sonnet-20241022 \
  --temperature 0.2 \
  --max_tokens 4096 \
  --delay 1.5 \
  --max-retries 5 \
  --use-search
```

### Advanced Usage

#### Custom Prompts

You can customize the system and user prompts:

```bash
python main.py \
  --system_prompt "You are a constitutional historian..." \
  --user_prompt "Analyze {country} from {start_year} to {end_year}"
```

#### Model Naming Convention

Models are specified in `KEY=IDENTIFIER` format:

- **OpenAI**: `GPT=gpt-4o`, `GPT4=gpt-4-turbo`
- **Anthropic**: `Claude=claude-3-5-sonnet-20241022`
- **Gemini**: `Gemini=gemini-2.5-pro`, `Gemini15=gemini-1.5-pro`
- **Bedrock**: `Bedrock=arn:aws:bedrock:us-east-1::foundation-model/...`

The `KEY` becomes part of the output column names (e.g., `constitution_gpt`, `constitution_claude`).

## Architecture

### Project Structure

```
constitution_llm/
├── main.py                      # Main entry point
├── run.sh                       # Shell wrapper script
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (create this)
├── .env.example                 # Example environment file
├── README.md                    # This file
├── utils/
│   ├── config.py               # Configuration constants
│   ├── llm_clients.py          # LLM client implementations
│   ├── search_agents.py        # Web search agents
│   ├── prompt.py               # Prompt templates
│   ├── encoding_fix.py         # CSV encoding utilities
│   └── sanity_check.py         # Data validation utilities
└── Dataset/
    ├── polity_level_data.csv   # Input data (you provide)
    └── llm_predictions.csv     # Output results
```

### Module Overview

- **main.py**: Orchestrates the entire pipeline, handles data loading, processing, and saving
- **utils/config.py**: Centralized configuration management
- **utils/llm_clients.py**: Individual LLM provider implementations (OpenAI, Anthropic, Gemini, Bedrock)
- **utils/search_agents.py**: Web search-enabled agents for each provider
- **utils/prompt.py**: System and user prompt templates
- **utils/encoding_fix.py**: Tools for fixing CSV encoding issues
- **utils/sanity_check.py**: Data validation and reprocessing utilities

### Processing Flow

```
1. Load polity data from CSV
2. For each polity:
   a. Create prompt with polity information
   b. Query multiple LLMs concurrently
   c. Parse and validate responses
   d. Aggregate results
3. Save checkpoint at 50% completion
4. Save final results at 100% completion
5. Clean up temporary files
```

## Input Data Format

The input CSV must contain these required columns:

- `territorynamehistorical`: Name of the historical polity
- `start_year`: Start year of the polity period
- `end_year`: End year of the polity period

Example:

```csv
territorynamehistorical,start_year,end_year
France,1789,1799
United States,1776,1789
Ancient Rome,-500,476
```

## Output Format

The pipeline generates a CSV with the following columns:

### Input Columns (preserved)
- `territorynamehistorical`
- `start_year`
- `end_year`
- Any other columns from input file

### Generated Columns (per model)
- `constitution_{model}`: Binary indicator (1=Yes, 0=No, -1=Error)
- `constitution_year`: Year of earliest constitution
- `constitution_name_{model}`: Name of the constitutional document
- `explanation_{model}`: Detailed explanation from the model
- `explanation_length_{model}`: Length of explanation
- `confidence_score_{model}`: Confidence score (1-5)

Example:

```csv
territorynamehistorical,start_year,end_year,constitution_gpt,constitution_year,constitution_name_gpt,explanation_gpt,explanation_length_gpt,confidence_score_gpt
France,1789,1799,1,1791,Constitution of 1791,"...",1250,5
```

## Utilities

### Encoding Fix Utility

Fix CSV encoding issues:

```python
from utils.encoding_fix import convert_csv_to_utf8

# Convert single file
convert_csv_to_utf8(
    'data_latin1.csv',
    'data_utf8.csv',
    source_encoding='latin-1'
)

# Auto-detect encoding
convert_csv_to_utf8(
    'data.csv',
    'data_utf8.csv',
    source_encoding='autodetect'
)
```

### Sanity Check Utility

Validate and reprocess results:

```python
from utils.sanity_check import sanity_check_and_reprocess

# Check and reprocess failed rows
sanity_check_and_reprocess(
    input_csv='results.csv',
    output_csv='results_fixed.csv',
    min_confidence=3,
    min_length=100
)
```

## Troubleshooting

### Common Issues

#### 1. API Key Errors

**Error**: `ERROR: Anthropic API key not provided.`

**Solution**: Ensure your `.env` file contains the required API keys:

```bash
# Check if .env file exists
ls -la .env

# Verify API keys are set
cat .env | grep API_KEY
```

#### 2. Module Import Errors

**Error**: `ModuleNotFoundError: No module named 'anthropic'`

**Solution**: Install missing dependencies:

```bash
pip install -r requirements.txt
```

#### 3. Rate Limiting

**Error**: `WARN: Bedrock API throttling detected`

**Solution**: Increase the delay between API calls:

```bash
./run.sh --delay 2.0
```

#### 4. Encoding Issues

**Error**: `UnicodeDecodeError: 'utf-8' codec can't decode byte...`

**Solution**: Use the encoding fix utility:

```python
python utils/encoding_fix.py
```

#### 5. Web Search Not Working

**Error**: `Error: Serper API key is not configured.`

**Solution**: Add your Serper API key to `.env`:

```bash
echo "SERPER_API_KEY=your_key_here" >> .env
```

### Debug Mode

For detailed debugging, you can modify the print statements in the code or use Python's logging module.

### Getting Help

If you encounter issues:

1. Check the error messages carefully
2. Review the [Troubleshooting](#troubleshooting) section
3. Ensure all API keys are correctly configured
4. Verify input data format matches requirements
5. Try running with `--test 1` to isolate the issue

## Model-Specific Notes

### OpenAI (GPT)
- Supports all GPT models (gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.)
- Web search works via function calling
- Generally fast with good availability

### Anthropic (Claude)
- Supports Claude 3.5 Sonnet and other Claude models
- Excellent reasoning capabilities
- Web search via tool use

### Google Gemini
- Supports Gemini 2.5 Pro, 1.5 Pro, etc.
- Fast and cost-effective
- Web search via function declarations

### AWS Bedrock
- Supports Claude on Bedrock and other Bedrock models
- Requires AWS credentials
- Good for enterprise use cases
- May have throttling limits

## Performance Tips

1. **Use concurrent processing**: The pipeline automatically runs multiple models in parallel
2. **Adjust delay**: Balance between speed and rate limits (`--delay 1.0`)
3. **Use checkpoints**: The pipeline automatically saves at 50% and 100%
4. **Test first**: Always test with `--test 5` before running full dataset
5. **Choose appropriate models**: Different models have different speeds and costs

## Best Practices

1. **Always use version control** for your data and results
2. **Test with small datasets** before running on large datasets
3. **Monitor API usage** to avoid unexpected costs
4. **Keep API keys secure** (never commit `.env` to git)
5. **Document your experiments** (model versions, parameters used, etc.)
6. **Validate results** using the sanity check utility

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude models
- Google for Gemini models
- AWS for Bedrock platform
- Serper for web search API

## Contact

For questions or support, please open an issue on GitHub.

---

Made with ❤️ for constitutional history research
