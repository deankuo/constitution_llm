#!/bin/bash

# Constitution Analysis Pipeline Shell Script
# Wrapper script for easy execution of the constitution analysis pipeline

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

# Functions for colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Show usage information
show_usage() {
    print_header "Constitution Analysis Pipeline"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Common Options:"
    echo "  -i, --input <path>           Input CSV file path"
    echo "                               (default: ./Dataset/polity_level_data.csv)"
    echo ""
    echo "  -o, --output <path>          Output CSV file path"
    echo "                               (default: ./Dataset/llm_predictions.csv)"
    echo ""
    echo "  -m, --models <models>        Model specifications (KEY=IDENTIFIER format)"
    echo "                               Examples:"
    echo "                                 GPT=gpt-4o"
    echo "                                 Claude=claude-3-5-sonnet-20241022"
    echo "                                 Gemini=gemini-2.5-pro"
    echo "                               (default: Gemini=gemini-2.5-pro)"
    echo ""
    echo "  -t, --test <N>               Test mode: process only first N entries"
    echo "                               Examples: --test 10, --test 5:15"
    echo ""
    echo "  --use-search                 Enable web search (requires SERPER_API_KEY)"
    echo ""
    echo "Advanced Options:"
    echo "  -d, --delay <seconds>        Delay between API calls (default: 1.0)"
    echo "  -b, --batch-size <size>      Batch size for checkpoints (default: 100)"
    echo "  --temperature <value>        LLM temperature (default: 0)"
    echo "  --max-tokens <value>         Max tokens for response (default: 4096)"
    echo "  --max-retries <value>        Max retries for failed calls (default: 3)"
    echo ""
    echo "Other Options:"
    echo "  -h, --help                   Show this help message"
    echo "  --check-env                  Check environment setup"
    echo ""
    echo "Examples:"
    echo ""
    echo "  # Quick test with 5 entries using default model"
    echo "  $0 --test 5"
    echo ""
    echo "  # Use GPT-4 with web search"
    echo "  $0 -m GPT=gpt-4o --use-search"
    echo ""
    echo "  # Compare multiple models"
    echo "  $0 -m GPT=gpt-4o Claude=claude-3-5-sonnet-20241022 Gemini=gemini-2.5-pro"
    echo ""
    echo "  # Custom input/output with specific delay"
    echo "  $0 -i custom_data.csv -o results.csv --delay 2.0"
    echo ""
    echo "  # Full production run with all settings"
    echo "  $0 -i data.csv -o results.csv -m GPT=gpt-4o -d 1.5 --temperature 0.2"
    echo ""
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY              OpenAI API key"
    echo "  ANTHROPIC_API_KEY           Anthropic/Claude API key"
    echo "  GEMINI_API_KEY              Google Gemini API key"
    echo "  AWS_ACCESS_KEY_ID           AWS Access Key (for Bedrock)"
    echo "  AWS_SECRET_ACCESS_KEY       AWS Secret Key (for Bedrock)"
    echo "  SERPER_API_KEY              Serper API key (for web search)"
    echo ""
}

# Check environment setup
check_environment() {
    print_header "Environment Check"
    echo ""

    local all_good=true

    # Check Python
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python 3 found: version $python_version"
    else
        print_error "Python 3 not found"
        all_good=false
    fi

    # Check main.py
    if [ -f "main.py" ]; then
        print_success "main.py found"
    else
        print_error "main.py not found in current directory"
        all_good=false
    fi

    # Check requirements.txt
    if [ -f "requirements.txt" ]; then
        print_success "requirements.txt found"
    else
        print_warning "requirements.txt not found"
    fi

    # Check .env file
    if [ -f ".env" ]; then
        print_success ".env file found"
    else
        print_warning ".env file not found (API keys should be set as environment variables)"
    fi

    # Check API keys
    echo ""
    print_info "Checking API keys..."

    if [ -n "$OPENAI_API_KEY" ]; then
        print_success "OPENAI_API_KEY is set"
    else
        print_warning "OPENAI_API_KEY is not set"
    fi

    if [ -n "$ANTHROPIC_API_KEY" ]; then
        print_success "ANTHROPIC_API_KEY is set"
    else
        print_warning "ANTHROPIC_API_KEY is not set"
    fi

    if [ -n "$GEMINI_API_KEY" ]; then
        print_success "GEMINI_API_KEY is set"
    else
        print_warning "GEMINI_API_KEY is not set"
    fi

    if [ -n "$AWS_ACCESS_KEY_ID" ]; then
        print_success "AWS_ACCESS_KEY_ID is set"
    else
        print_warning "AWS_ACCESS_KEY_ID is not set"
    fi

    if [ -n "$SERPER_API_KEY" ]; then
        print_success "SERPER_API_KEY is set (web search available)"
    else
        print_warning "SERPER_API_KEY is not set (web search unavailable)"
    fi

    echo ""
    if [ "$all_good" = true ]; then
        print_success "Environment check complete!"
    else
        print_error "Some environment checks failed. Please fix the issues above."
        exit 1
    fi
}

# Initialize variables
INPUT_PATH=""
OUTPUT_PATH=""
MODELS=""
TEST_SIZE=""
DELAY=""
BATCH_SIZE=""
TEMPERATURE=""
MAX_TOKENS=""
MAX_RETRIES=""
USE_SEARCH=""
PYTHON_SCRIPT="main.py"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -m|--models)
            shift
            MODELS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                MODELS="$MODELS $1"
                shift
            done
            ;;
        -t|--test)
            TEST_SIZE="$2"
            shift 2
            ;;
        -d|--delay)
            DELAY="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --use-search)
            USE_SEARCH="--use-search"
            shift
            ;;
        --check-env)
            check_environment
            exit 0
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    print_info "Make sure you're running this script from the project root directory"
    exit 1
fi

# Check if input file exists (only if specified)
if [ -n "$INPUT_PATH" ] && [ ! -f "$INPUT_PATH" ]; then
    print_error "Input file does not exist: $INPUT_PATH"
    exit 1
fi

# Create output directory if needed
if [ -n "$OUTPUT_PATH" ]; then
    OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
    if [ ! -d "$OUTPUT_DIR" ]; then
        print_warning "Output directory does not exist. Creating: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    fi
fi

# Build Python command
PYTHON_CMD="python3 $PYTHON_SCRIPT"

if [ -n "$INPUT_PATH" ]; then
    PYTHON_CMD="$PYTHON_CMD --input \"$INPUT_PATH\""
fi

if [ -n "$OUTPUT_PATH" ]; then
    PYTHON_CMD="$PYTHON_CMD --output \"$OUTPUT_PATH\""
fi

if [ -n "$MODELS" ]; then
    PYTHON_CMD="$PYTHON_CMD --models$MODELS"
fi

if [ -n "$TEST_SIZE" ]; then
    PYTHON_CMD="$PYTHON_CMD --test $TEST_SIZE"
fi

if [ -n "$DELAY" ]; then
    PYTHON_CMD="$PYTHON_CMD --delay $DELAY"
fi

if [ -n "$BATCH_SIZE" ]; then
    PYTHON_CMD="$PYTHON_CMD --batch-size $BATCH_SIZE"
fi

if [ -n "$TEMPERATURE" ]; then
    PYTHON_CMD="$PYTHON_CMD --temperature $TEMPERATURE"
fi

if [ -n "$MAX_TOKENS" ]; then
    PYTHON_CMD="$PYTHON_CMD --max_tokens $MAX_TOKENS"
fi

if [ -n "$MAX_RETRIES" ]; then
    PYTHON_CMD="$PYTHON_CMD --max-retries $MAX_RETRIES"
fi

if [ -n "$USE_SEARCH" ]; then
    PYTHON_CMD="$PYTHON_CMD $USE_SEARCH"
fi

# Display configuration
print_header "Constitution Analysis Pipeline - Starting"
echo ""
print_info "Configuration:"
echo "  Input:        ${INPUT_PATH:-Using default (./Dataset/polity_level_data.csv)}"
echo "  Output:       ${OUTPUT_PATH:-Using default (./Dataset/llm_predictions.csv)}"
echo "  Models:       ${MODELS:-Using default (Gemini=gemini-2.5-pro)}"
echo "  Test mode:    ${TEST_SIZE:-Disabled (processing all data)}"
echo "  Web search:   ${USE_SEARCH:-Disabled}"
echo ""

# Execute the Python script
print_info "Executing command:"
echo "  $PYTHON_CMD"
echo ""

print_info "Starting analysis..."
start_time=$(date +%s)

if eval $PYTHON_CMD; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))

    echo ""
    print_header "Analysis Complete"
    print_success "Total execution time: ${minutes}m ${seconds}s"

    # Show output file info if it exists
    if [ -n "$OUTPUT_PATH" ] && [ -f "$OUTPUT_PATH" ]; then
        line_count=$(wc -l < "$OUTPUT_PATH")
        print_info "Output file: $OUTPUT_PATH"
        print_info "Contains $((line_count - 1)) data rows (plus header)"
    fi

    echo ""
    print_success "Pipeline completed successfully!"
else
    print_error "Pipeline failed! Check the error messages above."
    exit 1
fi
