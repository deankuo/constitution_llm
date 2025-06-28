#!/bin/bash

# Constitution Analysis Pipeline Shell Script
# This shell script is for executing the constitution analysis prompt engineering process.

# Set script to exit on any error
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to show usage
show_usage() {
    echo "Constitution Analysis Pipeline"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Optional Arguments (all have defaults in Python script):"
    echo "  -i, --input <path>          Input CSV file (default: ./Dataset/polity_level_data.csv)"
    echo "  -o, --output <path>         Output CSV file (default: ./Dataset/llm_predictions.csv)"
    echo "  -u, --user-prompt <text>    Custom user prompt template"
    echo "  -s, --system-prompt <text>  Custom system prompt"
    echo "  -m, --models <models>       Model names (space-separated, default: gpt-4.1-nano)"
    echo "  -k, --api-key <key>         API key (can also use OPENAI_API_KEY env var)"
    echo "  -b, --batch-size <size>     Batch size for temporary saves (default: 10)"
    echo "  -d, --delay <seconds>       Delay between API calls (default: 1.0)"
    echo "  -t, --test <number>         Test mode: process only first N entries"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                              # Use all defaults"
    echo "  $0 -i my_data.csv -o my_results.csv"
    echo "  $0 -m \"gpt-3.5-turbo gpt-4\" -k your_api_key"
    echo "  $0 -t 5 -d 2.0                                  # Test with 5 entries, 2s delay"
    echo "  $0 -s \"You are a historian\" -u \"Analyze: {country}\""
    echo ""
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY              OpenAI API key"
    echo ""
    echo "Note: All arguments are optional due to defaults in Python script"
}

# Initialize variables (empty means use Python script defaults)
INPUT_PATH=""
OUTPUT_PATH=""
USER_PROMPT=""
SYSTEM_PROMPT=""
MODELS=""
API_KEY=""
BATCH_SIZE=""
DELAY=""
TEST_SIZE=""
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
        -u|--user-prompt)
            USER_PROMPT="$2"
            shift 2
            ;;
        -s|--system-prompt)
            SYSTEM_PROMPT="$2"
            shift 2
            ;;
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -d|--delay)
            DELAY="$2"
            shift 2
            ;;
        -t|--test)
            TEST_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if required arguments are provided (now all are optional due to Python defaults)
# Only check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    print_info "Make sure $PYTHON_SCRIPT is in the current directory"
    exit 1
fi

# Check if input file exists (only if specified)
if [ -n "$INPUT_PATH" ] && [ ! -f "$INPUT_PATH" ]; then
    print_error "Input file does not exist: $INPUT_PATH"
    exit 1
fi

# Check if output directory exists, create if not (only if output path specified)
if [ -n "$OUTPUT_PATH" ]; then
    OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
    if [ ! -d "$OUTPUT_DIR" ]; then
        print_warning "Output directory does not exist. Creating: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    fi
fi

# Check for API key
if [ -z "$API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    print_warning "No API key provided. Make sure OPENAI_API_KEY environment variable is set"
fi

# Print configuration
print_info "Starting Constitution Analysis Pipeline"
print_info "Configuration:"
if [ -n "$INPUT_PATH" ]; then
    echo "  Input file: $INPUT_PATH"
else
    echo "  Input file: Using Python default (./Dataset/polity_level_data.csv)"
fi

if [ -n "$OUTPUT_PATH" ]; then
    echo "  Output file: $OUTPUT_PATH"
else
    echo "  Output file: Using Python default (./Dataset/llm_predictions.csv)"
fi

if [ -n "$MODELS" ]; then
    echo "  Models: $MODELS"
else
    echo "  Models: Using Python default (gpt-4.1-nano)"
fi

if [ -n "$USER_PROMPT" ]; then
    echo "  User prompt: Custom prompt provided"
else
    echo "  User prompt: Using Python default template"
fi

if [ -n "$SYSTEM_PROMPT" ]; then
    echo "  System prompt: Custom prompt provided"
else
    echo "  System prompt: Using Python default"
fi

if [ -n "$BATCH_SIZE" ]; then
    echo "  Batch size: $BATCH_SIZE"
else
    echo "  Batch size: Using Python default (10)"
fi

if [ -n "$DELAY" ]; then
    echo "  Delay: $DELAY seconds"
else
    echo "  Delay: Using Python default (1.0 seconds)"
fi

if [ -n "$TEST_SIZE" ]; then
    echo "  Test mode: Processing first $TEST_SIZE entries"
fi
echo ""

# Build Python command
PYTHON_CMD="python $PYTHON_SCRIPT"

# Add optional parameters only if they are provided
if [ -n "$INPUT_PATH" ]; then
    PYTHON_CMD="$PYTHON_CMD --input \"$INPUT_PATH\""
fi

if [ -n "$OUTPUT_PATH" ]; then
    PYTHON_CMD="$PYTHON_CMD --output \"$OUTPUT_PATH\""
fi

if [ -n "$USER_PROMPT" ]; then
    PYTHON_CMD="$PYTHON_CMD --user-prompt \"$USER_PROMPT\""
fi

if [ -n "$SYSTEM_PROMPT" ]; then
    PYTHON_CMD="$PYTHON_CMD --system-prompt \"$SYSTEM_PROMPT\""
fi

if [ -n "$MODELS" ]; then
    PYTHON_CMD="$PYTHON_CMD --models $MODELS"
fi

if [ -n "$API_KEY" ]; then
    PYTHON_CMD="$PYTHON_CMD --api-key \"$API_KEY\""
fi

if [ -n "$BATCH_SIZE" ]; then
    PYTHON_CMD="$PYTHON_CMD --batch-size $BATCH_SIZE"
fi

if [ -n "$DELAY" ]; then
    PYTHON_CMD="$PYTHON_CMD --delay $DELAY"
fi

if [ -n "$TEST_SIZE" ]; then
    PYTHON_CMD="$PYTHON_CMD --test $TEST_SIZE"
fi

# Print command for debugging
print_info "Executing command:"
echo "  $PYTHON_CMD"
echo ""

# Execute the Python script
print_info "Starting analysis..."
start_time=$(date +%s)

if eval $PYTHON_CMD; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    print_success "Analysis completed successfully!"
    print_info "Total execution time: ${duration} seconds"
    print_info "Results saved to: $OUTPUT_PATH"
    
    # Show basic statistics if output file exists
    if [ -f "$OUTPUT_PATH" ]; then
        line_count=$(wc -l < "$OUTPUT_PATH")
        print_info "Output file contains $((line_count - 1)) data rows (plus header)"
    fi
else
    print_error "Analysis failed!"
    exit 1
fi

# Clean up temporary files (optional)
temp_files=$(find . -name "temp_polity_results_*.csv" 2>/dev/null || true)
if [ -n "$temp_files" ]; then
    print_info "Cleaning up temporary files..."
    rm -f temp_polity_results_*.csv
    print_success "Temporary files removed"
fi

print_success "Script execution completed!"