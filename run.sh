#!/bin/bash

# ============================================================================
# Constitution Analysis Pipeline - Run Script
#
# A user-friendly wrapper for running the prediction pipeline.
# Supports both leader and polity pipelines with all configuration options.
#
# Usage:
#   ./run.sh --help                          # Show full help
#   ./run.sh --check-env                     # Verify environment setup
#   ./run.sh --quick-test                    # Quick test (5 rows, leader pipeline)
#   ./run.sh --pipeline leader [OPTIONS]     # Leader pipeline
#   ./run.sh --pipeline polity [OPTIONS]     # Polity pipeline
#
# To run in background (survives sleep/logout):
#   ./run.sh --background [OPTIONS]
#
# ============================================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Helpers ──────────────────────────────────────────────────────────────────

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

print_header() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ── Usage / Help ─────────────────────────────────────────────────────────────

show_usage() {
    print_header "Constitution Analysis Pipeline"
    cat <<'USAGE'

  A pipeline for predicting political indicators for historical polities
  using LLMs (Gemini, GPT, Claude) with optional web search and verification.

GETTING STARTED
  1. Install dependencies:   pip install -r requirements.txt
  2. Set API keys:           cp .env.example .env && edit .env
  3. Check setup:            ./run.sh --check-env
  4. Quick test:             ./run.sh --quick-test

USAGE
    echo -e "${BOLD}SYNOPSIS${NC}"
    echo "  ./run.sh [--pipeline leader|polity] [OPTIONS]"
    echo ""
    echo -e "${BOLD}PIPELINE SELECTION${NC}"
    echo "  --pipeline leader     Leader-level pipeline (all 8 indicators, default)"
    echo "  --pipeline polity     Polity-level pipeline (constitution only, multi-model)"
    echo ""
    echo -e "${BOLD}COMMON OPTIONS${NC}"
    echo "  -i, --input <path>              Input CSV file"
    echo "  -o, --output <path>             Output CSV file"
    echo "  -m, --models <model ...>        Model(s) — see examples below"
    echo "  -t, --test <N|start:end>        Test mode: first N rows or range"
    echo "  -d, --delay <sec>               Delay between API calls (default: 1.0)"
    echo "  --temperature <val>             LLM temperature (default: 0)"
    echo "  --max-tokens <val>              Max response tokens (default: 32768)"
    echo ""
    echo -e "${BOLD}LEADER PIPELINE OPTIONS${NC}"
    echo "  --mode <single|multiple|seq>    Prompt mode (default: multiple)"
    echo "  --indicators <ind ...>          Indicators to predict (space-separated)"
    echo "                                  Options: constitution sovereign assembly"
    echo "                                           appointment tenure exit"
    echo "                                           collegiality separate_powers"
    echo "  --reasoning <True|False>        Include reasoning columns (default: True)"
    echo "  --parallel-rows <N>             Concurrent rows (default: 1)"
    echo "  --checkpoint-interval <N>       Rows per checkpoint/batch (default: 500)"
    echo ""
    echo -e "${BOLD}SEARCH OPTIONS${NC}"
    echo "  --search-mode <none|agentic|forced>"
    echo "                                  none    = LLM only (default)"
    echo "                                  agentic = LLM decides when to search"
    echo "                                  forced  = always search before LLM"
    echo ""
    echo -e "${BOLD}BATCH API (GEMINI ONLY)${NC}"
    echo "  --use-batch                     Submit via Gemini Batch API (50% savings)"
    echo "                                  Incompatible with --search-mode agentic"
    echo ""
    echo -e "${BOLD}VERIFICATION OPTIONS${NC}"
    echo "  --verify <none|self_consistency|cove|both>"
    echo "  --verify-indicators <ind ...>   Which indicators to verify"
    echo "  --verifier-model <model>        Model for CoVe verification"
    echo "  --n-samples <N>                 Self-consistency samples (default: 3)"
    echo ""
    echo -e "${BOLD}SEQUENTIAL MODE OPTIONS${NC}"
    echo "  --sequence <ind ...>            Custom indicator order"
    echo "  --random-sequence               Randomize indicator order"
    echo ""
    echo -e "${BOLD}BACKGROUND EXECUTION${NC}"
    echo "  --background                    Run with caffeinate + nohup (survives"
    echo "                                  sleep, screen lock, and terminal close)"
    echo "  --notify                        macOS notification when job finishes"
    echo ""
    echo -e "${BOLD}UTILITIES${NC}"
    echo "  --check-env                     Check environment (Python, API keys, etc.)"
    echo "  --quick-test                    Quick 5-row test with defaults"
    echo "  --dry-run                       Show the command without running it"
    echo "  -h, --help                      Show this help"
    echo ""
    echo -e "${BOLD}EXAMPLES${NC}"
    echo ""
    echo -e "  ${DIM}# Quick test (5 rows, leader pipeline, default model)${NC}"
    echo "  ./run.sh --quick-test"
    echo ""
    echo -e "  ${DIM}# Leader pipeline: predict 3 indicators for 20 rows${NC}"
    echo "  ./run.sh --pipeline leader \\"
    echo "      --indicators sovereign assembly exit \\"
    echo "      --models gemini-2.5-pro --test 20"
    echo ""
    echo -e "  ${DIM}# Leader pipeline: single mode, 4 parallel rows${NC}"
    echo "  ./run.sh --pipeline leader \\"
    echo "      --mode single \\"
    echo "      --indicators sovereign assembly appointment tenure exit \\"
    echo "      --parallel-rows 4 \\"
    echo "      -i data/plt_leaders_data.csv -o data/results/exp001.csv"
    echo ""
    echo -e "  ${DIM}# Leader pipeline: with forced search + batch API${NC}"
    echo "  ./run.sh --pipeline leader \\"
    echo "      --indicators sovereign assembly \\"
    echo "      --search-mode forced --use-batch \\"
    echo "      -i data/plt_leaders_data.csv -o data/results/search_exp.csv"
    echo ""
    echo -e "  ${DIM}# Leader pipeline: with self-consistency verification${NC}"
    echo "  ./run.sh --pipeline leader \\"
    echo "      --indicators assembly \\"
    echo "      --verify self_consistency --verify-indicators assembly \\"
    echo "      --n-samples 5 --test 10"
    echo ""
    echo -e "  ${DIM}# Polity pipeline: two models, with search${NC}"
    echo "  ./run.sh --pipeline polity \\"
    echo "      --models Gemini=gemini-2.5-pro GPT=gpt-4o \\"
    echo "      --search-mode agentic \\"
    echo "      -i data/plt_polity_data_v2.csv -o data/results/polity.csv"
    echo ""
    echo -e "  ${DIM}# Full production run in background (survives sleep)${NC}"
    echo "  ./run.sh --background --pipeline leader \\"
    echo "      --indicators sovereign assembly appointment tenure exit \\"
    echo "                   collegiality separate_powers \\"
    echo "      --parallel-rows 4 \\"
    echo "      -i data/plt_leaders_data.csv -o data/results/full_run.csv"
    echo ""
    echo -e "${BOLD}ENVIRONMENT VARIABLES${NC}"
    echo "  GEMINI_API_KEY              Google Gemini API key"
    echo "  OPENAI_API_KEY              OpenAI API key"
    echo "  ANTHROPIC_API_KEY           Anthropic/Claude API key"
    echo "  AWS_ACCESS_KEY_ID           AWS Access Key (for Bedrock)"
    echo "  AWS_SECRET_ACCESS_KEY       AWS Secret Key (for Bedrock)"
    echo "  SERPER_API_KEY              Serper API key (for agentic search)"
    echo ""
}

# ── Environment Check ────────────────────────────────────────────────────────

check_environment() {
    print_header "Environment Check"

    local all_good=true
    local warnings=0

    # Python
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python 3: $python_version"
    else
        print_error "Python 3 not found"
        all_good=false
    fi

    # Core files
    for f in main.py config.py requirements.txt; do
        if [ -f "$f" ]; then
            print_success "$f found"
        else
            print_error "$f not found (run from project root)"
            all_good=false
        fi
    done

    # .env
    if [ -f ".env" ]; then
        print_success ".env file found"
        # Source it so key checks below work
        set -a; source .env 2>/dev/null; set +a
    else
        print_warning ".env not found — API keys must be set as env vars"
        ((warnings++)) || true
    fi

    # Key Python packages
    echo ""
    print_info "Checking Python packages..."
    local missing_pkgs=0
    for pkg in pandas tqdm google.generativeai openai anthropic boto3; do
        if python3 -c "import $pkg" 2>/dev/null; then
            print_success "  $pkg"
        else
            print_warning "  $pkg not installed"
            ((missing_pkgs++)) || true
        fi
    done
    if [ $missing_pkgs -gt 0 ]; then
        echo ""
        print_info "Install missing packages: pip install -r requirements.txt"
    fi

    # API keys
    echo ""
    print_info "Checking API keys..."

    local key_count=0
    for key_name in GEMINI_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY; do
        if [ -n "${!key_name}" ]; then
            # Show first 8 chars only
            masked="${!key_name:0:8}..."
            print_success "  $key_name = $masked"
            ((key_count++)) || true
        else
            print_warning "  $key_name not set"
        fi
    done

    if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
        print_success "  AWS credentials set (Bedrock available)"
        ((key_count++)) || true
    else
        print_warning "  AWS credentials not set (Bedrock unavailable)"
    fi

    if [ -n "$SERPER_API_KEY" ]; then
        print_success "  SERPER_API_KEY set (web search available)"
    else
        print_warning "  SERPER_API_KEY not set (agentic search unavailable)"
    fi

    echo ""
    if [ $key_count -eq 0 ]; then
        print_error "No API keys found. Set at least one provider key."
        all_good=false
    fi

    if [ "$all_good" = true ]; then
        print_success "Environment is ready."
    else
        print_error "Fix the errors above before running."
        exit 1
    fi
}

# ── Parse Arguments ──────────────────────────────────────────────────────────

PIPELINE=""
INPUT_PATH=""
OUTPUT_PATH=""
MODELS=""
MODE=""
INDICATORS=""
TEST_SIZE=""
DELAY=""
TEMPERATURE=""
MAX_TOKENS=""
MAX_RETRIES=""
SEARCH_MODE=""
USE_BATCH=""
VERIFY=""
VERIFY_INDICATORS=""
VERIFIER_MODEL=""
N_SAMPLES=""
SEQUENCE=""
RANDOM_SEQUENCE=""
REASONING=""
PARALLEL_ROWS=""
CHECKPOINT_INTERVAL=""
BACKGROUND=false
NOTIFY=false
DRY_RUN=false
QUICK_TEST=false
SC_TEMPERATURES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --pipeline)
            PIPELINE="$2"; shift 2 ;;
        -i|--input)
            INPUT_PATH="$2"; shift 2 ;;
        -o|--output)
            OUTPUT_PATH="$2"; shift 2 ;;
        -m|--models)
            shift
            MODELS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]] && [[ ! "$1" =~ ^-[a-z]$ ]]; do
                MODELS="$MODELS $1"; shift
            done
            ;;
        --mode)
            MODE="$2"; shift 2 ;;
        --indicators)
            shift
            INDICATORS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]] && [[ ! "$1" =~ ^-[a-z]$ ]]; do
                INDICATORS="$INDICATORS $1"; shift
            done
            ;;
        -t|--test)
            TEST_SIZE="$2"; shift 2 ;;
        -d|--delay)
            DELAY="$2"; shift 2 ;;
        --temperature)
            TEMPERATURE="$2"; shift 2 ;;
        --max-tokens)
            MAX_TOKENS="$2"; shift 2 ;;
        --max-retries)
            MAX_RETRIES="$2"; shift 2 ;;
        --search-mode)
            SEARCH_MODE="$2"; shift 2 ;;
        --use-batch)
            USE_BATCH="true"; shift ;;
        --verify)
            VERIFY="$2"; shift 2 ;;
        --verify-indicators)
            shift
            VERIFY_INDICATORS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]] && [[ ! "$1" =~ ^-[a-z]$ ]]; do
                VERIFY_INDICATORS="$VERIFY_INDICATORS $1"; shift
            done
            ;;
        --verifier-model)
            VERIFIER_MODEL="$2"; shift 2 ;;
        --n-samples)
            N_SAMPLES="$2"; shift 2 ;;
        --sc-temperatures)
            shift
            SC_TEMPERATURES=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]] && [[ ! "$1" =~ ^-[a-z]$ ]]; do
                SC_TEMPERATURES="$SC_TEMPERATURES $1"; shift
            done
            ;;
        --sequence)
            shift
            SEQUENCE=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]] && [[ ! "$1" =~ ^-[a-z]$ ]]; do
                SEQUENCE="$SEQUENCE $1"; shift
            done
            ;;
        --random-sequence)
            RANDOM_SEQUENCE="true"; shift ;;
        --reasoning)
            REASONING="$2"; shift 2 ;;
        --parallel-rows)
            PARALLEL_ROWS="$2"; shift 2 ;;
        --checkpoint-interval)
            CHECKPOINT_INTERVAL="$2"; shift 2 ;;
        --background)
            BACKGROUND=true; shift ;;
        --notify)
            NOTIFY=true; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --quick-test)
            QUICK_TEST=true; shift ;;
        --check-env)
            check_environment; exit 0 ;;
        -h|--help)
            show_usage; exit 0 ;;
        *)
            print_error "Unknown option: $1"
            echo "Run ./run.sh --help for usage."
            exit 1
            ;;
    esac
done

# ── Quick Test shortcut ──────────────────────────────────────────────────────

if [ "$QUICK_TEST" = true ]; then
    PIPELINE="${PIPELINE:-leader}"
    TEST_SIZE="${TEST_SIZE:-5}"
    INDICATORS="${INDICATORS:- sovereign assembly}"
    print_info "Quick test mode: $TEST_SIZE rows, pipeline=$PIPELINE"
fi

# ── Defaults ─────────────────────────────────────────────────────────────────

PIPELINE="${PIPELINE:-leader}"

# ── Validation ───────────────────────────────────────────────────────────────

if [ ! -f "main.py" ]; then
    print_error "main.py not found. Run this script from the project root directory."
    exit 1
fi

# Source .env if present
if [ -f ".env" ]; then
    set -a; source .env 2>/dev/null; set +a
fi

# Validate input file exists
if [ -n "$INPUT_PATH" ] && [ ! -f "$INPUT_PATH" ]; then
    print_error "Input file not found: $INPUT_PATH"
    exit 1
fi

# Create output directory if needed
if [ -n "$OUTPUT_PATH" ]; then
    OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
    if [ "$OUTPUT_DIR" != "." ] && [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
        print_info "Created output directory: $OUTPUT_DIR"
    fi
fi

# ── Build Command ────────────────────────────────────────────────────────────

CMD=(python3 main.py --pipeline "$PIPELINE")

[ -n "$INPUT_PATH" ]           && CMD+=(--input "$INPUT_PATH")
[ -n "$OUTPUT_PATH" ]          && CMD+=(--output "$OUTPUT_PATH")
[ -n "$MODELS" ]               && CMD+=(--models $MODELS)
[ -n "$MODE" ]                 && CMD+=(--mode "$MODE")
[ -n "$INDICATORS" ]           && CMD+=(--indicators $INDICATORS)
[ -n "$TEST_SIZE" ]            && CMD+=(--test "$TEST_SIZE")
[ -n "$DELAY" ]                && CMD+=(--delay "$DELAY")
[ -n "$TEMPERATURE" ]          && CMD+=(--temperature "$TEMPERATURE")
[ -n "$MAX_TOKENS" ]           && CMD+=(--max-tokens "$MAX_TOKENS")
[ -n "$MAX_RETRIES" ]          && CMD+=(--max-retries "$MAX_RETRIES")
[ -n "$SEARCH_MODE" ]          && CMD+=(--search-mode "$SEARCH_MODE")
[ "$USE_BATCH" = "true" ]      && CMD+=(--use-batch)
[ -n "$VERIFY" ]               && CMD+=(--verify "$VERIFY")
[ -n "$VERIFY_INDICATORS" ]    && CMD+=(--verify-indicators $VERIFY_INDICATORS)
[ -n "$VERIFIER_MODEL" ]       && CMD+=(--verifier-model "$VERIFIER_MODEL")
[ -n "$N_SAMPLES" ]            && CMD+=(--n-samples "$N_SAMPLES")
[ -n "$SC_TEMPERATURES" ]      && CMD+=(--sc-temperatures $SC_TEMPERATURES)
[ -n "$SEQUENCE" ]             && CMD+=(--sequence $SEQUENCE)
[ "$RANDOM_SEQUENCE" = "true" ] && CMD+=(--random-sequence)
[ -n "$REASONING" ]            && CMD+=(--reasoning "$REASONING")
[ -n "$PARALLEL_ROWS" ]        && CMD+=(--parallel-rows "$PARALLEL_ROWS")
[ -n "$CHECKPOINT_INTERVAL" ]  && CMD+=(--checkpoint-interval "$CHECKPOINT_INTERVAL")

# ── Display Configuration ───────────────────────────────────────────────────

print_header "Constitution Analysis Pipeline"
echo ""
echo -e "  ${BOLD}Pipeline:${NC}     $PIPELINE"
echo -e "  ${BOLD}Input:${NC}        ${INPUT_PATH:-<default>}"
echo -e "  ${BOLD}Output:${NC}       ${OUTPUT_PATH:-<default>}"
echo -e "  ${BOLD}Model(s):${NC}     ${MODELS:- <default: gemini-2.5-pro>}"

if [ "$PIPELINE" = "leader" ]; then
    echo -e "  ${BOLD}Mode:${NC}         ${MODE:-multiple}"
    echo -e "  ${BOLD}Indicators:${NC}   ${INDICATORS:- <default: constitution>}"
    [ -n "$PARALLEL_ROWS" ] && \
    echo -e "  ${BOLD}Parallel:${NC}     $PARALLEL_ROWS rows"
fi

echo -e "  ${BOLD}Test:${NC}         ${TEST_SIZE:-<full dataset>}"
echo -e "  ${BOLD}Search:${NC}       ${SEARCH_MODE:-none}"
[ "$USE_BATCH" = "true" ] && \
echo -e "  ${BOLD}Batch API:${NC}    enabled (50% cost savings)"
[ -n "$VERIFY" ] && [ "$VERIFY" != "none" ] && \
echo -e "  ${BOLD}Verify:${NC}       $VERIFY → ${VERIFY_INDICATORS:- <all>}"
[ "$BACKGROUND" = true ] && \
echo -e "  ${BOLD}Background:${NC}   yes (survives sleep & terminal close)"
echo ""

# Show the command
print_info "Command:"
echo -e "  ${DIM}${CMD[*]}${NC}"
echo ""

# ── Dry Run ──────────────────────────────────────────────────────────────────

if [ "$DRY_RUN" = true ]; then
    print_info "Dry run — command not executed."
    exit 0
fi

# ── Run ──────────────────────────────────────────────────────────────────────

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [ "$BACKGROUND" = true ]; then
    # ── Background execution ─────────────────────────────────────────────
    # Uses caffeinate to prevent sleep and nohup to survive terminal close.
    # Output goes to a timestamped log file.

    LOG_DIR="data/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

    print_info "Starting in background..."
    print_info "Log file: $LOG_FILE"
    print_info "Monitor with: tail -f $LOG_FILE"
    echo ""

    # Build the full command string for nohup
    CMD_STR="${CMD[*]}"

    # Notification command (macOS only)
    if [ "$NOTIFY" = true ] && command -v osascript &> /dev/null; then
        NOTIFY_CMD='osascript -e "display notification \"Pipeline finished\" with title \"Constitution LLM\" sound name \"Glass\""'
    else
        NOTIFY_CMD=""
    fi

    # Write a wrapper script that caffeinate will keep alive
    WRAPPER="/tmp/constitution_llm_run_${TIMESTAMP}.sh"
    cat > "$WRAPPER" <<WRAPPER_EOF
#!/bin/bash
cd "$(pwd)"
# Source .env for API keys
[ -f .env ] && set -a && source .env 2>/dev/null && set +a

echo "========================================"
echo "Constitution LLM Pipeline"
echo "Started: \$(date)"
echo "Command: $CMD_STR"
echo "========================================"
echo ""

START_TIME=\$(date +%s)

if $CMD_STR; then
    END_TIME=\$(date +%s)
    DURATION=\$(( END_TIME - START_TIME ))
    MINS=\$(( DURATION / 60 ))
    SECS=\$(( DURATION % 60 ))
    echo ""
    echo "========================================"
    echo "SUCCESS - Completed in \${MINS}m \${SECS}s"
    echo "Finished: \$(date)"
    echo "========================================"
    ${NOTIFY_CMD}
else
    echo ""
    echo "========================================"
    echo "FAILED - Check log for errors"
    echo "Finished: \$(date)"
    echo "========================================"
    ${NOTIFY_CMD:+osascript -e 'display notification "Pipeline FAILED" with title "Constitution LLM" sound name "Basso"'}
fi
WRAPPER_EOF
    chmod +x "$WRAPPER"

    # caffeinate -i  → prevent idle sleep (system stays awake)
    # nohup          → survives terminal close
    # &              → run in background
    nohup caffeinate -i bash "$WRAPPER" > "$LOG_FILE" 2>&1 &
    BG_PID=$!

    print_success "Background process started (PID: $BG_PID)"
    echo ""
    echo -e "  ${BOLD}Check progress:${NC}  tail -f $LOG_FILE"
    echo -e "  ${BOLD}Stop the job:${NC}    kill $BG_PID"
    echo -e "  ${BOLD}Check if alive:${NC}  ps -p $BG_PID"
    echo ""
    print_info "You can close this terminal. The job will keep running."

else
    # ── Foreground execution ─────────────────────────────────────────────
    print_info "Starting pipeline..."
    START_TIME=$(date +%s)

    if "${CMD[@]}"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINS=$((DURATION / 60))
        SECS=$((DURATION % 60))

        echo ""
        print_header "Pipeline Complete"
        print_success "Execution time: ${MINS}m ${SECS}s"

        if [ -n "$OUTPUT_PATH" ] && [ -f "$OUTPUT_PATH" ]; then
            LINE_COUNT=$(wc -l < "$OUTPUT_PATH" | tr -d ' ')
            ROWS=$((LINE_COUNT - 1))
            print_info "Output: $OUTPUT_PATH ($ROWS rows)"
        fi

        # Check for cost report
        if [ -n "$OUTPUT_PATH" ]; then
            COST_FILE="data/logs/$(basename "${OUTPUT_PATH%.csv}")_costs.json"
            if [ -f "$COST_FILE" ]; then
                print_info "Cost report: $COST_FILE"
            fi
        fi

        echo ""
        print_success "Done!"

        # macOS notification
        if [ "$NOTIFY" = true ] && command -v osascript &> /dev/null; then
            osascript -e 'display notification "Pipeline finished" with title "Constitution LLM" sound name "Glass"'
        fi
    else
        echo ""
        print_error "Pipeline failed. Check the errors above."

        if [ "$NOTIFY" = true ] && command -v osascript &> /dev/null; then
            osascript -e 'display notification "Pipeline FAILED" with title "Constitution LLM" sound name "Basso"'
        fi
        exit 1
    fi
fi
