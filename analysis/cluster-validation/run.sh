#!/usr/bin/env bash
#
# run.sh — Bootstrap and run the RCP cluster validation tool.
#
# Handles: Python version check, virtual environment creation,
# dependency installation, model pre-download, and execution.
#
# Usage:
#   ./run.sh --config ../../config.json
#   ./run.sh --config ../../config.json --output results/
#   ./run.sh --config ../../config.json --model all-mpnet-base-v2
#   ./run.sh --test                # run unit tests instead
#   ./run.sh --clean               # remove venv and cached model
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=9
DEFAULT_MODEL="all-MiniLM-L6-v2"

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' NC=''
fi

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
info()  { echo -e "${BLUE}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
fail()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

check_python() {
    # Try python3 first, then python
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            local version
            version="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            local major="${version%%.*}"
            local minor="${version#*.}"
            if [ "$major" -ge "$MIN_PYTHON_MAJOR" ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; then
                PYTHON_CMD="$cmd"
                ok "Found $cmd $version"
                return 0
            else
                warn "$cmd $version found, but need >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}"
            fi
        fi
    done
    fail "Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} not found.\n\n  macOS:   brew install python@3.11\n  Ubuntu:  sudo apt install python3.11 python3.11-venv\n  Windows: https://www.python.org/downloads/\n"
}

setup_venv() {
    if [ -d "$VENV_DIR" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
        ok "Virtual environment exists at .venv/"
    else
        info "Creating virtual environment..."
        "$PYTHON_CMD" -m venv "$VENV_DIR" || fail "Could not create venv. On Ubuntu/Debian you may need:\n  sudo apt install python3-venv"
        ok "Virtual environment created"
    fi

    # Activate
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
}

install_deps() {
    # Check if deps are already installed by trying to import the heaviest one
    if python -c "import sentence_transformers" &>/dev/null; then
        ok "Dependencies already installed"
        return 0
    fi

    info "Installing dependencies (this may take a few minutes the first time)..."
    pip install --upgrade pip --quiet 2>/dev/null
    pip install -r "$REQUIREMENTS" --quiet || fail "Dependency installation failed"
    ok "Dependencies installed"
}

preload_model() {
    local model="${1:-$DEFAULT_MODEL}"
    # sentence-transformers caches models in ~/.cache/torch/sentence_transformers/
    # Check if already downloaded
    if python -c "
from sentence_transformers import SentenceTransformer
import os
cache_dir = os.path.expanduser('~/.cache/torch/sentence_transformers/')
model_dir = os.path.join(cache_dir, 'sentence-transformers_${model}')
if not os.path.isdir(model_dir):
    # Also check the newer cache location
    from huggingface_hub import try_to_load_from_cache
    result = try_to_load_from_cache('sentence-transformers/${model}', 'config.json')
    if result is None:
        exit(1)
" &>/dev/null 2>&1; then
        ok "Model '${model}' already cached"
    else
        info "Downloading model '${model}' (~90MB, one-time)..."
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${model}')" \
            || fail "Model download failed. Check your internet connection."
        ok "Model downloaded and cached"
    fi
}

# ---------------------------------------------------------------------------
# Special modes
# ---------------------------------------------------------------------------
handle_clean() {
    info "Cleaning up..."
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        ok "Removed .venv/"
    fi
    echo ""
    info "Note: cached models live in ~/.cache/huggingface/ and ~/.cache/torch/"
    info "Remove those manually if you want to reclaim disk space."
    exit 0
}

handle_test() {
    check_python
    setup_venv
    install_deps
    echo ""
    info "Running unit tests..."
    echo ""
    cd "$SCRIPT_DIR"
    python -m pytest test_cluster_validate.py -v
    exit $?
}

handle_help() {
    cat <<'EOF'
RCP Cluster Validation Tool
============================

Validates that concept inventory domain assignments correspond to
semantic clustering in embedding space.

USAGE:
  ./run.sh --config <path>  [--output <dir>] [--model <name>] [--no-plots]
  ./run.sh --test
  ./run.sh --clean
  ./run.sh --help

ARGUMENTS:
  --config <path>    Path to RCP config.json (required for analysis)
  --output <dir>     Save report, JSON, and plots to this directory
  --model <name>     sentence-transformers model (default: all-MiniLM-L6-v2)
  --no-plots         Skip generating PNG scatter plot and dendrogram

SPECIAL MODES:
  --test             Run the unit test suite
  --clean            Remove the virtual environment
  --help             Show this message

EXAMPLES:
  # Basic run, prints report to terminal
  ./run.sh --config ../../config.json

  # Save everything to a results folder
  ./run.sh --config ../../config.json --output results/

  # Use a different embedding model
  ./run.sh --config ../../config.json --model all-mpnet-base-v2

  # Run unit tests
  ./run.sh --test

FIRST RUN:
  The first run creates a Python virtual environment, installs
  dependencies (~500MB), and downloads the embedding model (~90MB).
  Subsequent runs start in seconds.
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    echo "  RCP Cluster Validation Tool"
    echo "  ==========================="
    echo ""

    # Parse for special modes first
    for arg in "$@"; do
        case "$arg" in
            --clean) handle_clean ;;
            --test)  handle_test ;;
            --help|-h) handle_help ;;
        esac
    done

    # Need at least --config for a real run
    if [ $# -eq 0 ]; then
        handle_help
    fi

    # Extract model name for preload (default if not specified)
    local model="$DEFAULT_MODEL"
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [ "${args[$i]}" = "--model" ] && [ $((i+1)) -lt ${#args[@]} ]; then
            model="${args[$((i+1))]}"
        fi
    done

    check_python
    setup_venv
    install_deps
    preload_model "$model"

    echo ""
    info "Running analysis..."
    echo ""

    cd "$SCRIPT_DIR"
    python cluster_validate.py "$@"
}

main "$@"
