#!/usr/bin/env bash
#
# Factor validation runner for RCP concept inventories.
# Creates a virtual environment, installs dependencies, and runs the tool.
#
# Usage:
#   ./run.sh --data ../../runs/20260324-1-Sonnet-Moral-Data/
#   ./run.sh --data ../../runs/*-Moral-Data/ --output results/
#   ./run.sh --data ../../runs/20260324-1-Sonnet-Moral-Data/ --config ../../config.json
#   ./run.sh --test
#   ./run.sh --clean

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# --- Handle --clean ---
if [[ "${1:-}" == "--clean" ]]; then
    echo "Removing virtual environment..."
    rm -rf "$VENV_DIR"
    echo "Done."
    exit 0
fi

# --- Ensure venv exists ---
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Installing dependencies..."
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
    echo "Environment ready."
fi

PYTHON="$VENV_DIR/bin/python"

# --- Handle --test ---
if [[ "${1:-}" == "--test" ]]; then
    echo "Running unit tests..."
    "$PYTHON" -m pytest "$SCRIPT_DIR/test_factor_validate.py" -v
    exit $?
fi

# --- Run the tool ---
exec "$PYTHON" "$SCRIPT_DIR/factor_validate.py" "$@"
