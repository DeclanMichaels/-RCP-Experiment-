#!/bin/bash
#
# RCP Experiment -- Run with virtual environment
#
# Activates the venv and passes all arguments to run_experiment.py.
# Run getstarted.sh first if you haven't.
#
# Usage:
#   ./run.sh --pilot
#   ./run.sh --analysis-only
#   ./run.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "No virtual environment found. Run ./getstarted.sh first."
    exit 1
fi

source .venv/bin/activate
python run_experiment.py "$@"
