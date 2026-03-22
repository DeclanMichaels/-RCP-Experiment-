#!/bin/bash
#
# RCP Experiment -- Environment Setup
#
# Run once. Creates a virtual environment, installs dependencies,
# runs unit tests to confirm everything works. After this, use:
#
#   source .venv/bin/activate
#   python run_experiment.py --pilot
#
# Or just use ./run.sh which activates and runs for you.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  RCP Experiment -- Environment Setup"
echo "============================================================"
echo

# Find Python 3
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" --version 2>&1)
        if echo "$version" | grep -q "Python 3"; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3 not found. Install it from https://python.org"
    exit 1
fi

echo "  Python: $PYTHON ($($PYTHON --version 2>&1))"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment..."
    $PYTHON -m venv .venv
else
    echo "  Virtual environment exists."
fi

# Activate
source .venv/bin/activate
echo "  Activated: $(which python)"

# Install dependencies
echo "  Installing dependencies..."
pip install -q -r requirements.txt
echo "  Dependencies installed."

# Run unit tests
echo
echo "  Running unit tests..."
python run_tests.py

echo
echo "============================================================"
echo "  Setup complete."
echo "============================================================"
echo
echo "  To run the experiment:"
echo "    source .venv/bin/activate"
echo "    export ANTHROPIC_API_KEY=sk-ant-..."
echo "    export ANTHROPIC_RPM=5              # optional: your rate limit"
echo "    python run_experiment.py --pilot"
echo
echo "  Or use the shortcut:"
echo "    export ANTHROPIC_API_KEY=sk-ant-..."
echo "    ./run.sh --pilot"
echo
