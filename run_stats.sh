#!/bin/bash
#
# Run pre-registered statistical tests (H1, H2, H3, Cohen's d)
# against all moral-domain data directories found under runs/.
#
# Usage: bash run_stats.sh           (exact permutation tests)
#        bash run_stats.sh --quick   (Monte Carlo, 100 permutations, fast sanity check)
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "No virtual environment found. Run ./getstarted.sh first."
    exit 1
fi

source .venv/bin/activate

# Quick mode: Monte Carlo with 100 permutations
# Full mode: exact enumeration (H1: 17M partitions, H2: 18K groups)
PERM_ARGS=""
MODE="EXACT"
if [ "$1" = "--quick" ]; then
    PERM_ARGS="--n-perm-h1 100 --n-perm-h2 100"
    MODE="QUICK (100 MC permutations)"
fi

# Auto-discover all moral-domain data directories under runs/
DATA_DIRS=$(find runs -maxdepth 1 -type d -name "*-Moral-Data" | sort)

if [ -z "$DATA_DIRS" ]; then
    echo "No moral-domain data directories found under runs/"
    echo "Expected pattern: runs/*-Moral-Data"
    exit 1
fi

echo "============================================================"
echo "  RCP STATISTICAL TESTS (Pre-Registered)"
echo "  Mode: $MODE"
echo "  H1: Domain ordering permutation test"
echo "  H2: Framing sensitivity permutation test"
echo "  H3: Control discrimination ratio"
echo "  Effect sizes: Cohen's d"
echo "============================================================"
echo ""
echo "  Data directories found:"
for d in $DATA_DIRS; do
    echo "    $d"
done

FAILURES=0
COUNT=0

for DATA_DIR in $DATA_DIRS; do
    # Skip aborted or archived directories
    if echo "$DATA_DIR" | grep -qiE "ABORTED|ARCHIVED|STALE"; then
        echo ""
        echo "  SKIP: $DATA_DIR (aborted/archived)"
        continue
    fi

    COUNT=$((COUNT + 1))
    echo ""
    python analysis/permutation_tests.py \
        --data-dir "$DATA_DIR" \
        $PERM_ARGS \
        --seed 42

    if [ $? -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
    fi
done

echo ""
echo "============================================================"
echo "  $COUNT data directories processed"
if [ $FAILURES -eq 0 ]; then
    echo "  ALL COMPLETE"
else
    echo "  $FAILURES HAD ERRORS"
fi
echo "============================================================"
