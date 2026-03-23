#!/bin/bash
# split_multi_code.sh — Split Multi-Code run into per-model directories
# Run from the -RCP-Experiment- directory

SRC="20260322-1-Multi-Code-Data"
SONNET_DIR="20260322-1-Sonnet-Code-Data"
GPT4O_DIR="20260322-2-GPT4o-Code-Data"

# Create directories if needed
mkdir -p "$SONNET_DIR" "$GPT4O_DIR"

# Copy per-model main data files
cp "$SRC/main_claude-sonnet.jsonl" "$SONNET_DIR/main_claude-sonnet.jsonl"
cp "$SRC/main_gpt-4o.jsonl" "$GPT4O_DIR/main_gpt-4o.jsonl"

# Filter manipulation check by model
grep '"model_name": "claude-sonnet"' "$SRC/manipulation_check.jsonl" > "$SONNET_DIR/manipulation_check.jsonl"
grep '"model_name": "gpt-4o"' "$SRC/manipulation_check.jsonl" > "$GPT4O_DIR/manipulation_check.jsonl"

# Filter explanations by model
grep '"model_name": "claude-sonnet"' "$SRC/explanations.jsonl" > "$SONNET_DIR/explanations.jsonl"
grep '"model_name": "gpt-4o"' "$SRC/explanations.jsonl" > "$GPT4O_DIR/explanations.jsonl"

# Create per-model metadata
cat > "$SONNET_DIR/run_metadata.json" << 'EOF'
{
  "pair_direction_seed": 2038780684,
  "models": ["claude-sonnet"],
  "framings": ["neutral", "functional", "enterprise", "systems", "startup", "irrelevant", "nonsense"],
  "temperatures": [0.0],
  "timestamp": "2026-03-22T18:19:35.969064+00:00",
  "note": "Split from 20260322-1-Multi-Code-Data"
}
EOF

cat > "$GPT4O_DIR/run_metadata.json" << 'EOF'
{
  "pair_direction_seed": 2038780684,
  "models": ["gpt-4o"],
  "framings": ["neutral", "functional", "enterprise", "systems", "startup", "irrelevant", "nonsense"],
  "temperatures": [0.0],
  "timestamp": "2026-03-22T18:19:35.969064+00:00",
  "note": "Split from 20260322-1-Multi-Code-Data"
}
EOF

echo "Data split complete."
echo ""
echo "Sonnet: $(wc -l < "$SONNET_DIR/main_claude-sonnet.jsonl") main records"
echo "GPT-4o: $(wc -l < "$GPT4O_DIR/main_gpt-4o.jsonl") main records"
echo ""
echo "Now run analysis on each:"
echo "  python3 analyze.py --data-dir $SONNET_DIR --output-dir 20260322-1-Sonnet-Code-Results --config config-code.json"
echo "  python3 analyze.py --data-dir $GPT4O_DIR --output-dir 20260322-2-GPT4o-Code-Results --config config-code.json"
