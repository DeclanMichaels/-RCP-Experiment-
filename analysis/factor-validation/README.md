# RCP Factor Validation Tool

Validates that concept inventory domain assignments (physical, institutional, moral) correspond to factor structure in the actual LLM similarity rating data.

This complements the embedding-based cluster validation (`../cluster-validation/`) by validating against response structure rather than a proxy embedding model.

## Quick start

```bash
cd analysis/factor-validation
chmod +x run.sh
./run.sh --data ../../runs/20260324-1-Sonnet-Moral-Data/
```

The first run creates a Python venv and installs dependencies. Subsequent runs start immediately.

## Commands

```bash
# Print report to terminal (one model)
./run.sh --data ../../runs/20260324-1-Sonnet-Moral-Data/

# Multiple models at once
./run.sh --data ../../runs/20260324-1-Sonnet-Moral-Data/ ../../runs/20260324-1-GPT4o-Moral-Data/

# Save reports + plots to a folder
./run.sh --data ../../runs/*-Moral-Data/ --output results/

# Use config.json for canonical concept ordering
./run.sh --data ../../runs/20260324-1-Sonnet-Moral-Data/ --config ../../config.json

# Extract 2 factors instead of 3
./run.sh --data ../../runs/20260324-1-Sonnet-Moral-Data/ --n-factors 2

# Run unit tests
./run.sh --test

# Remove the virtual environment
./run.sh --clean
```

## What it measures

- **Factor loadings**: Do concepts load on the factor corresponding to their assigned domain? Principal axis factoring with varimax rotation.
- **Domain recovery rate**: What fraction of concepts have their highest loading on the "correct" domain factor?
- **KMO score**: Is the correlation matrix suitable for factor analysis? (>0.6 adequate, >0.8 good)
- **Bartlett's test**: Is the correlation matrix significantly different from identity?
- **Parallel analysis**: How many factors does the data actually support? Compares eigenvalues to random matrices.
- **Tucker's congruence coefficient**: How well do the extracted factors match the target domain structure? (>0.95 excellent, >0.85 good)

## Output files (when using --output)

Per model directory:
- `factor_validation_report.txt` -- Human-readable report
- `factor_validation.json` -- Machine-readable results
- `loading_heatmap.png` -- Factor loadings colored by domain
- `scree_plot.png` -- Scree plot with parallel analysis overlay

## Data input

Reads `main_*.jsonl` files from RCP Data directories. Filters to unframed/neutral condition only. Handles both `framing: "neutral"` (v1 data) and `framing: "unframed"` (v2 data).

If multiple ratings exist for the same pair (stochastic runs with multiple reps), they are averaged.

## Method

1. Extract unframed pairwise similarity ratings from JSONL
2. Build 18x18 similarity matrix (concepts sorted alphabetically within domain)
3. Normalize to correlation scale: (rating - 1) / 6, diagonal = 1.0
4. Ensure positive semi-definiteness (Higham projection if needed)
5. Extract factors via principal axis factoring (eigendecomposition)
6. Apply varimax rotation for simple structure
7. Map factors to domains by highest average absolute loading
8. Compute recovery rate, Tucker's congruence, KMO, Bartlett, parallel analysis

## Why this matters

The embedding-based cluster validation (`../cluster-validation/`) checks whether the words themselves cluster by domain in a general-purpose embedding model. That's a necessary sanity check, but it validates against a proxy, not the actual experimental data.

Factor analysis validates against the LLM similarity ratings themselves. If concepts load on the expected domain factors, it means the models are constructing domain-differentiated relational structure from the probes. This is construct validity from the response data, which is the standard a psychometrics reviewer would expect.

## Requirements

- macOS, Linux, or WSL
- Python 3.9+

No internet connection required (no model downloads). Everything is handled by `run.sh`.
