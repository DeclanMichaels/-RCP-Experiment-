# RCP Cluster Validation Tool

Validates that concept inventory domain assignments (physical, institutional, moral) correspond to semantic clustering in embedding space.

Produces: cosine similarity analysis, silhouette scores, hierarchical cluster recovery (ARI/NMI), PCA scatter plot, and dendrogram.

## Quick start

```bash
cd analysis/cluster-validation
chmod +x run.sh
./run.sh --config ../../config.json --output results/
```

The first run takes a few minutes (creates a Python venv, installs dependencies, downloads the embedding model). Subsequent runs start in seconds.

## Commands

```bash
# Print report to terminal
./run.sh --config ../../config.json

# Save report + plots to a folder
./run.sh --config ../../config.json --output results/

# Use a different embedding model
./run.sh --config ../../config.json --model all-mpnet-base-v2

# Run unit tests
./run.sh --test

# Remove the virtual environment
./run.sh --clean
```

## What it measures

- **Within vs between domain similarity**: Are concepts in the same domain closer in embedding space than concepts in different domains?
- **Silhouette score**: Per-concept and per-domain clustering quality (-1 to 1, higher is better).
- **Cluster recovery**: If we hierarchically cluster the embeddings and cut at 3 clusters, how well do they match the assigned domains? (Adjusted Rand Index, Normalized Mutual Information)
- **PCA projection**: 2D visualization of the 384-dimensional embedding space.

## Output files (when using --output)

- `cluster_validation_report.txt` — Human-readable report
- `cluster_validation.json` — Machine-readable results
- `pca_scatter.png` — 2D PCA scatter plot colored by domain
- `dendrogram.png` — Hierarchical clustering dendrogram

## Requirements

- macOS, Linux, or WSL
- Python 3.9+
- Internet connection (first run only, for model download)

Everything else is handled automatically by `run.sh`.
