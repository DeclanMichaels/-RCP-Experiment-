"""
Embedding-based cluster validation for RCP concept inventories.

Validates that domain assignments (physical, institutional, moral) correspond
to semantic clustering in embedding space. Uses sentence-transformers for
embeddings, hierarchical clustering, and silhouette analysis.

Requires: sentence-transformers, scikit-learn, scipy, numpy, matplotlib

Usage:
    python cluster_validate.py --config ../../config.json
    python cluster_validate.py --config ../../config.json --output results/
    python cluster_validate.py --config ../../config.json --model all-mpnet-base-v2
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Concept:
    """A single concept with its domain label."""
    word: str
    domain: str


@dataclass(frozen=True)
class EmbeddingResult:
    """Embeddings for a set of concepts."""
    concepts: tuple  # tuple of Concept
    vectors: np.ndarray  # shape (n_concepts, embedding_dim)
    model_name: str

    class Config:
        arbitrary_types_allowed = True


@dataclass
class ClusterReport:
    """Full cluster validation report for one concept inventory."""
    model_name: str
    concepts: list
    domains: list
    n_concepts: int
    n_domains: int
    embedding_dim: int

    # Cosine similarity matrix (n x n)
    cosine_sim_matrix: np.ndarray = field(repr=False)

    # Within-domain vs between-domain similarity
    mean_within_domain_similarity: float = 0.0
    mean_between_domain_similarity: float = 0.0
    separation_ratio: float = 0.0

    # Per-domain within similarity
    per_domain_within: dict = field(default_factory=dict)

    # Silhouette analysis
    silhouette_avg: float = 0.0
    per_domain_silhouette: dict = field(default_factory=dict)
    per_concept_silhouette: dict = field(default_factory=dict)

    # Cluster recovery (hierarchical clustering cut at n_domains)
    adjusted_rand_index: float = 0.0
    normalized_mutual_info: float = 0.0
    cluster_assignments: list = field(default_factory=list)
    misassigned_concepts: list = field(default_factory=list)

    # PCA projection (2D)
    pca_coords: np.ndarray = field(default=None, repr=False)
    pca_variance_explained: tuple = field(default_factory=tuple)

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_concepts_from_config(config_path: str) -> list:
    """Load concept inventory from an RCP config.json file.

    Args:
        config_path: Path to config.json with a "concepts" key mapping
                     domain names to lists of concept words.

    Returns:
        List of Concept objects, sorted by (domain, word) for determinism.
    """
    with open(config_path) as f:
        config = json.load(f)

    if "concepts" not in config:
        raise ValueError(f"Config file {config_path} has no 'concepts' key")

    concepts = []
    for domain, words in config["concepts"].items():
        for word in words:
            concepts.append(Concept(word=word, domain=domain))

    # Sort for deterministic ordering
    concepts.sort(key=lambda c: (c.domain, c.word))
    return concepts


def get_embeddings(concepts: list, model_name: str = "all-MiniLM-L6-v2") -> EmbeddingResult:
    """Compute embeddings for a list of concepts.

    Args:
        concepts: List of Concept objects.
        model_name: sentence-transformers model name.

    Returns:
        EmbeddingResult with vectors shape (n_concepts, embedding_dim).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    words = [c.word for c in concepts]
    vectors = model.encode(words, show_progress_bar=False)

    return EmbeddingResult(
        concepts=tuple(concepts),
        vectors=np.array(vectors),
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Analysis functions (all operate on numpy arrays, no model dependency)
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        vectors: shape (n, d) array of embeddings.

    Returns:
        shape (n, n) symmetric matrix with 1.0 on diagonal.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Guard against zero vectors
    norms = np.where(norms == 0, 1e-10, norms)
    normalized = vectors / norms
    sim = normalized @ normalized.T
    # Clip to [-1, 1] for numerical stability
    return np.clip(sim, -1.0, 1.0)


def within_between_similarity(sim_matrix: np.ndarray, labels: list) -> dict:
    """Compute mean within-domain and between-domain cosine similarity.

    Args:
        sim_matrix: (n, n) cosine similarity matrix.
        labels: list of domain labels, length n.

    Returns:
        Dict with keys: mean_within, mean_between, separation_ratio,
        per_domain_within (dict of domain -> mean within similarity).
    """
    n = len(labels)
    unique_domains = sorted(set(labels))

    within_vals = []
    between_vals = []
    per_domain = {}

    for domain in unique_domains:
        indices = [i for i, l in enumerate(labels) if l == domain]
        domain_within = []
        for i_idx in range(len(indices)):
            for j_idx in range(i_idx + 1, len(indices)):
                val = sim_matrix[indices[i_idx], indices[j_idx]]
                within_vals.append(val)
                domain_within.append(val)
        if domain_within:
            per_domain[domain] = float(np.mean(domain_within))
        else:
            per_domain[domain] = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                between_vals.append(sim_matrix[i, j])

    mean_within = float(np.mean(within_vals)) if within_vals else 0.0
    mean_between = float(np.mean(between_vals)) if between_vals else 0.0

    # Separation ratio: how much higher within is than between.
    # >1 means within-domain concepts are more similar than between-domain.
    if mean_between != 0:
        separation_ratio = mean_within / mean_between
    else:
        separation_ratio = float("inf") if mean_within > 0 else 0.0

    return {
        "mean_within": mean_within,
        "mean_between": mean_between,
        "separation_ratio": separation_ratio,
        "per_domain_within": per_domain,
    }


def compute_silhouette(vectors: np.ndarray, labels: list) -> dict:
    """Compute silhouette scores using cosine distance.

    Args:
        vectors: (n, d) embedding array.
        labels: list of domain labels, length n.

    Returns:
        Dict with silhouette_avg, per_domain (dict), per_concept (dict).
    """
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 domains for silhouette analysis")

    # Encode labels as integers for sklearn
    label_map = {l: i for i, l in enumerate(unique_labels)}
    int_labels = [label_map[l] for l in labels]

    avg = silhouette_score(vectors, int_labels, metric="cosine")
    samples = silhouette_samples(vectors, int_labels, metric="cosine")

    per_domain = {}
    for domain in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == domain]
        per_domain[domain] = float(np.mean(samples[mask]))

    per_concept = {}
    for i, (label, score) in enumerate(zip(labels, samples)):
        # Return by index; caller maps to concept names.
        per_concept[i] = float(score)

    return {
        "silhouette_avg": float(avg),
        "per_domain": per_domain,
        "per_concept": per_concept,
    }


def hierarchical_cluster_recovery(vectors: np.ndarray, labels: list,
                                   n_clusters: int) -> dict:
    """Run hierarchical clustering and measure domain recovery.

    Uses average linkage on cosine distances, cuts at n_clusters,
    and compares to ground-truth domain labels.

    Args:
        vectors: (n, d) embedding array.
        labels: list of domain labels, length n.
        n_clusters: number of clusters to cut at.

    Returns:
        Dict with adjusted_rand_index, normalized_mutual_info,
        cluster_assignments, misassigned (list of (concept_idx, true, predicted)).
    """
    # Cosine distance for linkage
    cos_dist = pdist(vectors, metric="cosine")
    # Average linkage with cosine distance (Ward requires euclidean)
    Z = linkage(cos_dist, method="average")
    predicted = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Encode true labels as integers
    unique_labels = sorted(set(labels))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    true_int = [label_map[l] for l in labels]

    ari = adjusted_rand_score(true_int, predicted)
    nmi = normalized_mutual_info_score(true_int, predicted)

    # Find misassigned: concepts whose cluster doesn't match majority domain
    # Map each cluster to its majority domain
    cluster_to_domain = {}
    for c_id in set(predicted):
        members = [labels[i] for i in range(len(labels)) if predicted[i] == c_id]
        from collections import Counter
        majority = Counter(members).most_common(1)[0][0]
        cluster_to_domain[c_id] = majority

    misassigned = []
    for i, (true_label, pred_cluster) in enumerate(zip(labels, predicted)):
        pred_domain = cluster_to_domain[pred_cluster]
        if true_label != pred_domain:
            misassigned.append((i, true_label, pred_domain))

    return {
        "adjusted_rand_index": float(ari),
        "normalized_mutual_info": float(nmi),
        "cluster_assignments": predicted.tolist(),
        "misassigned": misassigned,
        "linkage_matrix": Z,
    }


def pca_projection(vectors: np.ndarray) -> dict:
    """Project embeddings to 2D via PCA.

    Args:
        vectors: (n, d) embedding array.

    Returns:
        Dict with coords (n, 2) array and variance_explained (2-tuple).
    """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    return {
        "coords": coords,
        "variance_explained": tuple(pca.explained_variance_ratio_),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(embedding_result: EmbeddingResult) -> ClusterReport:
    """Run full cluster validation and return a report.

    Args:
        embedding_result: EmbeddingResult from get_embeddings().

    Returns:
        ClusterReport with all analysis results.
    """
    concepts = list(embedding_result.concepts)
    vectors = embedding_result.vectors
    labels = [c.domain for c in concepts]
    words = [c.word for c in concepts]
    unique_domains = sorted(set(labels))
    n_domains = len(unique_domains)

    sim_matrix = cosine_similarity_matrix(vectors)
    wb = within_between_similarity(sim_matrix, labels)
    sil = compute_silhouette(vectors, labels)
    clust = hierarchical_cluster_recovery(vectors, labels, n_domains)
    pca = pca_projection(vectors)

    # Map per-concept silhouette from index to word
    per_concept_sil = {words[i]: score for i, score in sil["per_concept"].items()}

    # Map misassigned from index to word
    misassigned = [(words[i], true, pred) for i, true, pred in clust["misassigned"]]

    return ClusterReport(
        model_name=embedding_result.model_name,
        concepts=words,
        domains=unique_domains,
        n_concepts=len(concepts),
        n_domains=n_domains,
        embedding_dim=vectors.shape[1],
        cosine_sim_matrix=sim_matrix,
        mean_within_domain_similarity=wb["mean_within"],
        mean_between_domain_similarity=wb["mean_between"],
        separation_ratio=wb["separation_ratio"],
        per_domain_within=wb["per_domain_within"],
        silhouette_avg=sil["silhouette_avg"],
        per_domain_silhouette=sil["per_domain"],
        per_concept_silhouette=per_concept_sil,
        adjusted_rand_index=clust["adjusted_rand_index"],
        normalized_mutual_info=clust["normalized_mutual_info"],
        cluster_assignments=clust["cluster_assignments"],
        misassigned_concepts=misassigned,
        pca_coords=pca["coords"],
        pca_variance_explained=pca["variance_explained"],
    )


def format_report(report: ClusterReport) -> str:
    """Format a ClusterReport as human-readable text.

    Args:
        report: ClusterReport to format.

    Returns:
        Multi-line string.
    """
    lines = []
    lines.append(f"Cluster Validation Report")
    lines.append(f"Embedding model: {report.model_name}")
    lines.append(f"Concepts: {report.n_concepts} across {report.n_domains} domains")
    lines.append(f"Embedding dimensionality: {report.embedding_dim}")
    lines.append("")

    lines.append("--- Within vs Between Domain Similarity ---")
    lines.append(f"Mean within-domain cosine similarity:  {report.mean_within_domain_similarity:.4f}")
    lines.append(f"Mean between-domain cosine similarity: {report.mean_between_domain_similarity:.4f}")
    lines.append(f"Separation ratio (within/between):     {report.separation_ratio:.4f}")
    lines.append("")

    lines.append("Per-domain within-domain similarity:")
    for domain in sorted(report.per_domain_within.keys()):
        lines.append(f"  {domain:15s} {report.per_domain_within[domain]:.4f}")
    lines.append("")

    lines.append("--- Silhouette Analysis (cosine distance) ---")
    lines.append(f"Average silhouette score: {report.silhouette_avg:.4f}")
    lines.append("")
    lines.append("Per-domain silhouette:")
    for domain in sorted(report.per_domain_silhouette.keys()):
        lines.append(f"  {domain:15s} {report.per_domain_silhouette[domain]:.4f}")
    lines.append("")

    lines.append("Per-concept silhouette:")
    # List all concepts sorted by score
    sorted_concepts = sorted(report.per_concept_silhouette.items(), key=lambda x: x[1])
    for word, score in sorted_concepts:
        marker = " <-- LOW" if score < 0.0 else ""
        lines.append(f"  {word:15s} {score:.4f}{marker}")
    lines.append("")

    lines.append("--- Hierarchical Cluster Recovery ---")
    lines.append(f"Adjusted Rand Index:       {report.adjusted_rand_index:.4f}")
    lines.append(f"Normalized Mutual Info:    {report.normalized_mutual_info:.4f}")
    if report.misassigned_concepts:
        lines.append(f"Misassigned concepts ({len(report.misassigned_concepts)}):")
        for word, true_domain, pred_domain in report.misassigned_concepts:
            lines.append(f"  {word:15s} assigned={true_domain}, clustered with={pred_domain}")
    else:
        lines.append("All concepts clustered with their assigned domain.")
    lines.append("")

    lines.append("--- PCA Projection ---")
    lines.append(f"Variance explained: PC1={report.pca_variance_explained[0]:.3f}, "
                 f"PC2={report.pca_variance_explained[1]:.3f}")
    lines.append("")

    lines.append("--- Interpretation Guide ---")
    lines.append("Silhouette score: -1 to 1. Above 0.5 = strong clustering. "
                 "0.25-0.5 = moderate. Below 0.25 = weak.")
    lines.append("Separation ratio: >1 means within-domain similarity exceeds "
                 "between-domain. Higher = better separation.")
    lines.append("ARI: 0 = random, 1 = perfect recovery. "
                 "Above 0.65 = good agreement with domain labels.")
    lines.append("NMI: 0 = no mutual information, 1 = perfect. "
                 "Above 0.5 = meaningful overlap.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_scatter_plot(report: ClusterReport, output_path: str):
    """Save a 2D PCA scatter plot colored by domain.

    Args:
        report: ClusterReport with pca_coords populated.
        output_path: Path to save PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = report.pca_coords
    concepts = report.concepts
    domain_list = report.domains
    per_domain_count = report.n_concepts // report.n_domains

    colors = {"physical": "#2196F3", "institutional": "#FF9800", "moral": "#E91E63"}
    default_colors = ["#2196F3", "#FF9800", "#E91E63", "#4CAF50", "#9C27B0", "#795548"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Concepts are sorted by (domain, word), domains are sorted
    idx = 0
    for d_idx, domain in enumerate(sorted(domain_list)):
        color = colors.get(domain, default_colors[d_idx % len(default_colors)])
        end_idx = idx + per_domain_count
        ax.scatter(coords[idx:end_idx, 0], coords[idx:end_idx, 1],
                   c=color, label=domain, s=100, alpha=0.8, edgecolors="white", linewidth=0.5)
        for i in range(idx, end_idx):
            ax.annotate(concepts[i], (coords[i, 0], coords[i, 1]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        idx = end_idx

    ax.set_xlabel(f"PC1 ({report.pca_variance_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({report.pca_variance_explained[1]:.1%} variance)")
    ax.set_title(f"Concept Inventory Cluster Validation ({report.model_name})")
    ax.legend(title="Domain", loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_dendrogram(vectors: np.ndarray, labels: list, words: list,
                    output_path: str):
    """Save a dendrogram of hierarchical clustering.

    Args:
        vectors: (n, d) embedding array.
        labels: domain labels.
        words: concept words.
        output_path: Path to save PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cos_dist = pdist(vectors, metric="cosine")
    Z = linkage(cos_dist, method="average")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    leaf_labels = [f"{w} [{l[:4]}]" for w, l in zip(words, labels)]

    dendrogram(Z, labels=leaf_labels, ax=ax, leaf_rotation=45, leaf_font_size=9)
    ax.set_ylabel("Cosine distance")
    ax.set_title("Hierarchical Clustering of Concept Inventory")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Embedding-based cluster validation for RCP concept inventories."
    )
    parser.add_argument("--config", required=True, help="Path to RCP config.json")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="sentence-transformers model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--output", default=None,
                        help="Output directory for report and plots (default: print to stdout)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation (useful for CI/testing)")
    args = parser.parse_args()

    print(f"Loading concepts from {args.config}...")
    concepts = load_concepts_from_config(args.config)
    print(f"  {len(concepts)} concepts across {len(set(c.domain for c in concepts))} domains")

    print(f"Computing embeddings with {args.model}...")
    embedding_result = get_embeddings(concepts, model_name=args.model)
    print(f"  Embedding shape: {embedding_result.vectors.shape}")

    print("Running analysis...")
    report = generate_report(embedding_result)

    text = format_report(report)

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "cluster_validation_report.txt"
        with open(report_path, "w") as f:
            f.write(text)
        print(f"Report saved to {report_path}")

        # Save JSON for programmatic use
        json_path = output_dir / "cluster_validation.json"
        json_data = {
            "model_name": report.model_name,
            "concepts": report.concepts,
            "domains": report.domains,
            "n_concepts": report.n_concepts,
            "n_domains": report.n_domains,
            "embedding_dim": report.embedding_dim,
            "mean_within_domain_similarity": report.mean_within_domain_similarity,
            "mean_between_domain_similarity": report.mean_between_domain_similarity,
            "separation_ratio": report.separation_ratio,
            "per_domain_within": report.per_domain_within,
            "silhouette_avg": report.silhouette_avg,
            "per_domain_silhouette": report.per_domain_silhouette,
            "per_concept_silhouette": report.per_concept_silhouette,
            "adjusted_rand_index": report.adjusted_rand_index,
            "normalized_mutual_info": report.normalized_mutual_info,
            "cluster_assignments": report.cluster_assignments,
            "misassigned_concepts": [
                {"word": w, "assigned": t, "clustered_with": p}
                for w, t, p in report.misassigned_concepts
            ],
            "pca_variance_explained": list(report.pca_variance_explained),
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, cls=NumpyEncoder)
        print(f"JSON saved to {json_path}")

        if not args.no_plots:
            scatter_path = output_dir / "pca_scatter.png"
            save_scatter_plot(report, str(scatter_path))
            print(f"Scatter plot saved to {scatter_path}")

            labels = [c.domain for c in concepts]
            words = [c.word for c in concepts]
            dendro_path = output_dir / "dendrogram.png"
            save_dendrogram(embedding_result.vectors, labels, words, str(dendro_path))
            print(f"Dendrogram saved to {dendro_path}")
    else:
        print()
        print(text)


if __name__ == "__main__":
    main()
