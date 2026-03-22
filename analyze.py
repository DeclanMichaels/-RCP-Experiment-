#!/usr/bin/env python3
"""
RCP Analysis Pipeline -- From JSONL to drift metrics and visualizations.

Usage:
    python analyze.py --data-dir data/               # Analyze all collected data
    python analyze.py --data-dir data/ --model claude-sonnet  # Single model
    python analyze.py --data-dir data/ --figures-only # Regenerate figures from cached matrices
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def load_records(data_dir, tag="main"):
    """Load all JSONL records from data directory matching the tag."""
    records = []
    data_dir = Path(data_dir)
    for fpath in sorted(data_dir.glob(f"{tag}_*.jsonl")):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def get_all_concepts(config):
    all_c = []
    for domain_concepts in config["concepts"].values():
        all_c.extend(domain_concepts)
    return sorted(all_c)


def get_concept_domain(concept, config):
    for domain, concepts in config["concepts"].items():
        if concept in concepts:
            return domain
    return None


# ---------------------------------------------------------------------------
# Matrix Construction
# ---------------------------------------------------------------------------

def build_similarity_matrices(records, config):
    """
    Build similarity matrices from records.
    Returns dict: (model_name, framing, temperature) -> {
        'matrix': np.array (n x n),
        'std_matrix': np.array (n x n),
        'concepts': list[str],
        'n_records': int,
        'parse_rate': float
    }
    """
    concepts = get_all_concepts(config)
    n = len(concepts)
    concept_idx = {c: i for i, c in enumerate(concepts)}

    # Group records by (model, framing, temp)
    groups = defaultdict(list)
    for r in records:
        if r.get("parsed") and r.get("rating") is not None:
            key = (r["model_name"], r["framing"], r["temperature"])
            groups[key].append(r)

    # Count total records per group for parse rate
    total_groups = defaultdict(int)
    for r in records:
        key = (r["model_name"], r["framing"], r["temperature"])
        total_groups[key] += 1

    matrices = {}
    for key, recs in groups.items():
        # Collect ratings per pair
        pair_ratings = defaultdict(list)
        for r in recs:
            i = concept_idx.get(r["concept_a"])
            j = concept_idx.get(r["concept_b"])
            if i is not None and j is not None:
                pair_ratings[(i, j)].append(r["rating"])

        # Build matrix
        mat = np.full((n, n), np.nan)
        std_mat = np.full((n, n), np.nan)
        np.fill_diagonal(mat, 7.0)  # self-similarity = max
        np.fill_diagonal(std_mat, 0.0)

        for (i, j), ratings in pair_ratings.items():
            mean_r = np.mean(ratings)
            std_r = np.std(ratings) if len(ratings) > 1 else 0.0
            mat[i, j] = mean_r
            mat[j, i] = mean_r  # symmetric
            std_mat[i, j] = std_r
            std_mat[j, i] = std_r

        total = total_groups[key]
        parsed = len(recs)
        parse_rate = parsed / total if total > 0 else 0.0

        matrices[key] = {
            "matrix": mat,
            "std_matrix": std_mat,
            "concepts": concepts,
            "n_records": parsed,
            "parse_rate": parse_rate,
        }

    return matrices


def similarity_to_distance(sim_matrix):
    """Convert similarity (1-7) to distance. Higher similarity = lower distance."""
    return 8.0 - sim_matrix


# ---------------------------------------------------------------------------
# Geometry: MDS
# ---------------------------------------------------------------------------

def compute_mds(distance_matrix, n_components=2, random_state=42):
    """
    Non-metric MDS embedding (ordinal -- respects rank ordering only).
    Returns (embedding, stress).
    """
    # Handle NaN: replace with median distance
    median_d = np.nanmedian(distance_matrix)
    dist_clean = np.where(np.isnan(distance_matrix), median_d, distance_matrix)
    np.fill_diagonal(dist_clean, 0.0)

    mds = MDS(
        n_components=n_components,
        metric=False,  # non-metric MDS for ordinal data
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress="auto",
        n_init=4,
    )
    embedding = mds.fit_transform(dist_clean)
    stress = mds.stress_
    return embedding, stress


def compute_mds_multidim(distance_matrix, dims=(2, 3, 5), random_state=42):
    """
    Run non-metric MDS at multiple dimensionalities.
    Returns dict: {n_components: (embedding, stress)}
    """
    results = {}
    for d in dims:
        emb, stress = compute_mds(distance_matrix, n_components=d, random_state=random_state)
        results[d] = (emb, stress)
    return results


# ---------------------------------------------------------------------------
# Drift Metrics
# ---------------------------------------------------------------------------

def compute_procrustes_drift(embedding_base, embedding_framed):
    """
    Procrustes alignment and residual.
    Returns (aligned_base, aligned_framed, disparity).
    disparity is the sum of squared differences after optimal alignment.
    """
    mtx1, mtx2, disparity = procrustes(embedding_base, embedding_framed)
    return mtx1, mtx2, disparity


def compute_rank_correlation(dist_matrix_base, dist_matrix_framed):
    """
    Spearman correlation of the flattened upper triangles.
    Returns (rho, p_value).
    """
    n = dist_matrix_base.shape[0]
    idx = np.triu_indices(n, k=1)
    vec_base = dist_matrix_base[idx]
    vec_framed = dist_matrix_framed[idx]

    # Remove NaN pairs
    valid = ~(np.isnan(vec_base) | np.isnan(vec_framed))
    if valid.sum() < 3:
        return np.nan, np.nan

    rho, p = spearmanr(vec_base[valid], vec_framed[valid])
    return rho, p


def compute_silhouette(embedding, config):
    """
    Silhouette score for the 3-domain clustering.
    Returns score (-1 to 1).
    """
    concepts = get_all_concepts(config)
    labels = [get_concept_domain(c, config) for c in concepts]

    # Encode labels as integers
    domain_map = {"physical": 0, "institutional": 1, "moral": 2}
    label_ints = np.array([domain_map[label] for label in labels])

    try:
        return silhouette_score(embedding, label_ints)
    except Exception:
        return np.nan


def compute_subdomain_drift(dist_base, dist_framed, concepts, config):
    """
    Compute drift separately for each domain's sub-matrix.
    Returns dict: domain -> mean_absolute_drift
    """
    domain_indices = defaultdict(list)
    for i, c in enumerate(concepts):
        d = get_concept_domain(c, config)
        domain_indices[d].append(i)

    drift_by_domain = {}
    for domain, idxs in domain_indices.items():
        sub_base = dist_base[np.ix_(idxs, idxs)]
        sub_framed = dist_framed[np.ix_(idxs, idxs)]
        # Mean absolute difference in within-domain distances
        valid = ~(np.isnan(sub_base) | np.isnan(sub_framed))
        if valid.sum() > 0:
            drift_by_domain[domain] = np.mean(np.abs(sub_base[valid] - sub_framed[valid]))
        else:
            drift_by_domain[domain] = np.nan

    return drift_by_domain


# ---------------------------------------------------------------------------
# Drift Decomposition
# ---------------------------------------------------------------------------

def decompose_drift(embedding_base, embedding_framed, concepts, config):
    """
    After Procrustes alignment, decompose residuals into systematic vs. random.
    Returns dict with total_residual, systematic_variance, random_variance, ratio.
    """
    mtx1, mtx2, disparity = procrustes(embedding_base, embedding_framed)
    residuals = mtx2 - mtx1  # per-point residual vectors

    # Systematic: compute mean residual per domain
    domain_indices = defaultdict(list)
    for i, c in enumerate(concepts):
        d = get_concept_domain(c, config)
        domain_indices[d].append(i)

    domain_means = {}
    for domain, idxs in domain_indices.items():
        domain_means[domain] = np.mean(residuals[idxs], axis=0)

    # Systematic component: replace each point's residual with its domain mean
    systematic = np.zeros_like(residuals)
    for domain, idxs in domain_indices.items():
        for i in idxs:
            systematic[i] = domain_means[domain]

    random_component = residuals - systematic

    total_var = np.sum(residuals ** 2)
    systematic_var = np.sum(systematic ** 2)
    random_var = np.sum(random_component ** 2)

    ratio = systematic_var / random_var if random_var > 0 else float("inf")

    return {
        "disparity": disparity,
        "total_residual_ss": float(total_var),
        "systematic_ss": float(systematic_var),
        "random_ss": float(random_var),
        "systematic_random_ratio": float(ratio),
        "domain_mean_residuals": {d: v.tolist() for d, v in domain_means.items()},
    }


# ---------------------------------------------------------------------------
# Centroid Baseline Analysis
# ---------------------------------------------------------------------------

def compute_centroid_baseline(matrices, config):
    """
    Compute distance from neutral geometry to each cultural framing.
    Also compute a centroid geometry (average across cultural framings).
    Returns dict per model: {framing: distance_to_neutral, 'centroid': avg_matrix}
    """
    cultural_framings = ["individualist", "collectivist", "hierarchical", "egalitarian"]
    results = {}

    models = set(k[0] for k in matrices)
    for model in sorted(models):
        baseline_key = (model, "neutral", 0.0)
        if baseline_key not in matrices:
            continue

        neutral_dist = similarity_to_distance(matrices[baseline_key]["matrix"])
        n = neutral_dist.shape[0]
        idx = np.triu_indices(n, k=1)
        neutral_vec = neutral_dist[idx]

        model_results = {}
        cultural_dists = []

        for framing in cultural_framings:
            key = (model, framing, 0.0)
            if key not in matrices:
                continue
            framed_dist = similarity_to_distance(matrices[key]["matrix"])
            framed_vec = framed_dist[idx]

            valid = ~(np.isnan(neutral_vec) | np.isnan(framed_vec))
            if valid.sum() > 0:
                rho, _ = spearmanr(neutral_vec[valid], framed_vec[valid])
                mean_abs_diff = np.mean(np.abs(neutral_vec[valid] - framed_vec[valid]))
                model_results[framing] = {
                    "spearman_rho": float(rho),
                    "mean_abs_distance": float(mean_abs_diff),
                }
            cultural_dists.append(framed_dist)

        # Compute centroid (average distance matrix across cultural framings)
        if cultural_dists:
            centroid = np.nanmean(cultural_dists, axis=0)
            centroid_vec = centroid[idx]
            valid = ~(np.isnan(neutral_vec) | np.isnan(centroid_vec))
            if valid.sum() > 0:
                rho, _ = spearmanr(neutral_vec[valid], centroid_vec[valid])
                model_results["neutral_to_centroid"] = {
                    "spearman_rho": float(rho),
                    "mean_abs_distance": float(np.mean(np.abs(neutral_vec[valid] - centroid_vec[valid]))),
                }

        results[model] = model_results

    return results


# ---------------------------------------------------------------------------
# Moral Flattening Detection
# ---------------------------------------------------------------------------

def detect_moral_flattening(matrices, config):
    """
    Check if moral sub-matrix variance compresses under framing.
    Returns dict per (model, framing): {neutral_var, framed_var, ratio, is_flattened}
    """
    concepts = get_all_concepts(config)
    moral_indices = [i for i, c in enumerate(concepts) if get_concept_domain(c, config) == "moral"]
    results = {}

    models = set(k[0] for k in matrices)
    for model in sorted(models):
        baseline_key = (model, "neutral", 0.0)
        if baseline_key not in matrices:
            continue

        neutral_sim = matrices[baseline_key]["matrix"]
        neutral_moral = neutral_sim[np.ix_(moral_indices, moral_indices)]
        # Variance of upper triangle only (exclude diagonal)
        idx = np.triu_indices(len(moral_indices), k=1)
        neutral_var = np.nanvar(neutral_moral[idx])

        for key in matrices:
            m, framing, temp = key
            if m != model or framing == "neutral" or temp != 0.0:
                continue

            framed_sim = matrices[key]["matrix"]
            framed_moral = framed_sim[np.ix_(moral_indices, moral_indices)]
            framed_var = np.nanvar(framed_moral[idx])
            framed_mean = np.nanmean(framed_moral[idx])

            ratio = framed_var / neutral_var if neutral_var > 0 else float("inf")
            # Flattened if variance drops below 50% and mean is near midpoint (3-5)
            is_flattened = bool(ratio < 0.5 and 2.5 < framed_mean < 5.5)

            results[(model, framing)] = {
                "neutral_variance": float(neutral_var),
                "framed_variance": float(framed_var),
                "variance_ratio": float(ratio),
                "framed_mean": float(framed_mean),
                "is_flattened": is_flattened,
            }

    return results


# ---------------------------------------------------------------------------
# Tie Density Computation
# ---------------------------------------------------------------------------

def compute_tie_density(matrices, config):
    """
    Compute tie density for each (model, framing) condition.
    Tie = two or more pairs with the same rating.
    Reports overall and within-moral-domain tie density.
    """
    concepts = get_all_concepts(config)
    moral_indices = [i for i, c in enumerate(concepts) if get_concept_domain(c, config) == "moral"]
    n = len(concepts)
    full_idx = np.triu_indices(n, k=1)
    moral_n = len(moral_indices)
    moral_idx = np.triu_indices(moral_n, k=1)

    results = {}
    for key, mat_data in matrices.items():
        model, framing, temp = key
        sim = mat_data["matrix"]

        # Full matrix tie density
        full_vals = sim[full_idx]
        full_valid = full_vals[~np.isnan(full_vals)]
        if len(full_valid) > 0:
            unique_count = len(np.unique(full_valid))
            full_tie_density = 1.0 - (unique_count / len(full_valid))
        else:
            full_tie_density = float("nan")

        # Moral sub-matrix tie density
        moral_sub = sim[np.ix_(moral_indices, moral_indices)]
        moral_vals = moral_sub[moral_idx]
        moral_valid = moral_vals[~np.isnan(moral_vals)]
        if len(moral_valid) > 0:
            moral_unique = len(np.unique(moral_valid))
            moral_tie_density = 1.0 - (moral_unique / len(moral_valid))
        else:
            moral_tie_density = float("nan")

        results[(model, framing, temp)] = {
            "full_tie_density": float(full_tie_density),
            "moral_tie_density": float(moral_tie_density),
            "full_unique_values": int(unique_count) if not np.isnan(full_tie_density) else 0,
            "moral_unique_values": int(moral_unique) if not np.isnan(moral_tie_density) else 0,
            "moral_pair_count": int(len(moral_valid)),
        }

    return results


# ---------------------------------------------------------------------------
# Full Analysis
# ---------------------------------------------------------------------------

def run_analysis(matrices, config, output_dir):
    """
    Run full analysis pipeline on constructed matrices.
    Returns a summary dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts = get_all_concepts(config)
    results = []

    # Group matrices by model
    models = set()
    framings = set()
    temps = set()
    for (model, framing, temp) in matrices.keys():
        models.add(model)
        framings.add(framing)
        temps.add(temp)

    print(f"Analyzing: {len(models)} models, {len(framings)} framings, {len(temps)} temperatures")
    print(f"Models: {sorted(models)}")

    for model in sorted(models):
        # Get baseline (neutral) matrix at temperature 0.0 (prefer deterministic)
        baseline_temp = 0.0 if 0.0 in temps else min(temps)
        baseline_key = (model, "neutral", baseline_temp)
        if baseline_key not in matrices:
            print(f"  Warning: no neutral baseline for {model} at temp={baseline_temp}")
            continue

        baseline_sim = matrices[baseline_key]["matrix"]
        baseline_dist = similarity_to_distance(baseline_sim)
        baseline_emb, baseline_stress = compute_mds(baseline_dist)
        baseline_silhouette = compute_silhouette(baseline_emb, config)

        # Multi-dimensional stress comparison (2D, 3D, 5D)
        baseline_multidim = compute_mds_multidim(baseline_dist)
        baseline_stress_by_dim = {d: s for d, (_, s) in baseline_multidim.items()}

        print(f"\n  {model} baseline: stress_2d={baseline_stress:.4f}, "
              f"stress_3d={baseline_stress_by_dim.get(3, 0):.4f}, "
              f"stress_5d={baseline_stress_by_dim.get(5, 0):.4f}, "
              f"silhouette={baseline_silhouette:.3f}")

        for framing in sorted(framings):
            if framing == "neutral":
                continue

            for temp in sorted(temps):
                key = (model, framing, temp)
                if key not in matrices:
                    continue

                mat_data = matrices[key]
                framed_sim = mat_data["matrix"]
                framed_dist = similarity_to_distance(framed_sim)
                framed_emb, framed_stress = compute_mds(framed_dist)

                # Multi-dimensional stress and cross-dim Procrustes
                framed_multidim = compute_mds_multidim(framed_dist)
                procrustes_by_dim = {}
                for dim, (f_emb, f_stress) in framed_multidim.items():
                    b_emb = baseline_multidim[dim][0]
                    _, _, disp = compute_procrustes_drift(b_emb, f_emb)
                    procrustes_by_dim[dim] = float(disp)

                # Procrustes (2D, primary)
                _, _, disparity = compute_procrustes_drift(baseline_emb, framed_emb)

                # Rank correlation
                rho, rho_p = compute_rank_correlation(baseline_dist, framed_dist)

                # Silhouette
                sil = compute_silhouette(framed_emb, config)

                # Per-domain drift
                domain_drift = compute_subdomain_drift(
                    baseline_dist, framed_dist, concepts, config
                )

                # Decomposition
                decomp = decompose_drift(
                    baseline_emb, framed_emb, concepts, config
                )

                row = {
                    "model": model,
                    "framing": framing,
                    "temperature": temp,
                    "procrustes_disparity": float(disparity),
                    "procrustes_3d": procrustes_by_dim.get(3, float("nan")),
                    "procrustes_5d": procrustes_by_dim.get(5, float("nan")),
                    "spearman_rho": float(rho),
                    "spearman_p": float(rho_p),
                    "silhouette_baseline": float(baseline_silhouette),
                    "silhouette_framed": float(sil),
                    "silhouette_delta": float(sil - baseline_silhouette),
                    "drift_physical": float(domain_drift.get("physical", np.nan)),
                    "drift_institutional": float(domain_drift.get("institutional", np.nan)),
                    "drift_moral": float(domain_drift.get("moral", np.nan)),
                    "systematic_random_ratio": decomp["systematic_random_ratio"],
                    "parse_rate": mat_data["parse_rate"],
                    "n_records": mat_data["n_records"],
                    "mds_stress_2d": float(framed_stress),
                    "mds_stress_3d": float(framed_multidim[3][1]),
                    "mds_stress_5d": float(framed_multidim[5][1]),
                }
                results.append(row)

                print(f"    {framing} (t={temp}): procrustes={disparity:.4f}, "
                      f"rho={rho:.3f}, drift P/I/M="
                      f"{domain_drift.get('physical', 0):.3f}/"
                      f"{domain_drift.get('institutional', 0):.3f}/"
                      f"{domain_drift.get('moral', 0):.3f}")

    # Save results
    results_path = output_dir / "drift_metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDrift metrics saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

DOMAIN_COLORS = {
    "physical": "#2196F3",
    "institutional": "#FF9800",
    "moral": "#E91E63",
}

FRAMING_MARKERS = {
    "individualist": "o",
    "collectivist": "s",
    "hierarchical": "^",
    "egalitarian": "D",
}


def plot_domain_drift(results, output_dir):
    """
    Centerpiece figure: domain drift by model across framings.
    """
    output_dir = Path(output_dir)
    # Filter to deterministic temperature
    det_results = [r for r in results if r["temperature"] == 0.0]
    if not det_results:
        det_results = results  # fall back

    models = sorted(set(r["model"] for r in det_results))
    domains = ["physical", "institutional", "moral"]

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_results = [r for r in det_results if r["model"] == model]

        for framing in sorted(set(r["framing"] for r in model_results)):
            framing_results = [r for r in model_results if r["framing"] == framing]
            if not framing_results:
                continue
            r = framing_results[0]  # one row per framing at this temp
            drifts = [r[f"drift_{d}"] for d in domains]
            marker = FRAMING_MARKERS.get(framing, "x")
            ax.plot(domains, drifts, marker=marker, label=framing, linewidth=1.5, markersize=8)

        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xlabel("Domain")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean Absolute Drift from Neutral")
    axes[-1].legend(fontsize=8, loc="upper left")

    fig.suptitle("Representational Drift by Domain Under Cultural Framing", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "domain_drift.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_mds_projections(matrices, config, output_dir):
    """
    MDS scatter plots: one panel per (model, framing), colored by domain.
    """
    output_dir = Path(output_dir)
    concepts = get_all_concepts(config)
    domain_labels = [get_concept_domain(c, config) for c in concepts]

    # Filter to deterministic, or whatever is available
    relevant_keys = [k for k in matrices if k[2] == 0.0]
    if not relevant_keys:
        relevant_keys = list(matrices.keys())

    models = sorted(set(k[0] for k in relevant_keys))
    framings = sorted(set(k[1] for k in relevant_keys))

    n_rows = len(models)
    n_cols = len(framings)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, model in enumerate(models):
        for j, framing in enumerate(framings):
            ax = axes[i, j]
            temp = 0.0 if (model, framing, 0.0) in matrices else min(
                k[2] for k in matrices if k[0] == model and k[1] == framing
            )
            key = (model, framing, temp)
            if key not in matrices:
                ax.set_visible(False)
                continue

            sim = matrices[key]["matrix"]
            dist = similarity_to_distance(sim)
            emb, stress = compute_mds(dist)

            for k_idx, (x, y) in enumerate(emb):
                domain = domain_labels[k_idx]
                color = DOMAIN_COLORS.get(domain, "gray")
                ax.scatter(x, y, c=color, s=30, zorder=3)
                ax.annotate(
                    concepts[k_idx], (x, y),
                    fontsize=5, ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points",
                )

            if i == 0:
                ax.set_title(framing, fontsize=9)
            if j == 0:
                ax.set_ylabel(model, fontsize=9)
            ax.tick_params(labelsize=6)

    fig.suptitle("MDS Projections (2D) by Model and Framing", fontsize=12)
    fig.tight_layout()
    path = output_dir / "mds_projections.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_decomposition(results, output_dir):
    """
    Systematic vs random residual ratio by model and framing.
    """
    output_dir = Path(output_dir)
    det_results = [r for r in results if r["temperature"] == 0.0]
    if not det_results:
        det_results = results

    models = sorted(set(r["model"] for r in det_results))
    framings = sorted(set(r["framing"] for r in det_results))

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(framings))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        ratios = []
        for framing in framings:
            rows = [r for r in det_results if r["model"] == model and r["framing"] == framing]
            if rows:
                ratios.append(rows[0]["systematic_random_ratio"])
            else:
                ratios.append(0)
        ax.bar(x + i * width, ratios, width, label=model)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(framings, rotation=30)
    ax.set_ylabel("Systematic / Random Ratio")
    ax.set_title("Drift Decomposition: Structured Rotation vs. Noise")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "decomposition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_vector_displacement(matrices, config, output_dir):
    """
    Vector displacement plots: overlay neutral and framed positions,
    draw arrows colored by domain.
    """
    output_dir = Path(output_dir)
    concepts = get_all_concepts(config)
    domain_labels = [get_concept_domain(c, config) for c in concepts]

    models = sorted(set(k[0] for k in matrices if k[2] == 0.0))
    cultural_framings = [f for f in ["individualist", "collectivist", "hierarchical", "egalitarian"]
                         if any(k[1] == f for k in matrices)]

    if not models or not cultural_framings:
        return

    fig, axes = plt.subplots(len(models), len(cultural_framings),
                             figsize=(3.5 * len(cultural_framings), 3.5 * len(models)))
    if len(models) == 1:
        axes = axes[np.newaxis, :]
    if len(cultural_framings) == 1:
        axes = axes[:, np.newaxis]

    for i, model in enumerate(models):
        baseline_key = (model, "neutral", 0.0)
        if baseline_key not in matrices:
            continue
        baseline_dist = similarity_to_distance(matrices[baseline_key]["matrix"])
        baseline_emb, _ = compute_mds(baseline_dist)

        for j, framing in enumerate(cultural_framings):
            ax = axes[i, j]
            key = (model, framing, 0.0)
            if key not in matrices:
                ax.set_visible(False)
                continue

            framed_dist = similarity_to_distance(matrices[key]["matrix"])
            framed_emb, _ = compute_mds(framed_dist)

            # Procrustes align framed to baseline for comparable coordinates
            base_aligned, framed_aligned, _ = procrustes(baseline_emb, framed_emb)

            for k_idx in range(len(concepts)):
                domain = domain_labels[k_idx]
                color = DOMAIN_COLORS.get(domain, "gray")
                x0, y0 = base_aligned[k_idx]
                x1, y1 = framed_aligned[k_idx]

                ax.scatter(x0, y0, c=color, s=20, zorder=3, alpha=0.6)
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.7))
                ax.annotate(concepts[k_idx], (x0, y0), fontsize=4.5, ha="center",
                            va="bottom", xytext=(0, 2), textcoords="offset points")

            if i == 0:
                ax.set_title(framing, fontsize=9)
            if j == 0:
                ax.set_ylabel(model, fontsize=9)
            ax.tick_params(labelsize=6)

    fig.suptitle("Vector Displacement: Neutral to Framed (arrows colored by domain)", fontsize=11)
    fig.tight_layout()
    path = output_dir / "vector_displacement.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RCP Analysis Pipeline")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-dir", default="data", help="Directory with JSONL files")
    parser.add_argument("--output-dir", default="results", help="Output directory for figures/metrics")
    parser.add_argument("--model", default=None, help="Analyze single model")
    parser.add_argument("--figures-only", action="store_true",
                        help="Regenerate figures from cached metrics")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.figures_only:
        metrics_path = Path(args.output_dir) / "drift_metrics.json"
        if not metrics_path.exists():
            print(f"Error: {metrics_path} not found. Run full analysis first.")
            sys.exit(1)
        with open(metrics_path) as f:
            results = json.load(f)
        plot_domain_drift(results, args.output_dir)
        plot_decomposition(results, args.output_dir)
        print("Figures regenerated.")
        return

    # Load and filter records
    records = load_records(args.data_dir, tag="main")
    if not records:
        print(f"No records found in {args.data_dir}/main_*.jsonl")
        sys.exit(1)

    if args.model:
        records = [r for r in records if r["model_name"] == args.model]

    print(f"Loaded {len(records)} records")

    # Build matrices
    matrices = build_similarity_matrices(records, config)
    print(f"Built {len(matrices)} similarity matrices")

    # Run analysis
    results = run_analysis(matrices, config, args.output_dir)

    # Generate figures
    if results:
        plot_domain_drift(results, args.output_dir)
        plot_mds_projections(matrices, config, args.output_dir)
        plot_decomposition(results, args.output_dir)
        plot_vector_displacement(matrices, config, args.output_dir)

    # Centroid baseline analysis
    centroid_results = compute_centroid_baseline(matrices, config)
    if centroid_results:
        centroid_path = Path(args.output_dir) / "centroid_baseline.json"
        with open(centroid_path, "w") as f:
            json.dump(centroid_results, f, indent=2)
        print(f"\nCentroid baseline analysis saved to {centroid_path}")
        for model, data in centroid_results.items():
            print(f"  {model}:")
            for framing, metrics in data.items():
                if isinstance(metrics, dict):
                    print(f"    {framing}: rho={metrics.get('spearman_rho', 'N/A'):.3f}, "
                          f"dist={metrics.get('mean_abs_distance', 'N/A'):.3f}")

    # Moral flattening detection
    flattening = detect_moral_flattening(matrices, config)
    if flattening:
        flat_path = Path(args.output_dir) / "moral_flattening.json"
        flat_serializable = {f"{k[0]}/{k[1]}": v for k, v in flattening.items()}
        with open(flat_path, "w") as f:
            json.dump(flat_serializable, f, indent=2)
        print(f"\nMoral flattening analysis saved to {flat_path}")
        flattened_count = sum(1 for v in flattening.values() if v["is_flattened"])
        print(f"  {flattened_count}/{len(flattening)} conditions show moral flattening")

    # Tie density analysis
    tie_density = compute_tie_density(matrices, config)
    if tie_density:
        td_path = Path(args.output_dir) / "tie_density.json"
        td_serializable = {f"{k[0]}/{k[1]}/t={k[2]}": v for k, v in tie_density.items()}
        with open(td_path, "w") as f:
            json.dump(td_serializable, f, indent=2)
        print(f"\nTie density analysis saved to {td_path}")
        # Flag conditions where moral tie density exceeds 50%
        high_tie = [(k, v) for k, v in tie_density.items() if v["moral_tie_density"] > 0.5]
        if high_tie:
            print(f"  WARNING: {len(high_tie)} condition(s) have >50% moral tie density:")
            for (m, f_name, t), v in high_tie:
                print(f"    {m}/{f_name}: {v['moral_tie_density']:.0%} "
                      f"({v['moral_unique_values']} unique values in {v['moral_pair_count']} pairs)")
        else:
            print("  All conditions below 50% moral tie density threshold")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
