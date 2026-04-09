"""
Factor analysis construct validation for RCP concept inventories.

Validates that domain assignments (physical, institutional, moral) correspond
to factor structure in the actual LLM similarity rating data. Uses exploratory
factor analysis (EFA) on unframed similarity matrices, with parallel analysis
for factor count determination and Tucker's congruence coefficient for
target-structure comparison.

This complements the embedding-based cluster validation by validating against
response structure rather than a proxy model.

Requires: scikit-learn, scipy, numpy, matplotlib

Usage:
    python factor_validate.py --data runs/20260324-1-Sonnet-Moral-Data/
    python factor_validate.py --data runs/*-Moral-Data/ --output results/
    python factor_validate.py --data runs/20260324-1-Sonnet-Moral-Data/ --config config.json
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.linalg import sqrtm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairRating:
    """A single pairwise similarity rating."""
    concept_a: str
    concept_b: str
    domain_a: str
    domain_b: str
    rating: int
    model_name: str


@dataclass
class SimilarityMatrix:
    """An 18x18 similarity matrix from one model's unframed ratings."""
    model_name: str
    concepts: list          # sorted concept names
    domains: list           # domain label per concept (same order)
    matrix: np.ndarray      # (n, n) raw similarity ratings (1-7 scale)
    correlation: np.ndarray # (n, n) normalized to correlation-like [0, 1] with 1.0 diagonal


@dataclass
class FactorReport:
    """Factor analysis results for one model."""
    model_name: str
    concepts: list
    domains: list
    unique_domains: list
    n_concepts: int
    n_factors: int

    # Factor loadings (n_concepts x n_factors)
    loadings: np.ndarray = field(repr=False)

    # Per-concept primary factor and domain match
    concept_assignments: list = field(default_factory=list)

    # Summary statistics
    domain_recovery_rate: float = 0.0
    variance_explained: list = field(default_factory=list)
    cumulative_variance: float = 0.0

    # KMO and Bartlett
    kmo_score: float = 0.0
    bartlett_chi2: float = 0.0
    bartlett_p: float = 0.0

    # Parallel analysis suggested factors
    parallel_n_factors: int = 0

    # Tucker's congruence coefficient (per factor, vs target)
    tucker_congruence: list = field(default_factory=list)

    # Eigenvalues (for scree)
    eigenvalues: list = field(default_factory=list)
    parallel_eigenvalues: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ratings_from_jsonl(data_path: str) -> list:
    """Load pairwise ratings from a Data directory's JSONL files.

    Reads all main_*.jsonl files in the directory. Filters to
    framing == "neutral" or "unframed", parsed == true.

    Args:
        data_path: Path to a Data directory containing JSONL files.

    Returns:
        List of PairRating objects.
    """
    data_dir = Path(data_path)
    if not data_dir.is_dir():
        raise ValueError(f"Not a directory: {data_path}")

    ratings = []
    jsonl_files = sorted(data_dir.glob("main_*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No main_*.jsonl files in {data_path}")

    for fpath in jsonl_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                framing = record.get("framing", "")
                if framing not in ("neutral", "unframed"):
                    continue
                if not record.get("parsed", False):
                    continue
                if record.get("is_refusal", False):
                    continue

                ratings.append(PairRating(
                    concept_a=record["concept_a"],
                    concept_b=record["concept_b"],
                    domain_a=record["domain_a"],
                    domain_b=record["domain_b"],
                    rating=int(record["rating"]),
                    model_name=record.get("model_name", "unknown"),
                ))

    return ratings


def build_similarity_matrix(ratings: list, concepts: list = None,
                            domain_map: dict = None) -> SimilarityMatrix:
    """Build an 18x18 similarity matrix from pairwise ratings.

    If multiple ratings exist for the same pair (e.g., stochastic reps),
    they are averaged.

    Args:
        ratings: List of PairRating objects (should be from one model).
        concepts: Optional sorted list of concept names.
        domain_map: Optional dict of concept -> domain.

    Returns:
        SimilarityMatrix with raw and correlation-normalized matrices.
    """
    if not ratings:
        raise ValueError("No ratings provided")

    if domain_map is None:
        domain_map = {}
        for r in ratings:
            domain_map[r.concept_a] = r.domain_a
            domain_map[r.concept_b] = r.domain_b

    if concepts is None:
        concepts = sorted(domain_map.keys())

    n = len(concepts)
    concept_idx = {c: i for i, c in enumerate(concepts)}
    domains = [domain_map[c] for c in concepts]

    sum_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n))

    for r in ratings:
        if r.concept_a not in concept_idx or r.concept_b not in concept_idx:
            continue
        i = concept_idx[r.concept_a]
        j = concept_idx[r.concept_b]
        sum_matrix[i, j] += r.rating
        sum_matrix[j, i] += r.rating
        count_matrix[i, j] += 1
        count_matrix[j, i] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_matrix = np.where(count_matrix > 0, sum_matrix / count_matrix, 0.0)

    np.fill_diagonal(raw_matrix, 7.0)
    corr_matrix = (raw_matrix - 1.0) / 6.0

    model_name = ratings[0].model_name if ratings else "unknown"

    return SimilarityMatrix(
        model_name=model_name,
        concepts=concepts,
        domains=domains,
        matrix=raw_matrix,
        correlation=corr_matrix,
    )


# ---------------------------------------------------------------------------
# Matrix conditioning
# ---------------------------------------------------------------------------

def nearest_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    """Find the nearest positive semi-definite matrix.

    Uses eigenvalue clamping with diagonal restoration.
    Factor analysis requires a PSD correlation matrix.

    Args:
        matrix: Symmetric matrix that may not be PSD.

    Returns:
        Nearest PSD matrix (Frobenius norm).
    """
    B = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.maximum(eigvals, 1e-10)
    result = eigvecs @ np.diag(eigvals) @ eigvecs.T
    result = (result + result.T) / 2.0
    d = np.sqrt(np.diag(result))
    d = np.where(d == 0, 1e-10, d)
    result = result / np.outer(d, d)
    np.fill_diagonal(result, 1.0)
    return result


def is_positive_semidefinite(matrix: np.ndarray, tol: float = -1e-8) -> bool:
    """Check if matrix is positive semi-definite."""
    eigvals = np.linalg.eigvalsh(matrix)
    return np.all(eigvals >= tol)


# ---------------------------------------------------------------------------
# Factor analysis primitives (pure numpy/scipy, no external packages)
# ---------------------------------------------------------------------------

def run_parallel_analysis(corr_matrix: np.ndarray, n_obs: int = 153,
                          n_iter: int = 100, percentile: int = 95) -> dict:
    """Run parallel analysis to determine number of factors.

    Generates random correlation matrices of the same size,
    extracts eigenvalues, and compares to actual eigenvalues.

    Args:
        corr_matrix: (n, n) correlation matrix.
        n_obs: Number of observations (pairs).
        n_iter: Number of random iterations.
        percentile: Percentile of random eigenvalues to use as threshold.

    Returns:
        Dict with actual_eigenvalues, random_eigenvalues (at percentile),
        and suggested_n_factors.
    """
    n = corr_matrix.shape[0]
    actual_eigvals = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]

    random_eigvals = np.zeros((n_iter, n))
    rng = np.random.default_rng(42)
    for i in range(n_iter):
        random_data = rng.standard_normal((n_obs, n))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigvals[i] = np.sort(np.linalg.eigvalsh(random_corr))[::-1]

    threshold = np.percentile(random_eigvals, percentile, axis=0)

    n_factors = 0
    for k in range(n):
        if actual_eigvals[k] > threshold[k]:
            n_factors += 1
        else:
            break

    return {
        "actual_eigenvalues": actual_eigvals.tolist(),
        "random_eigenvalues": threshold.tolist(),
        "suggested_n_factors": max(1, n_factors),
    }


def varimax_rotation(loadings: np.ndarray, max_iter: int = 100,
                     tol: float = 1e-6) -> np.ndarray:
    """Apply varimax rotation to factor loadings.

    Maximizes the sum of variances of squared loadings within each factor,
    producing a simpler structure (each variable loads heavily on fewer factors).

    Args:
        loadings: (n_variables, n_factors) matrix.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Rotated loadings matrix, same shape as input.
    """
    n, k = loadings.shape
    rotation = np.eye(k)
    rotated = loadings.copy()

    for _ in range(max_iter):
        old_rotation = rotation.copy()
        for i in range(k):
            for j in range(i + 1, k):
                x = rotated[:, i]
                y = rotated[:, j]
                u = x**2 - y**2
                v = 2 * x * y
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u**2 - v**2)
                D = 2 * np.sum(u * v)
                num = D - 2 * A * B / n
                den = C - (A**2 - B**2) / n
                phi = 0.25 * np.arctan2(num, den)

                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                new_i = rotated[:, i] * cos_phi + rotated[:, j] * sin_phi
                new_j = -rotated[:, i] * sin_phi + rotated[:, j] * cos_phi
                rotated[:, i] = new_i
                rotated[:, j] = new_j

        if np.max(np.abs(rotated - loadings @ np.linalg.lstsq(loadings, rotated, rcond=None)[0])) < tol:
            pass
        diff = np.max(np.abs(old_rotation - rotation))

    return rotated


def compute_kmo(corr: np.ndarray) -> float:
    """Compute Kaiser-Meyer-Olkin measure of sampling adequacy.

    KMO compares partial correlations to zero-order correlations.
    Values above 0.6 suggest the data is suitable for factor analysis.

    Args:
        corr: (n, n) correlation matrix.

    Returns:
        KMO overall score (0 to 1).
    """
    n = corr.shape[0]
    try:
        inv_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr)

    d = np.diag(inv_corr)
    d_safe = np.where(d == 0, 1e-10, d)
    partial = -inv_corr / np.sqrt(np.outer(d_safe, d_safe))
    np.fill_diagonal(partial, 0.0)

    corr_sq = corr.copy()
    np.fill_diagonal(corr_sq, 0.0)
    corr_sq = corr_sq ** 2

    partial_sq = partial ** 2

    num = np.sum(corr_sq)
    denom = num + np.sum(partial_sq)

    return float(num / denom) if denom > 0 else 0.0


def compute_bartlett(corr: np.ndarray, n_obs: int = 153) -> tuple:
    """Compute Bartlett's test of sphericity.

    Tests whether the correlation matrix is significantly different from
    an identity matrix (i.e., whether there are relationships to factor).

    Args:
        corr: (n, n) correlation matrix.
        n_obs: Number of observations.

    Returns:
        (chi_square, p_value) tuple.
    """
    from scipy.stats import chi2

    n = corr.shape[0]
    det = np.linalg.det(corr)
    if det <= 0:
        det = 1e-300

    statistic = -np.log(det) * (n_obs - 1 - (2 * n + 5) / 6)
    df = n * (n - 1) / 2
    p_value = 1.0 - chi2.cdf(statistic, df)

    return float(statistic), float(p_value)


def principal_axis_factoring(corr: np.ndarray, n_factors: int) -> tuple:
    """Extract factor loadings via principal axis factoring.

    Eigendecomposes the correlation matrix and takes the top n_factors
    eigenvectors scaled by sqrt(eigenvalue) as loadings.

    Args:
        corr: (n, n) correlation matrix (must be PSD).
        n_factors: Number of factors to extract.

    Returns:
        (loadings, eigenvalues) where loadings is (n, n_factors)
        and eigenvalues is the full sorted array.
    """
    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    k = min(n_factors, len(eigvals))
    top_vals = np.maximum(eigvals[:k], 0.0)
    loadings = eigvecs[:, :k] * np.sqrt(top_vals)

    return loadings, eigvals


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_factor_analysis(sim_matrix: SimilarityMatrix,
                        n_factors: int = 3) -> FactorReport:
    """Run exploratory factor analysis on unframed similarity data.

    Uses principal axis factoring with varimax rotation. All computations
    done with numpy/scipy (no external factor analysis package needed).

    Args:
        sim_matrix: SimilarityMatrix from one model.
        n_factors: Number of factors to extract (default 3 for 3 domains).

    Returns:
        FactorReport with loadings, assignments, and fit statistics.
    """
    corr = sim_matrix.correlation.copy()
    concepts = sim_matrix.concepts
    domains = sim_matrix.domains
    unique_domains = sorted(set(domains))
    n = len(concepts)

    if not is_positive_semidefinite(corr):
        corr = nearest_positive_semidefinite(corr)

    try:
        kmo_score = compute_kmo(corr)
    except Exception:
        kmo_score = 0.0

    try:
        bartlett_chi2, bartlett_p = compute_bartlett(corr, n_obs=153)
    except Exception:
        bartlett_chi2 = 0.0
        bartlett_p = 1.0

    parallel = run_parallel_analysis(corr, n_obs=153)

    raw_loadings, eigvals = principal_axis_factoring(corr, n_factors)
    loadings = varimax_rotation(raw_loadings)

    ss_loadings = np.sum(loadings ** 2, axis=0)
    prop_var = (ss_loadings / n).tolist()
    cum_var = float(np.sum(ss_loadings) / n)
    orig_eigenvalues = eigvals.tolist()

    factor_to_domain = {}
    for f_idx in range(n_factors):
        domain_avg = {}
        for domain in unique_domains:
            mask = [i for i, d in enumerate(domains) if d == domain]
            domain_avg[domain] = float(np.mean(np.abs(loadings[mask, f_idx])))
        factor_to_domain[f_idx] = max(domain_avg, key=domain_avg.get)

    concept_assignments = []
    matches = 0
    for i, (concept, domain) in enumerate(zip(concepts, domains)):
        primary_factor = int(np.argmax(np.abs(loadings[i])))
        primary_loading = float(loadings[i, primary_factor])
        predicted_domain = factor_to_domain[primary_factor]
        match = (predicted_domain == domain)
        if match:
            matches += 1
        concept_assignments.append({
            "concept": concept,
            "domain": domain,
            "primary_factor": primary_factor,
            "primary_loading": primary_loading,
            "predicted_domain": predicted_domain,
            "match": match,
        })

    recovery_rate = matches / n if n > 0 else 0.0

    target = np.zeros_like(loadings)
    for f_idx in range(n_factors):
        assigned_domain = factor_to_domain[f_idx]
        for i, domain in enumerate(domains):
            if domain == assigned_domain:
                target[i, f_idx] = 1.0

    tucker = []
    for f_idx in range(n_factors):
        num = np.dot(loadings[:, f_idx], target[:, f_idx])
        denom = np.sqrt(
            np.dot(loadings[:, f_idx], loadings[:, f_idx]) *
            np.dot(target[:, f_idx], target[:, f_idx])
        )
        tc = float(num / denom) if denom > 0 else 0.0
        tucker.append({
            "factor": f_idx,
            "domain": factor_to_domain[f_idx],
            "congruence": tc,
        })

    return FactorReport(
        model_name=sim_matrix.model_name,
        concepts=concepts,
        domains=domains,
        unique_domains=unique_domains,
        n_concepts=n,
        n_factors=n_factors,
        loadings=loadings,
        concept_assignments=concept_assignments,
        domain_recovery_rate=recovery_rate,
        variance_explained=prop_var,
        cumulative_variance=cum_var,
        kmo_score=kmo_score,
        bartlett_chi2=bartlett_chi2,
        bartlett_p=bartlett_p,
        parallel_n_factors=parallel["suggested_n_factors"],
        tucker_congruence=tucker,
        eigenvalues=orig_eigenvalues,
        parallel_eigenvalues=parallel["random_eigenvalues"],
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(report: FactorReport) -> str:
    """Format a FactorReport as human-readable text."""
    lines = []
    lines.append("Factor Analysis Construct Validation Report")
    lines.append(f"Model: {report.model_name}")
    lines.append(f"Concepts: {report.n_concepts} across {len(report.unique_domains)} domains")
    lines.append(f"Factors extracted: {report.n_factors}")
    lines.append("")

    lines.append("--- Sampling Adequacy ---")
    lines.append(f"KMO score: {report.kmo_score:.4f}")
    kmo_interp = ("excellent" if report.kmo_score >= 0.9 else
                  "good" if report.kmo_score >= 0.8 else
                  "adequate" if report.kmo_score >= 0.6 else
                  "poor" if report.kmo_score >= 0.5 else "unacceptable")
    lines.append(f"  Interpretation: {kmo_interp}")
    lines.append(f"Bartlett chi-square: {report.bartlett_chi2:.2f}, p = {report.bartlett_p:.6f}")
    lines.append("")

    lines.append("--- Parallel Analysis ---")
    lines.append(f"Suggested number of factors: {report.parallel_n_factors}")
    lines.append(f"Requested factors: {report.n_factors}")
    if report.parallel_n_factors != report.n_factors:
        lines.append(f"  NOTE: Parallel analysis suggests {report.parallel_n_factors}, "
                     f"not {report.n_factors}. This may indicate the domain structure "
                     f"doesn't cleanly separate into {report.n_factors} factors.")
    lines.append("")

    lines.append("--- Variance Explained ---")
    for i, v in enumerate(report.variance_explained):
        lines.append(f"  Factor {i+1}: {v:.4f} ({v*100:.1f}%)")
    lines.append(f"  Cumulative: {report.cumulative_variance:.4f} ({report.cumulative_variance*100:.1f}%)")
    lines.append("")

    lines.append("--- Factor Loadings (varimax rotation) ---")
    header = f"  {'Concept':15s} {'Domain':15s}"
    for i in range(report.n_factors):
        tc = [t for t in report.tucker_congruence if t["factor"] == i]
        domain_label = tc[0]["domain"] if tc else f"F{i+1}"
        header += f" {domain_label:>10s}"
    header += "  Primary"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for ca in report.concept_assignments:
        row = f"  {ca['concept']:15s} {ca['domain']:15s}"
        i = report.concepts.index(ca["concept"])
        for f_idx in range(report.n_factors):
            loading = report.loadings[i, f_idx]
            marker = "*" if f_idx == ca["primary_factor"] else " "
            row += f" {loading:>9.3f}{marker}"
        match_marker = "OK" if ca["match"] else "MISS"
        row += f"  {match_marker}"
        lines.append(row)
    lines.append("")

    lines.append("--- Domain Recovery ---")
    lines.append(f"Recovery rate: {report.domain_recovery_rate:.2%} "
                 f"({sum(1 for ca in report.concept_assignments if ca['match'])}/{report.n_concepts})")
    misses = [ca for ca in report.concept_assignments if not ca["match"]]
    if misses:
        lines.append("Misassigned concepts:")
        for ca in misses:
            lines.append(f"  {ca['concept']:15s} domain={ca['domain']}, "
                        f"loaded on factor for {ca['predicted_domain']}")
    else:
        lines.append("All concepts loaded on their expected domain factor.")
    lines.append("")

    lines.append("--- Tucker's Congruence Coefficient ---")
    for tc in report.tucker_congruence:
        interp = ("excellent" if abs(tc["congruence"]) >= 0.95 else
                  "good" if abs(tc["congruence"]) >= 0.85 else
                  "fair" if abs(tc["congruence"]) >= 0.65 else "poor")
        lines.append(f"  Factor {tc['factor']+1} ({tc['domain']:15s}): "
                     f"{tc['congruence']:.4f} ({interp})")
    lines.append("")

    lines.append("--- Interpretation Guide ---")
    lines.append("KMO: >0.8 good, >0.6 adequate, <0.5 unacceptable for factor analysis.")
    lines.append("Bartlett: p < 0.05 means the correlation matrix is not identity (good).")
    lines.append("Tucker's congruence: >0.95 excellent, >0.85 good, >0.65 fair match to target.")
    lines.append("Recovery rate: fraction of concepts whose highest loading matches their domain.")
    lines.append("Parallel analysis: data-driven estimate of number of meaningful factors.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_loading_heatmap(report: FactorReport, output_path: str):
    """Save a heatmap of factor loadings colored by domain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loadings = report.loadings
    n_concepts = report.n_concepts
    n_factors = report.n_factors
    concepts = report.concepts
    domains = report.domains

    domain_colors = {"physical": "#2196F3", "institutional": "#FF9800", "moral": "#E91E63"}

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    im = ax.imshow(loadings, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n_factors))
    factor_labels = []
    for tc in report.tucker_congruence:
        factor_labels.append(f"F{tc['factor']+1}\n({tc['domain'][:4]})")
    ax.set_xticklabels(factor_labels)

    ax.set_yticks(range(n_concepts))
    y_labels = [f"{c} [{d[:4]}]" for c, d in zip(concepts, domains)]
    ax.set_yticklabels(y_labels, fontsize=8)

    for i, (concept, domain) in enumerate(zip(concepts, domains)):
        color = domain_colors.get(domain, "#333333")
        ax.get_yticklabels()[i].set_color(color)

    for i in range(n_concepts):
        for j in range(n_factors):
            val = loadings[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title(f"Factor Loadings: {report.model_name}")
    fig.colorbar(im, ax=ax, label="Loading", shrink=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_scree_plot(report: FactorReport, output_path: str):
    """Save a scree plot with parallel analysis overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(len(report.eigenvalues), 10)
    x = range(1, n_show + 1)
    actual = report.eigenvalues[:n_show]
    parallel = report.parallel_eigenvalues[:n_show]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(x, actual, "bo-", label="Actual eigenvalues", markersize=8)
    ax.plot(x, parallel, "r--s", label="95th percentile (random)", markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Kaiser criterion")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Scree Plot with Parallel Analysis: {report.model_name}")
    ax.legend()
    ax.set_xticks(list(x))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Factor analysis construct validation for RCP concept inventories."
    )
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to Data directories containing JSONL files")
    parser.add_argument("--config", default=None,
                        help="Path to config.json (for concept/domain ordering)")
    parser.add_argument("--n-factors", type=int, default=3,
                        help="Number of factors to extract (default: 3)")
    parser.add_argument("--output", default=None,
                        help="Output directory for reports and plots")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    concepts = None
    domain_map = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        if "concepts" in config:
            domain_map = {}
            for domain, words in config["concepts"].items():
                for word in words:
                    domain_map[word] = domain
            concepts = sorted(domain_map.keys())

    for data_path in args.data:
        data_path = data_path.rstrip("/")
        dir_name = Path(data_path).name
        print(f"\n{'='*60}")
        print(f"Processing: {dir_name}")
        print(f"{'='*60}")

        try:
            ratings = load_ratings_from_jsonl(data_path)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  Loaded {len(ratings)} unframed ratings")
        if not ratings:
            print("  SKIP: no unframed ratings found")
            continue

        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        print(f"  Built {sim.matrix.shape[0]}x{sim.matrix.shape[1]} similarity matrix")

        report = run_factor_analysis(sim, n_factors=args.n_factors)
        text = format_report(report)

        if args.output:
            output_dir = Path(args.output) / dir_name
            output_dir.mkdir(parents=True, exist_ok=True)

            report_path = output_dir / "factor_validation_report.txt"
            with open(report_path, "w") as f:
                f.write(text)
            print(f"  Report: {report_path}")

            json_path = output_dir / "factor_validation.json"
            json_data = {
                "model_name": report.model_name,
                "n_concepts": report.n_concepts,
                "n_factors": report.n_factors,
                "kmo_score": report.kmo_score,
                "bartlett_chi2": report.bartlett_chi2,
                "bartlett_p": report.bartlett_p,
                "parallel_n_factors": report.parallel_n_factors,
                "domain_recovery_rate": report.domain_recovery_rate,
                "variance_explained": report.variance_explained,
                "cumulative_variance": report.cumulative_variance,
                "tucker_congruence": report.tucker_congruence,
                "concept_assignments": report.concept_assignments,
                "loadings": report.loadings.tolist(),
                "eigenvalues": report.eigenvalues,
                "parallel_eigenvalues": report.parallel_eigenvalues,
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2, cls=NumpyEncoder)

            if not args.no_plots:
                heatmap_path = output_dir / "loading_heatmap.png"
                save_loading_heatmap(report, str(heatmap_path))
                print(f"  Heatmap: {heatmap_path}")

                scree_path = output_dir / "scree_plot.png"
                save_scree_plot(report, str(scree_path))
                print(f"  Scree: {scree_path}")
        else:
            print()
            print(text)


if __name__ == "__main__":
    main()
