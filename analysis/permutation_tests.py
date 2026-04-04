#!/usr/bin/env python3
"""
RCP Statistical Tests -- Pre-registered permutation tests and effect sizes.

Implements H1 (domain ordering), H2 (framing sensitivity), H3 (control
discrimination), and Cohen's d effect sizes as specified in the OSF
pre-registration (https://osf.io/cp4d3/overview).

Operates on similarity matrices already built by analyze.py.

Usage:
    python permutation_tests.py --data-dir runs/20260324-1-Sonnet-Moral-Data
    python permutation_tests.py --data-dir runs/20260324-1-Sonnet-Moral-Data --output-dir runs/20260324-1-Sonnet-Moral-Results
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from analyze import (
    load_config,
    load_records,
    build_similarity_matrices,
    similarity_to_distance,
    compute_subdomain_drift,
    get_all_concepts,
    get_concept_domain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress(i, n, label="permutations"):
    """Print progress every 10% for long runs. Silent for n <= 500."""
    if n <= 500:
        return
    step = max(1, n // 10)
    if i > 0 and i % step == 0:
        pct = (i * 100) // n
        print(f"    {label}: {pct}% ({i:,}/{n:,})", flush=True)


def _find_matrix_key(matrices, model, framing, config):
    """Find the best available matrix key for (model, framing).
    Prefers stochastic over deterministic because stochastic data
    averages multiple reps and is the paper's primary reporting condition.
    Reads stochastic temperature from config rather than hardcoding.
    """
    stochastic_temp = config.get("collection", {}).get("temp_stochastic", 0.7)
    deterministic_temp = config.get("collection", {}).get("temp_deterministic", 0.0)

    # Prefer stochastic
    key = (model, framing, stochastic_temp)
    if key in matrices:
        return key
    # Fall back to deterministic
    key = (model, framing, deterministic_temp)
    if key in matrices:
        return key
    # Fall back to whatever exists
    for k in matrices:
        if k[0] == model and k[1] == framing:
            return k
    return None


def _precompute_group_sums(mean_delta, n_concepts, group_size):
    """Precompute within-group delta sums for all possible groups.
    Returns dict mapping sorted tuple of indices to sum of within-group deltas.
    """
    triu = np.triu_indices(group_size, k=1)
    group_sums = {}
    for group in combinations(range(n_concepts), group_size):
        sub = mean_delta[np.ix_(group, group)]
        group_sums[group] = float(np.sum(sub[triu]))
    return group_sums


def _compute_mean_delta(neutral_dist, framed_dists):
    """Compute mean absolute distance difference across framings.
    Accepts a single distance matrix or a list of them.
    Returns the mean |neutral - framed| matrix.
    """
    if isinstance(framed_dists, np.ndarray) and framed_dists.ndim == 2:
        # Single framing
        return np.abs(neutral_dist - framed_dists)
    # Multiple framings: average the absolute deltas (not delta of averages)
    deltas = [np.abs(neutral_dist - fd) for fd in framed_dists]
    return np.nanmean(deltas, axis=0)


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------

def cohens_d(group_a, group_b):
    """
    Compute Cohen's d (absolute value) between two groups.
    Uses pooled standard deviation as denominator.
    Returns 0.0 if both groups have zero variance.
    """
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)

    na, nb = len(a), len(b)
    mean_diff = abs(np.mean(a) - np.mean(b))

    var_a = np.var(a, ddof=1) if na > 1 else 0.0
    var_b = np.var(b, ddof=1) if nb > 1 else 0.0

    if na + nb - 2 <= 0:
        return float("inf") if mean_diff > 0 else 0.0

    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    pooled_sd = np.sqrt(pooled_var)

    if pooled_sd == 0:
        return float("inf") if mean_diff > 0 else 0.0

    return float(mean_diff / pooled_sd)


# ---------------------------------------------------------------------------
# Control discrimination ratio (H3)
# ---------------------------------------------------------------------------

def control_discrimination_ratio(nonsense_target_drift, cultural_mean_target_drift):
    """
    H3: Ratio of nonsense drift to cultural mean drift in the target domain.
    Pre-registered decision boundary: ratio < 0.5 means the model discriminates.
    """
    if cultural_mean_target_drift == 0:
        return float("inf") if nonsense_target_drift > 0 else float("nan")
    return nonsense_target_drift / cultural_mean_target_drift


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------

def holm_bonferroni(results):
    """
    Apply Holm-Bonferroni correction to a list of results with 'p_value' keys.
    Adds 'p_corrected' and 'significant' (at alpha=0.05) to each result.
    Returns results in original order.
    """
    n = len(results)
    if n == 0:
        return results

    indexed = [(i, r) for i, r in enumerate(results)]
    indexed.sort(key=lambda x: x[1]["p_value"])

    corrected = [None] * n
    max_so_far = 0.0
    for rank, (orig_idx, r) in enumerate(indexed):
        multiplier = n - rank
        adjusted_p = min(r["p_value"] * multiplier, 1.0)
        adjusted_p = max(adjusted_p, max_so_far)
        max_so_far = adjusted_p

        result_copy = dict(r)
        result_copy["p_corrected"] = adjusted_p
        result_copy["significant"] = adjusted_p < 0.05
        corrected[orig_idx] = result_copy

    return corrected


# ---------------------------------------------------------------------------
# H1: Domain ordering permutation test (exact)
# ---------------------------------------------------------------------------

def permutation_test_domain_ordering(neutral_dist, framed_dists, concepts,
                                     config, test_ordering=None,
                                     n_permutations=None, seed=None):
    """
    Exact or Monte Carlo permutation test for domain ordering.

    Tests whether a specific domain ordering is significant. The pre-registered
    ordering is config domain key order (physical < institutional < moral).

    If n_permutations is None, runs exact test enumerating all C(18,6)*C(12,6)
    = 17,153,136 labeled partitions. Otherwise samples n_permutations random
    partitions.

    framed_dists: a single distance matrix or a list of them. If a list,
    drift is computed per framing and then averaged (matching the pre-registered
    procedure "mean drift across cultural framings").

    test_ordering: list of domain names in expected ascending-drift order.
    Defaults to config domain key order (the pre-registered prediction).

    Returns dict with p-values for both the pre-registered and observed orderings.
    """
    rng = np.random.default_rng(seed)

    domain_keys = list(config["concepts"].keys())
    n_domains = len(domain_keys)
    n_concepts = len(concepts)
    n_per_domain = n_concepts // n_domains

    if test_ordering is None:
        test_ordering = domain_keys  # pre-registered: physical < institutional < moral

    # Compute mean delta matrix (average |neutral - framed| across framings)
    mean_delta = _compute_mean_delta(neutral_dist, framed_dists)

    # Observed drift per domain using mean_delta
    concept_domains = [get_concept_domain(c, config) for c in concepts]
    domain_indices = {}
    for d in domain_keys:
        domain_indices[d] = [i for i, cd in enumerate(concept_domains) if cd == d]

    n_pairs = n_per_domain * (n_per_domain - 1) // 2
    triu = np.triu_indices(n_per_domain, k=1)
    observed_drifts = {}
    for d in domain_keys:
        idx = domain_indices[d]
        sub = mean_delta[np.ix_(idx, idx)]
        observed_drifts[d] = float(np.mean(sub[triu]))

    # Observed ordering
    observed_sorted = sorted(domain_keys, key=lambda d: observed_drifts[d])
    observed_ordering_str = " < ".join(observed_sorted)
    observed_strictly_ordered = all(
        observed_drifts[observed_sorted[i]] < observed_drifts[observed_sorted[i + 1]]
        for i in range(n_domains - 1)
    )

    # Pre-registered ordering
    preregistered_ordering_str = " < ".join(test_ordering)

    # Precompute within-group sums for all possible groups
    group_sums = _precompute_group_sums(mean_delta, n_concepts, n_per_domain)

    def _check_ordering(drift_by_position, ordering):
        """Check if drift values follow the given ordering strictly."""
        return all(
            drift_by_position[ordering[i]] < drift_by_position[ordering[i + 1]]
            for i in range(len(ordering) - 1)
        )

    # Count matches for both pre-registered and observed orderings
    preregistered_count = 0
    observed_count = 0
    total = 0
    all_indices = set(range(n_concepts))

    if n_permutations is None:
        # Exact enumeration: C(n,k) * C(n-k,k) labeled partitions
        from math import comb
        total_expected = comb(n_concepts, n_per_domain)
        for i in range(1, n_domains - 1):
            total_expected *= comb(n_concepts - i * n_per_domain, n_per_domain)

        print(f"    Exact test: {total_expected:,} partitions to enumerate")

        for group_a in combinations(range(n_concepts), n_per_domain):
            _progress(total, total_expected, "H1 exact enumeration")
            remaining_a = sorted(all_indices - set(group_a))
            sum_a = group_sums[group_a]

            for group_b in combinations(remaining_a, n_per_domain):
                group_c = tuple(sorted(set(remaining_a) - set(group_b)))
                sum_b = group_sums[group_b]
                sum_c = group_sums[group_c]

                # Map groups to domain labels (group order = domain_keys order)
                drifts = {
                    domain_keys[0]: sum_a / n_pairs,
                    domain_keys[1]: sum_b / n_pairs,
                    domain_keys[2]: sum_c / n_pairs,
                }

                if _check_ordering(drifts, test_ordering):
                    preregistered_count += 1
                if _check_ordering(drifts, observed_sorted):
                    observed_count += 1

                total += 1

    else:
        # Monte Carlo
        labels = np.array(concept_domains)
        for perm_i in range(n_permutations):
            _progress(perm_i, n_permutations, "H1 domain ordering")
            permuted = rng.permutation(labels)

            groups = {}
            for d in domain_keys:
                groups[d] = [i for i, lbl in enumerate(permuted) if lbl == d]

            drifts = {}
            for d in domain_keys:
                idx = groups[d]
                sub = mean_delta[np.ix_(idx, idx)]
                vals = sub[np.triu_indices(len(idx), k=1)]
                valid = vals[~np.isnan(vals)]
                drifts[d] = float(np.mean(valid)) if len(valid) > 0 else 0.0

            if _check_ordering(drifts, test_ordering):
                preregistered_count += 1
            if _check_ordering(drifts, observed_sorted):
                observed_count += 1

            total += 1

    # P-values (+1 includes the observed data itself)
    preregistered_p = (preregistered_count + 1) / (total + 1)
    if observed_strictly_ordered:
        observed_p = (observed_count + 1) / (total + 1)
    else:
        observed_p = 1.0  # No strict ordering to test

    return {
        "preregistered_ordering": preregistered_ordering_str,
        "preregistered_p_value": float(preregistered_p),
        "preregistered_count": preregistered_count,
        "observed_ordering": observed_ordering_str,
        "observed_p_value": float(observed_p),
        "observed_count": observed_count,
        "observed_strictly_ordered": observed_strictly_ordered,
        "observed_drifts": {d: float(v) for d, v in observed_drifts.items()},
        "total_permutations": total,
        "exact": n_permutations is None,
    }


# ---------------------------------------------------------------------------
# H2: Framing sensitivity permutation test (exact)
# ---------------------------------------------------------------------------

def permutation_test_framing_sensitivity(neutral_dist, framed_dist, concepts,
                                         config, n_permutations=None, seed=None):
    """
    Exact or Monte Carlo test for whether the target domain's drift exceeds
    what a random group of concepts would produce.

    If n_permutations is None, runs exact test: evaluates all C(18,6) = 18,564
    possible 6-concept groups. Otherwise samples n_permutations random groups.

    Null hypothesis: the target domain's drift is not special compared to
    any random group of the same size. (This is the only statistically
    meaningful interpretation of "permute concept labels within the target
    domain" from the pre-registration, since permuting labels within a
    fixed group produces an identity operation on the drift metric.)

    Returns dict with p_value, observed_target_drift, null_mean, null_std.
    """
    rng = np.random.default_rng(seed)

    domain_keys = list(config["concepts"].keys())
    target_domain = domain_keys[-1]

    delta = np.abs(neutral_dist - framed_dist)

    # Observed target drift
    concept_domains = [get_concept_domain(c, config) for c in concepts]
    target_indices = [i for i, cd in enumerate(concept_domains) if cd == target_domain]
    n_target = len(target_indices)
    n_total = len(concepts)

    triu = np.triu_indices(n_target, k=1)
    sub_obs = delta[np.ix_(target_indices, target_indices)]
    observed_target = float(np.mean(sub_obs[triu]))

    if n_permutations is None:
        # Exact: enumerate all C(n_total, n_target) possible groups
        group_sums = _precompute_group_sums(delta, n_total, n_target)
        n_pairs = n_target * (n_target - 1) // 2

        null_drifts = []
        for group, s in group_sums.items():
            null_drifts.append(s / n_pairs)

        null_drifts = np.array(null_drifts)
        total = len(null_drifts)
        print(f"    Exact test: {total:,} possible groups evaluated")
    else:
        # Monte Carlo
        null_drifts = []
        for perm_i in range(n_permutations):
            _progress(perm_i, n_permutations, "H2 framing sensitivity")
            fake_indices = rng.choice(n_total, size=n_target, replace=False).tolist()
            sub = delta[np.ix_(fake_indices, fake_indices)]
            vals = sub[triu]
            valid = vals[~np.isnan(vals)]
            null_drifts.append(float(np.mean(valid)) if len(valid) > 0 else 0.0)

        null_drifts = np.array(null_drifts)
        total = len(null_drifts)

    p_value = float((np.sum(null_drifts >= observed_target) + 1) / (total + 1))

    return {
        "p_value": float(p_value),
        "observed_target_drift": float(observed_target),
        "null_mean": float(np.mean(null_drifts)),
        "null_std": float(np.std(null_drifts)),
        "total_permutations": total,
        "exact": n_permutations is None,
    }


# ---------------------------------------------------------------------------
# Full test suite for one model
# ---------------------------------------------------------------------------

def run_all_statistical_tests(matrices, config, model,
                              n_perm_h1=None, n_perm_h2=None, seed=None):
    """
    Run all pre-registered statistical tests for a single model.

    Prefers stochastic data when available. Falls back to deterministic.

    n_perm_h1/h2: None for exact test, integer for Monte Carlo.
    Defaults to exact (None).

    Returns dict with h1_domain_ordering, h2_framing_sensitivity,
    h3_control_discrimination, effect_sizes, and metadata.
    """
    concepts = get_all_concepts(config)
    domain_keys = list(config["concepts"].keys())
    control_domain = domain_keys[0]
    target_domain = domain_keys[-1]

    cultural_framings = [
        f for f in config["framings"]
        if f not in ("neutral", "irrelevant", "nonsense")
    ]

    # Find baseline
    baseline_key = _find_matrix_key(matrices, model, "neutral", config)
    if not baseline_key:
        return {"error": f"No neutral baseline found for {model}"}

    baseline_temp = baseline_key[2]
    neutral_dist = similarity_to_distance(matrices[baseline_key]["matrix"])
    print(f"  Baseline temperature: {baseline_temp}")

    # --- H1: Domain ordering ---
    # Collect framed distance matrices and per-framing drifts
    cultural_framed_dists = []
    framing_temps = {}
    phys_drifts_per_framing = []
    moral_drifts_per_framing = []
    inst_drifts_per_framing = []

    for framing in cultural_framings:
        key = _find_matrix_key(matrices, model, framing, config)
        if not key:
            continue

        framed_dist = similarity_to_distance(matrices[key]["matrix"])
        cultural_framed_dists.append(framed_dist)
        framing_temps[framing] = key[2]

        # Per-framing drift for effect sizes
        dd = compute_subdomain_drift(neutral_dist, framed_dist, concepts, config)
        phys_drifts_per_framing.append(dd.get(control_domain, 0.0))
        moral_drifts_per_framing.append(dd.get(target_domain, 0.0))
        if len(domain_keys) > 2:
            intermediate = domain_keys[1]
            inst_drifts_per_framing.append(dd.get(intermediate, 0.0))

    if not cultural_framed_dists:
        return {"error": f"No cultural framing data found for {model}"}

    # Report temperatures
    unique_temps = set(framing_temps.values())
    temp_str = str(unique_temps.pop()) if len(unique_temps) == 1 else "mixed"
    print(f"  Cultural framing temperature: {temp_str}")

    # H1 test: pre-registered ordering is config domain key order
    h1_result = permutation_test_domain_ordering(
        neutral_dist, cultural_framed_dists, concepts, config,
        test_ordering=domain_keys,
        n_permutations=n_perm_h1, seed=seed,
    )

    # --- H2: Framing sensitivity (per framing, then Holm-Bonferroni) ---
    h2_per_framing = []
    for framing in cultural_framings:
        key = _find_matrix_key(matrices, model, framing, config)
        if not key:
            continue

        framed_dist = similarity_to_distance(matrices[key]["matrix"])
        framing_seed = seed + hash(framing) % 10000 if seed is not None else None

        result = permutation_test_framing_sensitivity(
            neutral_dist, framed_dist, concepts, config,
            n_permutations=n_perm_h2, seed=framing_seed,
        )
        result["framing"] = framing
        h2_per_framing.append(result)

    h2_corrected = holm_bonferroni(h2_per_framing)
    h2_any_significant = any(r["significant"] for r in h2_corrected)

    # --- H3: Control discrimination ---
    nonsense_key = _find_matrix_key(matrices, model, "nonsense", config)

    nonsense_target_drift = 0.0
    if nonsense_key:
        nonsense_dist = similarity_to_distance(matrices[nonsense_key]["matrix"])
        nonsense_drifts = compute_subdomain_drift(
            neutral_dist, nonsense_dist, concepts, config
        )
        nonsense_target_drift = nonsense_drifts.get(target_domain, 0.0)

    cultural_target_drifts = []
    for framing in cultural_framings:
        key = _find_matrix_key(matrices, model, framing, config)
        if key:
            framed_dist = similarity_to_distance(matrices[key]["matrix"])
            dd = compute_subdomain_drift(neutral_dist, framed_dist, concepts, config)
            cultural_target_drifts.append(dd.get(target_domain, 0.0))

    cultural_mean = float(np.mean(cultural_target_drifts)) if cultural_target_drifts else 0.0

    h3_ratio = control_discrimination_ratio(nonsense_target_drift, cultural_mean)
    h3_passes = h3_ratio < 0.5

    # --- Effect sizes (Cohen's d, n=4 per group, noted as low-n estimate) ---
    d_phys_moral = cohens_d(phys_drifts_per_framing, moral_drifts_per_framing)
    d_phys_inst = cohens_d(phys_drifts_per_framing, inst_drifts_per_framing) if inst_drifts_per_framing else None

    return {
        "model": model,
        "temperature_baseline": float(baseline_temp),
        "temperature_framings": framing_temps,
        "h1_domain_ordering": h1_result,
        "h2_framing_sensitivity": {
            "per_framing": h2_corrected,
            "any_significant_after_correction": h2_any_significant,
        },
        "h3_control_discrimination": {
            "nonsense_target_drift": float(nonsense_target_drift),
            "cultural_mean_target_drift": cultural_mean,
            "ratio": float(h3_ratio) if not np.isinf(h3_ratio) else "inf",
            "passes_threshold": bool(h3_passes),
        },
        "effect_sizes": {
            f"d_{control_domain}_vs_{target_domain}": float(d_phys_moral),
            f"d_{control_domain}_vs_{domain_keys[1]}": float(d_phys_inst) if d_phys_inst is not None else None,
            "note": f"Cohen's d computed from n={len(phys_drifts_per_framing)} per group (one per cultural framing). Low-n estimates; interpret magnitude cautiously.",
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RCP Statistical Tests (Pre-Registered)"
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-dir", required=True,
                        help="Data directory with JSONL files")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: companion Results dir)")
    parser.add_argument("--model", default=None,
                        help="Test single model (default: all in data)")
    parser.add_argument("--n-perm-h1", type=int, default=None,
                        help="Permutations for H1 (default: exact)")
    parser.add_argument("--n-perm-h2", type=int, default=None,
                        help="Permutations for H2 (default: exact)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Monte Carlo (ignored for exact)")

    args = parser.parse_args()
    config = load_config(args.config)

    # Load data
    records = load_records(args.data_dir, tag="main")
    if not records:
        print(f"No records found in {args.data_dir}/main_*.jsonl")
        sys.exit(1)

    matrices = build_similarity_matrices(records, config)
    print(f"Loaded {len(records)} records, {len(matrices)} matrices")

    # Determine models
    models = sorted(set(k[0] for k in matrices))
    if args.model:
        if args.model not in models:
            print(f"Model '{args.model}' not found. Available: {models}")
            sys.exit(1)
        models = [args.model]

    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.data_dir.replace("-Data", "-Results")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run tests
    all_results = {}
    for model in models:
        print(f"\n{'='*60}")
        print(f"  Statistical tests: {model}")
        print(f"{'='*60}")

        result = run_all_statistical_tests(
            matrices, config, model,
            n_perm_h1=args.n_perm_h1,
            n_perm_h2=args.n_perm_h2,
            seed=args.seed,
        )

        all_results[model] = result

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        # Report H1
        h1 = result["h1_domain_ordering"]
        print(f"\n  H1 Domain Ordering ({'exact' if h1['exact'] else 'Monte Carlo'}, {h1['total_permutations']:,} partitions):")
        print(f"    Observed: {h1['observed_ordering']}")
        for d, v in h1["observed_drifts"].items():
            print(f"      {d}: {v:.4f}")
        print(f"    Pre-registered ({h1['preregistered_ordering']}): p = {h1['preregistered_p_value']:.4f}")
        print(f"    Observed ({h1['observed_ordering']}): p = {h1['observed_p_value']:.4f}")

        # Report H2
        h2 = result["h2_framing_sensitivity"]
        print(f"\n  H2 Framing Sensitivity (Holm-Bonferroni corrected):")
        for r in h2["per_framing"]:
            exact_str = "exact" if r.get("exact") else "MC"
            sig = "***" if r["significant"] else ""
            print(f"    {r['framing']}: drift={r['observed_target_drift']:.4f}, "
                  f"p={r['p_value']:.4f}, p_corrected={r['p_corrected']:.4f} ({exact_str}) {sig}")
        print(f"    Any significant after correction: {h2['any_significant_after_correction']}")

        # Report H3
        h3 = result["h3_control_discrimination"]
        print(f"\n  H3 Control Discrimination:")
        print(f"    Nonsense target drift: {h3['nonsense_target_drift']:.4f}")
        print(f"    Cultural mean target drift: {h3['cultural_mean_target_drift']:.4f}")
        print(f"    Ratio: {h3['ratio']}")
        print(f"    {'PASSES' if h3['passes_threshold'] else 'FAILS'} 50% threshold")

        # Report effect sizes
        es = result["effect_sizes"]
        print(f"\n  Effect Sizes (Cohen's d, n={len(result.get('temperature_framings', {}))} per group):")
        for label, d in es.items():
            if label == "note":
                continue
            if d is not None:
                print(f"    {label}: {d:.3f}")

    # Save
    out_file = output_path / "statistical_tests.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
