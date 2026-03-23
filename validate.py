#!/usr/bin/env python3
"""
RCP Validation Tests -- V1 through V7 from the experiment protocol.

Each test prints PASS/FAIL with supporting data.
Run after data collection to verify methodology before interpreting results.

Usage:
    python validate.py --data-dir data/                    # Run all tests
    python validate.py --data-dir data/ --test V1 V2       # Run specific tests
    python validate.py --symmetry-dir data/ --test V3      # Symmetry test only
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Import shared utilities from analyze.py
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
# Test infrastructure
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name, passed, message, data=None):
        self.name = name
        self.passed = passed
        self.message = message
        self.data = data or {}

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


# ---------------------------------------------------------------------------
# V1: Physical Domain Stability
# ---------------------------------------------------------------------------

def test_v1_physical_stability(matrices, config):
    """
    Physical domain sub-matrix drift from neutral should be below threshold
    for all models and framings.
    """
    threshold = config["validation"]["physical_drift_threshold"]
    concepts = get_all_concepts(config)
    control_domain = list(config["concepts"].keys())[0]
    failures = []
    all_drifts = []

    models = set(k[0] for k in matrices)
    for model in sorted(models):
        baseline_key = (model, "neutral", 0.0)
        if baseline_key not in matrices:
            # Try any temperature
            baseline_key = next(
                (k for k in matrices if k[0] == model and k[1] == "neutral"), None
            )
        if not baseline_key or baseline_key not in matrices:
            failures.append(f"{model}: no neutral baseline found")
            continue

        baseline_dist = similarity_to_distance(matrices[baseline_key]["matrix"])

        for key, mat_data in matrices.items():
            m, framing, temp = key
            if m != model or framing == "neutral":
                continue

            framed_dist = similarity_to_distance(mat_data["matrix"])
            domain_drift = compute_subdomain_drift(
                baseline_dist, framed_dist, concepts, config
            )
            phys_drift = domain_drift.get(control_domain, np.nan)
            all_drifts.append((model, framing, temp, phys_drift))

            if not np.isnan(phys_drift) and phys_drift > threshold:
                failures.append(
                    f"{model}/{framing} (t={temp}): physical drift = {phys_drift:.4f} > {threshold}"
                )

    if failures:
        msg = f"{len(failures)} condition(s) exceeded threshold {threshold}:\n"
        msg += "\n".join(f"  - {f}" for f in failures)
        return TestResult("V1: Physical Domain Stability", False, msg, {"drifts": all_drifts})

    mean_drift = np.nanmean([d[3] for d in all_drifts]) if all_drifts else 0
    return TestResult(
        "V1: Physical Domain Stability",
        True,
        f"All physical drifts below {threshold}. Mean = {mean_drift:.4f}",
        {"drifts": all_drifts},
    )


# ---------------------------------------------------------------------------
# V2: Known-Pair Ordering
# ---------------------------------------------------------------------------

def test_v2_known_pair_ordering(matrices, config):
    """
    Under neutral framing, specified ordering constraints must hold.
    """
    orderings = config["known_pair_orderings"]
    concepts = get_all_concepts(config)
    concept_idx = {c: i for i, c in enumerate(concepts)}

    failures = []
    models = set(k[0] for k in matrices)

    for model in sorted(models):
        baseline_key = (model, "neutral", 0.0)
        if baseline_key not in matrices:
            baseline_key = next(
                (k for k in matrices if k[0] == model and k[1] == "neutral"), None
            )
        if not baseline_key or baseline_key not in matrices:
            continue

        sim = matrices[baseline_key]["matrix"]

        for ordering in orderings:
            ca, cb = ordering["closer_pair"]
            fa, fb = ordering["farther_pair"]

            ia, ib = concept_idx[ca], concept_idx[cb]
            ifa, ifb = concept_idx[fa], concept_idx[fb]

            sim_close = sim[ia, ib]
            sim_far = sim[ifa, ifb]

            if np.isnan(sim_close) or np.isnan(sim_far):
                failures.append(f"{model}: missing data for {ordering['description']}")
            elif sim_close <= sim_far:
                failures.append(
                    f"{model}: {ordering['description']} -- "
                    f"sim({ca},{cb})={sim_close:.1f} <= sim({fa},{fb})={sim_far:.1f}"
                )

    if failures:
        msg = f"{len(failures)} ordering violation(s):\n"
        msg += "\n".join(f"  - {f}" for f in failures)
        return TestResult("V2: Known-Pair Ordering", False, msg)

    return TestResult(
        "V2: Known-Pair Ordering",
        True,
        f"All {len(orderings)} orderings hold across {len(models)} model(s)",
    )


# ---------------------------------------------------------------------------
# V3: Symmetry
# ---------------------------------------------------------------------------

def test_v3_symmetry(data_dir, config):
    """
    For symmetry validation pairs, sim(A,B) and sim(B,A) should match within tolerance.
    """
    records = load_records(data_dir, tag="symmetry")
    if not records:
        return TestResult(
            "V3: Symmetry",
            False,
            "No symmetry validation data found. Run: python collect.py --validation-only",
        )

    # Group by (model, concept_a, concept_b) -- note the symmetry run has both orderings
    pair_ratings = defaultdict(list)
    for r in records:
        if r.get("parsed") and r.get("rating") is not None:
            # Canonical key: sorted pair
            pair = tuple(sorted([r["concept_a"], r["concept_b"]]))
            order = "AB" if r["concept_a"] == pair[0] else "BA"
            pair_ratings[(r["model_name"], pair, order)].append(r["rating"])

    violations = 0
    total_pairs = 0
    max_diff = 0

    models = set(k[0] for k in pair_ratings)
    for model in sorted(models):
        model_pairs = set(k[1] for k in pair_ratings if k[0] == model)
        for pair in model_pairs:
            ab_ratings = pair_ratings.get((model, pair, "AB"), [])
            ba_ratings = pair_ratings.get((model, pair, "BA"), [])
            if ab_ratings and ba_ratings:
                mean_ab = np.mean(ab_ratings)
                mean_ba = np.mean(ba_ratings)
                diff = abs(mean_ab - mean_ba)
                max_diff = max(max_diff, diff)
                total_pairs += 1
                if diff > 1.0:
                    violations += 1

    if total_pairs == 0:
        return TestResult("V3: Symmetry", False, "No valid pair comparisons found")

    violation_rate = violations / total_pairs
    passed = violation_rate < 0.10  # Less than 10% of pairs violate symmetry

    return TestResult(
        "V3: Symmetry",
        passed,
        f"{violations}/{total_pairs} pairs differ by >1 point "
        f"({violation_rate:.1%}). Max diff = {max_diff:.1f}",
    )


# ---------------------------------------------------------------------------
# V4: Reproducibility at temp=0.0
# ---------------------------------------------------------------------------

def test_v4_reproducibility(records, config):
    """
    At temperature 0.0, standard deviation across reps should be 0 for 90%+ of probes.
    """
    threshold = config["validation"]["deterministic_zero_variance_threshold"]

    # Filter to temp=0.0
    det_records = [r for r in records if r.get("temperature") == 0.0
                   and r.get("parsed") and r.get("rating") is not None]

    if not det_records:
        return TestResult("V4: Reproducibility", False, "No deterministic (temp=0.0) data found")

    # Group by (model, concept_a, concept_b, framing)
    groups = defaultdict(list)
    for r in det_records:
        key = (r["model_name"], r["concept_a"], r["concept_b"], r["framing"])
        groups[key].append(r["rating"])

    zero_var_count = 0
    total_groups = 0
    nonzero_details = []

    for key, ratings in groups.items():
        if len(ratings) < 2:
            continue
        total_groups += 1
        sd = np.std(ratings)
        if sd == 0:
            zero_var_count += 1
        else:
            nonzero_details.append((key, ratings, sd))

    if total_groups == 0:
        return TestResult("V4: Reproducibility", False, "No multi-rep groups found")

    zero_rate = zero_var_count / total_groups
    passed = zero_rate >= threshold

    msg = (f"{zero_var_count}/{total_groups} groups have zero variance "
           f"({zero_rate:.1%}, threshold = {threshold:.0%})")
    if nonzero_details:
        msg += f"\n  {len(nonzero_details)} groups with nonzero variance"
        # Show worst 5
        nonzero_details.sort(key=lambda x: x[2], reverse=True)
        for (model, ca, cb, fr), ratings, sd in nonzero_details[:5]:
            msg += f"\n    {model}/{fr}: ({ca},{cb}) ratings={ratings} sd={sd:.2f}"

    return TestResult("V4: Reproducibility", passed, msg)


# ---------------------------------------------------------------------------
# V5: Framing Sensitivity (Moral domain drift is significant)
# ---------------------------------------------------------------------------

def test_v5_framing_sensitivity(matrices, config):
    """
    At least N models show statistically significant moral domain drift
    under at least one framing.
    """
    min_models = config["validation"]["min_models_with_moral_drift"]
    concepts = get_all_concepts(config)

    models_with_sig_drift = set()
    details = []

    models = sorted(set(k[0] for k in matrices))
    for model in models:
        baseline_key = (model, "neutral", 0.0)
        if baseline_key not in matrices:
            baseline_key = next(
                (k for k in matrices if k[0] == model and k[1] == "neutral"), None
            )
        if not baseline_key or baseline_key not in matrices:
            continue

        baseline_dist = similarity_to_distance(matrices[baseline_key]["matrix"])

        for key in matrices:
            m, framing, temp = key
            if m != model or framing == "neutral":
                continue

            framed_dist = similarity_to_distance(matrices[key]["matrix"])
            domain_drift = compute_subdomain_drift(
                baseline_dist, framed_dist, concepts, config
            )
            domain_keys = list(config["concepts"].keys())
            control_domain = domain_keys[0]
            target_domain = domain_keys[-1]
            target_drift = domain_drift.get(target_domain, 0)
            control_drift = domain_drift.get(control_domain, 0)

            # Simple significance test: target drift > 2x control drift
            # (A proper permutation test would go here with more data)
            if target_drift > 0 and (control_drift == 0 or target_drift / max(control_drift, 0.001) > 2.0):
                models_with_sig_drift.add(model)
                details.append(f"{model}/{framing}: {target_domain}={target_drift:.3f}, "
                               f"{control_domain}={control_drift:.3f}")

    n_sig = len(models_with_sig_drift)
    passed = n_sig >= min_models

    msg = f"{n_sig}/{len(models)} models show significant moral domain drift (need {min_models})"
    if details:
        msg += "\n" + "\n".join(f"  - {d}" for d in details[:10])

    return TestResult("V5: Framing Sensitivity", passed, msg)


# ---------------------------------------------------------------------------
# V6: Domain Ordering (physical < institutional < moral)
# ---------------------------------------------------------------------------

def test_v6_domain_ordering(matrices, config):
    """
    Mean drift should follow: physical < institutional < moral for N+ models.
    """
    min_models = config["validation"]["min_models_with_domain_ordering"]
    concepts = get_all_concepts(config)

    models_with_ordering = set()
    details = []

    models = sorted(set(k[0] for k in matrices))
    for model in models:
        baseline_key = (model, "neutral", 0.0)
        if baseline_key not in matrices:
            baseline_key = next(
                (k for k in matrices if k[0] == model and k[1] == "neutral"), None
            )
        if not baseline_key or baseline_key not in matrices:
            continue

        baseline_dist = similarity_to_distance(matrices[baseline_key]["matrix"])

        # Average drift across all framings for this model
        domain_drifts = defaultdict(list)
        for key in matrices:
            m, framing, temp = key
            if m != model or framing == "neutral":
                continue

            framed_dist = similarity_to_distance(matrices[key]["matrix"])
            dd = compute_subdomain_drift(baseline_dist, framed_dist, concepts, config)
            for domain, drift in dd.items():
                if not np.isnan(drift):
                    domain_drifts[domain].append(drift)

        # Domain ordering: control < intermediate < target (config key order)
        domain_keys = list(config["concepts"].keys())
        mean_drifts = [np.mean(domain_drifts.get(d, [0])) for d in domain_keys]
        ordering_holds = all(mean_drifts[i] < mean_drifts[i+1] for i in range(len(mean_drifts)-1))
        if ordering_holds:
            models_with_ordering.add(model)

        drift_str = " < ".join(f"{d[0].upper()}={v:.3f}" for d, v in zip(domain_keys, mean_drifts))
        details.append(f"{model}: {drift_str} {'(holds)' if ordering_holds else '(VIOLATED)'}")


    n_ordered = len(models_with_ordering)
    passed = n_ordered >= min_models

    msg = f"{n_ordered}/{len(models)} models show expected domain ordering (need {min_models})"
    msg += "\n" + "\n".join(f"  - {d}" for d in details)

    return TestResult("V6: Domain Ordering", passed, msg)


# ---------------------------------------------------------------------------
# V7: Parse Rate
# ---------------------------------------------------------------------------

def test_v7_parse_rate(records, config):
    """
    Parse success rate exceeds threshold for all model/framing combinations.
    """
    threshold = config["validation"]["parse_rate_threshold"]

    # Group records by (model, framing)
    groups = defaultdict(lambda: {"total": 0, "parsed": 0})
    for r in records:
        key = (r["model_name"], r["framing"])
        groups[key]["total"] += 1
        if r.get("parsed"):
            groups[key]["parsed"] += 1

    failures = []
    for (model, framing), counts in sorted(groups.items()):
        rate = counts["parsed"] / counts["total"] if counts["total"] > 0 else 0
        if rate < threshold:
            failures.append(
                f"{model}/{framing}: {counts['parsed']}/{counts['total']} = {rate:.1%}"
            )

    if failures:
        msg = f"{len(failures)} combination(s) below {threshold:.0%} parse rate:\n"
        msg += "\n".join(f"  - {f}" for f in failures)
        return TestResult("V7: Parse Rate", False, msg)

    total_parsed = sum(g["parsed"] for g in groups.values())
    total_records = sum(g["total"] for g in groups.values())
    overall_rate = total_parsed / total_records if total_records > 0 else 0

    # Refusal reporting (separate from parse rate pass/fail)
    refusal_groups = defaultdict(lambda: {"total": 0, "refusals": 0})
    for r in records:
        key = (r["model_name"], r["framing"],
               get_concept_domain(r.get("concept_a", ""), config) or "unknown",
               get_concept_domain(r.get("concept_b", ""), config) or "unknown")
        refusal_groups[key]["total"] += 1
        if r.get("is_refusal"):
            refusal_groups[key]["refusals"] += 1

    total_refusals = sum(g["refusals"] for g in refusal_groups.values())
    refusal_msg = ""
    if total_refusals > 0:
        refusal_msg = f"\n  Refusals detected: {total_refusals} total"
        # Summarize by model/framing
        by_model_framing = defaultdict(int)
        for (model, framing, da, db), counts in refusal_groups.items():
            if counts["refusals"] > 0:
                by_model_framing[(model, framing)] += counts["refusals"]
        for (model, framing), count in sorted(by_model_framing.items()):
            refusal_msg += f"\n    {model}/{framing}: {count} refusals"

    return TestResult(
        "V7: Parse Rate & Refusals",
        True,
        f"All combinations above {threshold:.0%}. Overall: {overall_rate:.1%} "
        f"({total_parsed}/{total_records}){refusal_msg}",
    )


# ---------------------------------------------------------------------------
# V8: Control Framing Discrimination (Compliance Check)
# ---------------------------------------------------------------------------

def test_v8_control_framing(matrices, config):
    """
    Nonsense framing drift should be less than 50% of mean cultural framing drift
    in the moral domain. Irrelevant preamble should produce less drift than nonsense.
    """
    concepts = get_all_concepts(config)
    # Cultural framings = all except neutral, irrelevant, nonsense
    cultural_framings = [
        f for f in config["framings"]
        if f not in ("neutral", "irrelevant", "nonsense")
    ]
    domain_keys = list(config["concepts"].keys())
    target_domain = domain_keys[-1]

    models = sorted(set(k[0] for k in matrices))
    models_passing = set()
    details = []

    for model in models:
        baseline_key = (model, "neutral", 0.0)
        if baseline_key not in matrices:
            baseline_key = next(
                (k for k in matrices if k[0] == model and k[1] == "neutral"), None
            )
        if not baseline_key or baseline_key not in matrices:
            continue

        baseline_dist = similarity_to_distance(matrices[baseline_key]["matrix"])

        # Mean cultural drift in target domain
        cultural_target_drifts = []
        for framing in cultural_framings:
            key = (model, framing, 0.0)
            if key not in matrices:
                continue
            framed_dist = similarity_to_distance(matrices[key]["matrix"])
            dd = compute_subdomain_drift(baseline_dist, framed_dist, concepts, config)
            target_drift = dd.get(target_domain, 0)
            if not np.isnan(target_drift):
                cultural_target_drifts.append(target_drift)

        if not cultural_target_drifts:
            continue
        mean_cultural_target = np.mean(cultural_target_drifts)

        # Nonsense drift
        nonsense_target = 0
        nonsense_key = (model, "nonsense", 0.0)
        if nonsense_key in matrices:
            framed_dist = similarity_to_distance(matrices[nonsense_key]["matrix"])
            dd = compute_subdomain_drift(baseline_dist, framed_dist, concepts, config)
            nonsense_target = dd.get(target_domain, 0)

        # Irrelevant drift
        irrelevant_target = 0
        irrelevant_key = (model, "irrelevant", 0.0)
        if irrelevant_key in matrices:
            framed_dist = similarity_to_distance(matrices[irrelevant_key]["matrix"])
            dd = compute_subdomain_drift(baseline_dist, framed_dist, concepts, config)
            irrelevant_target = dd.get(target_domain, 0)

        # Check: nonsense < 50% of cultural mean
        nonsense_ratio = nonsense_target / mean_cultural_target if mean_cultural_target > 0 else 0
        # Check: irrelevant < nonsense
        irrelevant_lt_nonsense = irrelevant_target <= nonsense_target

        model_passes = nonsense_ratio < 0.5 and irrelevant_lt_nonsense
        if model_passes:
            models_passing.add(model)

        details.append(
            f"{model}: cultural_mean={mean_cultural_target:.3f}, "
            f"nonsense={nonsense_target:.3f} ({nonsense_ratio:.0%}), "
            f"irrelevant={irrelevant_target:.3f} "
            f"{'(PASS)' if model_passes else '(FAIL)'}"
        )

    n_passing = len(models_passing)
    min_required = config["validation"].get("min_models_with_moral_drift", 3)
    passed = n_passing >= min_required

    msg = (f"{n_passing}/{len(models)} models pass compliance check "
           f"(need {min_required})")
    if details:
        msg += "\n" + "\n".join(f"  - {d}" for d in details)

    if not passed and n_passing < len(models):
        msg += ("\n  Note: If nonsense drift matches cultural drift, the experiment "
                "measures instruction compliance, not cultural reasoning.")

    return TestResult("V8: Control Framing Discrimination", passed, msg)


# ---------------------------------------------------------------------------
# V9: Framing Manipulation Check
# ---------------------------------------------------------------------------

FRAMING_KEYWORDS = {
    "individualist": ["autonomy", "individual", "rights", "self-determination", "personal"],
    "collectivist": ["harmony", "community", "collective", "group", "obligation", "mutual"],
    "hierarchical": ["rank", "station", "authority", "tradition", "role", "duty", "order"],
    "egalitarian": ["equal", "participation", "distributed", "reject", "no person"],
    "irrelevant": ["weather", "temperature", "warm", "climate"],
    "nonsense": ["triangle", "circle", "angular", "geometric", "shape"],
}


def test_v9_manipulation_check(data_dir, config):
    """
    Verify that models articulate the cultural frame they were given.
    Each response must contain at least 2 keywords from the expected framing.
    """
    data_dir = Path(data_dir)
    mc_file = data_dir / "manipulation_check.jsonl"

    if not mc_file.exists():
        return TestResult(
            "V9: Framing Manipulation Check",
            False,
            "No manipulation check data found. Run: python collect.py --manipulation-check",
        )

    records = []
    with open(mc_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return TestResult("V9: Framing Manipulation Check", False, "Empty manipulation check file")

    failures = []
    passes = 0
    total = 0

    for r in records:
        framing = r.get("framing", "")
        raw = (r.get("raw_response") or "").lower()
        keywords = FRAMING_KEYWORDS.get(framing, [])

        if not keywords:
            continue

        total += 1
        matches = sum(1 for kw in keywords if kw in raw)

        if matches >= 2:
            passes += 1
        else:
            model = r.get("model_name", "unknown")
            failures.append(f"{model}/{framing}: {matches}/{len(keywords)} keywords matched")

    if total == 0:
        return TestResult("V9: Framing Manipulation Check", False, "No valid records to check")

    pass_rate = passes / total
    passed = pass_rate >= 0.75  # At least 75% of model/framing combos must adopt the frame

    msg = f"{passes}/{total} model/framing combinations adopted the frame ({pass_rate:.0%})"
    if failures:
        msg += "\n  Frame not adopted:\n"
        msg += "\n".join(f"    - {f}" for f in failures)
        msg += "\n  Exclude or flag these combinations in drift interpretation."

    return TestResult("V9: Framing Manipulation Check", passed, msg)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "V1": "Physical Domain Stability",
    "V2": "Known-Pair Ordering",
    "V3": "Symmetry",
    "V4": "Reproducibility",
    "V5": "Framing Sensitivity",
    "V6": "Domain Ordering",
    "V7": "Parse Rate & Refusals",
    "V8": "Control Framing Discrimination",
    "V9": "Framing Manipulation Check",
}


def run_tests(args):
    config = load_config(args.config)
    tests_to_run = args.test or list(ALL_TESTS.keys())

    # Load main data
    records = []
    matrices = {}
    if any(t in tests_to_run for t in ["V1", "V2", "V4", "V5", "V6", "V7", "V8"]):
        records = load_records(args.data_dir, tag="main")
        if records:
            matrices = build_similarity_matrices(records, config)
            print(f"Loaded {len(records)} records, {len(matrices)} matrices")
        else:
            print(f"Warning: No main data found in {args.data_dir}")

    results = []

    for test_id in tests_to_run:
        if test_id not in ALL_TESTS:
            print(f"Unknown test: {test_id}. Available: {list(ALL_TESTS.keys())}")
            continue

        print(f"\nRunning {test_id}: {ALL_TESTS[test_id]}...")

        if test_id == "V1":
            result = test_v1_physical_stability(matrices, config) if matrices else \
                TestResult("V1", False, "No data")
        elif test_id == "V2":
            result = test_v2_known_pair_ordering(matrices, config) if matrices else \
                TestResult("V2", False, "No data")
        elif test_id == "V3":
            data_dir = args.symmetry_dir or args.data_dir
            result = test_v3_symmetry(data_dir, config)
        elif test_id == "V4":
            result = test_v4_reproducibility(records, config) if records else \
                TestResult("V4", False, "No data")
        elif test_id == "V5":
            result = test_v5_framing_sensitivity(matrices, config) if matrices else \
                TestResult("V5", False, "No data")
        elif test_id == "V6":
            result = test_v6_domain_ordering(matrices, config) if matrices else \
                TestResult("V6", False, "No data")
        elif test_id == "V7":
            result = test_v7_parse_rate(records, config) if records else \
                TestResult("V7", False, "No data")
        elif test_id == "V8":
            result = test_v8_control_framing(matrices, config) if matrices else \
                TestResult("V8", False, "No data")
        elif test_id == "V9":
            result = test_v9_manipulation_check(args.data_dir, config)

        results.append(result)
        print(result)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}")
    print(f"\n{passed}/{total} tests passed")

    if passed < total:
        print("\nFailing tests must be resolved before interpreting results.")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="RCP Validation Tests")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--symmetry-dir", default=None,
                        help="Directory for symmetry data (default: same as data-dir)")
    parser.add_argument("--test", nargs="+", default=None,
                        help="Specific tests to run (e.g., V1 V2 V5)")

    args = parser.parse_args()
    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
