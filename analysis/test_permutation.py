#!/usr/bin/env python3
"""
Unit tests for permutation_tests.py functions.
Run: python -m pytest test_permutation.py -v
"""

import numpy as np
import pytest
import itertools

from analyze import (
    load_config,
    build_similarity_matrices,
    similarity_to_distance,
    compute_subdomain_drift,
    get_all_concepts,
    get_concept_domain,
)
from permutation_tests import (
    permutation_test_domain_ordering,
    permutation_test_framing_sensitivity,
    cohens_d,
    control_discrimination_ratio,
    run_all_statistical_tests,
)


@pytest.fixture
def config():
    return load_config("config.json")


# ---- Helper: build synthetic records with controllable domain drift ----

def make_records_with_drift(config, model="test-model", framing="neutral",
                            temp=0.0, domain_offsets=None, reps=1):
    """
    Build synthetic records where within-domain similarity is 6 (close),
    cross-domain similarity is 2 (far), and domain_offsets shifts the
    framed condition's within-domain ratings.

    domain_offsets: dict mapping domain name to integer offset from neutral.
    e.g. {"physical": 0, "institutional": 2, "moral": 1} means
    institutional pairs shift by 2 from neutral, moral by 1.
    """
    concepts = sorted(sum(config["concepts"].values(), []))
    pairs = list(itertools.combinations(concepts, 2))
    records = []

    for a, b in pairs:
        da = get_concept_domain(a, config)
        db = get_concept_domain(b, config)

        if da == db:
            base_rating = 6
        else:
            base_rating = 2

        if framing != "neutral" and domain_offsets and da == db:
            offset = domain_offsets.get(da, 0)
            rating = max(1, min(7, base_rating - offset))
        else:
            rating = base_rating

        for rep in range(1, reps + 1):
            records.append({
                "model": model, "model_name": model,
                "concept_a": a, "concept_b": b,
                "domain_a": da, "domain_b": db,
                "framing": framing, "temperature": temp,
                "rep": rep, "rating": rating, "parsed": True,
                "raw_response": str(rating), "error": None,
            })
    return records


def build_test_matrices(config, model, domain_offsets, temp=0.0):
    """Build matrices for neutral + one framing with given domain offsets."""
    neutral_recs = make_records_with_drift(config, model=model,
                                           framing="neutral", temp=temp)
    framed_recs = make_records_with_drift(config, model=model,
                                          framing="collectivist", temp=temp,
                                          domain_offsets=domain_offsets)
    all_recs = neutral_recs + framed_recs
    return build_similarity_matrices(all_recs, config)


# =========================================================================
# Cohen's d tests
# =========================================================================

class TestCohensD:
    def test_zero_difference(self):
        a = [1.0, 1.0, 1.0, 1.0]
        b = [1.0, 1.0, 1.0, 1.0]
        assert cohens_d(a, b) == 0.0

    def test_known_effect_size(self):
        a = [0.0, 1.0, 0.0, 1.0]  # mean=0.5, sd=0.577
        b = [2.0, 3.0, 2.0, 3.0]  # mean=2.5, sd=0.577
        d = cohens_d(a, b)
        assert d > 3.0

    def test_moderate_effect(self):
        np.random.seed(42)
        a = np.random.normal(0, 1, 100).tolist()
        b = np.random.normal(0.5, 1, 100).tolist()
        d = cohens_d(a, b)
        assert 0.2 < d < 0.8

    def test_single_values(self):
        d = cohens_d([1.0], [2.0])
        assert isinstance(d, float)

    def test_returns_absolute_value(self):
        a = [3.0, 4.0, 3.0, 4.0]
        b = [1.0, 2.0, 1.0, 2.0]
        d = cohens_d(a, b)
        assert d > 0


# =========================================================================
# Control discrimination ratio tests
# =========================================================================

class TestControlDiscriminationRatio:
    def test_zero_nonsense_drift(self):
        ratio = control_discrimination_ratio(0.0, 1.0)
        assert ratio == 0.0

    def test_equal_drift(self):
        ratio = control_discrimination_ratio(1.0, 1.0)
        assert ratio == 1.0

    def test_half_drift(self):
        ratio = control_discrimination_ratio(0.5, 1.0)
        assert ratio == 0.5

    def test_zero_cultural_drift(self):
        ratio = control_discrimination_ratio(0.5, 0.0)
        assert np.isinf(ratio) or np.isnan(ratio)

    def test_passes_threshold(self):
        ratio = control_discrimination_ratio(0.3, 1.0)
        assert ratio < 0.5


# =========================================================================
# H1: Domain ordering permutation test
# =========================================================================

class TestPermutationDomainOrdering:
    def test_clear_ordering_observed_p(self, config):
        """When physical < moral < institutional clearly holds,
        observed_p_value should be low (MC, relaxed for 6-concept power)."""
        offsets = {"physical": 0, "institutional": 5, "moral": 2}
        matrices = build_test_matrices(config, "test-model", offsets)

        concepts = get_all_concepts(config)
        neutral_dist = similarity_to_distance(
            matrices[("test-model", "neutral", 0.0)]["matrix"]
        )
        framed_dist = similarity_to_distance(
            matrices[("test-model", "collectivist", 0.0)]["matrix"]
        )

        result = permutation_test_domain_ordering(
            neutral_dist, framed_dist, concepts, config,
            test_ordering=["physical", "moral", "institutional"],
            n_permutations=1000, seed=42,
        )

        assert "observed_p_value" in result
        assert "preregistered_p_value" in result
        assert "observed_ordering" in result
        assert "total_permutations" in result
        assert result["observed_p_value"] < 0.20  # relaxed: synthetic data, 6 concepts per domain

    def test_flat_drift_high_pvalue(self, config):
        """When all domains drift equally (zero offset), no strict ordering.
        observed_p_value should be 1.0."""
        offsets = {"physical": 0, "institutional": 0, "moral": 0}
        matrices = build_test_matrices(config, "test-model", offsets)

        concepts = get_all_concepts(config)
        neutral_dist = similarity_to_distance(
            matrices[("test-model", "neutral", 0.0)]["matrix"]
        )
        framed_dist = similarity_to_distance(
            matrices[("test-model", "collectivist", 0.0)]["matrix"]
        )

        result = permutation_test_domain_ordering(
            neutral_dist, framed_dist, concepts, config,
            n_permutations=100, seed=42,
        )

        assert result["observed_strictly_ordered"] is False
        assert result["observed_p_value"] == 1.0

    def test_preregistered_ordering_tested(self, config):
        """Pre-registered ordering should be tested even when observed differs."""
        offsets = {"physical": 0, "institutional": 5, "moral": 2}
        # Observed ordering: physical < moral < institutional
        # Pre-registered (default): physical < institutional < moral
        matrices = build_test_matrices(config, "test-model", offsets)

        concepts = get_all_concepts(config)
        neutral_dist = similarity_to_distance(
            matrices[("test-model", "neutral", 0.0)]["matrix"]
        )
        framed_dist = similarity_to_distance(
            matrices[("test-model", "collectivist", 0.0)]["matrix"]
        )

        result = permutation_test_domain_ordering(
            neutral_dist, framed_dist, concepts, config,
            # default test_ordering = config domain keys = pre-registered
            n_permutations=500, seed=42,
        )

        assert result["preregistered_ordering"] == "physical < institutional < moral"
        assert result["observed_ordering"] == "physical < moral < institutional"
        # Pre-registered ordering doesn't match observed, so its p-value
        # is the chance rate of the pre-registered ordering under the null
        assert isinstance(result["preregistered_p_value"], float)

    def test_returns_observed_drifts(self, config):
        """Result should include the actual drift values per domain."""
        offsets = {"physical": 0, "institutional": 5, "moral": 2}
        matrices = build_test_matrices(config, "test-model", offsets)

        concepts = get_all_concepts(config)
        neutral_dist = similarity_to_distance(
            matrices[("test-model", "neutral", 0.0)]["matrix"]
        )
        framed_dist = similarity_to_distance(
            matrices[("test-model", "collectivist", 0.0)]["matrix"]
        )

        result = permutation_test_domain_ordering(
            neutral_dist, framed_dist, concepts, config,
            n_permutations=100, seed=42,
        )

        assert "observed_drifts" in result
        for domain in config["concepts"]:
            assert domain in result["observed_drifts"]

    def test_respects_seed(self, config):
        """Same seed should produce same p-values."""
        offsets = {"physical": 0, "institutional": 2, "moral": 1}
        matrices = build_test_matrices(config, "test-model", offsets)

        concepts = get_all_concepts(config)
        neutral_dist = similarity_to_distance(
            matrices[("test-model", "neutral", 0.0)]["matrix"]
        )
        framed_dist = similarity_to_distance(
            matrices[("test-model", "collectivist", 0.0)]["matrix"]
        )

        r1 = permutation_test_domain_ordering(
            neutral_dist, framed_dist, concepts, config,
            n_permutations=500, seed=123,
        )
        r2 = permutation_test_domain_ordering(
            neutral_dist, framed_dist, concepts, config,
            n_permutations=500, seed=123,
        )

        assert r1["observed_p_value"] == r2["observed_p_value"]
        assert r1["preregistered_p_value"] == r2["preregistered_p_value"]

    def test_accepts_list_of_framed_dists(self, config):
        """Should accept a list of distance matrices (averaging across framings)."""
        offsets = {"physical": 0, "institutional": 5, "moral": 2}
        matrices = build_test_matrices(config, "test-model", offsets)

        concepts = get_all_concepts(config)
        neutral_dist = similarity_to_distance(
            matrices[("test-model", "neutral", 0.0)]["matrix"]
        )
        framed_dist = similarity_to_distance(
            matrices[("test-model", "collectivist", 0.0)]["matrix"]
        )

        # Pass as list of one (should work the same as single matrix)
        result = permutation_test_domain_ordering(
            neutral_dist, [framed_dist], concepts, config,
            n_permutations=100, seed=42,
        )
        assert "observed_p_value" in result
        assert "observed_drifts" in result


# =========================================================================
# H2: Framing sensitivity permutation test
# =========================================================================

class TestPermutationFramingSensitivity:
    def test_significant_drift_low_pvalue(self, config):
        """When moral domain clearly drifts, p-value should be low."""
        offsets = {"physical": 0, "institutional": 0, "moral": 3}
        matrices = build_test_matrices(config, "test-model", offsets)

        concepts = get_all_concepts(config)
        neutral_dist = similarity_to_distance(
            matrices[("test-model", "neutral", 0.0)]["matrix"]
        )
        framed_dist = similarity_to_distance(
            matrices[("test-model", "collectivist", 0.0)]["matrix"]
        )

        result = permutation_test_framing_sensitivity(
            neutral_dist, framed_dist, concepts, config,
            n_permutations=1000, seed=42,
        )

        assert "p_value" in result
        assert "observed_target_drift" in result
        assert result["p_value"] < 0.05

    def test_no_drift_high_pvalue(self, config):
        """When no domain drifts, p-value should be high."""
        offsets = {"physical": 0, "institutional": 0, "moral": 0}
        matrices = build_test_matrices(config, "test-model", offsets)

        concepts = get_all_concepts(config)
        neutral_dist = similarity_to_distance(
            matrices[("test-model", "neutral", 0.0)]["matrix"]
        )
        framed_dist = similarity_to_distance(
            matrices[("test-model", "collectivist", 0.0)]["matrix"]
        )

        result = permutation_test_framing_sensitivity(
            neutral_dist, framed_dist, concepts, config,
            n_permutations=1000, seed=42,
        )

        assert result["p_value"] > 0.05

    def test_returns_null_distribution_summary(self, config):
        """Result should include summary stats of null distribution."""
        offsets = {"physical": 0, "institutional": 0, "moral": 2}
        matrices = build_test_matrices(config, "test-model", offsets)

        concepts = get_all_concepts(config)
        neutral_dist = similarity_to_distance(
            matrices[("test-model", "neutral", 0.0)]["matrix"]
        )
        framed_dist = similarity_to_distance(
            matrices[("test-model", "collectivist", 0.0)]["matrix"]
        )

        result = permutation_test_framing_sensitivity(
            neutral_dist, framed_dist, concepts, config,
            n_permutations=500, seed=42,
        )

        assert "null_mean" in result
        assert "null_std" in result
        assert "total_permutations" in result

    def test_exact_matches_monte_carlo(self, config):
        """Exact test should give similar p-value to large MC sample."""
        offsets = {"physical": 0, "institutional": 0, "moral": 3}
        matrices = build_test_matrices(config, "test-model", offsets)

        concepts = get_all_concepts(config)
        neutral_dist = similarity_to_distance(
            matrices[("test-model", "neutral", 0.0)]["matrix"]
        )
        framed_dist = similarity_to_distance(
            matrices[("test-model", "collectivist", 0.0)]["matrix"]
        )

        exact = permutation_test_framing_sensitivity(
            neutral_dist, framed_dist, concepts, config,
            n_permutations=None,
        )
        mc = permutation_test_framing_sensitivity(
            neutral_dist, framed_dist, concepts, config,
            n_permutations=5000, seed=42,
        )

        assert exact["exact"] is True
        assert mc["exact"] is False
        # Exact total should be C(18,6) = 18564
        assert exact["total_permutations"] == 18564
        # P-values should be in the same ballpark
        assert abs(exact["p_value"] - mc["p_value"]) < 0.05


# =========================================================================
# Holm-Bonferroni correction
# =========================================================================

class TestHolmBonferroni:
    def test_single_pvalue_unchanged(self):
        from permutation_tests import holm_bonferroni
        results = [{"framing": "a", "p_value": 0.03}]
        corrected = holm_bonferroni(results)
        assert corrected[0]["p_corrected"] == 0.03
        assert corrected[0]["significant"] is True

    def test_correction_increases_pvalues(self):
        from permutation_tests import holm_bonferroni
        results = [
            {"framing": "a", "p_value": 0.01},
            {"framing": "b", "p_value": 0.03},
            {"framing": "c", "p_value": 0.04},
            {"framing": "d", "p_value": 0.06},
        ]
        corrected = holm_bonferroni(results)
        assert corrected[0]["p_corrected"] == pytest.approx(0.04)
        for c in corrected:
            assert c["p_corrected"] >= c["p_value"]

    def test_preserves_ordering(self):
        from permutation_tests import holm_bonferroni
        results = [
            {"framing": "a", "p_value": 0.06},
            {"framing": "b", "p_value": 0.01},
            {"framing": "c", "p_value": 0.03},
        ]
        corrected = holm_bonferroni(results)
        assert corrected[0]["framing"] == "a"
        assert corrected[1]["framing"] == "b"
        assert corrected[2]["framing"] == "c"

    def test_capped_at_one(self):
        from permutation_tests import holm_bonferroni
        results = [
            {"framing": "a", "p_value": 0.5},
            {"framing": "b", "p_value": 0.6},
        ]
        corrected = holm_bonferroni(results)
        for c in corrected:
            assert c["p_corrected"] <= 1.0


# =========================================================================
# Integration: run_all_statistical_tests
# =========================================================================

class TestRunAllStatisticalTests:
    def test_returns_all_sections(self, config):
        """Output should have h1, h2, h3, effect_sizes sections."""
        offsets = {"physical": 0, "institutional": 5, "moral": 2}
        neutral_recs = make_records_with_drift(config, model="test-model",
                                               framing="neutral", temp=0.0)
        cultural_recs = []
        for framing in ["individualist", "collectivist", "hierarchical", "egalitarian"]:
            cultural_recs += make_records_with_drift(
                config, model="test-model", framing=framing, temp=0.0,
                domain_offsets=offsets
            )
        nonsense_recs = make_records_with_drift(
            config, model="test-model", framing="nonsense", temp=0.0,
            domain_offsets={"physical": 0, "institutional": 1, "moral": 0}
        )

        all_recs = neutral_recs + cultural_recs + nonsense_recs
        matrices = build_similarity_matrices(all_recs, config)

        result = run_all_statistical_tests(
            matrices, config, model="test-model",
            n_perm_h1=100, n_perm_h2=100, seed=42,
        )

        assert "h1_domain_ordering" in result
        assert "h2_framing_sensitivity" in result
        assert "h3_control_discrimination" in result
        assert "effect_sizes" in result
        assert "temperature_baseline" in result
        assert "temperature_framings" in result
