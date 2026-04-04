#!/usr/bin/env python3
"""
Unit tests for analyze.py functions.
Run: python -m pytest test_analyze.py -v
"""

import json
import numpy as np
import pytest
from collections import defaultdict

from analyze import (
    load_config,
    build_similarity_matrices,
    similarity_to_distance,
    compute_mds,
    compute_mds_multidim,
    compute_procrustes_drift,
    compute_rank_correlation,
    compute_silhouette,
    compute_subdomain_drift,
    decompose_drift,
    detect_target_flattening,
    compute_tie_density,
    get_all_concepts,
    get_concept_domain,
)


@pytest.fixture
def config():
    return load_config("config.json")


# ---- Helper to build fake records ----

def make_records(config, model="test-model", framing="neutral", temp=0.0,
                 rating_func=None, reps=1):
    """Build synthetic JSONL records with a configurable rating function."""
    import itertools
    concepts = sorted(sum(config["concepts"].values(), []))
    concept_idx = {c: i for i, c in enumerate(concepts)}
    pairs = list(itertools.combinations(concepts, 2))
    records = []
    for a, b in pairs:
        if rating_func:
            da = get_concept_domain(a, config)
            db = get_concept_domain(b, config)
            rating = rating_func(a, b, da, db)
        else:
            rating = 4  # default: everything is 4
        for rep in range(1, reps + 1):
            records.append({
                "model": model, "model_name": model,
                "concept_a": a, "concept_b": b,
                "domain_a": get_concept_domain(a, config),
                "domain_b": get_concept_domain(b, config),
                "framing": framing, "temperature": temp,
                "rep": rep, "rating": rating, "parsed": True,
                "raw_response": str(rating), "error": None,
            })
    return records


# ---- similarity_to_distance ----

class TestSimilarityToDistance:
    def test_max_similarity_min_distance(self):
        sim = np.array([[7, 7], [7, 7]])
        dist = similarity_to_distance(sim)
        assert np.allclose(dist, [[1, 1], [1, 1]])

    def test_min_similarity_max_distance(self):
        sim = np.array([[7, 1], [1, 7]])
        dist = similarity_to_distance(sim)
        assert dist[0, 1] == 7.0
        assert dist[1, 0] == 7.0
        assert dist[0, 0] == 1.0  # 8 - 7

    def test_inverse_relationship(self):
        sim = np.array([[7, 5, 3], [5, 7, 2], [3, 2, 7]])
        dist = similarity_to_distance(sim)
        # Higher similarity -> lower distance
        assert dist[0, 1] < dist[0, 2]  # 5 > 3, so dist should be less

    def test_symmetry_preserved(self):
        sim = np.array([[7, 4, 2], [4, 7, 5], [2, 5, 7]])
        dist = similarity_to_distance(sim)
        np.testing.assert_array_equal(dist, dist.T)


# ---- build_similarity_matrices ----

class TestBuildSimilarityMatrices:
    def test_returns_matrix(self, config):
        records = make_records(config, rating_func=lambda a, b, da, db: 4)
        matrices = build_similarity_matrices(records, config)
        key = ("test-model", "neutral", 0.0)
        assert key in matrices
        mat = matrices[key]["matrix"]
        assert mat.shape == (18, 18)

    def test_diagonal_is_seven(self, config):
        records = make_records(config, rating_func=lambda a, b, da, db: 3)
        matrices = build_similarity_matrices(records, config)
        mat = matrices[("test-model", "neutral", 0.0)]["matrix"]
        np.testing.assert_array_equal(np.diag(mat), np.full(18, 7.0))

    def test_symmetric(self, config):
        records = make_records(config, rating_func=lambda a, b, da, db: 5)
        matrices = build_similarity_matrices(records, config)
        mat = matrices[("test-model", "neutral", 0.0)]["matrix"]
        np.testing.assert_array_equal(mat, mat.T)

    def test_rating_values_correct(self, config):
        # Within-domain = 6, cross-domain = 2
        def rate(a, b, da, db):
            return 6 if da == db else 2
        records = make_records(config, rating_func=rate)
        matrices = build_similarity_matrices(records, config)
        mat = matrices[("test-model", "neutral", 0.0)]["matrix"]
        concepts = get_all_concepts(config)
        for i in range(18):
            for j in range(i + 1, 18):
                di = get_concept_domain(concepts[i], config)
                dj = get_concept_domain(concepts[j], config)
                expected = 6.0 if di == dj else 2.0
                assert mat[i, j] == expected, f"{concepts[i]}-{concepts[j]}: {mat[i,j]} != {expected}"

    def test_parse_rate(self, config):
        records = make_records(config)
        matrices = build_similarity_matrices(records, config)
        assert matrices[("test-model", "neutral", 0.0)]["parse_rate"] == 1.0

    def test_multiple_reps_averaged(self, config):
        # rep 1 = 4, rep 2 = 6 -> mean should be 5
        records = make_records(config, rating_func=lambda a, b, da, db: 4, reps=1)
        for r in records:
            r["rep"] = 1
        records2 = make_records(config, rating_func=lambda a, b, da, db: 6, reps=1)
        for r in records2:
            r["rep"] = 2
        all_records = records + records2
        matrices = build_similarity_matrices(all_records, config)
        mat = matrices[("test-model", "neutral", 0.0)]["matrix"]
        # Off-diagonal should average to 5
        for i in range(18):
            for j in range(i + 1, 18):
                assert mat[i, j] == 5.0


# ---- compute_mds ----

class TestComputeMDS:
    def test_output_shape(self):
        dist = np.array([[0, 3, 5], [3, 0, 4], [5, 4, 0]], dtype=float)
        emb, stress = compute_mds(dist, n_components=2)
        assert emb.shape == (3, 2)

    def test_stress_is_number(self):
        dist = np.array([[0, 3, 5], [3, 0, 4], [5, 4, 0]], dtype=float)
        _, stress = compute_mds(dist)
        assert isinstance(stress, float)
        assert stress >= 0

    def test_deterministic(self):
        dist = np.array([[0, 3, 5], [3, 0, 4], [5, 4, 0]], dtype=float)
        e1, s1 = compute_mds(dist, random_state=42)
        e2, s2 = compute_mds(dist, random_state=42)
        np.testing.assert_array_almost_equal(e1, e2)

    def test_handles_nan(self):
        dist = np.array([[0, 3, np.nan], [3, 0, 4], [np.nan, 4, 0]], dtype=float)
        emb, stress = compute_mds(dist)
        assert emb.shape == (3, 2)
        assert not np.any(np.isnan(emb))


class TestComputeMDSMultidim:
    def test_returns_all_dims(self):
        dist = np.random.RandomState(42).rand(10, 10)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        results = compute_mds_multidim(dist, dims=(2, 3, 5))
        assert set(results.keys()) == {2, 3, 5}

    def test_correct_shapes(self):
        n = 10
        dist = np.random.RandomState(42).rand(n, n)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        results = compute_mds_multidim(dist)
        for d, (emb, stress) in results.items():
            assert emb.shape == (n, d)


# ---- compute_procrustes_drift ----

class TestComputeProcrustesDrift:
    def test_identical_zero_drift(self):
        emb = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        _, _, disp = compute_procrustes_drift(emb, emb.copy())
        assert disp < 1e-10

    def test_translated_low_drift(self):
        base = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        shifted = base + 5.0  # pure translation
        _, _, disp = compute_procrustes_drift(base, shifted)
        assert disp < 1e-10  # Procrustes removes translation

    def test_distorted_nonzero_drift(self):
        base = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        distorted = np.array([[0, 0], [2, 0], [0, 0.5]], dtype=float)
        _, _, disp = compute_procrustes_drift(base, distorted)
        assert disp > 0


# ---- compute_rank_correlation ----

class TestComputeRankCorrelation:
    def test_identical_perfect_correlation(self):
        d = np.array([[0, 3, 5], [3, 0, 4], [5, 4, 0]], dtype=float)
        rho, p = compute_rank_correlation(d, d)
        assert abs(rho - 1.0) < 1e-10

    def test_opposite_negative_correlation(self):
        d1 = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
        d2 = np.array([[0, 3, 2], [3, 0, 1], [2, 1, 0]], dtype=float)
        rho, _ = compute_rank_correlation(d1, d2)
        assert rho < 0

    def test_nan_handled(self):
        # Needs enough non-NaN pairs for Spearman to work
        n = 6
        d1 = np.random.RandomState(42).rand(n, n) * 5
        d1 = (d1 + d1.T) / 2
        np.fill_diagonal(d1, 0)
        d1[0, 3] = np.nan
        d1[3, 0] = np.nan
        d2 = np.random.RandomState(99).rand(n, n) * 5
        d2 = (d2 + d2.T) / 2
        np.fill_diagonal(d2, 0)
        rho, p = compute_rank_correlation(d1, d2)
        assert not np.isnan(rho)


# ---- compute_subdomain_drift ----

class TestComputeSubdomainDrift:
    def test_no_change_zero_drift(self, config):
        concepts = get_all_concepts(config)
        n = len(concepts)
        d = np.random.RandomState(42).rand(n, n) * 5
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0)
        drift = compute_subdomain_drift(d, d, concepts, config)
        for domain in ["physical", "institutional", "moral"]:
            assert drift[domain] == 0.0

    def test_moral_only_change(self, config):
        concepts = get_all_concepts(config)
        n = len(concepts)
        d1 = np.ones((n, n)) * 3
        np.fill_diagonal(d1, 0)
        d2 = d1.copy()
        # Shift only moral pairs
        moral_idx = [i for i, c in enumerate(concepts) if get_concept_domain(c, config) == "moral"]
        for i in moral_idx:
            for j in moral_idx:
                if i != j:
                    d2[i, j] += 1.0
        drift = compute_subdomain_drift(d1, d2, concepts, config)
        assert drift["physical"] == 0.0
        assert drift["institutional"] == 0.0
        assert drift["moral"] > 0.0


# ---- decompose_drift ----

class TestDecomposeDrift:
    def test_identical_low_residual(self, config):
        concepts = get_all_concepts(config)
        n = len(concepts)
        emb = np.random.RandomState(42).rand(n, 2)
        result = decompose_drift(emb, emb.copy(), concepts, config)
        assert result["disparity"] < 1e-10

    def test_has_required_keys(self, config):
        concepts = get_all_concepts(config)
        n = len(concepts)
        e1 = np.random.RandomState(42).rand(n, 2)
        e2 = np.random.RandomState(99).rand(n, 2)
        result = decompose_drift(e1, e2, concepts, config)
        for key in ["disparity", "total_residual_ss", "systematic_ss",
                     "random_ss", "systematic_random_ratio", "domain_mean_residuals"]:
            assert key in result

    def test_systematic_plus_random_equals_total(self, config):
        concepts = get_all_concepts(config)
        n = len(concepts)
        e1 = np.random.RandomState(42).rand(n, 2)
        e2 = np.random.RandomState(99).rand(n, 2)
        result = decompose_drift(e1, e2, concepts, config)
        total = result["systematic_ss"] + result["random_ss"]
        np.testing.assert_almost_equal(result["total_residual_ss"], total, decimal=10)


# ---- detect_target_flattening ----

class TestDetectTargetFlattening:
    def test_no_flattening_when_variance_stable(self, config):
        # Normal variance in both neutral and framed
        def rate_varied(a, b, da, db):
            if da == "moral" and db == "moral":
                return hash((a, b)) % 5 + 2  # 2-6, varied
            return 4
        records_n = make_records(config, framing="neutral", rating_func=rate_varied)
        records_f = make_records(config, framing="individualist", rating_func=rate_varied)
        matrices = build_similarity_matrices(records_n + records_f, config)
        result = detect_target_flattening(matrices, config)
        for v in result.values():
            assert v["variance_ratio"] > 0.5

    def test_flattening_detected(self, config):
        # Neutral has varied moral ratings, framed has all 4s
        def rate_varied(a, b, da, db):
            if da == "moral" and db == "moral":
                return hash((a, b)) % 5 + 2
            return 4
        records_n = make_records(config, framing="neutral", rating_func=rate_varied)
        records_f = make_records(config, framing="individualist",
                                 rating_func=lambda a, b, da, db: 4)
        matrices = build_similarity_matrices(records_n + records_f, config)
        result = detect_target_flattening(matrices, config)
        key = ("test-model", "individualist")
        assert key in result
        assert result[key]["variance_ratio"] < 0.5


# ---- compute_tie_density ----

class TestComputeTieDensity:
    def test_all_same_rating_max_ties(self, config):
        records = make_records(config, rating_func=lambda a, b, da, db: 4)
        matrices = build_similarity_matrices(records, config)
        td = compute_tie_density(matrices, config)
        key = ("test-model", "neutral", 0.0)
        # All off-diagonal are 4, diagonal is 7, so 2 unique values in upper tri
        assert td[key]["full_tie_density"] > 0.9

    def test_varied_fewer_ties_than_uniform(self, config):
        # Uniform: all 4s
        records_uniform = make_records(config, model="uniform",
                                       rating_func=lambda a, b, da, db: 4)
        # Varied: cycling 1-7
        counter = {"n": 0}
        def rate_cycling(a, b, da, db):
            counter["n"] += 1
            return (counter["n"] % 7) + 1
        records_varied = make_records(config, model="varied", rating_func=rate_cycling)
        matrices = build_similarity_matrices(records_uniform + records_varied, config)
        td = compute_tie_density(matrices, config)
        uniform_td = td[("uniform", "neutral", 0.0)]["full_tie_density"]
        varied_td = td[("varied", "neutral", 0.0)]["full_tie_density"]
        assert varied_td < uniform_td

    def test_moral_tie_density_computed(self, config):
        records = make_records(config, rating_func=lambda a, b, da, db: 4)
        matrices = build_similarity_matrices(records, config)
        td = compute_tie_density(matrices, config)
        key = ("test-model", "neutral", 0.0)
        assert "moral_tie_density" in td[key]
        assert "moral_pair_count" in td[key]
        assert td[key]["moral_pair_count"] == 15  # C(6,2)


# ---- compute_silhouette ----

class TestComputeSilhouette:
    def test_perfect_clusters_high_score(self, config):
        # Place physical, institutional, moral in tight clusters far apart
        concepts = get_all_concepts(config)
        n = len(concepts)
        emb = np.zeros((n, 2))
        for i, c in enumerate(concepts):
            d = get_concept_domain(c, config)
            if d == "physical":
                emb[i] = [0 + np.random.RandomState(i).rand() * 0.1, 0]
            elif d == "institutional":
                emb[i] = [5 + np.random.RandomState(i).rand() * 0.1, 0]
            else:
                emb[i] = [10 + np.random.RandomState(i).rand() * 0.1, 0]
        score = compute_silhouette(emb, config)
        assert score > 0.8

    def test_random_embedding_lower_score(self, config):
        emb = np.random.RandomState(42).rand(18, 2)
        score = compute_silhouette(emb, config)
        assert score < 0.8  # random clusters shouldn't score high
