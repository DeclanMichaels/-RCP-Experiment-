"""
Unit tests for factor_validate.py

Tests all pure analysis functions with synthetic data where expected
outputs can be computed by hand or verified analytically. Does NOT
require real JSONL data -- all tests use pre-built numpy arrays or
temporary files.

Run:  python -m pytest test_factor_validate.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from factor_validate import (
    PairRating,
    SimilarityMatrix,
    FactorReport,
    load_ratings_from_jsonl,
    build_similarity_matrix,
    nearest_positive_semidefinite,
    is_positive_semidefinite,
    run_parallel_analysis,
    run_factor_analysis,
    format_report,
    NumpyEncoder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_domain_ratings():
    """Synthetic ratings with clear 3-domain structure.

    Physical concepts rate high with each other (5-7), low with others (1-2).
    Same for institutional and moral. Cross-domain pairs get 1-2.
    """
    concepts = {
        "physical": ["gravity", "friction", "combustion",
                      "pressure", "erosion", "conduction"],
        "institutional": ["authority", "property", "contract",
                          "citizenship", "hierarchy", "obligation"],
        "moral": ["fairness", "honor", "harm",
                  "loyalty", "purity", "care"],
    }

    ratings = []
    all_concepts = []
    domain_map = {}
    for domain, words in concepts.items():
        for w in words:
            all_concepts.append(w)
            domain_map[w] = domain
    all_concepts.sort()

    for i, c1 in enumerate(all_concepts):
        for j, c2 in enumerate(all_concepts):
            if i >= j:
                continue
            d1 = domain_map[c1]
            d2 = domain_map[c2]
            if d1 == d2:
                rating = np.random.choice([5, 6, 7])
            else:
                rating = np.random.choice([1, 2])
            ratings.append(PairRating(
                concept_a=c1, concept_b=c2,
                domain_a=d1, domain_b=d2,
                rating=rating, model_name="synthetic",
            ))
    return ratings, all_concepts, domain_map


@pytest.fixture
def weak_structure_ratings():
    """Ratings with weak domain structure (uniform 3-5 everywhere)."""
    concepts = {
        "physical": ["gravity", "friction", "combustion"],
        "moral": ["fairness", "honor", "harm"],
    }
    ratings = []
    all_concepts = []
    domain_map = {}
    for domain, words in concepts.items():
        for w in words:
            all_concepts.append(w)
            domain_map[w] = domain
    all_concepts.sort()

    for i, c1 in enumerate(all_concepts):
        for j, c2 in enumerate(all_concepts):
            if i >= j:
                continue
            ratings.append(PairRating(
                concept_a=c1, concept_b=c2,
                domain_a=domain_map[c1], domain_b=domain_map[c2],
                rating=np.random.choice([3, 4, 5]),
                model_name="weak-synthetic",
            ))
    return ratings, all_concepts, domain_map


@pytest.fixture
def sample_jsonl_dir(tmp_path):
    """Create a temporary directory with a valid JSONL file."""
    records = [
        {"model": "test", "model_name": "test-model",
         "concept_a": "gravity", "concept_b": "friction",
         "domain_a": "physical", "domain_b": "physical",
         "framing": "neutral", "temperature": 0.0, "rep": 1,
         "rating": 6, "parsed": True, "is_refusal": False,
         "raw_response": "6", "timestamp": "2026-01-01T00:00:00Z",
         "latency_ms": 100, "error": None},
        {"model": "test", "model_name": "test-model",
         "concept_a": "gravity", "concept_b": "fairness",
         "domain_a": "physical", "domain_b": "moral",
         "framing": "neutral", "temperature": 0.0, "rep": 1,
         "rating": 2, "parsed": True, "is_refusal": False,
         "raw_response": "2", "timestamp": "2026-01-01T00:00:01Z",
         "latency_ms": 100, "error": None},
        {"model": "test", "model_name": "test-model",
         "concept_a": "friction", "concept_b": "fairness",
         "domain_a": "physical", "domain_b": "moral",
         "framing": "neutral", "temperature": 0.0, "rep": 1,
         "rating": 1, "parsed": True, "is_refusal": False,
         "raw_response": "1", "timestamp": "2026-01-01T00:00:02Z",
         "latency_ms": 100, "error": None},
        # Non-neutral framing (should be filtered)
        {"model": "test", "model_name": "test-model",
         "concept_a": "gravity", "concept_b": "friction",
         "domain_a": "physical", "domain_b": "physical",
         "framing": "individualist", "temperature": 0.0, "rep": 1,
         "rating": 7, "parsed": True, "is_refusal": False,
         "raw_response": "7", "timestamp": "2026-01-01T00:00:03Z",
         "latency_ms": 100, "error": None},
        # Unparsed (should be filtered)
        {"model": "test", "model_name": "test-model",
         "concept_a": "gravity", "concept_b": "friction",
         "domain_a": "physical", "domain_b": "physical",
         "framing": "neutral", "temperature": 0.0, "rep": 1,
         "rating": 0, "parsed": False, "is_refusal": False,
         "raw_response": "I cannot", "timestamp": "2026-01-01T00:00:04Z",
         "latency_ms": 100, "error": None},
    ]
    jsonl_path = tmp_path / "main_test-model.jsonl"
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Tests: load_ratings_from_jsonl
# ---------------------------------------------------------------------------

class TestLoadRatings:

    def test_loads_neutral_only(self, sample_jsonl_dir):
        ratings = load_ratings_from_jsonl(sample_jsonl_dir)
        assert len(ratings) == 3
        for r in ratings:
            assert isinstance(r, PairRating)

    def test_filters_unparsed(self, sample_jsonl_dir):
        ratings = load_ratings_from_jsonl(sample_jsonl_dir)
        assert all(r.rating > 0 for r in ratings)

    def test_accepts_unframed_label(self, tmp_path):
        record = {
            "model": "test", "model_name": "test-model",
            "concept_a": "gravity", "concept_b": "friction",
            "domain_a": "physical", "domain_b": "physical",
            "framing": "unframed", "temperature": 0.0, "rep": 1,
            "rating": 5, "parsed": True, "is_refusal": False,
            "raw_response": "5", "timestamp": "2026-01-01",
            "latency_ms": 100, "error": None,
        }
        jsonl_path = tmp_path / "main_test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(record) + "\n")
        ratings = load_ratings_from_jsonl(str(tmp_path))
        assert len(ratings) == 1

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No main_"):
            load_ratings_from_jsonl(str(tmp_path))

    def test_not_a_directory_raises(self, tmp_path):
        fake = tmp_path / "notadir.txt"
        fake.write_text("hello")
        with pytest.raises(ValueError, match="Not a directory"):
            load_ratings_from_jsonl(str(fake))

    def test_model_name_extracted(self, sample_jsonl_dir):
        ratings = load_ratings_from_jsonl(sample_jsonl_dir)
        assert ratings[0].model_name == "test-model"


# ---------------------------------------------------------------------------
# Tests: build_similarity_matrix
# ---------------------------------------------------------------------------

class TestBuildSimilarityMatrix:

    def test_matrix_shape(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        assert sim.matrix.shape == (18, 18)

    def test_diagonal_is_max(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        np.testing.assert_allclose(np.diag(sim.matrix), 7.0)

    def test_correlation_diagonal_is_one(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        np.testing.assert_allclose(np.diag(sim.correlation), 1.0)

    def test_symmetry(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        np.testing.assert_allclose(sim.matrix, sim.matrix.T)

    def test_within_domain_higher(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        n = len(concepts)
        within = []
        between = []
        for i in range(n):
            for j in range(i + 1, n):
                if sim.domains[i] == sim.domains[j]:
                    within.append(sim.matrix[i, j])
                else:
                    between.append(sim.matrix[i, j])
        assert np.mean(within) > np.mean(between)

    def test_averages_multiple_reps(self):
        ratings = [
            PairRating("a", "b", "d1", "d1", 4, "test"),
            PairRating("a", "b", "d1", "d1", 6, "test"),
        ]
        sim = build_similarity_matrix(ratings,
                                       concepts=["a", "b"],
                                       domain_map={"a": "d1", "b": "d1"})
        assert sim.matrix[0, 1] == 5.0

    def test_empty_ratings_raises(self):
        with pytest.raises(ValueError, match="No ratings"):
            build_similarity_matrix([])

    def test_derives_concepts_from_ratings(self, three_domain_ratings):
        ratings, _, _ = three_domain_ratings
        sim = build_similarity_matrix(ratings)
        assert len(sim.concepts) == 18


# ---------------------------------------------------------------------------
# Tests: nearest_positive_semidefinite
# ---------------------------------------------------------------------------

class TestNearestPSD:

    def test_psd_unchanged(self):
        m = np.eye(3)
        result = nearest_positive_semidefinite(m)
        np.testing.assert_allclose(result, m, atol=1e-6)

    def test_fixes_non_psd(self):
        m = np.array([[1.0, 0.9, 0.9],
                       [0.9, 1.0, -0.9],
                       [0.9, -0.9, 1.0]])
        assert not is_positive_semidefinite(m)
        result = nearest_positive_semidefinite(m)
        assert is_positive_semidefinite(result)

    def test_preserves_symmetry(self):
        rng = np.random.default_rng(42)
        m = rng.standard_normal((5, 5))
        m = (m + m.T) / 2
        np.fill_diagonal(m, 1.0)
        result = nearest_positive_semidefinite(m)
        np.testing.assert_allclose(result, result.T, atol=1e-10)

    def test_unit_diagonal(self):
        rng = np.random.default_rng(99)
        m = rng.standard_normal((4, 4))
        m = (m + m.T) / 2
        np.fill_diagonal(m, 1.0)
        result = nearest_positive_semidefinite(m)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-8)


class TestIsPSD:

    def test_identity_is_psd(self):
        assert is_positive_semidefinite(np.eye(3))

    def test_negative_eigenvalue(self):
        m = np.array([[1.0, 2.0], [2.0, 1.0]])
        assert not is_positive_semidefinite(m)

    def test_zero_matrix_is_psd(self):
        assert is_positive_semidefinite(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Tests: run_parallel_analysis
# ---------------------------------------------------------------------------

class TestParallelAnalysis:

    def test_returns_correct_keys(self):
        corr = np.eye(5)
        result = run_parallel_analysis(corr, n_obs=50, n_iter=10)
        assert "actual_eigenvalues" in result
        assert "random_eigenvalues" in result
        assert "suggested_n_factors" in result

    def test_identity_suggests_zero_or_one(self):
        corr = np.eye(10)
        result = run_parallel_analysis(corr, n_obs=100, n_iter=50)
        assert result["suggested_n_factors"] <= 2

    def test_strong_structure_detected(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        corr = sim.correlation.copy()
        if not is_positive_semidefinite(corr):
            corr = nearest_positive_semidefinite(corr)
        result = run_parallel_analysis(corr, n_obs=153, n_iter=50)
        assert result["suggested_n_factors"] >= 2

    def test_eigenvalue_count(self):
        n = 8
        corr = np.eye(n)
        result = run_parallel_analysis(corr, n_obs=50, n_iter=10)
        assert len(result["actual_eigenvalues"]) == n


# ---------------------------------------------------------------------------
# Tests: run_factor_analysis
# ---------------------------------------------------------------------------

class TestRunFactorAnalysis:

    def test_report_type(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        report = run_factor_analysis(sim, n_factors=3)
        assert isinstance(report, FactorReport)

    def test_loadings_shape(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        report = run_factor_analysis(sim, n_factors=3)
        assert report.loadings.shape == (18, 3)

    def test_recovery_with_strong_structure(self, three_domain_ratings):
        np.random.seed(42)
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        report = run_factor_analysis(sim, n_factors=3)
        assert report.domain_recovery_rate >= 0.6

    def test_concept_assignments_complete(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        report = run_factor_analysis(sim, n_factors=3)
        assert len(report.concept_assignments) == 18
        for ca in report.concept_assignments:
            assert "concept" in ca
            assert "domain" in ca
            assert "primary_factor" in ca
            assert "match" in ca

    def test_variance_explained_sums(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        report = run_factor_analysis(sim, n_factors=3)
        assert len(report.variance_explained) == 3
        assert report.cumulative_variance > 0
        assert report.cumulative_variance <= 1.0 + 0.01

    def test_tucker_congruence_count(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        report = run_factor_analysis(sim, n_factors=3)
        assert len(report.tucker_congruence) == 3
        for tc in report.tucker_congruence:
            assert -1.0 <= tc["congruence"] <= 1.0

    def test_two_factors(self, weak_structure_ratings):
        ratings, concepts, domain_map = weak_structure_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        report = run_factor_analysis(sim, n_factors=2)
        assert report.n_factors == 2
        assert report.loadings.shape == (6, 2)

    def test_kmo_computed(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        report = run_factor_analysis(sim, n_factors=3)
        assert 0.0 <= report.kmo_score <= 1.0

    def test_parallel_analysis_run(self, three_domain_ratings):
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        report = run_factor_analysis(sim, n_factors=3)
        assert report.parallel_n_factors >= 1
        assert len(report.eigenvalues) > 0


# ---------------------------------------------------------------------------
# Tests: format_report
# ---------------------------------------------------------------------------

class TestFormatReport:

    def test_produces_string(self, three_domain_ratings):
        np.random.seed(42)
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        report = run_factor_analysis(sim, n_factors=3)
        text = format_report(report)
        assert isinstance(text, str)
        assert len(text) > 200

    def test_contains_key_sections(self, three_domain_ratings):
        np.random.seed(42)
        ratings, concepts, domain_map = three_domain_ratings
        sim = build_similarity_matrix(ratings, concepts=concepts,
                                       domain_map=domain_map)
        report = run_factor_analysis(sim, n_factors=3)
        text = format_report(report)
        assert "KMO" in text
        assert "Bartlett" in text
        assert "Parallel Analysis" in text
        assert "Factor Loadings" in text
        assert "Domain Recovery" in text
        assert "Tucker" in text


# ---------------------------------------------------------------------------
# Tests: NumpyEncoder
# ---------------------------------------------------------------------------

class TestNumpyEncoder:

    def test_float32(self):
        val = np.float32(3.14)
        result = json.dumps({"val": val}, cls=NumpyEncoder)
        assert "3.14" in result

    def test_int64(self):
        val = np.int64(42)
        result = json.dumps({"val": val}, cls=NumpyEncoder)
        assert "42" in result

    def test_array(self):
        val = np.array([1.0, 2.0, 3.0])
        result = json.dumps({"val": val}, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["val"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_rep_per_pair(self):
        ratings = [
            PairRating("a", "b", "d1", "d1", 6, "test"),
            PairRating("a", "c", "d1", "d2", 2, "test"),
            PairRating("b", "c", "d1", "d2", 1, "test"),
        ]
        sim = build_similarity_matrix(ratings,
                                       concepts=["a", "b", "c"],
                                       domain_map={"a": "d1", "b": "d1", "c": "d2"})
        assert sim.matrix[0, 1] == 6.0

    def test_non_psd_matrix_handled(self):
        n = 6
        rng = np.random.default_rng(123)
        m = rng.uniform(0.0, 1.0, (n, n))
        m = (m + m.T) / 2
        np.fill_diagonal(m, 1.0)

        sim = SimilarityMatrix(
            model_name="test",
            concepts=[f"c{i}" for i in range(n)],
            domains=["d1", "d1", "d1", "d2", "d2", "d2"],
            matrix=m * 6 + 1,
            correlation=m,
        )
        report = run_factor_analysis(sim, n_factors=2)
        assert report.loadings.shape == (6, 2)
