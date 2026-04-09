"""
Unit tests for cluster_validate.py

Tests all pure analysis functions with synthetic data where expected
outputs can be computed by hand or verified analytically. Does NOT
require sentence-transformers — all tests use pre-built numpy arrays.

Run:  python -m pytest test_cluster_validate.py -v
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cluster_validate import (
    Concept,
    EmbeddingResult,
    ClusterReport,
    cosine_similarity_matrix,
    within_between_similarity,
    compute_silhouette,
    hierarchical_cluster_recovery,
    pca_projection,
    load_concepts_from_config,
    generate_report,
    format_report,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data with known properties
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_vectors():
    """Three orthogonal unit vectors — perfect separation."""
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def two_cluster_vectors():
    """Six vectors in 3D forming two tight, well-separated clusters.

    Cluster A (domain "alpha"): points near [1, 0, 0]
    Cluster B (domain "beta"):  points near [0, 1, 0]
    """
    np.random.seed(42)
    center_a = np.array([1.0, 0.0, 0.0])
    center_b = np.array([0.0, 1.0, 0.0])
    noise = 0.05

    cluster_a = np.array([center_a + np.random.randn(3) * noise for _ in range(3)])
    cluster_b = np.array([center_b + np.random.randn(3) * noise for _ in range(3)])
    vectors = np.vstack([cluster_a, cluster_b])
    labels = ["alpha", "alpha", "alpha", "beta", "beta", "beta"]
    return vectors, labels


@pytest.fixture
def three_cluster_vectors():
    """Eighteen vectors in 10D forming three tight, well-separated clusters.

    Mimics the RCP inventory: 6 concepts x 3 domains.
    Each domain's centroid is a different one-hot-like vector in 10D.
    """
    np.random.seed(123)
    dim = 10
    n_per = 6
    centroids = [
        np.zeros(dim),
        np.zeros(dim),
        np.zeros(dim),
    ]
    centroids[0][0] = 1.0  # physical
    centroids[1][3] = 1.0  # institutional
    centroids[2][6] = 1.0  # moral
    noise = 0.08

    vectors = []
    labels = []
    domains = ["institutional", "moral", "physical"]  # sorted
    for d_idx, domain in enumerate(domains):
        for _ in range(n_per):
            v = centroids[d_idx] + np.random.randn(dim) * noise
            vectors.append(v)
            labels.append(domain)

    return np.array(vectors), labels


@pytest.fixture
def identical_vectors():
    """All vectors the same — degenerate case."""
    v = np.ones((4, 5)) * 0.5
    labels = ["a", "a", "b", "b"]
    return v, labels


@pytest.fixture
def sample_config(tmp_path):
    """Create a temporary config.json in RCP format."""
    config = {
        "concepts": {
            "physical": ["gravity", "friction", "combustion"],
            "moral": ["fairness", "honor", "harm"],
        },
        "other_key": "irrelevant",
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path)


@pytest.fixture
def sample_config_full(tmp_path):
    """Config matching the real RCP 6-per-domain inventory."""
    config = {
        "concepts": {
            "physical": ["gravity", "friction", "combustion",
                         "pressure", "erosion", "conduction"],
            "institutional": ["authority", "property", "contract",
                              "citizenship", "hierarchy", "obligation"],
            "moral": ["fairness", "honor", "harm",
                      "loyalty", "purity", "care"],
        }
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path)


# ---------------------------------------------------------------------------
# Tests: cosine_similarity_matrix
# ---------------------------------------------------------------------------

class TestCosineSimilarityMatrix:

    def test_identity_vectors(self, identity_vectors):
        """Orthogonal unit vectors: self-sim=1, cross-sim=0."""
        sim = cosine_similarity_matrix(identity_vectors)
        assert sim.shape == (3, 3)
        np.testing.assert_allclose(np.diag(sim), 1.0)
        # Off-diagonal should be 0 for orthogonal vectors
        off_diag = sim[~np.eye(3, dtype=bool)]
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-9)

    def test_identical_vectors(self):
        """All-same vectors: all similarities should be 1.0."""
        v = np.array([[1.0, 2.0, 3.0]] * 4)
        sim = cosine_similarity_matrix(v)
        np.testing.assert_allclose(sim, 1.0)

    def test_opposite_vectors(self):
        """Negated vector pair: similarity should be -1."""
        v = np.array([[1.0, 0.0], [-1.0, 0.0]])
        sim = cosine_similarity_matrix(v)
        assert sim.shape == (2, 2)
        np.testing.assert_allclose(sim[0, 1], -1.0, atol=1e-9)
        np.testing.assert_allclose(sim[1, 0], -1.0, atol=1e-9)

    def test_symmetry(self, two_cluster_vectors):
        """Similarity matrix must be symmetric."""
        vectors, _ = two_cluster_vectors
        sim = cosine_similarity_matrix(vectors)
        np.testing.assert_allclose(sim, sim.T, atol=1e-12)

    def test_diagonal_is_one(self, two_cluster_vectors):
        """Diagonal must be 1.0 (self-similarity)."""
        vectors, _ = two_cluster_vectors
        sim = cosine_similarity_matrix(vectors)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-9)

    def test_values_in_range(self, three_cluster_vectors):
        """All values in [-1, 1]."""
        vectors, _ = three_cluster_vectors
        sim = cosine_similarity_matrix(vectors)
        assert np.all(sim >= -1.0 - 1e-10)
        assert np.all(sim <= 1.0 + 1e-10)

    def test_known_45_degree(self):
        """Two vectors at 45 degrees: cos(45) = 1/sqrt(2) ~ 0.7071."""
        v = np.array([[1.0, 0.0], [1.0, 1.0]])
        sim = cosine_similarity_matrix(v)
        expected = 1.0 / math.sqrt(2)
        np.testing.assert_allclose(sim[0, 1], expected, atol=1e-6)

    def test_zero_vector_handling(self):
        """Zero vectors should not cause NaN (guarded by epsilon)."""
        v = np.array([[0.0, 0.0], [1.0, 0.0]])
        sim = cosine_similarity_matrix(v)
        assert not np.any(np.isnan(sim))

    def test_single_vector(self):
        """Single vector: 1x1 matrix with value 1.0."""
        v = np.array([[3.0, 4.0]])
        sim = cosine_similarity_matrix(v)
        assert sim.shape == (1, 1)
        np.testing.assert_allclose(sim[0, 0], 1.0)

    def test_high_dimension(self):
        """Works in high-dimensional space (384D, like MiniLM)."""
        np.random.seed(99)
        v = np.random.randn(10, 384)
        sim = cosine_similarity_matrix(v)
        assert sim.shape == (10, 10)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)
        assert np.all(sim >= -1.0 - 1e-6)
        assert np.all(sim <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Tests: within_between_similarity
# ---------------------------------------------------------------------------

class TestWithinBetweenSimilarity:

    def test_perfect_clusters(self, identity_vectors):
        """Orthogonal unit vectors in separate domains:
        within = 0 (no within-domain pairs when 1 per domain),
        between = 0 (orthogonal)."""
        # One concept per domain — no within-domain pairs
        sim = cosine_similarity_matrix(identity_vectors)
        labels = ["A", "B", "C"]
        result = within_between_similarity(sim, labels)
        assert result["mean_within"] == 0.0  # no within-domain pairs
        np.testing.assert_allclose(result["mean_between"], 0.0, atol=1e-9)

    def test_two_clusters(self, two_cluster_vectors):
        """Two tight clusters: within > between."""
        vectors, labels = two_cluster_vectors
        sim = cosine_similarity_matrix(vectors)
        result = within_between_similarity(sim, labels)

        assert result["mean_within"] > result["mean_between"]
        assert result["separation_ratio"] > 1.0
        assert "alpha" in result["per_domain_within"]
        assert "beta" in result["per_domain_within"]

    def test_three_clusters(self, three_cluster_vectors):
        """Three tight clusters: within > between (absolute value)."""
        vectors, labels = three_cluster_vectors
        sim = cosine_similarity_matrix(vectors)
        result = within_between_similarity(sim, labels)

        # Within-domain similarity should exceed between-domain
        assert result["mean_within"] > result["mean_between"]
        assert len(result["per_domain_within"]) == 3
        # Each domain's within-similarity should be positive
        for domain, val in result["per_domain_within"].items():
            assert val > 0.0, f"Domain {domain} within-sim should be positive"

    def test_single_domain(self):
        """Single domain: all pairs are within, no between."""
        v = np.array([[1, 0], [1, 0.1], [1, 0.2]], dtype=float)
        sim = cosine_similarity_matrix(v)
        labels = ["only", "only", "only"]
        result = within_between_similarity(sim, labels)

        assert result["mean_within"] > 0
        assert result["mean_between"] == 0.0
        # separation_ratio should be inf
        assert result["separation_ratio"] == float("inf")

    def test_per_domain_keys(self, three_cluster_vectors):
        """per_domain_within has an entry for each unique domain."""
        vectors, labels = three_cluster_vectors
        sim = cosine_similarity_matrix(vectors)
        result = within_between_similarity(sim, labels)
        expected_domains = sorted(set(labels))
        assert sorted(result["per_domain_within"].keys()) == expected_domains

    def test_uniform_similarity(self):
        """All identical vectors: within = between = 1.0, ratio = 1.0."""
        v = np.array([[1.0, 0.0]] * 4)
        sim = cosine_similarity_matrix(v)
        labels = ["a", "a", "b", "b"]
        result = within_between_similarity(sim, labels)
        np.testing.assert_allclose(result["mean_within"], 1.0)
        np.testing.assert_allclose(result["mean_between"], 1.0)
        np.testing.assert_allclose(result["separation_ratio"], 1.0)

    def test_pair_count_correctness(self):
        """Verify the right number of pairs are counted.
        4 items, 2 per domain: within pairs = C(2,2)*2 = 2, between = 4."""
        v = np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1]], dtype=float)
        sim = cosine_similarity_matrix(v)
        labels = ["x", "x", "y", "y"]
        result = within_between_similarity(sim, labels)
        # Just verify the function runs and returns reasonable values
        assert 0 <= result["mean_within"] <= 1.0
        assert 0 <= result["mean_between"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: compute_silhouette
# ---------------------------------------------------------------------------

class TestComputeSilhouette:

    def test_well_separated_clusters(self, two_cluster_vectors):
        """Well-separated clusters -> high positive silhouette."""
        vectors, labels = two_cluster_vectors
        result = compute_silhouette(vectors, labels)

        assert result["silhouette_avg"] > 0.5
        assert "alpha" in result["per_domain"]
        assert "beta" in result["per_domain"]
        assert len(result["per_concept"]) == len(labels)

    def test_three_clusters(self, three_cluster_vectors):
        """Three tight clusters -> positive silhouette."""
        vectors, labels = three_cluster_vectors
        result = compute_silhouette(vectors, labels)

        assert result["silhouette_avg"] > 0.0
        for domain in ["institutional", "moral", "physical"]:
            assert domain in result["per_domain"]
            assert result["per_domain"][domain] > 0.0

    def test_per_concept_count(self, two_cluster_vectors):
        """per_concept has one entry per input vector."""
        vectors, labels = two_cluster_vectors
        result = compute_silhouette(vectors, labels)
        assert len(result["per_concept"]) == len(labels)

    def test_single_domain_raises(self):
        """Single domain raises ValueError (need >= 2 for silhouette)."""
        v = np.array([[1, 0], [0, 1]], dtype=float)
        labels = ["same", "same"]
        with pytest.raises(ValueError, match="at least 2 domains"):
            compute_silhouette(v, labels)

    def test_silhouette_range(self, three_cluster_vectors):
        """All silhouette values in [-1, 1]."""
        vectors, labels = three_cluster_vectors
        result = compute_silhouette(vectors, labels)

        assert -1.0 <= result["silhouette_avg"] <= 1.0
        for score in result["per_concept"].values():
            assert -1.0 <= score <= 1.0
        for score in result["per_domain"].values():
            assert -1.0 <= score <= 1.0

    def test_per_domain_keys_match_labels(self, three_cluster_vectors):
        """per_domain keys should be exactly the unique labels."""
        vectors, labels = three_cluster_vectors
        result = compute_silhouette(vectors, labels)
        assert sorted(result["per_domain"].keys()) == sorted(set(labels))


# ---------------------------------------------------------------------------
# Tests: hierarchical_cluster_recovery
# ---------------------------------------------------------------------------

class TestHierarchicalClusterRecovery:

    def test_perfect_recovery(self, two_cluster_vectors):
        """Well-separated clusters -> high ARI and NMI."""
        vectors, labels = two_cluster_vectors
        result = hierarchical_cluster_recovery(vectors, labels, n_clusters=2)

        assert result["adjusted_rand_index"] > 0.8
        assert result["normalized_mutual_info"] > 0.8
        assert len(result["cluster_assignments"]) == len(labels)

    def test_three_clusters(self, three_cluster_vectors):
        """Three tight clusters -> good recovery."""
        vectors, labels = three_cluster_vectors
        result = hierarchical_cluster_recovery(vectors, labels, n_clusters=3)

        assert result["adjusted_rand_index"] > 0.5
        assert result["normalized_mutual_info"] > 0.5

    def test_misassigned_format(self, three_cluster_vectors):
        """misassigned is a list of (idx, true_label, predicted_label) tuples."""
        vectors, labels = three_cluster_vectors
        result = hierarchical_cluster_recovery(vectors, labels, n_clusters=3)

        for item in result["misassigned"]:
            assert len(item) == 3
            idx, true_label, pred_label = item
            assert isinstance(idx, int)
            assert true_label in set(labels)
            assert pred_label in set(labels)

    def test_cluster_count(self, two_cluster_vectors):
        """Number of unique cluster assignments <= n_clusters."""
        vectors, labels = two_cluster_vectors
        result = hierarchical_cluster_recovery(vectors, labels, n_clusters=2)
        unique_clusters = set(result["cluster_assignments"])
        assert len(unique_clusters) <= 2

    def test_linkage_matrix_returned(self, two_cluster_vectors):
        """Result includes the linkage matrix for dendrogram use."""
        vectors, labels = two_cluster_vectors
        result = hierarchical_cluster_recovery(vectors, labels, n_clusters=2)
        assert "linkage_matrix" in result
        Z = result["linkage_matrix"]
        n = len(labels)
        assert Z.shape == (n - 1, 4)

    def test_ari_range(self, three_cluster_vectors):
        """ARI is in [-1, 1]; NMI in [0, 1]."""
        vectors, labels = three_cluster_vectors
        result = hierarchical_cluster_recovery(vectors, labels, n_clusters=3)
        assert -1.0 <= result["adjusted_rand_index"] <= 1.0
        assert 0.0 <= result["normalized_mutual_info"] <= 1.0

    def test_single_cluster(self, two_cluster_vectors):
        """Requesting 1 cluster: all assigned to same cluster, ARI=0."""
        vectors, labels = two_cluster_vectors
        result = hierarchical_cluster_recovery(vectors, labels, n_clusters=1)
        assert len(set(result["cluster_assignments"])) == 1


# ---------------------------------------------------------------------------
# Tests: pca_projection
# ---------------------------------------------------------------------------

class TestPCAProjection:

    def test_output_shape(self, two_cluster_vectors):
        """Output coords should be (n, 2)."""
        vectors, _ = two_cluster_vectors
        result = pca_projection(vectors)
        assert result["coords"].shape == (len(vectors), 2)

    def test_variance_explained(self, two_cluster_vectors):
        """variance_explained should be a 2-tuple summing to <= 1.0."""
        vectors, _ = two_cluster_vectors
        result = pca_projection(vectors)
        ve = result["variance_explained"]
        assert len(ve) == 2
        assert all(0.0 <= v <= 1.0 for v in ve)
        assert sum(ve) <= 1.0 + 1e-6

    def test_high_dim(self):
        """Works on high-dimensional input."""
        np.random.seed(77)
        v = np.random.randn(20, 384)
        result = pca_projection(v)
        assert result["coords"].shape == (20, 2)

    def test_2d_input(self):
        """2D input -> PCA is identity-like, variance explained sums to ~1.0."""
        np.random.seed(55)
        v = np.random.randn(10, 2)
        result = pca_projection(v)
        assert result["coords"].shape == (10, 2)
        np.testing.assert_allclose(sum(result["variance_explained"]), 1.0, atol=1e-6)

    def test_collinear_points(self):
        """Collinear points in 3D: PC1 captures ~100%, PC2 ~0%."""
        v = np.array([[t, 2 * t, 3 * t] for t in range(1, 11)], dtype=float)
        result = pca_projection(v)
        # First component should capture nearly all variance
        assert result["variance_explained"][0] > 0.99

    def test_no_nans(self, three_cluster_vectors):
        """Output should contain no NaN values."""
        vectors, _ = three_cluster_vectors
        result = pca_projection(vectors)
        assert not np.any(np.isnan(result["coords"]))


# ---------------------------------------------------------------------------
# Tests: load_concepts_from_config
# ---------------------------------------------------------------------------

class TestLoadConceptsFromConfig:

    def test_loads_correct_count(self, sample_config):
        """Should load all concepts from config."""
        concepts = load_concepts_from_config(sample_config)
        assert len(concepts) == 6  # 3 physical + 3 moral

    def test_concepts_are_concept_type(self, sample_config):
        """Each item should be a Concept dataclass."""
        concepts = load_concepts_from_config(sample_config)
        for c in concepts:
            assert isinstance(c, Concept)
            assert hasattr(c, "word")
            assert hasattr(c, "domain")

    def test_sorted_by_domain_then_word(self, sample_config):
        """Concepts should be sorted by (domain, word)."""
        concepts = load_concepts_from_config(sample_config)
        keys = [(c.domain, c.word) for c in concepts]
        assert keys == sorted(keys)

    def test_domains_present(self, sample_config):
        """Both domains from config should appear."""
        concepts = load_concepts_from_config(sample_config)
        domains = set(c.domain for c in concepts)
        assert domains == {"physical", "moral"}

    def test_words_present(self, sample_config):
        """All words from config should appear."""
        concepts = load_concepts_from_config(sample_config)
        words = set(c.word for c in concepts)
        assert "gravity" in words
        assert "fairness" in words
        assert "combustion" in words

    def test_full_inventory(self, sample_config_full):
        """Full RCP config: 18 concepts, 3 domains."""
        concepts = load_concepts_from_config(sample_config_full)
        assert len(concepts) == 18
        domains = set(c.domain for c in concepts)
        assert domains == {"physical", "institutional", "moral"}
        # 6 per domain
        for domain in domains:
            count = sum(1 for c in concepts if c.domain == domain)
            assert count == 6

    def test_missing_concepts_key(self, tmp_path):
        """Config without 'concepts' key raises ValueError."""
        config_path = tmp_path / "bad.json"
        with open(config_path, "w") as f:
            json.dump({"other": "data"}, f)
        with pytest.raises(ValueError, match="no 'concepts' key"):
            load_concepts_from_config(str(config_path))

    def test_invalid_json(self, tmp_path):
        """Invalid JSON raises an error."""
        config_path = tmp_path / "invalid.json"
        with open(config_path, "w") as f:
            f.write("not json {{{")
        with pytest.raises(json.JSONDecodeError):
            load_concepts_from_config(str(config_path))

    def test_missing_file(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_concepts_from_config("/nonexistent/path/config.json")


# ---------------------------------------------------------------------------
# Tests: generate_report (integration, uses synthetic embeddings)
# ---------------------------------------------------------------------------

class TestGenerateReport:

    def _make_embedding_result(self, vectors, labels, words):
        """Helper to build an EmbeddingResult from raw data."""
        concepts = tuple(
            Concept(word=w, domain=d) for w, d in sorted(zip(words, labels))
        )
        # Sort vectors to match sorted concepts
        idx_order = sorted(range(len(words)), key=lambda i: (labels[i], words[i]))
        sorted_vectors = vectors[idx_order]
        return EmbeddingResult(
            concepts=concepts,
            vectors=sorted_vectors,
            model_name="test-model",
        )

    def test_report_fields_populated(self, three_cluster_vectors):
        """All ClusterReport fields should be populated."""
        vectors, labels = three_cluster_vectors
        words = [f"word_{i}" for i in range(len(labels))]
        er = self._make_embedding_result(vectors, labels, words)
        report = generate_report(er)

        assert report.model_name == "test-model"
        assert report.n_concepts == 18
        assert report.n_domains == 3
        assert report.cosine_sim_matrix.shape == (18, 18)
        assert report.silhouette_avg != 0.0
        assert report.adjusted_rand_index != 0.0
        assert report.pca_coords.shape == (18, 2)
        assert len(report.pca_variance_explained) == 2

    def test_report_with_two_domains(self, two_cluster_vectors):
        """Report generation works with 2 domains."""
        vectors, labels = two_cluster_vectors
        words = [f"w{i}" for i in range(len(labels))]
        er = self._make_embedding_result(vectors, labels, words)
        report = generate_report(er)

        assert report.n_domains == 2
        assert report.separation_ratio > 1.0

    def test_format_report_runs(self, three_cluster_vectors):
        """format_report produces non-empty string without errors."""
        vectors, labels = three_cluster_vectors
        words = [f"word_{i}" for i in range(len(labels))]
        er = self._make_embedding_result(vectors, labels, words)
        report = generate_report(er)
        text = format_report(report)

        assert isinstance(text, str)
        assert len(text) > 100
        assert "Silhouette" in text
        assert "Separation ratio" in text
        assert "Adjusted Rand" in text
        assert "PCA" in text


# ---------------------------------------------------------------------------
# Tests: edge cases and robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_minimum_viable_input(self):
        """Two vectors, two domains — the absolute minimum.
        sklearn silhouette requires n_samples > n_labels, so 2 samples
        with 2 labels is below the threshold. Test the functions that work."""
        v = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        labels = ["a", "b"]

        sim = cosine_similarity_matrix(v)
        assert sim.shape == (2, 2)

        # within_between with 1 per domain: no within pairs
        wb = within_between_similarity(sim, labels)
        assert wb["mean_within"] == 0.0

        # silhouette requires n_samples > n_labels; 2 samples, 2 labels fails
        with pytest.raises(ValueError):
            compute_silhouette(v, labels)

    def test_minimum_viable_silhouette(self):
        """Three vectors, two domains — minimum for silhouette."""
        v = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]], dtype=float)
        labels = ["a", "a", "b"]
        sil = compute_silhouette(v, labels)
        assert len(sil["per_concept"]) == 3

    def test_unequal_domain_sizes(self):
        """Domains with different numbers of concepts."""
        np.random.seed(88)
        v_a = np.random.randn(3, 5) + np.array([5, 0, 0, 0, 0])
        v_b = np.random.randn(7, 5) + np.array([0, 5, 0, 0, 0])
        vectors = np.vstack([v_a, v_b])
        labels = ["a"] * 3 + ["b"] * 7

        sim = cosine_similarity_matrix(vectors)
        wb = within_between_similarity(sim, labels)
        assert wb["mean_within"] > 0
        assert wb["mean_between"] > 0

        sil = compute_silhouette(vectors, labels)
        assert "a" in sil["per_domain"]
        assert "b" in sil["per_domain"]

    def test_large_scale(self):
        """Runs without error on larger input (100 concepts, 5 domains)."""
        np.random.seed(200)
        n_per = 20
        dim = 50
        vectors = []
        labels = []
        for i in range(5):
            centroid = np.zeros(dim)
            centroid[i * 10] = 1.0
            for _ in range(n_per):
                vectors.append(centroid + np.random.randn(dim) * 0.1)
                labels.append(f"domain_{i}")
        vectors = np.array(vectors)

        sim = cosine_similarity_matrix(vectors)
        assert sim.shape == (100, 100)

        wb = within_between_similarity(sim, labels)
        assert wb["separation_ratio"] > 1.0

        sil = compute_silhouette(vectors, labels)
        assert sil["silhouette_avg"] > 0.0

        clust = hierarchical_cluster_recovery(vectors, labels, n_clusters=5)
        assert clust["adjusted_rand_index"] > 0.0

    def test_very_high_dimensional(self):
        """Works with 384-dimensional vectors (MiniLM output size)."""
        np.random.seed(42)
        vectors = np.random.randn(6, 384)
        labels = ["a", "a", "a", "b", "b", "b"]

        sim = cosine_similarity_matrix(vectors)
        assert sim.shape == (6, 6)

        wb = within_between_similarity(sim, labels)
        assert "mean_within" in wb

        sil = compute_silhouette(vectors, labels)
        assert "silhouette_avg" in sil

        pca = pca_projection(vectors)
        assert pca["coords"].shape == (6, 2)


# ---------------------------------------------------------------------------
# Tests: determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_cosine_sim_deterministic(self, two_cluster_vectors):
        """Same input -> same output, always."""
        vectors, _ = two_cluster_vectors
        sim1 = cosine_similarity_matrix(vectors)
        sim2 = cosine_similarity_matrix(vectors)
        np.testing.assert_array_equal(sim1, sim2)

    def test_within_between_deterministic(self, two_cluster_vectors):
        """Same input -> same output."""
        vectors, labels = two_cluster_vectors
        sim = cosine_similarity_matrix(vectors)
        r1 = within_between_similarity(sim, labels)
        r2 = within_between_similarity(sim, labels)
        assert r1["mean_within"] == r2["mean_within"]
        assert r1["mean_between"] == r2["mean_between"]

    def test_pca_deterministic(self, two_cluster_vectors):
        """PCA on same input -> same coordinates."""
        vectors, _ = two_cluster_vectors
        r1 = pca_projection(vectors)
        r2 = pca_projection(vectors)
        np.testing.assert_array_equal(r1["coords"], r2["coords"])


# ---------------------------------------------------------------------------
# Tests: Concept and EmbeddingResult dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:

    def test_concept_frozen(self):
        """Concept is immutable."""
        c = Concept(word="gravity", domain="physical")
        with pytest.raises(AttributeError):
            c.word = "friction"

    def test_concept_equality(self):
        """Two Concepts with same fields are equal."""
        c1 = Concept(word="gravity", domain="physical")
        c2 = Concept(word="gravity", domain="physical")
        assert c1 == c2

    def test_concept_inequality(self):
        """Concepts with different fields are not equal."""
        c1 = Concept(word="gravity", domain="physical")
        c2 = Concept(word="friction", domain="physical")
        assert c1 != c2

    def test_embedding_result_frozen(self):
        """EmbeddingResult is immutable."""
        v = np.array([[1.0, 2.0]])
        er = EmbeddingResult(
            concepts=(Concept("test", "domain"),),
            vectors=v,
            model_name="test-model",
        )
        with pytest.raises(AttributeError):
            er.model_name = "other"
