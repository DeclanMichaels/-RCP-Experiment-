#!/usr/bin/env python3
"""
Unit tests for collect.py functions.
Run: python -m pytest test_collect.py -v
"""

import pytest
from collect import (
    load_config,
    get_all_concepts,
    get_concept_domain,
    get_all_pairs,
    randomize_pair_directions,
    get_target_domain_pairs,
    build_rating_prompt,
    build_explanation_prompt,
    build_manipulation_check_prompt,
    parse_rating,
    detect_refusal,
)


@pytest.fixture
def config():
    return load_config("config.json")


# ---- Config helpers ----

class TestGetAllConcepts:
    def test_returns_sorted(self, config):
        concepts = get_all_concepts(config)
        assert concepts == sorted(concepts)

    def test_count_is_18(self, config):
        assert len(get_all_concepts(config)) == 18

    def test_no_duplicates(self, config):
        concepts = get_all_concepts(config)
        assert len(concepts) == len(set(concepts))

    def test_includes_known_concepts(self, config):
        concepts = get_all_concepts(config)
        assert "gravity" in concepts
        assert "fairness" in concepts
        assert "authority" in concepts


class TestGetConceptDomain:
    def test_physical(self, config):
        for c in ["gravity", "friction", "combustion", "pressure", "erosion", "conduction"]:
            assert get_concept_domain(c, config) == "physical"

    def test_institutional(self, config):
        for c in ["authority", "property", "contract", "citizenship", "hierarchy", "obligation"]:
            assert get_concept_domain(c, config) == "institutional"

    def test_moral(self, config):
        for c in ["fairness", "honor", "harm", "loyalty", "purity", "care"]:
            assert get_concept_domain(c, config) == "moral"

    def test_unknown_returns_none(self, config):
        assert get_concept_domain("banana", config) is None
        assert get_concept_domain("", config) is None


class TestGetAllPairs:
    def test_count(self, config):
        concepts = get_all_concepts(config)
        pairs = get_all_pairs(concepts)
        n = len(concepts)
        assert len(pairs) == n * (n - 1) // 2

    def test_no_self_pairs(self, config):
        pairs = get_all_pairs(get_all_concepts(config))
        for a, b in pairs:
            assert a != b

    def test_no_duplicate_pairs(self, config):
        pairs = get_all_pairs(get_all_concepts(config))
        assert len(pairs) == len(set(pairs))

    def test_alphabetical_order(self, config):
        pairs = get_all_pairs(get_all_concepts(config))
        for a, b in pairs:
            assert a < b


class TestGetTargetDomainPairs:
    def test_count(self, config):
        assert len(get_target_domain_pairs(config)) == 15  # C(6,2)

    def test_all_moral(self, config):
        for a, b in get_target_domain_pairs(config):
            assert get_concept_domain(a, config) == "moral"
            assert get_concept_domain(b, config) == "moral"


# ---- Pair randomization ----

class TestRandomizePairDirections:
    def test_deterministic_with_seed(self):
        pairs = [("a", "b"), ("c", "d"), ("e", "f")]
        r1, s1 = randomize_pair_directions(pairs, seed=42)
        r2, s2 = randomize_pair_directions(pairs, seed=42)
        assert r1 == r2
        assert s1 == s2

    def test_preserves_pair_content(self):
        pairs = [("a", "b"), ("c", "d")]
        result, _ = randomize_pair_directions(pairs, seed=42)
        for ra, rb in result:
            assert {ra, rb} in [{"a", "b"}, {"c", "d"}]

    def test_preserves_count(self):
        pairs = [("a", "b"), ("c", "d"), ("e", "f")]
        result, _ = randomize_pair_directions(pairs, seed=42)
        assert len(result) == len(pairs)

    def test_returns_int_seed(self):
        _, seed = randomize_pair_directions([("a", "b")], seed=None)
        assert isinstance(seed, int)

    def test_some_pairs_flip(self):
        pairs = [(chr(65 + i), chr(65 + i + 1)) for i in range(0, 20, 2)]
        result, _ = randomize_pair_directions(pairs, seed=42)
        flipped = sum(1 for (a, _), (ra, _) in zip(pairs, result) if a != ra)
        assert 0 < flipped < len(pairs)


# ---- Prompt construction ----

class TestBuildRatingPrompt:
    def test_contains_concepts(self):
        p = build_rating_prompt("gravity", "friction", "")
        assert '"gravity"' in p
        assert '"friction"' in p

    def test_contains_scale(self):
        p = build_rating_prompt("a", "b", "")
        assert "1 to 7" in p
        assert "Respond with only the number" in p

    def test_framing_prepended(self):
        p = build_rating_prompt("a", "b", "Consider a collectivist society.")
        assert p.startswith("Consider a collectivist society.")

    def test_no_framing_no_leading_blank(self):
        p = build_rating_prompt("a", "b", "")
        assert not p.startswith("\n")


class TestBuildExplanationPrompt:
    def test_contains_concepts(self):
        p = build_explanation_prompt("fairness", "honor", "")
        assert '"fairness"' in p
        assert '"honor"' in p

    def test_asks_for_one_sentence(self):
        p = build_explanation_prompt("a", "b", "")
        assert "one sentence" in p.lower()


class TestBuildManipulationCheckPrompt:
    def test_contains_framing(self):
        f = "Consider a hierarchical society."
        p = build_manipulation_check_prompt(f)
        assert f in p

    def test_asks_about_society(self):
        p = build_manipulation_check_prompt("Framing text.")
        assert "society" in p.lower()
        assert "foundation of a good life" in p


# ---- parse_rating ----

class TestParseRating:
    @pytest.mark.parametrize("input_str,expected", [
        ("1", (1, True)),
        ("2", (2, True)),
        ("3", (3, True)),
        ("4", (4, True)),
        ("5", (5, True)),
        ("6", (6, True)),
        ("7", (7, True)),
    ])
    def test_valid_single_digits(self, input_str, expected):
        assert parse_rating(input_str) == expected

    def test_whitespace(self):
        assert parse_rating("  5  ") == (5, True)

    def test_trailing_period(self):
        assert parse_rating("5.") == (5, True)

    def test_newline(self):
        assert parse_rating("5\n") == (5, True)

    def test_text_with_number(self):
        assert parse_rating("Rating: 4")[0] == 4

    def test_zero_invalid(self):
        assert parse_rating("0")[1] is False

    def test_eight_invalid(self):
        assert parse_rating("8")[1] is False

    def test_nine_invalid(self):
        assert parse_rating("9")[1] is False

    def test_empty_string(self):
        assert parse_rating("")[1] is False

    def test_no_digits(self):
        assert parse_rating("hello")[1] is False

    def test_refusal_text_no_number(self):
        assert parse_rating("I cannot rate these concepts")[1] is False


# ---- detect_refusal ----

class TestDetectRefusal:
    @pytest.mark.parametrize("text", [
        "I cannot compare these concepts.",
        "I can't rate morality.",
        "As an AI, I don't have opinions.",
        "As a language model, I must decline.",
        "Morality is subjective and cannot be rated.",
        "I don't think it's appropriate to compare these.",
        "I'm not able to make this judgment.",
        "I am not able to rate these.",
        "It would not be appropriate to compare purity and harm.",
        "I must respectfully decline.",
        "I'm unable to provide a rating.",
        "I cannot rate these concepts objectively.",
        "Cannot compare moral concepts this way.",
        "I'm not comfortable making this comparison.",
    ])
    def test_detects_refusal(self, text):
        assert detect_refusal(text) is True

    @pytest.mark.parametrize("text", [
        "5",
        "3",
        "The concepts are moderately related.",
        "These share some overlap in meaning.",
        "",
    ])
    def test_no_false_positive(self, text):
        assert detect_refusal(text) is False

    def test_none_input(self):
        assert detect_refusal(None) is False

    def test_case_insensitive(self):
        assert detect_refusal("AS AN AI, I cannot do this.") is True
