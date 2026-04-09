#!/usr/bin/env python3
"""
Unit tests for analyze_explanations.py functions.
Run: python -m pytest test_analyze_explanations.py -v
"""

import json
import pytest
from itertools import combinations

from analyze_explanations import (
    ExplanationRecord,
    GroupKey,
    count_matches,
    has_match,
    echo_score,
    hedge_count,
    has_perspective_marker,
    group_by,
    by_model_framing,
    aggregate_echo_rates,
    aggregate_hedge_rates,
    aggregate_perspective_adoption,
    aggregate_response_lengths,
    compute_lexical_overlap,
    load_explanations,
    run_analysis,
    serialize_results,
    format_summary_tables,
    FRAMING_ECHO_PATTERNS,
    HEDGE_PATTERNS,
    PERSPECTIVE_MARKERS,
    DATA_DIRS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(model_name="claude-sonnet", concept_a="care",
                concept_b="fairness", framing="neutral",
                raw_response="Care and fairness are related.") -> ExplanationRecord:
    """Build a single synthetic ExplanationRecord."""
    return ExplanationRecord(
        model_name=model_name,
        concept_a=concept_a,
        concept_b=concept_b,
        framing=framing,
        raw_response=raw_response,
        latency_ms=2000.0,
    )


def make_record_set(model_name, framing, response_template):
    """Build 15 ExplanationRecords (all moral pairs) for one model+framing."""
    concepts = ["care", "fairness", "harm", "honor", "loyalty", "purity"]
    return [
        make_record(
            model_name=model_name,
            concept_a=a, concept_b=b,
            framing=framing,
            raw_response=response_template.format(concept_a=a, concept_b=b),
        )
        for a, b in combinations(concepts, 2)
    ]


# ---------------------------------------------------------------------------
# ExplanationRecord
# ---------------------------------------------------------------------------

class TestExplanationRecord:
    def test_from_json_normal(self):
        d = {
            "model_name": "grok", "concept_a": "care", "concept_b": "harm",
            "framing": "neutral", "raw_response": "Hello world.",
            "latency_ms": 500,
        }
        rec = ExplanationRecord.from_json(d)
        assert rec.model_name == "grok"
        assert rec.raw_response == "Hello world."

    def test_from_json_null_response(self):
        d = {
            "model_name": "grok", "concept_a": "care", "concept_b": "harm",
            "framing": "neutral", "raw_response": None, "latency_ms": 0,
        }
        rec = ExplanationRecord.from_json(d)
        assert rec.raw_response == ""
        assert rec.word_count == 0

    def test_from_json_missing_response(self):
        d = {"model_name": "grok", "concept_a": "a", "concept_b": "b",
             "framing": "neutral"}
        rec = ExplanationRecord.from_json(d)
        assert rec.raw_response == ""

    def test_pair_is_sorted(self):
        rec = make_record(concept_a="purity", concept_b="care")
        assert rec.pair == ("care", "purity")

    def test_word_count(self):
        rec = make_record(raw_response="One two three four.")
        assert rec.word_count == 4

    def test_word_set_lowercase_min_length(self):
        rec = make_record(raw_response="A big Cat is on the mat.")
        ws = rec.word_set
        assert "big" in ws
        assert "cat" in ws
        assert "mat" in ws
        assert "is" not in ws   # 2 chars, below 3-char minimum
        assert "the" in ws     # 3 chars, meets minimum
        assert "on" not in ws  # 2 chars, below minimum

    def test_frozen(self):
        rec = make_record()
        with pytest.raises(AttributeError):
            rec.model_name = "changed"


# ---------------------------------------------------------------------------
# count_matches / has_match
# ---------------------------------------------------------------------------

class TestCountMatches:
    def test_no_matches(self):
        assert count_matches("The sky is blue.", [r"\bred\b"]) == 0

    def test_single_match(self):
        assert count_matches("Individual rights.", [r"\bindividual\w*\b"]) == 1

    def test_multiple_patterns(self):
        text = "Individual autonomy and personal freedom are key."
        patterns = [r"\bindividual\w*\b", r"\bpersonal\s+(choice|freedom)\w*\b"]
        assert count_matches(text, patterns) == 2

    def test_case_insensitive(self):
        assert count_matches("TRIANGLE GEOMETRIC.", [r"\btriangle\w*\b", r"\bgeometr\w*\b"]) == 2

    def test_empty_text(self):
        assert count_matches("", [r"\bfoo\b"]) == 0

    def test_empty_patterns(self):
        assert count_matches("Some text.", []) == 0

    def test_multiple_occurrences(self):
        assert count_matches("triangle and triangles and more triangle",
                             [r"\btriangle\w*\b"]) == 3


class TestHasMatch:
    def test_true_on_match(self):
        assert has_match("Triangles are shapes.", [r"\btriangle\w*\b"]) is True

    def test_false_on_no_match(self):
        assert has_match("Nothing here.", [r"\btriangle\w*\b"]) is False

    def test_short_circuits(self):
        # Second pattern would also match, but has_match should return True
        # after the first. We can't directly test short-circuiting, but
        # we verify correctness.
        assert has_match("Triangles", [r"\btriangle\w*\b", r"\bfoo\b"]) is True


# ---------------------------------------------------------------------------
# Per-record measurement functions
# ---------------------------------------------------------------------------

class TestEchoScore:
    def test_neutral_returns_zero(self):
        rec = make_record(framing="neutral",
                          raw_response="Individual autonomy matters.")
        assert echo_score(rec) == 0

    def test_individualist_echo(self):
        rec = make_record(framing="individualist",
                          raw_response="Personal autonomy and individual rights.")
        assert echo_score(rec) >= 2

    def test_nonsense_echo(self):
        rec = make_record(framing="nonsense",
                          raw_response="Triangles are geometrically angular.")
        assert echo_score(rec) >= 2

    def test_no_echo_in_framed(self):
        rec = make_record(framing="individualist",
                          raw_response="Care and fairness relate to justice.")
        assert echo_score(rec) == 0


class TestHedgeCount:
    def test_no_hedges(self):
        rec = make_record(raw_response="Care requires fairness.")
        assert hedge_count(rec) == 0

    def test_hedges_detected(self):
        rec = make_record(raw_response="Care and fairness are often intertwined.")
        assert hedge_count(rec) >= 2  # "often", "intertwined"


class TestHasPerspectiveMarker:
    def test_no_marker(self):
        rec = make_record(raw_response="Care and fairness are related.")
        assert has_perspective_marker(rec) is False

    def test_from_perspective(self):
        rec = make_record(raw_response="From this individualistic perspective, care is key.")
        assert has_perspective_marker(rec) is True

    def test_in_context_of(self):
        rec = make_record(raw_response="In the context of warm weather, harm increases.")
        assert has_perspective_marker(rec) is True

    def test_angular_framework(self):
        rec = make_record(raw_response="From our angular moral framework, virtue flows.")
        assert has_perspective_marker(rec) is True


# ---------------------------------------------------------------------------
# group_by
# ---------------------------------------------------------------------------

class TestGroupBy:
    def test_groups_correctly(self):
        recs = [
            make_record(model_name="a", framing="neutral"),
            make_record(model_name="a", framing="nonsense"),
            make_record(model_name="b", framing="neutral"),
        ]
        groups = group_by(recs, by_model_framing)
        assert len(groups) == 3
        assert len(groups[GroupKey("a", "neutral")]) == 1
        assert len(groups[GroupKey("a", "nonsense")]) == 1
        assert len(groups[GroupKey("b", "neutral")]) == 1

    def test_empty_input(self):
        groups = group_by([], by_model_framing)
        assert groups == {}


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

class TestAggregateEchoRates:
    def test_neutral_zero(self):
        recs = [make_record(framing="neutral",
                            raw_response="Individual autonomy matters.")]
        groups = group_by(recs, by_model_framing)
        result = aggregate_echo_rates(groups)
        assert result[GroupKey("claude-sonnet", "neutral")]["echo_rate"] == 0.0

    def test_full_echo(self):
        recs = [make_record(framing="individualist",
                            raw_response="Personal autonomy and individual rights.")]
        groups = group_by(recs, by_model_framing)
        result = aggregate_echo_rates(groups)
        assert result[GroupKey("claude-sonnet", "individualist")]["echo_rate"] == 1.0

    def test_mixed_rate(self):
        recs = [
            make_record(framing="collectivist",
                        raw_response="Group harmony requires mutual obligation."),
            make_record(framing="collectivist", concept_a="harm", concept_b="honor",
                        raw_response="Harm and honor are opposing forces."),
        ]
        groups = group_by(recs, by_model_framing)
        result = aggregate_echo_rates(groups)
        assert result[GroupKey("claude-sonnet", "collectivist")]["echo_rate"] == 0.5


class TestAggregateHedgeRates:
    def test_no_hedges(self):
        recs = [make_record(raw_response="Care requires fairness.")]
        groups = group_by(recs, by_model_framing)
        result = aggregate_hedge_rates(groups)
        assert result[GroupKey("claude-sonnet", "neutral")]["mean_hedge_count"] == 0.0

    def test_hedges_counted(self):
        recs = [make_record(raw_response="Care and fairness are often intertwined and complementary.")]
        groups = group_by(recs, by_model_framing)
        result = aggregate_hedge_rates(groups)
        assert result[GroupKey("claude-sonnet", "neutral")]["mean_hedge_count"] >= 2.0


class TestAggregatePerspectiveAdoption:
    def test_no_markers(self):
        recs = [make_record(raw_response="Care and fairness are related.")]
        groups = group_by(recs, by_model_framing)
        result = aggregate_perspective_adoption(groups)
        assert result[GroupKey("claude-sonnet", "neutral")]["adoption_rate"] == 0.0

    def test_with_marker(self):
        recs = [make_record(framing="individualist",
                            raw_response="From this individualistic perspective, care is key.")]
        groups = group_by(recs, by_model_framing)
        result = aggregate_perspective_adoption(groups)
        assert result[GroupKey("claude-sonnet", "individualist")]["adoption_rate"] == 1.0


class TestAggregateResponseLengths:
    def test_basic(self):
        recs = [
            make_record(raw_response="One two three."),
            make_record(concept_a="harm", concept_b="honor",
                        raw_response="One two three four five."),
        ]
        groups = group_by(recs, by_model_framing)
        result = aggregate_response_lengths(groups)
        data = result[GroupKey("claude-sonnet", "neutral")]
        assert data["mean_words"] == 4.0
        assert data["min_words"] == 3
        assert data["max_words"] == 5

    def test_empty_response(self):
        recs = [make_record(raw_response="")]
        groups = group_by(recs, by_model_framing)
        result = aggregate_response_lengths(groups)
        assert result[GroupKey("claude-sonnet", "neutral")]["mean_words"] == 0.0


# ---------------------------------------------------------------------------
# compute_lexical_overlap
# ---------------------------------------------------------------------------

class TestLexicalOverlap:
    def test_identical_responses(self):
        text = "Care and fairness are deeply related moral concepts."
        recs = [
            make_record(framing="neutral", raw_response=text),
            make_record(framing="individualist", raw_response=text),
        ]
        result = compute_lexical_overlap(recs)
        assert result["claude-sonnet"][("neutral", "individualist")] == 1.0

    def test_different_responses(self):
        recs = [
            make_record(framing="neutral",
                        raw_response="Care and fairness are deeply related moral concepts."),
            make_record(framing="nonsense",
                        raw_response="Triangles represent geometric angular virtue edges shapes."),
        ]
        result = compute_lexical_overlap(recs)
        assert result["claude-sonnet"][("neutral", "nonsense")] < 0.5

    def test_multiple_pairs_averaged(self):
        recs = [
            make_record(framing="neutral", concept_a="care", concept_b="fairness",
                        raw_response="Care relates to fairness through justice."),
            make_record(framing="individualist", concept_a="care", concept_b="fairness",
                        raw_response="Care relates to fairness through justice."),
            make_record(framing="neutral", concept_a="harm", concept_b="honor",
                        raw_response="Completely different vocabulary here entirely."),
            make_record(framing="individualist", concept_a="harm", concept_b="honor",
                        raw_response="Nothing overlapping whatsoever at all indeed."),
        ]
        result = compute_lexical_overlap(recs)
        jaccard = result["claude-sonnet"][("neutral", "individualist")]
        assert 0.3 < jaccard < 0.7


# ---------------------------------------------------------------------------
# load_explanations
# ---------------------------------------------------------------------------

class TestLoadExplanations:
    def test_loads_from_directory(self, tmp_path):
        runs_dir = tmp_path / "runs"
        data_dir = runs_dir / "20260324-1-Sonnet-Moral-Data"
        data_dir.mkdir(parents=True)

        d = {
            "model_name": "claude-sonnet", "concept_a": "care",
            "concept_b": "fairness", "framing": "neutral",
            "raw_response": "Test.", "latency_ms": 100,
        }
        with open(data_dir / "explanations.jsonl", "w") as f:
            f.write(json.dumps(d) + "\n")

        records = load_explanations(str(tmp_path))
        assert len(records) == 1
        assert isinstance(records[0], ExplanationRecord)
        assert records[0].model_name == "claude-sonnet"

    def test_handles_missing_directory(self, tmp_path):
        (tmp_path / "runs").mkdir()
        records = load_explanations(str(tmp_path))
        assert len(records) == 0

    def test_skips_empty_lines(self, tmp_path):
        runs_dir = tmp_path / "runs"
        data_dir = runs_dir / "20260324-1-Sonnet-Moral-Data"
        data_dir.mkdir(parents=True)

        d = {"model_name": "claude-sonnet", "concept_a": "care",
             "concept_b": "fairness", "framing": "neutral",
             "raw_response": "Test.", "latency_ms": 100}
        with open(data_dir / "explanations.jsonl", "w") as f:
            f.write(json.dumps(d) + "\n\n" + json.dumps(d) + "\n")

        records = load_explanations(str(tmp_path))
        assert len(records) == 2

    def test_null_response_normalized(self, tmp_path):
        runs_dir = tmp_path / "runs"
        data_dir = runs_dir / "20260324-1-Sonnet-Moral-Data"
        data_dir.mkdir(parents=True)

        d = {"model_name": "claude-sonnet", "concept_a": "care",
             "concept_b": "fairness", "framing": "neutral",
             "raw_response": None, "latency_ms": 0}
        with open(data_dir / "explanations.jsonl", "w") as f:
            f.write(json.dumps(d) + "\n")

        records = load_explanations(str(tmp_path))
        assert records[0].raw_response == ""


# ---------------------------------------------------------------------------
# run_analysis + serialize_results (integration)
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    def test_with_synthetic_data(self):
        neutral = make_record_set(
            "claude-sonnet", "neutral",
            "{concept_a} and {concept_b} are often related moral concepts."
        )
        nonsense = make_record_set(
            "claude-sonnet", "nonsense",
            "From our angular framework, triangles connect {concept_a} and {concept_b}."
        )
        results = run_analysis(neutral + nonsense)

        assert results["total_records"] == 30
        assert "claude-sonnet" in results["models"]

        # Nonsense should have high echo rate
        key = GroupKey("claude-sonnet", "nonsense")
        assert results["echo_rates"][key]["echo_rate"] > 0.5

        # Neutral echo rate is zero
        key_n = GroupKey("claude-sonnet", "neutral")
        assert results["echo_rates"][key_n]["echo_rate"] == 0.0

    def test_empty_records(self):
        results = run_analysis([])
        assert "error" in results

    def test_serialize_roundtrip(self):
        recs = make_record_set("claude-sonnet", "neutral", "{concept_a} {concept_b}")
        results = run_analysis(recs)
        serialized = serialize_results(results)
        # Should be JSON-serializable
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)
        assert "summary" in parsed
        assert parsed["summary"]["total_records"] == 15


# ---------------------------------------------------------------------------
# format_summary_tables
# ---------------------------------------------------------------------------

class TestFormatSummaryTables:
    def test_returns_string(self):
        recs = make_record_set("claude-sonnet", "neutral", "{concept_a} {concept_b}")
        results = run_analysis(recs)
        output = format_summary_tables(results)
        assert isinstance(output, str)
        assert "claude-sonnet" in output
        assert "Echo Rate" in output

    def test_error_case(self):
        output = format_summary_tables({"error": "No data"})
        assert "Error" in output


# ---------------------------------------------------------------------------
# Pattern coverage (verify patterns match known real-data phrases)
# ---------------------------------------------------------------------------

class TestPatternCoverage:
    @pytest.mark.parametrize("text,framing,expected_min", [
        ("From an individualistic perspective, personal autonomy drives care.",
         "individualist", 1),
        ("Group harmony requires mutual obligation to the community.",
         "collectivist", 2),
        ("Social rank and hierarchical structures define duty.",
         "hierarchical", 2),
        ("Equal participation rejects rank distinctions.",
         "egalitarian", 2),
        ("In the context of warm weather and rising temperatures.",
         "irrelevant", 2),
        ("Triangles are geometrically superior angular shapes with edges.",
         "nonsense", 4),
    ])
    def test_echo_patterns(self, text, framing, expected_min):
        patterns = FRAMING_ECHO_PATTERNS[framing]
        assert count_matches(text, patterns) >= expected_min

    @pytest.mark.parametrize("text,expected_min", [
        ("Care and fairness are often intertwined.", 2),
        ("These concepts can be complementary and tend to overlap.", 2),
        ("They may sometimes conflict in practice.", 2),
    ])
    def test_hedge_patterns(self, text, expected_min):
        assert count_matches(text, HEDGE_PATTERNS) >= expected_min

    @pytest.mark.parametrize("text", [
        "From this individualistic perspective, care is key.",
        "From our angular moral framework, virtue flows.",
        "In the context of warm weather, harm increases.",
        "In this society, loyalty matters most.",
    ])
    def test_perspective_markers(self, text):
        assert has_match(text, PERSPECTIVE_MARKERS) is True
