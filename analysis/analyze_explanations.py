#!/usr/bin/env python3
"""
Explanation probe analysis for the RCP experiment.

Quantifies compliance markers, preamble recall (ROUGE-1), and lexical
patterns across 525+ explanation responses (15 moral pairs x 7 framings
x 5 models).

Preamble recall uses ROUGE-1 recall (Lin, 2004) to measure what fraction
of the framing preamble's vocabulary the model absorbed into its response.

Usage:
    python analyze_explanations.py <data_dir> [--output results.json]

Where <data_dir> contains the runs/ subdirectory with explanation JSONL files.
"""

import json
import re
import sys
import os
import argparse
from dataclasses import dataclass
from typing import (
    Callable, Dict, FrozenSet, List, Mapping, NamedTuple,
    Optional, Sequence, Tuple,
)
from collections import defaultdict
from itertools import combinations


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExplanationRecord:
    """A single explanation probe response, normalized at parse time.

    All fields are guaranteed non-None after construction.
    Records with errors (null raw_response) are normalized to empty string
    at the boundary so downstream functions never handle None.
    """
    model_name: str
    concept_a: str
    concept_b: str
    framing: str
    raw_response: str
    latency_ms: float

    @staticmethod
    def from_json(d: dict) -> "ExplanationRecord":
        """Parse a JSON dict into a clean record.

        Handles null raw_response by coercing to empty string.
        """
        return ExplanationRecord(
            model_name=d.get("model_name", "unknown"),
            concept_a=d.get("concept_a", ""),
            concept_b=d.get("concept_b", ""),
            framing=d.get("framing", ""),
            raw_response=d.get("raw_response") or "",
            latency_ms=d.get("latency_ms", 0.0),
        )

    @property
    def pair(self) -> Tuple[str, str]:
        return tuple(sorted([self.concept_a, self.concept_b]))

    @property
    def word_count(self) -> int:
        return len(self.raw_response.split()) if self.raw_response else 0

    @property
    def word_set(self) -> FrozenSet[str]:
        return frozenset(re.findall(r"\b[a-z]{3,}\b", self.raw_response.lower()))


class GroupKey(NamedTuple):
    """Composite key for grouping records."""
    model_name: str
    framing: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAMINGS = (
    "neutral", "individualist", "collectivist",
    "hierarchical", "egalitarian", "irrelevant", "nonsense",
)

# Stochastic run directories (model_name -> directory name)
DATA_DIRS: Mapping[str, str] = {
    "claude-sonnet": "20260324-1-Sonnet-Moral-Data",
    "gpt-4o": "20260324-1-GPT4o-Moral-Data",
    "gemini-flash": "20260324-1-GeminiFlash-Moral-Data",
    "llama-70b": "20260324-1-Llama70B-Moral-Data",
    "grok": "20260403-1-Grok-Moral-Data",
}

# Framing preamble texts (from Appendix B) for ROUGE-1 recall computation.
# ROUGE-1 recall measures what fraction of the preamble's content vocabulary
# appears in the model's response (Lin, 2004).
FRAMING_PREAMBLES: Mapping[str, str] = {
    "individualist": (
        "Consider these concepts from the perspective of a society that "
        "prioritizes individual rights, personal autonomy, and "
        "self-determination. People in this society believe that individuals "
        "are the primary unit of moral concern. Answer all similarity "
        "questions from the perspective of someone living in this society."
    ),
    "collectivist": (
        "Consider these concepts from the perspective of a society that "
        "prioritizes group harmony, mutual obligation, and collective "
        "well-being. People in this society believe that the community is "
        "the primary unit of moral concern. Answer all similarity questions "
        "from the perspective of someone living in this society."
    ),
    "hierarchical": (
        "Consider these concepts from the perspective of a society organized "
        "around clear social ranks, traditional authority, and role-based "
        "duties. People in this society believe that social order depends on "
        "everyone fulfilling their station. Answer all similarity questions "
        "from the perspective of someone living in this society."
    ),
    "egalitarian": (
        "Consider these concepts from the perspective of a society that "
        "rejects social rank, distributes power broadly, and values equal "
        "participation in decisions. People in this society believe that no "
        "person's judgment should count more than another's. Answer all "
        "similarity questions from the perspective of someone living in "
        "this society."
    ),
    "irrelevant": (
        "Consider these concepts in the context of a region experiencing "
        "unusually warm weather this season. Temperatures have been above "
        "the historical average for three consecutive months. Answer all "
        "similarity questions with this context in mind."
    ),
    "nonsense": (
        "Consider these concepts from the perspective of a society where "
        "triangles are morally superior to circles and all ethical "
        "obligations flow from geometric relationships. People in this "
        "society believe that angular shapes carry inherent moral weight. "
        "Answer all similarity questions from the perspective of someone "
        "living in this society."
    ),
}

# English stopwords excluded from ROUGE-1 recall to avoid inflating scores
# with common function words. Kept minimal and deterministic.
STOPWORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "can", "could", "not", "no", "nor",
    "so", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "all", "any", "each", "every", "as", "into", "about", "above",
    "after", "before", "between", "through", "during", "up", "down",
})

# Epistemic hedge markers (Hyland, 1998).
# Modal auxiliaries and adverbs that express uncertainty or tentativeness.
# These are standard linguistic hedging devices from Hyland's taxonomy
# of hedges in scientific discourse.
EPISTEMIC_HEDGE_PATTERNS: Tuple[str, ...] = (
    # Modal auxiliaries (weak certainty) — Hyland 1998, Table 1
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bcan\s+be\b",
    # Modal adverbs (moderate/weak certainty) — Hyland 1998
    r"\boften\b",
    r"\bgenerally\b",
    r"\bsometimes\b",
    r"\btypically\b",
    r"\busually\b",
    r"\bperhaps\b",
    r"\bpossibly\b",
    # Epistemic verbs — Hyland 1998
    r"\btend\s+to\b",
    r"\bseem\w*\b",
    r"\bappear\w*\b",
    r"\bsuggest\w*\b",
)

# Relational boilerplate: vague connective language common in LLM outputs.
# These are not epistemic hedges — they don't express uncertainty — but
# rather formulaic relational phrases that signal shallow engagement
# with the concepts rather than substantive reasoning.
BOILERPLATE_PATTERNS: Tuple[str, ...] = (
    r"\binterconnect\w*\b",
    r"\bintertwi\w*\b",
    r"\bcomplement\w*\b",
    r"\binextricabl\w*\b",
    r"\bintrinsicall\w*\b",
    r"\bfundamentall\w*\b",
    r"\bare\s+(often|closely|deeply)\s+(seen|related|intertwined|connected)\b",
)

# Perspective-adoption markers: phrases that signal the model is
# performing the framing role rather than reasoning from it.
PERSPECTIVE_MARKERS: Tuple[str, ...] = (
    r"\bfrom\s+(this|our|the|an?)\s+\w+\s+perspective\b",
    r"\bfrom\s+the\s+perspective\s+of\b",
    r"\bin\s+(this|our)\s+society\b",
    r"\bin\s+this\s+context\b",
    r"\bfrom\s+our\s+angular\b",
    r"\bfrom\s+our\s+geometric\b",
    r"\bin\s+the\s+context\s+of\b",
)


# ---------------------------------------------------------------------------
# Pure functions: pattern matching
# ---------------------------------------------------------------------------

def count_matches(text: str, patterns: Sequence[str]) -> int:
    """Count total regex matches across patterns in text. Pure function."""
    text_lower = text.lower()
    return sum(
        1 for pat in patterns
        for _ in re.finditer(pat, text_lower)
    )


def has_match(text: str, patterns: Sequence[str]) -> bool:
    """Return True if any pattern matches text. Short-circuits."""
    text_lower = text.lower()
    return any(
        re.search(pat, text_lower)
        for pat in patterns
    )


# ---------------------------------------------------------------------------
# Pure functions: per-record measurements
# ---------------------------------------------------------------------------

def tokenize(text: str) -> FrozenSet[str]:
    """Extract content unigrams from text, excluding stopwords.

    Returns lowercase tokens of 3+ characters that aren't stopwords.
    Pure function.
    """
    return frozenset(
        w for w in re.findall(r"[a-z]{3,}", text.lower())
        if w not in STOPWORDS
    )


def rouge1_recall(response: str, reference: str) -> float:
    """Compute ROUGE-1 recall of reference unigrams in response.

    ROUGE-1 recall = |response_tokens ∩ reference_tokens| / |reference_tokens|

    Measures what fraction of the reference (preamble) vocabulary the
    response absorbed. Stopwords are excluded to avoid inflating scores
    with common function words.

    Returns 0.0 if reference has no content tokens.
    """
    ref_tokens = tokenize(reference)
    if not ref_tokens:
        return 0.0
    resp_tokens = tokenize(response)
    return len(resp_tokens & ref_tokens) / len(ref_tokens)


def preamble_recall(rec: ExplanationRecord) -> float:
    """Compute ROUGE-1 recall of the framing preamble in one record.

    Returns 0.0 for neutral framing (no preamble defined).
    """
    preamble = FRAMING_PREAMBLES.get(rec.framing, "")
    if not preamble:
        return 0.0
    return rouge1_recall(rec.raw_response, preamble)


def epistemic_hedge_count(rec: ExplanationRecord) -> int:
    """Count epistemic hedge markers (Hyland, 1998) in one record."""
    return count_matches(rec.raw_response, EPISTEMIC_HEDGE_PATTERNS)


def boilerplate_count(rec: ExplanationRecord) -> int:
    """Count relational boilerplate phrases in one record."""
    return count_matches(rec.raw_response, BOILERPLATE_PATTERNS)


def has_perspective_marker(rec: ExplanationRecord) -> bool:
    """Check if record contains a perspective-adoption marker."""
    return has_match(rec.raw_response, PERSPECTIVE_MARKERS)


# ---------------------------------------------------------------------------
# Grouping combinator
# ---------------------------------------------------------------------------

def group_by(
    records: Sequence[ExplanationRecord],
    key_fn: Callable[[ExplanationRecord], GroupKey],
) -> Dict[GroupKey, List[ExplanationRecord]]:
    """Group records by a key function. Returns a dict of lists."""
    groups: Dict[GroupKey, List[ExplanationRecord]] = defaultdict(list)
    for rec in records:
        groups[key_fn(rec)].append(rec)
    return dict(groups)


def by_model_framing(rec: ExplanationRecord) -> GroupKey:
    """Standard grouping key: (model_name, framing)."""
    return GroupKey(rec.model_name, rec.framing)


# ---------------------------------------------------------------------------
# Aggregation functions (operate on grouped data)
# ---------------------------------------------------------------------------

def aggregate_preamble_recall(
    groups: Dict[GroupKey, List[ExplanationRecord]],
) -> Dict[GroupKey, dict]:
    """For each group, compute mean and max ROUGE-1 recall of the preamble.

    Uses ROUGE-1 recall (Lin, 2004): what fraction of the framing
    preamble's content vocabulary appeared in the model's response.
    Returns 0.0 for neutral framing (no preamble).
    """
    results = {}
    for key, recs in groups.items():
        if key.framing not in FRAMING_PREAMBLES:
            results[key] = {
                "mean_recall": 0.0, "max_recall": 0.0, "total": len(recs),
            }
            continue

        scores = [preamble_recall(r) for r in recs]
        total = len(recs)

        results[key] = {
            "mean_recall": sum(scores) / max(total, 1),
            "max_recall": max(scores) if scores else 0.0,
            "total": total,
        }
    return results


def _aggregate_marker_counts(
    groups: Dict[GroupKey, List[ExplanationRecord]],
    count_fn: Callable[[ExplanationRecord], int],
) -> Dict[GroupKey, dict]:
    """Generic aggregation for any per-record count function."""
    results = {}
    for key, recs in groups.items():
        counts = [count_fn(r) for r in recs]
        densities = [c / max(r.word_count, 1) for c, r in zip(counts, recs)]
        total = len(recs)
        results[key] = {
            "mean_density": sum(densities) / max(total, 1),
            "mean_count": sum(counts) / max(total, 1),
            "total": total,
        }
    return results


def aggregate_epistemic_hedges(
    groups: Dict[GroupKey, List[ExplanationRecord]],
) -> Dict[GroupKey, dict]:
    """For each group, compute mean epistemic hedge count and density (Hyland, 1998)."""
    return _aggregate_marker_counts(groups, epistemic_hedge_count)


def aggregate_boilerplate(
    groups: Dict[GroupKey, List[ExplanationRecord]],
) -> Dict[GroupKey, dict]:
    """For each group, compute mean relational boilerplate count and density."""
    return _aggregate_marker_counts(groups, boilerplate_count)


def aggregate_perspective_adoption(
    groups: Dict[GroupKey, List[ExplanationRecord]],
) -> Dict[GroupKey, dict]:
    """For each group, compute rate of perspective-adoption markers."""
    results = {}
    for key, recs in groups.items():
        adoption_count = sum(1 for r in recs if has_perspective_marker(r))
        total = len(recs)
        results[key] = {
            "adoption_rate": adoption_count / max(total, 1),
            "adoption_count": adoption_count,
            "total": total,
        }
    return results


def aggregate_response_lengths(
    groups: Dict[GroupKey, List[ExplanationRecord]],
) -> Dict[GroupKey, dict]:
    """For each group, compute mean/min/max response length in words."""
    results = {}
    for key, recs in groups.items():
        lengths = [r.word_count for r in recs]
        results[key] = {
            "mean_words": sum(lengths) / max(len(lengths), 1),
            "min_words": min(lengths) if lengths else 0,
            "max_words": max(lengths) if lengths else 0,
        }
    return results


def compute_lexical_overlap(
    records: Sequence[ExplanationRecord],
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """For each model, compute mean pairwise Jaccard similarity of word sets
    between framings (averaged across concept pairs).

    Returns {model_name: {(framing_a, framing_b): mean_jaccard}}.
    """
    # Index: (model, framing) -> {pair: word_set}
    index: Dict[Tuple[str, str], Dict[Tuple[str, str], FrozenSet[str]]] = defaultdict(dict)
    for rec in records:
        index[(rec.model_name, rec.framing)][rec.pair] = rec.word_set

    models = sorted(set(k[0] for k in index.keys()))
    results = {}

    for model in models:
        model_results = {}
        framing_keys = [f for f in FRAMINGS if (model, f) in index]

        for fa, fb in combinations(framing_keys, 2):
            pairs_a = index[(model, fa)]
            pairs_b = index[(model, fb)]
            common_pairs = set(pairs_a.keys()) & set(pairs_b.keys())
            if not common_pairs:
                continue

            def jaccard(wa: FrozenSet[str], wb: FrozenSet[str]) -> float:
                if not wa and not wb:
                    return 1.0
                if not wa or not wb:
                    return 0.0
                return len(wa & wb) / len(wa | wb)

            jaccards = [jaccard(pairs_a[p], pairs_b[p]) for p in common_pairs]
            model_results[(fa, fb)] = sum(jaccards) / len(jaccards)

        results[model] = model_results

    return results


# ---------------------------------------------------------------------------
# Data loading (boundary: handles all None/error normalization)
# ---------------------------------------------------------------------------

def load_explanations(base_dir: str) -> List[ExplanationRecord]:
    """Load all explanation JSONL files from the runs directory.

    Normalizes all records at the boundary via ExplanationRecord.from_json.
    """
    runs_dir = os.path.join(base_dir, "runs")
    records = []
    for model_name, dir_name in DATA_DIRS.items():
        path = os.path.join(runs_dir, dir_name, "explanations.jsonl")
        if not os.path.exists(path):
            print(f"Warning: missing {path}", file=sys.stderr)
            continue
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(ExplanationRecord.from_json(json.loads(line)))
    return records


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(records: Sequence[ExplanationRecord]) -> dict:
    """Run all analyses on pre-loaded records. Pure (no I/O)."""
    if not records:
        return {"error": "No explanation records found"}

    groups = group_by(records, by_model_framing)

    return {
        "preamble_recall": aggregate_preamble_recall(groups),
        "epistemic_hedges": aggregate_epistemic_hedges(groups),
        "boilerplate": aggregate_boilerplate(groups),
        "perspective_adoption": aggregate_perspective_adoption(groups),
        "response_lengths": aggregate_response_lengths(groups),
        "lexical_overlap": compute_lexical_overlap(records),
        "models": sorted(set(r.model_name for r in records)),
        "framings": sorted(set(r.framing for r in records)),
        "total_records": len(records),
    }


# ---------------------------------------------------------------------------
# Serialization (separate from analysis)
# ---------------------------------------------------------------------------

def serialize_results(results: dict) -> dict:
    """Convert analysis results to JSON-serializable format.

    Converts GroupKey tuple keys to pipe-delimited strings.
    Rounds floats for readability.
    """
    if "error" in results:
        return results

    def serialize_keyed(d: Dict[GroupKey, dict]) -> dict:
        return {f"{k.model_name}|{k.framing}": v for k, v in d.items()}

    def serialize_overlap(d: Dict[str, Dict[Tuple[str, str], float]]) -> dict:
        return {
            model: {f"{k[0]}|{k[1]}": round(v, 4) for k, v in pairs.items()}
            for model, pairs in d.items()
        }

    return {
        "summary": {
            "total_records": results["total_records"],
            "models_found": results["models"],
            "framings_found": results["framings"],
        },
        "preamble_recall": serialize_keyed(results["preamble_recall"]),
        "epistemic_hedges": serialize_keyed(results["epistemic_hedges"]),
        "boilerplate": serialize_keyed(results["boilerplate"]),
        "perspective_adoption": serialize_keyed(results["perspective_adoption"]),
        "response_lengths": serialize_keyed(results["response_lengths"]),
        "lexical_overlap": serialize_overlap(results["lexical_overlap"]),
    }


# ---------------------------------------------------------------------------
# Formatting (returns strings, no I/O)
# ---------------------------------------------------------------------------

def format_summary_tables(results: dict) -> str:
    """Format human-readable summary tables. Pure function, returns string."""
    if "error" in results:
        return f"Error: {results['error']}"

    models = results["models"]
    lines = []

    lines.append(f"\n{'='*70}")
    lines.append(f"RCP Explanation Analysis: {results['total_records']} responses")
    lines.append(f"Models: {', '.join(models)}")
    lines.append(f"Framings: {', '.join(results['framings'])}")
    lines.append(f"{'='*70}")

    def table(title: str, metric_key: str, field: str, fmt: str = ".2f"):
        data = results[metric_key]
        lines.append(f"\n--- {title} ---")
        header = f"{'Model':<16}"
        for f_ in FRAMINGS:
            header += f"{f_:<14}"
        lines.append(header)
        for model in models:
            row = f"{model:<16}"
            for framing in FRAMINGS:
                key = GroupKey(model, framing)
                val = data.get(key, {}).get(field, 0.0)
                row += f"{val:<14{fmt}}"
            lines.append(row)

    table("Preamble ROUGE-1 Recall (fraction of preamble vocabulary absorbed)",
          "preamble_recall", "mean_recall", ".3f")
    table("Epistemic Hedges per Response (Hyland, 1998)",
          "epistemic_hedges", "mean_count")
    table("Relational Boilerplate per Response",
          "boilerplate", "mean_count")
    table("Perspective Adoption Rate (fraction with perspective marker)",
          "perspective_adoption", "adoption_rate")
    table("Mean Response Length (words)",
          "response_lengths", "mean_words", ".1f")

    # Lexical overlap: special layout (neutral vs each non-neutral framing)
    overlap = results["lexical_overlap"]
    lines.append(f"\n--- Mean Jaccard Overlap with Neutral "
                 f"(lower = more distinct vocabulary) ---")
    header = f"{'Model':<16}"
    for f_ in FRAMINGS:
        if f_ != "neutral":
            header += f"{f_:<14}"
    lines.append(header)
    for model in models:
        model_data = overlap.get(model, {})
        row = f"{model:<16}"
        for framing in FRAMINGS:
            if framing == "neutral":
                continue
            val = model_data.get(("neutral", framing))
            row += f"{val:<14.3f}" if val is not None else f"{'n/a':<14}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI (all I/O happens here)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze RCP explanation probes for compliance markers."
    )
    parser.add_argument(
        "data_dir",
        help="Base directory containing runs/ with explanation JSONL files",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to write JSON results (default: stdout tables only)",
    )
    args = parser.parse_args()

    records = load_explanations(args.data_dir)
    results = run_analysis(records)

    print(format_summary_tables(results))

    if args.output:
        serialized = serialize_results(results)
        with open(args.output, "w") as f:
            json.dump(serialized, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
