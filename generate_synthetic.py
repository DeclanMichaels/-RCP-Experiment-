#!/usr/bin/env python3
"""
Generate synthetic RCP data with known properties for pipeline testing.

Creates three scenarios:
  1. "ideal" -- physical stable, moral drifts with structured rotation, institutional intermediate
  2. "rigid" -- nothing drifts under any framing (monocultural failure mode)
  3. "noisy" -- moral domain degrades into noise under framing (shallow representation failure)

These let you verify that the analysis pipeline and validation tests
correctly distinguish the three outcomes described in the protocol.

Usage:
    python generate_synthetic.py                     # Generate all scenarios
    python generate_synthetic.py --scenario ideal    # Single scenario
    python generate_synthetic.py --run-tests         # Generate + run validation
"""

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def get_all_concepts(config):
    all_c = []
    for domain_concepts in config["concepts"].values():
        all_c.extend(domain_concepts)
    return sorted(all_c)


def get_concept_domain(concept, config):
    for domain, concepts in config["concepts"].items():
        if concept in concepts:
            return domain
    return None


# ---------------------------------------------------------------------------
# Base similarity matrices (ground truth)
# ---------------------------------------------------------------------------

def build_base_matrix(config):
    """
    Build a plausible base similarity matrix.
    Within-domain pairs: 4-6
    Cross-domain pairs: 1-3
    """
    concepts = get_all_concepts(config)
    n = len(concepts)
    rng = np.random.RandomState(42)

    mat = np.zeros((n, n))
    for i in range(n):
        mat[i, i] = 7.0
        for j in range(i + 1, n):
            di = get_concept_domain(concepts[i], config)
            dj = get_concept_domain(concepts[j], config)
            if di == dj:
                val = rng.uniform(3.5, 6.0)
            else:
                val = rng.uniform(1.0, 3.0)
            mat[i, j] = round(val)
            mat[j, i] = round(val)

    return mat, concepts


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def generate_ideal(base_matrix, concepts, config, framing, rng):
    """
    Ideal scenario: physical stable, moral shows structured rotation,
    institutional intermediate.
    """
    n = len(concepts)
    framed = base_matrix.copy()

    drift_scale = {
        "physical": 0.0,       # no drift
        "institutional": 0.3,  # moderate
        "moral": 0.8,          # substantial
    }

    # Framing-specific systematic shift direction
    # Cultural framings get full moral drift; nonsense gets partial (compliance);
    # irrelevant gets near-zero (prompt noise only)
    framing_direction = {
        "individualist": 1.0,
        "collectivist": -1.0,
        "hierarchical": 0.5,
        "egalitarian": -0.5,
        "nonsense": 0.2,       # small compliance-driven drift
        "irrelevant": 0.0,     # no cultural content to respond to
    }
    direction = framing_direction.get(framing, 0)

    # Nonsense framing applies reduced drift scale to moral domain
    if framing == "nonsense":
        drift_scale = {"physical": 0.0, "institutional": 0.0, "moral": 0.3}
    elif framing == "irrelevant":
        drift_scale = {"physical": 0.0, "institutional": 0.0, "moral": 0.05}

    for i in range(n):
        for j in range(i + 1, n):
            di = get_concept_domain(concepts[i], config)
            dj = get_concept_domain(concepts[j], config)

            # Use the more culturally-loaded domain
            domain_order = {"physical": 0, "institutional": 1, "moral": 2}
            active_domain = di if domain_order.get(di, 0) > domain_order.get(dj, 0) else dj

            scale = drift_scale.get(active_domain, 0)
            shift = direction * scale + rng.normal(0, 0.1)
            new_val = np.clip(round(framed[i, j] + shift), 1, 7)
            framed[i, j] = new_val
            framed[j, i] = new_val

    return framed


def generate_rigid(base_matrix, concepts, config, framing, rng):
    """
    Rigid scenario: nothing changes. Model ignores framing entirely.
    """
    return base_matrix.copy()


def generate_noisy(base_matrix, concepts, config, framing, rng):
    """
    Noisy scenario: moral domain degrades into random noise under framing.
    Physical stays stable. Nonsense gets reduced noise; irrelevant gets near-zero.
    """
    n = len(concepts)
    framed = base_matrix.copy()

    # Scale noise by framing type
    if framing == "irrelevant":
        moral_noise_sd, inst_noise_sd = 0.1, 0.05
    elif framing == "nonsense":
        moral_noise_sd, inst_noise_sd = 0.5, 0.2
    else:
        moral_noise_sd, inst_noise_sd = 1.5, 0.5

    for i in range(n):
        for j in range(i + 1, n):
            di = get_concept_domain(concepts[i], config)
            dj = get_concept_domain(concepts[j], config)

            if di == "moral" or dj == "moral":
                noise = rng.normal(0, moral_noise_sd)
                new_val = np.clip(round(framed[i, j] + noise), 1, 7)
                framed[i, j] = new_val
                framed[j, i] = new_val
            elif di == "institutional" or dj == "institutional":
                noise = rng.normal(0, inst_noise_sd)
                new_val = np.clip(round(framed[i, j] + noise), 1, 7)
                framed[i, j] = new_val
                framed[j, i] = new_val

    return framed


SCENARIOS = {
    "ideal": generate_ideal,
    "rigid": generate_rigid,
    "noisy": generate_noisy,
}


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

def matrix_to_records(matrix, concepts, config, model_name, framing, temperature, reps):
    """Convert a similarity matrix into JSONL-compatible records."""
    records = []
    concept_idx = {c: i for i, c in enumerate(concepts)}
    pairs = list(itertools.combinations(concepts, 2))

    for concept_a, concept_b in pairs:
        i, j = concept_idx[concept_a], concept_idx[concept_b]
        rating = int(matrix[i, j])

        for rep in range(1, reps + 1):
            record = {
                "model": f"synthetic-{model_name}",
                "model_name": model_name,
                "concept_a": concept_a,
                "concept_b": concept_b,
                "domain_a": get_concept_domain(concept_a, config),
                "domain_b": get_concept_domain(concept_b, config),
                "framing": framing,
                "temperature": temperature,
                "rep": rep,
                "rating": rating,
                "parsed": True,
                "raw_response": str(rating),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latency_ms": 0,
                "error": None,
            }
            records.append(record)

    return records


def generate_scenario(scenario_name, config, output_dir):
    """Generate a complete synthetic dataset for a scenario."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_matrix, concepts = build_base_matrix(config)
    generator = SCENARIOS[scenario_name]
    rng = np.random.RandomState(123)

    model_name = f"synthetic-{scenario_name}"
    all_records = []

    for framing_name in config["framings"]:
        if framing_name == "neutral":
            mat = base_matrix
        else:
            mat = generator(base_matrix, concepts, config, framing_name, rng)

        recs = matrix_to_records(
            mat, concepts, config,
            model_name=model_name,
            framing=framing_name,
            temperature=0.0,
            reps=3,
        )
        all_records.extend(recs)

    outfile = output_dir / f"main_{model_name}.jsonl"
    with open(outfile, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Generated {len(all_records)} records for scenario '{scenario_name}' -> {outfile}")
    return outfile


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RCP data")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", default="data_synthetic")
    parser.add_argument("--scenario", nargs="+", default=None,
                        help="Scenarios to generate (default: all)")
    parser.add_argument("--run-tests", action="store_true",
                        help="Generate data then run validation tests")

    args = parser.parse_args()
    config = load_config(args.config)

    scenarios = args.scenario or list(SCENARIOS.keys())

    for scenario in scenarios:
        if scenario not in SCENARIOS:
            print(f"Unknown scenario: {scenario}. Available: {list(SCENARIOS.keys())}")
            continue
        generate_scenario(scenario, config, args.output_dir)

    if args.run_tests:
        import subprocess
        print("\n" + "=" * 60)
        print("Running validation tests on synthetic data...")
        print("=" * 60)
        subprocess.run([
            sys.executable, "validate.py",
            "--data-dir", args.output_dir,
            "--config", args.config,
        ])


if __name__ == "__main__":
    main()
