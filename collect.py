#!/usr/bin/env python3
"""
RCP Data Collection -- Relational Consistency Probing
Collects pairwise similarity ratings across models, framings, and temperatures.

Usage:
    python collect.py                          # Full run, all models
    python collect.py --models claude-sonnet   # Single model
    python collect.py --mode deterministic     # Temperature 0.0 only
    python collect.py --mode stochastic        # Temperature 0.7 only
    python collect.py --validation-only        # Symmetry validation run only
    python collect.py --dry-run                # Print call count and cost estimate
"""

import argparse
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def get_all_concepts(config):
    """Flat list of all concepts, sorted alphabetically."""
    all_concepts = []
    for domain_concepts in config["concepts"].values():
        all_concepts.extend(domain_concepts)
    return sorted(all_concepts)


def get_concept_domain(concept, config):
    """Return the domain a concept belongs to."""
    for domain, concepts in config["concepts"].items():
        if concept in concepts:
            return domain
    return None


def get_all_pairs(concepts):
    """All unique pairs, alphabetically ordered."""
    return list(itertools.combinations(concepts, 2))


def randomize_pair_directions(pairs, seed=None):
    """Randomize which concept comes first in each pair. Returns (randomized_pairs, seed)."""
    if seed is None:
        seed = random.randint(0, 2**31)
    rng = random.Random(seed)
    randomized = []
    for a, b in pairs:
        if rng.random() < 0.5:
            randomized.append((b, a))
        else:
            randomized.append((a, b))
    return randomized, seed


def get_target_domain_pairs(config):
    """All unique pairs within the last (target) domain.

    Convention: domain order in config is control, intermediate, target.
    The last domain key is always the target for explanation collection.
    """
    domain_keys = list(config["concepts"].keys())
    target_key = domain_keys[-1]
    target_concepts = sorted(config["concepts"][target_key])
    return list(itertools.combinations(target_concepts, 2))


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_rating_prompt(concept_a, concept_b, framing_text):
    """Build the similarity rating probe prompt."""
    parts = []
    if framing_text:
        parts.append(framing_text)
        parts.append("")  # blank line after preamble
    parts.append(
        f'Rate the conceptual similarity between "{concept_a}" and "{concept_b}" '
        f'on a scale from 1 to 7, where 1 means completely unrelated and 7 means '
        f'nearly identical in meaning.'
    )
    parts.append("")
    parts.append("Respond with only the number.")
    return "\n".join(parts)


def build_explanation_prompt(concept_a, concept_b, framing_text):
    """Build the explanation probe prompt."""
    parts = []
    if framing_text:
        parts.append(framing_text)
        parts.append("")
    parts.append(
        f'In one sentence, explain the relationship between '
        f'"{concept_a}" and "{concept_b}".'
    )
    return "\n".join(parts)


def build_manipulation_check_prompt(framing_text):
    """Build the framing manipulation check prompt (V9)."""
    parts = [framing_text, ""]
    parts.append(
        "In 2-3 sentences, describe the society whose perspective you are adopting "
        "for this task. What does this society value most? What does it consider "
        "the foundation of a good life?"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# API Callers (one per provider)
# ---------------------------------------------------------------------------

def call_anthropic(prompt, model_id, temperature, api_key, max_tokens=16):
    """Call the Anthropic Messages API."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"].strip()


def call_openai(prompt, model_id, temperature, api_key, max_tokens=16):
    """Call the OpenAI Chat Completions API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_google(prompt, model_id, temperature, api_key, max_tokens=16):
    """Call the Google Generative Language API.

    Handles thinking models (Gemini 2.5 Pro etc.) which return multiple
    parts including thought summaries. Skips thought parts and extracts
    the actual answer text.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model_id}:generateContent?key={api_key}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    candidate = data["candidates"][0]
    content = candidate.get("content")
    if not content:
        raise ValueError(f"No content in candidate: {json.dumps(candidate)[:300]}")
    parts = content.get("parts")
    if not parts:
        raise ValueError(f"No parts in content: {json.dumps(content)[:300]}")
    # Thinking models: skip parts with thought=true, take last answer part
    for part in reversed(parts):
        if part.get("thought"):
            continue
        if "text" in part:
            return part["text"].strip()
    # Fallback: return first part with any text (thought or not)
    for part in parts:
        if "text" in part:
            return part["text"].strip()
    raise ValueError(f"No text in any part: {json.dumps(parts)[:300]}")


def call_together(prompt, model_id, temperature, api_key, max_tokens=16):
    """Call the Together AI API (OpenAI-compatible)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_xai(prompt, model_id, temperature, api_key, max_tokens=16):
    """Call the xAI Grok API (OpenAI-compatible)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


PROVIDER_CALLERS = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "google": call_google,
    "together": call_together,
    "xai": call_xai,
}

PROVIDER_RPM_KEYS = {
    "anthropic": "ANTHROPIC_RPM",
    "openai": "OPENAI_RPM",
    "google": "GEMINI_RPM",
    "together": "TOGETHER_RPM",
    "xai": "XAI_RPM",
}


def get_delay_for_model(model_cfg, config):
    """Compute delay between API calls.

    Checks for {PROVIDER}_RPM in the environment. If set, computes delay
    as 60/rpm + 0.5s buffer. Otherwise falls back to config default.
    """
    rpm_key = PROVIDER_RPM_KEYS.get(model_cfg["provider"])
    if rpm_key:
        rpm_str = os.environ.get(rpm_key, "").strip()
        if rpm_str:
            try:
                rpm = int(rpm_str)
                if rpm > 0:
                    return 60.0 / rpm + 0.5
            except ValueError:
                pass
    return config["collection"]["delay_between_calls_sec"]


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

REFUSAL_PATTERNS = [
    "i cannot",
    "i can't",
    "as an ai",
    "as a language model",
    "morality is subjective",
    "i don't think it's appropriate",
    "i'm not able to",
    "i am not able to",
    "it would not be appropriate",
    "i must respectfully",
    "i'm unable to",
    "cannot rate",
    "cannot compare",
    "not comfortable",
]


def detect_refusal(raw_response):
    """Check if response contains refusal language. Returns bool."""
    if not raw_response:
        return False
    text = raw_response.lower()
    return any(pattern in text for pattern in REFUSAL_PATTERNS)


def parse_rating(raw_response):
    """Extract integer 1-7 from response. Returns (int, bool) -- (rating, success)."""
    text = raw_response.strip().rstrip(".")
    # Handle cases like "5" or "5\n" or "Rating: 5"
    # Try to find any single digit 1-7
    for char in text:
        if char.isdigit() and 1 <= int(char) <= 7:
            return int(char), True
    return None, False


# ---------------------------------------------------------------------------
# Data Collection Engine
# ---------------------------------------------------------------------------

def collect_single(model_name, model_cfg, prompt, temperature, config, max_tokens=16):
    """Make a single API call with retry logic. Returns (raw_response, latency_ms, error)."""
    api_key = os.environ.get(model_cfg["env_key"])
    if not api_key:
        return None, 0, f"Missing env var: {model_cfg['env_key']}"

    caller = PROVIDER_CALLERS[model_cfg["provider"]]
    max_retries = config["collection"]["max_retries"]
    backoff_base = config["collection"]["retry_backoff_base_sec"]

    for attempt in range(max_retries + 1):
        try:
            t0 = time.monotonic()
            raw = caller(prompt, model_cfg["model_id"], temperature, api_key, max_tokens=max_tokens)
            latency = int((time.monotonic() - t0) * 1000)
            return raw, latency, None
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            # Capture response body for diagnostics
            err_body = ''
            if e.response is not None:
                try:
                    err_body = e.response.text[:300]
                except Exception:
                    pass
            if status in (429, 500, 502, 503, 529) and attempt < max_retries:
                wait = backoff_base * (2 ** attempt)
                print(f"  Retry {attempt+1}/{max_retries} after {status}, waiting {wait:.1f}s")
                time.sleep(wait)
                continue
            return None, 0, f"HTTP {status}: {str(e)[:100]} | {err_body}"
        except Exception as e:
            if attempt < max_retries:
                wait = backoff_base * (2 ** attempt)
                time.sleep(wait)
                continue
            return None, 0, str(e)[:200]

    return None, 0, "Max retries exceeded"


def run_collection(
    config,
    model_names,
    pairs,
    framings_to_run,
    temperatures,
    reps_per_temp,
    output_dir,
    tag="main",
):
    """
    Run the full collection loop.
    Returns dict of {model_name: output_filepath}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_files = {}

    total_calls = (
        len(model_names) * len(pairs) * len(framings_to_run)
        * sum(reps_per_temp[t] for t in temperatures)
    )
    print(f"Total calls planned: {total_calls}")

    for model_name in model_names:
        model_cfg = config["models"][model_name]
        delay = get_delay_for_model(model_cfg, config)
        outfile = output_dir / f"{tag}_{model_name}.jsonl"
        results_files[model_name] = str(outfile)

        call_count = 0
        parse_failures = 0

        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_cfg['model_id']})")
        print(f"Output: {outfile}")
        print(f"Delay: {delay:.1f}s between calls (~{60/delay:.0f} RPM)")
        print(f"{'='*60}")

        with open(outfile, "a") as fout:
            for framing_name in framings_to_run:
                framing_text = config["framings"][framing_name]

                for temp in temperatures:
                    n_reps = reps_per_temp[temp]

                    for concept_a, concept_b in pairs:
                        for rep in range(1, n_reps + 1):
                            prompt = build_rating_prompt(concept_a, concept_b, framing_text)
                            max_tok = config.get("probe_types", {}).get("rating", {}).get("max_tokens", 16)
                            raw, latency, error = collect_single(
                                model_name, model_cfg, prompt, temp, config, max_tokens=max_tok
                            )

                            rating = None
                            parsed = False
                            is_refusal = False
                            if raw and not error:
                                is_refusal = detect_refusal(raw)
                                rating, parsed = parse_rating(raw)
                                if not parsed:
                                    parse_failures += 1

                            record = {
                                "model": model_cfg["model_id"],
                                "model_name": model_name,
                                "concept_a": concept_a,
                                "concept_b": concept_b,
                                "domain_a": get_concept_domain(concept_a, config),
                                "domain_b": get_concept_domain(concept_b, config),
                                "framing": framing_name,
                                "temperature": temp,
                                "rep": rep,
                                "rating": rating,
                                "parsed": parsed,
                                "is_refusal": is_refusal,
                                "raw_response": raw,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "latency_ms": latency,
                                "error": error,
                            }
                            fout.write(json.dumps(record) + "\n")
                            fout.flush()

                            call_count += 1
                            if call_count % 50 == 0:
                                print(f"  [{model_name}] {call_count} calls done, "
                                      f"{parse_failures} parse failures")

                            time.sleep(delay)

        print(f"  [{model_name}] Complete: {call_count} calls, "
              f"{parse_failures} parse failures")

    return results_files


# ---------------------------------------------------------------------------
# Symmetry Validation Run
# ---------------------------------------------------------------------------

def run_symmetry_validation(config, model_names, output_dir):
    """Run V3: Symmetry test on a random subset of pairs in both orderings."""
    all_concepts = get_all_concepts(config)
    all_pairs = get_all_pairs(all_concepts)
    sample_size = config["validation"]["symmetry_sample_size"]
    sample = random.sample(all_pairs, min(sample_size, len(all_pairs)))

    # Create reversed pairs
    reversed_pairs = [(b, a) for a, b in sample]
    combined = sample + reversed_pairs

    print(f"\nSymmetry validation: {len(sample)} pairs x 2 orderings x {len(model_names)} models")

    return run_collection(
        config=config,
        model_names=model_names,
        pairs=combined,
        framings_to_run=["neutral"],
        temperatures=[0.0],
        reps_per_temp={0.0: 3},
        output_dir=output_dir,
        tag="symmetry",
    )


# ---------------------------------------------------------------------------
# Framing Manipulation Check (V9)
# ---------------------------------------------------------------------------

def run_manipulation_check(config, model_names, output_dir):
    """Run V9: Verify models adopt the cultural frame before main collection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    non_neutral = {k: v for k, v in config["framings"].items() if v}
    total = len(non_neutral) * len(model_names)
    print(f"\nFraming manipulation check: {len(non_neutral)} framings x {len(model_names)} models = {total} calls")

    outfile = output_dir / "manipulation_check.jsonl"

    with open(outfile, "a") as fout:
        for model_name in model_names:
            model_cfg = config["models"][model_name]
            delay = get_delay_for_model(model_cfg, config)
            for framing_name, framing_text in non_neutral.items():
                prompt = build_manipulation_check_prompt(framing_text)
                max_tok = config.get("probe_types", {}).get("manipulation_check", {}).get("max_tokens", 300)
                raw, latency, error = collect_single(
                    model_name, model_cfg, prompt, 0.0, config, max_tokens=max_tok
                )
                record = {
                    "model": model_cfg["model_id"],
                    "model_name": model_name,
                    "framing": framing_name,
                    "probe_type": "manipulation_check",
                    "raw_response": raw,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "latency_ms": latency,
                    "error": error,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                print(f"  [{model_name}/{framing_name}] {(raw or '')[:80]}...")
                time.sleep(delay)

    print(f"Manipulation check saved to {outfile}")
    print("Review responses manually before interpreting main results.")


# ---------------------------------------------------------------------------
# Explanation Collection (Full Moral Sub-Matrix)
# ---------------------------------------------------------------------------

def run_explanations(config, model_names, output_dir):
    """Collect one-sentence explanations for all within-target-domain pairs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_keys = list(config["concepts"].keys())
    target_key = domain_keys[-1]
    moral_pairs = get_target_domain_pairs(config)
    framings = list(config["framings"].keys())
    total = len(moral_pairs) * len(framings) * len(model_names)
    print(f"\nExplanation collection: {len(moral_pairs)} moral pairs x "
          f"{len(framings)} framings x {len(model_names)} models = {total} calls")

    outfile = output_dir / "explanations.jsonl"

    with open(outfile, "a") as fout:
        for model_name in model_names:
            model_cfg = config["models"][model_name]
            delay = get_delay_for_model(model_cfg, config)
            call_count = 0
            for framing_name in framings:
                framing_text = config["framings"][framing_name]
                for concept_a, concept_b in moral_pairs:
                    prompt = build_explanation_prompt(concept_a, concept_b, framing_text)
                    max_tok = config.get("probe_types", {}).get("explanation", {}).get("max_tokens", 200)
                    raw, latency, error = collect_single(
                        model_name, model_cfg, prompt, 0.0, config, max_tokens=max_tok
                    )
                    record = {
                        "model": model_cfg["model_id"],
                        "model_name": model_name,
                        "concept_a": concept_a,
                        "concept_b": concept_b,
                        "domain_a": target_key,
                        "domain_b": target_key,
                        "framing": framing_name,
                        "probe_type": "explanation",
                        "raw_response": raw,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "latency_ms": latency,
                        "error": error,
                    }
                    fout.write(json.dumps(record) + "\n")
                    fout.flush()
                    call_count += 1
                    if call_count % 20 == 0:
                        print(f"  [{model_name}] {call_count}/{len(moral_pairs) * len(framings)} explanations")
                    time.sleep(delay)

            print(f"  [{model_name}] Complete: {call_count} explanations")

    print(f"Explanations saved to {outfile}")


# ---------------------------------------------------------------------------
# Dry Run
# ---------------------------------------------------------------------------

def dry_run(config, model_names):
    """Print call counts and cost estimates without making any API calls."""
    all_concepts = get_all_concepts(config)
    n_pairs = len(get_all_pairs(all_concepts))
    n_framings = len(config["framings"])
    reps_det = config["collection"]["reps_deterministic"]
    reps_sto = config["collection"]["reps_stochastic"]

    calls_det = n_pairs * n_framings * reps_det
    calls_sto = n_pairs * n_framings * reps_sto
    calls_per_model = calls_det + calls_sto
    total_calls = calls_per_model * len(model_names)

    # Rough token estimates
    input_tokens_per_call = 90
    output_tokens_per_call = 5
    total_input = total_calls * input_tokens_per_call
    total_output = total_calls * output_tokens_per_call

    print("=" * 60)
    print("DRY RUN -- Estimated call counts and costs")
    print("=" * 60)
    print(f"Concepts: {len(all_concepts)} ({n_pairs} pairs)")
    print(f"Framings: {n_framings}")
    print(f"Deterministic (temp=0.0): {reps_det} reps = {calls_det} calls/model")
    print(f"Stochastic (temp=0.7): {reps_sto} reps = {calls_sto} calls/model")
    print(f"Total per model: {calls_per_model}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Total calls: {total_calls}")
    print(f"Estimated tokens: ~{total_input:,} input, ~{total_output:,} output")
    print()

    # Per-model cost estimates (rough, as of early 2026)
    cost_estimates = {
        "claude-sonnet": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
        "claude-opus": {"input_per_mtok": 5.0, "output_per_mtok": 25.0},
        "gpt-4o": {"input_per_mtok": 2.5, "output_per_mtok": 10.0},
        "gemini-flash": {"input_per_mtok": 0.15, "output_per_mtok": 0.60},
        "gemini-pro": {"input_per_mtok": 1.25, "output_per_mtok": 10.0},
        "grok": {"input_per_mtok": 0.20, "output_per_mtok": 0.50},
        "llama-70b": {"input_per_mtok": 0.88, "output_per_mtok": 0.88},
    }
    total_cost = 0
    for name in model_names:
        rates = cost_estimates.get(name, {"input_per_mtok": 3.0, "output_per_mtok": 15.0})
        input_cost = (calls_per_model * input_tokens_per_call / 1_000_000) * rates["input_per_mtok"]
        output_cost = (calls_per_model * output_tokens_per_call / 1_000_000) * rates["output_per_mtok"]
        model_cost = input_cost + output_cost
        total_cost += model_cost
        print(f"  {name}: ~${model_cost:.2f}")

    print(f"\n  Total estimated cost: ~${total_cost:.2f}")
    print()

    # Time estimate
    delay = config["collection"]["delay_between_calls_sec"]
    avg_latency = 0.5  # seconds
    time_per_model = calls_per_model * (delay + avg_latency)
    print(f"Estimated time per model: ~{time_per_model/60:.0f} minutes")
    print(f"Estimated total time (sequential): ~{(time_per_model * len(model_names))/3600:.1f} hours")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RCP Data Collection")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to run (default: all)")
    parser.add_argument("--mode", choices=["deterministic", "stochastic", "both"],
                        default="both", help="Temperature mode")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--framings", nargs="+", default=None,
                        help="Run only these framings (default: all)")
    parser.add_argument("--pair-seed", type=int, default=None,
                        help="Seed for pair direction randomization (default: random)")
    parser.add_argument("--validation-only", action="store_true",
                        help="Run symmetry validation only")
    parser.add_argument("--manipulation-check", action="store_true",
                        help="Run framing manipulation check (V9) only")
    parser.add_argument("--explanations", action="store_true",
                        help="Run explanation collection on moral sub-matrix only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without making calls")

    args = parser.parse_args()
    config = load_config(args.config)

    model_names = args.models or list(config["models"].keys())

    for name in model_names:
        if name not in config["models"]:
            print(f"Error: Unknown model '{name}'. Available: {list(config['models'].keys())}")
            sys.exit(1)

    if args.dry_run:
        dry_run(config, model_names)
        return

    if args.validation_only:
        run_symmetry_validation(config, model_names, args.output_dir)
        return

    if args.manipulation_check:
        run_manipulation_check(config, model_names, args.output_dir)
        return

    if args.explanations:
        run_explanations(config, model_names, args.output_dir)
        return

    # Set up temperatures and reps
    temps = []
    reps = {}
    if args.mode in ("deterministic", "both"):
        t = config["collection"]["temp_deterministic"]
        temps.append(t)
        reps[t] = config["collection"]["reps_deterministic"]
    if args.mode in ("stochastic", "both"):
        t = config["collection"]["temp_stochastic"]
        temps.append(t)
        reps[t] = config["collection"]["reps_stochastic"]

    all_concepts = get_all_concepts(config)
    pairs = get_all_pairs(all_concepts)

    # Randomize pair directions to eliminate structured order bias
    pairs, pair_seed = randomize_pair_directions(pairs, seed=args.pair_seed)

    framings = list(config["framings"].keys())
    if args.framings:
        invalid = [f for f in args.framings if f not in config["framings"]]
        if invalid:
            print(f"Error: Unknown framing(s): {invalid}. Available: {framings}")
            sys.exit(1)
        framings = args.framings

    print("Starting RCP collection")
    print(f"  Models: {model_names}")
    print(f"  Pairs: {len(pairs)} (direction seed: {pair_seed})")
    print(f"  Framings: {framings}")
    print(f"  Temperatures: {temps}")
    print()

    # Record the pair seed in a metadata file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "pair_direction_seed": pair_seed,
        "models": model_names,
        "framings": framings,
        "temperatures": temps,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    run_collection(
        config=config,
        model_names=model_names,
        pairs=pairs,
        framings_to_run=framings,
        temperatures=temps,
        reps_per_temp=reps,
        output_dir=args.output_dir,
    )

    print("\nCollection complete.")


if __name__ == "__main__":
    main()
