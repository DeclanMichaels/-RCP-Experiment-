#!/usr/bin/env python3
"""
RCP Experiment Runner -- Single script that does everything.

Detects available API keys, runs unit tests, collects data from available
models, analyzes results, and runs validation. Stops early if anything fails.

Usage:
    python run_experiment.py                    # Full run with available models
    python run_experiment.py --pilot            # Deterministic only, skip stochastic
    python run_experiment.py --skip-tests       # Skip unit tests (not recommended)
    python run_experiment.py --analysis-only    # Re-run analysis on existing data

Output:
    data/           -- raw JSONL, manipulation checks, explanations, metadata
    results/        -- drift metrics, figures, centroid baseline, tie density
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Run directory naming
# ---------------------------------------------------------------------------


def generate_run_dirs(models):
    """Generate data and results directory names.

    Convention: YYYYMMDD-N-Model-Data / YYYYMMDD-N-Model-Results
    where N auto-increments for same model on same day.
    Multi-model runs use 'Multi' as the model label.
    """
    today = datetime.now().strftime("%Y%m%d")
    MODEL_LABELS = {
        "claude-sonnet": "Sonnet",
        "claude-opus": "Opus",
        "gpt-4o": "GPT4o",
        "gemini-flash": "GeminiFlash",
        "gemini-pro": "GeminiPro",
        "grok": "Grok",
        "llama-70b": "Llama70B",
    }
    if len(models) == 1:
        model_label = MODEL_LABELS.get(models[0], models[0].capitalize())
    else:
        model_label = "Multi"

    # Find next run number
    run_num = 1
    while Path(f"{today}-{run_num}-{model_label}-Data").exists():
        run_num += 1

    data_dir = f"{today}-{run_num}-{model_label}-Data"
    results_dir = f"{today}-{run_num}-{model_label}-Results"
    return data_dir, results_dir


# ---------------------------------------------------------------------------
# API key detection
# ---------------------------------------------------------------------------

MODEL_KEYS = {
    "claude-sonnet": ("ANTHROPIC_API_KEY", "ANTHROPIC_RPM"),
    "claude-opus": ("ANTHROPIC_API_KEY", "ANTHROPIC_RPM"),
    "gpt-4o": ("OPENAI_API_KEY", "OPENAI_RPM"),
    "gemini-flash": ("GEMINI_API_KEY", "GEMINI_RPM"),
    "gemini-pro": ("GEMINI_API_KEY", "GEMINI_RPM"),
    "grok": ("XAI_API_KEY", "XAI_RPM"),
    "llama-70b": ("TOGETHER_API_KEY", "TOGETHER_RPM"),
}


def detect_available_models():
    """Check which API keys are set in the environment."""
    available = []
    missing = []
    for model, (api_key, rpm_key) in MODEL_KEYS.items():
        val = os.environ.get(api_key, "").strip()
        if val:
            rpm = os.environ.get(rpm_key, "").strip()
            rate_info = f" ({rpm} RPM)" if rpm else " (using config default)"
            available.append((model, rate_info))
        else:
            missing.append((model, api_key, rpm_key))
    return available, missing


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_step(label, cmd, stop_on_fail=True):
    """Run a subprocess step. Returns True if it succeeded."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"\n  *** FAILED: {label} ***")
        if stop_on_fail:
            print("  Stopping. Fix the issue and re-run.")
            sys.exit(1)
        return False
    return True


# Use whatever Python is running this script for all subprocess calls.
# Fixes macOS where 'python' and 'python3' may not both exist.
PYTHON = sys.executable


def run_unit_tests():
    """Step 1: Unit tests."""
    return run_step(
        "STEP 1: Unit Tests",
        [PYTHON, "run_tests.py"],
    )


def run_manipulation_check(models, data_dir, config):
    """Step 2: Framing manipulation check (V9)."""
    return run_step(
        f"STEP 2: Manipulation Check ({', '.join(models)})",
        [PYTHON, "collect.py", "--config", config, "--manipulation-check",
         "--models"] + models + ["--output-dir", data_dir],
    )


def run_deterministic_collection(models, data_dir, config):
    """Step 3: Deterministic rating probes (temp=0.0)."""
    return run_step(
        f"STEP 3: Deterministic Collection ({', '.join(models)})",
        [PYTHON, "collect.py", "--config", config, "--mode", "deterministic",
         "--models"] + models + ["--output-dir", data_dir],
    )


def run_stochastic_collection(models, data_dir, config):
    """Step 4: Stochastic rating probes (temp=0.7)."""
    return run_step(
        f"STEP 4: Stochastic Collection ({', '.join(models)})",
        [PYTHON, "collect.py", "--config", config, "--mode", "stochastic",
         "--models"] + models + ["--output-dir", data_dir],
    )


def run_explanations(models, data_dir, config):
    """Step 5: Explanation collection (full moral sub-matrix)."""
    return run_step(
        f"STEP 5: Explanation Collection ({', '.join(models)})",
        [PYTHON, "collect.py", "--config", config, "--explanations",
         "--models"] + models + ["--output-dir", data_dir],
    )


def run_analysis(data_dir, results_dir, config="config.json"):
    """Step 6: Analysis pipeline."""
    return run_step(
        "STEP 6: Analysis Pipeline",
        [PYTHON, "analysis/analyze.py", "--config", config, "--data-dir", data_dir, "--output-dir", results_dir],
    )


def run_validation(data_dir, config="config.json"):
    """Step 7: Validation tests V1-V9."""
    return run_step(
        "STEP 7: Validation Tests (V1-V9)",
        [PYTHON, "analysis/validate.py", "--config", config, "--data-dir", data_dir],
        stop_on_fail=False,  # Report but don't abort -- some failures are expected
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RCP Experiment Runner")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: deterministic only, skip stochastic")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip unit tests (not recommended)")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Re-run analysis and validation on existing data")
    parser.add_argument("--data-dir", default=None,
                        help="Override data directory (default: auto-generated)")
    parser.add_argument("--results-dir", default=None,
                        help="Override results directory (default: auto-generated)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override model selection (default: auto-detect from API keys)")
    parser.add_argument("--config", default="config.json",
                        help="Config file path (default: config.json)")
    args = parser.parse_args()

    start_time = time.monotonic()

    print("=" * 60)
    print("  RCP EXPERIMENT RUNNER")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    # Detect models
    if args.models:
        models = args.models
        print(f"\n  Models (manual override): {', '.join(models)}")
    else:
        available, missing = detect_available_models()
        models = [m for m, _ in available]
        if available:
            print("\n  Models available:")
            for model, rate_info in available:
                print(f"    {model}{rate_info}")
        if missing:
            print("  Models skipped (no API key):")
            for model, api_key, rpm_key in missing:
                print(f"    {model}: set {api_key} (optional: {rpm_key})")

    if not models:
        print("\n  ERROR: No API keys found. Set at least one:")
        for model, (api_key, rpm_key) in MODEL_KEYS.items():
            print(f"    export {api_key}=your-key-here    # enables {model}")
        sys.exit(1)

    # Generate run directories
    data_dir = args.data_dir
    results_dir = args.results_dir
    if not data_dir:
        data_dir, auto_results = generate_run_dirs(models)
        if not results_dir:
            results_dir = auto_results
    if not results_dir:
        results_dir = data_dir.replace("-Data", "-Results")

    # Analysis-only mode
    if args.analysis_only:
        if not args.data_dir:
            print("\n  ERROR: --analysis-only requires --data-dir.")
            print("  Example: --analysis-only --data-dir 20260321-1-Sonnet-Data")
            sys.exit(1)
        if not Path(data_dir).exists():
            print(f"\n  ERROR: Data directory '{data_dir}' not found.")
            sys.exit(1)
        print(f"  Data:    {data_dir}")
        print(f"  Results: {results_dir}")
        run_analysis(data_dir, results_dir, args.config)
        run_validation(data_dir, args.config)
        elapsed = time.monotonic() - start_time
        print(f"\n  Analysis complete in {elapsed:.0f}s")
        return

    # Full run
    mode = "PILOT (deterministic only)" if args.pilot else "FULL (deterministic + stochastic)"
    print(f"  Mode: {mode}")
    print(f"  Data:    {data_dir}")
    print(f"  Results: {results_dir}")

    # Step 1: Unit tests
    if not args.skip_tests:
        run_unit_tests()
    else:
        print("\n  Skipping unit tests (--skip-tests)")

    # Step 2: Manipulation check
    run_manipulation_check(models, data_dir, args.config)

    # Step 3: Deterministic collection
    run_deterministic_collection(models, data_dir, args.config)

    # Step 4: Stochastic collection (skip in pilot mode)
    if not args.pilot:
        run_stochastic_collection(models, data_dir, args.config)
    else:
        print("\n  Skipping stochastic collection (--pilot mode)")

    # Step 5: Explanations
    run_explanations(models, data_dir, args.config)

    # Step 6: Analysis
    run_analysis(data_dir, results_dir, args.config)

    # Step 7: Validation
    run_validation(data_dir, args.config)

    # Summary
    elapsed = time.monotonic() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Models:   {', '.join(models)}")
    print(f"  Mode:     {mode}")
    print(f"  Duration: {minutes}m {seconds}s")
    print(f"  Data:     {data_dir}")
    print(f"  Results:  {results_dir}")
    print()

    # List output files
    for dirname in [data_dir, results_dir]:
        d = Path(dirname)
        if d.exists():
            files = sorted(d.iterdir())
            print(f"  {dirname}/")
            for f in files:
                size = f.stat().st_size
                if size > 1024:
                    print(f"    {f.name}  ({size/1024:.1f} KB)")
                else:
                    print(f"    {f.name}  ({size} B)")
            print()


if __name__ == "__main__":
    main()
