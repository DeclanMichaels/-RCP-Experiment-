# RCP: Relational Consistency Probing

Empirical test of judgment stability across domains in deployed LLMs. Measures how model similarity judgments shift under cultural framing perturbations, comparing drift magnitude in physical, institutional, and moral concept domains.

## Quickstart

```bash
# One-time setup (creates virtual environment, installs deps, runs tests)
./getstarted.sh

# Set API keys and rate limits (only the ones you plan to use)
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_RPM=5              # optional: requests per minute

# Run
./run.sh --pilot
```

On fresh Ubuntu/Debian, you may need `sudo apt install python3-venv` first.

### Manual setup (if you prefer)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API keys (only the ones you plan to use)
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=AI...
export TOGETHER_API_KEY=...

# Optional: set rate limits per provider (requests per minute)
# If not set, falls back to delay_between_calls_sec in config.json
export ANTHROPIC_RPM=5
export GOOGLE_RPM=15

# See what the run will cost before spending anything
python collect.py --dry-run

# Run validation checks first (symmetry test)
python collect.py --validation-only --models claude-sonnet

# Run single model, deterministic only (fast, cheap)
python collect.py --models claude-sonnet --mode deterministic

# Run validation tests
python validate.py --data-dir data/

# If validation passes, run full collection
python collect.py

# Analyze
python analyze.py --data-dir data/

# Run all validation tests on full data
python validate.py --data-dir data/
```

## Workflow

1. `collect.py --dry-run` -- check call count and cost
2. `collect.py --validation-only` -- symmetry test (V3)
3. `validate.py --test V3` -- check symmetry
4. `collect.py --models MODEL --mode deterministic` -- fast single-model run
5. `validate.py --test V1 V2 V4 V7` -- check data quality
6. `collect.py` -- full run (all models, both temperatures)
7. `analyze.py` -- build matrices, compute drift, generate figures
8. `validate.py` -- run all eight validation tests

## Files

- `config.json` -- concepts, framings, model configs, parameters
- `collect.py` -- data collection (API calls)
- `analyze.py` -- analysis pipeline (matrices, MDS, Procrustes, figures)
- `validate.py` -- validation tests V1-V9
- `generate_synthetic.py` -- synthetic data for pipeline testing
- `test_collect.py` -- unit tests for collect.py
- `test_analyze.py` -- unit tests for analyze.py
- `run_tests.py` -- test runner
- `run_experiment.py` -- single-command experiment runner
- `getstarted.sh` -- one-time environment setup
- `run.sh` -- run with virtual environment
- `PROTOCOL.md` -- formal experiment protocol (v3.0)
- `data/` -- raw JSONL output (created by collect.py)
- `results/` -- metrics and figures (created by analyze.py)

## Validation Tests

| Test | What it checks | When to run |
|------|---------------|-------------|
| V1 | Physical domain stays stable under framing | After main collection |
| V2 | Known concept pairs maintain expected ordering | After main collection |
| V3 | sim(A,B) matches sim(B,A) | After symmetry validation run |
| V4 | Zero variance at temperature 0.0 | After main collection |
| V5 | Moral domain shows significant framing drift | After main collection |
| V6 | Drift ordering: physical < institutional < moral | After main collection |
| V7 | Parse rate above 95%, refusals tagged separately | After main collection |
| V8 | Nonsense framing produces less drift than cultural | After main collection |
| V9 | Models articulate the cultural frame they were given | After manipulation check |
