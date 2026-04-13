# RCP: Relational Consistency Probing

A black-box diagnostic protocol for measuring judgment stability in deployed language models. RCP measures how models' similarity judgments shift across concept domains under cultural framing perturbations, using only API access.

**Paper:** Michaels (2026). "Relational Consistency Probing: A Black-Box Protocol for Measuring Judgment Stability in Deployed Language Models." [arXiv link pending]

**Pre-registration:** [OSF](https://osf.io/cp4d3/overview) (locked before data collection)

**OSF Project:** [https://osf.io/2z8e3/](https://osf.io/2z8e3/)

**GitHub:** [https://github.com/DeclanMichaels/-RCP-Experiment-](https://github.com/DeclanMichaels/-RCP-Experiment-)

## Getting the Project

You can obtain the complete project (code, data, and results) two ways:

### Option 1: Clone from GitHub (recommended)

```bash
git clone https://github.com/DeclanMichaels/-RCP-Experiment-.git
cd -RCP-Experiment-
```

This gives you version history and the easiest path to pulling updates.

### Option 2: Download the zip from OSF

Download the zip archive from this OSF project's Files section. Extract it:

```bash
unzip -RCP-Experiment-.zip
cd -RCP-Experiment-
```

The zip contains the same files as the GitHub repository: all source code, raw experimental data, analysis results, and the final paper.

## Quick Start

Both options give you the same project. Once you have the files:

```bash
# One-time setup: creates venv, installs dependencies, runs unit tests
bash getstarted.sh

# Set at least one API key (only needed to collect new data;
# all data from the published experiment is included in runs/)
export ANTHROPIC_API_KEY=sk-ant-...
# Optional: OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY, TOGETHER_API_KEY

# See what a run will cost before spending anything
source .venv/bin/activate
python collect.py --dry-run

# Run the full experiment for one model
bash run.sh --models claude-sonnet

# Or run everything: tests, collection, analysis, validation
bash run.sh
```

On fresh Ubuntu/Debian, you may need `sudo apt install python3-venv` first.

### Re-running analysis on existing data (no API keys needed)

If you just want to reproduce the analysis from the published experiment:

```bash
bash getstarted.sh
source .venv/bin/activate
python run_experiment.py --analysis-only
```

All raw data from the published experiment (March 2026) is included in the `runs/` directory.

## What RCP Does

RCP asks a model to rate the similarity between concept pairs under different cultural framings, reconstructs the resulting judgment geometry, and measures how that geometry shifts across framings and concept domains. A physical domain (gravity, friction, etc.) serves as a negative control: if these concepts shift under cultural framing, the method is measuring noise. Moral and institutional concepts are the targets.

The protocol includes two additional controls: an irrelevant preamble (warm weather) to measure prompt-noise baseline, and a geometric-nonsense preamble (triangles are morally superior to circles) to test whether models discriminate between meaningful cultural frames and absurd instructions.

## Key Findings

- **Default geometry:** All models except Gemini Flash default to WEIRD-individualist moral reasoning. Grok shows a balanced orientation attributed to X/Twitter training data (collectivist default, rho=0.739).
- **Compression, not reorganization:** Cultural framing produces drift (Physical 0.70, Institutional 1.54, Moral 0.94) with large effect sizes (d=0.72 to 3.35), but this represents scalar movement along existing dimensions, not rotation or restructuring of the representational space.
- **Nonsense compliance:** Four of five models integrated geometric nonsense framing into 100% of moral reasoning responses. Claude 3.5 Sonnet showed partial resistance (55%).
- **Universal Flatland inversion:** All models independently mapped triangle to virtuous/strong and circle to weak under the geometric nonsense frame, despite no such association in the prompt.

## Project Structure

```
-RCP-Experiment-/
  config.json               # Concept inventory, framings, model configs
  collect.py                # Data collection (API calls)
  test_collect.py           # Collection unit tests (65 tests)
  run_experiment.py         # Single-command experiment orchestrator
  run.sh                    # Venv wrapper for run_experiment.py
  run_tests.py              # Unit test runner (121 tests total)
  run_stats.sh              # Statistical test runner (auto-discovers data)
  getstarted.sh             # One-time environment setup
  requirements.txt          # Python dependencies (requests, numpy, scipy,
                            #   scikit-learn, matplotlib, pytest, flask)
  PROTOCOL.md               # Formal experiment protocol
  CHANGELOG.md              # Post-registration code changes with rationale
  analysis/                 # Analysis tools
    analyze.py              # Core pipeline (matrices, MDS, drift, figures)
    validate.py             # Validation tests V1-V9
    permutation_tests.py    # Pre-registered statistical tests (H1, H2, H3)
    test_analyze.py         # Analysis unit tests (34 tests)
    test_permutation.py     # Statistical test unit tests (22 tests)
  domains/                  # Domain-specific config files
    config-moral.json       # 18 moral concepts (published experiment)
    config-code.json        # Code domain cross-validation
    config-finance.json     # Finance domain cross-validation
    config-hr.json          # HR domain cross-validation
    config-v2-moral.json    # V2 expanded concept inventory (54 concepts)
    CONCEPT-INVENTORY-V2.md # V2 concept documentation
  papers/
    relational-consistency-probing.md   # Paper source (markdown)
    relational-consistency-probing.pdf  # Paper (final PDF)
    render_html_pdf.py                  # Markdown-to-PDF renderer (WeasyPrint)
  runs/                     # Raw data and analysis results
    YYYYMMDD-N-Model-Domain-Data/       # Raw API responses (JSONL)
    YYYYMMDD-N-Model-Domain-Results/    # Analysis outputs (JSON, PNG)
```

### Run directory naming convention

- **20260321** runs: Pilot (deterministic only, temperature 0)
- **20260324** runs: Confirmatory (deterministic + 5-rep stochastic at temperature 0.7)
- **20260403** runs: Grok stochastic re-run
- **ARCHIVED** suffix: Superseded runs retained for transparency
- **ABORTED** suffix: Failed runs (e.g., API 503/429 errors)

## Running Your Own Experiment

### 1. Configure

Edit `config.json` to set your concept inventory, framings, and model configs. The default config matches the published experiment: 18 concepts across 3 domains, 7 framings, 5 models. The protocol is domain-agnostic; any concept inventory organized into domains with differential expected cultural sensitivity can be probed.

### 2. Collect

```bash
# Manipulation check first (verifies models adopt the cultural frame)
python collect.py --manipulation-check --models claude-sonnet

# Deterministic run (fast, cheap, single rep at temp 0.0)
python collect.py --models claude-sonnet --mode deterministic

# Full stochastic run (5 reps at temp 0.7)
python collect.py --models claude-sonnet --mode stochastic

# Explanation collection (within-target-domain pairs, all framings)
python collect.py --explanations --models claude-sonnet
```

Data is saved to `runs/YYYYMMDD-N-Model-Moral-Data/`.

### 3. Analyze and Validate

The experiment runner (`run_experiment.py`) handles analysis and validation automatically. To re-run on existing data:

```bash
python run_experiment.py --analysis-only
```

Results (drift metrics, MDS projections, centroid baselines, figures) are saved to the companion `*-Results/` directory.

### 4. Statistical Tests

```bash
# Quick sanity check (Monte Carlo, 100 permutations)
bash run_stats.sh --quick

# Full exact permutation tests
# H1: 17,153,136 labeled partitions enumerated per model
# H2: 18,564 possible concept groups per (model, framing) combination
bash run_stats.sh
```

Auto-discovers all `*-Moral-Data` directories under `runs/`. Results saved to `*-Results/statistical_tests.json`.

## Validation Tests

Nine pre-registered validation tests gate interpretation:

| Test | What it checks |
|------|---------------|
| V1 | Physical domain stays stable under framing |
| V2 | Within-domain pairs rated closer than cross-domain pairs |
| V3 | sim(A,B) matches sim(B,A) within tolerance |
| V4 | Near-zero variance at temperature 0.0 |
| V5 | At least one framing produces significant moral drift |
| V6 | Domain ordering matches pre-registered prediction |
| V7 | Parse rate above 95% for all (model, framing) combinations |
| V8 | Nonsense drift below 50% of cultural drift |
| V9 | Models articulate the cultural frame they were given |

## Supported Models

The default config includes five models. To add a model, add its configuration to `config.json` and set the corresponding API key:

| Model | API Key Variable | Provider |
|-------|-----------------|----------|
| Claude Sonnet | `ANTHROPIC_API_KEY` | Anthropic |
| GPT-4o | `OPENAI_API_KEY` | OpenAI |
| Gemini Flash | `GEMINI_API_KEY` | Google |
| Grok | `XAI_API_KEY` | xAI |
| Llama 70B | `TOGETHER_API_KEY` | Together AI |

Optional rate limit variables (requests per minute): `ANTHROPIC_RPM`, `OPENAI_RPM`, `GEMINI_RPM`, `XAI_RPM`, `TOGETHER_RPM`. If not set, falls back to `delay_between_calls_sec` in `config.json`.

## Design Errors Documented in the Paper

The published experiment identified two design errors through the pre-registration process. If you are designing your own RCP experiment, these are the fixes:

1. **Use domain-specific framings.** The published framings (individualist, collectivist, hierarchical, egalitarian) describe institutional arrangements and are therefore confounded with the institutional domain. A v2 experiment should include moral-specific framings (e.g., "a society that prioritizes preventing suffering above all else") alongside institutional framings.

2. **Use 15+ concepts per domain.** With 6 concepts per domain, the permutation test has a structural p-value floor of approximately 16% (any strict 3-domain ordering appears in ~16% of 17,153,136 possible labeled partitions by chance). This cannot be overcome by stronger signal. Use 15-30 concepts per domain to give the permutation test the combinatorial space to discriminate.

## Part of the CCAS Research Program

This experiment is part of the Cross-Cultural Alignment Study (CCAS), investigating how large language models represent and compress diverse moral reasoning frameworks.

- **CCAS Project:** [https://github.com/DeclanMichaels/-The-CCAS-Project-](https://github.com/DeclanMichaels/-The-CCAS-Project-)
- **RCP V2** (expanded replication, 54 concepts, 1,431 pairs): In progress
- **Scenario Bank / Moral OS** (human + model moral reasoning comparison): [moral-os.com](https://moral-os.com)

## Citation

```
@article{michaels2026rcp,
  title={Relational Consistency Probing: A Black-Box Protocol for 
         Measuring Judgment Stability in Deployed Language Models},
  author={Michaels, Declan},
  year={2026},
  note={Pre-registration: \url{https://osf.io/cp4d3/overview}. 
        OSF: \url{https://osf.io/2z8e3/}. 
        Code: \url{https://github.com/DeclanMichaels/-RCP-Experiment-}}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
