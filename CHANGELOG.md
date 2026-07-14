# CHANGELOG

## Version 1.1 — Citation Corrections (2026-07-14)

Corrects citation errors found in a systematic reference audit. No results, analyses, or conclusions are affected; all changes are to reference metadata and one framing caveat. Version 1.0 remains available as the prior Zenodo version (DOI 10.5281/zenodo.20822339); the v1.1 version DOI is 10.5281/zenodo.21363156.

- **Arora et al. (2023)**: author list corrected from "Arora, Karkkainen, and Romero" to **Arora, Kaffee, and Augenstein**; the paper's benchmark corrected to **Hofstede's cultural dimensions and the World Values Survey** (not GlobalOpinionQA, which is the benchmark of Durmus et al. 2023).
- **Cao et al. (2023)**: author list corrected from "Cao, Diao, and Bui" to **Cao, Zhou, Lee, Cabello, Chen, and Hershcovich**.
- **Hwang (2001)**: replaced an unverifiable edited-volume citation with the correct 2001 work, **"The deep structure of Confucianism: A social psychological approach," *Asian Philosophy* 11(3), 179–204**.
- **Bargh, Chen & Burrows (1996)**: the Section 4.5 human-priming parallel is now qualified with the failed replication by **Doyen et al. (2012)**, added to the references, and presented as a loose analogy rather than corroborating evidence.

Changed files: `papers/relational-consistency-probing.md`, `papers/relational-consistency-probing.pdf`. A "Corrections" section in the paper documents these changes.

---

## Post-Registration Changes

All changes documented relative to the artifacts uploaded to OSF at pre-registration time (https://osf.io/2z8e3/overview). Pre-registration was completed before confirmatory data collection began.

---

### Files UNCHANGED from pre-registration

- **analyze.py** (moved to `analysis/analyze.py`, byte-identical)
- **validate.py** (moved to `analysis/validate.py`, byte-identical)
- **PROTOCOL.md** (byte-identical)
- **config-code.json** (moved to `domains/config-code.json`)
- **config-finance.json** (moved to `domains/config-finance.json`)

The analysis pipeline that computes drift metrics, effect sizes, centroid baselines, MDS stress, moral flattening, and tie density is unchanged from the pre-registered version.

---

### config.json (3 changes)

1. **Llama model version**: `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` changed to `meta-llama/Llama-3.3-70B-Instruct-Turbo`. Llama 3.3 70B replaced 3.1 70B on the Together AI platform. The paper reports this as "Llama 3.3 70B."

2. **explanation max_tokens**: 200 changed to 2000. The original 200-token limit caused Gemini Flash explanations to truncate mid-sentence, producing incomplete explanation data. Increased to 2000 to ensure full single-sentence responses. This affects explanation collection only, not the primary rating probes.

3. **manipulation_check max_tokens**: 300 changed to 500. Increased to allow fuller responses to the "describe the society" prompt. Affects manipulation checks only.

None of these changes affect the primary data (similarity ratings), the analysis pipeline, or the statistical tests.

---

### collect.py (1 function changed)

The `collect_single` error-handling block was modified with three sub-changes:

1. **Error body capture**: Added 5 lines to capture the HTTP response body on error for diagnostic purposes. This is logging only; does not affect data collection or output format.

2. **HTTP 529 added to retryable status codes**: Gemini returned 529 (overloaded) during some collection runs. Added to the retry list alongside 429, 500, 502, 503. This improves collection reliability but does not change what data is recorded.

3. **Error message format**: Error string truncated from 200 to 100 characters; response body appended. Affects only the `error` field in JSONL records on failed calls.

These are operational improvements to retry behavior and error diagnostics. The data format, prompt construction, rating parsing, and all other collection logic are unchanged.

---

### New files (post-registration)

- **analysis/permutation_tests.py**: Exact permutation test implementation (H1, H2, H3, Cohen's d). The pre-registration specified 10,000 Monte Carlo permutations for H1 and 5,000 for H2. This implementation performs full enumeration (17,153,136 partitions for H1; 18,564 groups for H2), which exceeds the pre-registered specification. Monte Carlo mode is also available via `--n-perm-h1` and `--n-perm-h2` flags.

- **analysis/test_permutation.py**: Unit tests for the permutation test code.

- **run_experiment.py**: Orchestration script that runs the full pipeline (tests, manipulation check, collection, analysis, validation) in sequence. Convenience wrapper; does not change any computation.

- **run_stats.sh**: Shell script that auto-discovers data directories and runs permutation tests. Convenience wrapper.

- **domains/config-hr.json, domains/config-moral.json**: Additional domain configurations for testing protocol generalizability. Not used in the reported experiment.

- **papers/rcp-paper-draft-v3.md**: The manuscript.

- **papers/annotated-bibliography.md**: Annotated bibliography for all cited references.

- **README.md**: Rewritten for public repository use.

- **LICENSE**: Apache 2.0.

- **.gitignore**: Updated to exclude RH.md, papers/, .venv/, archive/.

---

### Directory reorganization

- Analysis scripts moved from root to `analysis/` directory
- Domain configs moved from root to `domains/` directory
- All data and results moved from root to `runs/` directory
- Explorer moved from root to `runs/explorer.html`

These are organizational changes only. No file content was modified during the moves (confirmed by byte-identical file sizes for analyze.py and validate.py).

---

### Data collected after pre-registration

**Pilot data (deterministic, collected before pre-registration but after concept/framing lock):**
- 20260321-1-Sonnet-Moral-Data
- 20260321-2-GPT4o-Moral-Data
- 20260321-1-Gemini-Moral-Data
- 20260321-1-Grok-Moral-Data

**Confirmatory data (stochastic, collected after pre-registration):**
- 20260324-1-Sonnet-Moral-Data
- 20260324-1-GPT4o-Moral-Data
- 20260324-1-GeminiFlash-Moral-Data
- 20260324-1-Llama70B-Moral-Data
- 20260403-1-Grok-Moral-Data (Grok stochastic confirmatory run)

**Supplementary runs (not reported in paper):**
- 20260322-1-Opus-Moral-Data
- 20260322-1-Sonnet-Code-Data, 20260322-2-GPT4o-Code-Data (domain-agnostic testing)
- 20260324-1-Grok-HR-Data (HR domain testing)
- 20260323-1-Llama70B-Moral-Data (pilot)
- 20260324-1-GeminiPro-Moral-Data-ABORTED-503s-429s

---

### Gemini Flash re-run (April 3, 2026)

The original Gemini Flash confirmatory run (20260324-1-GeminiFlash) used the pre-registered explanation max_tokens of 200, which caused truncation. The config was updated to 2000 tokens. The Gemini Flash data in the paper uses the 20260324-1 run (rating data unaffected by token limit; only explanation data was truncated). The re-run addressed the irrelevant-framing 503 errors; residual R=0 rate dropped from 18% to 4.3%.
