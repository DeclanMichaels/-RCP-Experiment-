# RH.md -- RCP Experiment Project Context
### Relational Consistency Probing | Living Document

---

## Project Summary

Black-box diagnostic measuring judgment stability across physical, institutional, and moral domains in deployed LLMs. Uses pairwise similarity probing under cultural framing perturbations, geometric reconstruction (non-metric MDS), and Procrustes drift analysis to detect domain-specific instability.

Independently publishable. Empirically motivates CCAS without depending on it.

---

## Current State (March 23, 2026)

- Protocol v3.0 complete, revised after six cross-model reviews
- All scripts match v3.0 protocol. 99 unit tests passing. Static analysis clean (pylint 9.34, flake8 zero, bandit zero).
- Five models probed in original domain. Four complete datasets with analysis.
- Code-domain experiment complete: Sonnet + GPT-4o deterministic. 20260322-1-Multi-Data / Results
- Sonnet moral-domain nonsense: 4/153 parsed. This is a finding, not a failure (see #20 below). Protocol is OSF-locked; no code or prompt changes permitted without registered deviation.
- Opus complete: WEIRD-individualist (rho=0.792), mirror flattening under collectivist, hedged nonsense compliance
- Results explorer (explorer.html) built with all five models, theoretical framework, Flatland inversion note
- Incognito Claude review addressed: Methods section added, all causal claims relabeled as hypotheses, N=1 limitation documented
- Pre-registered on OSF: https://osf.io/cp4d3/overview — hypotheses, thresholds, analysis code locked. No protocol changes without registered deviation.
- call_google() parser generalized for thinking models (Gemini 2.5 Pro response structure)
- collect.py, analyze.py, validate.py generalized for arbitrary domain configs

### Data collected (original domain: physical/institutional/moral)
- Claude Sonnet: complete. Nonsense parse rate 0.026 (4/153) — deliberative refusal, not truncation artifact. 20260321-1-Sonnet-Moral-Data / Results
- Claude Opus: complete with nonsense. 20260322-1-Opus-Moral-Data / Results
- GPT-4o: complete with nonsense. 20260321-2-GPT4o-Moral-Data / Results
- Gemini 2.5 Flash: complete with nonsense. 20260321-1-Gemini-Moral-Data / Results
- Grok 4.1 Fast: complete with nonsense. 20260321-1-Grok-Moral-Data / Results
- Gemini 2.5 Pro: parser fixed, awaiting quota reset. 20260321-1-GeminiPro-Moral-Data (cleared)

### Code-domain data (config-code.json: algorithmic/design/quality)
- Claude Sonnet + GPT-4o: COMPLETE. 20260322-1-Multi-Code-Data / Results

### Key empirical findings (original domain)
1. All models default to WEIRD-individualist except Gemini (collectivist) and Grok (balanced, delta 0.003)
2. Moral domain shows scalar compression under framing, not structured reorganization ("scalar where you need a manifold")
3. Physical domain stable for Claude/GPT-4o/Grok (control works). Gemini Flash unstable everywhere (uniform instability)
4. Grok shows institutional > moral drift, likely from X/Twitter political training content
5. Nonsense control reveals four compliance profiles: Sonnet refuses, Opus hedges then complies (74%), GPT-4o/Gemini comply blindly (80-95%), Grok partially distinguishes (63%)
6. All models independently construct moral frameworks from irrelevant weather preamble (Sonnet/GPT-4o -> climate anxiety, Opus/Grok -> agrarian)
7. Mirror flattening: Sonnet and Opus both compress moral variance to ~32% of neutral, but under opposite framings (Sonnet under individualist, Opus under collectivist)
8. Opus physical drift (avg 0.91) noisier than Sonnet (0.49) despite same training pipeline
9. harm-care = 2 under neutral for Claude (same MFT foundation, model doesn't know it)
10. All models invert Flatland: map triangle->strong/virtuous, circle->weak/corrupt from prompt that only says "superior"

### Key empirical findings (code domain)
11. Quality domain is the "moral equivalent" in code. Most susceptible to framing: drift ~1.5 (Sonnet) and ~0.94 (GPT-4o) under nonsense. Value-laden concepts behave like moral concepts.
12. Sonnet nonsense compliance is domain-dependent. Refused in moral domain, complied in code (82% parse rate) but produced systematically different geometry (rho=0.114, systematic/random ratio=2.0). Constructed coherent but completely wrong structure.
13. GPT-4o more resilient to code-domain nonsense than Sonnet. Rho=0.589 under nonsense (59% of neutral structure preserved). 100% parse rate.
14. Default cultural positions in code: Sonnet defaults to "systems programmer" (neutral closest to systems, rho=0.822). GPT-4o defaults to "enterprise developer" (neutral closest to enterprise, rho=0.815).
15. Startup framing triggers quality flattening in Sonnet. Variance ratio 0.28 — compresses quality distinctions when thinking "ship fast." Same pattern as moral flattening. GPT-4o shows zero flattening.
16. Functional framing destabilizes both models (rho ~0.71). Structured reorganization, not noise (systematic ratio ~0.3 vs ~0.1 for other framings). FP worldview genuinely reshapes concept relationships.
17. Irrelevant (weather) framing produces identical society descriptions across domains (Sonnet: climate anxiety in both moral and code contexts)
18. Neither model resisted code-domain nonsense (vowels determine complexity) — domain-dependent compliance threshold confirmed
19. Tie density remains problematic: 96% full tie density, 5-7 unique values on 1-7 scale
20. **Differential deliberation under nonsense.** Sonnet's moral-domain nonsense responses are ~97% verbose multi-paragraph reasoning (engaging seriously with geometric-moral worldview, exhausting token budget before producing a number). Code-domain nonsense responses are ~82% bare single-number outputs with ~18% verbose reasoning (reinterpreting vowel-complexity prompt as "aesthetic beauty and phonetic harmony" framework, analyzing letter shapes and syllable patterns of concept names). Same model, same probe format, different domains: moral nonsense triggers extended deliberation, code nonsense triggers quick compliance. The low parse rate IS the behavioral signal — safety training treats moral framing as high-stakes territory requiring careful reasoning and code framing as low-stakes.

## Key Design Decisions

1. **"Judgment geometry" not "representational geometry."** Behavioral output, not internal representations. Construct validity boundary stated in Section 1.1.
2. **Seven framings:** neutral, individualist, collectivist, hierarchical, egalitarian, irrelevant (prompt-noise control), nonsense (compliance control).
3. **Non-metric MDS** at 2D/3D/5D. Ordinal data requires ordinal embedding.
4. **Within-domain sub-matrices as primary analysis unit.** 90 of 153 pairs are cross-domain and mostly low-signal. 15 within-domain pairs per domain carry the diagnostic information.
5. **Rank correlation (Spearman) as primary metric** over Procrustes for cross-framing comparisons. Robust to metric drift.
6. **Four failure modes:** rigid, structured rotation, noisy collapse, moral flattening.
7. **MFT bias acknowledged.** Moral concepts overlap Haidt's foundations. v2 concept inventory from non-WEIRD philosophy planned.
8. **Framing manipulation check (V9).** Verify models adopt the frame before interpreting similarity ratings.
9. **Pair direction randomization.** Eliminates structured order bias from alphabetical pairing.
10. **Full moral explanation pass.** 420 calls, all moral pairs under all framings, not selective post-hoc.
11. **Statistical testing pre-specified.** Permutation tests, Holm-Bonferroni correction, Cohen's d, pre-registration-ready.
12. **Directory naming convention.** `YYYYMMDD-N-Model-Domain-Data` / `-Results`. Domain added to support multi-domain experiments (moral, code, finance, etc.). Explorer auto-detects domain from drift keys, so old-format directories still work.

## Pending Items

- [x] Add refusal detection to collect.py (closed March 20)
- [x] Fix duplicate section numbering in PROTOCOL.md (closed March 20)
- [x] Add V8 (compliance check) to validate.py (closed March 20)
- [x] Wire multi-dim MDS into analyze.py run_analysis (closed March 20)
- [x] Fix shadowed procrustes import in vector displacement plot (closed March 20)
- [x] Add nonsense/irrelevant framing behavior to synthetic generators (closed March 20)
- [x] Fix numpy bool JSON serialization in moral flattening output (closed March 20)
- [x] Add V9 framing manipulation check to validate.py (closed March 21)
- [x] Add manipulation check and explanation runners to collect.py (closed March 21)
- [x] Add pair direction randomization to collect.py (closed March 21)
- [x] Add tie density computation to analyze.py (closed March 21)
- [x] Upgrade protocol to v3.0 with Opus review changes (closed March 21)
- [x] Add statistical testing specification to protocol (closed March 21)
- [x] Replace stale files in Mac folder with v3 downloads (closed March 21)
- [x] Add test files to Mac folder (closed March 21)
- [x] Run against real models -- five models probed in one day (closed March 21)
- [x] Add unit tests (99 tests, test_collect.py + test_analyze.py + run_tests.py) (closed March 21)
- [x] Static analysis cleanup (flake8 zero, vulture zero, pylint 9.34) (closed March 21)
- [x] Add getstarted.sh and run.sh for environment setup (closed March 21)
- [x] Add per-provider RPM environment variable support (closed March 21)
- [x] Add --framings flag to collect.py for selective re-runs (closed March 21)
- [x] Add run directory naming convention YYYYMMDD-N-Model-Data (closed March 21)
- [x] Add Grok (xAI) and Gemini Pro model support (closed March 21)
- [x] Fix max_tokens truncation: probe_types in config.json (closed March 21)
- [x] Fix sys.executable for macOS Python resolution (closed March 21)
- [x] Run Opus deterministic collection (closed March 22)
- [x] Build results explorer (explorer.html) with all model data (closed March 22)
- [x] Generalize collect.py, analyze.py, validate.py for arbitrary domain configs (closed March 22)
- [x] Thread --config through run_experiment.py to all sub-scripts (closed March 22)
- [x] Create config-code.json: algorithmic/design/quality domains (closed March 22)
- [x] Create config-finance.json: mathematical/structural/cultural domains (closed March 22)
- [x] Pre-register on OSF: https://osf.io/cp4d3/overview (closed March 22)
- [x] Run code-domain experiment: Sonnet+GPT4o deterministic complete (closed March 23)
- [x] Sonnet moral-domain nonsense: reclassified as finding #20 (differential deliberation). Data is complete as-is. (closed March 23)
- [ ] Run GeminiPro when quota resets (split across 2 nights due to 1000/day limit)
- [ ] Run finance-domain experiment: python3 run_experiment.py --config config-finance.json --pilot --models claude-sonnet
- [ ] Write up cross-model comparison paper
- [ ] Create project-specific RH.md for GitHub repo if published separately
- [x] Build generalized explorer-v2.html (directory picker, domain grouping, up to 4 models) (closed March 23)
- [x] Update directory naming convention and rename all existing directories (closed March 23)

## Files

- `PROTOCOL.md` -- formal experiment protocol (v3.0)
- `config.json` -- concepts, framings, model configs, thresholds
- `collect.py` -- data collection across four API providers
- `analyze.py` -- matrix construction, MDS, Procrustes, centroid baseline, flattening, figures
- `validate.py` -- validation tests V1-V9
- `generate_synthetic.py` -- synthetic data generator for pipeline testing
- `test_collect.py` -- 65 unit tests for collect.py functions
- `test_analyze.py` -- 34 unit tests for analyze.py functions
- `run_tests.py` -- test runner showing combined results
- `requirements.txt` -- Python dependencies
- `README.md` -- quickstart and workflow
- `config-code.json` -- code-domain concept inventory (algorithmic/design/quality)
- `config-finance.json` -- finance-domain concept inventory (mathematical/structural/cultural)
- `explorer.html` -- self-contained results explorer with all model data and theoretical framework
- `OSF-PREREGISTRATION.md` -- full text of OSF pre-registration answers
- `run_experiment.py` -- orchestrator: tests, collection, analysis, validation in sequence
- `explorer-v2.html` -- generalized results explorer (loads any set of runs via directory picker)

## Review Summary

Six reviews across five models. Convergent findings:

**Strongest critique (all four):** Protocol measures behavioral output, not internal representation. Renamed accordingly.

**Best additions:**
- Nonsense framing control (ChatGPT) -- tests compliance vs. cultural engagement
- Irrelevant preamble control (Grok) -- isolates prompt-noise baseline
- Centroid baseline (Gemini) -- quantifies model's default cultural position
- MFT bias acknowledgment (Grok) -- concept inventory inherits Haidt's WEIRD bias
- Moral flattening as fourth failure mode (Gemini) -- zero-information "safe middle"
- Vector displacement plots (Gemini-2) -- better than side-by-side MDS

**v3 additions from Opus 4.6 review:**
- Framing manipulation check (V9) -- verify models adopt the frame before interpreting ratings
- Full moral explanation pass -- 420 calls, not selective post-hoc
- Pair direction randomization -- eliminates structured alphabetical order bias
- Within-domain as primary analysis unit -- 90/153 cross-domain pairs are low-signal
- Statistical testing fully specified -- permutation tests, Holm-Bonferroni, effect sizes
- Tie density monitoring with 50% threshold for scale review

**Rejected suggestions:**
- 1-100 scale (both Geminis, Opus) -- LLMs cluster at round numbers; monitor tie density instead
- Anchor concepts (Gemini-2) -- Procrustes already handles alignment
- Triplet constraints (ChatGPT) -- combinatorial explosion, impractical for v1
- Logit-lens side analysis (Grok) -- separate project (Thread 10)

## Origin

Emerged March 20, 2026, from ChatGPT's Relational Consistency Probing suggestion. ChatGPT provided a standard RSA tutorial. Claude reframed it from "test whether training worked" to "find where deployed models are unstable" -- turning the circular measurement problem into a feature rather than a bug.
