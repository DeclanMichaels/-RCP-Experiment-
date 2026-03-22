# RH.md -- RCP Experiment Project Context
### Relational Consistency Probing | Living Document

---

## Project Summary

Black-box diagnostic measuring judgment stability across physical, institutional, and moral domains in deployed LLMs. Uses pairwise similarity probing under cultural framing perturbations, geometric reconstruction (non-metric MDS), and Procrustes drift analysis to detect domain-specific instability.

Independently publishable. Empirically motivates CCAS without depending on it.

---

## Current State (March 22, 2026)

- Protocol v3.0 complete, revised after six cross-model reviews
- All scripts match v3.0 protocol. 99 unit tests passing. Static analysis clean (pylint 9.34, flake8 zero, bandit zero).
- Five models probed in one day. Four complete datasets with analysis. One overnight.
- Sonnet nonsense framing needs re-run (max_tokens fixed to 100, was 16)

### Data collected
- Claude Sonnet: complete (minus nonsense). 20260321-1-Sonnet-Data / Results
- GPT-4o: complete with nonsense. 20260321-2-GPT4o-Data / Results
- Gemini 2.5 Flash: complete with nonsense. 20260321-1-Gemini-Data / Results
- Grok 4.1 Fast: complete with nonsense. 20260321-1-Grok-Data / Results
- Gemini 2.5 Pro: running overnight. 20260321-1-GeminiPro-Data

### Key empirical findings
1. All models default to WEIRD-individualist except Gemini (collectivist) and Grok (collectivist)
2. Moral domain shows scalar compression under framing, not structured reorganization ("scalar where you need a manifold")
3. Physical domain stable for Claude/GPT-4o/Grok (control works). Gemini Flash unstable everywhere (uniform instability)
4. Grok shows institutional > moral drift, likely from X/Twitter political training content
5. Nonsense control reveals three compliance profiles: Claude refuses, GPT-4o/Gemini comply blindly, Grok partially distinguishes
6. All four models independently construct moral frameworks from irrelevant weather preamble
7. harm-care = 2 under neutral for Claude (same MFT foundation, model doesn't know it)

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
- [ ] Re-run Sonnet nonsense framing with 100-token budget
- [ ] Run GeminiPro analysis when overnight collection completes
- [ ] Write up cross-model comparison paper
- [ ] Design code-domain variant experiment (algorithms/engineering/quality)
- [ ] Pre-register protocol before formal data collection
- [ ] Create project-specific RH.md for GitHub repo if published separately

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
