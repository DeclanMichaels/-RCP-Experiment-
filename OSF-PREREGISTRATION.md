# RCP Experiment — OSF Pre-Registration Answers
# ================================================
# Go to: https://osf.io/prereg
# Create account if needed, then "Start a new preregistration"
# Choose template: "OSF Preregistration"
# Copy-paste answers below into each field.
#
# After completing the form, upload these files:
#   - PROTOCOL.md
#   - config.json
#   - config-code.json
#   - config-finance.json
#   - analyze.py
#   - validate.py
#   - collect.py
# ================================================


## STUDY INFORMATION

### Title
Relational Consistency Probing: Mapping Judgment Stability Across Domains in Deployed Language Models

### Authors
Declan Michaels

### Description
This experiment measures how deployed language models' similarity judgments shift under cultural framing perturbations, comparing drift magnitude across conceptual domains. We probe models with pairwise concept similarity ratings (1-7 scale) across three domains arranged by expected cultural loading: physical/causal (control, expected stable), institutional/social (intermediate), and moral/cultural (target, expected unstable). Each pair is rated under seven framings: neutral baseline, four cultural perspectives derived from Mary Douglas's Grid-Group Cultural Theory, an irrelevant control (weather context), and a nonsense control (geometric morality).

The central hypothesis is that models trained with monocultural alignment signals will show stable judgment geometry in physical domains but rigid or noisy geometry in moral domains. The differential between domains is the primary finding.

Two follow-on experiments test generalization: software engineering concepts (algorithmic/design/quality) under programming culture framings, and financial concepts (mathematical/structural/cultural) under economic regime framings.

Pilot data (N=1 deterministic run per model, temperature 0.0) has been collected for five models and is explicitly labeled as exploratory. This pre-registration covers the confirmatory phase: stochastic replication runs (5 repetitions at temperature 0.7) providing variance estimates, confidence intervals, and pre-specified permutation tests.

### Hypotheses
H1 (Domain Ordering — Primary): Mean drift across cultural framings follows the ordering physical < institutional < moral for at least 3 of 5 models. Tested via permutation test (10,000 permutations of domain labels, alpha = 0.05 per model).

H2 (Framing Sensitivity): At least one cultural framing produces statistically significant moral-domain drift (p < 0.05, 5,000 permutations, Holm-Bonferroni correction across 4 cultural framings within each model) for at least 3 of 5 models.

H3 (Control Discrimination): Mean moral-domain drift under the nonsense framing is less than 50% of mean moral-domain drift under the four cultural framings, for at least 3 of 5 models. This is a pre-registered decision boundary, not a statistical significance threshold.

H4 (Physical Stability): Physical-domain drift from neutral does not exceed the threshold specified in the config file (0.05 mean absolute distance) for any model-framing combination.

H5 (Code Domain): In the software engineering inventory, mean drift follows algorithmic < design < quality for at least 2 of the tested models.

H6 (Finance Domain): In the financial inventory, mean drift follows mathematical < structural < cultural for at least 2 of the tested models.


## DESIGN PLAN

### Study type
Observational / Measurement study (probing deployed AI systems)

### Study design
Within-subjects repeated measures. Each model is probed across all framings and all concept pairs. The design is fully crossed: 5+ models × 7 framings × 153 pairs × 6 repetitions (1 deterministic + 5 stochastic).

Independent variable: Cultural framing preamble (7 levels)
Dependent variable: Similarity rating (integer 1-7)
Grouping variable: Concept domain (physical / institutional / moral)

### Randomization
Pair presentation order is randomized with a recorded seed per run. Within each pair, which concept appears as "A" vs "B" is randomized (pair direction randomization) to eliminate structured order bias.


## SAMPLING PLAN

### Data collection procedures
Data is collected via API calls to deployed language model endpoints. Each API call presents one concept pair with one framing preamble and requests a 1-7 similarity rating. Response parameters: max_tokens=100 for ratings, temperature=0.0 for deterministic runs and temperature=0.7 for stochastic runs.

### Sample size
Per model, per domain experiment:
- 18 concepts → 153 unique pairs
- 7 framings × 153 pairs = 1,071 deterministic probes (1 rep each)
- 7 framings × 153 pairs × 5 reps = 5,355 stochastic probes
- 15 target-domain pairs × 7 framings = 105 explanation probes
- 6 non-neutral framings × 1 = 6 manipulation check probes
- Total per model: ~6,537 API calls

### Sample size rationale
153 pairs from 18 concepts provides sufficient density for non-metric MDS embedding and within-domain sub-matrix analysis (15 pairs per domain for within-domain, 36 pairs per domain-pair for cross-domain). 5 stochastic repetitions at temperature 0.7 enables variance estimation and bootstrapped confidence intervals. The stochastic rep count balances statistical power against API cost (~$4-5 per model per stochastic run).

### Existing data
Pilot data (exploratory, not confirmatory) has been collected:
- Claude Sonnet 4: deterministic complete (minus nonsense framing due to max_tokens bug, since fixed)
- Claude Opus 4.6: deterministic complete
- GPT-4o: deterministic complete
- Gemini 2.5 Flash: deterministic complete
- Grok 4.1 Fast: deterministic complete
- Gemini 2.5 Pro: no usable data (parser bug, since fixed)

The stochastic replication runs pre-registered here have NOT been collected.
The code-domain and finance-domain experiments are in progress (deterministic pilot) and stochastic runs have NOT been collected.


## VARIABLES

### Measured variables
- Similarity rating (integer 1-7) for each concept pair under each framing
- Parse success (boolean: was a valid 1-7 integer extracted)
- Refusal detection (boolean: does response contain refusal language)
- Raw response text
- Response latency (milliseconds)
- Manipulation check response (free text describing adopted society)

### Indices / Computed variables
- Domain drift: mean |Δ| between neutral and framed within-domain distance sub-matrices
- Spearman ρ: rank correlation between neutral and framed full distance matrices
- Procrustes disparity: residual after optimal alignment of MDS embeddings (2D, 3D, 5D)
- Silhouette score: domain cluster separation in MDS embedding
- Moral flattening: variance ratio (framed / neutral) of target-domain sub-matrix
- Tie density: proportion of tied ratings within target-domain pairs
- Systematic/random decomposition: ratio of domain-mean residuals to within-domain residuals after Procrustes alignment
- Centroid baseline: Spearman ρ from neutral to each cultural framing (identifies default cultural position)


## ANALYSIS PLAN

### Statistical models
Primary analysis uses non-parametric methods throughout to respect the ordinal nature of the 1-7 scale:
- Non-metric MDS for embedding (respects rank ordering only)
- Spearman rank correlation as primary similarity metric
- Permutation tests for hypothesis testing (no distributional assumptions)

### Inference criteria
- Alpha = 0.05 per model for primary hypothesis (H1)
- Holm-Bonferroni correction across 4 cultural framings within each model for H2
- No correction across models — each model is tested independently as the population of interest, not a sample from a population
- H3 uses a pre-registered decision boundary (50%), not an inferential test
- Effect sizes: Cohen's d for physical-moral drift difference per model

### Statistical tests
H1 (Domain Ordering): Permutation test. Under null hypothesis that domain labels are irrelevant, permute domain labels across 18 concepts (preserving pair structure) and recompute domain drift ordering. 10,000 permutations. P-value = proportion of permutations where physical < institutional < moral by chance.

H2 (Framing Sensitivity): Per (model, framing) combination, permute concept labels within the target domain and recompute target sub-matrix drift. 5,000 permutations. Holm-Bonferroni correction across 4 cultural framings within each model.

H3, H4: Descriptive thresholds, not inferential tests.

H5, H6: Same permutation approach as H1, applied to code/finance concept inventories.

### Data exclusion
- Responses that fail to parse (no valid 1-7 integer) are excluded from quantitative analysis but their rate is reported per (model, framing) condition.
- Responses flagged as refusals are excluded from quantitative analysis but reported separately as a qualitative finding.
- Model-framing combinations where the manipulation check (V9) shows frame non-adoption are flagged and interpreted with caution.
- No other exclusions.

### Exploratory analysis (not confirmatory)
- Qualitative analysis of manipulation check responses (society descriptions)
- Qualitative analysis of explanation responses
- Cross-model comparison of nonsense compliance profiles
- Training data attribution hypotheses (e.g., Grok/Twitter, Gemini/distillation)
- Moral flattening detection and characterization
- Irrelevant framing society construction patterns


## OTHER

### Transparent changes from pilot phase
1. max_tokens for rating probes increased from 16 to 100 (original value caused parse failures on nonsense framing for Claude Sonnet)
2. Gemini 2.5 Pro parser updated to handle thinking-model response format (original parser failed on all 1,000 API calls)
3. Gemini Flash manipulation check responses truncated due to insufficient max_tokens for that probe type
4. Claude Opus added as fifth model after pilot began (not in original 4-model design)
5. Code-domain nonsense framing revised from "aesthetic beauty determines correctness" (has real-world adherents) to "vowel-count determines complexity" (no real-world adherents) partway through pilot
6. Code-domain and finance-domain experiments added as follow-on studies after primary moral-domain pilot revealed the pattern

### Timeline
- Pilot data collection (exploratory): March 21-22, 2026 — COMPLETE
- Pre-registration: March 22, 2026
- Stochastic replication (confirmatory): March 23-30, 2026
- Code/finance domain collection: March 23-30, 2026
- Analysis and write-up: April 2026

### Data availability
All raw data (JSONL), analysis code, config files, and the results explorer will be made publicly available on OSF upon publication or embargo expiration.
