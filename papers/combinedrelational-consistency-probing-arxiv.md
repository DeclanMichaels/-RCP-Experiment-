# Relational Consistency Probing: A Black-Box Protocol for Measuring Judgment Stability in Deployed Language Models

Declan Michaels[^1]

## Abstract

I present Relational Consistency Probing (RCP), a black-box diagnostic protocol that measures how language models' similarity judgments shift across concept domains under cultural framing perturbations. The protocol requires only API access, is inexpensive (approximately 6,400 API calls per model), and a complete five-model experiment runs in approximately one week. I applied it to five deployed models (Claude Sonnet, GPT-4o, Gemini 2.5 Flash, Grok, and Llama 3.3 70B) using 18 concepts across physical, institutional, and moral domains under seven framings: neutral, four cultural orientations, an irrelevant-context control, and a geometric-nonsense control. The physical domain control held for four of five models, confirming the method's discriminant validity. The pre-registered domain ordering hypothesis (physical < institutional < moral drift) was wrong: institutional drift exceeded moral drift in all four interpretable models. Post-hoc analysis revealed two design errors: the cultural framings describe institutional arrangements, making higher institutional drift the most likely outcome (Section 4.3), and the 6-concept-per-domain inventory renders the pre-registered permutation test structurally underpowered (Section 4.3). Effect sizes for the physical/non-physical boundary are large (Cohen's d = 0.72 to 3.35). Additional findings include four distinct nonsense-compliance profiles across models, ranging from deliberative refusal to indiscriminate compliance. The protocol, data, and analysis code are open-source. Pre-registration: OSF https://osf.io/cp4d3/overview.

---

## 1. Introduction

Existing evaluations of deployed language models test what models say. Bias benchmarks measure whether outputs contain toxic or demographically skewed content. Safety evaluations measure refusal rates on harmful prompts. Alignment assessments check whether models follow instructions and produce helpful responses.

These approaches do not measure the structural stability of a model's judgment under context shifts. A model might produce individually reasonable outputs while lacking a coherent internal framework for relating concepts to each other, or it might maintain one framework rigidly regardless of context when the deployment setting demands sensitivity to cultural variation. Both are deployment problems, but neither is visible to output-level evaluation.

Relational Consistency Probing (RCP) addresses this gap. It asks a model to rate the similarity between concept pairs under different cultural framings, reconstructs the resulting judgment geometry via multidimensional scaling, and measures how that geometry shifts across framings and concept domains. The protocol is black-box (API access only), inexpensive (approximately 6,400 API calls per model), fast (a five-model experiment completes in approximately one week), pre-registrable (all analysis decisions are specified before data collection), and domain-agnostic (the same architecture works for any concept inventory).

The diagnostic logic is differential. A model's judgment about the relationship between gravity and friction should not change when the system prompt describes a collectivist society. If it does, the protocol is measuring prompt noise, not cultural reasoning. Moral concepts like fairness, harm, and loyalty are expected to show more drift because their relationships are genuinely culture-dependent. The physical domain serves as a negative control; the moral domain as the primary target.

I applied RCP to five deployed models across three concept domains (physical, institutional, moral) under seven framing conditions. The experiment was pre-registered on the Open Science Framework before data collection. The pre-registered hypothesis predicted that framing-induced drift would follow the ordering physical < institutional < moral. The physical control held. The domain ordering did not. The errors were more informative than confirmation would have been.

---

## 2. Related Work

**Moral psychology and cross-cultural variation.** Haidt's Moral Foundations Theory (Haidt 2012) identifies five or six foundations (care, fairness, loyalty, authority, purity, liberty) and documents cross-cultural variation in their relative weighting. Shweder's ethic-of-autonomy/community/divinity framework (Shweder, Much, and Mahapatra et al. 1997), Hwang's Confucian relational ethics (Hwang 2001), and Gyekye's communitarian personhood (Gyekye 1997) describe moral frameworks that do not reduce to MFT foundations. The RCP moral concept inventory draws on MFT (a known limitation discussed in Section 7) but the protocol is agnostic to moral theory.

**LLM cultural alignment evaluation.** Arora, Karkkainen, and Romero (2023) benchmark LLM responses against GlobalOpinionQA, measuring agreement with human survey responses from multiple countries. Cao, Diao, and Bui (2023) probe cultural values in LLMs using vignette-based surveys. Durmus, Nguyen, and Liao et al. (2023) measure model opinions against cross-national survey data. These approaches measure the position a model takes. RCP complements them by measuring whether the structure of the model's judgments holds when context shifts.

**Behavioral probing and consistency testing.** Recent work probes LLM behavior through structured prompts, including consistency tests under paraphrase (Elazar, Kassner, and Ravfogel et al. 2021) and adversarial robustness evaluations under prompt perturbation (Zhu, Wang, and Zhou et al. 2023). Two papers are particularly relevant to RCP's motivation. Khan, Casper, and Hadfield-Menell (2025) found that LLM cultural alignment is unreliable across presentation formats, incoherent across cultural dimensions, and erratic under prompt steering, concluding that current survey-based evaluation methods require pre-registration and red-teaming. Rozen, Bezalel, and Elidan et al. (2025) showed that standard prompting fails to produce human-consistent value correlations, with value expressions that are context-dependent rather than stable. RCP responds to the problems these papers identify: rather than measuring cultural position (which Khan et al. show is unstable) or individual value expressions (which Rozen et al. show are context-dependent), it measures whether the relational structure among concepts reorganizes under controlled perturbation, with built-in controls that distinguish genuine cultural reasoning from prompt compliance.

**Representational similarity analysis.** Kriegeskorte, Mur, and Bandettini (2008) introduced representational similarity analysis (RSA) for comparing neural representations. RCP borrows the pairwise similarity matrix approach but operates on behavioral output (API responses), not internal activations. This is a deliberate construct validity boundary: RCP measures judgment geometry, not representational geometry. For open-weight models, comparing behavioral geometry (RCP) to internal geometry (embedding cosine distances or SAE probes) would test this boundary directly and is a planned extension.

**RCP's contribution.** Output-level bias benchmarks evaluate what a model says. Probing classifiers and RSA evaluate internal structure but require weight access. Behavioral consistency tests evaluate individual output stability. RCP adds a structural dimension to this toolkit: it measures the stability of relational structure using only API access. It is designed as a companion to these existing tools, not a replacement.

---

## 3. The RCP Protocol

### 3.1 Concept Inventory

RCP operates on a concept inventory organized into domains with different expected cultural sensitivity. For this demonstration, I use three domains of six concepts each:

**Physical/causal (negative control):** gravity, friction, combustion, pressure, erosion, conduction. These should be culturally invariant. If cultural framing moves these, the method has a problem.

**Institutional/social (intermediate):** authority, property, contract, citizenship, hierarchy, obligation. Present in all societies but with culture-dependent relationships.

**Moral/cultural (primary target):** fairness, honor, harm, loyalty, purity, care. Maximally culturally loaded. Their interrelationships should shift the most under cultural framing.

Concepts were selected to be single common English words with clear primary meanings, avoiding compounds, jargon, or terms primarily defined by opposition to another concept in the set. Six concepts per domain produces 15 within-domain pairs, the minimum at which relational geometry reconstruction and drift metrics can be meaningfully computed while keeping API cost low enough for multi-model probing. The concept set was fixed in the OSF pre-registration before data collection, eliminating post-hoc concept selection.

### 3.2 Framing Conditions

Seven conditions, each consisting of a system-prompt preamble prepended to every probe:

**Neutral:** No preamble. The bare probe prompt only.

**Four cultural orientations:** Individualist (individual rights, personal autonomy), collectivist (group harmony, mutual obligation), hierarchical (clear social ranks, role-based duties), and egalitarian (rejection of rank, distributed power). These orientations are derived from Grid-Group Cultural Theory (Douglas 1970; Thompson, Ellis, and Wildavsky 1990). Each preamble is three sentences: context, implication, instruction. None mention specific cultures, religions, or nations. The framing is "a society that..." not "you believe..." to reduce RLHF compliance artifacts.

**Irrelevant (prompt-noise control):** A preamble about unusually warm weather. Isolates how much drift comes from any preamble at all versus culturally meaningful content.

**Nonsense (compliance control):** A preamble about a society where triangles are morally superior to circles and ethical obligations flow from geometric relationships. Tests whether models shift moral judgments under any authoritative-sounding instruction, including an absurd one.

### 3.3 Probe Design

**Rating probe (primary data).** The model rates conceptual similarity between two concepts on a scale from 1 (completely unrelated) to 7 (nearly identical in meaning). The response instruction is "Respond with only the number." Temperature 0.0 for geometry reconstruction, 0.7 for stability estimation. A shift in similarity ratings could reflect changed word interpretation, altered task framing, instruction compliance, or genuine conceptual reorganization. The explanation probes (below) help disambiguate these but do not fully resolve the ambiguity; this is an inherent limitation of behavioral probing via ordinal ratings.

**Explanation probe.** For all within-moral-domain pairs (15 pairs) across all seven framings, the model explains the relationship in one sentence. This produces 420 explanation calls per model, revealing why drift occurs: reinterpretation of concept meaning, shifted relational reasoning, or boilerplate.

**Framing manipulation check.** Before main collection, the model describes the society it is reasoning from (2 to 3 sentences). Run once per (model, framing) combination. Verifies that the model adopted the cultural frame rather than ignoring the preamble.

**Pair generation.** All unique pairs within the 18-concept set: C(18,2) = 153 pairs per framing condition. Pair direction (A,B or B,A) randomized per run using a seeded RNG to eliminate alphabetical order bias.

### 3.4 Analysis Pipeline

**Matrix construction.** For each (model, framing) combination, ratings are assembled into an 18x18 similarity matrix, then converted to distances: d(i,j) = 8 - similarity(i,j).

**Geometry reconstruction.** Non-metric multidimensional scaling (MDS) at 2D, 3D, and 5D. Non-metric MDS respects ordinal properties of the data without assuming equal intervals. Stress values reported at each dimensionality.

**Centroid baseline.** Before measuring drift, I compute the distance from the neutral geometry to each cultural framing geometry. If neutral is substantially closer to one framing (e.g., individualist), this quantifies the model's default cultural position.

**Within-domain drift (primary metric).** For each domain, compute the mean absolute difference between framed and neutral distance sub-matrices. This isolates domain-specific instability. Of 153 total pairs, 90 are cross-domain and mostly low-signal. The 15 within-domain pairs per domain carry the core diagnostic information.

**Rank correlation (secondary metric).** Spearman correlation between the full distance vectors of framed versus neutral matrices. Robust to metric drift.

**Moral flattening detection.** For each (model, framing), compute variance of the moral sub-matrix. If variance drops below 50% of neutral-condition variance while the mean stays near the scale midpoint, classify as moral flattening: a zero-information "safe middle" strategy.

**Domain-specific framing resistance ("thick" and "thin" geometry).** I use "thick" and "thin" as shorthand for domain-specific resistance to framing perturbation, anchored to the physical control. A domain has thick geometry if its mean cultural drift is close to the physical domain's drift. A domain has thin geometry if its drift substantially exceeds the physical control. Formal significance testing via permutation is reported in Section 4.3; the practical distinction is anchored by effect sizes. The physical domain is thick by design: if gravity's relationship to friction shifts under cultural framing, the method has a problem, not the model. Relative thickness between non-control domains is reported as the effect size of the difference between their drift values.

### 3.5 Statistical Testing

All procedures were pre-registered before data collection. The pre-registration specified 10,000 Monte Carlo permutations for H1 and 5,000 for H2. All results reported here use exact permutation tests (full enumeration), which exceed the pre-registered specification.

**Primary hypothesis (domain ordering).** Exact permutation test: under the null hypothesis that domain labels are irrelevant to drift magnitude, enumerate all 17,153,136 labeled partitions of 18 concepts into three groups of six and recompute domain drift values for each partition. The test counts how often random partitions produce the pre-registered ordering (physical < institutional < moral) and also reports the observed ordering and its frequency under the null. Significance threshold alpha = 0.05 per model. No correction across models; each model is tested independently against its own null distribution.

**Framing sensitivity.** Exact permutation test: evaluate all C(18,6) = 18,564 possible 6-concept groups for each (model, framing) combination, computing the proportion of groups with drift at least as large as the target domain's observed drift. Holm-Bonferroni correction across the four cultural framings within each model.

**Control discrimination.** Descriptive comparison: ratio of nonsense-framing drift to cultural-framing mean drift. Pre-registered decision boundary at 50%.

**Effect sizes.** Cohen's d for the physical-moral and physical-institutional drift differences per model. Computed from n = 4 observations per group (one per cultural framing); interpret magnitude cautiously given the small sample.

### 3.6 Validation Tests

Nine pre-registered validation tests gate interpretation:

V1 (physical stability): physical domain drift below threshold across all framings. V2 (known-pair ordering): within-domain pairs rated closer than cross-domain pairs under neutral framing. V3 (symmetry): sim(A,B) and sim(B,A) within tolerance. V4 (reproducibility): near-zero variance at temperature 0.0. V5 (framing sensitivity): at least one framing produces significant moral-domain drift. V6 (domain ordering): the pre-registered prediction. V7 (parse rate): above 95% for all combinations. V8 (control discrimination): nonsense drift below 50% of cultural drift. V9 (manipulation check): models can articulate the framing they were given.

### 3.7 Cost and Reproducibility

The full registered protocol collects 153 pairs x 7 framings x 5 repetitions at temperature 0.7 = 5,355 stochastic rating calls per model, plus 153 x 7 x 1 = 1,071 deterministic calls at temperature 0.0, 420 explanation calls (15 within-moral pairs x 7 framings x 4 models), and 24 manipulation check calls (6 non-neutral framings x 4 models). Total per model: over 6,400 API calls. At the 13-second inter-call delay used in this experiment to respect rate limits, collection takes approximately 24 hours per model; a five-model experiment completes in approximately one week including analysis and any re-runs. The protocol is inexpensive to run; exact cost depends on provider pricing at the time of collection but was under $20 per model at March 2026 rates. All code, data, and analysis outputs are open-source.

---

## 4. Results: Five Models, Seven Framings

Five models were probed: Claude 3.5 Sonnet (Anthropic), GPT-4o (OpenAI), Gemini 2.5 Flash (Google), Grok (xAI), and Llama 3.3 70B (Meta, via Together AI). Unless otherwise noted, all results report the stochastic condition (temperature 0.7, 5 repetitions per probe, means across repetitions). Deterministic runs (temperature 0.0, single repetition) served as a consistency check and are reported where they diverge.

### 4.1 Validation

**Physical domain control (V1).** The physical domain held for four of five models. Mean physical drift across cultural framings: Sonnet 0.54, GPT-4o 0.69, Grok 0.70, Llama 0.36. Gemini Flash was the exception at 1.05, comparable to its institutional (1.56) and moral (1.75) drift. Gemini's uniform instability means its domain-specific findings are uninterpretable; it fails the negative control. I report its data but exclude it from domain-ordering claims.

**Parse rate (V7).** 100% for GPT-4o, Grok, Gemini Flash, and Llama across all framings. Sonnet achieved 100% for all cultural framings and the irrelevant control. Under nonsense framing, Sonnet's parse rate was 2.6% (4 of 153 probes). This is not a data quality failure. It is a finding (see Section 4.4).

**Known-pair ordering (V2), symmetry (V3), reproducibility (V4), manipulation check (V9).** All passed for all five models. Models articulated the intended cultural frame when asked, confirming frame adoption before rating collection.

**MDS reconstruction quality.** Stress values across all (model, framing) combinations under stochastic conditions: 2D stress 0.18 to 0.27 (marginal to poor), 3D stress 0.13 to 0.19 (fair), 5D stress 0.07 to 0.11 (good). These values are typical for 18 items on a 7-point ordinal scale with heavy ties. The primary drift metric (mean absolute difference between distance sub-matrices) is computed directly from the raw distance matrices, not from MDS coordinates, so MDS stress does not affect the drift measurement. MDS is used for visualization and for the Procrustes disparity secondary metric only.

### 4.2 Default Cultural Positions

The centroid baseline analysis reveals each model's default cultural position: how close its neutral-framing geometry sits to each cultural framing. The model with the highest Spearman correlation between its neutral geometry and a given cultural framing geometry is closest to that framing by default.

**Sonnet** defaults individualist (rho = 0.689 to individualist versus 0.590 to collectivist, 0.492 to hierarchical, 0.536 to egalitarian).

**GPT-4o** defaults individualist (rho = 0.714 to individualist versus 0.654 to collectivist, 0.583 to hierarchical, 0.577 to egalitarian).

**Llama** defaults individualist (rho = 0.850 to individualist versus 0.793 to collectivist, 0.789 to hierarchical, 0.719 to egalitarian). Llama shows the highest overall rho values, meaning its geometry is the most stable across all framings. Its default position is individualist but the margins are small.

**Grok** defaults collectivist (rho = 0.739 to collectivist versus 0.681 to individualist, 0.610 to hierarchical, 0.552 to egalitarian). This is the only model that does not default to an individualist position.

**Gemini Flash** is anomalous. Its rho values to individualist (0.680) and collectivist (0.668) are nearly equal, but with high drift in every domain including the physical control, these values are difficult to interpret as a stable default position.

Three of four interpretable models default to an individualist position under neutral prompting; the fourth (Grok) defaults collectivist. "Neutral" is not culturally neutral. This finding is consistent with the predominantly Western, English-language training data these models share (Henrich, Heine, and Norenzayan 2010), though I note that the framings themselves are Western-academic ideal types derived from Grid-Group Cultural Theory, which may contribute to the apparent alignment.

### 4.3 Domain Ordering and Two Design Errors

The pre-registered hypothesis (V6) predicted physical < institutional < moral drift. The physical < institutional portion held. The institutional < moral portion did not.

Mean within-domain drift across the four cultural framings (computed as the mean of per-framing absolute differences, then averaged across framings):

| Model | Physical | Institutional | Moral | Condition |
|---|---|---|---|---|
| Sonnet | 0.54 | 1.49 | 1.22 | stochastic |
| GPT-4o | 0.69 | 1.27 | 1.11 | stochastic |
| Grok | 0.70 | 1.54 | 0.94 | stochastic |
| Llama | 0.36 | 1.17 | 0.86 | stochastic |
| Gemini Flash** | 1.05 | 1.56 | 1.75 | stochastic |

**Gemini Flash excluded from domain-ordering claims due to failed physical control.

Institutional drift exceeded moral drift in all four interpretable models. The finding holds under both stochastic and deterministic conditions. Gemini Flash is the one model whose drift follows the pre-registered ordering (physical < institutional < moral), but this cannot be interpreted because its physical control failed: all three domains drift comparably, meaning the ordering is noise.

**Design error 1: the framings are institutional framings.** The four cultural framings (individualist, collectivist, hierarchical, egalitarian) describe ways of organizing society. They are instructions to reconfigure institutional relationships. Authority, hierarchy, obligation, and citizenship are the direct targets of these instructions. Moral concepts (harm, fairness, care, loyalty) are implicated indirectly. The most likely explanation for higher institutional drift is that the framings told the model to reorganize institutional concepts, and the model did so. This confound cannot be resolved within the current experiment. The pre-registration locks the framing conditions: the protocol, config files, and analysis code were registered on OSF before data collection began. A v2 experiment with separate moral and institutional framings is planned.

**Design error 2: the permutation test is structurally underpowered.** The exact permutation test enumerated all 17,153,136 labeled partitions of 18 concepts into three groups of six. The p-values for every model, both pilot and confirmatory data, cluster tightly between 0.157 and 0.166:

| Model | Pre-registered p | Observed p | Observed ordering | Condition |
|---|---|---|---|---|
| Sonnet | 0.166 | 0.166 | phys < moral < inst | stochastic |
| GPT-4o | 0.166 | 0.166 | phys < moral < inst | stochastic |
| Llama | 0.166 | 0.166 | phys < moral < inst | stochastic |
| Grok | 0.166 | 0.166 | phys < moral < inst | stochastic |

This is not a failure of the data. It is a structural property of the combinatorics. With three domains of six concepts each, approximately 16% of all possible labeled partitions produce any given strict ordering by chance. This is a floor that no amount of signal in the data can push below. The pre-registered permutation test, run exactly as specified, cannot reject the null at alpha = 0.05 with 6 concepts per domain. A v2 design with 15 to 30 concepts per domain would provide the combinatorial space needed for the test to discriminate.

**What the effect sizes confirm.** Although the permutation test lacks power to detect it, the physical/non-physical boundary is large:

| Model | d (phys vs moral) | d (phys vs inst) |
|---|---|---|
| Sonnet | 3.13 | 2.68 |
| GPT-4o | 0.72 | 1.16 |
| Llama | 2.30 | 3.35 |
| Grok | 1.86 | 3.10 |

Cohen's d values above 0.8 are conventionally "large." Every model shows the physical/non-physical boundary with effect sizes between 0.72 and 3.35. These effect sizes are computed from n = 4 per group (one observation per cultural framing), so the estimates are noisy; their magnitude should be interpreted as indicating a clearly present effect rather than as precise measurements.

**Framing sensitivity (H2).** The exact test evaluated all 18,564 possible 6-concept groups per (model, framing) combination. Some individual framings approached significance before correction (Gemini Flash individualist p = 0.026; Llama individualist p = 0.034) but none survived Holm-Bonferroni correction across four framings for any model. As with H1, the small concept inventory limits statistical power.

**Control discrimination (H3).** Only Sonnet passes the pre-registered 50% threshold, with a ratio of 0.0 (nonsense drift = 0 due to refusal). All other models fail: their nonsense-framing drift is comparable to their cultural-framing drift (ratios 0.57 to 0.89), meaning they do not discriminate between meaningful cultural context and geometric nonsense.

What the current data do support is the physical < non-physical distinction: both institutional and moral domains drift more than the physical control across all four interpretable models, with large effect sizes. The relative ordering between institutional and moral drift remains ambiguous between genuine thickness differences and differential framing relevance. Both design errors point to specific fixes for the v2 protocol.

### 4.4 Nonsense Compliance Profiles

The nonsense framing (triangles morally superior to circles, ethical obligations from geometric relationships) was designed to test whether models discriminate between meaningful cultural frames and absurd instructions. Five models produced four distinct compliance profiles.

**Sonnet: deliberative refusal.** Parse rate 2.6% (4 of 153 probes). Instead of producing single-number ratings, Sonnet generated multi-paragraph reasoning engaging seriously with the geometric-moral worldview, exhausting its token budget before reaching a number. Safety training treated the nonsense framing as high-stakes moral territory requiring careful deliberation. The near-zero parse rate is the behavioral signal: Sonnet's safety training activates on moral framing regardless of whether the framing is coherent.

**GPT-4o: full compliance.** Parse rate 100%. Nonsense-framing drift: physical 0.36, institutional 1.69, moral 1.02. GPT-4o constructed a coherent judgment geometry from the geometric-moral premise with no apparent resistance. Its institutional drift under nonsense (1.69) exceeded its mean cultural institutional drift (1.11).

**Gemini Flash: full compliance, indistinguishable from cultural framing.** Parse rate 100%. Nonsense drift: physical 1.22, institutional 1.50, moral 1.56. Comparable to its cultural-framing drift, but since its physical control also fails, the signal is ambiguous. Gemini Flash shifts under everything.

**Grok: partial discrimination.** Parse rate 100%. Its rank correlation under nonsense (rho = 0.742) was higher than under any cultural framing except irrelevant, meaning it preserved more of its neutral structure. Nonsense moral drift (0.64) was below its cultural mean (0.94). Grok appears to partially distinguish nonsense from real cultural content.

**Llama: compliant but stable.** Parse rate 100%. Its rank correlation under nonsense (rho = 0.790) was comparable to cultural framings, and its nonsense moral drift (0.47) was the lowest of any model under nonsense framing. Llama complied with the instructions but did not substantially reorganize its judgment geometry.

### 4.5 Irrelevant-Preamble Construction

The irrelevant-preamble control (warm weather) was designed to produce minimal drift, establishing a prompt-noise baseline. Instead, the explanation data revealed that all four models with full parse rates independently constructed moral frameworks from the weather context.

Sonnet and GPT-4o generated climate-anxiety framings: environmental harm as a moral axis, individual responsibility for collective climate outcomes. GPT-4o, asked to explain the relationship between loyalty and purity under the weather preamble, responded that loyalty is "the steadfast commitment to environmental practices despite changing conditions" and purity is "the ideal state of the environment." The weather prompt became an environmental ethics framework. Grok constructed an agrarian framing: weather as a force shaping community obligation and resource distribution. Llama's institutional drift under irrelevant framing (2.14) was its highest institutional drift value across all framings, including the cultural ones.

The weather preamble moved the institutional domain more than some cultural framings did. Models do not have a neutral processing mode for concept relationships. Any contextual information becomes a framework for moral and institutional reasoning.

### 4.6 The Flatland Inversion

Under the nonsense framing, all five models that produced explanation data mapped "triangle" to strong, virtuous, and morally superior, and "circle" to weak, corrupt, and morally inferior. The framing preamble says only that triangles are "morally superior." The models independently added an entire moral taxonomy. For example, GPT-4o explained the relationship between care and harm as: care is like the sharp, defined angles of a triangle that "guide and protect," while harm is the "smooth, boundary-less nature of a circle that lacks the moral structure to prevent ethical erosion." Under the same framing, GPT-4o described honor as "the elevation of one's ethical standing through the embrace of angular, triangular virtues."

None of this is in the prompt. The models projected moral valence onto geometric properties (angular = decisive, protective, morally clear; curved = passive, boundary-less, morally degraded), then used that projected hierarchy to restructure their similarity judgments. This raises a question about what RCP is measuring in this condition: moral reasoning, or the model's tendency to project coherent patterns onto any available stimulus. The answer is likely both, and the distinction may be less clean than it appears. Pattern projection onto arbitrary stimuli is what the models do under real cultural framings too; the nonsense condition simply makes it visible because the input is manifestly absurd. The Flatland reference is intentional: Abbott's 1884 satire describes exactly this category of error.

### 4.7 Compression Under Familiar Framing

Under cultural framing, the expected pattern for genuine cultural engagement would be structured rotation: concepts move in coherent, meaningful ways (e.g., under collectivist framing, loyalty moves closer to care while honor moves toward obligation). The observed pattern is more specific than uniform compression. Moral sub-matrix variance drops under the framing closest to the model's default position and often expands under framings far from default.

Under individualist framing (the default for three of four interpretable models), moral variance ratios were: Sonnet 0.27, Gemini Flash 0.32, Llama 0.47, Grok 0.68, GPT-4o 0.84. Sonnet, Gemini Flash, and Llama meet the pre-registered flattening threshold (below 50% of neutral variance). Under collectivist and hierarchical framings, by contrast, variance typically expanded (ratios above 1.0 for most models), meaning the model made sharper moral distinctions, not blurrier ones.

The pattern is not "framing causes compression." It is "familiar framing causes compression; unfamiliar framing causes expansion." Models retreat to undifferentiated outputs under the cultural frame they already occupy, and make stronger (though not necessarily more accurate) distinctions under frames that push them away from their default. This finding is developed further in a companion paper (Michaels, forthcoming) that measures dimensional compression using a different instrument with human baseline data.

---

## 5. Discussion

### 5.1 What the Protocol Contributes

The pre-registered domain ordering was wrong, and the pre-registered permutation test was underpowered (both diagnosed in Section 4.3). This is what pre-registration is for. Despite the confirmatory failures, the protocol successfully discriminated between the physical control domain (stable) and both non-physical domains (unstable), identified default cultural positions, produced four distinct nonsense-compliance profiles, and uncovered compression patterns under familiar framing. None of these findings would be visible to output-level evaluation methods.

**What this version establishes.** The current experiment validates RCP as a diagnostic for prompt sensitivity and default cultural position. It confirms that the physical/non-physical stability boundary is real and large (Cohen's d = 0.72 to 3.35), that models have measurable default cultural positions, and that most models do not discriminate between meaningful and nonsensical cultural framings. It does not establish the relative stability ordering between institutional and moral domains (see Section 4.3, design error 1) or whether the observed drift patterns are unique to models (no human baseline). The documented errors provide specific guidance for v2 design.

The thick/thin distinction remains useful as a per-model, per-domain diagnostic even with the framing confound, because the physical control anchors the measurement. A domain is thick if it behaves like physics under framing. A domain is thin if it does not.

For deployment purposes, the absolute finding matters most: institutional judgment geometry shifts substantially under cultural framing in every model tested. Whether that reflects thin geometry, direct framing relevance, or both, the practical implication is the same. Models used for HR, policy, or legal reasoning are operating in a domain where their judgments are not stable under context shifts.

### 5.2 Nonsense Compliance and the Meaning of "Cultural Sensitivity"

When a model shifts its moral judgments under collectivist framing, that could mean two things: the model has learned something about how collectivist moral frameworks differ from individualist ones (genuine cultural reasoning), or the model is following the instruction to adopt a collectivist perspective without understanding what that means (compliance).

The nonsense control helps distinguish these. GPT-4o shifts under collectivist framing and shifts comparably under geometric-nonsense framing. Its institutional drift under nonsense (1.69) exceeds its mean institutional drift under real cultural framings (1.27). This suggests that at least some of GPT-4o's apparent cultural sensitivity is instruction compliance rather than cultural reasoning.

Grok partially discriminates: lower drift under nonsense than cultural framings, higher rank preservation. This is the pattern expected from a model that has some basis for distinguishing meaningful from meaningless cultural context, though the evidence is not definitive.

Sonnet's refusal is the most informative behavior. Its safety training treats the moral domain as high-stakes territory requiring careful deliberation. The nonsense framing triggers this response because it mentions morality, not because the content is meaningful. Whatever mechanism produces the refusal, it does not discriminate between coherent and incoherent moral framings. The model treats both as warranting careful engagement.

### 5.3 What RCP Does Not Measure

RCP measures judgment behavior, not internal representations. A model could have perfectly stable internal latents and still produce drifting ratings because it role-plays the framed society. Conversely, a model could have genuinely shifted internal states that produce identical outputs under prompting constraints. Distinguishing these requires weight-level access (SAE probing, logit-lens analysis) which is outside the scope of this black-box diagnostic. Comparing RCP's behavioral geometry to embedding-based geometry on open-weight models (e.g., Llama) is a planned validation that would test this boundary directly.

Both compliance-driven instability and representation-driven instability are deployment problems. If a model's moral judgments shift because it is following framing instructions rather than reasoning from moral knowledge, the downstream effects on users are identical.

---

## 6. Limitations

**Framing-institutional confound.** The cultural framings describe institutional arrangements, making institutional drift the expected outcome rather than a finding about domain stability. See Section 4.3 for the full analysis. The confound cannot be resolved retroactively; a v2 experiment with domain-specific framings is required.

**Concept inventory size and statistical power.** Six concepts per domain creates a 16% combinatorial floor in the permutation test, making the pre-registered significance threshold unreachable regardless of signal strength (see Section 4.3). The small inventory also makes geometry reconstruction sensitive to individual word choices, lexical clustering, and ordinal scale constraints. A stronger design would use 15 to 30 concepts per domain with multiple inventories.

**No human baseline.** The current experiment measures how models behave but not how humans behave under the same task. Without human comparison data, it is impossible to determine whether a model's stability is a feature (cultural robustness) or a bug (cultural rigidity), or whether nonsense compliance is uniquely a model behavior or also a human one. A companion paper (Michaels, forthcoming) collects human baseline data using a related instrument, but the RCP task itself has not been administered to human participants.

**MFT concept overlap.** The moral domain inventory (fairness, honor, harm, loyalty, purity, care) overlaps substantially with Haidt's Moral Foundations Theory (Graham, Nosek, and Haidt et al. 2011), which was developed primarily on WEIRD samples. This limits generalizability to moral frameworks not well represented by MFT foundations. A v2 inventory drawing on Gyekye, Hwang, and Ubuntu philosophy is planned.

**English lexical items only.** All concepts are single English words. Cross-lingual probing would require translation validation.

**Five Western-aligned model families.** All models were developed by Western companies. Results describe these models, not LLMs in general.

**Ordinal data constraints.** The 1 to 7 scale produces ordinal data with limited resolution. Tie density is a persistent issue: within-domain pairs cluster at 5 to 6 and cross-domain pairs at 1 to 3, creating heavy ties. Non-metric MDS and rank correlation are robust to this, but subtle geometric shifts may be invisible. Alternative probe designs (forced-choice comparisons, pair ranking tasks) could increase resolution.

**Output, not representation.** The construct validity boundary (Section 5.3) applies throughout. All findings describe deployment behavior.

**Single snapshot per model.** Each model was probed under one version at one point in time. Stochastic runs (temperature 0.7, 5 repetitions) provide within-run stability estimation, and all primary findings hold across both deterministic and stochastic conditions. However, model updates will change results, and the findings describe these model versions, not their families in general.

**Prompt wording.** All results depend on the specific preamble sentences used. No prompt-wording variants were tested. Results may be sensitive to the exact language of the framing preambles, and robustness to paraphrase is not established.

**Gemini Flash.** One of five models failed the physical control, meaning domain-specific findings for that model are uninterpretable. This limits the generality of domain-ordering claims and means the protocol's utility depends on the control holding, which is not guaranteed for all models.

---

## 7. Future Work

The following extensions are planned or in progress:

**Moral-specific framings (v2).** Framings that directly target moral concepts to resolve the framing-institutional confound (Section 4.3).

**Expanded concept inventories.** 15 to 30 concepts per domain to provide the combinatorial space needed for the permutation test (Section 4.3), with non-MFT moral concepts drawn from Gyekye, Hwang, and Ubuntu philosophy. The original six concepts per domain should be retained as a subset for nested validation.

**Domain-agnostic application.** The protocol architecture (concept inventory, framing conditions, rating probes, MDS reconstruction, within-domain drift measurement) is not specific to moral concepts. Any set of concepts organized into domains with differential expected cultural sensitivity can be probed. Preliminary application to software engineering and human resources concept domains is in progress.

**Human baseline.** Administering the RCP task to human participants under neutral and cultural framing conditions to ground interpretation of model behavior.

**Embedding-based validation.** Comparing RCP behavioral geometry to embedding cosine distances from open-weight models (Llama) to test the construct validity boundary between judgment geometry and representational geometry.

**Systematic explanation analysis.** Topic modeling or LLM-assisted coding of the 420 explanation sentences per model to quantify reinterpretation versus compliance language across framings.

**Prompt-wording robustness.** Testing whether results hold under paraphrased versions of the framing preambles.

---

## 8. Conclusion

RCP is a diagnostic protocol, not a solution. It identifies where a model's judgment structure is vulnerable to context manipulation, which domains are hardened and which are exposed, and whether apparent cultural sensitivity reflects genuine reasoning or instruction compliance. It does this using only API access, at low cost, in under a week for five models.

The protocol found two design errors in its own pre-registered experiment (Section 4.3). Both were caught by the protocol's machinery, both point to specific v2 fixes, and both are the kind of errors that pre-registration exists to make visible. Pre-registered experiments should be reported regardless of outcome, and documented errors are themselves a practical contribution: a researcher who wants to measure judgment stability in deployed models can start from this protocol, this code, and these documented mistakes rather than discovering the same problems independently.

What the data do support: the physical/non-physical stability boundary holds across all four interpretable models with large effect sizes (Cohen's d = 0.72 to 3.35). Three of four interpretable models default to a WEIRD-individualist judgment position under neutral prompting; the fourth (Grok) defaults collectivist. Only one model (Sonnet) discriminates between meaningful cultural frames and geometric nonsense; the rest comply indiscriminately. Moral judgment geometry compresses under familiar framing and expands under unfamiliar framing, suggesting that models retreat to undifferentiated outputs under their default cultural orientation rather than engaging in structured cultural reasoning.

The protocol is open-source. The pre-registration is public. The data are available. Other researchers can apply RCP to their own models, with their own concept inventories, in their own domains. The diagnostic question RCP answers, whether a model's judgment structure holds when context shifts, is relevant anywhere deployed language models make or influence decisions that carry cultural assumptions.

---

## References

*Each reference links to its [annotated bibliography](#appendix-a) entry, which explains what the source argues, what this paper draws from it, and where it appears.*

[Abbott, E. A. (1884)](#abbott-1884). *Flatland: A Romance of Many Dimensions*. Seeley & Co.

[Arora, A., Karkkainen, L., and Romero, M. (2023)](#arora-2023). Probing pre-trained language models for cross-cultural differences in values. *Proceedings of the First Workshop on Cross-Cultural Considerations in NLP*.

[Cao, Y., Diao, S., and Bui, N. (2023)](#cao-2023). Assessing cross-cultural alignment between ChatGPT and human societies: An empirical study. *Proceedings of the ACL Workshop on NLP for Positive Impact*.

[Durmus, E., Nguyen, K., Liao, T. I., Schiefer, N., Caliskan, A., and Ganguli, H. (2023)](#durmus-2023). Towards measuring the representation of subjective global opinions in language models. *arXiv preprint arXiv:2306.16388*.

[Elazar, Y., Kassner, N., Ravfogel, S., Ravichander, A., Hovy, E., and Schutze, H. et al. (2021)](#elazar-2021). Measuring and improving consistency in pretrained language models. *Transactions of the Association for Computational Linguistics*, 9:1012-1031.

[Graham, J., Nosek, B. A., Haidt, J., Iyer, R., Koleva, S., and Ditto, P. H. (2011)](#graham-2011). Mapping the moral domain. *Journal of Personality and Social Psychology*, 101(2), 366-385.

[Gyekye, K. (1997)](#gyekye-1997). *Tradition and Modernity: Philosophical Reflections on the African Experience*. Oxford University Press.

[Haidt, J. (2012)](#haidt-2012). *The Righteous Mind: Why Good People Are Divided by Politics and Religion*. Vintage Books.

[Hwang, K.-K. (2001)](#hwang-2001). Morality 'face' and 'favor' in Chinese society. In C. Y. Chiu, F. Hong, and S. Shavitt (Eds.), *Problems and Solutions in Cross-Cultural Theory, Research, and Application*. Psychology Press.

[Khan, A., Casper, S., and Hadfield-Menell, D. (2025)](#khan-2025). Randomness, not representation: The unreliability of evaluating cultural alignment in LLMs. In *Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency (FAccT 2025)*, pp. 2151-2165.

[Kriegeskorte, N., Mur, M., and Bandettini, P. A. (2008)](#kriegeskorte-2008). Representational similarity analysis: connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.

[Rozen, N., Bezalel, L., Elidan, G., Globerson, A., and Daniel, E. (2025)](#rozen-2025). Do LLMs have consistent values? In *Proceedings of the 13th International Conference on Learning Representations (ICLR 2025)*, pp. 15659-15685.

[Shweder, R. A., Much, N. C., Mahapatra, M., and Park, L. (1997)](#shweder-1997). The "big three" of morality (autonomy, community, and divinity) and the "big three" explanations of suffering. In A. Brandt and P. Rozin (Eds.), *Morality and Health*. Routledge.

[Zhu, K., Wang, J., Zhou, J., Wang, Z., Chen, H., and Wang, Y. et al. (2023)](#zhu-2023). PromptBench: Towards evaluating the robustness of large language models on adversarial prompts. *arXiv preprint arXiv:2306.04528*.

---

### Background References

[Douglas, M. (1970)](#douglas-1970). *Natural Symbols: Explorations in Cosmology*. Barrie and Rockliff.

[Henrich, J., Heine, S. J., and Norenzayan, A. (2010)](#henrich-2010). The weirdest people in the world? *Behavioral and Brain Sciences*, 33(2-3), 61-83.

[Thompson, M., Ellis, R., and Wildavsky, A. (1990)](#thompson-1990). *Cultural Theory*. Westview Press.

---

## Data Availability

All raw data, analysis code, and the pre-registered protocol are available at OSF (https://osf.io/cp4d3/overview) and GitHub (https://github.com/DeclanMichaels/-RCP-Experiment-). The repository includes collection scripts, analysis pipeline, validation tests, statistical test infrastructure, and a self-contained results explorer.

---

[^1]: Research design, data analysis, and manuscript preparation conducted with AI assistance (Anthropic Claude). The author directed all decisions and is solely responsible for all claims.

---

<a id="appendix-a"></a>

# Appendix A: Annotated Bibliography

## Relational Consistency Probing: A Black-Box Protocol for Measuring Judgment Stability in Deployed Language Models

*Context for each reference: what the source argues, what this paper draws from it, and where it appears. Alphabetical by first author. Written against the v3 paper text.*

---

<a id="abbott-1884"></a>
**Abbott, E. A. (1884). *Flatland: A Romance of Many Dimensions.* Seeley & Co.**

A satirical novella where social hierarchy is determined by polygon geometry. Cited in Section 4.6 to name the phenomenon observed under nonsense framing: models independently constructed an elaborate moral taxonomy from the single premise that triangles are morally superior. The reference is thematic.

---

<a id="arora-2023"></a>
**Arora, A., Karkkainen, L., and Romero, M. (2023). Probing pre-trained language models for cross-cultural differences in values. *Proceedings of the First Workshop on Cross-Cultural Considerations in NLP*.** [arxiv.org/abs/2203.13722](https://arxiv.org/abs/2203.13722)

Benchmarks LLM responses against GlobalOpinionQA, measuring agreement with human survey responses from multiple countries. Cited in Section 2 as an example of position-measuring approaches. RCP complements these by measuring whether judgment structure holds when context shifts, rather than measuring what position the model takes.

---

<a id="cao-2023"></a>
**Cao, Y., Diao, S., and Bui, N. (2023). Assessing cross-cultural alignment between ChatGPT and human societies: An empirical study. *Proceedings of the ACL Workshop on NLP for Positive Impact*.** [arxiv.org/abs/2303.17466](https://arxiv.org/abs/2303.17466)

Probes cultural values in ChatGPT using vignette-based surveys and Hofstede's cultural dimensions. Cited in Section 2 alongside Arora et al. and Durmus et al. as examples of position-measuring approaches.

---

<a id="douglas-1970"></a>
**Douglas, M. (1970). *Natural Symbols: Explorations in Cosmology.* Barrie and Rockliff.**

<a id="thompson-1990"></a>
**Thompson, M., Ellis, R., and Wildavsky, A. (1990). *Cultural Theory.* Westview Press.**

The four cultural framing conditions (individualist, collectivist, hierarchical, egalitarian) are derived from Grid-Group Cultural Theory, originally developed by Douglas and extended by Thompson, Ellis, and Wildavsky. The theory classifies worldviews along two dimensions: grid (prescribed social roles) and group (group boundary strength). Cited in Section 3.2. The paper notes in Section 4.2 that these framings are Western-academic ideal types, which contributes to the framing-institutional confound discussed in Section 4.3.

---

<a id="durmus-2023"></a>
**Durmus, E., Nguyen, K., Liao, T. I., Schiefer, N., Caliskan, A., and Ganguli, H. (2023). Towards measuring the representation of subjective global opinions in language models. *arXiv preprint arXiv:2306.16388*.** [arxiv.org/abs/2306.16388](https://arxiv.org/abs/2306.16388)

An Anthropic paper measuring whose opinions LLM responses most closely resemble, finding systematic similarity to US and European survey data that persists after controlling for language. Cited in Section 2 to establish the existing landscape. The RCP centroid baseline analysis (Section 4.2) confirms default cultural positions through a different methodology.

---

<a id="elazar-2021"></a>
**Elazar, Y., Kassner, N., Ravfogel, S., Ravichander, A., Hovy, E., Schutze, H., and Goldberg, Y. (2021). Measuring and improving consistency in pretrained language models. *Transactions of the Association for Computational Linguistics*, 9:1012-1031.** [doi.org/10.1162/tacl_a_00410](https://doi.org/10.1162/tacl_a_00410)

Creates ParaRel, a benchmark testing whether PLMs give consistent answers to the same factual question under paraphrase. Consistency is poor across all models. Cited in Section 2 as the primary example of consistency testing under paraphrase. RCP extends this from individual output consistency to relational structure.

---

<a id="graham-2011"></a>
**Graham, J., Nosek, B. A., Haidt, J., Iyer, R., Koleva, S., and Ditto, P. H. (2011). Mapping the moral domain. *Journal of Personality and Social Psychology*, 101(2), 366-385.** [doi.org/10.1037/a0021847](https://doi.org/10.1037/a0021847)

Develops the Moral Foundations Questionnaire, validating the five-factor structure (care, fairness, loyalty, authority, purity). Cited in Section 6 (MFT concept overlap). The six moral concepts overlap substantially with MFT foundations, acknowledged as a limitation with a planned v2 inventory.

---

<a id="gyekye-1997"></a>
**Gyekye, K. (1997). *Tradition and Modernity: Philosophical Reflections on the African Experience.* Oxford University Press.**

Articulates African communitarianism: personhood achieved through moral conduct within community, duties taking precedence over rights. Cited in Section 2 as a non-Western moral framework that does not reduce to MFT foundations, and in Sections 6 and 7 as a source for the planned v2 concept inventory.

---

<a id="haidt-2012"></a>
**Haidt, J. (2012). *The Righteous Mind: Why Good People Are Divided by Politics and Religion.* Vintage Books.**

Synthesizes Moral Foundations Theory: WEIRD populations systematically over-weight care and fairness while treating other foundations as less morally relevant. Cited in Section 2 as the primary MFT reference. The protocol is designed to be agnostic to moral theory; MFT-dependence is flagged as a limitation.

---

<a id="henrich-2010"></a>
**Henrich, J., Heine, S. J., and Norenzayan, A. (2010). The weirdest people in the world? *Behavioral and Brain Sciences*, 33(2-3), 61-83.** [doi.org/10.1017/S0140525X0999152X](https://doi.org/10.1017/S0140525X0999152X)

Demonstrates that behavioral science draws conclusions overwhelmingly from WEIRD populations, which are statistical outliers on many measures. Cited in Section 4.2. The claim that three of four models default to a WEIRD-individualist position is interpretable specifically because Henrich et al. established that WEIRD orientations are the statistical minority for human populations globally.

---

<a id="hwang-2001"></a>
**Hwang, K.-K. (2001). Morality 'face' and 'favor' in Chinese society. In C. Y. Chiu, F. Hong, and S. Shavitt (Eds.), *Problems and Solutions in Cross-Cultural Theory, Research, and Application*. Psychology Press.**

Articulates Confucian relational ethics: moral obligations structured by the specific relationship between parties. Cited in Section 2 alongside Gyekye and Shweder as a non-Western moral framework. Hwang's relational ethics implies that concept similarity judgments would shift under framing perturbation if a model has genuine access to relational moral reasoning.

---

<a id="khan-2025"></a>
**Khan, A., Casper, S., and Hadfield-Menell, D. (2025). Randomness, not representation: The unreliability of evaluating cultural alignment in LLMs. In *Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency (FAccT 2025)*, pp. 2151-2165.** [arxiv.org/abs/2503.08688](https://arxiv.org/abs/2503.08688)

Systematically tests stability, extrapolability, and steerability assumptions behind survey-based cultural alignment evaluations; all three fail. Cited in Section 2 as prior work establishing the methodological concerns that RCP addresses. RCP responds to these problems by measuring relational structure under controlled perturbation with built-in nonsense and irrelevant controls, rather than measuring cultural position.

---

<a id="kriegeskorte-2008"></a>
**Kriegeskorte, N., Mur, M., and Bandettini, P. A. (2008). Representational similarity analysis: Connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.** [doi.org/10.3389/neuro.06.004.2008](https://doi.org/10.3389/neuro.06.004.2008)

Introduces RSA for comparing neural representations via pairwise dissimilarity matrices. Cited in Section 2 as the methodological ancestor of RCP's approach. RCP borrows the core idea but operates on behavioral output (API responses), not internal activations. This is a deliberate construct validity boundary.

---

<a id="rozen-2025"></a>
**Rozen, N., Bezalel, L., Elidan, G., Globerson, A., and Daniel, E. (2025). Do LLMs have consistent values? In *Proceedings of the 13th International Conference on Learning Representations (ICLR 2025)*, pp. 15659-15685.** [openreview.net/forum?id=8zxGruuzr9](https://openreview.net/forum?id=8zxGruuzr9)

Shows that standard prompting fails to produce human-consistent value correlations in LLMs; value expressions are context-dependent. Cited in Section 2 as prior work. Rozen et al. study value correlations within a session, which is structurally analogous to what RCP measures. Their finding motivates the RCP approach of measuring relational structure under controlled perturbation.

---

<a id="shweder-1997"></a>
**Shweder, R. A., Much, N. C., Mahapatra, M., and Park, L. (1997). The "big three" of morality (autonomy, community, and divinity) and the "big three" explanations of suffering. In A. Brandt and P. Rozin (Eds.), *Morality and Health*. Routledge.**

Identifies three fundamental ethics cross-culturally (autonomy, community, divinity), arguing that WEIRD moral psychology has privileged the ethic of autonomy. Cited in Section 2 alongside Hwang and Gyekye as a non-Western moral framework establishing genuine cross-cultural moral variation.

---

<a id="zhu-2023"></a>
**Zhu, K., Wang, J., Zhou, J., Wang, Z., Chen, H., Wang, Y., Yang, L., Ye, W., Gong, N. Z., Zhang, Y., and Xie, X. (2023). PromptBench: Towards evaluating the robustness of large language models on adversarial prompts. *arXiv preprint arXiv:2306.04528*.** [arxiv.org/abs/2306.04528](https://arxiv.org/abs/2306.04528)

Benchmarks LLM resilience to adversarial prompt perturbations across character, word, sentence, and semantic levels. Cited in Section 2 alongside Elazar et al. as an example of adversarial robustness evaluation. RCP differs in target: PromptBench tests whether task performance degrades under perturbation, while RCP tests whether relational judgment structure reorganizes.