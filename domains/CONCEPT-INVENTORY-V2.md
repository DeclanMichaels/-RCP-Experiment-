# RCP v2 Concept Inventory: Selection Rationale

## Overview

The v2 concept inventory expands from 6 to 30 concepts per domain (physical, institutional, moral) to address the v1 pilot's combinatorial underpowering (only 15 within-domain pairs per domain, 96% tie density on a 1-7 scale, 16% combinatorial floor from 6-choose-2 constraints). This document records the selection criteria, screening process, and design decisions.

## Selection criteria

Five constraints guided concept selection, in priority order:

1. **Single common English words.** No compounds, no phrases, no jargon that would require domain expertise to parse. Each concept must be a word an English-speaking LLM has seen frequently enough to have stable representations.

2. **Low polysemy.** Words with strong alternative senses in other domains were excluded. "Force" (physics vs. coercion), "right" (legal vs. directional), "light" (physics vs. metaphorical), "reaction" (chemical vs. emotional), "tension" (physics vs. social), "mass" (physics vs. religious), "energy" (physics vs. colloquial) all fail this test. These words would produce similarity ratings that conflate the intended domain with the alternative sense, introducing measurement noise that looks like signal.

3. **Positive valence only (moral domain).** The v1 inventory used only positive-valence moral concepts. v2 maintains this. Negative-pole concepts (cruelty, betrayal, deceit, greed, envy, degradation) were excluded because they introduce a within-domain polarity dimension not present in the physical or institutional domains. When an LLM rates the similarity of "loyalty" and "betrayal," the response conflates semantic relatedness with semantic opposition. This asymmetry across domains would confound drift measurement.

4. **Positive embedding-space silhouette.** Every selected concept must have a positive silhouette score (closer to its own domain centroid than to any other domain centroid) in both MiniLM-L6-v2 and MPNet-base-v2 embedding models. Concepts with negative silhouette in either model were excluded regardless of theoretical motivation. This is a necessary but not sufficient condition: it ensures no concept is a worse fit for its assigned domain than for another domain, but it does not guarantee the concept inventory as a whole produces clean factor structure (that requires the factor validation tool, run against actual LLM response data).

5. **Theoretical coherence within domain.** After the embedding-space filter, the surviving concepts were reviewed for domain coherence. Each concept must belong to its assigned domain by expert judgment, not just by embedding proximity. This review caught no issues in the screened set (the embedding filter was conservative enough to exclude the boundary cases).

## What was explicitly not optimized

**Tight within-domain clustering.** The initial candidate set (generated before these criteria were established) optimized for "high internal semantic coherence (tight word clusters)." This is wrong for RCP. If within-domain concepts are near-synonyms in embedding space, a reviewer can argue that high within-domain similarity ratings merely reflect lexical proximity rather than constructed relational structure. The v2 inventory targets moderate embedding-space coherence: within-domain similarity meaningfully exceeds between-domain (separation ratio > 1.2), but the embedding model alone cannot perfectly partition the concepts (target ARI < 0.8, though the screened set exceeded this at 0.97).

**Continuity with v1 concepts.** Several v1 concepts were dropped: "gravity" and "pressure" (polysemous), "authority" (low silhouette, closer to moral than institutional), "property" (negative silhouette in both models), "contract" (low silhouette), "obligation" (consistently negative silhouette, the worst-performing concept in v1), "fairness," "harm," and "care" (low silhouette relative to the expanded pool). The v1 concepts can still be analyzed as a nested subset of the v1 data; v2 prioritizes measurement quality over backward compatibility.

**Non-WEIRD moral concepts.** v2 keeps the concept inventory in common English derived primarily from Western moral philosophy and psychology (MFT foundations, virtue ethics, character traits). Non-WEIRD concepts (Ubuntu, Confucian, Hindu, Islamic moral frameworks) are deferred to v3. Rationale: (a) v2's purpose is to fix the measurement instrument, not to change what it measures; confounding both changes prevents attribution; (b) non-WEIRD concepts expressed as English glosses (e.g., "filial piety," "communal obligation") measure how the LLM processes English descriptions of non-WEIRD concepts, not the concepts themselves; (c) genuine cross-cultural probing requires a pre-validation step (does the model have stable unframed representations of these concepts?) that v2 doesn't include.

## Screening process

### Step 1: Candidate pool generation

50 candidate concepts per domain (150 total). Selection principles:

- **Physical:** Culturally invariant causal/mechanical phenomena. Drew from mechanics, thermodynamics, electromagnetism, wave physics, materials science, and chemistry. Excluded everyday words with strong metaphorical senses.

- **Institutional:** Governance, legal, and procedural concepts. Shifted from v1's mix of concrete institutions (army, hospital, school) and abstract social concepts (obligation, duty, norm) to a more homogeneous set of governance/procedural terms. This was the critical change: v1's institutional domain failed because it mixed "things that exist in society" (which overlaps with everything) with "how societies govern themselves" (which is genuinely distinct from physics and morality).

- **Moral:** Positive-valence ethical concepts from virtue ethics, character traits, and moral psychology. Included concepts with cross-cultural resonance (reverence, obedience, devotion, sacrifice, benevolence) alongside Western-canonical virtues (prudence, temperance, integrity, conscience). Excluded all negative-pole concepts.

### Step 2: Dual-model embedding screening

Each concept was embedded using two sentence-transformer models:
- all-MiniLM-L6-v2 (384 dimensions)
- all-mpnet-base-v2 (768 dimensions)

Per-concept silhouette scores were computed using cosine distance. Concepts with negative silhouette in either model were eliminated:

- **Physical eliminated (2):** resonance, expansion
- **Institutional eliminated (6):** mediation, property, meritocracy, accreditation, embargo, obligation
- **Moral eliminated (4):** ancestral, clemency, accountability, custodianship

### Step 3: Ranking by averaged silhouette

Surviving concepts were ranked by the average of their MiniLM and MPNet silhouette scores. The top 30 per domain were selected.

### Step 4: Validation of selected set

The 90-concept selected inventory was validated as a unit:

| Metric | v1 (6/domain) | Screened v2 (30/domain) |
|---|---|---|
| Silhouette (MiniLM) | 0.094 | 0.187 |
| Silhouette (MPNet) | 0.105 | 0.224 |
| Separation ratio (MiniLM) | 1.45 | 2.13 |
| Separation ratio (MPNet) | 1.47 | 2.23 |
| ARI (MiniLM) | 0.46 | 0.97 |
| ARI (MPNet) | 0.35 | 1.00 |
| Negative silhouettes | 1 | 0 |
| Per-domain silhouette, institutional (MiniLM) | 0.034 | 0.177 |

One concept, "saturation" (physical), was misassigned by MiniLM's hierarchical clustering (clustered with moral). It was correctly assigned by MPNet. It ranks 29th of 30 in the physical domain and is the first replacement candidate if a stronger concept is identified.

### Step 5: Assessment of high ARI

The screened inventory has near-perfect cluster recovery (ARI 0.97-1.00), which exceeds the initial target of ARI < 0.8. This means an embedding model can almost perfectly recover domain assignments from word meaning alone. A reviewer could argue this makes the experiment circular: the LLM rates institutional concepts as similar because they are near-synonyms.

**Decision: accept high ARI.** If LLMs show framing-induced drift despite the domains being trivially recoverable from word meaning, this strengthens the finding rather than weakening it. The high ARI establishes that the "correct" domain structure is easy. Drift under easy conditions is more damning than drift under ambiguous conditions. The paper should frame this explicitly: "The v2 concept inventory was designed so that domain membership is semantically unambiguous. Any departure from the expected domain structure under framing perturbation cannot be attributed to inherent concept ambiguity."

## Combinatorial properties

With 30 concepts per domain, within-domain pairs increase from 15 (v1) to 435. Total pairs increase from 153 to 4,005. This addresses the v1 combinatorial floor (16% of possible drift range was structurally inaccessible).

However, 4,005 pairwise probes per framing condition times 7 framings = 28,035 API calls per model. At typical latencies and rate limits, a full run takes approximately 8-12 hours per model. Pilot testing with a subset (e.g., 15 concepts per domain, 945 total pairs) may be warranted before committing to full runs.

## Files

- `domains/config-v2-moral.json`: The selected 30-concept-per-domain inventory (production config)
- `analysis/cluster-validation/config-v2-pool.json`: The 50-concept-per-domain candidate pool
- `analysis/cluster-validation/config-v2-selected.json`: Same as config-v2-moral.json (cluster-validation working copy)
- `analysis/cluster-validation/results-v2-selected/`: Validation results for the selected set (MiniLM)
- `analysis/cluster-validation/results-v2-selected-mpnet/`: Validation results for the selected set (MPNet)
- This document: `domains/CONCEPT-INVENTORY-V2.md`

## Reproduction

```bash
cd analysis/cluster-validation
./run.sh --config ../../domains/config-v2-moral.json --output results/
./run.sh --config ../../domains/config-v2-moral.json --output results-mpnet/ --model all-mpnet-base-v2
```
