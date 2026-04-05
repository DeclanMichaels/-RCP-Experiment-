# Annotated Bibliography

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
