# SPEAR literature / novelty audit (checked 2026-10-30)

This is an *editorial research note*, not a claim that the literature search is exhaustive. The associated manuscript cites the primary source for each method.

## Closest known categories

| Source | What it already does | What SPEAR would still need to demonstrate |
|---|---|---|
| [SEAL — NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/6b41e04c41726e2a60e456d0a2b961ab-Abstract-Conference.html) | Trains generated self-edits based on post-fine-tuning task reward | Choosing *which external diagnostic evidence to acquire* by impact on subsequent weight changes |
| [SIA (2026)](https://arxiv.org/abs/2605.27276) | Jointly changes agent harness and model weights | Calibrated source-selection VOI measured against counterfactual persistent updates |
| [SkillSmith (2026)](https://arxiv.org/abs/2607.27497) | Synthesizes prefix parameters from textual and parametric skills | Endogenous acquisition of discriminating diagnoses, with measured post-update value |
| [KV-Skill (2026)](https://arxiv.org/abs/2608.05475) | Converts text/reward into independently loadable skill operators | Durable evidence-seeking/update coupling rather than skill registration alone |
| [Distributional hypernetworks (Oct 2026)](https://arxiv.org/abs/2610.01934) | Models query-conditioned distributions over LoRA updates | Distinguish an expensive information source by verified adaptation benefit |
| [W2T (2026)](https://arxiv.org/abs/2603.15990) | Predicts LoRA properties from a canonical weight representation | Outcome-supervised, state/evidence-conditioned update choice |
| [BQ-LoRA (2026)](https://arxiv.org/abs/2609.12896) | Learns behavior-aware adaptation under a low-rank budget | Active heterogeneous evidence gathering for intervention decisions |
| [VI-MoLE (2026)](https://arxiv.org/abs/2608.02528) | Information-value allocation among LoRA experts | VOI over *persistent model changes* rather than inference-time expert activation |
| [Value of Information for Human–Agent Communication (ACL 2026)](https://aclanthology.org/2026.acl-long.1987/) | VOI for when to ask a human clarifying question | Focus on parameter update benefit, not immediate task communication |
| [DeepControl (EMNLP Findings 2026)](https://aclanthology.org/2026.findings-emnlp.52/) | Controls retrieval amount/detail for reasoning | Long-term parameter intervention instead of inference-time evidence management |
| [SkillCommit (2026)](https://arxiv.org/abs/2608.15165) | Behaviorally validated external skill-library growth | Independent learned adaptation of evidence and parameter-generation mechanisms |
| [SkillEvoReg (2026)](https://arxiv.org/abs/2609.30861) | Anti-overfitting and causal regression checks for external skills | The specific parameter-acquisition coupling and persistent weight audit |
| [Hyper-ES (2026)](https://arxiv.org/abs/2608.05541) / [CoPES (2026)](https://arxiv.org/abs/2608.02391) | Efficient evolution in meaningful parameter directions or subspaces | Evidence selection driven by future intervention outcomes |

**Do not present:** retrieval, teacher critiques, LoRA generation, low-rank functional bases, classical VOI, or iterative agent improvement as independently invented by this project.

**Potential new empirical finding:** source information and parameter direction interact. In the same observed failure stratum, another source changes which durable update is best; the learned acquisition policy predicts this reversal on held-out mechanisms, cheaply and with retention.

## Naming collision (important before public paper submission)

The acronym **SPEAR** is **already used in the same broad LLM-agent research area**:

- [Qin et al., *Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning*](https://arxiv.org/abs/2509.22601), submitted 2025; their method is named SPEAR and associated with ICLR 2026.
- [Yang et al., *SPEAR: A Unified SSL Framework for Learning Speech and Audio Representations*](https://proceedings.mlr.press/v306/yang26cv.html), ICML 2026 (different area).

This repository keeps **SPEAR** because it is the requested project name, but the paper title/acronym should be reconsidered before submission to avoid confusion. The scientific claims and equations do not depend on the acronym.

## Claim-to-source discipline

The attached research brief supplied the proposed problem framing: **parameter-aware evidence acquisition, counterfactual self-modification, functional intervention generation, and recursive policy learning**. The linked papers above support the related-work comparisons. They do **not** independently validate SPEAR or its expected performance.

The manuscript is dated 2026-10-30 and uses no future-looking publication claims. Any new references added later require a fresh source check.
