# SPEAR — parameter-aware self-improvement research

**Working manuscript:** [SPEAR: Learning What Evidence to Seek for Parameter-Level Self-Improvement in LLM Agents](paper/main.tex)

SPEAR (Self-improvement through Probing, Evidence Acquisition, and Reparameterization) investigates whether LLM agents can learn **which evidence to acquire in order to make a more beneficial durable model-parameter update**.

Core research objects:

1. **PI-VOI:** parameter-intervention value of information — a source is valuable only if it improves a subsequent parameter decision, net of cost.
2. **Evidence-conditioned functional parameter interventions:** constrained reuse/combination of validated update directions with an optional bounded residual.
3. **Counterfactual validation and recursive learning:** paired post-update outcomes train the evidence policy and intervention mechanism, with persistence and old-skill audits.

## Manuscript and research protocol

- [LaTeX entrypoint](paper/main.tex) and [paper directory guide](paper/README.md)
- [Experiment preregistration / validation gates](paper/EXPERIMENTS.md)
- [Literature and originality audit](paper/RELATED_WORK.md)
- [Bibliography](paper/references.bib)

Compile with \`cd paper && make\` in a TeX Live environment. A GitHub Actions build definition is included in the paper branch; PDF availability depends on GitHub Actions execution.

**Research status (2026-10-30):** method proposal and planned experiments; there are no measured SPEAR results in this draft. The paper does not claim an established scientific contribution before the causal and persistence tests succeed.

**Naming note:** a different agentic-RL method already used **SPEAR** in [Qin et al. (2025/ICLR 2026)](https://arxiv.org/abs/2509.22601). The acronym here is provisional pending a submission decision.
