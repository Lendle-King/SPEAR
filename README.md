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

Compile with `cd paper && make` in a TeX Live environment. A GitHub Actions build definition is included in the paper branch; PDF availability depends on GitHub Actions execution.

**Research status (2026-10-30):** method proposal and planned experiments; there are no measured SPEAR results in this draft. The paper does not claim an established scientific contribution before the causal and persistence tests succeed.

**Naming note:** a different agentic-RL method already used **SPEAR** in [Qin et al. (2025/ICLR 2026)](https://arxiv.org/abs/2509.22601). The acronym here is provisional pending a submission decision.

## Executable verification experiments (CPU smoke test)

- [Reproducible PI-VOI intervention experiments](experiments/parameter_voi/README.md)
- [Synthetic seed-42 result audit](experiments/parameter_voi/results/interpretation.md) and [machine-readable summary](experiments/parameter_voi/results/synthetic_seed42_summary.json)
- [Real LLM paired-run audit validator](experiments/parameter_voi/external_audit.py) (requires independent real runner logs)
- [GitHub Actions validation](.github/workflows/verify-parameter-voi.yml): 19 invariant/unit tests, deterministic synthetic experiment, raw trial artifacts

**Important distinction:** the CPU experiment changes persistent parameters of a tiny toy model, **not** a Qwen/AndroidWorld agent. On the seeded synthetic benchmark PI-VOI does not meaningfully beat the strong same-input readiness rule, and its learned intervention generator ties the static semantic rule. These negative controls remain visible; genuine LLM evidence is not yet reported.

## New: real-weights LoRA intervention pilot

The [counterfactual real-LoRA experiment](experiments/real_lora/README.md) adds an
explicitly opt-in CUDA Transformers/PEFT runner for three matched tool failure
mechanisms (schema, policy, stale state), four training diagnosis conditions
(blind, docs, teacher, probe), independent adapters from a pinned base,
paired full-trajectory validation/test, and old-skill/compute audits.
[CPU protocol CI](.github/workflows/verify-real-lora-protocol.yml) checks the
split, checker and leakage safeguards without downloading a model.
**No Qwen3.5-4B training or AndroidWorld results are reported yet.**
