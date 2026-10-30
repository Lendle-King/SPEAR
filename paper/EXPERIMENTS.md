# SPEAR — prospective experimental protocol (v0, 2026-10-30)

**Status:** research proposal, not a completed experiment. There are no measured SPEAR scores, claims of significant improvement, or performance tables yet. Do not replace "pending" with estimates.

## Central hypothesis

Under an equal end-to-end improvement budget, a policy trained to acquire evidence for the *best subsequent validated model-parameter intervention* yields better persistent held-out skill acquisition than a fixed evidence policy or one based solely on confidence, uncertainty, or visible error labels.

This claim is **conditional** on showing a source-choice/intervention-choice interaction. A fixed error-condition rule that matches the learned method invalidates the specific adaptive-acquisition claim in that setting.

## Required pilot: falsify before scaling

Construct a resettable tool-agent simulation with at least three distinct latent failure mechanisms but matched observable pre-query error messages:

- **Schema mismatch**: a tool field has changed. The source with unique new information is the current manual/schema record.
- **Task-invariant mismatch**: the tool call is well-formed but violates a hidden task constraint. An expert explanation of constraints can expose the cause.
- **Environment precondition mismatch**: the error requires testing a state-dependent prerequisite. An executable non-destructive probe provides discriminating observations.

These are *experimental designs*, not observations. Each source must have explicitly constrained access; if the teacher can infer every cause from hidden metadata, source selection becomes trivial and the experimental manipulation is invalid. Randomize tool IDs, descriptions, surface wording, evidence order, and combinations across train/dev/test.

For each matched visible-failure stratum, record both candidate-update outcomes from the same checkpoint and environment initialization. Require that the better candidate (including no update) differs for at least some latent causes. Without such reversals, **stop before fitting a large source router**.

## Four gates and go/no-go logic

| Gate | Minimum credible evidence | Fail / inconclusive outcome |
|---|---|---|
| 1. Evidence value | External evidence causes a measurable improvement in paired *post-update* utility versus trace-only and trace-paraphrase conditions at matched total cost | Return to intervention design; do not assert evidence advantage |
| 2. Source/intervention reversal | Same visible failure, different latent causes; choice of best update reverses after a source observation; learned acquisition beats fixed and simple-rule routing in held-out combinations | Do not claim parameter-aware acquisition is necessary |
| 3. Persistent skill and retention | Accepted checkpoint improves independent, full trajectories **without teacher text, documents, or query-specific adapter reload**; old-skill drop within prespecified tolerance | Describe transient routing/adaptation, not persistent learning |
| 4. Recursive meta-improvement | Continuing to update source policy and intervention generator beats **frozen** versions with the **same growing memory log** and evaluation budget on later skills | Do not claim recursive self-improvement |

Do not advance to later gates as established simply because an earlier pilot is promising.

## Trial structure and split isolation

1. Freeze base checkpoint (candidate Qwen3.5-4B), tokenizer, adapter configuration, tool schemas, model versions, and environment image.
2. Partition by **underlying mechanism and task family** first, then instantiate independent train, update-selection validation, meta-development, and final test episodes. No task paraphrase or reset seed may cross a boundary.
3. Use source-choice randomization with recorded propensities during evidence-value model data collection. Log the budget even if no source was queried.
4. Propose both rule-based or learned basis-only adapters and bounded residual candidates; always include zero update.
5. Branch identical pre-intervention checkpoints and resettable environment snapshots. Run baseline and candidate on paired seeds; record strict checker signals, episode completion, side effects, and actual cost.
6. Apply a prespecified accept/rollback rule to *validation* data only. Report both accepted and rejected candidates.
7. Persist the accepted adapter/checkpoint. For new episodes, turn off the original diagnosis, raw retrieved text, and the training-time source-query context.
8. Train the evidence and update policies on measured intervention outcomes, not the held-out test, and retain versioned logs.
9. Repeat in a sequential skill stream and compare against frozen-meta controls at exactly matched experience/archive growth.

**Do not use the same held-out task instance both to select a candidate and to report the final success rate.**

## Comparator inventory

- Unmodified student with no tools; memory-only/Reflexion-style external skill.
- Fixed teacher each failure + SFT/LoRA gradient update.
- Fixed heterogeneous sources + text-conditioned adapter prediction.
- Simple observable-state route; confidence/uncertainty source route.
- Contextual BO / low-dimensional functional ES with identical candidate budget.
- Learned source selector + fixed intervention generator.
- Fixed source selector + learned intervention generator.
- Joint source policy and learned update generator.
- Joint system with both meta-components frozen while memory continues to grow.

Budget **total** GPU-hours (candidate learning + evaluation), query calls, retrieved bytes/tokens, and simulator reset/execution counts. If baselines use fewer queries, either expose frontier plots or enforce matched caps; do not secretly gift one method extra trials.

## Metrics and analysis commitments

Primary: strict success over independently initialized *full held-out trajectories*, compared at matched cumulative budget.

Secondary: validated new-skill gain per budget; expected future skill success; old-skill average and worst-case degradation; update acceptance rate and false-positive acceptance; query cost; intervention-value MAE/regret/calibration; source-choice effect within visible-failure strata; later-skill learning efficiency; adaptive-minus-frozen-meta gain.

Report 95% uncertainty intervals, effect sizes, full denominators, and error categories. Use mechanism-level cluster/paired bootstraps or another predeclared appropriate method; correct for multiple confirmatory comparisons. Run a pilot power analysis to freeze the final sample count **before** final testing. Do not cite repeated stochastic rollouts on the same instance as independent task generalization.

## Mandatory source and intervention audit schema (illustrative)

Each record should include:

| Field | Meaning |
|---|---|
| `run_id`, `skill_family`, `mechanism_id`, `split` | Immutable trial identity and isolation |
| `model_checkpoint_sha`, `basis_sha`, `generator_version`, `source_policy_version` | Causal treatment provenance |
| `environment_image`, `reset_snapshot_id`, `task_seed` | Exact replay target |
| `visible_failure_id`, `trace_hash` | Matched observable state |
| `source_action`, `action_propensity`, `evidence_hash`, `source_provenance` | Acquisition assignment and response |
| `teacher_tokens`, `search_bytes`, `probe_steps`, `gpu_seconds` | Full intervention cost |
| `candidate_id`, `delta_hash`, `is_zero_update` | Proposal and null comparator |
| `baseline_checker`, `candidate_checker`, `episode_done`, `side_effects` | Actual paired outcome |
| `old_skill_pre`, `old_skill_post`, `accepted`, `rollback_reason` | Persistence / retention decision |
| `heldout_evaluation`, `selection_eligible` | Leakage control |

Unknown or technically errored outcomes must remain explicitly **unknown**; do not reclassify as model failure or exclude without reporting.

## Essential ablations

- No extra evidence; trace-only textual paraphrase; shuffled/unrelated text; oracle external evidence.
- Teacher vs documentation vs real probe and mixed-source availability.
- Basis-only, residual-only, random direction, static best direction, direct SFT/LoRA.
- No update; memory-only; temporary routing vs permanently saved update.
- No old-skill audit; no uncertainty correction; no acquisition cost.
- Learned acquisition only; learned intervention only; both; both frozen with growing memory.
- Observable-state rule with the **same input fields** as the learned router.
- Held-out combinations of surface text, tool names, error type, and latent mechanism.

## Reporting rule

Before a claimed result appears in the manuscript, provide the raw-run path, code commit, configuration digest, pre-registered analysis file, and checker audit. Empty result cells are preferable to invented numbers or comparisons across incompatible budgets.
