"""Reproducible PI-VOI mechanism validation (synthetic CPU-only experiment).

Example:
  python -m experiments.parameter_voi.run --seed 42 --train 360 --test 240 --out /tmp/spear

The output has real measured outcomes *from the supplied toy environment*.
It must never be labeled as a Qwen/AndroidWorld/LLM Agent result.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
from typing import Iterable, Optional

from .core import (Environment, PublicCase, Student, World, SOURCES,
                   PREFERRED_SOURCE, SOURCE_COST, VALIDATION_COST,
                   PROTECTION_COST, cases, stable_seed)
from .learning import (DirectionModel, StaticSemanticDirection,
                       TabularPIVOI, fit_pipeline, split_pairs, training_logs)

METHODS = (
    "no_update", "random_no_query", "exhaustive_no_query",
    "fixed_teacher_static", "fixed_teacher_learned",
    "fixed_docs_learned", "fixed_probe_learned", "random_source_learned",
    "kind_rule_learned", "ready_rule_learned",
    "global_best_learned", "pi_voi_static", "pi_voi_learned",
)


@dataclass(frozen=True)
class TrialRecord:
    method: str
    case_id: str
    pair_id: str
    split: str
    phase: int
    context_kind: str
    source: Optional[str]
    source_signal: int
    proposed_direction: int
    accepted: bool
    rollback_reason: str
    validation_calls: int
    source_calls: int
    candidate_cost: float
    source_cost: float
    heldout_new_gain: float
    heldout_old_drop: float
    strict_heldout_success: float
    realized_net: float
    persisted_new_weight: float
    persisted_old_weight: float
    heldout_evidence_disabled: bool


def choose_source(method: str, case: PublicCase, value: TabularPIVOI,
                  seed: int) -> str | None:
    if method in ("no_update", "random_no_query", "exhaustive_no_query"):
        return None
    if method.startswith("fixed_teacher"):
        return "teacher"
    if method.startswith("fixed_docs"):
        return "docs"
    if method.startswith("fixed_probe"):
        return "probe"
    if method == "random_source_learned":
        return SOURCES[stable_seed("random_source", seed, case.case_id) % len(SOURCES)]
    if method == "kind_rule_learned":
        return PREFERRED_SOURCE[case.context_kind]
    if method == "ready_rule_learned":
        preferred = PREFERRED_SOURCE[case.context_kind]
        return (preferred if case.is_ready(preferred) else
                next((s for s in SOURCES if case.is_ready(s)), preferred))
    if method == "global_best_learned":
        return max(SOURCES, key=lambda s: (sum(r.realized_net_utility
                                          for r in value.records if r.source == s)
                                        / max(1, sum(r.source == s for r in value.records))))
    if method.startswith("pi_voi"):
        return value.choose(case, allow_stop=True)
    raise ValueError(f"unknown method: {method}")


def apply_policy(world: World, method: str, learned: DirectionModel,
                 source_value: TabularPIVOI,
                 static_source_value: TabularPIVOI,
                 seed: int = 0) -> TrialRecord:
    if method not in METHODS:
        raise ValueError(method)
    c = world.context
    env = Environment(world)
    student = Student()
    source = choose_source(method, c,
                           static_source_value if method == "pi_voi_static" else source_value,
                           seed)
    signal = 0
    direction = 0
    source_calls = 0
    trial_count = 0
    source_cost = 0.0
    if method == "exhaustive_no_query":
        # This strong baseline asks for no diagnostic evidence and tests both
        # interventions. It incurs twice the *candidate evaluation* cost.
        pos = env.paired_trial(student, +1, protect_old=True, validation_seed=703)
        neg = env.paired_trial(student, -1, protect_old=True, validation_seed=703)
        accepted = [t for t in (pos, neg) if t.accepted]
        trial = max(accepted, key=lambda t: t.updated_new) if accepted else pos
        trial_count = pos.validation_calls + neg.validation_calls
        direction = trial.direction
        reason = trial.reason
    else:
        if method == "random_no_query":
            direction = +1 if stable_seed("random_direction", seed, c.pair_id) % 2 else -1
        elif source is not None:
            source_calls = 1
            source_cost = SOURCE_COST
            evidence = env.query(source)
            signal = evidence.signal
            generator = (StaticSemanticDirection() if method in
                         ("fixed_teacher_static", "pi_voi_static") else learned)
            direction = generator.propose(c, evidence)
        trial = env.paired_trial(student, direction, protect_old=True,
                                 validation_seed=703)
        trial_count = trial.validation_calls
        reason = trial.reason
    new_gain, old_drop = env.heldout_gain(student, trial, heldout_seed=1007)
    candidate_cost = (trial_count * VALIDATION_COST
                      + (PROTECTION_COST if trial_count and c.shared_parameters else 0.0))
    cost = source_cost + candidate_cost
    updated = student.updated(trial.direction, c.shared_parameters,
                              trial.protect_old) if trial.accepted else student
    return TrialRecord(
        method=method, case_id=c.case_id, pair_id=c.pair_id, split=c.split,
        phase=c.phase, context_kind=c.context_kind, source=source,
        source_signal=signal, proposed_direction=direction,
        accepted=trial.accepted, rollback_reason=reason,
        validation_calls=trial_count, source_calls=source_calls,
        candidate_cost=round(candidate_cost, 6), source_cost=round(source_cost, 6),
        heldout_new_gain=round(new_gain, 6), heldout_old_drop=round(old_drop, 6),
        strict_heldout_success=round(.5 + new_gain, 6),
        realized_net=round(new_gain - old_drop - cost, 6),
        persisted_new_weight=updated.new_weight,
        persisted_old_weight=updated.old_weight,
        heldout_evidence_disabled=True,
    )


def evaluate(worlds: list[World], generator: DirectionModel,
             source_value: TabularPIVOI, static_value: TabularPIVOI,
             seed: int, methods: Iterable[str] = METHODS) -> list[TrialRecord]:
    chosen = tuple(methods)
    return [apply_policy(w, m, generator, source_value, static_value, seed)
            for w in worlds for m in chosen]


def summary(rows: list[TrialRecord]) -> dict:
    groups: dict[str, list[TrialRecord]] = defaultdict(list)
    for row in rows:
        groups[row.method].append(row)
    result = {}
    for method, trials in sorted(groups.items()):
        n = len(trials)
        result[method] = {
            "n_cases": n,
            "n_independent_matched_pairs": len({t.pair_id for t in trials}),
            "strict_success": round(statistics.mean(t.strict_heldout_success for t in trials), 6),
            "new_skill_gain": round(statistics.mean(t.heldout_new_gain for t in trials), 6),
            "old_skill_drop": round(statistics.mean(t.heldout_old_drop for t in trials), 6),
            "net_utility": round(statistics.mean(t.realized_net for t in trials), 6),
            "mean_synthetic_cost": round(statistics.mean(t.candidate_cost + t.source_cost for t in trials), 6),
            "mean_source_calls": round(statistics.mean(t.source_calls for t in trials), 6),
            "mean_candidate_validations": round(statistics.mean(t.validation_calls for t in trials), 6),
            "accepted_updates": sum(t.accepted for t in trials),
            "rejected_updates": sum(t.proposed_direction != 0 and not t.accepted for t in trials),
            "no_update_proposals": sum(t.proposed_direction == 0 for t in trials),
        }
    return result


def paired_bootstrap(rows: list[TrialRecord], method: str, reference: str,
                     seed: int, repeats: int = 600) -> dict:
    by_pair: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.method in (method, reference):
            by_pair[row.pair_id][row.method].append(row.realized_net)
    pairs = sorted(pair for pair, d in by_pair.items()
                   if method in d and reference in d)
    if not pairs:
        return {"n_pairs": 0, "mean_difference": None, "ci_95": None}
    deltas = [statistics.mean(by_pair[p][method]) - statistics.mean(by_pair[p][reference])
              for p in pairs]
    rng = random.Random(stable_seed("bootstrap", seed, method, reference))
    boots = sorted(statistics.mean(deltas[rng.randrange(len(deltas))]
                                   for _ in deltas) for _ in range(repeats))
    lo = boots[int(.025 * (repeats-1))]
    hi = boots[int(.975 * (repeats-1))]
    return {"n_pairs": len(deltas),
            "mean_difference": round(statistics.mean(deltas), 6),
            "ci_95": [round(lo, 6), round(hi, 6)],
            "method": method, "reference": reference,
            "unit": "synthetic utility per case, grouped bootstrap by matched pair"}


def reversal_audit(worlds: list[World]) -> dict:
    grouped: dict[str, list[World]] = defaultdict(list)
    for world in worlds:
        grouped[world.context.pair_id].append(world)
    reversed_count = 0
    invalid = 0
    for pair_id, group in grouped.items():
        if len(group) != 2:
            invalid += 1
            continue
        contexts = [(w.context.context_kind, w.context.source_ready,
                     w.context.shared_parameters, w.context.phase,
                     w.context.visible_failure) for w in group]
        if contexts[0] != contexts[1]:
            invalid += 1
            continue
        winners = []
        for world in group:
            env = Environment(world)
            p = env.paired_trial(Student(), +1)
            n = env.paired_trial(Student(), -1)
            winners.append(+1 if p.updated_new > n.updated_new else -1)
        if winners[0] == -winners[1]:
            reversed_count += 1
    return {"matched_public_failure_pairs": len(grouped),
            "valid_matched_pairs": len(grouped) - invalid,
            "pairs_with_opposite_best_parameter_update": reversed_count,
            "constructed_by_design": True,
            "interpretation": "Synthetic reversal gate only; no inference about real LLMs."}


def retention_audit(worlds: list[World]) -> dict:
    shared = [w for w in worlds if w.context.shared_parameters]
    failures_blocked = 0
    for world in shared:
        env = Environment(world)
        t = env.paired_trial(Student(), +1, protect_old=False)
        if t.reason == "retention_regression" and not t.accepted:
            failures_blocked += 1
    return {"shared_parameter_cases": len(shared),
            "unsafe_updates_rejected": failures_blocked,
            "note": "Toy old-weight spillover; not evidence of retention for LLMs."}


def ensure_disjoint(*groups: list[World]) -> None:
    pair_ids = [set(w.context.pair_id for w in group) for group in groups]
    case_ids = [set(w.context.case_id for w in group) for group in groups]
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            if (pair_ids[i] & pair_ids[j]) or (case_ids[i] & case_ids[j]):
                raise ValueError("leakage: evaluation and training splits overlap")


def run_experiment(seed: int = 42, train: int = 360, test: int = 240,
                   meta_train: int = 240, meta_test: int = 240,
                   out: Path | None = None) -> dict:
    if min(train, test, meta_train, meta_test) < 4:
        raise ValueError("each split must contain at least 4 worlds")
    if any(n % 4 for n in (train, meta_train)) or test % 2 or meta_test % 2:
        raise ValueError("train and meta_train must be divisible by 4; test counts even")

    train_worlds = cases("train", train, seed, phase=0)
    gen_worlds, value_worlds = split_pairs(train_worlds)
    test_worlds = cases("test", test, seed+17, phase=0)
    adapt_worlds = cases("meta-adapt", meta_train, seed+29, phase=1)
    adapt_gen, adapt_value = split_pairs(adapt_worlds)
    final_worlds = cases("meta-test", meta_test, seed+31, phase=1)
    ensure_disjoint(gen_worlds, value_worlds, test_worlds,
                    adapt_gen, adapt_value, final_worlds)

    direction, value, static_value, training_audit = fit_pipeline(
        gen_worlds, value_worlds, seed)
    results = evaluate(test_worlds, direction, value, static_value, seed)
    means = summary(results)
    baseline_list = ("kind_rule_learned", "ready_rule_learned",
                     "fixed_teacher_learned", "global_best_learned",
                     "random_source_learned", "exhaustive_no_query",
                     "pi_voi_static")
    differences = [paired_bootstrap(results, "pi_voi_learned", name, seed)
                   for name in baseline_list]

    # Full stage-1 meta archive available identically to both arms; frozen
    # refuses to *train on it*; adapted refits the two meta-components.
    gen_new, value_new, static_new, meta_training_audit = fit_pipeline(
        gen_worlds + adapt_gen, value_worlds + adapt_value, seed+1)
    frozen_results = evaluate(final_worlds, direction, value, static_value,
                              seed, methods=("pi_voi_learned",))
    adapted_results = evaluate(final_worlds, gen_new, value_new, static_new,
                               seed, methods=("pi_voi_learned",))
    frozen_results = [TrialRecord(**{**asdict(r), "method": "frozen_meta"})
                      for r in frozen_results]
    adapted_results = [TrialRecord(**{**asdict(r), "method": "adaptive_meta"})
                       for r in adapted_results]
    meta_rows = frozen_results + adapted_results
    recursive = {
        "status": "controlled meta-learning stress test, NOT proof of recursive LLM improvement",
        "phase0_replay_cases": len(gen_worlds) + len(value_worlds),
        "phase1_replay_cases_both_arms": len(adapt_worlds),
        "frozen_archive_cases": train + meta_train,
        "adaptive_archive_cases": train + meta_train,
        "adapted_meta_fit_records": meta_training_audit,
        "method_summary": summary(meta_rows),
        "paired_difference": paired_bootstrap(meta_rows, "adaptive_meta",
                                               "frozen_meta", seed+5),
    }
    output = {
        "protocol": "SPEAR synthetic parameter-intervention experiment v0.1",
        "claim_boundary": "No LLM checkpoint, teacher API, real retrieval, AndroidWorld or AppWorld was run.",
        "seed": seed,
        "split_sizes": {"direction_train": len(gen_worlds),
                        "source_value_train": len(value_worlds),
                        "final_phase0_test": len(test_worlds),
                        "meta_adaptation": len(adapt_worlds),
                        "final_phase1_test": len(final_worlds)},
        "training_audit": training_audit,
        "reversal_audit": reversal_audit(test_worlds),
        "retention_audit": retention_audit(test_worlds),
        "phase0_summary": means,
        "phase0_paired_net_utility_comparisons": differences,
        "recursive_stress_test": recursive,
        "gate_outcomes": {
            "G1": "synthetic evidence intervention comparison recorded; check paired CIs before claiming improvement",
            "G2": "constructed latent-opposite counterfactuals; compare learned PI-VOI to strong metadata rule",
            "G3": "toy numeric weight persistence and guard verified; real LLM persistence NOT TESTED",
            "G4": "frozen-versus-updated meta-components compared with identical replay archive; real RSI NOT TESTED",
        },
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
        with (out / "phase0_trials.jsonl").open("w") as fp:
            for record in results:
                fp.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        with (out / "recursive_trials.jsonl").open("w") as fp:
            for record in meta_rows:
                fp.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=360)
    parser.add_argument("--test", type=int, default=240)
    parser.add_argument("--meta-train", type=int, default=240)
    parser.add_argument("--meta-test", type=int, default=240)
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_voi"))
    args = parser.parse_args()
    output = run_experiment(seed=args.seed, train=args.train, test=args.test,
                            meta_train=args.meta_train, meta_test=args.meta_test,
                            out=args.out)
    print(f"Wrote synthetic audit records to {args.out}")
    print(json.dumps({"phase0_summary": output["phase0_summary"],
                      "paired_comparisons": output["phase0_paired_net_utility_comparisons"],
                      "recursion": output["recursive_stress_test"]["paired_difference"]},
                     indent=2))


if __name__ == "__main__":
    main()
