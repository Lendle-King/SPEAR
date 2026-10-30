"""Outcome-supervised intervention and parameter-aware source-value learners.

The intervention labels are derived from paired *model executions* on a
separate development split. No learner ever reads World.latent_direction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

from .core import (Environment, Evidence, PublicCase, SOURCES, PREFERRED_SOURCE,
                   Student, SOURCE_COST, VALIDATION_COST, PROTECTION_COST,
                   World, stable_seed)


def logistic(v: float) -> float:
    v = max(-30.0, min(30.0, v))
    return 1.0 / (1.0 + math.exp(-v))


def direction_features(case: PublicCase, evidence: Evidence) -> tuple[float, ...]:
    """All features available after evidence, multiplied by signal polarity."""
    if evidence.signal not in (-1, 0, 1):
        raise ValueError("invalid evidence sign")
    x = [1.0]
    for source in SOURCES:
        selected = float(source == evidence.source)
        x.extend([selected,
                  selected * float(case.is_ready(source)),
                  selected * float(PREFERRED_SOURCE[case.context_kind] == source),
                  selected * float(case.phase == 1)])
    return tuple(evidence.signal * v for v in x)


@dataclass
class DirectionModel:
    """Small trainable functional-direction model: outputs {-1,0,+1}.

    The student owns *persistent* task parameters; this meta-model predicts
    their update direction from evidence. SGD updates the meta-model weights.
    """
    coefficients: list[float]
    examples_seen: int = 0

    @classmethod
    def empty(cls) -> "DirectionModel":
        return cls([0.0] * (1 + 4 * len(SOURCES)))

    def train(self, examples: list[tuple[PublicCase, Evidence, int]],
              epochs: int = 20, lr: float = .17, seed: int = 0) -> None:
        rng = random.Random(stable_seed("direction_training", seed))
        items = list(examples)
        self.examples_seen += len(items)
        for epoch in range(epochs):
            rng.shuffle(items)
            for c, e, target in items:
                if e.signal == 0:
                    continue
                if target not in (-1, 1):
                    raise ValueError("target must be validated winning direction")
                x = direction_features(c, e)
                p = logistic(sum(w * z for w, z in zip(self.coefficients, x)))
                err = (1.0 if target == +1 else 0.0) - p
                for j, z in enumerate(x):
                    self.coefficients[j] += lr * (err * z - .001 * self.coefficients[j])

    def propose(self, case: PublicCase, evidence: Evidence) -> int:
        if evidence.signal == 0:
            return 0
        x = direction_features(case, evidence)
        score = sum(w * z for w, z in zip(self.coefficients, x))
        return +1 if score >= 0 else -1


class StaticSemanticDirection:
    """Strong non-learned baseline respecting that docs mark forbidden actions."""
    examples_seen = 0

    @staticmethod
    def propose(case: PublicCase, evidence: Evidence) -> int:
        return -evidence.signal if evidence.source == "docs" else evidence.signal


@dataclass(frozen=True)
class AcquisitionLog:
    """Training evidence and *real* post-update benefit (synthetic world)."""
    context: PublicCase
    source: str
    signal: int
    proposed_direction: int
    realized_net_utility: float
    accepted: bool
    validation_calls: int


@dataclass
class TabularPIVOI:
    """Empirical parameter-intervention VOI, not raw explanation accuracy.

    We use source + observable condition + phase strata with shrinkage to
    phase-agnostic means. Each label measures a subsequent proposed weight
    update's validated future-task utility minus acquisition/test cost.
    """
    records: list[AcquisitionLog]
    smoothing: float = 5.0

    def __init__(self, smoothing: float = 5.0) -> None:
        self.records = []
        self.smoothing = smoothing

    def fit(self, records: Iterable[AcquisitionLog]) -> None:
        self.records = list(records)

    def estimated_net(self, case: PublicCase, source: str) -> float:
        if source not in SOURCES:
            raise ValueError(source)
        condition = [r for r in self.records
                     if (r.source == source and r.context.context_kind == case.context_kind
                         and r.context.is_ready(source) == case.is_ready(source))]
        prior = (sum(r.realized_net_utility for r in condition) / len(condition)
                 if condition else 0.0)
        same_phase = [r for r in condition if r.context.phase == case.phase]
        return ((sum(r.realized_net_utility for r in same_phase) + self.smoothing * prior)
                / (len(same_phase) + self.smoothing))

    def choose(self, case: PublicCase, allow_stop: bool = True) -> str | None:
        scored = [(self.estimated_net(case, s), s) for s in SOURCES]
        best = max(scored, key=lambda pair: pair[0])
        return best[1] if (best[0] > 0 or not allow_stop) else None


def single_source_net(env: Environment, proposal: int, case: PublicCase,
                      val_seed: int, heldout_seed: int) -> tuple[float, bool, int]:
    """An evidence action pays for its query and every tested candidate."""
    trial = env.paired_trial(Student(), proposal, protect_old=True,
                             validation_seed=val_seed)
    actual_gain, old_drop = env.heldout_gain(Student(), trial, heldout_seed)
    value = (actual_gain - SOURCE_COST - VALIDATION_COST * trial.validation_calls
             - (PROTECTION_COST if case.shared_parameters and trial.validation_calls else 0))
    # old_drop is zero if the guard is correct; retained here to prevent
    # accidental forgetting being counted as positive utility.
    return value - old_drop, trial.accepted, trial.validation_calls


def training_logs(worlds: list[World], generator: DirectionModel | StaticSemanticDirection,
                  val_seed: int, heldout_seed: int) -> list[AcquisitionLog]:
    logs: list[AcquisitionLog] = []
    for world in worlds:
        c = world.context
        for source in SOURCES:
            env = Environment(world)
            evidence = env.query(source)
            direction = generator.propose(c, evidence)
            value, accepted, calls = single_source_net(env, direction, c, val_seed, heldout_seed)
            logs.append(AcquisitionLog(c, source, evidence.signal, direction,
                                       value, accepted, calls))
    return logs


def shared_training_logs(worlds: list[World],
                         learned_generator: DirectionModel,
                         val_seed: int, heldout_seed: int) -> tuple[list[AcquisitionLog], list[AcquisitionLog]]:
    """Both learned/static generators reuse the *same* queried evidence.

    Their update candidates are evaluated independently. Report both sets
    of candidate-validation costs; the evidence acquisition is shared.
    """
    learned_records, static_records = [], []
    static = StaticSemanticDirection()
    for world in worlds:
        c = world.context
        for source in SOURCES:
            env = Environment(world)
            evidence = env.query(source)
            for generator, records in ((learned_generator, learned_records),
                                       (static, static_records)):
                direction = generator.propose(c, evidence)
                value, accepted, calls = single_source_net(env, direction, c,
                                                            val_seed, heldout_seed)
                records.append(AcquisitionLog(c, source, evidence.signal,
                                              direction, value, accepted, calls))
    return learned_records, static_records


def train_direction(worlds: list[World], seed: int) -> tuple[DirectionModel, dict]:
    examples = []
    examined = 0
    for world in worlds:
        for source in SOURCES:
            env = Environment(world)
            evidence = env.query(source)
            if evidence.signal == 0:
                continue
            up = env.paired_trial(Student(), +1, validation_seed=701)
            down = env.paired_trial(Student(), -1, validation_seed=701)
            # Labels are based solely on paired checker outcomes. These are
            # intervention trial outcomes, not an oracle latent-cause read.
            target = +1 if up.updated_new > down.updated_new else -1
            examples.append((world.context, evidence, target))
            examined += 2
    model = DirectionModel.empty()
    model.train(examples, seed=seed)
    return model, {"training_evidence_examples": len(examples),
                   "counterfactual_candidate_evaluations": examined}


def fit_pipeline(direction_worlds: list[World], source_worlds: list[World],
                 seed: int) -> tuple[DirectionModel, TabularPIVOI, TabularPIVOI, dict]:
    """Use disjoint *pairs* for the two fitted meta-components."""
    a = set(w.context.pair_id for w in direction_worlds)
    b = set(w.context.pair_id for w in source_worlds)
    if not a.isdisjoint(b):
        raise ValueError("source-value and direction-learning world pairs overlap")
    generator, audit = train_direction(direction_worlds, seed)
    learned_logs, static_logs = shared_training_logs(source_worlds, generator, 702, 703)
    source_model = TabularPIVOI()
    source_model.fit(learned_logs)
    static_source_model = TabularPIVOI()
    static_source_model.fit(static_logs)
    audit.update({"source_value_records": len(learned_logs),
                  "direction_training_source_queries": len(direction_worlds) * len(SOURCES),
                  "source_training_shared_queries": len(learned_logs),
                  "source_training_candidate_evaluations": sum(r.validation_calls for r in learned_logs),
                  "static_source_training_candidate_evaluations": sum(r.validation_calls for r in static_logs),
                  "static_source_value_records": len(static_logs),
                  "toy_training_query_cost_units": round((len(direction_worlds) * len(SOURCES) + len(learned_logs)) * SOURCE_COST, 6),
                  "toy_training_candidate_cost_units": round((audit["counterfactual_candidate_evaluations"]
                      + sum(r.validation_calls for r in learned_logs)
                      + sum(r.validation_calls for r in static_logs)) * VALIDATION_COST, 6)})
    return generator, source_model, static_source_model, audit


def split_pairs(worlds: list[World]) -> tuple[list[World], list[World]]:
    """Split by matched pair, not by individual opposite-sign examples."""
    pair_ids = sorted({w.context.pair_id for w in worlds})
    first = set(pair_ids[:len(pair_ids) // 2])
    return ([w for w in worlds if w.context.pair_id in first],
            [w for w in worlds if w.context.pair_id not in first])
