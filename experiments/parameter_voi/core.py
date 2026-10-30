"""Resettable and auditable toy-agent environment for PI-VOI gate tests.

Counterfactual world truth lives only inside Environment. Policies receive
public_context() and query() outputs; they never access the latent label.
Here the durable student model has two numeric parameters: a new skill weight
and an old skill weight. This is a mechanism smoke test, not an LLM surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Dict, Iterable, Literal, Optional, Tuple

SOURCES: Tuple[str, ...] = ("teacher", "docs", "probe")
KINDS: Tuple[str, ...] = ("schema", "constraint", "state")
PREFERRED_SOURCE = {"schema": "docs", "constraint": "teacher", "state": "probe"}
VISIBLE_FAILURE = "Tool execution failed: recovery required"
SOURCE_COST = 0.04                 # synthetic utility cost units, NOT dollars
VALIDATION_COST = 0.18             # per evaluated intervention, synthetic units
PROTECTION_COST = 0.02             # extra cost when preserving shared old weights


def stable_seed(*parts: object) -> int:
    message = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big")


@dataclass(frozen=True)
class PublicCase:
    """Everything available to a decision-maker before diagnostic queries."""
    case_id: str
    pair_id: str
    split: str
    context_kind: str
    source_ready: Tuple[bool, bool, bool]
    shared_parameters: bool
    phase: int
    visible_failure: str = VISIBLE_FAILURE

    def is_ready(self, source: str) -> bool:
        return self.source_ready[SOURCES.index(source)]

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id, "pair_id": self.pair_id, "split": self.split,
            "context_kind": self.context_kind,
            "source_ready": dict(zip(SOURCES, self.source_ready)),
            "shared_parameters": self.shared_parameters, "phase": self.phase,
            "visible_failure": self.visible_failure,
        }


@dataclass(frozen=True)
class World:
    context: PublicCase
    latent_direction: int       # intentionally withheld by Environment


@dataclass(frozen=True)
class Evidence:
    source: str
    signal: int                 # +1 / -1; docs signal means forbidden action
    text: str
    provenance: str
    def as_dict(self) -> dict:
        return {"source": self.source, "signal": self.signal,
                "text": self.text, "provenance": self.provenance}


@dataclass(frozen=True)
class Student:
    """A tiny persistent parameterized model; old/new skills are separate."""
    new_weight: float = 0.0
    old_weight: float = 1.5

    def predict(self, skill: Literal["new", "old"], x: int) -> int:
        weight = self.new_weight if skill == "new" else self.old_weight
        # A zero-weight model predicts +1 irrespective of its input.
        return 1 if abs(weight) < 1e-12 else (1 if weight * x > 0 else -1)

    def updated(self, direction: int, shared: bool, protect_old: bool) -> "Student":
        if direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0 or +1")
        leak = 2.5 if (direction and shared and not protect_old) else 0.0
        return Student(new_weight=self.new_weight + 2.0 * direction,
                       old_weight=self.old_weight - leak)


@dataclass(frozen=True)
class Trial:
    baseline_new: float
    updated_new: float
    baseline_old: float
    updated_old: float
    direction: int
    protect_old: bool
    accepted: bool
    reason: str
    validation_calls: int

    @property
    def gain(self) -> float:
        return self.updated_new - self.baseline_new if self.accepted else 0.0

    @property
    def old_drop(self) -> float:
        return max(0.0, self.baseline_old - self.updated_old) if self.accepted else 0.0


def cases(split: str, n: int, seed: int, phase: int = 0) -> list[World]:
    """Generate exact latent-opposite pairs with *identical public conditions*.

    n must be even; case/pair identifiers are unique across splits. The label
    is inaccessible via public_context, and the model never sees this object.
    """
    if n <= 0 or n % 2:
        raise ValueError("n must be a positive even integer")
    if phase not in (0, 1):
        raise ValueError("phase must be 0 or 1")
    rng = random.Random(stable_seed("worldgen", split, seed, phase))
    out = []
    for i in range(n // 2):
        kind = KINDS[rng.randrange(len(KINDS))]
        ready = tuple(rng.random() < 0.68 for _ in SOURCES)
        shared = rng.random() < 0.2
        pair_id = f"{split}-phase{phase}-pair{i:05d}"
        for sign in (-1, +1):
            context = PublicCase(
                case_id=f"{pair_id}-{rng.getrandbits(48):012x}",
                pair_id=pair_id, split=split, context_kind=kind,
                source_ready=ready, shared_parameters=shared, phase=phase,
            )
            out.append(World(context=context, latent_direction=sign))
    rng.shuffle(out)
    return out


class Environment:
    def __init__(self, world: World):
        self._world = world
        self._query_log: list[str] = []

    def public_context(self) -> PublicCase:
        return self._world.context

    @property
    def query_log(self) -> Tuple[str, ...]:
        return tuple(self._query_log)

    def query(self, source: str) -> Evidence:
        if source not in SOURCES:
            raise ValueError(f"Unknown evidence source: {source}")
        self._query_log.append(source)
        c = self._world.context
        if not c.is_ready(source):
            # Some queries return nothing. The chance is deterministic per
            # (world, source) for paired comparisons.
            if random.Random(stable_seed(c.case_id, source, "missing")).random() < 0.60:
                return Evidence(source, 0, "SOURCE_UNAVAILABLE", f"sim:{c.case_id}:{source}")
        p = 0.88 if source == PREFERRED_SOURCE[c.context_kind] else 0.68
        if not c.is_ready(source):
            p = 0.51
        if c.phase == 1 and c.is_ready(source):
            # Explicit environment change for the recursive-learning test.
            # Meta-adaptation must respond to a new source-quality regime.
            if source == "docs":
                p = max(0.51, p - 0.32)
            elif source == "probe":
                p = min(0.97, p + 0.20)
        correct = random.Random(stable_seed(c.case_id, source, "accuracy")).random() < p
        underlying = self._world.latent_direction if correct else -self._world.latent_direction
        # The manual reports the forbidden direction, not the target action.
        signal = -underlying if source == "docs" else underlying
        desc = ("FORBIDDEN_ACTION" if source == "docs" else "SUGGESTED_ACTION")
        word = "PLUS" if signal == +1 else "MINUS"
        return Evidence(source, signal, f"{desc}={word}", f"sim:{c.case_id}:{source}")

    def strict_success(self, student: Student, skill: str, seed: int, n: int = 24) -> float:
        if n <= 0 or n % 2:
            raise ValueError("n must be a positive even integer")
        if skill not in ("new", "old"):
            raise ValueError("invalid skill")
        # Task instance ids/seed differ between validation and final testing.
        rng = random.Random(stable_seed(self._world.context.case_id, skill, seed))
        inputs = [-1, 1] * (n // 2)
        rng.shuffle(inputs)
        target = self._world.latent_direction if skill == "new" else +1
        correct = sum(student.predict(skill, x) == target * x for x in inputs)
        return correct / n

    def paired_trial(self, student: Student, direction: int, protect_old: bool = True,
                     validation_seed: int = 70, retention_tolerance: float = 0.05) -> Trial:
        if direction not in (-1, 0, 1):
            raise ValueError("direction must be -1/0/+1")
        old = self.strict_success(student, "old", validation_seed)
        before = self.strict_success(student, "new", validation_seed)
        if direction == 0:
            return Trial(before, before, old, old, 0, protect_old, False,
                         "no_update", 0)
        new_student = student.updated(direction, self._world.context.shared_parameters,
                                       protect_old)
        after = self.strict_success(new_student, "new", validation_seed)
        old_after = self.strict_success(new_student, "old", validation_seed)
        if old - old_after > retention_tolerance:
            return Trial(before, after, old, old_after, direction, protect_old, False,
                         "retention_regression", 1)
        if after <= before:
            return Trial(before, after, old, old_after, direction, protect_old, False,
                         "no_positive_validation_gain", 1)
        return Trial(before, after, old, old_after, direction, protect_old, True,
                     "accepted", 1)

    def heldout_gain(self, student: Student, trial: Trial, heldout_seed: int = 1007) -> Tuple[float, float]:
        base = self.strict_success(student, "new", heldout_seed)
        old_base = self.strict_success(student, "old", heldout_seed)
        if trial.accepted:
            updated = student.updated(trial.direction,
                                      self._world.context.shared_parameters,
                                      trial.protect_old)
        else:
            updated = student
        return (self.strict_success(updated, "new", heldout_seed) - base,
                max(0., old_base - self.strict_success(updated, "old", heldout_seed)))
