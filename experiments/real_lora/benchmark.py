"""Resettable multi-step tool failures with matched initial observations.

This is a controlled *environment*, not AndroidWorld. Mechanism and payload
truth may be inspected by the benchmark only, never by the student prompt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Any, Callable

MECHANISMS = ("schema", "policy", "stale")
SOURCES = ("blind", "docs", "teacher", "probe")
SOURCE_FOR = {"schema": "docs", "policy": "teacher", "stale": "probe"}
SOURCE_FIXTURE_COST = {"blind": 0, "docs": 1, "teacher": 1, "probe": 1}
GENERIC_FAILURE = "E_REJECTED: operation rejected. Reason unavailable."
INITIAL_TASK = ("An attempted write to the transaction service failed: "
                + GENERIC_FAILURE + " Repair safely and submit the transaction. "
                "You can inspect the failure using the local tool; do not guess private values.")
SYSTEM = ("You are a deterministic transaction repair agent. Respond with exactly "
          "one JSON object per turn, no commentary. Tool calls must have shape "
          '{"tool":"tool_name","args":{...}}. Available tools: '
          "inspect_failure() to read local error metadata; "
          "rename_field(from,to) to rename the submitted field; "
          "cap_units(units) to set a safe quantity; "
          "refresh_revision(revision) to supply a current revision; "
          "submit() to finish. Start by inspecting, then apply one correct "
          "repair, then submit. Do not submit before the repair is confirmed.")


def stable_int(*parts: Any) -> int:
    value = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


@dataclass(frozen=True)
class Case:
    case_id: str
    family_id: str
    split: str
    mechanism: str
    old_field: str
    new_field: str
    requested_units: int
    cap: int
    old_revision: str
    current_revision: str

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, value: dict) -> "Case":
        if value["mechanism"] not in MECHANISMS:
            raise ValueError("unsupported hidden mechanism")
        return cls(**value)


def make_cases(seed: int, groups: dict[str, int]) -> list[Case]:
    if not groups or any(n < 1 for n in groups.values()):
        raise ValueError("nonempty split group counts must be positive")
    seen: set[str] = set()
    cases: list[Case] = []
    for split, count in groups.items():
        if split not in ("train", "validation", "test"):
            raise ValueError("unknown split")
        for index in range(count):
            family_id = f"{split}-family-{index:04d}"
            if family_id in seen:
                raise ValueError("duplicate family")
            seen.add(family_id)
            for mechanism in MECHANISMS:
                rng = random.Random(stable_int("case", seed, family_id, mechanism))
                old_field = f"key_{rng.randrange(1000, 9999)}"
                new_field = f"key_{rng.randrange(10000, 99999)}"
                cap = rng.randrange(3, 30)
                requested = cap + rng.randrange(1, 10)
                cases.append(Case(
                    case_id=f"{family_id}-{mechanism}", family_id=family_id,
                    split=split, mechanism=mechanism,
                    old_field=old_field, new_field=new_field,
                    requested_units=requested, cap=cap,
                    old_revision=f"rev_{rng.randrange(100, 999)}",
                    current_revision=f"rev_{rng.randrange(1000, 9999)}",
                ))
    validate_cases(cases)
    return cases


def validate_cases(cases: list[Case]) -> None:
    if not cases:
        raise ValueError("no cases")
    ids = [c.case_id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case_id")
    family_split: dict[str, str] = {}
    groups: dict[str, set[str]] = {}
    for case in cases:
        if case.mechanism not in MECHANISMS or case.split not in ("train", "validation", "test"):
            raise ValueError("invalid case metadata")
        if case.family_id in family_split and family_split[case.family_id] != case.split:
            raise ValueError("family contamination across splits")
        family_split[case.family_id] = case.split
        groups.setdefault(case.family_id, set()).add(case.mechanism)
    for family, kinds in groups.items():
        if kinds != set(MECHANISMS):
            raise ValueError(f"family {family} must contain all three mechanisms")


class Transaction:
    """No hidden attributes in observation; only tool results reveal state."""

    def __init__(self, case: Case, max_steps: int = 5):
        self.case = case
        self.max_steps = max_steps
        self.steps = 0
        self.did_inspect = False
        self.fixed = False
        self.done = False
        self.success = False
        self.side_effects = 0
        self.history: list[dict] = []

    def initial_messages(self) -> list[dict[str, str]]:
        # The full *initial* prompt is bit-identical across the three cases
        # inside every family: no mechanism, IDs, or private fields leaked.
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": INITIAL_TASK}]

    def inspect_data(self) -> dict:
        c = self.case
        if c.mechanism == "schema":
            return {"observed_field": c.old_field,
                    "required_field": c.new_field,
                    "observed_problem": "field mapping inconsistent"}
        if c.mechanism == "policy":
            return {"requested_units": c.requested_units,
                    "safe_cap": c.cap,
                    "observed_problem": "quantity exceeds allowed bound"}
        return {"submitted_revision": c.old_revision,
                "latest_revision": c.current_revision,
                "observed_problem": "revision is out of date"}

    def step(self, action: dict) -> dict:
        if self.done:
            raise RuntimeError("episode already done")
        if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
            action = {"tool": "INVALID", "args": {}}
        tool = action["tool"]
        args = action.get("args", {})
        if not isinstance(args, dict):
            args = {}
        self.steps += 1
        if tool == "inspect_failure":
            self.did_inspect = True
            response: dict = {"status": "observed", "detail": self.inspect_data()}
        elif tool in ("rename_field", "cap_units", "refresh_revision"):
            c = self.case
            correct = False
            if self.did_inspect:
                if c.mechanism == "schema" and tool == "rename_field":
                    correct = args.get("from") == c.old_field and args.get("to") == c.new_field
                elif c.mechanism == "policy" and tool == "cap_units":
                    correct = type(args.get("units")) is int and args["units"] == c.cap
                elif c.mechanism == "stale" and tool == "refresh_revision":
                    correct = args.get("revision") == c.current_revision
            if correct:
                self.fixed = True
                response = {"status": "prepared", "ok": True}
            else:
                self.side_effects += 1
                response = {"status": "rejected", "ok": False, "error": GENERIC_FAILURE}
        elif tool == "submit":
            self.done = True
            self.success = self.did_inspect and self.fixed and not self.side_effects
            response = {"status": "committed" if self.success else "rejected",
                        "ok": self.success}
        else:
            self.side_effects += 1
            response = {"status": "rejected", "ok": False, "error": "INVALID_TOOL"}
        if self.steps >= self.max_steps:
            self.done = True
        self.history.append({"action": action, "observation": response})
        return response

    def gold_actions(self) -> list[dict]:
        c = self.case
        if c.mechanism == "schema":
            repair = {"tool": "rename_field", "args": {"from": c.old_field, "to": c.new_field}}
        elif c.mechanism == "policy":
            repair = {"tool": "cap_units", "args": {"units": c.cap}}
        else:
            repair = {"tool": "refresh_revision", "args": {"revision": c.current_revision}}
        return [{"tool": "inspect_failure", "args": {}}, repair,
                {"tool": "submit", "args": {}}]


def source_report(case: Case, source: str) -> dict:
    """Controlled fixture, NOT a network teacher, web retrieval or real probe."""
    if source not in SOURCES:
        raise ValueError("unknown evidence source")
    env = Transaction(case)
    correct = source != "blind" and SOURCE_FOR[case.mechanism] == source
    if source == "blind":
        report = "No external diagnosis was supplied; use local observations."
    elif not correct:
        report = "The queried source found no applicable rule for this failure."
    elif source == "docs":
        detail = env.inspect_data()
        report = ("Current API documentation: a rejected field can be repaired "
                  f"by renaming {detail['observed_field']} to {detail['required_field']}.")
    elif source == "teacher":
        detail = env.inspect_data()
        report = ("Policy consultant: the requested quantity must be clamped "
                  f"to the authorized ceiling {detail['safe_cap']}.")
    else:
        detail = env.inspect_data()
        report = ("Controlled state probe: use the current revision "
                  f"{detail['latest_revision']} instead of the submitted revision.")
    return {"source": source, "provenance": "deterministic_fixture_not_external_api",
            "applicable": correct, "content": report,
            "fixture_query_count": 0 if source == "blind" else 1}


def as_json(action: dict) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def training_examples(case: Case, source: str, report_override: dict | None = None) -> list[dict]:
    """Equal action labels in all arms isolate evidence in training context.

    Teacher labels come from the *benchmark oracle* in every arm; source
    reports are extra text, not the unique origin of ground-truth labels.
    This is therefore a diagnostic-conditioning ablation, NOT a direct
    estimate of information gained by buying source-specific annotations.
    """
    env = Transaction(case)
    all_actions = env.gold_actions()
    transcript = env.initial_messages()
    examples = []
    for index, action in enumerate(all_actions):
        if index == 1 and source != "blind":
            report = report_override if report_override is not None else source_report(case, source)
            if report.get("source") != source or not isinstance(report.get("content"), str):
                raise ValueError("evidence source/content mismatch")
            transcript.append({"role": "user", "content": "Training-time diagnostic: " + report["content"]})
        examples.append({"messages": list(transcript), "target": as_json(action),
                         "split": case.split, "case_id": case.case_id,
                         "mechanism": case.mechanism, "source": source,
                         "step": index})
        response = env.step(action)
        transcript.append({"role": "assistant", "content": as_json(action)})
        transcript.append({"role": "user", "content": "Tool returned: " + json.dumps(response, sort_keys=True)})
    if not env.success:
        raise AssertionError("oracle fixture failed strict checker")
    return examples


def parse_action(text: str) -> dict:
    """Decode a generated JSON action, including fenced/extraneous text."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "tool" in value:
            return value
    return {"tool": "INVALID", "args": {}}


def rollout(case: Case, respond: Callable[[list[dict]], str], max_steps: int = 5) -> dict:
    """Evaluate a full independent trajectory, never inject source reports."""
    env = Transaction(case, max_steps=max_steps)
    messages = env.initial_messages()
    while not env.done:
        raw = respond(messages)
        action = parse_action(raw)
        observation = env.step(action)
        messages.extend([{"role": "assistant", "content": raw},
                         {"role": "user", "content": "Tool returned: " + json.dumps(observation, sort_keys=True)}])
    return {"case_id": case.case_id, "family_id": case.family_id,
            "split": case.split, "mechanism": case.mechanism,
            "strict_success": bool(env.done and env.success),
            "episode_done": env.done, "steps": env.steps,
            "side_effects": env.side_effects, "trace": env.history,
            "heldout_no_teacher_or_docs": True}


def same_input_source_rule(inspect_observation: dict) -> str:
    """Select a diagnosis source using ONLY a permissible local tool reply.

    No experiment-only mechanism ID, hidden case attributes, or test labels
    are supplied to this rule. Charge the local inspection cost separately.
    """
    fields = inspect_observation.get("detail", {})
    if {"observed_field", "required_field"} <= set(fields):
        return "docs"
    if {"requested_units", "safe_cap"} <= set(fields):
        return "teacher"
    if {"submitted_revision", "latest_revision"} <= set(fields):
        return "probe"
    raise ValueError("local inspection does not identify a supported symptom")
