"""Create an immutable deterministic private-case manifest; no model required.

python -m experiments.real_lora.prepare --out runs/real_lora/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .benchmark import MECHANISMS, Case, Transaction, make_cases, validate_cases


def canonical_hash(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def prepare(seed: int = 42, train_groups: int = 48, validation_groups: int = 12,
            test_groups: int = 24) -> dict:
    splits = {"train": train_groups, "validation": validation_groups, "test": test_groups}
    cases = make_cases(seed, splits)
    initial_prompts = {c.mechanism: Transaction(c).initial_messages()
                       for c in cases if c.family_id == cases[0].family_id}
    if len({json.dumps(x, sort_keys=True) for x in initial_prompts.values()}) != 1:
        raise AssertionError("initial visible failure varies across latent causes")
    payload = {"protocol": "SPEAR real-model tool-repair counterfactual v1",
               "environment": "synthetic_resettable_transaction_tools_not_androidworld",
               "evidence": "controlled_fixture_not_live_teacher_or_web",
               "seed": seed, "groups": splits,
               "cases": [c.as_dict() for c in cases],
               "rules": {"max_agent_steps": 5,
                         "candidate_checkpoint": "same operator-pinned base per arm",
                         "splits": "disjoint family IDs; no test results in candidate acceptance",
                         "training_labels": "oracle checker trajectories are equal in all evidence arms",
                         "test_diagnosis": "no teacher/docs/probe report injected; local inspect_failure remains allowed"}}
    payload["case_sha256"] = canonical_hash(payload["cases"])
    return payload


def load_manifest(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [Case.load(c) for c in payload["cases"]]
    validate_cases(cases)
    expected = canonical_hash(payload["cases"])
    if expected != payload.get("case_sha256"):
        raise ValueError("manifest integrity hash mismatch")
    for split, expected_groups in payload["groups"].items():
        actual = len({c.family_id for c in cases if c.split == split})
        if actual != expected_groups:
            raise ValueError("group count differs from manifest declaration")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-groups", type=int, default=48)
    parser.add_argument("--validation-groups", type=int, default=12)
    parser.add_argument("--test-groups", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("runs/real_lora/manifest.json"))
    args = parser.parse_args()
    manifest = prepare(args.seed, args.train_groups, args.validation_groups, args.test_groups)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        if json.loads(args.out.read_text()) != manifest:
            raise FileExistsError("existing manifest differs; use a new run directory")
    else:
        args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(args.out), "cases": len(manifest["cases"]),
                      "groups": manifest["groups"], "sha": manifest["case_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
