"""Validate and summarize paired post-update outcomes from REAL agent runs.

Input is an externally collected JSONL file. This script does not launch a
model, infer a missing strict checker, or fabricate held-out outcomes.

python -m experiments.parameter_voi.external_audit real_trials.jsonl --output report.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any

VALID_SPLITS = ("train", "validation", "test", "retention")
VALID_STATUSES = ("complete", "technical_error")
REQUIRED = (
    "run_id", "group_id", "split", "policy", "source", "source_propensity",
    "visible_failure_id", "pre_checkpoint_sha", "post_checkpoint_sha",
    "candidate_sha", "environment_snapshot", "pre_seed", "post_seed",
    "baseline_success", "candidate_success", "baseline_old_success",
    "candidate_old_success", "source_cost", "validation_cost", "accepted",
    "heldout_no_evidence", "episode_done_pre", "episode_done_post", "status",
)


class AuditError(ValueError):
    pass


def _one_row(row: dict, line: int) -> None:
    missing = set(REQUIRED) - row.keys()
    if missing:
        raise AuditError(f"line {line}: required fields missing: {sorted(missing)}")
    if row["split"] not in VALID_SPLITS or row["status"] not in VALID_STATUSES:
        raise AuditError(f"line {line}: unsupported split/status")
    for key in ("run_id", "group_id", "policy", "pre_checkpoint_sha",
                "post_checkpoint_sha", "candidate_sha", "environment_snapshot",
                "visible_failure_id"):
        if not isinstance(row[key], str) or not row[key]:
            raise AuditError(f"line {line}: {key} must be a nonempty string")
    if row["source"] is not None and not isinstance(row["source"], str):
        raise AuditError(f"line {line}: source must be string or null")
    for key in ("accepted", "heldout_no_evidence", "episode_done_pre",
                "episode_done_post"):
        if not isinstance(row[key], bool):
            raise AuditError(f"line {line}: {key} must be a bool")
    for key in ("source_cost", "validation_cost"):
        if (isinstance(row[key], bool) or not isinstance(row[key], (int, float))
                or not 0 <= row[key] < 1e7):
            raise AuditError(f"line {line}: invalid {key}")
    if not 0 < row["source_propensity"] <= 1:
        raise AuditError(f"line {line}: source_propensity must be in (0,1]")
    if row["pre_seed"] != row["post_seed"]:
        raise AuditError(f"line {line}: the baseline and candidate must use the same reset seed")
    for key in ("baseline_success", "candidate_success", "baseline_old_success",
                "candidate_old_success"):
        val = row[key]
        if row["status"] == "technical_error":
            if val is not None:
                raise AuditError(f"line {line}: technical error outcomes must be null (unknown)")
        else:
            if not isinstance(val, bool):
                raise AuditError(f"line {line}: complete {key} must be strict checker bool")
    if row["status"] == "technical_error":
        if row["accepted"]:
            raise AuditError(f"line {line}: technical error cannot be accepted")
        return
    if row["accepted"] and row["pre_checkpoint_sha"] == row["post_checkpoint_sha"]:
        raise AuditError(f"line {line}: an accepted parameter update must change checkpoint SHA")
    if not row["accepted"] and row["pre_checkpoint_sha"] != row["post_checkpoint_sha"]:
        raise AuditError(f"line {line}: rejected trial must roll back to original checkpoint")
    if row["split"] in ("test", "retention") and not row["heldout_no_evidence"]:
        raise AuditError(f"line {line}: evidence was injected during heldout evaluation")
    # Acceptance is made from validation evidence; a subsequently failed test
    # or old-skill regression is a legitimate negative result, not malformed.
    if row["baseline_success"] and not row["episode_done_pre"]:
        raise AuditError(f"line {line}: baseline strict success contradicts episode_done_pre")
    if row["candidate_success"] and not row["episode_done_post"]:
        raise AuditError(f"line {line}: candidate strict success contradicts episode_done_post")


def audit_rows(rows: list[dict]) -> dict:
    if not rows:
        raise AuditError("empty input")
    identities = set()
    group_splits: dict[str, str] = {}
    for n, row in enumerate(rows, 1):
        _one_row(row, n)
        id_key = (row["run_id"], row["group_id"], row["policy"], row["pre_seed"])
        if id_key in identities:
            raise AuditError(f"line {n}: duplicate run/paired-world/policy/seed")
        identities.add(id_key)
        group = row["group_id"]
        if group in group_splits and group_splits[group] != row["split"]:
            raise AuditError(f"line {n}: group_id occurs in different splits")
        group_splits[group] = row["split"]

    completed = [r for r in rows if r["status"] == "complete"]
    failed = [r for r in rows if r["status"] == "technical_error"]
    methods: dict[str, list[dict]] = defaultdict(list)
    for r in completed:
        methods[r["policy"]].append(r)
    report = {
        "protocol": "SPEAR external paired-checker audit v0.1",
        "data_origin": "external JSONL supplied by experiment operator; no model was executed by this script",
        "total_records": len(rows),
        "complete_records": len(completed),
        "unknown_technical_error_records": len(failed),
        "splits": {s: sum(r["split"] == s for r in rows) for s in VALID_SPLITS},
        "policies": {},
        "matched_policy_comparisons": [],
        "warnings": [],
    }
    for policy, group in sorted(methods.items()):
        report["policies"][policy] = {
            "complete_pairs": len(group),
            "paired_strict_success_gain": round(statistics.mean(
                int(r["candidate_success"]) - int(r["baseline_success"]) for r in group), 6),
            "old_skill_regression_rate": round(statistics.mean(
                int(r["baseline_old_success"] and not r["candidate_old_success"])
                for r in group), 6),
            "mean_reported_cost": round(statistics.mean(
                r["source_cost"] + r["validation_cost"] for r in group), 6),
            "mean_net_gain": round(statistics.mean(
                (int(r["candidate_success"]) - int(r["baseline_success"]))
                - int(r["baseline_old_success"] and not r["candidate_old_success"])
                - r["source_cost"] - r["validation_cost"] for r in group), 6),
            "accepted": sum(r["accepted"] for r in group),
        }
    # Important: matched pairs only, not unpaired comparisons across groups.
    test_pairs: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in completed:
        if row["split"] == "test":
            test_pairs[(row["group_id"], row["pre_seed"])][row["policy"]] = row
    policies = sorted(methods)
    for i, a in enumerate(policies):
        for b in policies[i+1:]:
            eligible = [(d[a], d[b]) for d in test_pairs.values() if a in d and b in d]
            if not eligible:
                continue
            delta = statistics.mean(
                (int(ra["candidate_success"]) - int(rb["candidate_success"]))
                - (ra["source_cost"] + ra["validation_cost"]
                   - rb["source_cost"] - rb["validation_cost"])
                for ra, rb in eligible)
            report["matched_policy_comparisons"].append({
                "a": a, "b": b, "matched_reset_seed_pairs": len(eligible),
                "paired_net_difference_a_minus_b": round(delta, 6),
            })
    if failed:
        report["warnings"].append("Technical-error outcomes are unknown, not classified as failures; report counts by policy/split separately.")
    if any(r["source_propensity"] == 1. for r in completed):
        report["warnings"].append("Some source assignments have propensity 1; causal comparisons across different sources may lack positivity.")
    if not report["matched_policy_comparisons"]:
        report["warnings"].append("No two policies share a paired test group and reset seed; causal policy comparison unavailable.")
    report["warnings"].append("Differences are descriptive unless source assignment, calibration, sample sizes and confidence intervals are independently verified.")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    rows = []
    for n, raw in enumerate(args.jsonl.read_text().splitlines(), 1):
        if raw.strip():
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise AuditError(f"malformed JSON line {n}: {exc}") from exc
    result = audit_rows(rows)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
