"""Real-model counterfactual LoRA protocol (no model simulated in train/evaluate).

CLI:
  python -m experiments.real_lora.prepare --out runs/spear/manifest.json
  python -m experiments.real_lora.run plan --manifest runs/spear/manifest.json
  python -m experiments.real_lora.run train --help
  python -m experiments.real_lora.run evaluate --help
  python -m experiments.real_lora.run report --help
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from .benchmark import (Case, MECHANISMS, SOURCES, SOURCE_FOR, Transaction,
                        same_input_source_rule, rollout, source_report, training_examples)
from .prepare import canonical_hash, load_manifest


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def all_cases(manifest: dict, *, split: str, mechanism: str | None = None) -> list[Case]:
    return [Case.load(c) for c in manifest["cases"]
            if c["split"] == split and (mechanism is None or c["mechanism"] == mechanism)]


def build_training_rows(manifest: dict, target: str, source: str,
                        source_overrides: dict[str, dict] | None = None) -> tuple[list[dict], list[dict]]:
    if target not in MECHANISMS or source not in SOURCES:
        raise ValueError("unsupported target mechanism or evidence source")
    rows: list[dict] = []
    provenance: list[dict] = []
    for case in all_cases(manifest, split="train", mechanism=target):
        report = (source_overrides or {}).get(case.case_id)
        if report is None:
            report = source_report(case, source)
        if source == "blind" and report.get("source") != "blind":
            raise ValueError("blind arm cannot carry external evidence")
        if report.get("source") != source or not isinstance(report.get("content"), str):
            raise ValueError("invalid source report")
        if report.get("provenance") != "deterministic_fixture_not_external_api" and not report.get("provenance"):
            raise ValueError("real evidence requires provenance")
        rows.extend(training_examples(case, source, report_override=report))
        provenance.append({"case_id": case.case_id, "source": source,
                           "provenance": report["provenance"],
                           "report_sha256": hashlib.sha256(report["content"].encode()).hexdigest(),
                           "fixture_query_count": report.get("fixture_query_count", 0)})
    return rows, provenance


def load_external_reports(path: Path, manifest: dict, target: str, source: str) -> dict[str, dict]:
    """Strict text-only external evidence format; no unsupported API calls."""
    if source == "blind":
        raise ValueError("blind arm may not use external reports")
    eligible = {c.case_id for c in all_cases(manifest, split="train", mechanism=target)}
    reports: dict[str, dict] = {}
    for line_n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        report = json.loads(line)
        case_id = report.get("case_id")
        if case_id not in eligible or case_id in reports:
            raise ValueError(f"line {line_n}: wrong/duplicate case_id or contaminated split")
        if report.get("source") != source:
            raise ValueError(f"line {line_n}: unexpected source")
        if not isinstance(report.get("content"), str) or not report["content"].strip():
            raise ValueError(f"line {line_n}: missing evidence content")
        if not isinstance(report.get("provenance"), str) or not report["provenance"].strip():
            raise ValueError(f"line {line_n}: evidence provenance is required")
        if "fixture_query_count" in report and report["fixture_query_count"] != 0:
            raise ValueError("external source reports cannot claim fixture query counts")
        reports[case_id] = {"source": source, "content": report["content"],
                            "provenance": report["provenance"], "fixture_query_count": 0}
    if reports.keys() != eligible:
        raise ValueError(f"external reports must cover all {len(eligible)} eligible train cases")
    return reports


def metadata_for_training(*, manifest: dict, target: str, source: str,
                          model: str, revision: str, base_id: str, seed: int,
                          quantization: str, steps: int, evidence_origin: str,
                          cases_count: int, query_count: int) -> dict:
    return {"protocol": "SPEAR real-weights LoRA intervention v1",
            "manifest_case_sha256": manifest["case_sha256"],
            "target_mechanism": target, "source": source,
            "checkpoint_ref": model, "checkpoint_revision": revision,
            "operator_supplied_base_id": base_id,
            "base_id_is_independently_verified_sha": False,
            "train_seed": seed, "quantization": quantization,
            "gradient_optimizer_steps": steps,
            "train_cases": cases_count,
            "evidence_origin": evidence_origin,
            "source_query_count": query_count,
            "ground_truth_action_labels": "oracle checker demonstrations in every arm; labels controlled equal",
            "not_claimed": "No actual teacher API or AndroidWorld unless independently connected and documented."}


def train(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.grad_accum < 1 or args.max_length < 96:
        raise ValueError("invalid training hyperparameters")
    if args.base_id is None or not args.base_id.strip():
        raise ValueError("--base-id is mandatory; use one immutable base-model identifier across arms")
    manifest = load_manifest(args.manifest)
    overrides = (load_external_reports(args.external_reports, manifest, args.target, args.source)
                 if args.external_reports else None)
    rows, provenance = build_training_rows(manifest, args.target, args.source, overrides)
    if not rows:
        raise ValueError("no training examples")
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("refusing to overwrite existing adapter directory")
    from .hf_backend import train_adapter, adapter_hash
    backend = train_adapter(model_id=args.model, revision=args.revision,
                            model_class=args.model_class, quantization=args.quantization,
                            rows=rows, output=out, steps=args.steps, lr=args.lr,
                            rank=args.rank, target_modules=args.target_modules.split(","),
                            seed=args.seed, max_length=args.max_length,
                            grad_accum=args.grad_accum)
    meta = metadata_for_training(manifest=manifest, target=args.target, source=args.source,
                    model=args.model, revision=args.revision, base_id=args.base_id,
                    seed=args.seed, quantization=args.quantization, steps=args.steps,
                    evidence_origin="external_captured" if overrides else "controlled_fixture",
                    cases_count=len(provenance), query_count=0 if overrides else sum(
                        x["fixture_query_count"] for x in provenance))
    meta["adapter_sha256"] = adapter_hash(out)
    meta["backend"] = backend
    meta["supervised_dataset_sha256"] = canonical_hash(rows)
    write_json(out / "spear_training.json", meta)
    write_jsonl(out / "source_provenance.jsonl", provenance)
    print(json.dumps({"adapter": str(out), "sha256": meta["adapter_sha256"],
                      "training_seconds": backend["elapsed_seconds"],
                      "source": args.source, "target": args.target}, indent=2))


def mean_success(records: list[dict], name: str) -> float:
    if not records:
        raise ValueError("empty evaluation cohort")
    return statistics.mean(int(row[name]["strict_success"]) for row in records)


def decision(records: list[dict], target: str, min_gain: float, old_tolerance: float) -> dict:
    new = [r for r in records if r["mechanism"] == target]
    old = [r for r in records if r["mechanism"] != target]
    new_gain = mean_success(new, "candidate") - mean_success(new, "baseline")
    old_drop = max(0., mean_success(old, "baseline") - mean_success(old, "candidate"))
    accepted = new_gain > min_gain and old_drop <= old_tolerance
    return {"accepted": accepted, "new_skill_gain": round(new_gain, 6),
            "old_skill_drop": round(old_drop, 6), "min_gain": min_gain,
            "old_tolerance": old_tolerance,
            "reason": ("validation_accept" if accepted else
                       "validation_gain_insufficient" if new_gain <= min_gain
                       else "validation_retention_regression"),
            "acceptance_split": "validation_only_no_test_peeking",
            "statistical_significance_not_established": True}


def check_metadata(meta: dict, manifest: dict, args: argparse.Namespace) -> None:
    values = (("manifest_case_sha256", manifest["case_sha256"]),
              ("target_mechanism", args.target), ("checkpoint_ref", args.model),
              ("checkpoint_revision", args.revision),
              ("operator_supplied_base_id", args.base_id),
              ("quantization", args.quantization))
    for field, expected in values:
        if meta.get(field) != expected:
            raise ValueError(f"adapter metadata mismatch for {field}")


def evaluate(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 16 or not 0 <= args.old_tolerance <= 1:
        raise ValueError("invalid evaluation controls")
    manifest = load_manifest(args.manifest)
    meta = json.loads((args.adapter / "spear_training.json").read_text(encoding="utf-8"))
    check_metadata(meta, manifest, args)
    from .hf_backend import load_pair, make_responder, adapter_hash
    if adapter_hash(args.adapter) != meta["adapter_sha256"]:
        raise ValueError("LoRA adapter weight digest changed after training")
    model, tok, model_config_sha = load_pair(args.model, args.revision,
                                      args.model_class, args.quantization, args.adapter)
    if model_config_sha != meta["backend"]["model_config_sha256"]:
        raise ValueError("incompatible base model config at evaluation")
    responder = make_responder(model, tok, max_new_tokens=args.max_new_tokens)
    results: list[dict] = []
    started = time.monotonic()
    for split in ("validation", "test"):
        for case in all_cases(manifest, split=split):
            with model.disable_adapter():
                baseline = rollout(case, responder, max_steps=args.max_agent_steps)
            candidate = rollout(case, responder, max_steps=args.max_agent_steps)
            results.append({"case_id": case.case_id, "family_id": case.family_id,
                "mechanism": case.mechanism, "split": split,
                "baseline": baseline, "candidate": candidate,
                "paired_environment_case": True,
                "test_teacher_docs_probe_text_injected": False})
    validation = [row for row in results if row["split"] == "validation"]
    chosen = decision(validation, args.target, args.min_gain, args.old_tolerance)
    test = [row for row in results if row["split"] == "test"]
    target_test = [row for row in test if row["mechanism"] == args.target]
    # The strong rule sees only an ordinary local inspection, never a hidden
    # mechanism label. It must be computed before final test selection.
    public_choices = {same_input_source_rule(Transaction(case).step(
                         {"tool": "inspect_failure", "args": {}}))
                      for case in all_cases(manifest, split="validation", mechanism=args.target)}
    if len(public_choices) != 1:
        raise ValueError("validation symptoms do not admit a single public rule choice")
    public_rule_source = next(iter(public_choices))
    by_mechanism = {name: {"n": len([r for r in test if r["mechanism"] == name]),
                           "baseline": round(mean_success([r for r in test if r["mechanism"] == name], "baseline"), 6),
                           "candidate": round(mean_success([r for r in test if r["mechanism"] == name], "candidate"), 6)}
                    for name in MECHANISMS}
    other_test = [row for row in test if row["mechanism"] != args.target]
    summary = {"protocol": "SPEAR real-model paired tool trajectories v1",
               "manifest_case_sha256": manifest["case_sha256"],
               "model": args.model, "revision": args.revision, "base_id": args.base_id,
               "source": meta["source"], "evidence_origin": meta["evidence_origin"],
               "target_mechanism": args.target,
               "same_input_rule_source_from_local_inspection": public_rule_source,
               "same_input_rule_local_inspection_count": len(all_cases(manifest, split="validation", mechanism=args.target)),
               "adapter_sha256": meta["adapter_sha256"],
               "train": meta["backend"],
               "selection": chosen,
               "test_mechanism_matrix": by_mechanism,
               "test_target_n": len(target_test),
               "test_retention_n": len(other_test),
               "target_baseline_success": round(mean_success(target_test, "baseline"), 6),
               "target_candidate_success": round(mean_success(target_test, "candidate"), 6),
               "target_deployed_success": round(mean_success(target_test, "candidate" if chosen["accepted"] else "baseline"), 6),
               "retention_baseline_success": round(mean_success(other_test, "baseline"), 6),
               "retention_candidate_success": round(mean_success(other_test, "candidate"), 6),
               "test_used_for_selection": False,
               "independent_family_holdout": True,
               "test_auxiliary_evidence_disabled": True,
               "source_query_count_fixture": meta["source_query_count"],
               "resource_ledger": {"gpu_eval_wall_seconds": round(time.monotonic()-started, 3),
                                   "gpu_train_wall_seconds": meta["backend"]["elapsed_seconds"],
                                   "training_reported_peak_cuda_memory_bytes": meta["backend"].get("peak_cuda_memory_bytes"),
                                   "actual_api_cost_usd": None,
                                   "fixture_queries_are_not_billable_api_calls": True},
               "claim_limits": ["controlled tool environment, not AndroidWorld",
                                "validation threshold is a pilot gate, not significance",
                                "all arms use oracle-correct SFT labels",
                                "no learned PI-VOI source-selection policy is trained here"]}
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("refusing to overwrite evaluation results")
    write_json(out / "summary.json", summary)
    write_jsonl(out / "paired_trajectories.jsonl", results)
    print(json.dumps({"result": str(out), "target": args.target, "source": meta["source"],
                      "validation_accepted": chosen["accepted"],
                      "heldout_target_candidate_success": summary["target_candidate_success"]}, indent=2))


def paired_cluster_bootstrap(rows_a: list[dict], rows_b: list[dict],
                             *, mechanism: str, seed: int = 42, repeats: int = 600) -> dict:
    """Cluster by resettable family; all compared model runs see same cases."""
    a = {r["case_id"]: r for r in rows_a
         if r["split"] == "test" and r["mechanism"] == mechanism}
    b = {r["case_id"]: r for r in rows_b
         if r["split"] == "test" and r["mechanism"] == mechanism}
    if set(a) != set(b) or not a:
        raise ValueError("source arms need identical heldout case IDs")
    delta_by_family: dict[str, list[int]] = defaultdict(list)
    for case_id in a:
        if a[case_id]["family_id"] != b[case_id]["family_id"]:
            raise ValueError("mismatched family or reset")
        delta_by_family[a[case_id]["family_id"]].append(
            int(a[case_id]["candidate"]["strict_success"])
            - int(b[case_id]["candidate"]["strict_success"]))
    diffs = [statistics.mean(v) for _, v in sorted(delta_by_family.items())]
    rng = random.Random(seed)
    boot = sorted(statistics.mean(diffs[rng.randrange(len(diffs))]
                                   for _ in diffs) for _ in range(repeats))
    return {"n_family_clusters":len(diffs),
            "success_rate_difference":round(statistics.mean(diffs),6),
            "bootstrap_ci_95":[round(boot[int(.025*(repeats-1))],6),
                               round(boot[int(.975*(repeats-1))],6)],
            "unit":"heldout strict success; paired family-cluster bootstrap",
            "not_multiple_comparison_corrected":True}


def assemble_report(paths: list[Path]) -> dict:
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if not summaries:
        raise ValueError("no real-model arm results found")
    keys = [(x["target_mechanism"], x["source"]) for x in summaries]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate target/source arm")
    for field in ("manifest_case_sha256", "model", "revision", "base_id"):
        if len({x[field] for x in summaries}) != 1:
            raise ValueError(f"arms are not checkpoint/split matched: {field}")
    table = []
    for x in sorted(summaries, key=lambda r:(r["target_mechanism"],r["source"])):
        table.append({"mechanism":x["target_mechanism"],"source":x["source"],
                      "accepted_on_validation":x["selection"]["accepted"],
                      "validation_gain":x["selection"]["new_skill_gain"],
                      "validation_old_drop":x["selection"]["old_skill_drop"],
                      "test_n":x["test_target_n"],
                      "test_baseline_success":x["target_baseline_success"],
                      "test_candidate_success":x["target_candidate_success"],
                      "test_deployed_success":x["target_deployed_success"],
                      "retention_n":x["test_retention_n"],
                      "retention_candidate_success":x["retention_candidate_success"],
                      "training_seconds":x["resource_ledger"]["gpu_train_wall_seconds"],
                      "evaluation_seconds":x["resource_ledger"]["gpu_eval_wall_seconds"],
                      "source_fixture_queries":x["source_query_count_fixture"]})
    by_target = defaultdict(dict)
    for entry in table:
        by_target[entry["mechanism"]][entry["source"]] = entry
    comparisons_vs_blind = []
    trajectories: dict[tuple[str, str], list[dict]] = {}
    for path, summary in zip(paths, summaries):
        raw = path.with_name("paired_trajectories.jsonl")
        if raw.exists():
            trajectories[(summary["target_mechanism"], summary["source"])] = [
                json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    for (target, source), candidate_rows in trajectories.items():
        if source == "blind" or (target, "blind") not in trajectories:
            continue
        comparison = paired_cluster_bootstrap(candidate_rows,
                      trajectories[(target, "blind")], mechanism=target,
                      seed=42)
        comparisons_vs_blind.append({"target":target, "source":source,
                                     "baseline":"blind", **comparison})
    heuristic = []
    val_select = []
    for target, arms in by_target.items():
        # Use only the public tool-response-derived value recorded at validation.
        choices = {x.get("same_input_rule_source_from_local_inspection")
                   for x in summaries if x["target_mechanism"] == target}
        if len(choices) != 1:
            raise ValueError("inconsistent public rule source across adapters")
        public_choice = next(iter(choices))
        if public_choice in arms:
            heuristic.append({"target":target,"source":public_choice,
                              "test_deployed_success":arms[public_choice]["test_deployed_success"],
                              "requires_prequery_local_inspection":True,
                              "mechanism_id_was_not_used_for_rule":True})
        # Selection based on the *validation split only*, with all arm training
        # and validation costs charged to the protocol, never retrospectively
        # claiming online acquisition efficiency.
        eligible = sorted(arms.values(), key=lambda x:(x["validation_gain"] - x["validation_old_drop"],
                                                       -x["training_seconds"],x["source"]))
        best = eligible[-1]
        val_select.append({"target":target,"source":best["source"],
                           "test_deployed_success":best["test_deployed_success"],
                           "arms_trained_and_checked":len(arms),
                           "total_train_eval_seconds_for_selection":round(sum(
                                a["training_seconds"]+a["evaluation_seconds"] for a in arms.values()),3)})
    return {"status":"real LoRA counterfactual feasibility; NOT a validated PI-VOI policy",
            "base_id":summaries[0]["base_id"],
            "manifest_case_sha256":summaries[0]["manifest_case_sha256"],
            "n_arms":len(table), "arms":table,
            "heldout_pairwise_comparisons_vs_blind":comparisons_vs_blind,
            "same_input_rule_selection":heuristic,
            "retrospective_validation_selection":val_select,
            "interpretation_limits": ["Held-out test was never used to select an adapter",
                "Source selection by best validation arm is expensive offline search, not a cost-effective online PI-VOI learner",
                "Same-input heuristic uses the ordinary inspect_failure response, not the hidden mechanism field",
                "Comparative causal effect requires repeat seeds, balanced source assignment and uncertainty intervals",
                "A genuine PI-VOI learned evidence-acquisition controller is not yet implemented for this real model"]}


def print_plan(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    run_dir = args.out
    print(f"3 mechanisms x {len(SOURCES)} diagnosis arms = {len(MECHANISMS)*len(SOURCES)} LoRA candidates.")
    print(f"Immutable manifest: {manifest['case_sha256']}; train groups {manifest['groups']['train']}")
    print("Each arm must use identical --base-id / --model / --revision and an independent process.")
    print("Each evaluation runs baseline + adapter on matched validation/test episodes and records full tool actions.")
    for target in MECHANISMS:
        for source in SOURCES:
            print(f"  {target:7s} {source:7s} train -> {run_dir}/adapters/{target}/{source}; "
                  f"eval -> {run_dir}/eval/{target}/{source}")
    print("GPU execution is NOT started by plan. See README for command templates.")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="command", required=True)
    plan = cmds.add_parser("plan", help="CPU-only reproducible experiment manifest plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--out", type=Path, default=Path("runs/real_lora"))

    train = cmds.add_parser("train", help="Train one REAL independent LoRA candidate")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--target", choices=MECHANISMS, required=True)
    train.add_argument("--source", choices=SOURCES, required=True)
    train.add_argument("--model", required=True, help="local checkpoint or pinned HF model identifier")
    train.add_argument("--revision", default="main")
    train.add_argument("--base-id", required=True, help="operator-pinned immutable base checkpoint ID")
    train.add_argument("--model-class", choices=("causal","image-text"), default="causal")
    train.add_argument("--quantization", choices=("4bit","none"), default="4bit")
    train.add_argument("--external-reports", type=Path)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--steps", type=int, default=80)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--rank", type=int, default=8)
    train.add_argument("--target-modules", default="q_proj,v_proj")
    train.add_argument("--max-length", type=int, default=1024)
    train.add_argument("--grad-accum", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-4)

    evalp = cmds.add_parser("evaluate", help="Baseline vs candidate paired full trajectories")
    evalp.add_argument("--manifest", type=Path, required=True)
    evalp.add_argument("--adapter", type=Path, required=True)
    evalp.add_argument("--target", choices=MECHANISMS, required=True)
    evalp.add_argument("--model", required=True)
    evalp.add_argument("--revision", default="main")
    evalp.add_argument("--base-id", required=True)
    evalp.add_argument("--model-class", choices=("causal","image-text"), default="causal")
    evalp.add_argument("--quantization", choices=("4bit","none"), default="4bit")
    evalp.add_argument("--out", type=Path, required=True)
    evalp.add_argument("--max-new-tokens", type=int, default=160)
    evalp.add_argument("--max-agent-steps", type=int, default=5)
    evalp.add_argument("--min-gain", type=float, default=0)
    evalp.add_argument("--old-tolerance", type=float, default=0.025)

    report = cmds.add_parser("report", help="Aggregate real-model arm summaries; no model needed")
    report.add_argument("--root", type=Path, required=True,
                        help="directory containing eval/<target>/<source>/summary.json")
    report.add_argument("--out", type=Path, required=True)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "plan":
        print_plan(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "report":
        paths = sorted((args.root / "eval").glob("*/*/summary.json"))
        result = assemble_report(paths)
        write_json(args.out, result)
        print(json.dumps({"saved":str(args.out),"n_arms":result["n_arms"]},indent=2))


if __name__ == "__main__":
    main()
