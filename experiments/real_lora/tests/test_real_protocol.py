"""CPU-only invariants for real-LoRA protocol; does not download model weights."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from experiments.real_lora.benchmark import (
    MECHANISMS, SOURCES, Transaction, as_json, make_cases, parse_action,
    rollout, source_report, training_examples)
from experiments.real_lora.prepare import load_manifest, prepare
from experiments.real_lora.run import (
    assemble_report, build_training_rows, check_metadata, decision,
    load_external_reports, metadata_for_training, paired_cluster_bootstrap)


class RealLoRAProtocolTests(unittest.TestCase):
    def setUp(self):
        self.m = prepare(seed=42, train_groups=3, validation_groups=2, test_groups=2)
        self.cases = make_cases(42, {"train":3,"validation":2,"test":2})

    def test_identical_initial_failure(self):
        triple = [c for c in self.cases if c.family_id==self.cases[0].family_id]
        prompts=[Transaction(c).initial_messages() for c in triple]
        self.assertEqual(len(triple),3)
        self.assertEqual(prompts[0],prompts[1])
        self.assertEqual(prompts[0],prompts[2])

    def test_local_inspection_identifies_different_repairs(self):
        triple = self.cases[:3]
        inspections=[]
        tools=[]
        for c in triple:
            env=Transaction(c)
            inspections.append(env.step({"tool":"inspect_failure","args":{}}))
            tools.append(env.gold_actions()[1]["tool"])
        self.assertEqual(len({json.dumps(o,sort_keys=True) for o in inspections}),3)
        self.assertEqual(set(tools),{"rename_field","cap_units","refresh_revision"})

    def test_gold_trajectories_all_strict(self):
        for c in self.cases:
            env=Transaction(c)
            for a in env.gold_actions(): env.step(a)
            self.assertTrue(env.success)
            self.assertEqual(env.steps,3)
            self.assertEqual(env.side_effects,0)

    def test_wrong_repair_fails(self):
        for c in self.cases:
            env=Transaction(c)
            env.step({"tool":"inspect_failure","args":{}})
            env.step({"tool":"cap_units","args":{"units":-7}})
            env.step({"tool":"submit","args":{}})
            self.assertFalse(env.success)
            self.assertGreater(env.side_effects,0)

    def test_full_rollout_no_external_text(self):
        c=self.cases[0]
        actions=iter(Transaction(c).gold_actions())
        r=rollout(c,lambda _:as_json(next(actions)))
        self.assertTrue(r["strict_success"])
        self.assertTrue(r["heldout_no_teacher_or_docs"])
        self.assertEqual(len(r["trace"]),3)

    def test_invalid_model_json_fails(self):
        r=rollout(self.cases[0],lambda _: "not a JSON tool call",max_steps=4)
        self.assertFalse(r["strict_success"])

    def test_decode_json_inside_other_text(self):
        self.assertEqual(parse_action('leading {"tool":"submit","args":{}} trailing')["tool"],"submit")

    def test_sources_are_explicit_fixtures(self):
        for c in self.cases:
            for source in SOURCES:
                self.assertIn("fixture",source_report(c,source)["provenance"])

    def test_equal_supervision_labels(self):
        for c in self.cases[:3]:
            groups=[[x["target"] for x in training_examples(c,s)] for s in SOURCES]
            self.assertTrue(all(x==groups[0] for x in groups))
            self.assertEqual(len(groups[0]),3)

    def test_deterministic_manifest(self):
        self.assertEqual(self.m,prepare(42,3,2,2))
        self.assertNotEqual(self.m["case_sha256"],prepare(43,3,2,2)["case_sha256"])

    def test_manifest_tampering_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"manifest.json"
            p.write_text(json.dumps(self.m))
            self.assertEqual(load_manifest(p)["case_sha256"],self.m["case_sha256"])
            bad=json.loads(p.read_text())
            bad["cases"][0]["split"]="test"
            p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):load_manifest(p)

    def test_train_only_target_and_train_split(self):
        rows,logs=build_training_rows(self.m,"schema","docs")
        self.assertEqual({x["split"] for x in rows},{"train"})
        self.assertEqual({x["mechanism"] for x in rows},{"schema"})
        self.assertEqual(len(rows),9)
        self.assertEqual(len(logs),3)

    def test_all_arms_have_same_action_targets(self):
        labels={}
        for s in SOURCES:
            rows,_=build_training_rows(self.m,"policy",s)
            labels[s]=[(x["case_id"],x["step"],x["target"]) for x in rows]
        self.assertTrue(all(v==labels["blind"] for v in labels.values()))

    def test_external_source_requires_complete_training_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"evidence.jsonl"
            p.write_text(json.dumps({"case_id":self.m["cases"][0]["case_id"],
                "source":"docs","content":"text","provenance":"external"})+"\n")
            with self.assertRaises(ValueError):
                load_external_reports(p,self.m,"schema","docs")

    def test_external_source_rejects_test_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"evidence.jsonl"
            c=next(x for x in self.m["cases"] if x["split"]=="test")
            p.write_text(json.dumps({"case_id":c["case_id"],
                "source":"docs","content":"leak","provenance":"external"})+"\n")
            with self.assertRaises(ValueError):
                load_external_reports(p,self.m,"schema","docs")

    def test_validation_gate_and_retention(self):
        rows=[{"mechanism":k,"baseline":{"strict_success":k!="schema"},
               "candidate":{"strict_success":True}} for k in MECHANISMS]
        self.assertTrue(decision(rows,"schema",0,0)["accepted"])
        rows[1]["candidate"]["strict_success"]=False
        self.assertFalse(decision(rows,"schema",0,0)["accepted"])

    def test_checkpoint_mismatch_fails(self):
        meta=metadata_for_training(manifest=self.m,target="schema",source="docs",
            model="local",revision="rev",base_id="base-a",seed=42,
            quantization="4bit",steps=3,evidence_origin="fixture",cases_count=3,query_count=3)
        args=argparse.Namespace(target="schema",model="local",revision="rev",
                                base_id="base-b",quantization="4bit")
        with self.assertRaises(ValueError):check_metadata(meta,self.m,args)

    def test_paired_bootstrap_clusters_by_family(self):
        a=[{"case_id":"c1","family_id":"f1","split":"test","mechanism":"schema",
            "candidate":{"strict_success":True}},
           {"case_id":"c2","family_id":"f2","split":"test","mechanism":"schema",
            "candidate":{"strict_success":True}}]
        b=[{"case_id":"c1","family_id":"f1","split":"test","mechanism":"schema",
            "candidate":{"strict_success":False}},
           {"case_id":"c2","family_id":"f2","split":"test","mechanism":"schema",
            "candidate":{"strict_success":True}}]
        r=paired_cluster_bootstrap(a,b,mechanism="schema",repeats=100)
        self.assertEqual(r["n_family_clusters"],2)
        self.assertEqual(r["success_rate_difference"],.5)
        with self.assertRaises(ValueError):paired_cluster_bootstrap(a,b[:1],mechanism="schema")

    def test_report_disallows_mixed_base_checkpoints(self):
        def row(s,base):
            return {"target_mechanism":"schema","source":s,"model":"Qwen",
                "revision":"r","base_id":base,"manifest_case_sha256":"digest",
                "selection":{"accepted":False,"new_skill_gain":0,"old_skill_drop":0},
                "target_baseline_success":0,"target_candidate_success":0,
                "target_deployed_success":0,"retention_candidate_success":0,
                "test_target_n":2,"test_retention_n":4,
                "resource_ledger":{"gpu_train_wall_seconds":1,"gpu_eval_wall_seconds":1},
                "source_query_count_fixture":1}
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/"a.json";b=Path(tmp)/"b.json"
            a.write_text(json.dumps(row("docs","base-a")))
            b.write_text(json.dumps(row("teacher","base-b")))
            with self.assertRaises(ValueError):assemble_report([a,b])

    def test_validation_source_selection_not_test_leakage(self):
        def row(s,vg,ts):
            return {"target_mechanism":"schema","source":s,"model":"Qwen",
                "revision":"r","base_id":"base","manifest_case_sha256":"digest",
                "selection":{"accepted":True,"new_skill_gain":vg,"old_skill_drop":0},
                "target_baseline_success":0,"target_candidate_success":ts,
                "target_deployed_success":ts,"retention_candidate_success":0,
                "test_target_n":2,"test_retention_n":4,
                "resource_ledger":{"gpu_train_wall_seconds":1,"gpu_eval_wall_seconds":1},
                "source_query_count_fixture":1}
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/"a.json";b=Path(tmp)/"b.json"
            a.write_text(json.dumps(row("docs",.1,1)))
            b.write_text(json.dumps(row("teacher",.2,0)))
            self.assertEqual(assemble_report([a,b])["retrospective_validation_selection"][0]["source"],"teacher")


if __name__ == "__main__":
    unittest.main()
