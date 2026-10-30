import copy
import unittest

from experiments.parameter_voi.external_audit import audit_rows, AuditError


def example(policy, group, *, split="test", status="complete", pre=False,
            post=True, accepted=True):
    return {
        "run_id": f"run-{policy}-{group}", "group_id": group,
        "split": split, "policy": policy, "source": "probe",
        "source_propensity": 0.5, "visible_failure_id": "err-1",
        "pre_checkpoint_sha": "base-sha",
        "post_checkpoint_sha": "adapted-sha" if accepted else "base-sha",
        "candidate_sha": "candidate-sha", "environment_snapshot": f"snapshot-{group}",
        "pre_seed": 17, "post_seed": 17,
        "baseline_success": pre if status == "complete" else None,
        "candidate_success": post if status == "complete" else None,
        "baseline_old_success": True if status == "complete" else None,
        "candidate_old_success": True if status == "complete" else None,
        "source_cost": .02, "validation_cost": .15,
        "accepted": accepted if status == "complete" else False,
        "heldout_no_evidence": True,
        "episode_done_pre": pre, "episode_done_post": post,
        "status": status,
    }


class ExternalAuditTests(unittest.TestCase):
    def test_valid_paired_and_error_not_failed(self):
        rows = [example("voi", "g1"),
                example("rule", "g1", pre=False, post=False),
                example("voi", "g2", status="technical_error")]
        rows[1]["accepted"] = False
        rows[1]["post_checkpoint_sha"] = "base-sha"
        audit = audit_rows(rows)
        self.assertEqual(audit["complete_records"], 2)
        self.assertEqual(audit["unknown_technical_error_records"], 1)
        self.assertEqual(audit["policies"]["voi"]["paired_strict_success_gain"], 1)
        self.assertEqual(audit["matched_policy_comparisons"][0]["matched_reset_seed_pairs"], 1)

    def test_reject_invalid_incomplete_success(self):
        r = example("voi", "g1")
        r["episode_done_post"] = False
        with self.assertRaises(AuditError):
            audit_rows([r])

    def test_reject_unpaired_seed_and_split_leak(self):
        r = example("voi", "g1")
        r["post_seed"] = 18
        with self.assertRaises(AuditError):
            audit_rows([r])
        a, b = example("voi", "g1"), example("rule", "g1", split="train")
        with self.assertRaises(AuditError):
            audit_rows([a,b])

    def test_reject_nonzero_errors_and_test_evidence_injection(self):
        r = example("voi", "g1", status="technical_error")
        r["candidate_success"] = False
        with self.assertRaises(AuditError):
            audit_rows([r])
        r = example("voi", "g2")
        r["heldout_no_evidence"] = False
        with self.assertRaises(AuditError):
            audit_rows([r])

    def test_failures_after_acceptance_are_reported_not_hidden(self):
        r = example("voi", "g1", post=False, accepted=True)
        r["candidate_old_success"] = False
        report = audit_rows([r])
        self.assertEqual(report["policies"]["voi"]["old_skill_regression_rate"], 1)
        self.assertEqual(report["policies"]["voi"]["paired_strict_success_gain"], 0)

    def test_rejected_changes_must_rollback(self):
        r = example("voi", "g1", post=False, accepted=False)
        r["post_checkpoint_sha"] = "no-rollback"
        with self.assertRaises(AuditError):
            audit_rows([r])


if __name__ == "__main__":
    unittest.main()
