import unittest
from dataclasses import fields

from experiments.parameter_voi.core import (Environment, Student, cases,
    PREFERRED_SOURCE, SOURCE_COST, VALIDATION_COST)
from experiments.parameter_voi.run import reversal_audit, retention_audit, ensure_disjoint


class EnvironmentTests(unittest.TestCase):
    def test_world_pairs_same_visible_state_but_opposite_best_update(self):
        worlds = cases("toy", 60, 33)
        summary = reversal_audit(worlds)
        self.assertEqual(summary["valid_matched_pairs"], 30)
        self.assertEqual(summary["pairs_with_opposite_best_parameter_update"], 30)
        self.assertTrue(all(w.context.visible_failure == worlds[0].context.visible_failure
                            for w in worlds))

    def test_no_latent_cause_field_exposed_before_evidence(self):
        worlds = cases("toy", 20, 12)
        visible = worlds[0].context.as_dict()
        self.assertNotIn("latent_direction", visible)
        self.assertFalse(any(f.name == "latent_direction"
                             for f in fields(type(worlds[0].context))))

    def test_evidence_provenance_and_determinism(self):
        world = cases("toy", 8, 21)[0]
        a, b = Environment(world), Environment(world)
        self.assertEqual(a.query("docs"), b.query("docs"))
        self.assertEqual(a.query("probe"), b.query("probe"))
        self.assertEqual(a.query_log, ("docs", "probe"))
        with self.assertRaises(ValueError):
            a.query("made_up_source")

    def test_persistent_numeric_parameters_and_no_evidence_at_eval(self):
        world = next(w for w in cases("toy", 10, 5) if w.latent_direction == 1)
        env = Environment(world)
        before = Student()
        result = env.paired_trial(before, +1, protect_old=True)
        self.assertTrue(result.accepted)
        self.assertEqual(result.updated_new, 1.0)
        gain, forgetting = env.heldout_gain(before, result)
        self.assertEqual(gain, .5)
        self.assertEqual(forgetting, 0.0)
        self.assertEqual(env.query_log, ())  # checker never queries source
        committed = before.updated(+1, world.context.shared_parameters, True)
        self.assertEqual(committed.new_weight, 2.0)
        self.assertEqual(committed.old_weight, 1.5)

    def test_bad_or_zero_update_cannot_improve_heldout_metric(self):
        world = next(w for w in cases("toy", 10, 19) if w.latent_direction == +1)
        env = Environment(world)
        no = env.paired_trial(Student(), 0)
        wrong = env.paired_trial(Student(), -1)
        self.assertEqual(no.validation_calls, 0)
        self.assertFalse(wrong.accepted)
        self.assertEqual(env.heldout_gain(Student(), wrong)[0], 0.0)

    def test_retention_interference_rejected(self):
        world = next(w for w in cases("toy", 50, 8) if w.context.shared_parameters)
        env = Environment(world)
        bad = env.paired_trial(Student(), +1, protect_old=False)
        self.assertFalse(bad.accepted)
        self.assertEqual(bad.reason, "retention_regression")
        self.assertEqual(bad.updated_old, 0.0)
        safe = env.paired_trial(Student(), +1, protect_old=True)
        self.assertEqual(safe.updated_old, 1.0)
        self.assertEqual(env.heldout_gain(Student(), bad)[1], 0.0)
        audit = retention_audit(cases("toy", 50, 8))
        self.assertEqual(audit["unsafe_updates_rejected"], audit["shared_parameter_cases"])

    def test_split_overlap_protected(self):
        a = cases("train", 20, 1)
        b = cases("test", 20, 1)
        ensure_disjoint(a, b)
        with self.assertRaises(ValueError):
            ensure_disjoint(a, a)

    def test_inputs_and_count_validation(self):
        with self.assertRaises(ValueError):
            cases("fail", 3, 1)
        with self.assertRaises(ValueError):
            cases("fail", 0, 1)
        with self.assertRaises(ValueError):
            cases("fail", 4, 1, phase=2)


if __name__ == "__main__":
    unittest.main()
