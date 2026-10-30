import unittest

from experiments.parameter_voi.core import Environment, cases, SOURCES
from experiments.parameter_voi.learning import (fit_pipeline, split_pairs,
                                                 StaticSemanticDirection,
                                                 DirectionModel)
from experiments.parameter_voi.run import (run_experiment, summary, apply_policy,
                                            evaluate, paired_bootstrap)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        worlds = cases("train-a", 96, 4)
        cls.generator, cls.value, cls.static_value, cls.audit = fit_pipeline(
            *split_pairs(worlds), seed=4)
        cls.test = cases("test-a", 40, 8)

    def test_generator_is_trained_from_outcomes(self):
        self.assertEqual(self.audit["counterfactual_candidate_evaluations"],
                         2 * self.audit["training_evidence_examples"])
        self.assertGreater(self.generator.examples_seen, 0)
        self.assertTrue(any(abs(w) > 0 for w in self.generator.coefficients))

    def test_query_budget_and_no_hidden_evidence_on_final_rollout(self):
        for method in ("no_update", "exhaustive_no_query", "fixed_docs_learned",
                       "pi_voi_learned", "ready_rule_learned"):
            sample = apply_policy(self.test[0], method, self.generator,
                                  self.value, self.static_value, seed=9)
            self.assertTrue(sample.heldout_evidence_disabled)
            self.assertIn(sample.source_calls, (0, 1))
            self.assertEqual(sample.validation_calls,
                             2 if method == "exhaustive_no_query" else
                             0 if sample.proposed_direction == 0 else 1)
            if sample.accepted:
                self.assertNotEqual(sample.persisted_new_weight, 0)
            else:
                self.assertEqual(sample.persisted_new_weight, 0)

    def test_all_baselines_have_metrics(self):
        data = evaluate(self.test, self.generator, self.value,
                        self.static_value, seed=1)
        summaries = summary(data)
        self.assertEqual(len(summaries), 13)
        self.assertEqual(summaries["no_update"]["strict_success"], .5)
        self.assertEqual(summaries["exhaustive_no_query"]["strict_success"], 1.0)
        bootstrap = paired_bootstrap(data, "pi_voi_learned", "ready_rule_learned", 1)
        self.assertEqual(bootstrap["n_pairs"], 20)
        self.assertEqual(len(bootstrap["ci_95"]), 2)

    def test_full_run_determinism(self):
        a = run_experiment(seed=2, train=48, test=24, meta_train=48,
                           meta_test=24)
        b = run_experiment(seed=2, train=48, test=24, meta_train=48,
                           meta_test=24)
        self.assertEqual(a, b)
        self.assertEqual(a["split_sizes"]["final_phase0_test"], 24)
        self.assertIn("adaptive_meta", a["recursive_stress_test"]["method_summary"])
        self.assertIn("frozen_meta", a["recursive_stress_test"]["method_summary"])
        self.assertEqual(a["recursive_stress_test"]["adaptive_archive_cases"],
                         a["recursive_stress_test"]["frozen_archive_cases"])

    def test_split_parameter_validation(self):
        with self.assertRaises(ValueError):
            run_experiment(train=10, test=10, meta_train=12, meta_test=10)


if __name__ == "__main__":
    unittest.main()
