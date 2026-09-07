"""Experiment iteration: redesign the test until it discriminates."""

from __future__ import annotations

import unittest

from farfield.extras.iterate import (
    DEFAULT_EXPERIMENT_ROUNDS,
    EXPERIMENT_AUTO_CAP,
    classify_bottleneck,
    decide_retry,
    pipeline_bottleneck,
    resolve_experiment_rounds,
    switch_mechanism_now,
)


class BudgetTests(unittest.TestCase):
    def test_product_default_is_auto(self) -> None:
        self.assertEqual(DEFAULT_EXPERIMENT_ROUNDS, -1)
        cap, mode = resolve_experiment_rounds(-1)
        self.assertEqual(cap, EXPERIMENT_AUTO_CAP)
        self.assertEqual(mode, "auto")
        self.assertEqual(resolve_experiment_rounds(0), (0, "fixed"))
        self.assertEqual(resolve_experiment_rounds(2), (2, "fixed"))


class BottleneckTests(unittest.TestCase):
    def test_supports_and_weakens_are_informative(self) -> None:
        self.assertIsNone(classify_bottleneck(verdict="supports"))
        self.assertIsNone(classify_bottleneck(verdict="weakens", status="ran"))

    def test_a_crash_is_implementation(self) -> None:
        self.assertEqual(
            classify_bottleneck(status="crashed"),
            "implementation",
        )
        self.assertEqual(
            classify_bottleneck(refused="probe"),
            "implementation",
        )

    def test_uninformative_and_a_refused_diagnosis_are_method(self) -> None:
        self.assertEqual(
            classify_bottleneck(status="ran", verdict="uninformative"),
            "method",
        )
        self.assertEqual(
            classify_bottleneck(refused="diagnosis"),
            "method",
        )


class PipelineBottleneckTests(unittest.TestCase):
    def test_prior_outranks_an_informative_probe(self) -> None:
        self.assertEqual(
            pipeline_bottleneck({"prior_kills": True, "verdict": "supports"}),
            "prior",
        )

    def test_graph_kills_and_text_kills_split(self) -> None:
        self.assertEqual(
            pipeline_bottleneck({"killed_by": ["pair_not_already_combined"]}),
            "graph",
        )
        self.assertEqual(
            pipeline_bottleneck({"killed_by": ["prediction_names_a_quantity"]}),
            "predicate",
        )

    def test_host_block_is_named_when_the_probe_already_spoke(self) -> None:
        self.assertEqual(
            pipeline_bottleneck(
                {"verdict": "supports", "host_ok": False, "probe_kind": "WORLD"}
            ),
            "host",
        )

    def test_a_constructed_world_is_not_a_catalog_stop(self) -> None:
        self.assertEqual(
            pipeline_bottleneck(
                {
                    "world_incompatible": True,
                    "generated_world": True,
                    "experiment_bottleneck": "method",
                }
            ),
            "probe_method",
        )


class RetryRuleTests(unittest.TestCase):
    def test_an_informative_verdict_never_redesigns(self) -> None:
        decision = decide_retry(attempt=0, cap=2, bottleneck=None)
        self.assertFalse(decision["continue"])
        self.assertEqual(decision["reason"], "informative")
        self.assertIn("chase", decision["detail"])

    def test_uninformative_redesigns_the_diagnosis_not_the_claim(self) -> None:
        decision = decide_retry(attempt=0, cap=2, bottleneck="method")
        self.assertTrue(decision["continue"])
        self.assertEqual(decision["action"], "rewrite_diagnosis")

    def test_a_crash_rewrites_the_script_and_keeps_the_registration(self) -> None:
        decision = decide_retry(attempt=0, cap=2, bottleneck="implementation")
        self.assertTrue(decision["continue"])
        self.assertEqual(decision["action"], "rewrite_probe")

    def test_one_shot_stops_after_the_first_attempt(self) -> None:
        decision = decide_retry(attempt=0, cap=0, bottleneck="method")
        self.assertFalse(decision["continue"])
        self.assertEqual(decision["reason"], "hit_cap")
        self.assertEqual(decision["bottleneck"], "method")

    def test_a_spent_budget_is_never_booked_as_an_executable_plan(self) -> None:
        # Three 0/0 probes on a constructed world used to close the
        # mission as `plan_executable`; the budget says nothing about
        # whether the registration can run.
        for bottleneck in ("method", "implementation", "world_model", "host"):
            decision = decide_retry(attempt=2, cap=2, bottleneck=bottleneck)
            self.assertFalse(decision["continue"])
            self.assertEqual(decision["reason"], "hit_cap", bottleneck)
            self.assertNotIn("executable plan", decision["detail"])

    def test_the_cap_is_extra_attempts_after_the_first(self) -> None:
        self.assertTrue(decide_retry(attempt=1, cap=2, bottleneck="method")["continue"])
        self.assertFalse(decide_retry(attempt=2, cap=2, bottleneck="method")["continue"])

    def test_no_handle_stops_the_experiment_loop(self) -> None:
        self.assertEqual(classify_bottleneck(refused="no_handle"), "no_handle")
        decision = decide_retry(attempt=0, cap=2, bottleneck="no_handle")
        self.assertFalse(decision["continue"])
        self.assertEqual(decision["reason"], "no_handle")
        self.assertIn("lever table", decision["detail"])
        self.assertEqual(
            pipeline_bottleneck({"experiment_bottleneck": "no_handle"}),
            "no_handle",
        )

    def test_a_crash_on_a_constructed_world_adapts_it_in_place(self) -> None:
        self.assertEqual(
            classify_bottleneck(
                status="crashed",
                error="FileNotFoundError: data/world.json",
                world_role="generated",
            ),
            "world_model",
        )
        decision = decide_retry(attempt=0, cap=2, bottleneck="world_model")
        self.assertTrue(decision["continue"])
        self.assertEqual(decision["action"], "adapt_world")

    def test_a_second_world_crash_rewrites_the_script(self) -> None:
        self.assertEqual(
            classify_bottleneck(
                status="crashed",
                error="FileNotFoundError: data/world.json",
                world_role="generated",
                last_action="adapt_world",
            ),
            "implementation",
        )

    def test_an_attested_world_crash_is_still_implementation(self) -> None:
        self.assertEqual(
            classify_bottleneck(
                status="crashed",
                error="FileNotFoundError: data/world.json",
                world_role="world",
            ),
            "implementation",
        )


    def test_a_second_world_uninformative_switches_the_lever_in_this_mission(self) -> None:
        self.assertFalse(switch_mechanism_now(uninformative_runs=1, probe_kind="WORLD"))
        self.assertTrue(switch_mechanism_now(uninformative_runs=2, probe_kind="WORLD"))
        self.assertFalse(
            switch_mechanism_now(uninformative_runs=2, probe_kind="SYNTHETIC")
        )
        self.assertTrue(
            switch_mechanism_now(uninformative_runs=2, probe_kind="GENERATED")
        )

    def test_object_absent_on_a_generated_world_adapts_it(self) -> None:
        self.assertEqual(
            classify_bottleneck(
                status="ran",
                verdict="uninformative",
                world_role="generated",
                object_absent=True,
            ),
            "world_model",
        )
        self.assertEqual(
            classify_bottleneck(
                status="ran",
                verdict="uninformative",
                world_role="world",
                object_absent=True,
            ),
            "method",
        )


if __name__ == "__main__":
    unittest.main()
