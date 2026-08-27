"""Explore and polish budgets are a cap plus an attested stop rule."""

from __future__ import annotations

import unittest

from farfield.extras.explore import (
    DEFAULT_EXPLORE,
    POLISH_AUTO_CAP,
    decide_idea_refine,
    decide_next,
    has_live_line,
    has_research_plan,
    normalize_explore,
    pair_structurally_dead,
    resolve_idea_rounds,
    resolve_polish_rounds,
)


class ExploreRuleTests(unittest.TestCase):
    def test_product_default_is_auto(self) -> None:
        self.assertEqual(normalize_explore(None), "auto")
        self.assertEqual(normalize_explore(""), DEFAULT_EXPLORE)
        self.assertEqual(normalize_explore("FIXED"), "fixed")

    def test_a_supported_line_stops_further_generation(self) -> None:
        decision = decide_next(
            opened=1,
            cap=4,
            records=[
                {
                    "verdict": "supports",
                    "prior_kills": False,
                    "probe_kind": "WORLD",
                }
            ],
        )
        self.assertFalse(decision["continue"])
        self.assertEqual(decision["reason"], "has_plan")

    def test_synthetic_support_does_not_stop_spray(self) -> None:
        decision = decide_next(
            opened=1,
            cap=4,
            records=[{"verdict": "supports", "prior_kills": False}],
        )
        self.assertTrue(decision["continue"])
        self.assertEqual(decision["reason"], "no_plan")

    def test_a_registered_experiment_does_not_stop_spray(self) -> None:
        decision = decide_next(
            opened=1,
            cap=4,
            records=[
                {
                    "verdict": "uninformative",
                    "experiment": "walk the constructed trace with the mechanism flag",
                    "generated_world": True,
                    "ledger": "diagnostic",
                }
            ],
        )
        self.assertTrue(decision["continue"])
        self.assertEqual(decision["reason"], "no_plan")
        self.assertFalse(
            has_research_plan(
                [{"experiment": "two-arm test", "verdict": "uninformative"}]
            )
        )

    def test_weakened_or_closed_lines_keep_exploring(self) -> None:
        decision = decide_next(
            opened=1,
            cap=4,
            records=[
                {"verdict": "weakens"},
                {"verdict": "supports", "prior_kills": True},
            ],
        )
        self.assertTrue(decision["continue"])
        self.assertEqual(decision["reason"], "no_plan")

    def test_nothing_entered_continues_until_the_cap(self) -> None:
        self.assertTrue(decide_next(opened=1, cap=4, records=[])["continue"])
        hit = decide_next(opened=4, cap=4, records=[])
        self.assertFalse(hit["continue"])
        self.assertEqual(hit["reason"], "hit_cap")

    def test_entered_without_a_verdict_is_not_a_live_line(self) -> None:
        self.assertFalse(has_live_line([{"verdict": None, "prior_kills": False}]))
        self.assertFalse(has_live_line([{"verdict": "weakens"}]))
        self.assertTrue(has_live_line([{"verdict": "supports", "probe_kind": "WORLD"}]))
        self.assertFalse(has_live_line([{"verdict": "supports"}]))
        self.assertFalse(has_research_plan([{"verdict": None}]))

    def test_polish_minus_one_is_auto_with_a_cap(self) -> None:
        cap, mode = resolve_polish_rounds(-1)
        self.assertEqual(cap, POLISH_AUTO_CAP)
        self.assertEqual(mode, "auto")
        self.assertEqual(resolve_polish_rounds(0), (0, "fixed"))
        self.assertEqual(resolve_polish_rounds(2), (2, "fixed"))

    def test_idea_refine_rewrites_the_same_line_until_it_enters(self) -> None:
        cap, mode = resolve_idea_rounds(-1)
        self.assertEqual(cap, 2)
        self.assertEqual(mode, "auto")
        keep = decide_idea_refine(
            attempt=0, cap=2, killed_by=["prediction_names_a_measurable_quantity"]
        )
        self.assertTrue(keep["continue"])
        self.assertEqual(keep["action"], "refine_idea")
        stop = decide_idea_refine(attempt=0, cap=2, alive=True)
        self.assertFalse(stop["continue"])
        hit = decide_idea_refine(
            attempt=2, cap=2, killed_by=["prediction_names_a_measurable_quantity"]
        )
        self.assertFalse(hit["continue"])

    def test_a_graph_dead_pair_does_not_burn_idea_rounds(self) -> None:
        self.assertTrue(pair_structurally_dead(["endpoint_is_not_a_concept_hub"]))
        decision = decide_idea_refine(
            attempt=0, cap=2, killed_by=["endpoint_is_not_a_concept_hub"]
        )
        self.assertFalse(decision["continue"])
        self.assertEqual(decision["reason"], "graph_pair")


if __name__ == "__main__":
    unittest.main()
