"""Explore and polish budgets are a cap plus an attested stop rule."""

from __future__ import annotations

import unittest

from farfield.extras.explore import (
    COMPILE_REFINE_CAP,
    DEFAULT_EXPLORE,
    POLISH_AUTO_CAP,
    ExplorationPrior,
    compile_debt,
    continue_reason,
    decide_compile_refine,
    decide_idea_refine,
    decide_next,
    has_live_line,
    has_research_plan,
    is_continue_program,
    is_live_program,
    landing_category,
    normalize_explore,
    pair_structurally_dead,
    plan_executable,
    stored_plan_executable,
    resolve_idea_rounds,
    resolve_polish_rounds,
)


class ExploreRuleTests(unittest.TestCase):
    def test_product_default_is_auto(self) -> None:
        self.assertEqual(normalize_explore(None), "auto")
        self.assertEqual(normalize_explore(""), DEFAULT_EXPLORE)
        self.assertEqual(normalize_explore("FIXED"), "fixed")

    def test_a_neighbourhood_live_line_stops_spray_without_a_deepen_generate(self) -> None:
        decision = decide_next(
            opened=0,
            cap=4,
            records=[],
            prior_live=True,
        )
        self.assertFalse(decision["continue"])
        self.assertEqual(decision["reason"], "plan_executable")
        self.assertFalse(
            decide_next(opened=1, cap=4, records=[], prior_live=True)["continue"]
        )

    def test_one_executable_plan_is_not_a_choice_before_the_breadth_floor(self) -> None:
        # cwm-iclr2027 mission3: jumps=4, one WORLD uninformative on the
        # first landing, spray stopped, nothing to rank. With a floor of 3
        # the mission keeps opening distinct landings; the plan keeps its seat.
        records = [{"verdict": "uninformative", "prior_kills": False, "probe_kind": "WORLD"}]
        first = decide_next(opened=1, cap=4, records=records, min_landings=3)
        self.assertTrue(first["continue"])
        self.assertEqual(first["reason"], "breadth_floor")
        third = decide_next(opened=3, cap=4, records=records, min_landings=3)
        self.assertFalse(third["continue"])
        self.assertEqual(third["reason"], "plan_executable")
        # The floor never exceeds the cap, and 1 restores stop-at-first-plan.
        capped = decide_next(opened=1, cap=1, records=records, min_landings=3)
        self.assertFalse(capped["continue"])
        legacy = decide_next(opened=1, cap=4, records=records, min_landings=1)
        self.assertEqual(legacy["reason"], "plan_executable")
        # An inherited neighbourhood plan is a continue, not a floor case.
        inherited = decide_next(opened=0, cap=4, records=[], prior_plan=True, min_landings=3)
        self.assertEqual(inherited["reason"], "plan_executable")

    def test_a_blind_measure_does_not_make_the_plan_executable(self) -> None:
        # The DV sanity gate books a measure that ignores the oracle as
        # object_absent; the seat stays open exactly as for a 0/0 probe.
        row = {"verdict": "uninformative", "probe_kind": "WORLD", "object_absent": True, "dv_blind": True}
        self.assertFalse(plan_executable(row))
        self.assertTrue(decide_next(opened=1, cap=4, records=[row], min_landings=1)["continue"])

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
        self.assertEqual(decision["reason"], "plan_executable")

    def test_synthetic_support_stops_spray_because_the_plan_is_executable(self) -> None:
        decision = decide_next(
            opened=1,
            cap=4,
            records=[{"verdict": "supports", "prior_kills": False}],
        )
        self.assertFalse(decision["continue"])
        self.assertEqual(decision["reason"], "plan_executable")

    def test_a_world_uninformative_experiment_stops_spray(self) -> None:
        decision = decide_next(
            opened=1,
            cap=4,
            records=[
                {
                    "verdict": "uninformative",
                    "experiment": "mask failures on the attested freeze",
                    "probe_kind": "WORLD",
                }
            ],
        )
        self.assertFalse(decision["continue"])
        self.assertEqual(decision["reason"], "plan_executable")
        self.assertFalse(
            has_research_plan(
                [{"experiment": "two-arm test", "verdict": "uninformative"}]
            )
        )
        self.assertTrue(
            plan_executable(
                {
                    "verdict": "uninformative",
                    "experiment": "two-arm test",
                    "probe_kind": "WORLD",
                }
            )
        )
        self.assertTrue(
            stored_plan_executable(
                None,
                {"verdict": "supports", "probe_kind": "SYNTHETIC"},
            )
        )
        self.assertFalse(stored_plan_executable(None))
        self.assertFalse(
            plan_executable(
                {
                    "experiment": (
                        "Skip the probe: the frozen lifecycle automaton "
                        "has no self-edit lever"
                    ),
                    "experiment_bottleneck": "no_handle",
                }
            )
        )
        self.assertFalse(
            plan_executable(
                {
                    "idea_kind": "acquire",
                    "experiment": "harvest a validators_mutable world",
                    "verdict": None,
                }
            )
        )

    def test_a_generated_uninformative_does_not_occupy_the_seat(self) -> None:
        """Three 0/0 rehearsals on a constructed world are not a plan.

        The live failure: a GENERATED probe went uninformative, decide_next
        said plan_executable, and the mission welded itself to a dead
        fixture. A constructed rehearsal that failed to speak leaves the
        landing open; a probe that never met its object leaves it open even
        on a freeze.
        """
        decision = decide_next(
            opened=1,
            cap=4,
            records=[
                {
                    "verdict": "uninformative",
                    "experiment": "walk the constructed trace with the mechanism flag",
                    "probe_kind": "GENERATED",
                    "generated_world": True,
                }
            ],
        )
        self.assertTrue(decision["continue"])
        self.assertFalse(
            plan_executable(
                {
                    "verdict": "uninformative",
                    "experiment": "two-arm test",
                    "probe_kind": "GENERATED",
                }
            )
        )
        self.assertFalse(
            plan_executable(
                {
                    "verdict": "uninformative",
                    "experiment": "two-arm test",
                    "probe_kind": "WORLD",
                    "object_absent": True,
                }
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
        self.assertEqual(decision["reason"], "idea_closed")

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
        self.assertTrue(
            is_live_program({"verdict": "supports", "probe_kind": "WORLD"})
        )
        self.assertFalse(
            is_live_program({"verdict": "supports", "probe_kind": "SYNTHETIC"})
        )
        self.assertTrue(
            is_continue_program({"verdict": "supports", "probe_kind": "WORLD"})
        )
        self.assertTrue(
            is_continue_program(
                {
                    "verdict": "uninformative",
                    "probe_kind": "WORLD",
                    "must_switch_mechanism": True,
                }
            )
        )
        self.assertTrue(
            is_continue_program(
                {
                    "verdict": "uninformative",
                    "probe_kind": "WORLD",
                }
            )
        )
        self.assertFalse(
            is_continue_program(
                {
                    "verdict": "uninformative",
                    "probe_kind": "SYNTHETIC",
                    "must_switch_mechanism": True,
                }
            )
        )
        self.assertFalse(
            is_continue_program(
                {
                    "verdict": "uninformative",
                    "probe_kind": "GENERATED",
                    "must_switch_mechanism": True,
                }
            )
        )
        self.assertFalse(
            is_continue_program(
                {
                    "verdict": "weakens",
                    "probe_kind": "WORLD",
                    "must_switch_mechanism": True,
                }
            )
        )
        self.assertEqual(
            continue_reason({"verdict": "supports", "probe_kind": "WORLD"}),
            "plan_executable",
        )
        self.assertEqual(
            continue_reason(
                {
                    "verdict": "uninformative",
                    "probe_kind": "WORLD",
                    "must_switch_mechanism": True,
                }
            ),
            "plan_executable",
        )
        self.assertEqual(
            continue_reason(
                {"verdict": "uninformative", "probe_kind": "WORLD"}
            ),
            "plan_executable",
        )
        self.assertEqual(continue_reason(None), "")

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

    def test_compile_refine_is_one_same_pair_rewrite_not_spray(self) -> None:
        self.assertEqual(COMPILE_REFINE_CAP, 1)
        self.assertEqual(
            compile_debt(structural_notes=["arms share a cache"]),
            ["structural"],
        )
        self.assertEqual(
            compile_debt(
                world_lever="none",
                levers=("mask_tools", "delay_tools"),
                alternative="",
            ),
            ["no_handle", "competing_explanation"],
        )
        keep = decide_compile_refine(attempt=0, reasons=["structural"])
        self.assertTrue(keep["continue"])
        self.assertEqual(keep["reason"], "compile")
        self.assertIn("same pair", keep["detail"])
        idle = decide_compile_refine(attempt=0, reasons=[])
        self.assertFalse(idle["continue"])
        self.assertEqual(idle["reason"], "no_compile_debt")
        hit = decide_compile_refine(attempt=1, reasons=["structural"])
        self.assertFalse(hit["continue"])
        self.assertEqual(hit["reason"], "hit_cap")
        entered = decide_idea_refine(attempt=0, cap=2, alive=True)
        self.assertEqual(entered["reason"], "entered")


class ExplorationNotebookTests(unittest.TestCase):
    def test_a_later_jump_sees_tried_routes_not_another_ideas_program(self) -> None:
        prior = ExplorationPrior(object_phrases=("formal verification", "agent security"))
        prior.absorb_landing(
            pair=("formal verification", "compression ratio"),
            far_menu=("compression ratio", "sparse cut"),
            category="world_weakens",
            operator="directional",
        )
        lines = "\n".join(prior.prompt_lines())
        self.assertIn("formal verification × compression ratio [world_weakens]", lines)
        self.assertIn("sparse cut", lines)
        self.assertIn("Topic object phrases", lines)
        self.assertNotIn("landing vector", lines.lower())
        snap = prior.snapshot()
        self.assertEqual(snap["used_far"], ["compression ratio"])
        self.assertNotIn("index", snap.get("how_found") or {})
        self.assertNotIn("landing", snap)

    def test_world_support_records_how_the_pair_was_found(self) -> None:
        prior = ExplorationPrior()
        prior.absorb_landing(
            pair=("succinct data structure", "wavelet tree"),
            far_menu=("wavelet tree", "rank query"),
            category="world_supports",
            operator="analogy",
        )
        self.assertEqual(prior.how_found["operator"], "analogy")
        self.assertEqual(prior.how_found["pair"], ["succinct data structure", "wavelet tree"])
        self.assertEqual(prior.how_found["unused_menu"], ["rank query"])
        self.assertNotIn("index", prior.how_found)

    def test_landing_category_reads_attested_records_only(self) -> None:
        self.assertEqual(
            landing_category(
                {"verdict": "supports", "probe_kind": "WORLD", "prior_kills": False}
            ),
            "world_supports",
        )
        self.assertEqual(
            landing_category({"verdict": "weakens", "probe_kind": "WORLD"}),
            "world_weakens",
        )
        self.assertEqual(
            landing_category(killed_by=["endpoint_is_not_a_concept_hub"]),
            "graph_pair",
        )
        self.assertEqual(
            landing_category(generated=False),
            "generation_refused",
        )
        self.assertEqual(
            landing_category({"experiment_bottleneck": "no_handle"}),
            "no_handle",
        )
        self.assertEqual(
            landing_category({"world_incompatible": True}),
            "world_incompatible",
        )

    def test_no_handle_refines_the_same_pair_instead_of_spraying(self) -> None:
        keep = decide_idea_refine(
            attempt=0, cap=2, killed_by=["no_handle"], alive=True
        )
        self.assertTrue(keep["continue"])
        self.assertEqual(keep["reason"], "no_handle")
        stop = decide_idea_refine(
            attempt=0, cap=2, killed_by=["world_incompatible"], alive=True
        )
        self.assertFalse(stop["continue"])
        self.assertEqual(stop["reason"], "entered")

    def test_adversarial_must_change_spends_an_idea_round(self) -> None:
        keep = decide_idea_refine(
            attempt=0,
            cap=1,
            alive=True,
            review_changes=("name the replayed frozen state",),
        )
        self.assertTrue(keep["continue"])
        self.assertEqual(keep["reason"], "review")
        skip = decide_idea_refine(attempt=0, cap=1, alive=True)
        self.assertFalse(skip["continue"])
        self.assertEqual(skip["reason"], "entered")

    def test_used_levers_are_notebook_not_a_new_far_menu(self) -> None:
        prior = ExplorationPrior()
        prior.absorb_landing(
            pair=("agent trace", "knapsack"),
            far_menu=("knapsack", "lz77"),
            category="no_handle",
            operator="directional",
            world_lever="mask_tools",
        )
        snap = prior.snapshot()
        self.assertEqual(snap["used_levers"], ["mask_tools"])
        lines = "\n".join(prior.prompt_lines())
        self.assertIn("mask_tools", lines)
        self.assertIn("Spray breadth", lines)

    def test_a_lesson_reaches_the_next_jump_without_another_ideas_program(self) -> None:
        prior = ExplorationPrior()
        prior.absorb_landing(
            pair=("recursive algorithm", "stochastic reward"),
            far_menu=("stochastic reward",),
            category="uninformative",
            why="14-of-16 certificate had no tampering labels",
            iterate="measure hidden-seed rewards on the same object",
        )
        lines = "\n".join(prior.prompt_lines())
        self.assertIn("tampering labels", lines)
        self.assertIn("hidden-seed rewards", lines)
        self.assertNotIn("landing vector", lines.lower())
        self.assertEqual(prior.snapshot()["lessons"][0]["category"], "uninformative")

    def test_absorb_without_why_keeps_the_old_notebook_shape(self) -> None:
        prior = ExplorationPrior()
        prior.absorb_landing(
            pair=("formal verification", "compression ratio"),
            category="world_weakens",
        )
        self.assertNotIn("lessons", prior.snapshot())
        self.assertNotIn("Approaches already tried", "\n".join(prior.prompt_lines()))

    def test_landing_feedback_steers_the_next_jump(self) -> None:
        from farfield.extras.researchspace import JumpFeedback

        prior = ExplorationPrior()
        prior.absorb_landing(
            pair=("formal verification", "compression ratio"),
            category="entered",
        )
        prior.absorb_feedback(
            JumpFeedback(
                support="strong",
                novelty="linear",
                value="unclear",
                feasibility="open",
                distance="too_close",
                policy="increase_distance",
                detail="too close to the last paper",
            ),
            far="compression ratio",
        )
        snap = prior.snapshot()
        self.assertEqual(snap["search"]["policy"], "increase_distance")
        lines = "\n".join(prior.prompt_lines())
        self.assertIn("increase_distance", lines)
        self.assertIn("too_close", lines)


if __name__ == "__main__":
    unittest.main()
