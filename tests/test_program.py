"""The generation program constrains the next generate in this mission."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from farfield.extras.generate import generate_card
from farfield.extras.mission import PRODUCTION_CORPUS
from farfield.extras.program import (
    compile_program,
    continue_duty_lines,
    prompt_lines,
    target_line,
)
from farfield.extras.state import load, program_for
from tests.test_generate import FakeClient, LABELS, answer
from tests.test_mission import TOPIC, FakeFeed, SchemingClient, run_mission


class CompileProgramTests(unittest.TestCase):
    def test_a_supported_line_becomes_the_next_generation_constraint(self) -> None:
        program = compile_program(
            "succinct pivots",
            ranked=[{"rank": 1, "card_id": "live", "verdict": "supports"}],
            found=[
                {
                    "card_id": "live",
                    "claim": "store pivot history in linear space",
                    "mechanism": "the log repeats",
                    "prediction": "later work reports sub-100ns rank",
                    "experiment": "time rank on a public trace",
                    "verdict": "supports",
                    "probe_kind": "WORLD",
                    "pair": ["succinct data structure", "pivot rule"],
                    "has_brief": True,
                    "idea_world": {
                        "objects": ["pivot history", "rank query"],
                        "iterate": "measure rank on the attested log, not a new graph noun",
                    },
                }
            ],
            kills=[{"pair": ["succinct data structure", "hash table"], "killed_by": ["endpoint_is_not_a_concept_hub"]}],
        )
        self.assertIsNotNone(program)
        self.assertIn("linear space", program["commit"])
        self.assertIn("farfield execute", program["next_card_must"])
        self.assertIn("Do not generate", program["next_card_must"])
        self.assertIn("hub", program["do_not_generate"])
        self.assertIn("this-mission probes: supports/WORLD", program["h"]["verifiers"])
        self.assertIn("idea-world objects", program["h"]["tools"])
        self.assertIn("idea-world iterate", program["h"]["verifiers"])
        self.assertTrue(prompt_lines(program))
        self.assertIn("committed program", target_line(program) or "")

    def test_a_world_gap_does_not_construct_a_substitute(self) -> None:
        program = compile_program(
            "diffusion sampler invariants",
            found=[
                {
                    "card_id": "fv",
                    "claim": "a Lyapunov potential certifies the sampler invariant",
                    "mechanism": "reverse Kolmogorov drift is nonpositive",
                    "prediction": "unsafe trajectory fraction is 0.0",
                    "pair": ["formal verification", "zero weight"],
                    "world_incompatible": True,
                    "ledger": "diagnostic",
                }
            ],
            kills=[
                {
                    "pair": ["formal verification", "terminal set"],
                    "killed_by": ["prediction_names_a_measurable_quantity"],
                }
            ],
        )
        self.assertIsNotNone(program)
        self.assertIn("Lyapunov", program["commit"])
        self.assertIn("this pair has no attested freeze", program["next_card_must"])
        self.assertIn("do not construct a substitute", program["next_card_must"])
        self.assertNotIn("construct the executable world", program["next_card_must"])
        self.assertIn("measurable", program["do_not_generate"])
        self.assertNotIn("terminal set", program["do_not_generate"])

    def test_a_synthetic_support_tells_the_next_card_to_bind_a_world(self) -> None:
        program = compile_program(
            "succinct pivots",
            ranked=[{"rank": 1, "card_id": "live", "verdict": "supports"}],
            found=[
                {
                    "card_id": "live",
                    "claim": "store pivot history in linear space",
                    "verdict": "supports",
                    "probe_kind": "SYNTHETIC",
                    "experiment": "invent ten random instances",
                    "has_brief": True,
                }
            ],
        )
        self.assertIn("SYNTHETIC", program["next_card_must"])
        self.assertIn("attested fixture", program["next_card_must"])
        self.assertIn("farfield execute", program["next_card_must"])
        self.assertIn("no attested world bound", program["h"]["tools"])

    def test_a_generated_support_does_not_forbid_spray(self) -> None:
        program = compile_program(
            "succinct pivots",
            ranked=[{"rank": 1, "card_id": "live", "verdict": "supports"}],
            found=[
                {
                    "card_id": "live",
                    "claim": "store pivot history in linear space",
                    "verdict": "supports",
                    "probe_kind": "GENERATED",
                    "experiment": "two-arm on a constructed stream",
                    "has_brief": True,
                }
            ],
        )
        self.assertIn("constructed world", program["next_card_must"])
        self.assertIn("farfield freeze", program["next_card_must"])
        self.assertIn("cannot occupy the continue seat", program["next_card_must"])
        self.assertIn("corroborate", program["next_card_must"])

    def test_runtime_h_is_compiled_and_a_blocked_host_is_not_a_discovery(self) -> None:
        program = compile_program(
            "graphs",
            ranked=[{"rank": 1, "card_id": "live", "verdict": "supports"}],
            found=[
                {
                    "card_id": "live",
                    "claim": "dynamic connectivity on an attested collaboration graph",
                    "mechanism": "the spanning forest is compressible",
                    "verdict": "supports",
                    "probe_kind": "WORLD",
                    "host_ok": False,
                    "host_status": "parent cache missing",
                    "world_id": "snap-ca-grqc",
                    "world_schema": "undirected_graph",
                    "operator": "directional",
                    "has_brief": True,
                }
            ],
            kills=[{"pair": ["graph", "transformer"], "killed_by": ["endpoint_is_not_a_concept_hub"]}],
        )
        self.assertIn("farfield execute", program["next_card_must"])
        self.assertIn("20s slice is a filter", program["next_card_must"])
        self.assertIn("snap-ca-grqc", program["h"]["tools"])
        self.assertIn("undirected_graph", program["h"]["tools"])
        self.assertNotIn("WORLD-supported", program["h"]["routing"])
        self.assertIn("operators that entered", program["h"]["routing"])
        self.assertIn("endpoint_is_not_a_concept_hub", program["h"]["verifiers"])
        self.assertTrue(any("Runtime H" in line for line in prompt_lines(program)))

    def test_a_spray_that_never_entered_tells_the_next_card_what_not_to_do(self) -> None:
        program = compile_program(
            "graphs",
            found=[],
            kills=[
                {
                    "pair": ["graph connectivity", "transformer"],
                    "killed_by": ["endpoint_is_not_a_concept_hub"],
                }
            ],
        )
        self.assertIn("did not enter research", program["commit"])
        self.assertIn("endpoint_is_not_a_concept_hub", program["next_card_must"])


    def test_a_world_uninformative_line_beats_a_synthetic_support_as_lead(self) -> None:
        program = compile_program(
            "graphs",
            ranked=[
                {"rank": 1, "card_id": "synth", "verdict": "supports"},
                {"rank": 2, "card_id": "world", "verdict": "uninformative"},
            ],
            found=[
                {
                    "card_id": "synth",
                    "claim": "invented instances always separate",
                    "verdict": "supports",
                    "probe_kind": "SYNTHETIC",
                    "value_score": 9.0,
                    "has_brief": True,
                },
                {
                    "card_id": "world",
                    "claim": "dynamic connectivity on an attested collaboration graph",
                    "mechanism": "the spanning forest is compressible",
                    "verdict": "uninformative",
                    "probe_kind": "WORLD",
                    "value_score": 0.1,
                    "has_brief": True,
                    "experiment": "time cuts on ca-GrQc",
                    "pipeline_bottleneck": "probe_method",
                },
            ],
        )
        self.assertIn("attested collaboration graph", program["commit"])
        self.assertIn("execute", program["next_card_must"])
        self.assertIn("EXPERIMENT_PLAN.md", program["next_card_must"])
        self.assertIn("pipeline probe_method", program["h"]["verifiers"])
        self.assertIn("pipeline probe_method", program["h"]["verifiers"])

    def test_ablation_debt_is_the_next_card_constraint(self) -> None:
        program = compile_program(
            "graphs",
            ranked=[{"rank": 1, "card_id": "live", "verdict": "supports"}],
            found=[
                {
                    "card_id": "live",
                    "claim": "dynamic connectivity on an attested collaboration graph",
                    "verdict": "supports",
                    "probe_kind": "WORLD",
                    "host_ok": True,
                    "ablation_required": True,
                    "mechanism_identified": False,
                    "experiment": "cut vs random on ca-GrQc",
                    "pair": ["graph connectivity", "min cut"],
                    "has_brief": True,
                    "world_id": "snap-ca-grqc",
                }
            ],
        )
        self.assertIn("execute 实验块 2", program["next_card_must"])
        self.assertIn("THIS freeze", program["next_card_must"])
        self.assertIn("ablation_required", program["h"]["verifiers"])
        self.assertEqual(program["world_id"], "snap-ca-grqc")
        self.assertIn("skills=", " ".join(prompt_lines(program)))

    def test_a_mechanism_switch_debt_rides_with_the_compiled_program(self) -> None:
        program = compile_program(
            "graphs",
            ranked=[{"rank": 1, "card_id": "world", "verdict": "uninformative"}],
            found=[
                {
                    "card_id": "world",
                    "claim": "dynamic connectivity on an attested collaboration graph",
                    "verdict": "uninformative",
                    "probe_kind": "WORLD",
                    "must_switch_mechanism": True,
                    "pair": ["graph connectivity", "min cut"],
                    "has_brief": True,
                }
            ],
        )
        self.assertTrue(program["must_switch_mechanism"])
        self.assertEqual(program["verdict"], "uninformative")
        self.assertEqual(program["probe_kind"], "WORLD")
        duty = continue_duty_lines(program)
        self.assertTrue(any("Keep pair" in line for line in duty))
        self.assertTrue(any("executable" in line for line in duty))
        self.assertFalse(any("Deepen THIS line" in line for line in duty))
        retry = compile_program(
            "graphs",
            found=[
                {
                    "card_id": "once",
                    "claim": "dynamic connectivity on an attested collaboration graph",
                    "verdict": "uninformative",
                    "probe_kind": "WORLD",
                    "pair": ["graph connectivity", "min cut"],
                    "has_brief": True,
                }
            ],
        )
        retry_duty = continue_duty_lines(retry)
        self.assertTrue(any("executable" in line for line in retry_duty))
        self.assertFalse(any("Stagnation rule" in line for line in retry_duty))
        synthetic = compile_program(
            "graphs",
            found=[
                {
                    "card_id": "synth",
                    "claim": "invented instances always separate",
                    "verdict": "uninformative",
                    "probe_kind": "SYNTHETIC",
                    "must_switch_mechanism": True,
                    "pair": ["graph connectivity", "hash table"],
                    "has_brief": True,
                }
            ],
        )
        self.assertFalse(synthetic.get("must_switch_mechanism"))
        self.assertEqual(continue_duty_lines(synthetic), ())


class PromptSlotTests(unittest.TestCase):
    def test_an_empty_program_leaves_the_generation_prompt_byte_identical(self) -> None:
        client = FakeClient(answer())
        generate_card(
            client,
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="analogy",
            alienness=0.5,
            label_to_node=LABELS,
        )
        self.assertNotIn("Committed research program", client.prompts[0])
        self.assertIn(
            "(operator: analogy): protein folding\n\nPropose ONE",
            client.prompts[0],
        )

    def test_a_program_reaches_the_generation_prompt(self) -> None:
        client = FakeClient(answer())
        generate_card(
            client,
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="analogy",
            alienness=0.5,
            label_to_node=LABELS,
            program=("Continue this line: keep the cache",),
        )
        self.assertIn("Committed research program", client.prompts[0])
        self.assertIn("keep the cache", client.prompts[0])


class MissionProgramTests(unittest.TestCase):
    def test_synthetic_support_stays_on_its_own_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shared = {
                "state_store": Path(tmp) / "state.json",
                "policy_log": Path(tmp) / "log.json",
                "policy_file": Path(tmp) / "policy.json",
                "polish_rounds": 0,
                "explore": "fixed",
            }
            first = list(
                run_mission(
                    TOPIC,
                    jumps=2,
                    candidates=1,
                    client=SchemingClient(),
                    feed=FakeFeed(),
                    **shared,
                )
            )
            prog = next(e for e in first if e["stage"] == "program")
            self.assertTrue(prog["commit"])
            anchor = next(e for e in first if e["stage"] == "anchor")["memory_anchor"]
            stored = program_for(load(shared["state_store"]), PRODUCTION_CORPUS, anchor)
            self.assertIsNone(stored)
            card = next(e for e in first if e["stage"] == "card")
            idea = program_for(
                load(shared["state_store"]),
                PRODUCTION_CORPUS,
                anchor,
                pair=card["pair"],
            )
            self.assertIsNotNone(idea)
            self.assertEqual(idea["commit"], prog["commit"])
            client = SchemingClient()
            second = list(
                run_mission(
                    TOPIC,
                    jumps=4,
                    candidates=1,
                    client=client,
                    feed=FakeFeed(),
                    **shared,
                )
            )
            agenda = next(e for e in second if e["stage"] == "agenda")
            self.assertFalse(agenda["deepen"])
            far = [s for s in agenda["slots"] if s["track"] == "farfield"]
            self.assertTrue(all(s.get("role") != "exploit" for s in far))
            self.assertEqual(agenda["explore_jumps"], 0)
            self.assertEqual(agenda["continue_reason"], "plan_executable")
            self.assertFalse(any(e["stage"] == "card" for e in second))
            self.assertFalse(
                any("Committed research program" in p for p in client.prompts),
            )

    def test_a_world_support_closes_spray_because_the_plan_is_executable(self) -> None:
        from tests.test_mission import FakeFeed, WorldProbeClient

        with tempfile.TemporaryDirectory() as tmp:
            shared = {
                "state_store": Path(tmp) / "state.json",
                "policy_log": Path(tmp) / "log.json",
                "policy_file": Path(tmp) / "policy.json",
                "polish_rounds": 0,
                "explore": "fixed",
            }
            first = list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=WorldProbeClient(),
                    feed=FakeFeed(),
                    world="path-trace",
                    **shared,
                )
            )
            prog = next(e for e in first if e["stage"] == "program")
            self.assertEqual(prog.get("probe_kind"), "WORLD")
            anchor = next(e for e in first if e["stage"] == "anchor")["memory_anchor"]
            stored = program_for(load(shared["state_store"]), PRODUCTION_CORPUS, anchor)
            self.assertIsNotNone(stored)
            self.assertEqual(stored["commit"], prog["commit"])
            client = WorldProbeClient()
            second = list(
                run_mission(
                    TOPIC,
                    jumps=4,
                    candidates=1,
                    client=client,
                    feed=FakeFeed(),
                    world="path-trace",
                    **shared,
                )
            )
            agenda = next(e for e in second if e["stage"] == "agenda")
            self.assertFalse(agenda["deepen"])
            self.assertTrue(agenda["continue"])
            self.assertEqual(agenda["continue_reason"], "plan_executable")
            self.assertEqual(agenda["explore_jumps"], 0)
            far = [s for s in agenda["slots"] if s["track"] == "farfield"]
            self.assertEqual(far, [])
            self.assertFalse(any(e["stage"] == "card" for e in second))
            decision = next(
                e
                for e in second
                if e["stage"] == "explore_decision" and e.get("track") == "farfield"
            )
            self.assertFalse(decision["continue"])
            self.assertEqual(decision["reason"], "plan_executable")

    def test_auto_explore_does_not_spray_over_an_executable_plan(self) -> None:
        from tests.test_mission import FakeFeed, WorldProbeClient

        with tempfile.TemporaryDirectory() as tmp:
            shared = {
                "state_store": Path(tmp) / "state.json",
                "policy_log": Path(tmp) / "log.json",
                "policy_file": Path(tmp) / "policy.json",
                "polish_rounds": 0,
                "explore": "auto",
            }
            first = list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=WorldProbeClient(),
                    feed=FakeFeed(),
                    world="path-trace",
                    **shared,
                )
            )
            self.assertTrue(any(e["stage"] == "card" for e in first))
            client = WorldProbeClient()
            second = list(
                run_mission(
                    TOPIC,
                    jumps=4,
                    candidates=1,
                    client=client,
                    feed=FakeFeed(),
                    world="path-trace",
                    **shared,
                )
            )
            far_decisions = [
                e
                for e in second
                if e["stage"] == "explore_decision" and e.get("track") == "farfield"
            ]
            self.assertEqual(len(far_decisions), 1)
            self.assertFalse(far_decisions[0]["continue"])
            self.assertEqual(far_decisions[0]["reason"], "plan_executable")
            self.assertFalse(any(e["stage"] == "jump" for e in second))
            self.assertFalse(any(e["stage"] == "card" for e in second))

    def test_a_world_uninformative_closes_spray_because_the_plan_is_executable(self) -> None:
        from farfield.extras.state import record_program

        with tempfile.TemporaryDirectory() as tmp:
            shared = {
                "state_store": Path(tmp) / "state.json",
                "policy_log": Path(tmp) / "log.json",
                "policy_file": Path(tmp) / "policy.json",
                "polish_rounds": 0,
                "explore": "auto",
            }
            first = list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=SchemingClient(),
                    feed=FakeFeed(),
                    **shared,
                )
            )
            card = next(e for e in first if e["stage"] == "card")
            anchor = next(e for e in first if e["stage"] == "anchor")["anchors"][0][
                "concept"
            ]
            record_program(
                shared["state_store"],
                PRODUCTION_CORPUS,
                anchor,
                {
                    "commit": "the last two WORLD tests did not discriminate",
                    "pair": list(card["pair"]),
                    "verdict": "uninformative",
                    "probe_kind": "WORLD",
                    "must_switch_mechanism": True,
                    "world_id": "path-trace",
                },
            )
            client = SchemingClient()
            second = list(
                run_mission(
                    TOPIC,
                    jumps=4,
                    candidates=1,
                    client=client,
                    feed=FakeFeed(),
                    **shared,
                )
            )
            agenda = next(e for e in second if e["stage"] == "agenda")
            self.assertFalse(agenda["deepen"])
            self.assertTrue(agenda["continue"])
            self.assertEqual(agenda["continue_reason"], "plan_executable")
            self.assertEqual(agenda["explore_jumps"], 0)
            self.assertFalse(any(e["stage"] == "card" for e in second))
            joined = "\n".join(client.prompts)
            self.assertNotIn("Continue THIS line", joined)
            self.assertNotIn("Deepen THIS line", joined)
            self.assertNotIn("Stagnation rule", joined)
            far_decisions = [
                e
                for e in second
                if e["stage"] == "explore_decision" and e.get("track") == "farfield"
            ]
            self.assertEqual(len(far_decisions), 1)
            self.assertFalse(far_decisions[0]["continue"])
            self.assertEqual(far_decisions[0]["reason"], "plan_executable")

    def test_the_first_world_uninformative_closes_spray_because_the_plan_is_executable(
        self,
    ) -> None:
        from farfield.extras.state import record_program

        with tempfile.TemporaryDirectory() as tmp:
            shared = {
                "state_store": Path(tmp) / "state.json",
                "policy_log": Path(tmp) / "log.json",
                "policy_file": Path(tmp) / "policy.json",
                "polish_rounds": 0,
                "explore": "auto",
            }
            first = list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=SchemingClient(),
                    feed=FakeFeed(),
                    **shared,
                )
            )
            card = next(e for e in first if e["stage"] == "card")
            anchor = next(e for e in first if e["stage"] == "anchor")["anchors"][0][
                "concept"
            ]
            record_program(
                shared["state_store"],
                PRODUCTION_CORPUS,
                anchor,
                {
                    "commit": "the first WORLD test did not discriminate",
                    "pair": list(card["pair"]),
                    "verdict": "uninformative",
                    "probe_kind": "WORLD",
                    "world_id": "path-trace",
                },
            )
            client = SchemingClient()
            second = list(
                run_mission(
                    TOPIC,
                    jumps=4,
                    candidates=1,
                    client=client,
                    feed=FakeFeed(),
                    **shared,
                )
            )
            agenda = next(e for e in second if e["stage"] == "agenda")
            self.assertFalse(agenda["deepen"])
            self.assertTrue(agenda["continue"])
            self.assertEqual(agenda["continue_reason"], "plan_executable")
            self.assertEqual(agenda["explore_jumps"], 0)
            self.assertFalse(any(e["stage"] == "jump" for e in second))
            joined = "\n".join(client.prompts)
            self.assertNotIn("Continue THIS line", joined)
            self.assertNotIn("Sharpen the experiment", joined)

    def test_a_missing_far_does_not_spray_over_an_executable_plan(self) -> None:
        from farfield.extras.state import record_program

        with tempfile.TemporaryDirectory() as tmp:
            shared = {
                "state_store": Path(tmp) / "state.json",
                "policy_log": Path(tmp) / "log.json",
                "policy_file": Path(tmp) / "policy.json",
                "polish_rounds": 0,
                "explore": "fixed",
            }
            first = list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=SchemingClient(),
                    feed=FakeFeed(),
                    **shared,
                )
            )
            anchor = next(e for e in first if e["stage"] == "anchor")["anchors"][0][
                "concept"
            ]
            record_program(
                shared["state_store"],
                PRODUCTION_CORPUS,
                anchor,
                {
                    "commit": "continue a pair the graph no longer names",
                    "pair": [anchor, "__far_not_in_this_graph__"],
                    "verdict": "uninformative",
                    "probe_kind": "WORLD",
                },
            )
            client = SchemingClient()
            second = list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=client,
                    feed=FakeFeed(),
                    **shared,
                )
            )
            agenda = next(e for e in second if e["stage"] == "agenda")
            self.assertEqual(agenda["explore_jumps"], 0)
            self.assertEqual(agenda["continue_reason"], "plan_executable")
            self.assertFalse(any(e["stage"] == "jump" for e in second))
            self.assertFalse(any(e["stage"] == "refused" for e in second))
            self.assertFalse(
                any("Continue THIS line" in p for p in client.prompts),
            )


if __name__ == "__main__":
    unittest.main()
