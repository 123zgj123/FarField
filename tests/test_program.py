"""The generation program constrains the next generate in this mission."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from farfield.extras.generate import generate_card
from farfield.extras.mission import PRODUCTION_CORPUS
from farfield.extras.program import compile_program, prompt_lines, target_line
from farfield.extras.state import load, program_for
from tests.test_generate import FakeClient, LABELS, answer
from tests.test_mission import TOPIC, SchemingClient, run_mission


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
                }
            ],
            kills=[{"pair": ["succinct data structure", "hash table"], "killed_by": ["endpoint_is_not_a_concept_hub"]}],
        )
        self.assertIsNotNone(program)
        self.assertIn("linear space", program["commit"])
        self.assertIn("public trace", program["next_card_must"])
        self.assertIn("hub", program["do_not_generate"])
        self.assertIn("this-mission probes: supports/WORLD", program["h"]["verifiers"])
        self.assertTrue(prompt_lines(program))
        self.assertIn("committed program", target_line(program) or "")

    def test_a_world_gap_constructs_the_experiment_world(self) -> None:
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
        self.assertIn("this mission", program["next_card_must"])
        self.assertIn("construct the executable world", program["next_card_must"])
        self.assertIn("measurable", program["do_not_generate"])

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
        self.assertIn("attested world", program["next_card_must"])
        self.assertIn("no attested world bound", program["h"]["tools"])

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
        self.assertIn("parent was not executed", program["next_card_must"])
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
        self.assertIn("UNINFORMATIVE", program["next_card_must"])
        self.assertIn("switch the mechanism", program["next_card_must"])
        self.assertIn("pipeline probe_method", program["h"]["verifiers"])


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
    def test_a_finished_mission_banks_a_program_that_shrinks_the_next_spray(self) -> None:
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
                    TOPIC, jumps=2, candidates=1, client=SchemingClient(), **shared
                )
            )
            prog = next(e for e in first if e["stage"] == "program")
            self.assertTrue(prog["commit"])
            anchor = next(e for e in first if e["stage"] == "anchor")["anchors"][0]["concept"]
            stored = program_for(load(shared["state_store"]), PRODUCTION_CORPUS, anchor)
            self.assertIsNotNone(stored)
            self.assertEqual(stored["commit"], prog["commit"])
            client = SchemingClient()
            second = list(
                run_mission(
                    TOPIC,
                    jumps=4,
                    candidates=1,
                    client=client,
                    **shared,
                )
            )
            agenda = next(e for e in second if e["stage"] == "agenda")
            self.assertTrue(agenda["deepen"])
            far = [s for s in agenda["slots"] if s["track"] == "farfield"]
            self.assertEqual(len(far), 4)
            self.assertEqual(far[0].get("role"), "exploit")
            self.assertTrue(
                any("Committed research program" in p for p in client.prompts),
                client.prompts[0][:200] if client.prompts else "no prompts",
            )
            self.assertTrue(
                any("Runtime H" in p for p in client.prompts),
                "next mission must see compiled runtime H",
            )


if __name__ == "__main__":
    unittest.main()
