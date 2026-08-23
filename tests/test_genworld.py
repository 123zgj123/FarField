"""Constructed experimental worlds cannot corroborate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.genworld import (
    KIND_GENERATED,
    construct_world,
    execute_world,
    load_world_payload,
)
from farfield.extras.generate import GeneratedCard
from types import SimpleNamespace

from farfield.extras.world import (
    KIND_WORLD,
    WorldRequirement,
    infer_requirement,
    refine_requirement,
    world_attested,
)


def _card() -> GeneratedCard:
    return GeneratedCard(
        card_id="gen_test",
        operator="directional",
        claim="a Lyapunov potential certifies the sampler invariant",
        mechanism="reverse Kolmogorov drift is nonpositive",
        prediction="unsafe trajectory fraction is 0.0",
        falsifier="pair_not_already_combined",
        pair=("formal verification", "zero weight"),
        pair_nodes=("a", "b"),
        alienness=0.4,
        model="test",
        artifact_digest="d",
        artifact_uri="file:///dev/null",
        replay_mode="replay",
    )


class ConstructedWorldTests(unittest.TestCase):
    def test_a_stub_world_executes_and_is_not_attested(self) -> None:
        card = _card()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = construct_world(
                None,
                card,
                WorldRequirement(object_type="formula", environment=("formula",)),
                dest=Path(tmp) / "world",
            )
            self.assertEqual(fixture.role, "generated")
            self.assertEqual(fixture.schema, "symbolic_trace")
            payload = load_world_payload(fixture.root)
            self.assertEqual(execute_world(payload), [])
            honesty = json.loads((fixture.root / "world.json").read_text())["honesty"]
            self.assertNotIn("diagnostic", honesty.lower())
        self.assertFalse(
            world_attested({"probe_kind": KIND_GENERATED, "verdict": "supports"})
        )
        self.assertTrue(
            world_attested({"probe_kind": KIND_WORLD, "verdict": "supports"})
        )

    def test_a_dangling_transition_fails_execution(self) -> None:
        errors = execute_world(
            {
                "object_type": "formula",
                "invariant": "mass conserved",
                "states": [
                    {"id": "start", "label": "a"},
                    {"id": "safe", "label": "b"},
                ],
                "transitions": [{"src": "start", "dst": "missing", "action": "step"}],
            }
        )
        self.assertTrue(any("not a state" in item for item in errors))

    def test_adapt_rewrites_the_same_dest(self) -> None:
        card = _card()

        class Client:
            def complete(self, prompt, *, purpose, system=None):
                self.prompt = prompt

                class Done:
                    text = json.dumps(
                        {
                            "object_type": "formula",
                            "invariant": "mass conserved",
                            "states": [
                                {"id": "start", "label": "initial"},
                                {"id": "safe", "label": "ok"},
                            ],
                            "transitions": [
                                {"src": "start", "dst": "safe", "action": "step"}
                            ],
                        }
                    )

                    def assert_usable(self) -> None:
                        return None

                return Done()

        client = Client()
        broken = {
            "object_type": "formula",
            "invariant": "mass conserved",
            "states": [
                {"id": "start", "label": "a"},
                {"id": "safe", "label": "b"},
            ],
            "transitions": [{"src": "start", "dst": "missing", "action": "step"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "world"
            fixture = construct_world(
                client,
                card,
                {"object_type": "formula"},
                dest=dest,
                previous=broken,
                errors=["transition 0 dst 'missing' is not a state"],
            )
            self.assertEqual(fixture.root, dest)
            self.assertIn("Adapt THIS same world", client.prompt)
            self.assertEqual(execute_world(load_world_payload(dest)), [])

    def test_an_io_requirement_is_not_a_formula_automaton(self) -> None:
        card = _card()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = construct_world(
                None,
                card,
                WorldRequirement(object_type="io", environment=("io",)),
                dest=Path(tmp) / "world",
            )
            self.assertEqual(fixture.role, "generated")
            self.assertNotEqual(fixture.schema, "symbolic_trace")
            payload = load_world_payload(fixture.root)
            self.assertEqual(payload["acquisition"], "incomplete")
            self.assertTrue(execute_world(payload))
            self.assertFalse(
                world_attested({"probe_kind": KIND_GENERATED, "verdict": "supports"})
            )

    def test_a_graph_requirement_is_constructed_not_incomplete(self) -> None:
        card = _card()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = construct_world(
                None,
                card,
                WorldRequirement(object_type="graph", schema="undirected_graph"),
                dest=Path(tmp) / "world",
            )
            payload = load_world_payload(fixture.root)
            self.assertNotEqual(payload.get("acquisition"), "incomplete")
            self.assertEqual(fixture.schema, "undirected_graph")
            self.assertEqual(execute_world(payload), [])
            self.assertTrue((fixture.root / "graph.json").is_file())
        self.assertFalse(
            world_attested({"probe_kind": KIND_GENERATED, "verdict": "supports"})
        )

    def test_empty_requirement_follows_the_claim_object(self) -> None:
        card = GeneratedCard(
            card_id="gen_auth",
            operator="directional",
            claim=(
                "Diffusion-trained agent-tool authorization policies require a "
                "path metric embedding of denoising trajectories; endpoint-only "
                "embeddings cannot separate order-sensitive bypass traces from "
                "authorized tool-call traces."
            ),
            mechanism="path order is preserved along the sampling path",
            prediction="path-metric search cuts bypass-candidate count by 2x",
            falsifier="pair_not_already_combined",
            pair=("metric embedding", "quasipolynomial time algorithm"),
            pair_nodes=("a", "b"),
            alienness=0.4,
            model="test",
            artifact_digest="d",
            artifact_uri="file:///dev/null",
            replay_mode="replay",
        )
        req = refine_requirement(
            WorldRequirement(object_type=""),
            card.claim,
            card.prediction,
        )
        self.assertIsNotNone(req)
        self.assertEqual(req.object_type, "trace")
        with tempfile.TemporaryDirectory() as tmp:
            fixture = construct_world(
                None,
                card,
                {"object_type": ""},
                dest=Path(tmp) / "world",
            )
            payload = load_world_payload(fixture.root)
            self.assertEqual(payload.get("schema"), "labeled_traces")
            self.assertGreaterEqual(len(payload.get("traces") or []), 2)
            self.assertEqual(execute_world(payload), [])

    def test_incomplete_previous_can_be_reconstructed(self) -> None:
        card = _card()
        diagnosis = SimpleNamespace(
            experiment="On a planar-graph shortest-path world both arms call measure.",
            treatment_arm="path_order true",
            control_arm="path_order false",
            mechanism_flag="path_order",
            alternative="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "world"
            fixture = construct_world(
                None,
                card,
                {"object_type": ""},
                dest=dest,
                diagnosis=diagnosis,
                previous={"acquisition": "incomplete", "object_type": ""},
            )
            payload = load_world_payload(fixture.root)
            self.assertNotEqual(payload.get("acquisition"), "incomplete")
            self.assertEqual(execute_world(payload), [])

    def test_infer_demo_claim_is_traces_not_empty(self) -> None:
        req = infer_requirement(
            "combining diffusion models with agent-tool authorization security",
            "endpoint-only embeddings cannot separate bypass traces from "
            "authorized tool-call traces along denoising trajectories",
        )
        self.assertIsNotNone(req)
        self.assertEqual(req.object_type, "trace")


if __name__ == "__main__":
    unittest.main()
