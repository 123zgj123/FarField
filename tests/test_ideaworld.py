"""Idea-world analysis is diagnostic and never replaces a freeze."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from farfield.extras.genworld import construct_world, execute_world, load_world_payload
from farfield.extras.ideaworld import analyze_idea_world, deterministic_idea_world
from farfield.extras.world import WorldRequirement
from tests.test_genworld import _card


class IdeaWorldTests(unittest.TestCase):
    def test_deterministic_analysis_stays_on_the_topic_object(self) -> None:
        card = _card()
        topic = "code world models of executable program state as the world for an agent harness"
        world = deterministic_idea_world(card, topic)
        self.assertTrue(world.objects)
        self.assertTrue(any("code" in item or "program" in item for item in world.objects))
        self.assertIn("this claim object", world.iterate)
        self.assertEqual(world.source, "deterministic")
        self.assertTrue(world.prompt_lines())
        self.assertIn("must store fields", world.construct_notes())

    def test_a_missing_client_does_not_block(self) -> None:
        world = analyze_idea_world(None, _card(), "agent harness traces")
        self.assertEqual(world.source, "deterministic")
        self.assertIn("cannot corroborate", world.to_dict()["honesty"])

    def test_construct_stub_stores_named_quantities(self) -> None:
        card = _card()
        analysis = SimpleNamespace(
            objects=("self-edit proposals", "hidden-seed rewards"),
            missing=("tampering labels",),
            construct_notes=lambda: "Must store countable fields for: self-edit proposals.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            fixture = construct_world(
                None,
                card,
                WorldRequirement(object_type="formula", environment=("formula",)),
                dest=Path(tmp) / "world",
                idea_world=analysis,
            )
            payload = load_world_payload(fixture.root)
            self.assertEqual(execute_world(payload), [])
            quantities = [str(item) for item in (payload.get("quantities") or [])]
            self.assertTrue(
                any("self-edit" in item or "hidden-seed" in item for item in quantities)
            )


if __name__ == "__main__":
    unittest.main()
