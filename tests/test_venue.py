"""Declared venue lines join the value rubric. They cannot kill."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from farfield.extras.venue import (
    VenueError,
    normalize_venue,
    venue_value_lines,
    venue_writing_note,
)
from farfield.extras.mission import run_mission
from tests.test_mission import FakeFeed, SchemingClient, TOPIC


class VenueProfileTests(unittest.TestCase):
    def test_empty_venue_keeps_generation_prompts_identical(self) -> None:
        self.assertEqual(normalize_venue(""), "")
        self.assertEqual(normalize_venue(None), "")
        self.assertEqual(venue_value_lines(""), ())
        self.assertEqual(venue_writing_note(""), "")

    def test_unknown_venue_refuses_instead_of_silently_skipping(self) -> None:
        with self.assertRaises(VenueError) as caught:
            normalize_venue("not-a-conference")
        self.assertIn("known:", str(caught.exception))
        self.assertIn("cannot kill", str(caught.exception))

    def test_value_lines_are_review_questions_not_a_kill_sheet(self) -> None:
        lines = venue_value_lines("neurips")
        self.assertTrue(lines)
        joined = "\n".join(lines)
        self.assertIn("Declared venue neurips", joined)
        self.assertIn("cannot kill", joined)
        self.assertIn("cannot rename the object", joined)
        self.assertIn("ablation", joined.lower())
        self.assertNotIn("this card is killed", joined.lower())
        self.assertNotIn("score this idea", joined.lower())
        self.assertIn("jin-s13/ai-research-writing-skill", joined)

    def test_aliases_normalize_to_declared_ids(self) -> None:
        self.assertEqual(normalize_venue("nips"), "neurips")
        self.assertEqual(normalize_venue("NeurIPS 2026"), "neurips")
        self.assertEqual(normalize_venue("ICLR 2027"), "iclr")


class VenueMissionTests(unittest.TestCase):
    def test_unknown_venue_blocks_the_mission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events = list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=SchemingClient(),
                    venue="not-a-conference",
                    polish_rounds=0,
                    explore="fixed",
                    experiment_rounds=0,
                    idea_rounds=0,
                    state_store=Path(tmp) / "state.json",
                    policy_log=Path(tmp) / "log.json",
                    policy_file=Path(tmp) / "policy.json",
                )
            )
        blocked = next(e for e in events if e["stage"] == "blocked")
        self.assertEqual(blocked["missing_capability"], "venue_profile")
        self.assertFalse(any(e["stage"] == "card" for e in events))

    def test_declared_venue_reaches_the_value_rubric_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = SchemingClient()
            events = list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=client,
                    feed=FakeFeed(),
                    venue="neurips",
                    polish_rounds=0,
                    explore="fixed",
                    experiment_rounds=0,
                    idea_rounds=0,
                    state_store=Path(tmp) / "state.json",
                    policy_log=Path(tmp) / "log.json",
                    policy_file=Path(tmp) / "policy.json",
                )
            )
        corpus = next(e for e in events if e["stage"] == "corpus")
        self.assertEqual(corpus["venue"], "neurips")
        joined = "\n".join(client.prompts)
        self.assertIn("Declared venue neurips", joined)
        self.assertIn("cannot kill", joined)
        self.assertIn("These lines cannot kill a card", joined)
        self.assertNotIn("this card is killed", joined.lower())


if __name__ == "__main__":
    unittest.main()
