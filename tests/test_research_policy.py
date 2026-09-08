"""Research-time evolution records proposals without granting new authority."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from farfield.extras.openworld import OpenWorld
from farfield.extras.openworld.evolve import Pathology, propose_patch
from farfield.extras.openworld.state import ScientificState


class ResearchTimeEvolutionTests(unittest.TestCase):
    def test_repeated_failure_only_proposes_and_preserves_active_harness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = OpenWorld(
                workspace=Path(tmp),
                state=ScientificState(goal="compare validator histories", world_id="W0"),
            )
            for index in range(3):
                env.note_harness_failure(
                    where="question_observation_executor",
                    why="cannot_align_versioned_records",
                    example=f"trajectory-{index}",
                )

            result = env.evolve()

            self.assertEqual(result["status"], "proposed")
            self.assertFalse(result["unlocked"])
            self.assertEqual(
                result["reason"],
                "research_policy_requires_completed_mission_and_heldout_receipt",
            )
            self.assertEqual(env.harness.version_id, "H0")
            self.assertEqual(env.harness.artifacts, [])
            self.assertNotIn("compare_validator_snapshots", env.capabilities.names())
            self.assertEqual(env.state.world_id, "W0")
            self.assertEqual(
                [event.event_type for event in env.log.events],
                ["HarnessPatchProposed"],
            )
            self.assertEqual(env.archives.harness[-1]["status"], "proposed")
            self.assertEqual(
                env.archives.harness[-1]["patch"]["artifact"]["name"],
                "compare_validator_snapshots",
            )
            env.persist()
            restored = OpenWorld.load(Path(tmp))
            self.assertEqual(restored.harness.version_id, "H0")
            self.assertEqual(restored.archives.harness[-1]["status"], "proposed")
            self.assertEqual(restored.log.events[-1].event_type, "HarnessPatchProposed")

    def test_supplied_passing_cases_cannot_install_during_research(self) -> None:
        env = OpenWorld(workspace=None, state=ScientificState(goal="ongoing research"))
        candidate = propose_patch(
            Pathology("question_observation_executor", "cannot_align_versioned_records", 3)
        )
        records = [{"validator_version": "v1"}, {"validator_version": "v2"}]

        result = env.evolve(
            candidate,
            failing=[{"records": records}],
            nearby=[{"records": records}],
            sealed=[{"records": records}],
            regression=[{"name": "prior survey", "passed": True}],
            previous=[{"name": "schema", "passed": True}],
            cost={"tokens": 1, "token_cap": 10},
        )

        self.assertEqual(result["status"], "proposed")
        self.assertFalse(result["unlocked"])
        self.assertEqual(env.harness.version_id, "H0")
        self.assertEqual(env.capabilities.names(), ())
        self.assertFalse(result["harness_general"])
        self.assertEqual(
            [event.event_type for event in env.log.events],
            ["HarnessPatchProposed"],
        )


if __name__ == "__main__":
    unittest.main()
