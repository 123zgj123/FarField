"""Deterministic fixtures for the verified open-world environment."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from farfield.extras.mission import _run_entry
from farfield.extras.openworld import (
    OpenWorld,
    after_card,
    dry_run_openworld,
)
from farfield.extras.openworld.actions import (
    ACQUIRE,
    EVOLVE_HARNESS,
    FORK_LINEAGE,
    OBSERVE,
    PROBE,
    THEORIZE,
    ActionInstance,
    SPECS,
    eligible_actions,
    types_of,
)
from farfield.extras.openworld.capability import compare_validator_snapshots
from farfield.extras.openworld.env import HarnessArtifact, HarnessVersion, project_harness
from farfield.extras.openworld.evolve import (
    HARNESS_FAILURE,
    SCIENTIFIC_FAILURE,
    classify_failure,
    evaluate_patch,
    propose_patch,
    Pathology,
)
from farfield.extras.openworld.kernel import (
    TRUSTED_KERNEL,
    KernelViolation,
    assert_patch_files,
    can_promote_to_world,
    make_run_context,
    preserve_evidence_provenance,
    social_consensus_is_not_evidence,
)
from farfield.extras.openworld.memory import ArchiveStore, project_archive
from farfield.extras.openworld.progress import fork_trajectories, maybe_stagnation
from farfield.extras.openworld.scheduler import schedule
from farfield.extras.openworld.state import ScientificState


def _question_state(**extra) -> ScientificState:
    state = ScientificState(goal="validator versions", world_id="W0")
    state.world_versions.append({"id": "W0", "world_id": "W0"})
    state.add_question(
        question_id="Q13",
        text="does validator-version history change acceptance?",
        required_capability="compare_validator_snapshots",
        lineage_id="line-q13",
        epistemic="GENERATED",
    )
    state.note_missing("compare_validator_snapshots")
    for key, value in extra.items():
        setattr(state, key, value)
    return state


class AffordanceTests(unittest.TestCase):
    def test_case_a_same_state_has_multiple_legal_actions(self) -> None:
        state = _question_state()
        types = set(types_of(eligible_actions(state)))
        self.assertIn(ACQUIRE, types)
        self.assertIn(THEORIZE, types)
        self.assertNotIn(PROBE, types)
        self.assertGreaterEqual(len(types), 3)

    def test_case_b_missing_observable_schedules_acquire_not_probe(self) -> None:
        state = _question_state()
        actions = eligible_actions(state)
        chosen = schedule(state, actions)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.action.action_type, ACQUIRE)
        self.assertNotIn(PROBE, types_of(actions))


class PathologyTests(unittest.TestCase):
    def test_case_c_repeated_executor_failure_is_harness_not_science(self) -> None:
        self.assertEqual(
            classify_failure(
                {
                    "where": "question_observation_executor",
                    "why": "cannot_align_versioned_records",
                }
            ),
            HARNESS_FAILURE,
        )
        self.assertEqual(
            classify_failure({"verdict": "weakens", "hypothesis_false": True}),
            SCIENTIFIC_FAILURE,
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = OpenWorld(workspace=Path(tmp), state=_question_state())
            for _ in range(3):
                env.execute(
                    ActionInstance(
                        SPECS[OBSERVE],
                        target="Q13",
                        extra={
                            "question": {
                                "id": "Q13",
                                "required_capability": "compare_validator_snapshots",
                            }
                        },
                    )
                )
            self.assertTrue(
                any(row.count >= 3 for row in env.pathologies.values())
            )
            types = types_of(env.eligible())
            self.assertIn(EVOLVE_HARNESS, types)
            self.assertFalse(
                any(row.get("epistemic") == "WORLD" and row.get("verdict") == "weakens" for row in env.state.evidence_records)
            )


class CreditGateTests(unittest.TestCase):
    def test_case_d_regression_rejects_patch(self) -> None:
        patch = propose_patch(
            Pathology(
                where="question_observation_executor",
                why="cannot_align_versioned_records",
                count=3,
            )
        )
        report = evaluate_patch(
            patch,
            failing=[
                {
                    "records": [
                        {"validator_version": "v1"},
                        {"validator_version": "v2"},
                    ]
                }
            ],
            nearby=[{"records": [{"validator_version": "a"}, {"validator_version": "b"}]}],
            regression=[{"name": "old_survey", "passed": False}],
            previous=[{"name": "schema", "passed": True}],
            cost={"tokens": 1, "token_cap": 10},
        )
        self.assertTrue(report.failing_replay)
        self.assertFalse(report.regression_ok)
        self.assertFalse(report.admit)

    def test_case_e_sealed_no_gain_is_not_general(self) -> None:
        patch = propose_patch(
            Pathology(
                where="question_observation_executor",
                why="cannot_align_versioned_records",
                count=3,
            )
        )
        report = evaluate_patch(
            patch,
            failing=[
                {
                    "records": [
                        {"validator_version": "v1"},
                        {"validator_version": "v2"},
                    ]
                }
            ],
            nearby=[{"records": [{"validator_version": "a"}, {"validator_version": "b"}]}],
            regression=[{"name": "old_survey", "passed": True}],
            previous=[{"name": "schema", "passed": True}],
            sealed=[{"records": [{"validator_version": "only-one"}], "aligned": True}],
            cost={"tokens": 1, "token_cap": 10},
        )
        self.assertTrue(report.failing_replay)
        self.assertFalse(report.sealed_gain)
        self.assertFalse(report.general)
        self.assertFalse(report.admit)


class CapabilityTests(unittest.TestCase):
    def test_case_f_validated_skill_unlocks_observe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = OpenWorld(workspace=Path(tmp), state=_question_state())
            for _ in range(3):
                env.note_harness_failure(
                    where="question_observation_executor",
                    why="cannot_align_versioned_records",
                )
            result = env.evolve()
            self.assertEqual(result["status"], "admitted")
            self.assertIn("compare_validator_snapshots", env.capabilities.names())
            types = types_of(env.eligible())
            self.assertIn(OBSERVE, types)


class ProvenanceTests(unittest.TestCase):
    def test_case_g_harness_change_does_not_rewrite_old_evidence(self) -> None:
        old = {
            "evidence_id": "E-old",
            "world_id": "W0",
            "harness_version": "H0",
            "epistemic": "WORLD",
        }
        kept = preserve_evidence_provenance(old, new_harness="H1")
        self.assertEqual(kept["harness_version"], "H0")
        self.assertEqual(kept["evidence_id"], "E-old")
        ctx0 = make_run_context(world_id="W0", evidence_id="E-old", harness_version="H0")
        ctx1 = make_run_context(world_id="W0", evidence_id="E-new", harness_version="H1")
        self.assertNotEqual(ctx0.run_context_id, ctx1.run_context_id)

    def test_case_h_scientific_acquire_does_not_change_harness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = OpenWorld(workspace=Path(tmp), state=_question_state())
            env.state.world_id = "W0"
            before = env.harness.version_id
            result = env.execute(
                ActionInstance(SPECS[ACQUIRE], target="Q13", extra={"gap": "world"})
            )
            self.assertEqual(env.harness.version_id, before)
            self.assertEqual(result["harness_version"], before)
            self.assertEqual(env.state.world_id, "W0")

    def test_case_i_harness_admit_does_not_change_world(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = OpenWorld(workspace=Path(tmp), state=_question_state())
            for _ in range(3):
                env.note_harness_failure(
                    where="question_observation_executor",
                    why="cannot_align_versioned_records",
                )
            result = env.evolve()
            self.assertEqual(result["status"], "admitted")
            self.assertEqual(env.state.world_id, "W0")
            self.assertTrue(result["world_unchanged"])
            self.assertEqual(env.harness.version_id, "H1")


class EpistemicTests(unittest.TestCase):
    def test_case_j_generated_narratives_never_become_world(self) -> None:
        row = {
            "epistemic": "GENERATED",
            "sibling_agreement": 6,
            "citation_count": 12,
            "harness_repetition": 20,
            "archive_discussion": True,
            "lineage_references": 4,
            "claim": "the same hypothesis again",
        }
        social = social_consensus_is_not_evidence(row)
        self.assertTrue(social)
        ok, gaps = can_promote_to_world(row)
        self.assertFalse(ok)
        self.assertTrue(any("GENERATED" in item for item in gaps))


class ProjectionTests(unittest.TestCase):
    def test_case_k_harness_history_is_projected(self) -> None:
        harness = HarnessVersion(version_id="H3")
        for index in range(12):
            harness.artifacts.append(
                HarnessArtifact(
                    name=f"acquire-recipe-{index}",
                    module="acquire_executor",
                    scope="action-kind",
                    activation="ACQUIRE",
                    body="x" * 80,
                )
            )
        harness.artifacts.append(
            HarnessArtifact(
                name="observe-align",
                module="question_executor",
                scope="action-kind",
                activation="OBSERVE",
                body="align versions",
            )
        )
        slice_ = project_harness(harness, action_type="OBSERVE")
        names = [item.name for item in slice_]
        self.assertIn("observe-align", names)
        self.assertFalse(any(name.startswith("acquire-recipe") for name in names))
        store = ArchiveStore()
        for index in range(20):
            store.working.append({"kind": "acquire", "n": index})
            store.harness.append({"name": f"acquire-recipe-{index}", "module": "acquire_executor"})
        store.harness.append({"name": "observe-align", "module": "question_executor"})
        projected = project_archive(store, action_type="OBSERVE")
        self.assertTrue(projected["harness"])
        self.assertTrue(all("question" in str(row.get("module") or row.get("name")).lower() for row in projected["harness"]))
        self.assertLess(len(projected["working"]) + len(projected["harness"]), 40)


class StagnationTests(unittest.TestCase):
    def test_case_l_stagnation_forks_not_only_far_jump(self) -> None:
        state = _question_state()
        state.ticks = 8
        state.last_progress_tick = 0
        event = maybe_stagnation(state)
        self.assertIsNotNone(event)
        self.assertEqual(event.forbid, "far_jump_only")
        self.assertIn("infrastructure", event.lanes)
        self.assertIn("understanding", event.lanes)
        types = types_of(eligible_actions(state, stagnation=True))
        self.assertIn(FORK_LINEAGE, types)
        branches = fork_trajectories(state, harness_version="H0", n=5)
        next_actions = {row.next_action for row in branches}
        self.assertNotEqual(next_actions, {"SURVEY"})
        self.assertTrue({"EVOLVE_HARNESS", "THEORIZE"} & next_actions)


class KernelBoundaryTests(unittest.TestCase):
    def test_trusted_kernel_cannot_be_patched(self) -> None:
        self.assertIn("evidence_promotion", TRUSTED_KERNEL)
        with self.assertRaises(KernelViolation):
            assert_patch_files(["src/farfield/extras/evidence.py"])
        with self.assertRaises(KernelViolation):
            assert_patch_files(["src/farfield/extras/mission.py"])


class MissionHookTests(unittest.TestCase):
    def test_after_card_does_not_force_a_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = OpenWorld(workspace=Path(tmp), state=_question_state())
            env.persist()
            events = after_card(Path(tmp), {"idea_kind": "question", "claim": "q", "card_id": "c1"})
            self.assertEqual(events[0]["stage"], "openworld_tick")
            self.assertFalse(events[0]["executed"])
            self.assertIn(ACQUIRE, events[0]["eligible_types"])
            self.assertNotIn(PROBE, events[0]["eligible_types"])

    def test_mission_hook_is_not_kind_if_else(self) -> None:
        source = inspect.getsource(_run_entry)
        self.assertIn("after_card", source)
        self.assertNotIn('if idea_kind == "acquire": skip', source)


class DryRunTests(unittest.TestCase):
    def test_openworld_chain_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events = dry_run_openworld(Path(tmp))
            summary = events[-1]
            self.assertEqual(summary["stage"], "dry_run_summary")
            self.assertEqual(summary["world_id"], "W0")
            self.assertEqual(summary["harness_version"], "H1")
            self.assertIn("compare_validator_snapshots", summary["capabilities"])
            self.assertNotIn("Q13", summary["resolved"], "generated comparison cannot settle acceptance hypothesis")
            observed = [row for row in events if row.get("stage") == "observe_after_h1"]
            self.assertTrue(observed)
            evidence = observed[0]["evidence"]
            self.assertEqual(evidence["role"], "QuestionEvidence")
            self.assertEqual(evidence["harness_version"], "H1")
            self.assertEqual(evidence["world_id"], "W0")
            self.assertEqual(evidence["causal_scope"], "none")
            theory = [row for row in events if row.get("stage") == "theory_eligible"]
            self.assertTrue(theory)
            env = OpenWorld.load(Path(tmp))
            types = types_of(env.eligible())
            self.assertTrue({THEORIZE, OBSERVE} & set(types) or env.state.theories)


class SnapshotHelperTests(unittest.TestCase):
    def test_compare_validator_snapshots_aligns(self) -> None:
        result = compare_validator_snapshots(
            [{"validator_version": "v1"}, {"validator_version": "v2"}]
        )
        self.assertTrue(result["aligned"])
        self.assertEqual(result["contrast"], "v1 vs v2")


if __name__ == "__main__":
    unittest.main()
