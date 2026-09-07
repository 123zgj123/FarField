"""Long-horizon OpenWorld production path: control, events, scenarios A–C."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from farfield.extras.openworld import OpenWorld
from farfield.extras.openworld.actions import (
    ACQUIRE,
    ACTION_COMPLETENESS,
    OBSERVE,
    THEORIZE,
    VERIFY,
    ActionInstance,
    SPECS,
    eligible_actions,
    types_of,
)
from farfield.extras.openworld.env import HarnessArtifact
from farfield.extras.openworld.executors import SNAPSHOT_FILE, resolve_world_root
from farfield.extras.openworld.kernel import can_promote_to_world
from farfield.extras.openworld.memory import project_archive
from farfield.extras.openworld.progress import TrajectoryBranch, merge_branches
from farfield.extras.openworld.reducer import rebuild_state, state_digest
from farfield.extras.openworld.runtime import bootstrap_research, run_research
from farfield.extras.openworld.scheduler import schedule
from farfield.extras.researchspace import ProblemDefinition, ResearchState
from farfield.extras.world import digest_files


def _write_w0(workspace: Path) -> Path:
    root = workspace / "worlds" / "W0"
    root.mkdir(parents=True, exist_ok=True)
    (root / "world.json").write_text(
        '{"n": 2, "traces": ["ep-v1", "ep-v2"]}\n', encoding="utf-8"
    )
    return root


class ControlOwnershipTests(unittest.TestCase):
    def test_production_next_action_is_openworld_scheduler(self) -> None:
        source = inspect.getsource(run_research)
        self.assertIn("env.tick()", source)
        self.assertNotIn("decide_next", source)
        self.assertNotIn("_evidence_pipeline", source)
        cli = Path("src/farfield/cli.py").read_text(encoding="utf-8")
        self.assertIn("legacy_pipeline", cli)
        self.assertIn("run_research", cli)
        self.assertLess(cli.find("run_research"), cli.find("if args.legacy_pipeline"))

    def test_problem_definition_is_not_control_state(self) -> None:
        self.assertIs(ProblemDefinition, ResearchState)


class EventSourcingTests(unittest.TestCase):
    def test_rebuild_digest_matches_stored_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            env = bootstrap_research("does validator history change acceptance?", workspace)
            env.run(8, stop_on_exhaust=False)
            rebuilt = rebuild_state(env.log)
            self.assertEqual(state_digest(env.state), state_digest(rebuilt))


class ScenarioATests(unittest.TestCase):
    def test_world_grows_then_observe_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write_w0(workspace)
            env = bootstrap_research(
                "is false acceptance higher on later validator versions?",
                workspace,
            )
            parent = resolve_world_root(workspace, "W0")
            self.assertIsNotNone(parent)
            parent_digest = digest_files(parent, ["world.json"])
            harness0 = env.harness.version_id
            events = env.run(12, stop_on_exhaust=False)
            types = [row.get("result", {}).get("action_type") for row in events if row.get("result")]
            self.assertIn(ACQUIRE, types)
            self.assertIn(OBSERVE, types)
            self.assertEqual(env.harness.version_id, harness0)
            self.assertNotEqual(env.state.world_id, "W0")
            child = resolve_world_root(workspace, env.state.world_id)
            self.assertIsNotNone(child)
            self.assertTrue((child / SNAPSHOT_FILE).is_file())
            self.assertTrue((child / "binaries" / "config.json").is_file())
            self.assertTrue((child / "validation" / "outputs.json").is_file())
            child_files = [
                p.relative_to(child).as_posix()
                for p in child.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            ]
            child_digest = digest_files(child, child_files)
            self.assertNotEqual(child_digest, parent_digest)
            self.assertTrue(env.state.resolved_questions)
            self.assertEqual(env.state.resolved_questions[0]["id"], "Q1")
            world_ev = [
                row
                for row in env.state.evidence_records
                if str(row.get("epistemic") or "") == "WORLD"
            ]
            self.assertTrue(world_ev)


class ScenarioBTests(unittest.TestCase):
    def test_harness_grows_world_stays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            env = OpenWorld(workspace=workspace)
            env.state.goal = "validator versions"
            env.state.world_id = "W0"
            env.state.add_question(
                question_id="Q2",
                text="can we align validator versions?",
                required_capability="compare_validator_snapshots",
                lineage_id="line-q2",
            )
            env.state.note_missing("compare_validator_snapshots")
            old_evidence = {
                "evidence_id": "E-old",
                "world_id": "W0",
                "harness_version": "H0",
                "epistemic": "WORLD",
                "attested": True,
                "role": "QuestionEvidence",
            }
            env.state.record_evidence(old_evidence)
            for _ in range(3):
                env.execute(
                    ActionInstance(
                        SPECS[OBSERVE],
                        target="Q2",
                        frontier_target_id="Q2",
                        extra={
                            "question": {
                                "id": "Q2",
                                "required_capability": "compare_validator_snapshots",
                                "text": env.state.goal,
                            }
                        },
                    )
                )
            self.assertIn("EVOLVE_HARNESS", types_of(env.eligible()))
            result = env.evolve()
            self.assertEqual(result["status"], "admitted")
            self.assertEqual(env.state.world_id, "W0")
            self.assertEqual(env.harness.version_id, "H1")
            observed = env.execute(
                ActionInstance(
                    SPECS[OBSERVE],
                    target="Q2",
                    frontier_target_id="Q2",
                    extra={
                        "question": {
                            "id": "Q2",
                            "required_capability": "compare_validator_snapshots",
                            "text": env.state.goal,
                        },
                        "records": [
                            {"validator_version": "v1", "ok": True},
                            {"validator_version": "v2", "ok": False},
                        ],
                    },
                )
            )
            self.assertEqual(observed["status"], "observed")
            self.assertEqual(env.state.world_id, "W0")
            kept = next(
                row for row in env.state.evidence_records if row.get("evidence_id") == "E-old"
            )
            self.assertEqual(kept["harness_version"], "H0")
            self.assertEqual(kept["world_id"], "W0")
            cap = env.capabilities.items["compare_validator_snapshots"]
            self.assertGreaterEqual(cap.activation_count, 1)
            self.assertIn(cap.trust_level, {"SHADOW", "TRUSTED", "validated", "VALIDATED"})


class ScenarioCTests(unittest.TestCase):
    def test_branch_merge_keeps_contradiction(self) -> None:
        state = OpenWorld(workspace=None).state
        state.goal = "T1"
        a = TrajectoryBranch(
            lane="A",
            harness_version="H0",
            next_action=OBSERVE,
            score=9.0,
            world_evidence=[
                {
                    "evidence_id": "E2",
                    "world_id": "W1",
                    "epistemic": "WORLD",
                    "attested": True,
                    "question_id": "Q1",
                    "theory_id": "T1",
                    "verdict": "supports",
                }
            ],
            generated=[{"epistemic": "GENERATED", "text": "narrate a win"}],
        )
        b = TrajectoryBranch(
            lane="B",
            harness_version="H0",
            next_action=OBSERVE,
            score=1.0,
            world_evidence=[
                {
                    "evidence_id": "E3",
                    "world_id": "W1",
                    "epistemic": "WORLD",
                    "attested": True,
                    "question_id": "Q1",
                    "theory_id": "T1",
                    "verdict": "weakens",
                }
            ],
        )
        result = merge_branches([a, b], state)
        ids = {row.get("evidence_id") for row in state.evidence_records}
        self.assertIn("E2", ids)
        self.assertIn("E3", ids)
        self.assertTrue(result["contradictions"] or state.contradictions)
        self.assertFalse(result["promoted_generated"])
        self.assertFalse(
            any(str(row.get("epistemic") or "") == "GENERATED" for row in result["merged"])
        )
        actions = eligible_actions(state)
        self.assertIn(VERIFY, types_of(actions))
        chosen = schedule(state, actions)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.action.action_type, VERIFY)


class CompletenessTests(unittest.TestCase):
    def test_core_actions_are_production_ready(self) -> None:
        for name in (OBSERVE, ACQUIRE, THEORIZE, VERIFY):
            row = ACTION_COMPLETENESS[name]
            self.assertTrue(all(row.values()), name)


class LongHorizonTests(unittest.TestCase):
    def _run(self, ticks: int, *, stop: bool = False) -> OpenWorld:
        workspace = Path(tempfile.mkdtemp())
        _write_w0(workspace)
        env = bootstrap_research("does validator-version history change acceptance?", workspace)
        env.run(ticks, stop_on_exhaust=stop)
        return env

    def test_20_50_100_ticks_converge(self) -> None:
        env20 = self._run(20)
        env50 = self._run(50)
        env100 = self._run(100)
        for env, n in ((env20, 20), (env50, 50), (env100, 100)):
            rebuilt = rebuild_state(env.log)
            self.assertEqual(state_digest(env.state), state_digest(rebuilt), n)
            self.assertGreaterEqual(len(env.state.resolved_questions), 1, n)
            self.assertLessEqual(len(env.state.open_questions), 2, n)
            self.assertGreaterEqual(
                len(
                    [
                        row
                        for row in env.state.evidence_records
                        if str(row.get("epistemic") or "") == "WORLD"
                    ]
                ),
                1,
                n,
            )
            leakage = [
                row
                for row in env.state.evidence_records
                if str(row.get("epistemic") or "") == "WORLD"
                and not row.get("attested")
            ]
            self.assertFalse(leakage, n)
            ok, _ = can_promote_to_world(
                {"epistemic": "GENERATED", "claim": "story", "sibling_agreement": 9}
            )
            self.assertFalse(ok)
            theories = env.state.theories
            if theories:
                self.assertTrue(
                    any(row.get("discriminating_predictions") for row in theories), n
                )
            self.assertLessEqual(len(env.harness.artifacts), 8, n)
            self.assertLessEqual(int((env.state.debt or {}).get("verification_debt") or 0), 3, n)
            scientific = [
                row
                for row in env.history
                if row.get("action_type")
                in {OBSERVE, ACQUIRE, THEORIZE, VERIFY, "PROBE", "ASK", "SURVEY"}
            ]
            drifted = [
                row
                for row in scientific
                if not row.get("frontier_target_id")
            ]
            self.assertFalse(drifted, n)
        self.assertGreaterEqual(len(env100.state.resolved_questions), 1)
        self.assertGreaterEqual(
            len(env100.state.resolved_questions),
            len(env20.state.resolved_questions),
        )
        archive = project_archive(
            env100.archives, action_type=OBSERVE, world_id=env100.state.world_id
        )
        self.assertTrue(archive["scientific"] or env100.state.archive_ids)
        stale = project_archive(env100.archives, action_type=OBSERVE, world_id="W-missing")
        for row in stale["scientific"]:
            self.assertNotEqual(str(row.get("world_id") or ""), "W1")

    def test_harness_gc_deprecates_useless_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = OpenWorld(workspace=Path(tmp))
            env.state.ticks = 20
            env.harness.artifacts.append(
                HarnessArtifact(
                    name="dead-skill",
                    module="skill_registry",
                    scope="task",
                    activation="REFLECT",
                    body="noise",
                    activations=3,
                    benefit=0.0,
                    context_cost=400,
                    success_delta=0.0,
                )
            )
            from farfield.extras.openworld.env import gc_harness

            child, deprecated = gc_harness(env.harness, tick=20)
            self.assertTrue(deprecated)
            self.assertTrue(all(item.status == "DEPRECATED" for item in deprecated))
            self.assertFalse(any(item.name == "dead-skill" for item in child.artifacts))


if __name__ == "__main__":
    unittest.main()
