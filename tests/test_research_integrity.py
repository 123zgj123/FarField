"""End-to-end research contracts: real bytes, honest provenance and outcomes."""

import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from farfield.cli import main
from farfield.extras.lifecycle import derive_world_version
from farfield.extras.openworld import OpenWorld
from farfield.extras.openworld.actions import ActionInstance, SPECS, OBSERVE, ACQUIRE, VERIFY, THEORIZE
from farfield.extras.openworld.executors import fixture_from_root
from farfield.extras.openworld.runtime import bootstrap_research, run_research
from farfield.extras.openworld.events import make_event
from farfield.extras.openworld.kernel import CandidateResult
from farfield.extras.world import load_fixture

ROOT = Path(__file__).resolve().parents[1]


class ResearchIntegrityTests(unittest.TestCase):
    def setUp(self):
        # Integrity tests are offline. Live literature parity is separately labelled.
        from farfield.extras.knowledge import Harvest
        self.harvest = patch('farfield.extras.openworld.workers.harvest_survey', return_value=Harvest())
        self.harvest.start()
        self.addCleanup(self.harvest.stop)

    def test_missing_recorded_digest_cannot_become_attested_by_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            (root / "iris.json").write_text('{"rows": []}')
            (root / "manifest.json").write_text(json.dumps({
                "id": "iris", "files": ["iris.json"], "source": "unit-test",
                "retrieved_at": "2026-09-07", "schema": "numeric_table",
            }))
            with self.assertRaisesRegex(ValueError, "recorded digest"):
                bootstrap_research("iris", Path(tmp) / "mission", world=load_fixture(root))

    def test_verification_recomputes_the_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = bootstrap_research("iris species", Path(tmp), world="fisher-iris")
            action = ActionInstance(SPECS[OBSERVE], target="Q1", extra={"question": env.state.open_questions[0]})
            env.execute(action)
            env.state.evidence_records[0]["result"]["n_rows"] = 999
            result = env.execute(ActionInstance(SPECS[VERIFY], target="Q1"))
            self.assertFalse(result["replay_ok"])

    def test_generic_theory_does_not_invent_validator_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = bootstrap_research("iris species", Path(tmp))
            result = env.execute(ActionInstance(SPECS[THEORIZE], target="Q1"))
            self.assertEqual(result["status"], "blocked")
            self.assertFalse(env.state.theories)

    def test_evidence_events_cannot_bypass_world_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = bootstrap_research("iris", Path(tmp))
            forged = {"role": "QuestionEvidence", "epistemic": "WORLD", "attested": True,
                      "evidence_id": "forged", "world_id": "W0", "question_id": "Q1"}
            env.executors[OBSERVE] = lambda *_: CandidateResult(
                status="observed", events=[make_event("EvidenceAdded", forged),
                make_event("QuestionResolved", {"id": "Q1", "evidence_id": "forged"})])
            env.execute(ActionInstance(SPECS[OBSERVE], target="Q1"))
            self.assertFalse(env.state.resolved_questions)
            self.assertFalse([r for r in env.state.evidence_records if r.get("epistemic") == "WORLD"])

    def test_matching_world_digest_does_not_erase_generated_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = bootstrap_research('iris', Path(tmp), world='fisher-iris')
            for tag in ('kind', 'probe_kind', 'world_role', 'provenance', 'world_provenance'):
                forged = {'role': 'ProbeEvidence', 'epistemic': 'WORLD', 'attested': True,
                    'world_id': env.state.world_id, 'world_digest': env.state.world_versions[0]['digest'],
                    'evidence_id': 'forged-' + tag, tag: 'GENERATED'}
                env.executors[OBSERVE] = lambda *_, forged=forged: CandidateResult(status='observed', evidence=forged)
                env.execute(ActionInstance(SPECS[OBSERVE], target='Q1'))
            self.assertFalse([r for r in env.state.evidence_records if r.get('epistemic') == 'WORLD'])

    def test_cli_iris_binds_exact_bytes_and_keeps_prediction_question_open(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
            rc = main(["research", tmp, "--topic", "does sepal geometry predict iris species?",
                       "--world", "fisher-iris", "--horizon", "4"])
            self.assertEqual(rc, 0, out.getvalue())
            states = list(Path(tmp).rglob("SCIENTIFIC_STATE.json"))
            self.assertEqual(len(states), 1)
            workspace = states[0].parent
            iris = workspace / "worlds/fisher-iris/iris.json"
            self.assertTrue(iris.is_file(), "explicit fixture was replaced with a fabricated world")
            self.assertEqual(iris.read_bytes(), (ROOT / "worlds/fisher-iris/iris.json").read_bytes())
            state = json.loads(states[0].read_text())
            self.assertFalse(state["resolved_questions"])
            self.assertTrue(state["open_questions"] + state["blocked_questions"])
            observations = [r for r in state["evidence_records"] if r.get("role") == "QuestionEvidence"]
            self.assertTrue(observations)
            self.assertEqual(observations[0]["result"]["n_rows"], 150)
            self.assertTrue(observations[0]["cannot_corroborate"])
            done = json.loads(out.getvalue().splitlines()[-1])
            self.assertFalse(done["research_complete"])
            self.assertEqual(done["status"], "incomplete")
            self.assertTrue((workspace / "RESEARCH_STATUS.md").is_file())

    def test_unknown_world_fails_without_creating_a_substitute(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
            rc = main(["research", tmp, "--topic", "iris", "--world", "missing-world", "--horizon", "1"])
            self.assertEqual(rc, 2, out.getvalue())
            self.assertFalse(list(Path(tmp).rglob("world.json")))

    def test_auto_does_not_invent_observations_or_resolve_the_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = list(run_research("does sepal geometry predict iris species?", workspace=Path(tmp), horizon=6))
            state = OpenWorld.load(Path(tmp)).state
            self.assertFalse(state.resolved_questions)
            self.assertFalse([r for r in state.evidence_records if r.get("epistemic") == "WORLD"])
            self.assertNotIn("validator_snapshots", json.dumps(state.open_questions))
            self.assertFalse(events[-1]["research_complete"])

    def test_manifestless_world_is_generated_regardless_of_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "world.json").write_text('{"records": [{"validator_version":"v1"}, {"validator_version":"v2"}]}')
            parent = fixture_from_root(root, "real-sounding-name")
            self.assertEqual(parent.provenance, "generated")
            child = derive_world_version(parent, root / "child", new_id="W2", acquisition_event="test")
            self.assertEqual(child.provenance, "generated")
            self.assertEqual(child.role, "generated")
            self.assertEqual(fixture_from_root(child.root, "W2").provenance, "generated")

    def test_generated_snapshots_do_not_become_world_after_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "worlds/W0"
            root.mkdir(parents=True)
            (root / "world.json").write_text('{"records": [{"validator_version":"v1"}, {"validator_version":"v2"}]}')
            env = bootstrap_research("compare validator snapshots", workspace)
            env.execute(ActionInstance(SPECS[ACQUIRE], target="Q1", extra={"gap": "validator_snapshots"}))
            env.execute(ActionInstance(SPECS[OBSERVE], target="Q1", extra={"observable": "validator_snapshots"}))
            self.assertFalse([r for r in env.state.evidence_records if r.get("epistemic") == "WORLD"])
            self.assertFalse(env.archives.scientific if hasattr(env.archives, "scientific") else [])

    def test_fixture_object_is_copied_and_digest_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = bootstrap_research("iris", Path(tmp), world=load_fixture(ROOT / "worlds/fisher-iris"))
            root = Path(tmp) / "worlds/fisher-iris"
            self.assertTrue((root / "iris.json").is_file())
            (root / "iris.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                fixture_from_root(root, env.state.world_id)


if __name__ == "__main__":
    unittest.main()
