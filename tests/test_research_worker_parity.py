"""Recorded-model replay parity, not live mission or literature evidence.

Both paths execute real sandbox subprocesses against identical local frozen
bytes. Model responses and literature are explicitly offline test records.
"""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import test_research_replication as fixtures
from test_openworld_workers import ReplayClient
from farfield.extras import chain
from farfield.extras import mission
from farfield.extras.openworld import workers
from farfield.extras.brief import compile_brief
from farfield.extras.evidence import sha256_text
from farfield.extras.generate import GenerationRefused
from farfield.extras.hostexp import execute_protocol
from farfield.extras.llm import LLMUnavailable
from farfield.extras.mission import _evidence_pipeline
from farfield.extras.openworld import OpenWorld
from farfield.extras.openworld.workers import LegacyResearchWorkers, _card
from farfield.extras.packet import render_protocol
from farfield.extras.workspace import write_idea
from farfield.models import BlockedRecord


class RecordedModelReplay:
    """Purpose-addressed recorded responses; cannot make network requests."""

    def __init__(self, diagnosis=None, failure=None, source=fixtures.SOURCE):
        self.diagnosis = copy.deepcopy(diagnosis or fixtures.DIAGNOSIS)
        self.failure = failure
        self.source = source
        self.calls = []

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.calls.append(purpose)
        if purpose.startswith("diagnosis:"):
            if self.failure is not None:
                raise self.failure
            answer = self.diagnosis
        elif purpose.startswith("research_probe:"):
            answer = {"measure": "mean_column_spread", "source": self.source}
        elif purpose == "idea_world_analysis":
            answer = {}  # Mature worker's deterministic idea-world fallback.
        else:
            raise AssertionError(f"Unrecorded model purpose: {purpose}")
        return ReplayClient([answer]).complete(prompt, purpose=purpose, system=system)


class ResearchWorkerParityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReplicationTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.workspace = self.fixture.workspace
        self.card = _card(self.fixture.state.theories[0]["card"])

    def legacy(self, client):
        bundle = {}
        events = list(_evidence_pipeline(
            client, None, self.card, self.fixture.state.goal,
            workspace=self.workspace / "legacy", world=self.fixture.world,
            lock_world=True, writeup=False, world_sim=False,
            experiment_rounds=0, bundle=bundle,
        ))
        return bundle, events

    def openworld(self, client):
        worker = LegacyResearchWorkers(client=client, workspace=self.workspace)
        env = OpenWorld(workspace=self.workspace, state=copy.deepcopy(self.fixture.state))
        env.executors["PROBE"] = lambda env, action: worker.probe(action, env.state.view())
        result = env.execute(self.fixture.action("PROBE"))
        return env, result

    def test_positive_full_mission_evidence_path_matches_openworld_execution(self):
        # These observers wrap actual executors; neither fabricates their results.
        # Assertions happen BEFORE the real subprocess, not just post hoc.
        real_legacy_run = mission._run_probe_attempt
        real_worker_run = workers.run_probe
        registration_checks = []

        def legacy_run(spec, workspace, card_id, attempt, **kwargs):
            path = Path(workspace) / "candidates" / card_id / "chain.jsonl"
            rows = chain.read_events(path)
            registered = [row for row in rows if row["kind"] == chain.REGISTER_PROTOCOL]
            self.assertEqual(registered[-1]["payload"]["experiment_digest"], sha256_text(spec.source))
            self.assertEqual(registered[-1]["payload"]["world_digest"], self.fixture.world.digest)
            self.assertNotIn(chain.EXECUTE_PROTOCOL, [row["kind"] for row in rows])
            registration_checks.append("legacy")
            return real_legacy_run(spec, workspace, card_id, attempt, **kwargs)

        def worker_run(spec, dest, **kwargs):
            if Path(dest).name == "execution":
                folder = Path(dest).parent.parent
                protocol = json.loads((folder / "protocol.json").read_text())
                self.assertEqual(protocol["experiment_digest"], sha256_text(spec.source))
                self.assertFalse((Path(dest) / "metrics.json").exists())
                self.assertIsNone((protocol.get("cheap_probe") or {}).get("treatment"))
                self.assertEqual(protocol["experiment_proposal"], self.fixture.proposal)
                registered = chain.read_events(folder / "chain.jsonl")
                self.assertIn(chain.REGISTER_PROTOCOL, [row["kind"] for row in registered])
                registration_checks.append("openworld")
            return real_worker_run(spec, dest, **kwargs)

        with patch.object(mission, "_run_probe_attempt", side_effect=legacy_run):
            legacy, events = self.legacy(RecordedModelReplay())
        with patch.object(workers, "run_probe", side_effect=worker_run):
            env, result = self.openworld(RecordedModelReplay())
        self.assertEqual(registration_checks, ["legacy", "openworld"])
        self.assertIsNotNone(legacy.get("probe_payload"), events)
        old = legacy["probe_payload"]
        new = result["evidence"]
        self.assertEqual(result["status"], "probe_positive", result)
        for key in ("experiment_digest", "world_digest", "data_digest", "evidence_id", "treatment", "control"):
            self.assertEqual(old[key], new[key], key)
        self.assertEqual(old["verdict"], new["outcome"])
        self.assertEqual((new["treatment"], new["control"]), (9.0, 109.0))
        self.assertEqual(new["experiment_digest"], sha256_text(fixtures.SOURCE.strip()))
        self.assertEqual(new["world_digest"], self.fixture.world.digest)
        self.assertEqual(old["source"], (Path(new["protocol_dir"]) / "experiment.py").read_text())
        self.assertEqual(legacy["diagnosis"].to_dict(), new["diagnosis"])
        self.assertEqual(old["kind"], "WORLD")
        self.assertTrue(old["world_consumed"])
        self.assertTrue(new["world_consumed"])
        self.assertEqual(old["scoped_verdict"], new["scoped_verdict"])
        self.assertEqual(new["causal_scope"], "within_tested_conditions")
        self.assertNotIn(self.card.card_id, env.state.verified_research_state)

        # writeup=False deliberately leaves packet publication to the mature
        # deterministic writer. This is not a second source generation.
        brief = compile_brief(self.card, [], diagnosis=legacy["diagnosis"], topic=env.state.goal)
        protocol = render_protocol(env.state.goal, self.card, brief, [], probe=old,
                                   diagnosis=legacy["diagnosis"], world=self.fixture.world)
        old_folder = write_idea(self.workspace / "legacy", card_id=self.card.card_id,
                               brief=brief.to_dict(), works=[], probe=old, protocol=protocol.payload)
        new_folder = Path(new["protocol_dir"])
        registered_bytes = [(folder / name, (folder / name).read_bytes())
                            for folder in (old_folder, new_folder)
                            for name in ("protocol.json", "experiment.py")]
        old_host = execute_protocol(old_folder, catalog_root=self.workspace, prefer_parent=False)
        verifier = LegacyResearchWorkers(workspace=self.workspace)
        env.executors["VERIFY"] = lambda env, action: verifier.verify(action, env.state.view())
        verified = env.execute(self.fixture.action("VERIFY", new["evidence_id"], replication_world_id=""))
        self.assertEqual(verified["status"], "verified", verified)
        reproduced = next(row for row in env.state.evidence_records if row["evidence_id"] == new["evidence_id"])
        new_host = reproduced["host_record"]
        for key in ("evidence_id", "experiment_digest", "world_digest", "data_digest", "treatment", "control", "verdict"):
            self.assertEqual(old_host[key], new_host[key], key)
        for path, before in registered_bytes:
            self.assertEqual(path.read_bytes(), before, str(path))
        self.assertFalse(reproduced["transfer_ok"])
        self.assertNotIn(self.card.card_id, env.state.verified_research_state)

    def test_scientifically_negative_is_not_execution_failure_in_either_path(self):
        diagnosis = {**fixtures.DIAGNOSIS, "expected_direction": "treatment_higher",
                     "alternative_direction": "treatment_lower"}
        legacy, events = self.legacy(RecordedModelReplay(diagnosis))
        env, result = self.openworld(RecordedModelReplay(diagnosis))
        self.assertIsNotNone(legacy.get("probe_payload"), events)
        self.assertEqual(legacy["probe_payload"]["verdict"], "weakens")
        self.assertEqual(result["status"], "probe_scientifically_negative")
        self.assertEqual(result["evidence"]["outcome"], "weakens")
        self.assertEqual(result["evidence"]["execution_status"], "ran")
        self.assertEqual(legacy["probe_payload"]["treatment"], result["evidence"]["treatment"])
        for key in ("experiment_digest", "world_digest", "data_digest", "evidence_id", "control"):
            self.assertEqual(legacy["probe_payload"][key], result["evidence"][key], key)
        self.assertFalse(result["evidence"]["mechanism_validated"])
        self.assertNotIn(self.card.card_id, env.state.verified_research_state)

    def test_network_unavailable_and_policy_refusal_never_become_negative_evidence(self):
        record = BlockedRecord(missing_capability="recorded_unavailable_backend",
                               attempted="recorded diagnosis", unlock_condition="offline replay failure")
        for failure, legacy_stage, status in (
            (LLMUnavailable(record), "blocked", "probe_execution_blocked"),
            (GenerationRefused(record), "diagnosis_refused", "probe_policy_refused"),
        ):
            with self.subTest(failure=type(failure).__name__):
                legacy, events = self.legacy(RecordedModelReplay(failure=failure))
                env, result = self.openworld(RecordedModelReplay(failure=failure))
                self.assertIn(legacy_stage, [event["stage"] for event in events])
                self.assertFalse(legacy.get("probe_payload"))
                self.assertEqual(result["status"], status, result)
                self.assertFalse(result.get("evidence"))
                self.assertFalse(env.state.evidence_records)

    def test_network_import_is_refused_by_both_real_source_compilers(self):
        source = "import socket\n" + fixtures.SOURCE
        legacy, events = self.legacy(RecordedModelReplay(source=source))
        env, result = self.openworld(RecordedModelReplay(source=source))
        self.assertIn("probe_refused", [event["stage"] for event in events])
        self.assertFalse(legacy.get("probe_payload"))
        self.assertEqual(result["status"], "probe_policy_refused", result)
        self.assertFalse(env.state.evidence_records)
        self.assertFalse(result.get("evidence"))


if __name__ == "__main__":
    unittest.main()
