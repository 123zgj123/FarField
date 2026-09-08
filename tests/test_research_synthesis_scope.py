"""Author prompts and rendered notes retain the registered evidence ceiling."""

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.evidence import sha256_text
from farfield.extras.paperplan import PaperError, compile_paper, gather_facts
from test_paperplan import ScriptedClient, _author
import test_openworld_workers as worker_fixtures


class CapturingAuthor(ScriptedClient):
    def __init__(self, answer=None):
        super().__init__({"paper_author": answer or _author(hypotheses=[])})
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return super().complete(prompt, **kwargs)


def registration(root: Path):
    workspace = root / "research_workers" / "E1"
    folder = workspace / "candidates" / "H1"
    folder.mkdir(parents=True)
    source = "print('registered experiment')\n"
    protocol = {
        "claim": "A feedback mechanism reduces decision errors.", "card_id": "H1",
        "mechanism": "A feedback controller updates decisions.", "prediction": "Errors decrease.",
        "world": {"id": "exact-world", "schema": "numeric_table", "digest": "world-bytes",
                  "provenance": "published", "source": "https://example.invalid/frozen.csv"},
        "world_digest": "world-bytes", "data_digest": "world-bytes",
        "experiment_digest": sha256_text(source), "evidence_id": "E1",
    }
    text = json.dumps(protocol)
    (folder / "protocol.json").write_text(text)
    (folder / "experiment.py").write_text(source)
    probe = {
        "kind": "WORLD", "epistemic": "WORLD", "attested": True,
        "execution_status": "ran", "status": "ran", "verdict": "supports",
        "treatment": 0.1, "control": 0.4,
        "claim_consistent": False, "cannot_corroborate": True,
        "scope_gap": "registered_competing_mechanism_ablation",
        "causal_scope": "unestablished", "prohibited_inferences": ["causal", "mechanism"],
        "validation_gaps": ["ablation missing"], "replication_required": True,
        "world_id": "exact-world", "world_digest": "world-bytes", "data_digest": "world-bytes",
        "experiment_digest": protocol["experiment_digest"], "evidence_id": "E1",
        "protocol_digest": sha256_text(text),
    }
    (folder / "probe.json").write_text(json.dumps(probe))
    return workspace, folder, probe


class RegisteredScopeTests(unittest.TestCase):
    def test_nested_registration_world_and_scope_reach_author_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws, folder, probe = registration(Path(tmp))
            (folder / "PAPER_FACTS.json").write_text(json.dumps({
                "world": {"id": "fabricated-world"},
                "scientific_scope": {"cannot_corroborate": False, "claim_consistent": True},
            }))
            client = CapturingAuthor()
            compile_paper(ws, folder, "H1", client=client, rounds=0, lang="en", catalog_root=Path(tmp))
            facts = json.loads((folder / "paper" / "PAPER_FACTS.json").read_text())
            self.assertEqual(facts["world"]["id"], "exact-world")
            self.assertEqual(facts["world"]["digest"], "world-bytes")
            self.assertTrue(facts["scientific_scope"]["cannot_corroborate"])
            self.assertEqual(facts["scientific_scope"]["scope_gap"], probe["scope_gap"])
            self.assertEqual(facts["scientific_scope"]["prohibited_inferences"], ["causal", "mechanism"])
            self.assertIn('"cannot_corroborate": true', client.prompts[0])
            self.assertIn('"id": "exact-world"', client.prompts[0])
            self.assertNotIn("fabricated-world", client.prompts[0])

    def test_generated_probe_cannot_clear_scope_with_positive_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws, folder, probe = registration(Path(tmp))
            probe.update(epistemic="GENERATED", kind="GENERATED", cannot_corroborate=False,
                         claim_consistent=True, scope_gap="", validation_gaps=[])
            (folder / "probe.json").write_text(json.dumps(probe))
            scope = gather_facts(ws, folder, "H1", catalog_root=Path(tmp))["scientific_scope"]
            self.assertTrue(scope["cannot_corroborate"])
            self.assertFalse(scope["claim_consistent"])

    def test_registration_identity_mismatch_cannot_be_written_as_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws, folder, probe = registration(Path(tmp))
            probe["world_digest"] = "different-world"
            (folder / "probe.json").write_text(json.dumps(probe))
            with self.assertRaises(PaperError):
                gather_facts(ws, folder, "H1", catalog_root=Path(tmp))

    def test_author_cannot_turn_noncorroborating_measurement_into_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws, folder, _ = registration(Path(tmp))
            client = CapturingAuthor(_author(
                proposal_markdown="We confirm the causal mechanism from these results.",
                hypotheses=[{"id": "H1", "world": "exact-world", "evidence": "supports"}],
            ))
            with self.assertRaises(PaperError):
                compile_paper(ws, folder, "H1", client=client, rounds=0, catalog_root=Path(tmp))
            self.assertFalse((folder / "paper" / "RESEARCH_PROPOSAL.md").exists())

    def test_supported_association_still_cannot_be_written_as_causal_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws, folder, probe = registration(Path(tmp))
            probe.update(cannot_corroborate=False, claim_consistent=True, scope_gap="",
                         validation_gaps=[], scoped_verdict="supports_association")
            (folder / "probe.json").write_text(json.dumps(probe))
            client = CapturingAuthor(_author(
                proposal_markdown="We confirm the causal mechanism from these results.", hypotheses=[],
            ))
            with self.assertRaises(PaperError):
                compile_paper(ws, folder, "H1", client=client, rounds=0, catalog_root=Path(tmp))

    def test_missing_world_provenance_is_not_labelled_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws, folder, probe = registration(Path(tmp))
            protocol = json.loads((folder / "protocol.json").read_text())
            protocol["world"].pop("provenance")
            text = json.dumps(protocol)
            (folder / "protocol.json").write_text(text)
            probe["protocol_digest"] = sha256_text(text)
            (folder / "probe.json").write_text(json.dumps(probe))
            compile_paper(ws, folder, "H1", client=CapturingAuthor(), rounds=0, lang="en", catalog_root=Path(tmp))
            draft = (folder / "paper" / "RESEARCH_PROPOSAL.md").read_text()
            self.assertIn("provenance `unknown`", draft)
            self.assertNotIn("provenance `published`", draft)


class WorkerSynthesisScopeTests(unittest.TestCase):
    setUp = worker_fixtures.WorkerTests.setUp
    worker = worker_fixtures.WorkerTests.worker
    theory = worker_fixtures.WorkerTests.theory
    world = worker_fixtures.WorkerTests.world
    executed = worker_fixtures.WorkerTests.executed

    def test_optional_paper_keeps_same_scope_as_note_and_exact_world(self):
        evidence = self.executed()
        client = CapturingAuthor()
        worker = self.worker_class(client=client, workspace=self.workspace)
        result = worker.synthesize(worker_fixtures.action("SYNTHESIZE", target=evidence["evidence_id"], paper_draft=True),
                                   self.state.view())
        self.assertEqual(result.status, "synthesized", result.to_dict())
        folder = Path(evidence["protocol_dir"])
        facts = json.loads((folder / "paper" / "PAPER_FACTS.json").read_text())
        self.assertTrue(facts["scientific_scope"]["cannot_corroborate"])
        self.assertEqual(facts["world"]["id"], evidence["world_id"])
        self.assertIn('"cannot_corroborate": true', client.prompts[0])
        note = (folder / "research_note.md").read_text()
        self.assertIn(evidence["world_id"], note)
        self.assertIn("cannot_corroborate", note)
        self.assertNotIn("arms run on a constructed dataset", note)


if __name__ == "__main__":
    unittest.main()
