"""Direct research workers: real compilation/execution, offline model replay."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.freeze import freeze_world
from farfield.extras.livefeed import FeedBlocked, FreshWork
from farfield.extras.llm import Completion
from farfield.extras.openworld.actions import ActionInstance, SPECS
from farfield.extras.openworld.state import ScientificState


class ReplayClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.prompts.append((purpose, prompt))
        if not self.answers:
            raise AssertionError("unexpected model call")
        return Completion(text=json.dumps(self.answers.pop(0)), model="replay",
                          request_digest="request", digest="d" * 16,
                          artifact_uri="file:///dev/null", finish_reason="stop",
                          prompt_tokens=1, completion_tokens=1, reasoning_tokens=0,
                          mode="replay")


THEORY = {
    "assumption": "uniform sampling",
    "claim": "Removing uniform sampling reduces query cost for skewed table records",
    "mechanism": "uniform sampling overlooks repeated keys; caching repeated keys avoids scans",
    "prediction": "the cache reduces average scanned records by at least ten percent",
    "objection": "the apparent speedup may instead reflect shorter queries",
}
DIAGNOSIS = {
    "alternative": "shorter queries rather than caching account for the reduction",
    "experiment": "use the same table records with caching enabled or disabled",
    "treatment_arm": "reuse previously seen keys in a cache",
    "control_arm": "scan every record for every key",
    "expected_direction": "treatment_lower", "alternative_direction": "treatment_higher",
    "expected_if_alternative": "cost does not fall on the same query sequence",
    "margin": 0.05, "margin_reason": "a five percent reduction exceeds timer noise",
}
SOURCE = """import json
from pathlib import Path
table = json.loads(Path('data/table.json').read_text())
def measure(enabled):
    total = 0
    seen = set()
    for row in table['rows']:
        key = str(row)
        if not enabled or key not in seen:
            total += 1
        seen.add(key)
    return float(total)
Path('metrics.json').write_text(json.dumps({'treatment': measure(True), 'control': measure(False)}))
"""


def action(kind, target="q1", **extra):
    return ActionInstance(SPECS[kind], target=target, frontier_target_id="q1", extra=extra)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec("farfield.extras.openworld.workers")
        self.assertIsNotNone(spec, "direct research worker adapter is missing")
        from farfield.extras.openworld.workers import LegacyResearchWorkers
        self.worker_class = LegacyResearchWorkers
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        self.state = ScientificState(goal="query costs for skewed table records")
        self.state.add_question(question_id="q1", text="Does caching avoid repeated scans?")

    def worker(self, answers=(), feed=None):
        return self.worker_class(client=ReplayClient(answers), feed=feed, workspace=self.workspace)

    def test_nonprobe_card_fails_preflight_without_model_or_execution(self):
        theory = self.theory()
        self.world()
        theory['card']['idea_kind'] = 'theory'
        result = self.worker().probe(action('PROBE', target=theory['id']), self.state.view())
        self.assertEqual(result.status, 'probe_execution_blocked')
        self.assertIn('requires_probe_compilation_or_world_acquisition', result.extras['reason'])
        self.assertFalse([e for e in result.events if e.event_type == 'EvidenceAdded'])

    def theory(self):
        worker = self.worker([THEORY])
        result = worker.executors()["THEORIZE"](action("THEORIZE"), self.state.view())
        row = result.events[0].payload
        self.state.theories.append(dict(row))
        return row

    def world(self):
        source = self.workspace / "source.csv"
        source.write_text("key,value\na,1\na,1\nb,2\nb,2\n")
        fixture = freeze_world(world_id="table-world", schema="numeric_table",
                               slice_rule="first_rows:4", catalog=self.workspace / "worlds",
                               url="https://example.invalid/table.csv", source_file=source)
        self.state.world_id = fixture.id
        return fixture

    def test_generic_theory_preserves_generated_card_and_question(self):
        row = self.theory()
        self.assertEqual(row["question_id"], "q1")
        self.assertEqual(row["card"]["claim"], THEORY["claim"])
        self.assertEqual(row["epistemic"], "GENERATED")
        self.assertIn(THEORY["objection"], row["competing"])
        self.assertNotIn("validator", json.dumps(row))

    def test_missing_model_is_blocked_without_invented_theory(self):
        result = self.worker_class(workspace=self.workspace).executors()["THEORIZE"](
            action("THEORIZE"), self.state.view())
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.events[0].event_type, "ActionBlocked")

    def test_survey_verifies_identifier_and_preserves_actual_abstract_span(self):
        paper = FreshWork("Caching skewed table queries", "2026-01-01", arxiv_id="2601.12345",
                          abstract="Caching reduces scans. Uniform sampling misses repeated keys.")
        class Feed:
            verify_literature = True
            def survey_around(self, *args, **kwargs):
                return [paper]
            def verify_fetcher(self, url):
                return b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Caching skewed table queries</title></entry></feed>'
        result = self.worker(feed=Feed()).executors()["SURVEY"](action("SURVEY"), self.state.view())
        self.assertEqual(result.status, "surveyed")
        row = result.events[0].payload
        self.assertEqual(row["kind"], "literature_survey")
        self.assertEqual(row["read_first"], ["2601.12345"])
        self.assertEqual(row["spans"][0]["text"], paper.abstract)
        self.assertTrue(Path(row["path"]).is_file())

    def test_empty_survey_is_blocked_without_placeholder_citations(self):
        class Feed:
            verify_literature = False
            def survey_around(self, *args, **kwargs):
                return []
        result = self.worker(feed=Feed()).executors()["SURVEY"](action("SURVEY"), self.state.view())
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.extras["citations"], [])

    def test_probe_ignores_supplied_arm_numbers_and_executes_registered_source(self):
        row, fixture = self.theory(), self.world()
        result = self.worker([DIAGNOSIS, {"measure": "scanned records", "source": SOURCE}]).executors()["PROBE"](
            action("PROBE", target=row["id"], treatment=-999, control=999), self.state.view(), world=fixture)
        self.assertIn(result.status, {"probe_positive", "probe_inconclusive"})
        self.assertEqual(result.evidence["treatment"], 2.0)
        self.assertEqual(result.evidence["control"], 4.0)
        self.assertEqual(result.evidence["executor_status"], "ran")
        self.assertEqual(result.evidence["epistemic"], "WORLD")
        folder = Path(result.evidence["protocol_dir"])
        self.assertEqual((folder / "experiment.py").read_text(), SOURCE.strip())
        protocol = json.loads((folder / "protocol.json").read_text())
        self.assertEqual(protocol["world_digest"], fixture.digest)
        self.assertEqual(protocol["experiment_digest"], result.evidence["experiment_digest"])
        self.assertNotIn("treatment", protocol.get("cheap_probe") or {})
        self.assertEqual(result.evidence["outcome"], result.evidence["verdict"])
        self.assertEqual(result.evidence["execution_status"], "ran")
        self.assertFalse(result.evidence["claim_consistent"])
        self.assertFalse(result.evidence["literature_checked"])
        self.assertFalse(result.evidence["mechanism_validated"])

    def test_no_attested_world_blocks_before_model_or_execution(self):
        row = self.theory()
        result = self.worker().executors()["PROBE"](action("PROBE", target=row["id"]), self.state.view())
        self.assertEqual(result.status, "probe_execution_blocked")
        self.assertIsNone(result.evidence)

    def test_policy_refusal_is_not_a_negative_result(self):
        row, fixture = self.theory(), self.world()
        bad = {"measure": "bad", "source": "import os\nos.system('true')"}
        result = self.worker([DIAGNOSIS, bad, bad]).executors()["PROBE"](
            action("PROBE", target=row["id"]), self.state.view(), world=fixture)
        self.assertEqual(result.status, "probe_policy_refused")
        self.assertIsNone(result.evidence)

    def test_synthesis_without_executed_evidence_is_blocked(self):
        self.theory()
        result = self.worker().executors()["SYNTHESIZE"](action("SYNTHESIZE"), self.state.view())
        self.assertEqual(result.status, "blocked")
        self.assertFalse(list(self.workspace.rglob("paper.md")))

    def executed(self, source=SOURCE):
        row, fixture = self.theory(), self.world()
        result = self.worker([DIAGNOSIS, {"measure": "scanned records", "source": source}]).executors()["PROBE"](
            action("PROBE", target=row["id"]), self.state.view(), world=fixture)
        self.assertIsNotNone(result.evidence, result.to_dict())
        self.state.evidence_records.append(result.evidence)
        return result.evidence

    def test_host_reproduces_same_identity_without_claiming_independent_replication(self):
        evidence = self.executed()
        result = self.worker().executors()["VERIFY"](
            action("VERIFY", target=evidence["evidence_id"]), self.state.view())
        self.assertEqual(result.status, "verified", result.to_dict())
        row = result.events[0].payload
        self.assertEqual(row["evidence_id"], evidence["evidence_id"])
        self.assertTrue(row["reproduction_ok"])
        self.assertTrue(row["identity_ok"])
        self.assertTrue(row["executable_ok"])
        self.assertFalse(row["claim_consistent"])
        self.assertFalse(row["transfer_ok"])
        self.assertTrue(row["replication_required"])

    def test_mutating_registered_margin_blocks_host_reproduction(self):
        evidence = self.executed()
        path = Path(evidence["protocol_dir"]) / "protocol.json"
        protocol = json.loads(path.read_text())
        protocol["diagnosis"]["margin"] = 0.001
        path.write_text(json.dumps(protocol))
        result = self.worker().executors()["VERIFY"](
            action("VERIFY", target=evidence["evidence_id"]), self.state.view())
        self.assertEqual(result.status, "blocked")
        self.assertFalse((path.parent / "host_run.json").exists())

    def test_mutating_registered_script_blocks_host_reproduction(self):
        evidence = self.executed()
        path = Path(evidence["protocol_dir"]) / "experiment.py"
        path.write_text(path.read_text() + "\n# different experiment\n")
        result = self.worker().executors()["VERIFY"](
            action("VERIFY", target=evidence["evidence_id"]), self.state.view())
        self.assertEqual(result.status, "blocked")
        self.assertFalse((path.parent / "host_run.json").exists())

    def test_synthesis_uses_executed_numbers_and_marks_scientific_scope_gap(self):
        evidence = self.executed()
        result = self.worker().executors()["SYNTHESIZE"](
            action("SYNTHESIZE", target=evidence["evidence_id"]), self.state.view())
        self.assertEqual(result.status, "synthesized", result.to_dict())
        artifact = result.events[0].payload
        self.assertTrue(artifact["cannot_corroborate"])
        text = Path(artifact["path"]).read_text()
        self.assertIn("2", text)
        self.assertIn("4", text)
        self.assertIn("claim_spec_not_registered", text)

    def test_runtime_crash_keeps_registered_design_and_is_not_negative_evidence(self):
        row, fixture = self.theory(), self.world()
        broken = SOURCE.replace("table['rows']", "table['missing_rows']")
        result = self.worker([DIAGNOSIS, {"measure": "scanned records", "source": broken}]).executors()["PROBE"](
            action("PROBE", target=row["id"]), self.state.view(), world=fixture)
        self.assertEqual(result.status, "probe_execution_blocked")
        self.assertEqual(result.extras["executor_status"], "crashed")
        self.assertIsNone(result.evidence)
        self.assertEqual(result.extras["diagnosis"]["experiment"], DIAGNOSIS["experiment"])

    def test_probe_that_rewrites_bound_data_cannot_emit_world_evidence(self):
        row, fixture = self.theory(), self.world()
        changed = SOURCE + "\nPath('data/table.json').write_text('{}')\n"
        result = self.worker([DIAGNOSIS, {"measure": "scanned records", "source": changed}]).executors()["PROBE"](
            action("PROBE", target=row["id"]), self.state.view(), world=fixture)
        self.assertEqual(result.status, "probe_execution_blocked")
        self.assertIsNone(result.evidence)
        self.assertIn("digest", result.extras["reason"])

    def test_stale_claim_spec_cannot_validate_a_handle_missing_from_the_current_world(self):
        row, fixture = self.theory(), self.world()
        row["card"]["claim_spec"] = {"disposition": "object_validated", "spec": {
            "claim_id": row["id"], "statement": row["card"]["claim"],
            "target_object": "table", "claim_type": "association",
            "mechanism": {"name": "cache", "status": "observed", "world_refs": []},
            "handle": {"id": "foreign_handle", "kind": "observational"},
            "measurement": {"dv": "foreign_field"}}}
        result = self.worker().executors()["PROBE"](
            action("PROBE", target=row["id"]), self.state.view(), world=fixture)
        self.assertEqual(result.status, "probe_execution_blocked")
        self.assertIsNone(result.evidence)
        self.assertIn("claim_object_validation", result.extras["reason"])

    def test_negative_scientific_result_is_not_retried(self):
        row, fixture = self.theory(), self.world()
        diagnosis = {**DIAGNOSIS, "expected_direction": "treatment_higher", "alternative_direction": "treatment_lower"}
        result = self.worker([diagnosis, {"measure": "scanned records", "source": SOURCE}]).executors()["PROBE"](
            action("PROBE", target=row["id"]), self.state.view(), world=fixture)
        self.assertEqual(result.status, "probe_scientifically_negative")
        self.assertEqual(result.evidence["outcome"], "weakens")

    def test_optional_paper_draft_runs_one_existing_writer_without_review_loop(self):
        evidence = self.executed()
        draft = {"title": "Caching repeated table scans", "abstract": "An exploratory comparison.",
                 "proposal_markdown": "# Caching\nTreatment scanned 2 records; control scanned 4 records.",
                 "experiment_design_markdown": "# Design\nRun the registered script on the frozen table."}
        worker = self.worker([draft])
        result = worker.executors()["SYNTHESIZE"](
            action("SYNTHESIZE", target=evidence["theory_id"], paper_draft=True), self.state.view())
        self.assertEqual(result.status, "synthesized", result.to_dict())
        self.assertEqual(result.extras.get("paper", {}).get("rounds"), 0)
        self.assertTrue((Path(evidence["protocol_dir"]) / "paper" / "RESEARCH_PROPOSAL.md").is_file())


if __name__ == "__main__":
    unittest.main()
