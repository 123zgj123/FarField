"""Heldout replication reuses one registration and executes actual sandboxes."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.freeze import freeze_world
from farfield.extras.generate import GeneratedCard
from farfield.extras.openworld.actions import ActionInstance, SPECS
from farfield.extras.openworld import OpenWorld
from farfield.extras.openworld.events import make_event
from farfield.extras.openworld.reducer import apply_events
from farfield.extras.openworld.state import ScientificState
from farfield.extras.openworld.workers import LegacyResearchWorkers
from test_openworld_workers import ReplayClient


SOURCE = """import json
from pathlib import Path
table = json.loads(Path('data/table.json').read_text())
def measure(enabled):
    rows = table['rows']
    selected = rows[:len(rows)//2] if enabled else rows
    values = [float(row['x']) for row in selected]
    return max(values) - min(values)
Path('metrics.json').write_text(json.dumps({'treatment': measure(True), 'control': measure(False)}))
"""
DIAGNOSIS = {
    "alternative": "noise widens the column spread rather than dropout reducing it",
    "experiment": "dropout removes the last half of ordered rows; noise is held absent in both arms",
    "treatment_arm": "dropout of the last half of rows with no added noise",
    "control_arm": "all rows with no added noise",
    "expected_direction": "treatment_lower", "alternative_direction": "treatment_higher",
    "expected_if_alternative": "column spread increases if noise accounts for the contrast",
    "margin": 0.1, "margin_reason": "a ten percent change exceeds numerical roundoff",
    "world_lever": "dropout", "world_observable": "mean_column_spread",
}


class ReplicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        self.world = self.freeze("training", list(range(10)) + list(range(100, 110)))
        self.heldout = self.freeze("heldout", list(range(10, 20)) + list(range(200, 210)))
        claim = "Removing the final half of ordered numeric table rows decreases mean_column_spread"
        card = GeneratedCard(card_id="gen_dropout", operator="assumption_removal", claim=claim,
            mechanism="dropout removes the high-valued regime", prediction="mean_column_spread decreases",
            falsifier="claim_contains_a_falsifiable_assertion", pair=("numeric table", "ordering"),
            pair_nodes=("concept:table", "assumption:ordering"), alienness=0.0,
            model="replay", artifact_digest="d", artifact_uri="file:///dev/null", replay_mode="replay",
            world_lever="dropout", world_observable="mean_column_spread", far_maps_to_lever="ordering dropout",
            claim_spec={"disposition": "object_validated", "spec": {
                "claim_id": "gen_dropout", "target_object": "numeric table", "statement": claim,
                "claim_type": "intervention", "mechanism": {"name": "dropout", "status": "attested",
                    "world_refs": ["dropout"]}, "handle": {"id": "dropout", "kind": "interventional"},
                "measurement": {"dv": "mean_column_spread"}}})
        self.proposal = {"hypotheses_compared": [card.card_id, "H-noise"],
            "expected_outcomes": {card.card_id: {"observable": "mean_column_spread", "direction": "decrease"},
                                  "H-noise": {"observable": "mean_column_spread", "direction": "increase"}},
            "nuisance_factors": ["noise"], "minimal_discriminating_intervention": DIAGNOSIS["experiment"]}
        self.state = ScientificState(goal="numeric table sampling", world_id=self.world.id)
        self.state.theories = [{"id": card.card_id, "question_id": "q1", "card": card.to_dict(),
                               "competing": ["H-noise"], "experiment_proposal": self.proposal},
                              {"id": "H-noise", "question_id": "q1", "text": DIAGNOSIS["alternative"]}]
        self.state.artifacts = [{"kind": "literature_survey", "question_id": "q1", "works": [{
            "title": "Protocols for reporting numerical measurements", "published": "2020",
            "work_id": "doi:10.1234/example", "source": "replay"}]}]
        self.state.world_versions = [{"id": self.world.id, "digest": self.world.digest},
                                    {"id": self.heldout.id, "digest": self.heldout.digest, "role": "heldout"}]

    def freeze(self, name, values):
        path = self.workspace / f"{name}.csv"
        path.write_text("x,label\n" + "\n".join(f"{value},sample" for value in values) + "\n")
        return freeze_world(world_id=name, schema="numeric_table", slice_rule=f"first_rows:{len(values)}",
                            source_file=path, url=f"https://example.invalid/{name}.csv",
                            catalog=self.workspace / "worlds")

    def action(self, kind, target="gen_dropout", **extra):
        return ActionInstance(SPECS[kind], target=target, frontier_target_id="q1", extra=extra)

    def initial_probe(self):
        client = ReplayClient([DIAGNOSIS, {"measure": "mean_column_spread", "source": SOURCE}])
        result = LegacyResearchWorkers(client=client, workspace=self.workspace).probe(
            self.action("PROBE"), self.state.view())
        self.assertEqual(result.status, "probe_positive", result.to_dict())
        admitted = OpenWorld(workspace=self.workspace, state=self.state)._admit(self.action("PROBE"), result)
        self.state = apply_events(self.state, admitted.events)
        return admitted.evidence

    def verify(self, original, **extra):
        action = self.action("VERIFY", original["evidence_id"], **extra)
        result = LegacyResearchWorkers(workspace=self.workspace).verify(action, self.state.view())
        return OpenWorld(workspace=self.workspace, state=self.state)._admit(action, result)

    def test_discriminator_is_registered_before_execution_and_validated_within_conditions(self):
        original = self.initial_probe()
        self.assertEqual(original.get("mechanism_validated"), "registered_discriminator_within_tested_conditions")
        self.assertEqual(original.get("causal_scope"), "within_tested_conditions")
        protocol = json.loads((Path(original["protocol_dir"]) / "protocol.json").read_text())
        self.assertEqual(protocol.get("experiment_proposal"), self.proposal)

    def test_heldout_reuses_source_and_diagnosis_and_mints_new_executed_evidence(self):
        original = self.initial_probe()
        same = self.verify(original, replication_world_id="")
        self.assertEqual(same.status, "verified", same.to_dict())
        self.state = apply_events(self.state, same.events)
        preserved = copy.deepcopy(self.state.evidence_records[0])
        replicated = self.verify(original, replication_world_id=self.heldout.id)
        self.assertEqual(replicated.status, "replicated", replicated.to_dict())
        row = replicated.events[0].payload
        self.assertNotEqual(row["evidence_id"], original["evidence_id"])
        self.assertEqual(row["replication_of"], original["evidence_id"])
        self.assertEqual(row["world_digest"], self.heldout.digest)
        self.assertEqual(row["experiment_digest"], original["experiment_digest"])
        self.assertEqual(row["diagnosis"], original["diagnosis"])
        self.assertEqual(row["treatment"], 9.0)
        self.assertEqual(row["control"], 199.0)
        self.assertTrue(row["transfer_ok"])
        self.assertTrue(row["reproduction_ok"])
        self.assertEqual(row["host_record"]["evidence_id"], row["evidence_id"])
        self.assertEqual(self.state.evidence_records[0], preserved)
        self.state = apply_events(self.state, replicated.events)
        self.assertIn("gen_dropout", self.state.verified_research_state)

    def test_same_bytes_under_another_world_name_cannot_count_as_transfer(self):
        original = self.initial_probe()
        duplicate = self.freeze("duplicate", list(range(10)) + list(range(100, 110)))
        result = self.verify(original, replication_world_id=duplicate.id)
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.extras.get("transfer_ok", False))

    def test_altered_source_refuses_heldout_execution(self):
        original = self.initial_probe()
        script = Path(original["protocol_dir"]) / "experiment.py"
        script.write_text(script.read_text() + "\n# modified after results\n")
        result = self.verify(original, replication_world_id=self.heldout.id)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(len(self.state.evidence_records), 1)

    def test_generated_replication_world_is_not_an_attested_transfer(self):
        original = self.initial_probe()
        path = self.heldout.root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["role"] = "generated"
        path.write_text(json.dumps(manifest))
        result = self.verify(original, replication_world_id=self.heldout.id)
        self.assertEqual(result.status, "blocked")

    def test_indistinguishable_proposal_does_not_validate_a_mechanism(self):
        self.proposal["expected_outcomes"]["H-noise"]["direction"] = "decrease"
        result = self.initial_probe()
        self.assertFalse(result.get("mechanism_validated"))

    def test_inconclusive_heldout_run_is_new_evidence_without_transfer_success(self):
        original = self.initial_probe()
        null = self.freeze("null-heldout", list(range(10)) + list(range(10)))
        result = self.verify(original, replication_world_id=null.id)
        self.assertEqual(result.status, "replication_inconclusive", result.to_dict())
        row = result.events[0].payload
        self.assertEqual(row["outcome"], "uninformative")
        self.assertEqual(row["treatment"], row["control"])
        self.assertFalse(row["transfer_ok"])
        self.assertTrue(row["reproduction_ok"])
        self.assertFalse(row["mechanism_validated"])

    def test_claiming_a_different_measured_observable_cannot_validate_discriminator(self):
        client = ReplayClient([DIAGNOSIS, {"measure": "unrelated performance score", "source": SOURCE}])
        result = LegacyResearchWorkers(client=client, workspace=self.workspace).probe(
            self.action("PROBE"), self.state.view())
        self.assertEqual(result.status, "probe_execution_blocked", result.to_dict())
        self.assertIn('probe_measure_differs_from_registered_claim', result.extras['reason'])
        self.assertFalse(result.evidence)

    def test_diagnosis_cannot_change_registered_intervention(self):
        client = ReplayClient([{**DIAGNOSIS, 'world_lever': 'noise'}])
        result = LegacyResearchWorkers(client=client, workspace=self.workspace).probe(
            self.action('PROBE'), self.state.view())
        self.assertEqual(result.status, 'probe_execution_blocked')
        self.assertIn('diagnosis_differs_from_registered_claim', result.extras['reason'])
        self.assertFalse(result.evidence)

    def test_bound_claim_cannot_execute_against_another_world_identity(self):
        self.state.theories[0]['card']['claim_spec']['world_id'] = 'different-world'
        client = ReplayClient([])
        result = LegacyResearchWorkers(client=client, workspace=self.workspace).probe(
            self.action('PROBE'), self.state.view())
        self.assertEqual(result.status, 'probe_execution_blocked')
        self.assertIn('claim_world_identity_mismatch', result.extras['reason'])
        self.assertFalse(result.evidence)

    def test_unresolved_ablation_does_not_validate_mechanism(self):
        diagnosis = {**DIAGNOSIS, "alternative": "unmeasured background drift accounts for the effect"}
        result = LegacyResearchWorkers(client=ReplayClient([diagnosis, {
            "measure": "mean_column_spread", "source": SOURCE}]), workspace=self.workspace).probe(
                self.action("PROBE"), self.state.view())
        self.assertTrue(result.evidence["ablation_required"])
        self.assertTrue(result.evidence["cannot_corroborate"])
        self.assertFalse(result.evidence["mechanism_validated"])


if __name__ == "__main__":
    unittest.main()
