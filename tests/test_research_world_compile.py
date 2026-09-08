"""Actual research workers compile only against a bound, attested world."""
import unittest
import tempfile
from dataclasses import replace
from pathlib import Path

from farfield.extras.openworld.actions import ActionInstance, SPECS, THEORIZE
from farfield.extras.openworld.state import ScientificState
from farfield.extras.openworld.workers import LegacyResearchWorkers
from farfield.extras.reframe import generate_reframe
from farfield.extras.world import load_fixture
from test_openworld_workers import ReplayClient
import test_research_branches as branch_fixtures


TOPIC = "Iris feature measurements"


def answer(**extra):
    metadata = branch_fixtures.BranchTests().metadata()
    metadata["experiment_proposal"]["expected_outcomes"] = {
        "self": {"observable": "mean_column_spread", "direction": "increase"},
        "H2": {"observable": "mean_column_spread", "direction": "unchanged"},
    }
    return {"assumption": "stable feature scales",
        "claim": "Iris feature measurements with stable feature scales hide a noise-dependent spread boundary",
        "mechanism": "Perturbing stable feature scales makes Iris feature measurements expose anisotropic spread",
        "prediction": "noise increases mean_column_spread on Iris feature measurements",
        "falsifier": "Noise does not increase mean_column_spread on the frozen Iris measurements",
        "pair": [TOPIC, "unconstrained mechanism exploration"],
        "idea_kind": "probe", "world_lever": "noise", "world_observable": "mean_column_spread",
        "far_maps_to_lever": "unconstrained mechanism exploration becomes noise applied to the attested Iris feature measurements",
        "target_object": "fisher-iris", "research_metadata": metadata, **extra}


class WorldCompilationTests(unittest.TestCase):
    def world(self):
        return load_fixture(Path(__file__).resolve().parents[1] / "worlds" / "fisher-iris")

    def state(self):
        state = ScientificState(goal=TOPIC, world_id="fisher-iris")
        state.add_question(question_id="Q1", text=TOPIC)
        state.artifacts.append({"kind": "literature_survey", "question_id": "Q1",
                                "works": [{"title": "Source methodology", "work_id": "P1"}]})
        return state

    def test_both_live_shaped_worker_routes_compile_canonical_object_identity(self):
        for lane in ("unconstrained", "assumption_reframe"):
            with self.subTest(lane=lane):
                client = ReplayClient([answer()])
                result = LegacyResearchWorkers(client=client).theorize(
                    ActionInstance(SPECS[THEORIZE], target="Q1", extra={"research_lane": lane}),
                    self.state().view(), world=self.world())
                self.assertEqual(result.status, "theorized", result.extras)
                card = result.extras["theory"]["card"]
                self.assertIsNotNone(card.get("claim_spec"), "actual world was not offered to claim compiler")
                spec = card["claim_spec"]
                self.assertEqual(spec["claim_id"], card["card_id"])
                self.assertEqual(spec["target_object"], "fisher-iris")
                self.assertEqual(spec["world_id"], "fisher-iris")
                self.assertEqual(spec["world_digest"], self.world().digest)
                self.assertEqual(spec["handle"]["id"], "noise")
                self.assertEqual(spec["measurement"]["dv"], "mean_column_spread")
                self.assertEqual(spec["disposition"], "object_validated")
                self.assertEqual(card["idea_kind"], "probe")
                self.assertIn("fisher-iris", client.prompts[0][1])
                self.assertIn("mean_column_spread", client.prompts[0][1])

    def test_invalid_explicit_object_is_rejected_on_both_routes(self):
        for lane in ("unconstrained", "assumption_reframe"):
            with self.subTest(lane=lane):
                result = LegacyResearchWorkers(client=ReplayClient([answer(target_object="unrelated-protein")])).theorize(
                    ActionInstance(SPECS[THEORIZE], target="Q1", extra={"research_lane": lane}),
                    self.state().view(), world=self.world())
                self.assertEqual(result.status, "blocked", result.extras)
                self.assertFalse(any(event.event_type == "TheoryCreated" for event in result.events))

    def test_no_actual_world_keeps_speculation_without_llm_claim_spec(self):
        with tempfile.TemporaryDirectory() as workspace:
            for folder in (None, workspace):
                with self.subTest(workspace=folder):
                    result = LegacyResearchWorkers(workspace=folder,
                        client=ReplayClient([answer(claim_spec={"disposition": "OBJECT_VALIDATED"})])).theorize(
                        ActionInstance(SPECS[THEORIZE], target="Q1", extra={"research_lane": "assumption_reframe"}), self.state().view())
                    self.assertEqual(result.status, "theorized", result.extras)
                    self.assertIsNone(result.extras["theory"]["card"].get("claim_spec"))

    def test_supplied_world_identity_mismatch_does_not_compile(self):
        result = LegacyResearchWorkers(client=ReplayClient([answer()])).theorize(
            ActionInstance(SPECS[THEORIZE], target="Q1", extra={"research_lane": "assumption_reframe"}),
            self.state().view(), world=replace(self.world(), id="other-world"))
        self.assertEqual(result.status, "blocked", result.extras)

    def test_question_is_not_silently_promoted_to_probe_by_reframe(self):
        client = ReplayClient([answer(idea_kind="question", world_lever="", world_observable="")])
        card = generate_reframe(client, seed_label=TOPIC, near_labels=(TOPIC,), operator="assumption_removal")
        self.assertEqual(card.idea_kind, "question")
        self.assertIsNone(card.claim_spec)

    def test_graph_check_is_not_a_scientific_falsifier_in_either_lane(self):
        for lane in ("unconstrained", "assumption_reframe"):
            with self.subTest(lane=lane):
                result = LegacyResearchWorkers(client=ReplayClient([answer(falsifier="pair_not_already_combined")])).theorize(
                    ActionInstance(SPECS[THEORIZE], target="Q1", extra={"research_lane": lane}), self.state().view())
                self.assertEqual(result.status, "blocked", result.extras)

    def test_question_cannot_compile_an_interventional_handle_as_observation(self):
        result = LegacyResearchWorkers(client=ReplayClient([answer(idea_kind="question")])).theorize(
            ActionInstance(SPECS[THEORIZE], target="Q1", extra={"research_lane": "assumption_reframe"}),
            self.state().view(), world=self.world())
        self.assertEqual(result.status, "blocked", result.extras)

    def test_evidence_driven_reframe_applies_the_operator_it_records(self):
        state = self.state()
        state.theories = [{"id": "parent", "hop": 1, "question_id": "Q1"}]
        state.evidence_records = [{"theory_id": "parent", "evidence_id": "E1",
            "epistemic": "WORLD", "attested": True, "execution_status": "ran",
            "world_digest": self.world().digest, "experiment_digest": "recorded-source", "outcome": "weakens"}]
        result = LegacyResearchWorkers(client=ReplayClient([answer()])).theorize(
            ActionInstance(SPECS[THEORIZE], target="Q1", extra={"research_lane": "assumption_reframe", "parent_theory_id": "parent"}),
            state.view(), world=self.world())
        self.assertEqual(result.status, "theorized", result.extras)
        theory = result.extras["theory"]
        self.assertEqual(theory["operator_reused"], "representation_change")
        self.assertEqual(theory["card"]["operator"], theory["operator_reused"])


if __name__ == "__main__":
    unittest.main()
