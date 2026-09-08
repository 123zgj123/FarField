"""Production routing admission consumes protected, candidate-specific replays."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from farfield.extras import routing
from farfield.extras.evidence import evidence_id


def observation(mid, operator, verdict):
    treatment, control = (2.0, 4.0) if verdict == "supports" else (4.0, 4.0)
    exp, world = ("a" * 64), ("b" * 64)
    cid = f"{mid}-{operator}"
    eid = evidence_id(claim_id=cid, experiment_digest=exp, world_digest=world, data_digest=world)
    return {
        "card_id": cid, "operator": operator, "probe_kind": "WORLD",
        "execution_status": "ran", "ledger": "scientific", "claim_consistent": True,
        "cannot_corroborate": False, "evidence_id": eid, "experiment_digest": exp,
        "world_digest": world, "data_digest": world, "verdict": verdict,
        "treatment": treatment, "control": control,
        "diagnosis": {"alternative": "query length accounts for the effect",
            "experiment": "same log with and without caching", "treatment_arm": "cache on",
            "control_arm": "cache off", "expected_direction": "treatment_lower",
            "alternative_direction": "treatment_higher", "expected_if_alternative": "tie",
            "margin": 0.05, "margin_reason": "five percent exceeds timer noise"},
        "host_record": {"status": "ran", "evidence_id": eid, "experiment_digest": exp,
            "world_digest": world, "data_digest": world, "treatment": treatment,
            "control": control, "replication": "E1_same_instance"},
        "measured_cost": {"tokens": 10, "wall_seconds": 0.1},
    }


def mission(mid, failure=False):
    row = {"mission_id": mid, "status": "completed", "cards": [
        observation(mid, "directional", "supports"), observation(mid, "analogy", "uninformative")]}
    if failure:
        row["failures"] = [{"id": f"f-{mid}", "class": "SCIENTIFIC_FAILURE",
                            "signature": "non_discriminating_design", "evidence_id": row["cards"][1]["evidence_id"]}]
    return row


class ProtectedReplay:
    """An application-owned offline replay over previously validated executions."""
    def __init__(self, regress=False, expensive=False):
        self.calls = []
        self.regress = regress
        self.expensive = expensive

    def __call__(self, policy, source, *, partition):
        weights = (policy or {}).get("weights") or {}
        candidate = bool(weights)
        pick = "directional" if candidate else "analogy"
        if self.regress and partition == "regression":
            pick = "analogy" if candidate else "directional"
        row = copy.deepcopy(next(card for card in source["cards"] if card["operator"] == pick))
        if self.expensive and candidate:
            row["measured_cost"]["tokens"] = 1000
        self.calls.append((routing._digest(policy), source["mission_id"], partition))
        return {"mission_id": source["mission_id"], "source_digest": routing._digest(source),
                "policy_digest": routing._digest(policy), "run_id": f"run-{len(self.calls)}",
                "status": "completed", "observations": [row],
                "measured_cost": dict(row["measured_cost"])}


class PolicyAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log, self.path = self.root / "log.json", self.root / "policy.json"
        self.rows = [mission(f"m{i}", failure=i < 3) for i in range(6)]
        for row in self.rows:
            routing.append_mission(self.log, row)
        self.suite = {"training": ["m0", "m1", "m2"], "nearby": ["m3"],
                      "heldout": ["m4"], "regression": ["m5"]}
        self.candidate = {"weights": {"directional": 1.0, "analogy": 0.1}, "fitted_on": 3}

    def evaluate(self, **kwargs):
        fn = getattr(routing, "evaluate_research_policy", None)
        self.assertTrue(callable(fn), "production research-policy evaluator is missing")
        return fn(self.log, self.path, suite=kwargs.pop("suite", self.suite),
                  candidate=kwargs.pop("candidate", self.candidate),
                  evaluator=kwargs.pop("evaluator", ProtectedReplay()), **kwargs)

    def admit(self, receipt):
        return routing.maybe_update(self.log, self.path, evaluation=receipt)

    def test_missing_evaluation_never_installs_and_persists_rejection(self):
        result = routing.maybe_update(self.log, self.path)
        self.assertFalse(result["committed"])
        self.assertFalse(self.path.exists())
        decisions = self.path.with_name("policy.json.decisions.jsonl")
        self.assertTrue(decisions.is_file())
        self.assertIn("evaluation", decisions.read_text())

    def test_caller_asserted_boolean_evaluation_is_not_a_receipt(self):
        result = self.admit({"heldout_improved": True, "regression_ok": True, "cost_ok": True})
        self.assertFalse(result["committed"])
        self.assertFalse(self.path.exists())

    def test_positive_protected_replays_install_once_and_bind_policy(self):
        replay = ProtectedReplay()
        receipt = self.evaluate(evaluator=replay)
        result = self.admit(receipt)
        self.assertTrue(result["committed"], result)
        self.assertEqual(routing.load_policy(self.path), self.candidate)
        self.assertEqual({partition for _, _, partition in replay.calls},
                         {"nearby", "heldout", "regression", "historical_failure"})
        before = self.path.read_bytes()
        again = self.admit(receipt)
        self.assertFalse(again["committed"])
        self.assertIn("consumed", again["reason"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_overlapping_mission_partitions_reject_before_any_replay(self):
        suite = {**self.suite, "heldout": ["m0"]}
        replay = ProtectedReplay()
        result = self.admit(self.evaluate(suite=suite, evaluator=replay))
        self.assertFalse(result["committed"])
        self.assertEqual(replay.calls, [])

    def test_infrastructure_failures_do_not_authorize_policy_repair(self):
        rows = routing.load_log(self.log)
        for row in rows[:3]:
            row["failures"][0]["class"] = "HARNESS_FAILURE"
        self.log.write_text(json.dumps({"missions": rows, "digest": routing._digest(rows)}))
        result = self.admit(self.evaluate())
        self.assertFalse(result["committed"])
        self.assertIn("scientific", result["reason"])

    def test_uncompleted_mission_rejects(self):
        rows = routing.load_log(self.log)
        rows[-1]["status"] = "running"
        self.log.write_text(json.dumps({"missions": rows, "digest": routing._digest(rows)}))
        result = self.admit(self.evaluate())
        self.assertFalse(result["committed"])

    def test_regression_failure_preserves_incumbent_bytes(self):
        result = self.admit(self.evaluate(evaluator=ProtectedReplay(regress=True)))
        self.assertFalse(result["committed"])
        self.assertFalse(self.path.exists())

    def test_unmatched_cost_cannot_buy_improved_capability(self):
        result = self.admit(self.evaluate(evaluator=ProtectedReplay(expensive=True)))
        self.assertFalse(result["committed"])
        self.assertFalse(self.path.exists())

    def test_changed_baseline_invalidates_receipt(self):
        receipt = self.evaluate()
        incumbent = {"weights": {"directional": 0.9, "analogy": 0.1}}
        self.path.write_text(json.dumps({"policy": incumbent, "digest": routing._digest(incumbent)}))
        before = self.path.read_bytes()
        result = self.admit(receipt)
        self.assertFalse(result["committed"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_historical_precision_without_execution_evaluator_is_not_capability(self):
        result = self.admit(self.evaluate(evaluator=None))
        self.assertFalse(result["committed"])
        self.assertIn("replay", result["reason"])

    def test_generated_or_unconfirmed_promotions_are_not_validated_capability(self):
        for kind in ("GENERATED", "SYNTHETIC"):
            with self.subTest(kind=kind):
                rows = routing.load_log(self.log)
                rows[4]["cards"][0]["probe_kind"] = kind
                rows[4]["cards"][0]["promoted"] = "verified"
                self.log.write_text(json.dumps({"missions": rows, "digest": routing._digest(rows)}))
                result = self.admit(self.evaluate())
                self.assertFalse(result["committed"])

    def test_receipt_json_cannot_be_edited_to_pass_a_rejected_evaluation(self):
        receipt = self.evaluate(evaluator=None)
        path = Path(receipt.path)
        body = json.loads(path.read_text())
        body["rejections"] = []
        body["metrics"] = {"heldout": {"candidate": 1, "baseline": 0}}
        path.write_text(json.dumps(body))
        result = self.admit(receipt)
        self.assertFalse(result["committed"])
        self.assertIn("mutated", result["reason"])

    def test_changed_evaluation_missions_invalidate_the_receipt(self):
        receipt = self.evaluate()
        routing.append_mission(self.log, mission("m6"))
        result = self.admit(receipt)
        self.assertFalse(result["committed"])
        self.assertIn("inputs changed", result["reason"])

    def test_mutating_the_callers_candidate_cannot_change_issued_receipt(self):
        receipt = self.evaluate()
        self.candidate["weights"]["analogy"] = 99
        result = self.admit(receipt)
        self.assertTrue(result["committed"], result)
        self.assertEqual(routing.load_policy(self.path)["weights"]["analogy"], 0.1)

    def test_equal_replay_outcomes_do_not_count_as_positive_improvement(self):
        class EqualReplay(ProtectedReplay):
            def __call__(self, policy, source, *, partition):
                result = super().__call__(policy, source, partition=partition)
                result["observations"] = [copy.deepcopy(source["cards"][0])]
                return result
        result = self.admit(self.evaluate(evaluator=EqualReplay()))
        self.assertFalse(result["committed"])
        self.assertIn("strictly improve", result["reason"])

    def test_nonfinite_measured_cost_is_rejected(self):
        rows = routing.load_log(self.log)
        rows[4]["cards"][0]["measured_cost"]["wall_seconds"] = math.nan
        self.log.write_text(json.dumps({"missions": rows, "digest": routing._digest(rows)}))
        result = self.admit(self.evaluate())
        self.assertFalse(result["committed"])
        self.assertIn("finite", result["reason"])

    def test_infrastructure_signature_cannot_be_relabelled_as_scientific_failure(self):
        rows = routing.load_log(self.log)
        for row in rows[:3]:
            row["failures"][0]["signature"] = "infra_timeout"
        self.log.write_text(json.dumps({"missions": rows, "digest": routing._digest(rows)}))
        result = self.admit(self.evaluate())
        self.assertFalse(result["committed"])
        self.assertIn("scientific", result["reason"])

    def test_repeating_one_failure_receipt_is_not_three_distinct_failures(self):
        rows = routing.load_log(self.log)
        for row in rows[:3]:
            row["failures"][0]["id"] = "same-failure"
        self.log.write_text(json.dumps({"missions": rows, "digest": routing._digest(rows)}))
        result = self.admit(self.evaluate())
        self.assertFalse(result["committed"])

    def test_unconfirmed_world_boolean_does_not_replace_host_execution(self):
        rows = routing.load_log(self.log)
        row = rows[4]["cards"][0]
        del row["host_record"]
        row.update(host_ok=True, promoted="verified", reproduction_ok=True)
        self.log.write_text(json.dumps({"missions": rows, "digest": routing._digest(rows)}))
        result = self.admit(self.evaluate())
        self.assertFalse(result["committed"])
        self.assertIn("unconfirmed", result["reason"])

    def test_world_label_cannot_override_generated_provenance(self):
        rows = routing.load_log(self.log)
        rows[4]["cards"][0]["epistemic"] = "GENERATED"
        rows[4]["cards"][0]["world_role"] = "generated"
        self.log.write_text(json.dumps({"missions": rows, "digest": routing._digest(rows)}))
        result = self.admit(self.evaluate())
        self.assertFalse(result["committed"])

    def test_unresolved_required_ablation_is_not_resolved_capability(self):
        rows = routing.load_log(self.log)
        rows[4]["cards"][0].update(ablation_required=True, mechanism_identified=False)
        self.log.write_text(json.dumps({"missions": rows, "digest": routing._digest(rows)}))
        result = self.admit(self.evaluate())
        self.assertFalse(result["committed"])


if __name__ == "__main__":
    unittest.main()
