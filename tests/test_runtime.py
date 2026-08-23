from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Callable

from farfield.ledger import EventLedger
from farfield.models import (
    Candidate,
    CandidateMode,
    Evidence,
    ObjectiveLevel,
    PairedTrial,
    UpdateKind,
)
from farfield.policy import PromotionPolicy
from farfield.portfolio import select_portfolio
from farfield.runtime import (
    ArtifactIntegrityError,
    AuthorityError,
    FarfieldRuntime,
    StateError,
)


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.runtime = FarfieldRuntime.create(
            self.root,
            "Discover externally testable mechanisms without hiding failure.",
        )
        self.objective_id = self.runtime.declare_initial_objective(
            text="Which mechanism explains the observed failure?",
            success_criteria=["A separating intervention resolves the hypothesis"],
            constraints=["equal compute"],
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _artifact(self, name: str, content: str = "evidence\n") -> str:
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path.resolve().as_uri()

    def _open_mission(self, suffix: str) -> tuple[str, tuple[str, ...]]:
        mission_id = self.runtime.export_mission(
            objective_id=self.runtime.current_objective()["id"],
            goal=f"Run bounded mission {suffix}",
            completion_evidence=["raw result", "verifier report"],
            constraints=["frozen objective"],
            budget={"wall_time_minutes": 10, "compute_units": 2},
        )
        contract = self.runtime.load_mission_contract(mission_id)
        clauses = tuple(item["id"] for item in contract["completion_clauses"])
        return mission_id, clauses

    def _record(
        self,
        mission_id: str,
        clauses: tuple[str, ...],
        suffix: str,
        *,
        sealed: bool = True,
        evidence_id: str | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if evidence_id is not None:
            kwargs["id"] = evidence_id
        return self.runtime.record_evidence(
            Evidence(
                claim=f"claim-{suffix}",
                artifact_uri=self._artifact(f"{suffix}.json", f'{{"id":"{suffix}"}}\n'),
                verifier_id="task-native-v1",
                outcome="supports",
                scope="local-test",
                mission_id=mission_id,
                clause_ids=clauses,
                sealed=sealed,
                **kwargs,
            ),
            actor="protected_vault" if sealed else "verifier",
        )

    def _settled_mission(
        self, suffix: str, *, status: str = "done", sealed: bool = True
    ) -> tuple[str, str]:
        mission_id, clauses = self._open_mission(suffix)
        evidence_id = self._record(
            mission_id, clauses, f"mission-{suffix}", sealed=sealed
        )
        self.runtime.settle_mission(
            mission_id,
            status=status,
            evidence_ids=[evidence_id],
        )
        return mission_id, evidence_id

    def _update(self, suffix: str, mission_id: str) -> str:
        return self.runtime.propose_update(
            kind=UpdateKind.SKILL,
            artifact_uri=self._artifact(f"skill-{suffix}.md", f"# Skill {suffix}\n"),
            scope="cross-project",
            rationale="paired transport candidate",
            source_mission_id=mission_id,
        )

    @staticmethod
    def _trials(effect: float = 0.10) -> list[PairedTrial]:
        return [
            PairedTrial(
                task_family=f"family-{index % 3}",
                candidate_score=0.70 + effect + index * 0.001,
                baseline_score=0.70,
                candidate_compute=10,
                baseline_compute=10,
                protected_regression=0.0,
            )
            for index in range(8)
        ]

    def _receipt(
        self,
        update_id: str,
        evidence_id: str,
        eval_id: str,
    ) -> str:
        return self.runtime.record_protected_evaluation(
            update_id=update_id,
            protected_eval_id=eval_id,
            trials=self._trials(),
            evidence_ids=[evidence_id],
            frozen_baseline=True,
            independent_verifier=True,
            rollback_ref="snapshot://baseline",
        )

    def test_policy_is_genesis_pinned(self) -> None:
        reopened = FarfieldRuntime(self.root)
        self.assertEqual(reopened.policy.digest(), self.runtime.policy.digest())
        with self.assertRaises(StateError):
            FarfieldRuntime(
                self.root,
                PromotionPolicy(
                    min_paired_trials=1,
                    min_task_families=1,
                    allow_verifier_promotion=True,
                ),
            )

    def test_lower_level_cannot_supersede_research_question(self) -> None:
        _, evidence_id = self._settled_mission("level")
        with self.assertRaises(StateError):
            self.runtime.propose_objective_revision(
                level=ObjectiveLevel.MISSION,
                text="Pretend this is only a mission",
                success_criteria=["easier result"],
                constraints=[],
                supersedes=self.objective_id,
                falsified_assumptions=["old assumption"],
                evidence_ids=[evidence_id],
                intent_invariant_argument="claimed invariant",
                independent_difficulty_change=0.0,
                removed_core_criteria=[],
                expected_standing_utility_effect="claimed neutral",
                affected_descendants=[],
                rollback_ref="objective://old",
            )

    def test_nonfinite_values_are_rejected(self) -> None:
        trial = self._trials()[0]
        with self.assertRaises(ValueError):
            PairedTrial(
                task_family=trial.task_family,
                candidate_score=math.nan,
                baseline_score=trial.baseline_score,
                candidate_compute=trial.candidate_compute,
                baseline_compute=trial.baseline_compute,
                protected_regression=trial.protected_regression,
            ).validate()
        _, evidence_id = self._settled_mission("nan")
        with self.assertRaises(ValueError):
            self.runtime.propose_objective_revision(
                level=ObjectiveLevel.RESEARCH_QUESTION,
                text="new",
                success_criteria=["criterion"],
                constraints=[],
                supersedes=self.objective_id,
                falsified_assumptions=["assumption"],
                evidence_ids=[evidence_id],
                intent_invariant_argument="same intent",
                independent_difficulty_change=math.nan,
                removed_core_criteria=[],
                expected_standing_utility_effect="unknown",
                affected_descendants=[],
                rollback_ref="objective://old",
            )

    def test_semantic_verify_rejects_raw_privileged_forgery(self) -> None:
        evil_capability = object()
        raw = EventLedger(
            self.root / ".farfield" / FarfieldRuntime.LEDGER_NAME,
            evil_capability,
        )
        raw._transact(
            evil_capability,
            lambda events: [
                (
                    "update_decided",
                    "promotion_reviewer",
                    {
                        "id": "decision_forged",
                        "update_id": "upd_forged",
                        "receipt_id": "receipt_forged",
                        "accepted": True,
                        "installed_digest": "0" * 64,
                        "reasons": [],
                        "promotion_policy_digest": self.runtime.policy.digest(),
                    },
                )
            ],
            lambda events: None,
        )
        ok, message = self.runtime.verify()
        self.assertFalse(ok)
        self.assertIn("semantic", message)

    def test_mission_mirror_tamper_is_detected(self) -> None:
        mission_id, _ = self._open_mission("tamper")
        event = [
            item
            for item in self.runtime.events("mission_exported")
            if item["payload"]["id"] == mission_id
        ][0]
        mirror = Path(event["payload"]["mirror_path"])
        payload = json.loads(mirror.read_text(encoding="utf-8"))
        payload["contract"]["goal"] = "silently easier goal"
        mirror.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ArtifactIntegrityError):
            self.runtime.load_mission_contract(mission_id)

    def test_settlement_is_mission_bound_and_reviewer_labelled(self) -> None:
        mission_one, clauses_one = self._open_mission("one")
        evidence_one = self._record(mission_one, clauses_one, "one")
        with self.assertRaises(AuthorityError):
            self.runtime.settle_mission(
                mission_one,
                status="done",
                evidence_ids=[evidence_one],
                actor="planner",
            )

        mission_two, _ = self._open_mission("two")
        with self.assertRaises(StateError):
            self.runtime.settle_mission(
                mission_two,
                status="done",
                evidence_ids=[evidence_one],
            )

    def test_duplicate_evidence_id_and_post_settlement_evidence_are_rejected(
        self,
    ) -> None:
        mission_id, clauses = self._open_mission("duplicate")
        evidence_id = "ev_fixed-duplicate"
        first = self._record(
            mission_id, clauses, "duplicate-one", evidence_id=evidence_id
        )
        with self.assertRaises(StateError):
            self._record(mission_id, clauses, "duplicate-two", evidence_id=evidence_id)
        self.runtime.settle_mission(
            mission_id,
            status="done",
            evidence_ids=[first],
        )
        with self.assertRaises(StateError):
            self._record(mission_id, clauses, "too-late")

    def test_stale_objective_cannot_export_after_revision(self) -> None:
        _, evidence_id = self._settled_mission("revision")
        proposal_id = self.runtime.propose_objective_revision(
            level=ObjectiveLevel.RESEARCH_QUESTION,
            text="Which boundary condition falsifies the mechanism?",
            success_criteria=["A separating intervention resolves the boundary"],
            constraints=["equal compute"],
            supersedes=self.objective_id,
            falsified_assumptions=["mechanism is regime invariant"],
            evidence_ids=[evidence_id],
            intent_invariant_argument="still seeks an externally testable mechanism",
            independent_difficulty_change=0.1,
            removed_core_criteria=[],
            expected_standing_utility_effect="higher information gain",
            affected_descendants=["old hypothesis"],
            rollback_ref="objective://original",
        )
        self.runtime.review_objective_revision(
            proposal_id,
            verdict="endorse",
            evidence_ids=[evidence_id],
        )
        revised_id = self.runtime.activate_objective_revision(
            proposal_id, actor="human"
        )
        self.assertEqual(self.runtime.current_objective()["id"], revised_id)
        with self.assertRaises(StateError):
            self.runtime.export_mission(
                objective_id=self.objective_id,
                goal="stale mission",
                completion_evidence=["artifact"],
                constraints=[],
                budget={"wall_time_minutes": 1},
            )

    def test_failed_mission_only_yields_scoped_negative_memory(self) -> None:
        mission_id, evidence_id = self._settled_mission("failed", status="failed")
        failure_id = self.runtime.record_failure(
            mission_id=mission_id,
            context="same model and dataset",
            intervention="increased regularization",
            observation="validation worsened",
            mechanism="underfit under this regime",
            retry_when="data regime changes",
            scope="this dataset family",
            evidence_ids=[evidence_id],
        )
        with self.assertRaises(StateError):
            self.runtime.propose_update(
                kind=UpdateKind.SKILL,
                artifact_uri=self._artifact("bad-skill.md", "bad\n"),
                scope="global",
                rationale="overgeneralized failure",
                source_mission_id=mission_id,
                source_failure_id=failure_id,
            )
        memory_id = self.runtime.propose_update(
            kind=UpdateKind.MEMORY,
            artifact_uri=self._artifact("failure.md", "scoped negative result\n"),
            scope="this dataset family",
            rationale="retain bounded negative knowledge",
            source_mission_id=mission_id,
            source_failure_id=failure_id,
        )
        self.assertTrue(memory_id.startswith("upd_"))

    def test_quarantined_bytes_are_immutable_for_decision(self) -> None:
        mission_id, evidence_id = self._settled_mission("cas")
        source = self.root / "skill-cas.md"
        source.write_text("original\n", encoding="utf-8")
        update_id = self.runtime.propose_update(
            kind=UpdateKind.SKILL,
            artifact_uri=source.resolve().as_uri(),
            scope="test",
            rationale="content binding",
            source_mission_id=mission_id,
        )
        update = [
            item
            for item in self.runtime.events("update_quarantined")
            if item["payload"]["id"] == update_id
        ][0]["payload"]
        source.write_text("replacement\n", encoding="utf-8")
        receipt_id = self._receipt(update_id, evidence_id, "pev_cas-unique")
        decision = self.runtime.evaluate_update(update_id, receipt_id)
        self.assertTrue(decision.accepted)
        decided = [
            item
            for item in self.runtime.events("update_decided")
            if item["payload"]["update_id"] == update_id
        ][0]["payload"]
        self.assertEqual(decided["installed_digest"], update["artifact_digest"])

    def test_corrupted_quarantine_is_rejected(self) -> None:
        mission_id, evidence_id = self._settled_mission("corrupt")
        update_id = self._update("corrupt", mission_id)
        update = [
            item
            for item in self.runtime.events("update_quarantined")
            if item["payload"]["id"] == update_id
        ][0]["payload"]
        path = Path(update["quarantine_path"])
        os.chmod(path, 0o600)
        path.write_text("corrupted\n", encoding="utf-8")
        with self.assertRaises(ArtifactIntegrityError):
            self._receipt(update_id, evidence_id, "pev_corrupt-unique")

    def test_protected_id_is_canonical(self) -> None:
        mission_id, evidence_id = self._settled_mission("canonical")
        update_id = self._update("canonical", mission_id)
        with self.assertRaises(ValueError):
            self._receipt(update_id, evidence_id, " pev_alias-001 ")

    def test_concurrent_same_protected_id_has_one_winner(self) -> None:
        mission_id, evidence_id = self._settled_mission("race-receipt")
        update_ids = [self._update(str(index), mission_id) for index in range(2)]
        outcomes: list[str] = []
        barrier = threading.Barrier(2)

        def worker(update_id: str) -> None:
            barrier.wait()
            try:
                self._receipt(update_id, evidence_id, "pev_shared-race")
                outcomes.append("ok")
            except StateError:
                outcomes.append("blocked")

        threads = [threading.Thread(target=worker, args=(item,)) for item in update_ids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertCountEqual(outcomes, ["ok", "blocked"])

    def test_concurrent_receipt_consumption_has_one_decision(self) -> None:
        mission_id, evidence_id = self._settled_mission("race-decision")
        update_id = self._update("decision", mission_id)
        receipt_id = self._receipt(update_id, evidence_id, "pev_decision-race")
        outcomes: list[str] = []
        barrier = threading.Barrier(2)

        def worker() -> None:
            barrier.wait()
            try:
                self.runtime.evaluate_update(update_id, receipt_id)
                outcomes.append("ok")
            except StateError:
                outcomes.append("blocked")

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertCountEqual(outcomes, ["ok", "blocked"])
        decisions = [
            item
            for item in self.runtime.events("update_decided")
            if item["payload"]["update_id"] == update_id
        ]
        self.assertEqual(len(decisions), 1)

    def test_rollback_removes_update_from_active_view(self) -> None:
        mission_id, evidence_id = self._settled_mission("rollback")
        update_id = self._update("rollback", mission_id)
        receipt_id = self._receipt(update_id, evidence_id, "pev_rollback-001")
        self.assertTrue(self.runtime.evaluate_update(update_id, receipt_id).accepted)
        self.assertIn(update_id, self.runtime.active_update_ids())
        self.runtime.rollback_update(
            update_id,
            reason="post-install regression",
            evidence_ids=[evidence_id],
        )
        self.assertNotIn(update_id, self.runtime.active_update_ids())

    def test_uninitialized_project_is_not_valid(self) -> None:
        empty = self.root / "empty"
        with self.assertRaises(StateError):
            FarfieldRuntime(empty)

    def test_deep_verification_passes_hardened_history(self) -> None:
        mission_id, evidence_id = self._settled_mission("verify")
        update_id = self._update("verify", mission_id)
        receipt_id = self._receipt(update_id, evidence_id, "pev_verify-001")
        self.runtime.evaluate_update(update_id, receipt_id)
        ok, message = self.runtime.verify()
        self.assertTrue(ok, message)
        self.assertIn("semantic replay verified", message)


class PortfolioTests(unittest.TestCase):
    def test_portfolio_preserves_falsifier_and_mechanism_diversity(self) -> None:
        candidates = [
            Candidate(
                "Fast benchmark gain",
                "Tune an existing component",
                CandidateMode.BRIDGE,
                0.4,
                0.3,
                0.4,
                0.9,
                1.0,
                "same-benchmark",
            ),
            Candidate(
                "Counterexample",
                "Try to falsify the central causal assumption",
                CandidateMode.FALSIFIER,
                0.9,
                0.8,
                1.0,
                0.5,
                1.5,
                "causal-intervention",
            ),
            Candidate(
                "Mechanism",
                "Explain an observation outside the seed literature",
                CandidateMode.MECHANISM,
                0.8,
                0.9,
                0.8,
                0.4,
                1.5,
                "external-anomaly",
            ),
        ]
        selected = select_portfolio(candidates, budget=3.0)
        modes = {item.mode for item in selected}
        self.assertIn(CandidateMode.FALSIFIER, modes)
        self.assertIn(CandidateMode.MECHANISM, modes)
        signatures = [item.independence_signature for item in selected]
        self.assertEqual(len(signatures), len(set(signatures)))


if __name__ == "__main__":
    unittest.main()
