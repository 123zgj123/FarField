"""Hardened Farfield alpha control plane.

The implementation enforces transactional state transitions and content binding
inside one local process/filesystem trust domain. Actor names remain labels until
a deployment binds them to authenticated principals and isolated services.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, cast
from urllib.parse import unquote, urlparse

from .ledger import Event, EventDraft, EventLedger, canonical_json, content_digest
from .models import (
    Evidence,
    ObjectiveLevel,
    PairedTrial,
    PromotionEvidence,
    UpdateKind,
    new_id,
    require_canonical_id,
    require_finite_number,
)
from .policy import PromotionDecision, PromotionPolicy


class AuthorityError(PermissionError):
    pass


class StateError(RuntimeError):
    pass


class ArtifactIntegrityError(RuntimeError):
    pass


@dataclass
class ReplayState:
    initialized: bool = False
    standing_intent: str | None = None
    policy: PromotionPolicy | None = None
    policy_digest: str | None = None
    ids: set[str] = field(default_factory=set)
    objectives: dict[str, dict[str, Any]] = field(default_factory=dict)
    objective_head: str | None = None
    revision_proposals: dict[str, dict[str, Any]] = field(default_factory=dict)
    revision_proposers: dict[str, str] = field(default_factory=dict)
    revision_reviews: dict[str, dict[str, Any]] = field(default_factory=dict)
    activated_revisions: set[str] = field(default_factory=set)
    missions: dict[str, dict[str, Any]] = field(default_factory=dict)
    mission_actors: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_actors: dict[str, str] = field(default_factory=dict)
    settlements: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: dict[str, dict[str, Any]] = field(default_factory=dict)
    updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    update_proposers: dict[str, str] = field(default_factory=dict)
    evaluation_receipts: dict[str, dict[str, Any]] = field(default_factory=dict)
    protected_eval_ids: set[str] = field(default_factory=set)
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    consumed_receipts: set[str] = field(default_factory=set)
    rolled_back_updates: set[str] = field(default_factory=set)


Planner = Callable[[tuple[Event, ...], ReplayState], Sequence[EventDraft]]


class FarfieldRuntime:
    LEDGER_NAME = "farfield.events.jsonl"
    SCHEMA_VERSION = 2
    ROLE_LABELS = {
        "human",
        "manager",
        "planner",
        "engineer",
        "reviewer",
        "verifier",
        "protected_vault",
        "updater",
        "promotion_reviewer",
    }

    def __init__(
        self,
        project_dir: Path | str,
        policy: PromotionPolicy | None = None,
        *,
        _allow_empty: bool = False,
    ):
        self.project_dir = Path(project_dir).resolve()
        self._control_dir = self.project_dir / ".farfield"
        self._control_dir.mkdir(parents=True, exist_ok=True)
        self.__writer_capability = object()
        self._ledger_store = EventLedger(
            self._control_dir / self.LEDGER_NAME, self.__writer_capability
        )

        events = self._ledger_store.read()
        if not events:
            if not _allow_empty:
                raise StateError(
                    "project is not initialized; call FarfieldRuntime.create"
                )
            self.policy = policy or PromotionPolicy()
            self.policy.validate()
            return

        ok, message = self._ledger_store.verify_hash_chain()
        if not ok:
            raise StateError(message)
        state = self._replay(tuple(events))
        if state.policy is None:
            raise StateError("initialized project has no pinned policy")
        if policy is not None and policy.digest() != state.policy_digest:
            raise StateError(
                "supplied promotion policy differs from the genesis-pinned policy; "
                "use an owner-gated system migration"
            )
        self.policy = state.policy

    @classmethod
    def create(
        cls,
        project_dir: Path | str,
        standing_intent: str,
        actor: str = "human",
        policy: PromotionPolicy | None = None,
    ) -> "FarfieldRuntime":
        if actor != "human":
            raise AuthorityError("only actor label 'human' may declare standing intent")
        if not standing_intent.strip():
            raise ValueError("standing_intent must be non-empty")
        pinned_policy = policy or PromotionPolicy()
        pinned_policy.validate()
        runtime = cls(project_dir, pinned_policy, _allow_empty=True)
        project_id = new_id("project")

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            if events or state.initialized:
                raise StateError("project already has a Farfield genesis event")
            return [
                (
                    "project_initialized",
                    actor,
                    {
                        "id": project_id,
                        "schema_version": cls.SCHEMA_VERSION,
                        "standing_intent": standing_intent,
                        "promotion_policy": pinned_policy.to_dict(),
                        "promotion_policy_digest": pinned_policy.digest(),
                        "actor_labels_are_authenticated": False,
                    },
                )
            ]

        runtime._transact(plan)
        runtime.policy = pinned_policy
        return runtime

    def standing_intent(self) -> str:
        state = self._state()
        if state.standing_intent is None:
            raise StateError("standing intent missing")
        return state.standing_intent

    def current_objective(self) -> dict[str, Any]:
        state = self._state()
        if state.objective_head is None:
            raise StateError("no active objective")
        return dict(state.objectives[state.objective_head])

    def events(self, event_type: str | None = None) -> list[Event]:
        return list(self._ledger_store.find(event_type))

    def declare_initial_objective(
        self,
        *,
        text: str,
        success_criteria: list[str],
        constraints: list[str],
        actor: str = "human",
    ) -> str:
        self._require_actor(actor, {"human"})
        objective_id = new_id("obj")
        payload = {
            "id": objective_id,
            "level": ObjectiveLevel.RESEARCH_QUESTION.value,
            "text": self._nonempty(text, "text"),
            "success_criteria": self._nonempty_list(
                success_criteria, "success_criteria"
            ),
            "constraints": self._string_list(constraints, "constraints"),
            "supersedes": None,
            "revision_proposal_id": None,
        }

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            if state.objective_head is not None:
                raise StateError("an objective is already active")
            return [("objective_activated", actor, payload)]

        self._transact(plan)
        return objective_id

    def propose_objective_revision(
        self,
        *,
        level: ObjectiveLevel,
        text: str,
        success_criteria: list[str],
        constraints: list[str],
        supersedes: str,
        falsified_assumptions: list[str],
        evidence_ids: list[str],
        intent_invariant_argument: str,
        independent_difficulty_change: float,
        removed_core_criteria: list[str],
        expected_standing_utility_effect: str,
        affected_descendants: list[str],
        rollback_ref: str,
        actor: str = "manager",
    ) -> str:
        self._require_actor(actor, {"manager", "human"})
        difficulty = require_finite_number(
            independent_difficulty_change, "independent_difficulty_change"
        )
        proposal_id = new_id("orev")
        require_canonical_id(supersedes, "supersedes")
        self._unique_ids(evidence_ids, "evidence_ids")
        payload = {
            "id": proposal_id,
            "level": level.value,
            "text": self._nonempty(text, "text"),
            "success_criteria": self._nonempty_list(
                success_criteria, "success_criteria"
            ),
            "constraints": self._string_list(constraints, "constraints"),
            "supersedes": supersedes,
            "falsified_assumptions": self._nonempty_list(
                falsified_assumptions, "falsified_assumptions"
            ),
            "evidence_ids": evidence_ids,
            "intent_invariant_argument": self._nonempty(
                intent_invariant_argument, "intent_invariant_argument"
            ),
            "independent_difficulty_change": difficulty,
            "removed_core_criteria": self._string_list(
                removed_core_criteria, "removed_core_criteria"
            ),
            "expected_standing_utility_effect": self._nonempty(
                expected_standing_utility_effect,
                "expected_standing_utility_effect",
            ),
            "affected_descendants": self._string_list(
                affected_descendants, "affected_descendants"
            ),
            "rollback_ref": self._nonempty(rollback_ref, "rollback_ref"),
        }

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            if state.objective_head != supersedes:
                raise StateError(
                    "revision must supersede the unique current objective head"
                )
            current = state.objectives[supersedes]
            if current["level"] != level.value:
                raise StateError(
                    "objective revisions are same-level versions; lower-level objects "
                    "must be created as children, not used to supersede an ancestor"
                )
            self._require_known_evidence(state, evidence_ids)
            return [("objective_revision_proposed", actor, payload)]

        self._transact(plan)
        return proposal_id

    def review_objective_revision(
        self,
        proposal_id: str,
        *,
        verdict: str,
        evidence_ids: list[str],
        actor: str = "reviewer",
    ) -> None:
        self._require_actor(actor, {"reviewer"})
        require_canonical_id(proposal_id, "proposal_id")
        if verdict not in {"endorse", "reject"}:
            raise ValueError("verdict must be endorse or reject")
        self._unique_ids(evidence_ids, "evidence_ids")

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            if proposal_id not in state.revision_proposals:
                raise StateError(f"unknown revision proposal: {proposal_id}")
            if proposal_id in state.revision_reviews:
                raise StateError("revision proposal already has a final review")
            if state.revision_proposers[proposal_id] == actor:
                raise AuthorityError(
                    "objective proposer cannot review the same revision"
                )
            self._require_known_evidence(state, evidence_ids)
            return [
                (
                    "objective_revision_reviewed",
                    actor,
                    {
                        "proposal_id": proposal_id,
                        "verdict": verdict,
                        "evidence_ids": evidence_ids,
                    },
                )
            ]

        self._transact(plan)

    def activate_objective_revision(
        self,
        proposal_id: str,
        *,
        actor: str,
    ) -> str:
        require_canonical_id(proposal_id, "proposal_id")
        objective_id = new_id("obj")

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            proposal = state.revision_proposals.get(proposal_id)
            if proposal is None:
                raise StateError(f"unknown revision proposal: {proposal_id}")
            if proposal_id in state.activated_revisions:
                raise StateError("objective revision has already been activated")
            review = state.revision_reviews.get(proposal_id)
            if review is None or review["verdict"] != "endorse":
                raise StateError("independent review must endorse the revision")
            if state.objective_head != proposal["supersedes"]:
                raise StateError("revision is stale because the objective head changed")
            old_level = state.objectives[proposal["supersedes"]]["level"]
            if old_level != proposal["level"]:
                raise StateError("revision cannot change semantic objective level")
            drift_sensitive = proposal["independent_difficulty_change"] < 0 or bool(
                proposal["removed_core_criteria"]
            )
            required_actor = (
                "human"
                if old_level == ObjectiveLevel.RESEARCH_QUESTION.value
                or drift_sensitive
                else "manager"
            )
            self._require_actor(actor, {required_actor})
            return [
                (
                    "objective_activated",
                    actor,
                    {
                        "id": objective_id,
                        "level": proposal["level"],
                        "text": proposal["text"],
                        "success_criteria": proposal["success_criteria"],
                        "constraints": proposal["constraints"],
                        "supersedes": proposal["supersedes"],
                        "revision_proposal_id": proposal_id,
                    },
                )
            ]

        self._transact(plan)
        return objective_id

    def export_mission(
        self,
        *,
        objective_id: str,
        goal: str,
        completion_evidence: list[str],
        constraints: list[str],
        budget: dict[str, float],
        actor: str = "planner",
    ) -> str:
        self._require_actor(actor, {"planner"})
        require_canonical_id(objective_id, "objective_id")
        mission_id = new_id("mission")
        criteria = self._nonempty_list(completion_evidence, "completion_evidence")
        clauses = [
            {
                "id": f"clause_{mission_id.removeprefix('mission_')}_{index}",
                "description": description,
            }
            for index, description in enumerate(criteria, start=1)
        ]
        normalized_budget = self._budget(budget)

        # The contract is constructed inside the transaction so it freezes the
        # current objective version under the same lock as the export event.
        created_contract: dict[str, Any] = {}

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            if state.objective_head != objective_id:
                raise StateError(
                    "missions may only use the unique current objective head"
                )
            objective = dict(state.objectives[objective_id])
            contract = {
                "mission_id": mission_id,
                "standing_intent": state.standing_intent,
                "frozen_objective": objective,
                "goal": self._nonempty(goal, "goal"),
                "completion_clauses": clauses,
                "constraints": self._string_list(constraints, "constraints"),
                "budget": normalized_budget,
                "mutation_authority": "execution_plan_only",
                "on_objective_doubt": (
                    "return evidence and an objective challenge; do not redefine success"
                ),
            }
            digest = content_digest(contract)
            created_contract.update(contract)
            return [
                (
                    "mission_exported",
                    actor,
                    {
                        "id": mission_id,
                        "objective_id": objective_id,
                        "contract": contract,
                        "contract_digest": digest,
                        "mirror_path": str(self._mission_path(mission_id)),
                    },
                )
            ]

        self._transact(plan)
        self._write_mission_mirror(mission_id, created_contract)
        return mission_id

    def load_mission_contract(self, mission_id: str) -> dict[str, Any]:
        require_canonical_id(mission_id, "mission_id")
        state = self._state()
        mission = state.missions.get(mission_id)
        if mission is None:
            raise StateError(f"unknown mission: {mission_id}")
        contract = mission["contract"]
        if content_digest(contract) != mission["contract_digest"]:
            raise ArtifactIntegrityError(
                "authoritative mission contract digest mismatch"
            )
        mirror = self._mission_path(mission_id)
        if not mirror.is_file():
            self._write_mission_mirror(mission_id, contract)
        mirrored = self._strict_json_file(mirror)
        if mirrored.get("contract_digest") != mission["contract_digest"]:
            raise ArtifactIntegrityError("mission mirror digest header changed")
        if mirrored.get("contract") != contract:
            raise ArtifactIntegrityError("mission mirror content changed after export")
        return cast(dict[str, Any], json.loads(canonical_json(contract)))

    def record_evidence(self, evidence: Evidence, actor: str = "verifier") -> str:
        self._require_actor(actor, {"verifier", "protected_vault"})
        evidence.validate()
        if evidence.sealed and actor != "protected_vault":
            raise AuthorityError(
                "sealed evidence requires actor label 'protected_vault'"
            )
        stored = self._store_file_uri(evidence.artifact_uri, "evidence")
        payload = {
            **evidence.to_dict(),
            "artifact_digest": stored["digest"],
            "artifact_vault_path": stored["path"],
            "fingerprint_basis": "content",
        }

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            if evidence.id in state.ids:
                raise StateError(f"duplicate record ID: {evidence.id}")
            mission = state.missions.get(evidence.mission_id)
            if mission is None:
                raise StateError(f"unknown mission: {evidence.mission_id}")
            if evidence.mission_id in state.settlements:
                raise StateError(
                    "evidence cannot be added after final mission settlement"
                )
            required = {
                clause["id"] for clause in mission["contract"]["completion_clauses"]
            }
            unknown = set(evidence.clause_ids) - required
            if unknown:
                raise StateError(
                    f"evidence references unknown clause IDs: {sorted(unknown)}"
                )
            return [("evidence_recorded", actor, payload)]

        self._transact(plan)
        return evidence.id

    def settle_mission(
        self,
        mission_id: str,
        *,
        status: str,
        evidence_ids: list[str],
        actor: str = "reviewer",
    ) -> None:
        self._require_actor(actor, {"reviewer"})
        require_canonical_id(mission_id, "mission_id")
        self._unique_ids(evidence_ids, "evidence_ids")
        if status not in {"done", "failed", "blocked"}:
            raise ValueError("status must be done, failed, or blocked")

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            mission = state.missions.get(mission_id)
            if mission is None:
                raise StateError(f"unknown mission: {mission_id}")
            if mission_id in state.settlements:
                raise StateError("mission has already reached final settlement")
            if state.mission_actors[mission_id] == actor:
                raise AuthorityError("mission planner cannot settle its own mission")
            records = self._require_known_evidence(state, evidence_ids)
            if any(record["mission_id"] != mission_id for record in records):
                raise StateError("settlement evidence must be bound to this mission")
            for record in records:
                self._verify_vault_blob(
                    record["artifact_vault_path"], record["artifact_digest"]
                )
            covered = {
                clause_id for record in records for clause_id in record["clause_ids"]
            }
            required = {
                clause["id"] for clause in mission["contract"]["completion_clauses"]
            }
            if status == "done" and covered != required:
                raise StateError(
                    "done settlement must cover every frozen completion clause exactly"
                )
            receipt_body = {
                "mission_id": mission_id,
                "mission_digest": mission["contract_digest"],
                "status": status,
                "evidence_ids": evidence_ids,
                "covered_clause_ids": sorted(covered),
                "artifact_digests": sorted(
                    {record["artifact_digest"] for record in records}
                ),
            }
            return [
                (
                    "mission_settled",
                    actor,
                    {
                        "id": mission_id,
                        **receipt_body,
                        "settlement_digest": content_digest(receipt_body),
                        "final": True,
                    },
                )
            ]

        self._transact(plan)

    def record_failure(
        self,
        *,
        mission_id: str,
        context: str,
        intervention: str,
        observation: str,
        mechanism: str,
        retry_when: str,
        scope: str,
        evidence_ids: list[str],
        actor: str = "engineer",
    ) -> str:
        self._require_actor(actor, {"engineer", "reviewer"})
        require_canonical_id(mission_id, "mission_id")
        fields = (context, intervention, observation, mechanism, retry_when, scope)
        if any(not value.strip() for value in fields):
            raise ValueError("all failure-capsule fields must be non-empty")
        self._unique_ids(evidence_ids, "evidence_ids")
        failure_id = new_id("fail")

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            settlement = state.settlements.get(mission_id)
            if settlement is None:
                raise StateError("failure capsules require final mission settlement")
            records = self._require_known_evidence(state, evidence_ids)
            if any(record["mission_id"] != mission_id for record in records):
                raise StateError("failure evidence must be mission-bound")
            return [
                (
                    "failure_recorded",
                    actor,
                    {
                        "id": failure_id,
                        "mission_id": mission_id,
                        "context": context,
                        "intervention": intervention,
                        "observation": observation,
                        "mechanism": mechanism,
                        "retry_when": retry_when,
                        "scope": scope,
                        "evidence_ids": evidence_ids,
                    },
                )
            ]

        self._transact(plan)
        return failure_id

    def propose_update(
        self,
        *,
        kind: UpdateKind,
        artifact_uri: str,
        scope: str,
        rationale: str,
        source_mission_id: str,
        source_failure_id: str | None = None,
        actor: str = "updater",
    ) -> str:
        self._require_actor(actor, {"updater"})
        require_canonical_id(source_mission_id, "source_mission_id")
        if source_failure_id is not None:
            require_canonical_id(source_failure_id, "source_failure_id")
        stored = self._store_file_uri(artifact_uri, "quarantine")
        update_id = new_id("upd")
        payload = {
            "id": update_id,
            "kind": kind.value,
            "source_uri": artifact_uri,
            "artifact_digest": stored["digest"],
            "quarantine_path": stored["path"],
            "scope": self._nonempty(scope, "scope"),
            "rationale": self._nonempty(rationale, "rationale"),
            "source_mission_id": source_mission_id,
            "source_failure_id": source_failure_id,
        }

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            settlement = state.settlements.get(source_mission_id)
            if settlement is None:
                raise StateError("update extraction requires final mission settlement")
            if settlement["status"] != "done":
                if kind is not UpdateKind.MEMORY:
                    raise StateError(
                        "failed/blocked missions may only yield scoped MEMORY candidates"
                    )
                failure = state.failures.get(source_failure_id or "")
                if failure is None or failure["mission_id"] != source_mission_id:
                    raise StateError(
                        "negative-memory extraction requires a matching failure capsule"
                    )
            self._verify_vault_blob(stored["path"], stored["digest"])
            return [("update_quarantined", actor, payload)]

        self._transact(plan)
        return update_id

    def record_protected_evaluation(
        self,
        *,
        update_id: str,
        protected_eval_id: str,
        trials: list[PairedTrial],
        evidence_ids: list[str],
        frozen_baseline: bool,
        independent_verifier: bool,
        rollback_ref: str,
        actor: str = "protected_vault",
    ) -> str:
        """Ingest one local protected-evaluation receipt.

        Statistics are derived from immutable trial rows rather than accepted as
        caller-supplied summary fields. The alpha still lacks an external vault;
        ``protected_vault`` is an unauthenticated actor label.
        """

        self._require_actor(actor, {"protected_vault"})
        require_canonical_id(update_id, "update_id")
        require_canonical_id(protected_eval_id, "protected_eval_id")
        self._unique_ids(evidence_ids, "evidence_ids")
        if not trials:
            raise ValueError("protected evaluation requires paired trials")
        for trial in trials:
            trial.validate()
        receipt_id = new_id("receipt")

        differences = [trial.candidate_score - trial.baseline_score for trial in trials]
        mean_difference = statistics.mean(differences)
        standard_error = (
            statistics.stdev(differences) / math.sqrt(len(differences))
            if len(differences) > 1
            else 0.0
        )
        effect_lower_bound = mean_difference - 1.96 * standard_error
        promotion = PromotionEvidence(
            paired_trials=len(trials),
            distinct_task_families=len({trial.task_family for trial in trials}),
            effect_lower_bound=effect_lower_bound,
            worst_regression=max(trial.protected_regression for trial in trials),
            matched_compute=all(
                math.isclose(
                    trial.candidate_compute,
                    trial.baseline_compute,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for trial in trials
            ),
            frozen_baseline=frozen_baseline,
            sealed_holdout=True,
            independent_verifier=independent_verifier,
            rollback_ref=self._nonempty(rollback_ref, "rollback_ref"),
            protected_eval_id=protected_eval_id,
            evidence_ids=tuple(evidence_ids),
        )
        promotion.validate()

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            update = state.updates.get(update_id)
            if update is None:
                raise StateError(f"unknown update: {update_id}")
            if update_id in state.decisions:
                raise StateError("update already has a final promotion decision")
            if protected_eval_id in state.protected_eval_ids:
                raise StateError("protected evaluation ID has already been registered")
            records = self._require_known_evidence(state, evidence_ids)
            if any(
                not record["sealed"]
                or state.evidence_actors[record["id"]] != "protected_vault"
                for record in records
            ):
                raise StateError(
                    "protected evaluation receipts require sealed vault-labelled evidence"
                )
            self._verify_vault_blob(
                update["quarantine_path"], update["artifact_digest"]
            )
            receipt_body = {
                "id": receipt_id,
                "update_id": update_id,
                "update_digest": update["artifact_digest"],
                "protected_eval_id": protected_eval_id,
                "trials": [trial.to_dict() for trial in trials],
                "promotion_evidence": promotion.to_dict(),
            }
            return [
                (
                    "protected_evaluation_recorded",
                    actor,
                    {**receipt_body, "receipt_digest": content_digest(receipt_body)},
                )
            ]

        self._transact(plan)
        return receipt_id

    def evaluate_update(
        self,
        update_id: str,
        receipt_id: str,
        *,
        actor: str = "promotion_reviewer",
    ) -> PromotionDecision:
        self._require_actor(actor, {"promotion_reviewer"})
        require_canonical_id(update_id, "update_id")
        require_canonical_id(receipt_id, "receipt_id")
        returned: list[PromotionDecision] = []

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            update = state.updates.get(update_id)
            receipt = state.evaluation_receipts.get(receipt_id)
            if update is None or receipt is None:
                raise StateError("unknown update or protected-evaluation receipt")
            if update_id in state.decisions:
                raise StateError("update version already has a final decision")
            if receipt_id in state.consumed_receipts:
                raise StateError(
                    "protected-evaluation receipt has already been consumed"
                )
            if receipt["update_id"] != update_id:
                raise StateError("receipt is bound to a different update")
            if receipt["update_digest"] != update["artifact_digest"]:
                raise ArtifactIntegrityError("receipt/update digest mismatch")
            if state.update_proposers[update_id] == actor:
                raise AuthorityError("update proposer cannot perform promotion review")
            self._verify_vault_blob(
                update["quarantine_path"], update["artifact_digest"]
            )
            promotion = PromotionEvidence.from_dict(receipt["promotion_evidence"])
            decision = self.policy.evaluate(UpdateKind(update["kind"]), promotion)
            returned.append(decision)
            # One atomic transition both consumes the receipt and records whether
            # the exact quarantined digest became active.
            return [
                (
                    "update_decided",
                    actor,
                    {
                        "id": new_id("decision"),
                        "update_id": update_id,
                        "receipt_id": receipt_id,
                        "accepted": decision.accepted,
                        "installed_digest": (
                            update["artifact_digest"] if decision.accepted else None
                        ),
                        "reasons": list(decision.reasons),
                        "promotion_policy_digest": self.policy.digest(),
                    },
                )
            ]

        self._transact(plan)
        return returned[0]

    def rollback_update(
        self,
        update_id: str,
        *,
        reason: str,
        evidence_ids: list[str],
        actor: str = "promotion_reviewer",
    ) -> None:
        self._require_actor(actor, {"promotion_reviewer", "human"})
        require_canonical_id(update_id, "update_id")
        self._unique_ids(evidence_ids, "evidence_ids")

        def plan(events: tuple[Event, ...], state: ReplayState) -> Sequence[EventDraft]:
            decision = state.decisions.get(update_id)
            if decision is None or not decision["accepted"]:
                raise StateError("only an accepted update can be rolled back")
            if update_id in state.rolled_back_updates:
                raise StateError("update has already been rolled back")
            self._require_known_evidence(state, evidence_ids)
            receipt = state.evaluation_receipts[decision["receipt_id"]]
            promotion = receipt["promotion_evidence"]
            return [
                (
                    "update_rolled_back",
                    actor,
                    {
                        "id": new_id("rollback"),
                        "update_id": update_id,
                        "reason": self._nonempty(reason, "reason"),
                        "evidence_ids": evidence_ids,
                        "restored_ref": promotion["rollback_ref"],
                    },
                )
            ]

        self._transact(plan)

    def active_update_ids(self) -> list[str]:
        state = self._state()
        return sorted(
            update_id
            for update_id, decision in state.decisions.items()
            if decision["accepted"] and update_id not in state.rolled_back_updates
        )

    def verify(self, *, deep: bool = True) -> tuple[bool, str]:
        events = self._ledger_store.read()
        if not events:
            return False, "empty ledger is not an initialized Farfield project"
        ok, message = self._ledger_store.verify_hash_chain()
        if not ok:
            return False, message
        try:
            state = self._replay(tuple(events))
            if deep:
                for mission in state.missions.values():
                    if (
                        content_digest(mission["contract"])
                        != mission["contract_digest"]
                    ):
                        raise ArtifactIntegrityError("mission contract digest mismatch")
                for evidence in state.evidence.values():
                    self._verify_vault_blob(
                        evidence["artifact_vault_path"], evidence["artifact_digest"]
                    )
                for update in state.updates.values():
                    self._verify_vault_blob(
                        update["quarantine_path"], update["artifact_digest"]
                    )
        except (ValueError, StateError, AuthorityError, ArtifactIntegrityError) as exc:
            return False, f"semantic verification failed: {exc}"
        return True, f"{message}; semantic replay verified"

    def _state(self) -> ReplayState:
        events = tuple(self._ledger_store.read())
        if not events:
            raise StateError("project is not initialized")
        return self._replay(events)

    def _transact(self, planner: Planner) -> list[Event]:
        def ledger_planner(events: tuple[Event, ...]) -> Sequence[EventDraft]:
            return planner(events, self._replay(events, allow_empty=True))

        return self._ledger_store._transact(
            self.__writer_capability, ledger_planner, self._validate_history
        )

    def _validate_history(self, events: tuple[Event, ...]) -> None:
        self._replay(events, allow_empty=True)

    def _replay(
        self, events: tuple[Event, ...], *, allow_empty: bool = False
    ) -> ReplayState:
        if not events:
            if allow_empty:
                return ReplayState()
            raise StateError("empty ledger")
        state = ReplayState()
        for event in events:
            event_type = event.get("event_type")
            actor = event.get("actor")
            payload = event.get("payload")
            if actor not in self.ROLE_LABELS or not isinstance(payload, dict):
                raise StateError("unknown actor label or malformed payload")
            handler = getattr(self, f"_replay_{event_type}", None)
            if handler is None:
                raise StateError(f"unknown or privileged raw event type: {event_type}")
            handler(state, actor, payload)
        if not state.initialized:
            raise StateError("history lacks a valid genesis event")
        return state

    def _replay_project_initialized(
        self, state: ReplayState, actor: str, payload: dict[str, Any]
    ) -> None:
        if state.initialized or actor != "human":
            raise StateError("project genesis must be unique and human-labelled")
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise StateError("unsupported schema version")
        project_id = str(payload.get("id"))
        require_canonical_id(project_id, "project.id")
        policy = PromotionPolicy.from_dict(payload.get("promotion_policy", {}))
        if payload.get("promotion_policy_digest") != policy.digest():
            raise StateError("genesis promotion-policy digest mismatch")
        if not str(payload.get("standing_intent", "")).strip():
            raise StateError("standing intent missing")
        state.initialized = True
        state.standing_intent = payload["standing_intent"]
        state.policy = policy
        state.policy_digest = policy.digest()
        state.ids.add(project_id)

    def _replay_objective_activated(
        self, state: ReplayState, actor: str, payload: dict[str, Any]
    ) -> None:
        self._require_initialized(state)
        objective_id = self._register_id(state, payload, "objective")
        level = ObjectiveLevel(payload.get("level"))
        supersedes = payload.get("supersedes")
        proposal_id = payload.get("revision_proposal_id")
        if supersedes is None:
            if state.objective_head is not None or actor != "human":
                raise StateError("initial objective must be unique and human-labelled")
            if level is not ObjectiveLevel.RESEARCH_QUESTION or proposal_id is not None:
                raise StateError("initial objective must be a research question")
        else:
            if state.objective_head != supersedes:
                raise StateError("activated revision does not supersede current head")
            proposal = state.revision_proposals.get(str(proposal_id))
            review = state.revision_reviews.get(str(proposal_id))
            if proposal is None or review is None or review["verdict"] != "endorse":
                raise StateError("activation lacks endorsed proposal")
            if str(proposal_id) in state.activated_revisions:
                raise StateError("revision proposal activated more than once")
            old_level = state.objectives[supersedes]["level"]
            if proposal["supersedes"] != supersedes or proposal["level"] != old_level:
                raise StateError("objective level/supersedes mismatch")
            drift_sensitive = proposal["independent_difficulty_change"] < 0 or bool(
                proposal["removed_core_criteria"]
            )
            required_actor = (
                "human"
                if old_level == ObjectiveLevel.RESEARCH_QUESTION.value
                or drift_sensitive
                else "manager"
            )
            if actor != required_actor:
                raise StateError("objective activation actor violates authority policy")
            state.activated_revisions.add(str(proposal_id))
        self._nonempty(str(payload.get("text", "")), "objective.text")
        self._nonempty_list(payload.get("success_criteria", []), "success_criteria")
        state.objectives[objective_id] = dict(payload)
        state.objective_head = objective_id

    def _replay_objective_revision_proposed(
        self, state: ReplayState, actor: str, payload: dict[str, Any]
    ) -> None:
        if actor not in {"manager", "human"}:
            raise StateError("invalid revision proposer role")
        proposal_id = self._register_id(state, payload, "revision proposal")
        supersedes = payload.get("supersedes")
        if state.objective_head != supersedes:
            raise StateError("revision proposal is not based on current objective head")
        current = state.objectives[str(supersedes)]
        if payload.get("level") != current["level"]:
            raise StateError("objective revision changed semantic level")
        difficulty = require_finite_number(
            cast(float, payload.get("independent_difficulty_change")),
            "independent_difficulty_change",
        )
        payload["independent_difficulty_change"] = difficulty
        self._require_evidence_refs(state, payload.get("evidence_ids", []))
        self._nonempty_list(payload.get("falsified_assumptions", []), "falsifiers")
        state.revision_proposals[proposal_id] = dict(payload)
        state.revision_proposers[proposal_id] = actor

    def _replay_objective_revision_reviewed(
        self, state: ReplayState, actor: str, payload: dict[str, Any]
    ) -> None:
        if actor != "reviewer":
            raise StateError("revision review must use reviewer label")
        proposal_id = str(payload.get("proposal_id"))
        if proposal_id not in state.revision_proposals:
            raise StateError("review references unknown proposal")
        if proposal_id in state.revision_reviews:
            raise StateError("duplicate revision review")
        if state.revision_proposers[proposal_id] == actor:
            raise StateError("revision proposer reviewed itself")
        if payload.get("verdict") not in {"endorse", "reject"}:
            raise StateError("invalid revision verdict")
        self._require_evidence_refs(state, payload.get("evidence_ids", []))
        state.revision_reviews[proposal_id] = dict(payload)

    def _replay_mission_exported(
        self, state: ReplayState, actor: str, payload: dict[str, Any]
    ) -> None:
        if actor != "planner":
            raise StateError("mission export requires planner label")
        mission_id = self._register_id(state, payload, "mission")
        if payload.get("objective_id") != state.objective_head:
            raise StateError("mission uses a stale/superseded objective")
        contract = payload.get("contract")
        if not isinstance(contract, dict) or content_digest(contract) != payload.get(
            "contract_digest"
        ):
            raise StateError("mission contract content binding failed")
        if contract.get("mission_id") != mission_id:
            raise StateError("mission ID mismatch inside contract")
        clause_ids = [item.get("id") for item in contract.get("completion_clauses", [])]
        if not clause_ids or len(clause_ids) != len(set(clause_ids)):
            raise StateError("mission completion clauses must be non-empty and unique")
        for clause_id in clause_ids:
            require_canonical_id(clause_id, "mission clause ID")
        self._budget(contract.get("budget", {}))
        state.missions[mission_id] = dict(payload)
        state.mission_actors[mission_id] = actor

    def _replay_evidence_recorded(
        self, state: ReplayState, actor: str, payload: dict[str, Any]
    ) -> None:
        if actor not in {"verifier", "protected_vault"}:
            raise StateError("invalid evidence actor label")
        evidence_id = self._register_id(state, payload, "evidence")
        mission_id = str(payload.get("mission_id"))
        mission = state.missions.get(mission_id)
        if mission is None or mission_id in state.settlements:
            raise StateError("evidence must be recorded during an open mission")
        if payload.get("sealed") and actor != "protected_vault":
            raise StateError("sealed evidence lacks protected-vault label")
        required = {item["id"] for item in mission["contract"]["completion_clauses"]}
        clauses = payload.get("clause_ids", [])
        if not clauses or not set(clauses).issubset(required):
            raise StateError("evidence clause binding is invalid")
        if payload.get("fingerprint_basis") != "content":
            raise StateError("evidence artifact is not content-bound")
        state.evidence[evidence_id] = dict(payload)
        state.evidence_actors[evidence_id] = actor

    def _replay_mission_settled(
        self, state: ReplayState, actor: str, payload: dict[str, Any]
    ) -> None:
        if actor != "reviewer":
            raise StateError("mission settlement requires reviewer label")
        mission_id = str(payload.get("id"))
        mission = state.missions.get(mission_id)
        if mission is None or mission_id in state.settlements:
            raise StateError("unknown or duplicate mission settlement")
        evidence_ids = payload.get("evidence_ids", [])
        records = self._require_evidence_refs(state, evidence_ids)
        if any(record["mission_id"] != mission_id for record in records):
            raise StateError("settlement contains cross-mission evidence")
        covered = {clause for record in records for clause in record["clause_ids"]}
        required = {item["id"] for item in mission["contract"]["completion_clauses"]}
        status = payload.get("status")
        if status not in {"done", "failed", "blocked"}:
            raise StateError("invalid settlement status")
        if status == "done" and covered != required:
            raise StateError("done settlement lacks completion-clause coverage")
        receipt_body = {
            "mission_id": mission_id,
            "mission_digest": payload.get("mission_digest"),
            "status": status,
            "evidence_ids": evidence_ids,
            "covered_clause_ids": sorted(covered),
            "artifact_digests": sorted(
                {record["artifact_digest"] for record in records}
            ),
        }
        if payload.get("mission_digest") != mission["contract_digest"]:
            raise StateError("settlement is bound to a different mission contract")
        if payload.get("settlement_digest") != content_digest(receipt_body):
            raise StateError("settlement receipt digest mismatch")
        state.settlements[mission_id] = dict(payload)

    def _replay_failure_recorded(
        self, state: ReplayState, actor: str, payload: dict[str, Any]
    ) -> None:
        if actor not in {"engineer", "reviewer"}:
            raise StateError("invalid failure recorder label")
        failure_id = self._register_id(state, payload, "failure")
        mission_id = str(payload.get("mission_id"))
        if mission_id not in state.settlements:
            raise StateError("failure capsule precedes final settlement")
        records = self._require_evidence_refs(state, payload.get("evidence_ids", []))
        if any(record["mission_id"] != mission_id for record in records):
            raise StateError("failure capsule has cross-mission evidence")
        state.failures[failure_id] = dict(payload)

    def _replay_update_quarantined(
        self, state: ReplayState, actor: str, payload: dict[str, Any]
    ) -> None:
        if actor != "updater":
            raise StateError("update proposal requires updater label")
        update_id = self._register_id(state, payload, "update")
        mission_id = str(payload.get("source_mission_id"))
        settlement = state.settlements.get(mission_id)
        if settlement is None:
            raise StateError("update precedes mission settlement")
        kind = UpdateKind(payload.get("kind"))
        if settlement["status"] != "done":
            failure_id = payload.get("source_failure_id")
            if kind is not UpdateKind.MEMORY:
                raise StateError("non-done mission yielded executable update")
            if failure_id not in state.failures:
                raise StateError("negative update lacks a failure capsule")
            if state.failures[str(failure_id)]["mission_id"] != mission_id:
                raise StateError("failure capsule belongs to another mission")
        state.updates[update_id] = dict(payload)
        state.update_proposers[update_id] = actor

    def _replay_protected_evaluation_recorded(
        self, state: ReplayState, actor: str, payload: dict[str, Any]
    ) -> None:
        if actor != "protected_vault":
            raise StateError("protected receipt lacks protected-vault label")
        receipt_id = self._register_id(state, payload, "receipt")
        update_id = str(payload.get("update_id"))
        update = state.updates.get(update_id)
        if update is None or update_id in state.decisions:
            raise StateError("receipt references unknown or decided update")
        protected_eval_id = str(payload.get("protected_eval_id"))
        require_canonical_id(protected_eval_id, "protected_eval_id")
        if protected_eval_id in state.protected_eval_ids:
            raise StateError("protected evaluation ID reused")
        if payload.get("update_digest") != update["artifact_digest"]:
            raise StateError("protected receipt update digest mismatch")
        promotion = PromotionEvidence.from_dict(payload.get("promotion_evidence", {}))
        promotion.validate()
        records = self._require_evidence_refs(state, list(promotion.evidence_ids))
        if any(
            not record["sealed"]
            or state.evidence_actors[record["id"]] != "protected_vault"
            for record in records
        ):
            raise StateError("protected receipt references non-sealed evidence")
        receipt_body = {
            "id": receipt_id,
            "update_id": update_id,
            "update_digest": payload["update_digest"],
            "protected_eval_id": protected_eval_id,
            "trials": payload.get("trials"),
            "promotion_evidence": payload["promotion_evidence"],
        }
        if payload.get("receipt_digest") != content_digest(receipt_body):
            raise StateError("protected receipt digest mismatch")
        state.evaluation_receipts[receipt_id] = dict(payload)
        state.protected_eval_ids.add(protected_eval_id)

    def _replay_update_decided(
        self, state: ReplayState, actor: str, payload: dict[str, Any]
    ) -> None:
        if actor != "promotion_reviewer":
            raise StateError("promotion decision requires reviewer label")
        self._register_id(state, payload, "decision")
        update_id = str(payload.get("update_id"))
        receipt_id = str(payload.get("receipt_id"))
        update = state.updates.get(update_id)
        receipt = state.evaluation_receipts.get(receipt_id)
        if update is None or receipt is None:
            raise StateError("decision references unknown update/receipt")
        if update_id in state.decisions or receipt_id in state.consumed_receipts:
            raise StateError("duplicate decision or receipt consumption")
        if receipt["update_id"] != update_id:
            raise StateError("decision consumed a receipt for another update")
        if payload.get("promotion_policy_digest") != state.policy_digest:
            raise StateError("decision used an unpinned promotion policy")
        assert state.policy is not None
        evidence = PromotionEvidence.from_dict(receipt["promotion_evidence"])
        expected = state.policy.evaluate(UpdateKind(update["kind"]), evidence)
        if payload.get("accepted") != expected.accepted or payload.get(
            "reasons"
        ) != list(expected.reasons):
            raise StateError("promotion decision does not match pinned policy")
        expected_digest = update["artifact_digest"] if expected.accepted else None
        if payload.get("installed_digest") != expected_digest:
            raise StateError(
                "installed digest does not match exact quarantined artifact"
            )
        state.decisions[update_id] = dict(payload)
        state.consumed_receipts.add(receipt_id)

    def _replay_update_rolled_back(
        self, state: ReplayState, actor: str, payload: dict[str, Any]
    ) -> None:
        if actor not in {"promotion_reviewer", "human"}:
            raise StateError("invalid rollback actor label")
        self._register_id(state, payload, "rollback")
        update_id = str(payload.get("update_id"))
        decision = state.decisions.get(update_id)
        if decision is None or not decision["accepted"]:
            raise StateError("rollback references a non-active update")
        if update_id in state.rolled_back_updates:
            raise StateError("duplicate rollback")
        self._require_evidence_refs(state, payload.get("evidence_ids", []))
        state.rolled_back_updates.add(update_id)

    def _mission_path(self, mission_id: str) -> Path:
        return self._control_dir / "missions" / f"{mission_id}.json"

    def _write_mission_mirror(self, mission_id: str, contract: dict[str, Any]) -> None:
        path = self._mission_path(mission_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "contract": contract,
            "contract_digest": content_digest(contract),
        }
        serialized = (
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            existing = self._strict_json_file(path)
            if existing != payload:
                raise ArtifactIntegrityError(
                    "mission mirror already exists with other bytes"
                )
            return
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

    def _store_file_uri(self, uri: str, category: str) -> dict[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError(
                "only local file:// artifacts are accepted in hardened alpha"
            )
        source = Path(unquote(parsed.path)).resolve()
        if source == self._ledger_store.path.resolve():
            raise ValueError("ledger cannot be used as an artifact")
        if not source.is_file():
            raise ValueError(f"artifact does not exist: {source}")
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        destination = self._control_dir / category / "sha256" / digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
            )
        except FileExistsError:
            self._verify_vault_blob(str(destination), digest)
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        return {"digest": digest, "path": str(destination)}

    @staticmethod
    def _verify_vault_blob(path: str, expected_digest: str) -> None:
        artifact = Path(path)
        if not artifact.is_file():
            raise ArtifactIntegrityError(
                f"content-addressed artifact missing: {artifact}"
            )
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != expected_digest or artifact.name != expected_digest:
            raise ArtifactIntegrityError("content-addressed artifact digest mismatch")

    @staticmethod
    def _strict_json_file(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=lambda item: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {item}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ArtifactIntegrityError(f"invalid strict JSON mirror: {path}") from exc
        if not isinstance(value, dict):
            raise ArtifactIntegrityError("JSON mirror is not an object")
        return value

    @staticmethod
    def _require_actor(actor: str, allowed: set[str]) -> None:
        if actor not in allowed:
            raise AuthorityError(
                f"actor label {actor!r} is not allowed; expected one of {sorted(allowed)}"
            )

    @staticmethod
    def _require_initialized(state: ReplayState) -> None:
        if not state.initialized:
            raise StateError("event precedes project genesis")

    @staticmethod
    def _register_id(
        state: ReplayState, payload: dict[str, Any], record_type: str
    ) -> str:
        if not state.initialized and record_type != "project":
            raise StateError("record precedes project genesis")
        record_id = str(payload.get("id"))
        require_canonical_id(record_id, f"{record_type}.id")
        if record_id in state.ids:
            raise StateError(f"duplicate global record ID: {record_id}")
        state.ids.add(record_id)
        return record_id

    @staticmethod
    def _require_evidence_refs(
        state: ReplayState, evidence_ids: list[str]
    ) -> list[dict[str, Any]]:
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise StateError("evidence references must be non-empty and unique")
        missing = [item for item in evidence_ids if item not in state.evidence]
        if missing:
            raise StateError(f"unknown evidence IDs: {missing}")
        return [state.evidence[item] for item in evidence_ids]

    @classmethod
    def _require_known_evidence(
        cls, state: ReplayState, evidence_ids: list[str]
    ) -> list[dict[str, Any]]:
        return cls._require_evidence_refs(state, evidence_ids)

    @staticmethod
    def _nonempty(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty")
        return value

    @classmethod
    def _nonempty_list(cls, values: list[str], name: str) -> list[str]:
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a non-empty list")
        return [cls._nonempty(value, name) for value in values]

    @classmethod
    def _string_list(cls, values: list[str], name: str) -> list[str]:
        if not isinstance(values, list):
            raise ValueError(f"{name} must be a list")
        return [cls._nonempty(value, name) for value in values]

    @staticmethod
    def _unique_ids(values: list[str], name: str) -> None:
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be non-empty and unique")
        for value in values:
            require_canonical_id(value, name)

    @staticmethod
    def _budget(budget: dict[str, float]) -> dict[str, float]:
        if not isinstance(budget, dict) or not budget:
            raise ValueError("budget must define at least one bounded resource")
        normalized: dict[str, float] = {}
        for key, value in budget.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("budget keys must be non-empty strings")
            number = require_finite_number(value, f"budget.{key}")
            if number <= 0:
                raise ValueError("budget values must be positive")
            normalized[key] = number
        return normalized
