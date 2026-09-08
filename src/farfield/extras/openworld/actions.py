"""Action / Affordance registry. Actions are choices; idea kinds are objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .state import ScientificState


SURVEY = "SURVEY"
ASK = "ASK"
OBSERVE = "OBSERVE"
ACQUIRE = "ACQUIRE"
THEORIZE = "THEORIZE"
PROBE = "PROBE"
VERIFY = "VERIFY"
VERIFY_REPLICATION = "VERIFY_REPLICATION"
VERIFY_MEASUREMENT = "VERIFY_MEASUREMENT"
VERIFY_ALTERNATIVE_EXPLANATION = "VERIFY_ALTERNATIVE_EXPLANATION"
VERIFY_WORLD_COMPATIBILITY = "VERIFY_WORLD_COMPATIBILITY"
VERIFY_NEGATIVE_RESULT = "VERIFY_NEGATIVE_RESULT"
VERIFY_HIGH_IMPACT_CLAIM = "VERIFY_HIGH_IMPACT_CLAIM"
REFLECT = "REFLECT"
SYNTHESIZE = "SYNTHESIZE"
ARCHIVE = "ARCHIVE"
FORK_LINEAGE = "FORK_LINEAGE"
EVOLVE_HARNESS = "EVOLVE_HARNESS"

ACTION_TYPES = (
    SURVEY,
    ASK,
    OBSERVE,
    ACQUIRE,
    THEORIZE,
    PROBE,
    VERIFY,
    REFLECT,
    SYNTHESIZE,
    ARCHIVE,
    FORK_LINEAGE,
    EVOLVE_HARNESS,
)


@dataclass(frozen=True)
class ActionSpec:
    action_type: str
    preconditions: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    reads: tuple[str, ...]
    possible_outputs: tuple[str, ...]
    executor: str
    validator: str
    estimated_cost: float
    risk: str
    state_effect: str
    evidence_effect: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "preconditions": list(self.preconditions),
            "required_capabilities": list(self.required_capabilities),
            "reads": list(self.reads),
            "possible_outputs": list(self.possible_outputs),
            "executor": self.executor,
            "validator": self.validator,
            "estimated_cost": self.estimated_cost,
            "risk": self.risk,
            "state_effect": self.state_effect,
            "evidence_effect": self.evidence_effect,
        }


@dataclass(frozen=True)
class ActionInstance:
    spec: ActionSpec
    target: str = ""
    reason: str = ""
    lineage_id: str = ""
    frontier_target_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def action_type(self) -> str:
        return self.spec.action_type

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.spec.to_dict(),
            "target": self.target,
            "reason": self.reason,
            "lineage_id": self.lineage_id,
            "frontier_target_id": self.frontier_target_id,
            "extra": dict(self.extra),
        }


SPECS: dict[str, ActionSpec] = {
    SURVEY: ActionSpec(
        SURVEY,
        preconditions=("open_question_or_goal",),
        required_capabilities=(),
        reads=("scientific_archive", "literature"),
        possible_outputs=("citations", "gaps"),
        executor="survey_executor",
        validator="verify_papers",
        estimated_cost=0.4,
        risk="low",
        state_effect="record_citations",
        evidence_effect="none",
    ),
    ASK: ActionSpec(
        ASK,
        preconditions=("research_goal",),
        required_capabilities=(),
        reads=("working_memory",),
        possible_outputs=("question",),
        executor="ask_executor",
        validator="question_schema",
        estimated_cost=0.2,
        risk="low",
        state_effect="add_open_question",
        evidence_effect="none",
    ),
    OBSERVE: ActionSpec(
        OBSERVE,
        preconditions=("open_question", "observable_present"),
        required_capabilities=(),
        reads=("world_version", "capability_registry"),
        possible_outputs=("QuestionEvidence",),
        executor="question_executor",
        validator="observation_attestation",
        estimated_cost=0.5,
        risk="medium",
        state_effect="maybe_resolve_question",
        evidence_effect="question_evidence_if_attested",
    ),
    ACQUIRE: ActionSpec(
        ACQUIRE,
        preconditions=("capability_or_world_gap",),
        required_capabilities=(),
        reads=("missing_capabilities", "world_lineage"),
        possible_outputs=("AcquisitionEvidence", "world_version"),
        executor="acquire_executor",
        validator="world_attestation",
        estimated_cost=0.8,
        risk="medium",
        state_effect="new_world_version",
        evidence_effect="acquisition_evidence",
    ),
    THEORIZE: ActionSpec(
        THEORIZE,
        preconditions=("unexplained_phenomenon",),
        required_capabilities=(),
        reads=("open_questions", "evidence_records"),
        possible_outputs=("theory", "predictions"),
        executor="theory_discriminator",
        validator="theory_schema",
        estimated_cost=0.3,
        risk="low",
        state_effect="add_generated_theory",
        evidence_effect="none",
    ),
    PROBE: ActionSpec(
        PROBE,
        preconditions=("causal_handle", "valid_freeze", "measure"),
        required_capabilities=(),
        reads=("world_version", "harness_probe_executor"),
        possible_outputs=("ProbeEvidence",),
        executor="probe_executor",
        validator="fair_two_arm",
        estimated_cost=1.0,
        risk="high",
        state_effect="record_probe",
        evidence_effect="probe_evidence_if_world",
    ),
    VERIFY: ActionSpec(
        VERIFY,
        preconditions=("existing_evidence",),
        required_capabilities=(),
        reads=("evidence_id", "world_id"),
        possible_outputs=("confirmation_or_gap",),
        executor="verify_executor",
        validator="host_confirmation",
        estimated_cost=0.9,
        risk="medium",
        state_effect="confirmation_gaps",
        evidence_effect="same_evidence_id_only",
    ),
    REFLECT: ActionSpec(
        REFLECT,
        preconditions=("recent_events",),
        required_capabilities=(),
        reads=("failure_archive", "working_memory"),
        possible_outputs=("pathology_or_scientific_note",),
        executor="reflect_executor",
        validator="classify_failure",
        estimated_cost=0.2,
        risk="low",
        state_effect="record_pathology_or_anomaly",
        evidence_effect="none",
    ),
    SYNTHESIZE: ActionSpec(
        SYNTHESIZE,
        preconditions=("evidence_or_theory",),
        required_capabilities=(),
        reads=("scientific_archive",),
        possible_outputs=("brief",),
        executor="synthesize_executor",
        validator="attested_writeup_lock",
        estimated_cost=0.6,
        risk="low",
        state_effect="draft_only",
        evidence_effect="none",
    ),
    ARCHIVE: ActionSpec(
        ARCHIVE,
        preconditions=("closed_or_negative_result",),
        required_capabilities=(),
        reads=("working_memory",),
        possible_outputs=("archive_row",),
        executor="archive_executor",
        validator="epistemic_label",
        estimated_cost=0.1,
        risk="low",
        state_effect="move_to_archive",
        evidence_effect="none",
    ),
    FORK_LINEAGE: ActionSpec(
        FORK_LINEAGE,
        preconditions=("stagnation_or_diversity",),
        required_capabilities=(),
        reads=("scientific_state", "harness_version"),
        possible_outputs=("branches",),
        executor="fork_executor",
        validator="generated_not_merged_as_world",
        estimated_cost=0.7,
        risk="medium",
        state_effect="open_branches",
        evidence_effect="verified_only_merge",
    ),
    EVOLVE_HARNESS: ActionSpec(
        EVOLVE_HARNESS,
        preconditions=("reproducible_harness_pathology",),
        required_capabilities=(),
        reads=("harness_archive", "failure_archive"),
        possible_outputs=("harness_version", "capability"),
        executor="harness_evolver",
        validator="credit_gate",
        estimated_cost=0.9,
        risk="high",
        state_effect="harness_version_if_admitted",
        evidence_effect="none",
    ),
}


def spec_of(action_type: str) -> ActionSpec:
    return SPECS[str(action_type)]


def _has_capability(names: Iterable[str], needed: str) -> bool:
    wanted = str(needed or "").strip()
    if not wanted:
        return True
    have = {str(item).strip() for item in names}
    return wanted in have


def _observable_present(state: ScientificState, question: Mapping[str, Any]) -> bool:
    required = str(question.get("required_observable") or "").strip()
    if required:
        blob = " ".join(
            str(row.get("observables") or row.get("id") or "")
            for row in state.world_versions
        )
        if required in blob or required in " ".join(state.available_capabilities):
            return True
        return False
    cap = str(question.get("required_capability") or "").strip()
    if cap:
        return _has_capability(state.available_capabilities, cap)
    return bool(state.world_id)


def _has_theory_for(state: ScientificState, question_id: str) -> bool:
    return any(
        str(row.get("question_id") or row.get("target") or "") == question_id
        or str(row.get("id") or "") == question_id
        for row in state.theories
    )


def _needs_verify(state: ScientificState) -> bool:
    if state.contradictions:
        return True
    seen: set[str] = set()
    for row in state.evidence_records:
        eid = str(row.get("evidence_id") or "")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        if str(row.get("epistemic") or "") != "WORLD":
            continue
        if int(row.get("verify_count") or 0) < 1:
            return True
        if row.get("high_impact") and int(row.get("source_count") or 1) <= 1:
            return True
    return False


def _verify_kind(state: ScientificState) -> str:
    if state.contradictions:
        return VERIFY_ALTERNATIVE_EXPLANATION
    for row in state.evidence_records:
        if row.get("high_impact") and int(row.get("source_count") or 1) <= 1:
            return VERIFY_HIGH_IMPACT_CLAIM
        if str(row.get("outcome") or row.get("verdict") or "") in {"weakens", "negative"}:
            return VERIFY_NEGATIVE_RESULT
    return VERIFY_REPLICATION


def _causal_ready(state: ScientificState, extra: Mapping[str, Any] | None = None) -> bool:
    row = dict(extra or {})
    if row.get("causal_handle") and row.get("measure") and (row.get("freeze") or state.world_id):
        return True
    return bool(row.get("causal_identification_ready"))


def eligible_actions(
    state: ScientificState,
    *,
    capabilities: Iterable[str] = (),
    pathology_ready: bool = False,
    stagnation: bool = False,
    probe_ready: Mapping[str, Any] | None = None,
    workers_connected: bool = False,
) -> list[ActionInstance]:
    """Deterministic eligibility. Not a pipeline stage list."""
    caps = tuple(str(item) for item in capabilities) + tuple(state.available_capabilities)
    found: list[ActionInstance] = []
    goal_id = "goal:G0" if state.goal else ""

    if (state.goal or state.open_questions) and (
        state.ticks == 0 or not state.archive_ids
    ):
        found.append(
            ActionInstance(
                SPECS[SURVEY],
                target=state.goal,
                reason="goal_or_question",
                frontier_target_id=goal_id,
            )
        )
    if state.goal and not state.open_questions and not state.resolved_questions and not state.blocked_questions:
        found.append(
            ActionInstance(
                SPECS[ASK],
                target=state.goal,
                reason="research_goal",
                frontier_target_id=goal_id,
            )
        )

    for question in list(state.open_questions) + list(state.blocked_questions):
        qid = str(question.get("id") or "")
        lineage = str(question.get("lineage_id") or "")
        required_cap = str(question.get("required_capability") or "").strip()
        if _observable_present(state, question) and (
            not required_cap or _has_capability(caps, required_cap)
        ):
            found.append(
                ActionInstance(
                    SPECS[OBSERVE],
                    target=qid,
                    reason="observable_present",
                    lineage_id=lineage,
                    frontier_target_id=qid,
                    extra={"question": dict(question)},
                )
            )
        else:
            gap = required_cap or str(question.get("required_observable") or "observable")
            found.append(
                ActionInstance(
                    SPECS[ACQUIRE],
                    target=qid,
                    reason="blocked_on_capability_or_world",
                    lineage_id=lineage,
                    frontier_target_id=qid,
                    extra={"gap": gap, "question": dict(question)},
                )
            )
        if not _has_theory_for(state, qid):
            found.append(
                ActionInstance(
                    SPECS[THEORIZE],
                    target=qid,
                    reason="unexplained_question",
                    lineage_id=lineage,
                    frontier_target_id=qid,
                )
            )

    for question in state.resolved_questions:
        qid = str(question.get("id") or "")
        if qid and not _has_theory_for(state, qid):
            found.append(
                ActionInstance(
                    SPECS[THEORIZE],
                    target=qid,
                    reason="discriminate_resolved_phenomenon",
                    lineage_id=str(question.get("lineage_id") or ""),
                    frontier_target_id=qid,
                )
            )

    if state.missing_capabilities and not any(item.action_type == ACQUIRE for item in found):
        found.append(
            ActionInstance(
                SPECS[ACQUIRE],
                target=state.missing_capabilities[0],
                reason="capability_or_world_gap",
                frontier_target_id=state.open_questions[0]["id"] if state.open_questions else goal_id,
                extra={"gap": state.missing_capabilities[0]},
            )
        )

    if (state.open_questions or state.anomalies) and not any(
        item.action_type == THEORIZE for item in found
    ):
        found.append(
            ActionInstance(
                SPECS[THEORIZE],
                reason="unexplained_phenomenon",
                frontier_target_id=goal_id,
            )
        )

    if _causal_ready(state, probe_ready):
        found.append(
            ActionInstance(
                SPECS[PROBE],
                target=str((probe_ready or {}).get("hypothesis") or ""),
                reason="causal_handle_and_freeze",
                frontier_target_id=str((probe_ready or {}).get("question_id") or goal_id),
                extra=dict(probe_ready or {}),
            )
        )

    if _needs_verify(state):
        found.append(
            ActionInstance(
                SPECS[VERIFY],
                reason="existing_evidence",
                frontier_target_id=goal_id or (
                    str(state.evidence_records[-1].get("question_id") or "")
                    if state.evidence_records
                    else ""
                ),
                extra={"verify_kind": _verify_kind(state)},
            )
        )

    if state.ticks > 0:
        found.append(ActionInstance(SPECS[REFLECT], reason="recent_events"))

    last_synth = int((state.metrics or {}).get("last_synthesis_events") or 0)
    meaningful = int((state.metrics or {}).get("meaningful_events") or 0)
    if (state.evidence_records or state.theories) and (
        meaningful - last_synth >= 5 or state.contradictions or state.ticks == 1
    ):
        found.append(ActionInstance(SPECS[SYNTHESIZE], reason="evidence_or_theory"))

    if state.resolved_questions or state.failed_designs:
        found.append(ActionInstance(SPECS[ARCHIVE], reason="closed_or_negative_result"))

    if stagnation and len(state.active_lineages) < 4:
        found.append(
            ActionInstance(
                SPECS[FORK_LINEAGE],
                reason="stagnation_or_diversity",
            )
        )

    if pathology_ready:
        found.append(
            ActionInstance(
                SPECS[EVOLVE_HARNESS],
                reason="reproducible_harness_pathology",
            )
        )

    if workers_connected:
        found = worker_actions(state, found)
    return found


def worker_actions(state: ScientificState, actions: list[ActionInstance]) -> list[ActionInstance]:
    """Readiness from persisted scientific artifacts, not a second pipeline."""
    surveys = [r for r in state.artifacts if r.get("kind") == "literature_survey" and r.get("works")]
    # The control-plane observer is still useful, but cannot substitute for survey.
    found = [a for a in actions if a.action_type not in {SURVEY, THEORIZE, PROBE, VERIFY, SYNTHESIZE}]
    found = [a for a in found if a.action_type != OBSERVE or not any(
        r.get("role") == "QuestionEvidence" and r.get("question_id") == a.target
        and r.get("world_id") == state.world_id for r in state.evidence_records)]
    for row in state.evidence_records:
        if row.get("epistemic") == "WORLD" and not row.get("verify_count"):
            found.append(ActionInstance(SPECS[VERIFY], target=str(row.get("evidence_id") or ""),
                frontier_target_id=str(row.get("question_id") or "goal:G0"),
                reason="check_exact_evidence_identity", extra={"verify_kind": VERIFY_REPLICATION,
                                                               "evidence_role": row.get("role")}))
        elif row.get("role") == "ProbeEvidence" and row.get("reproduction_ok") and not row.get("replication_of"):
            for world in state.world_versions:
                wid = str(world.get("id") or world.get("world_id") or "")
                if world.get("research_role") != "heldout" or wid == row.get("world_id"):
                    continue
                if any(r.get("replication_of") == row.get("evidence_id") and r.get("world_id") == wid for r in state.evidence_records):
                    continue
                found.append(ActionInstance(SPECS[VERIFY], target=str(row.get("evidence_id")),
                    frontier_target_id=str(row.get("question_id") or "goal:G0"), reason="heldout_replication_debt",
                    extra={"replication_world_id": wid, "evidence_role": "ProbeEvidence"}))
    if not surveys:
        questions = state.open_questions + state.blocked_questions
        target = str(questions[0].get("id")) if questions else "goal:G0"
        found.append(ActionInstance(SPECS[SURVEY], target=target,
                                   frontier_target_id=target, reason="literature_gap",
                                   extra={"extract_trajectories": True}))
        return unattempted_actions(state, found)
    questions = state.open_questions + state.blocked_questions + state.resolved_questions
    for question in questions:
        qid = str(question.get("id") or "")
        if not any(r.get("question_id") in {"", qid} for r in surveys):
            found.append(ActionInstance(SPECS[SURVEY], target=qid, frontier_target_id=qid,
                reason="question_literature_gap", extra={"extract_trajectories": True}))
            continue
        theories = [t for t in state.theories if t.get("question_id") == qid]
        for lane in ("trajectory_supported", "assumption_reframe", "unconstrained"):
            if not any(t.get("research_lane") == lane and int(t.get("hop") or 1) == 1 for t in theories):
                found.append(ActionInstance(SPECS[THEORIZE], target=qid, frontier_target_id=qid,
                    lineage_id=str(question.get("lineage_id") or ""), reason="first_hop_branch_gap",
                    extra={"research_lane": lane, "hop": 1}))
        for theory in theories:
            if theory.get("role") == "competing_explanation":
                continue
            tid = str(theory.get("id") or "")
            card = theory.get("card") or {}
            measured = any(r.get("theory_id") == tid for r in state.evidence_records)
            attempted = any(r.get("theory_id") == tid and r.get("action_type") == PROBE
                            for r in state.failed_designs)
            literature_checked = any(r.get("theory_id") == tid for r in surveys)
            if card and not literature_checked:
                found.append(ActionInstance(SPECS[SURVEY], target=qid, frontier_target_id=qid,
                    reason="hypothesis_literature_check", extra={"theory_id": tid, "claim": theory.get("claim") or card.get("claim")}))
            if card and literature_checked and not measured and not attempted:
                found.append(ActionInstance(SPECS[PROBE], target=tid, frontier_target_id=qid,
                    reason="discriminate_competing_hypotheses" if str(card.get("idea_kind") or "probe").lower() == "probe"
                           else "check_missing_probe_compilation", extra={
                        "theory_id": tid, "question_id": qid, "card": card,
                        "causal_handle": card.get("world_lever") or theory.get("mechanism"),
                        "measure": card.get("world_observable") or card.get("prediction"),
                        "freeze": state.world_id,
                        "experiment_proposal": theory.get("experiment_proposal"),
                    }))
            if measured:
                found.append(ActionInstance(SPECS[SYNTHESIZE], target=tid, frontier_target_id=qid,
                                           reason="synthesize_measured_branch"))
            evidence = [r for r in state.evidence_records if r.get("theory_id") == tid
                        and r.get("epistemic") == "WORLD" and r.get("attested")
                        and r.get("execution_status") == "ran" and r.get("world_digest")
                        and r.get("experiment_digest")]
            failures = [r for r in state.failed_designs if r.get("theory_id") == tid]
            has_child = any(t.get("parent_theory_id") == tid for t in theories)
            # A blocked method can be reframed once; it is never called an evidence hop.
            pivot = failures and int(theory.get("failure_pivots") or 0) < 1
            if not has_child and (evidence or pivot):
                found.append(ActionInstance(SPECS[THEORIZE], target=qid, frontier_target_id=qid,
                    lineage_id=str(question.get("lineage_id") or ""),
                    reason="evidence_driven_transition" if evidence else "failure_driven_reframe",
                    extra={"research_lane": "trajectory_supported" if evidence else "assumption_reframe",
                           "hop": int(theory.get("hop") or 1) + 1, "parent_theory_id": tid,
                           "transition_kind": "evidence_driven" if evidence else "failure_reframe",
                           "evidence_context": evidence, "failure_context": failures,
                           "failure_pivots": int(theory.get("failure_pivots") or 0) + (0 if evidence else 1)}))
    return unattempted_actions(state, found)


def action_input_key(state: ScientificState, action: ActionInstance) -> str:
    """A retry needs changed scientific inputs, not just a later clock tick."""
    import hashlib
    import json
    inputs = {"goal": state.goal, "worlds": state.world_versions,
              "evidence": [(r.get("evidence_id"), r.get("verify_count")) for r in state.evidence_records],
              "theories": [r.get("id") for r in state.theories],
              "surveys": [r.get("id") for r in state.artifacts if r.get("kind") == "literature_survey"],
              "capabilities": state.available_capabilities,
              "service_session": next((r.get("id") for r in reversed(state.artifacts) if r.get("kind") == "research_service_session"), ""),
              "action": action.action_type, "target": action.target,
              "branch": {k: action.extra.get(k) for k in ("research_lane", "parent_theory_id", "hop", "theory_id", "replication_world_id")}}
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()


def unattempted_actions(state: ScientificState, actions: list[ActionInstance]) -> list[ActionInstance]:
    attempted = {r.get("input_key") for r in state.artifacts if r.get("kind") == "action_attempt"}
    return [a for a in actions if action_input_key(state, a) not in attempted]


def types_of(actions: Iterable[ActionInstance]) -> tuple[str, ...]:
    return tuple(item.action_type for item in actions)


ACTION_COMPLETENESS: dict[str, dict[str, bool]] = {
    SURVEY: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    ASK: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    OBSERVE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": True,
        "reducer": True,
        "regression_test": True,
    },
    ACQUIRE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": True,
        "reducer": True,
        "regression_test": True,
    },
    THEORIZE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": True,
        "reducer": True,
        "regression_test": True,
    },
    PROBE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": True,
        "reducer": True,
        "regression_test": True,
    },
    VERIFY: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": True,
        "reducer": True,
        "regression_test": True,
    },
    REFLECT: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    SYNTHESIZE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    ARCHIVE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": False,
    },
    FORK_LINEAGE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    EVOLVE_HARNESS: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
}
