"""Real scientific action executors. They return candidates, never write state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..lifecycle import derive_world_version, next_world_label
from ..world import WorldFixture, digest_files, load_fixture
from .worldio import is_attested, summarize
from .actions import (
    ACQUIRE,
    ARCHIVE,
    ASK,
    FORK_LINEAGE,
    OBSERVE,
    PROBE,
    REFLECT,
    SURVEY,
    SYNTHESIZE,
    THEORIZE,
    VERIFY,
    ActionInstance,
)
from .capability import run_capability
from .env import gc_harness
from .events import make_event
from .evolve import HARNESS_FAILURE
from .frontier import frontier_from_state
from .kernel import CandidateResult
from .progress import fork_trajectories


RESOLVED = "RESOLVED"
PARTIALLY_RESOLVED = "PARTIALLY_RESOLVED"
AMBIGUOUS = "AMBIGUOUS"
NO_DATA = "NO_DATA"
MEASUREMENT_INVALID = "MEASUREMENT_INVALID"
WORLD_INCOMPATIBLE = "WORLD_INCOMPATIBLE"

SNAPSHOT_FILE = "validator_snapshots.json"


@dataclass
class ObservationPlan:
    target_question: str
    required_world: str
    required_artifacts: tuple[str, ...] = ()
    observable: str = ""
    alignment: str = "validator_version"
    contrast: str = ""
    measurement: str = "compare_validator_snapshots"
    sanity_checks: tuple[str, ...] = ("aligned", "contrast")
    missingness_policy: str = "block"
    interpretation_scope: str = "observational"
    expected_failure_modes: tuple[str, ...] = (
        NO_DATA,
        MEASUREMENT_INVALID,
        WORLD_INCOMPATIBLE,
    )
    causal_scope: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_question": self.target_question,
            "required_world": self.required_world,
            "required_artifacts": list(self.required_artifacts),
            "observable": self.observable,
            "alignment": self.alignment,
            "contrast": self.contrast,
            "measurement": self.measurement,
            "sanity_checks": list(self.sanity_checks),
            "missingness_policy": self.missingness_policy,
            "interpretation_scope": self.interpretation_scope,
            "expected_failure_modes": list(self.expected_failure_modes),
            "causal_scope": self.causal_scope,
        }


@dataclass
class AcquirePlan:
    required_inputs: tuple[str, ...] = ("parent_world",)
    actions: tuple[str, ...] = ("extract_or_build_validator_snapshots",)
    tools: tuple[str, ...] = ("derive_world_version",)
    expected_artifacts: tuple[str, ...] = (
        SNAPSHOT_FILE,
        "binaries/config.json",
        "validation/outputs.json",
        "replay_interface.json",
        "provenance.json",
    )
    artifact_schema: str = "validator_snapshots/v1"
    validator: str = "snapshot_files_present"
    success_predicates: tuple[str, ...] = ("digest_differs", "artifacts_present")
    failure_predicates: tuple[str, ...] = ("no_parent_world",)
    rollback: str = "leave_parent_untouched"
    cost: float = 0.8
    timeout: float = 30.0
    provenance: str = "harvested"

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_inputs": list(self.required_inputs),
            "actions": list(self.actions),
            "tools": list(self.tools),
            "expected_artifacts": list(self.expected_artifacts),
            "artifact_schema": self.artifact_schema,
            "validator": self.validator,
            "success_predicates": list(self.success_predicates),
            "failure_predicates": list(self.failure_predicates),
            "rollback": self.rollback,
            "cost": self.cost,
            "timeout": self.timeout,
            "provenance": self.provenance,
        }


def resolve_world_root(workspace: Path | None, world_id: str) -> Path | None:
    if workspace is None or not world_id:
        return None
    dest = Path(workspace)
    for candidate in (
        dest / "worlds" / world_id,
        dest / world_id,
        dest / "data",
    ):
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return None


def fixture_from_root(root: Path, world_id: str) -> WorldFixture:
    if (root / "manifest.json").is_file():
        fixture = load_fixture(root)
        if fixture.id != world_id:
            raise ValueError("world id differs from workspace manifest")
        return fixture
    files = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    )
    digest = digest_files(root, files) if files else ""
    return WorldFixture(
        id=world_id,
        title=world_id,
        source="workspace",
        retrieved_at="2026-09-07",
        digest=digest,
        files=tuple(files),
        root=root,
        role="generated",
        provenance="generated",
        version=world_id,
    )


def load_json_artifact(root: Path | None, rel: str) -> Any | None:
    if root is None:
        return None
    path = Path(root) / rel
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def observation_plan_from(action: ActionInstance, state: Mapping[str, Any]) -> ObservationPlan:
    question = dict(action.extra.get("question") or {})
    observable = str(
        question.get("required_observable")
        or action.extra.get("observable")
        or "validator_snapshots"
    )
    artifacts = tuple(
        item
        for item in (
            SNAPSHOT_FILE,
            observable if observable.endswith(".json") else "",
        )
        if item
    )
    return ObservationPlan(
        target_question=str(question.get("id") or action.target or ""),
        required_world=str(state.get("world_id") or action.extra.get("world_id") or ""),
        required_artifacts=artifacts or (SNAPSHOT_FILE,),
        observable=observable,
        contrast=str(question.get("text") or action.target or ""),
    )


def execute_observe(env: Any, action: ActionInstance) -> CandidateResult:
    view = env.state.view()
    question = dict(action.extra.get("question") or {})
    qid = str(question.get("id") or action.target or "")
    needed = str(question.get("required_capability") or "").strip()
    plan = observation_plan_from(action, view)
    if needed and not env.capabilities.has(needed):
        env.note_harness_failure(
            where="question_observation_executor",
            why="cannot_align_versioned_records",
            example=qid,
        )
        return CandidateResult(
            status="observe_failed",
            events=[
                make_event(
                    "ActionFailed",
                    {
                        "class": HARNESS_FAILURE,
                        "where": "question_observation_executor",
                        "why": "cannot_align_versioned_records",
                        "question_id": qid,
                    },
                    action_type=OBSERVE,
                    frontier_target_id=qid,
                )
            ],
            extras={
                "class": HARNESS_FAILURE,
                "where": "question_observation_executor",
                "why": "cannot_align_versioned_records",
                "plan": plan.to_dict(),
            },
        )

    world_id = str(view.get("world_id") or "")
    world_root = resolve_world_root(env.workspace, world_id)
    fixture = fixture_from_root(world_root, world_id) if world_root else None
    if fixture is not None and plan.observable != "validator_snapshots" and not needed:
        result = summarize(fixture)
        evidence_id = "obs-" + hashlib.sha256(
            json.dumps([world_id, fixture.digest, qid, "descriptives-v1"]).encode()
        ).hexdigest()
        return CandidateResult(
            status="observed",
            evidence={
                "role": "QuestionEvidence", "question_id": qid,
                "question": question.get("text") or action.target,
                "world_id": world_id, "world_digest": fixture.digest,
                "evidence_id": evidence_id, "result": result,
                "epistemic": "WORLD" if is_attested(fixture) else "GENERATED",
                "attested": is_attested(fixture), "outcome": PARTIALLY_RESOLVED,
                "exploratory": True, "cannot_corroborate": True,
                "interpretation": "Dataset descriptives; the research question remains open.",
                "measurement": "descriptives-v1", "read_from_world": True,
                "harness_version": env.harness.version_id,
            },
            world_id=world_id, evidence_id=evidence_id,
            extras={"outcome": PARTIALLY_RESOLVED},
        )
    records = list(action.extra.get("records") or [])
    read_from_world = False
    if not records and world_root is not None:
        for rel in plan.required_artifacts:
            if fixture is not None and rel not in fixture.files:
                continue
            payload = load_json_artifact(world_root, rel)
            if isinstance(payload, Mapping) and payload.get("records"):
                records = list(payload.get("records") or [])
                read_from_world = True
                break
            if isinstance(payload, list):
                records = list(payload)
                read_from_world = True
                break

    if not records:
        if plan.missingness_policy == "block":
            return CandidateResult(
                status="blocked",
                events=[
                    make_event(
                        "ActionBlocked",
                        {
                            "reason": NO_DATA,
                            "question_id": qid,
                            "gap": plan.observable or SNAPSHOT_FILE,
                            "unlocks": ACQUIRE,
                        },
                        action_type=OBSERVE,
                        frontier_target_id=qid,
                    )
                ],
                extras={"outcome": NO_DATA, "plan": plan.to_dict(), "gap": plan.observable},
            )
        return CandidateResult(status="observe_failed", extras={"outcome": NO_DATA})

    if world_id and action.extra.get("required_world") and action.extra.get("required_world") != world_id:
        return CandidateResult(
            status="observe_failed",
            extras={"outcome": WORLD_INCOMPATIBLE, "plan": plan.to_dict()},
        )

    if needed:
        observed = run_capability(needed, records=records)
        env.capabilities.record_use(needed, success=bool(observed.get("aligned")), tick=int(view.get("ticks") or 0))
    else:
        versions = sorted(
            {
                str(row.get("validator_version") or row.get("version") or "")
                for row in records
                if isinstance(row, Mapping)
            }
            - {""}
        )
        observed = {
            "versions": versions,
            "aligned": len(versions) >= 2,
            "contrast": f"{versions[0]} vs {versions[-1]}" if len(versions) >= 2 else "",
        }

    if not isinstance(observed, Mapping):
        outcome = MEASUREMENT_INVALID
    elif not observed.get("aligned"):
        outcome = AMBIGUOUS
    elif observed.get("aligned") and observed.get("contrast"):
        outcome = RESOLVED
    else:
        outcome = PARTIALLY_RESOLVED

    attested = bool(read_from_world and fixture and is_attested(fixture))
    evidence_id = "obs-" + hashlib.sha256(json.dumps(
        [world_id, fixture.digest if fixture else "", qid, observed], sort_keys=True
    ).encode()).hexdigest()
    evidence = {
        "role": "QuestionEvidence",
        "question": question.get("text") or action.target,
        "question_id": qid,
        "contrast": observed.get("contrast") if isinstance(observed, Mapping) else "",
        "result": observed,
        "outcome": outcome,
        "interpretation": "observational contrast; not a causal claim",
        "causal_scope": "none",
        "epistemic": "WORLD" if attested else "GENERATED",
        "attested": attested,
        "world_id": world_id,
        "world_digest": fixture.digest if fixture else "",
        "evidence_id": evidence_id,
        "harness_version": env.harness.version_id,
        "capability": needed,
        "read_from_world": read_from_world,
        "plan": plan.to_dict(),
    }
    events = []
    if outcome == RESOLVED and question.get("resolution_contract") == "validator_version_contrast":
        events.append(
            make_event(
                "QuestionResolved",
                {"id": qid, "evidence_id": evidence_id, "outcome": outcome},
                action_type=OBSERVE,
                frontier_target_id=qid,
            )
        )
    elif outcome == PARTIALLY_RESOLVED and question.get("resolution_contract") == "validator_version_contrast":
        events.append(
            make_event(
                "QuestionPartiallyResolved",
                {"id": qid, "evidence_id": evidence_id, "outcome": outcome},
                action_type=OBSERVE,
                frontier_target_id=qid,
            )
        )
    return CandidateResult(
        status="observed",
        events=events,
        evidence=evidence,
        evidence_id=evidence_id,
        world_id=world_id,
        unlocked=bool(events),
        extras={"outcome": outcome, "plan": plan.to_dict()},
    )


def _parent_records(root: Path | None) -> list[dict[str, Any]]:
    payload = load_json_artifact(root, "world.json")
    if isinstance(payload, Mapping):
        traces = payload.get("records") or payload.get("traces") or []
        if isinstance(traces, list) and traces:
            records = []
            for index, item in enumerate(traces):
                if isinstance(item, Mapping) and (item.get("validator_version") or item.get("version")):
                    row = dict(item)
                    row.setdefault("validator_version", row.get("version"))
                    records.append(row)
            return records
    return []


def build_acquire_artifacts(parent_root: Path | None, *, child_id: str, parent_id: str) -> dict[str, str]:
    records = _parent_records(parent_root)
    snapshots = {
        "schema": "validator_snapshots/v1",
        "world_id": child_id,
        "parent_id": parent_id,
        "records": records,
        "binaries": {"validator": "offline-config"},
    }
    config = {
        "child_id": child_id,
        "parent_id": parent_id,
        "dependency_digest": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16],
        "replay": "replay_interface.json",
    }
    validation = {
        "passed": len({r["validator_version"] for r in records}) >= 2,
        "checks": ["schema", "two_versions", "parent_untouched"],
        "n_records": len(records),
    }
    replay = {
        "interface": "compare_validator_snapshots",
        "input": SNAPSHOT_FILE,
        "output": ["versions", "contrast", "aligned"],
    }
    provenance = {
        "parent_id": parent_id,
        "child_id": child_id,
        "recipe": "extract_or_build_validator_snapshots",
        "attested": bool(parent_root and is_attested(fixture_from_root(parent_root, parent_id))),
    }
    return {
        SNAPSHOT_FILE: json.dumps(snapshots, ensure_ascii=False, indent=2) + "\n",
        "binaries/config.json": json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        "validation/outputs.json": json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        "replay_interface.json": json.dumps(replay, ensure_ascii=False, indent=2) + "\n",
        "provenance.json": json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
    }


def execute_acquire(env: Any, action: ActionInstance) -> CandidateResult:
    plan = AcquirePlan()
    view = env.state.view()
    parent_id = str(view.get("world_id") or "")
    parent_root = resolve_world_root(env.workspace, parent_id)
    extra_root = action.extra.get("world_root")
    if extra_root:
        parent_root = Path(extra_root)
        parent_id = parent_id or parent_root.name
    if parent_root is None:
        return CandidateResult(
            status="acquire_proposed",
            extras={
                "gap": action.extra.get("gap") or action.target,
                "world_id": parent_id,
                "harness_version": env.harness.version_id,
                "plan": plan.to_dict(),
                "reason": "no_parent_world_on_disk",
            },
            world_id=parent_id,
        )
    parent = fixture_from_root(parent_root, parent_id or parent_root.name)
    gap = str(action.extra.get("gap") or "validator_snapshots")
    if gap != "validator_snapshots" or not _parent_records(parent_root):
        return CandidateResult(
            status="blocked", world_id=parent.id,
            events=[make_event("ActionBlocked", {
                "reason": "no_acquisition_recipe", "question_id": action.target,
                "gap": gap,
            }, action_type=ACQUIRE, frontier_target_id=action.frontier_target_id or action.target)],
            extras={"reason": "no_acquisition_recipe", "gap": gap,
                    "next": "Provide a frozen dataset and a matching acquisition recipe."},
        )
    child_id = next_world_label(parent_id or "W0")
    dest = Path(env.workspace) / "worlds" / child_id if env.workspace else parent_root.parent / child_id
    artifacts = build_acquire_artifacts(parent_root, child_id=child_id, parent_id=parent.id)
    child = derive_world_version(
        parent,
        dest,
        new_id=child_id,
        acquisition_event=f"acquire:{action.target or child_id}",
        idea_id=str(action.target or ""),
        parent_version=parent_id or "W0",
        extra_artifacts=artifacts,
    )
    events = [
        make_event(
            "WorldCreated",
            {
                "id": child.id,
                "world_id": child.id,
                "parent_id": parent.id,
                "digest": child.digest,
                "observables": "validator_snapshots",
                "files": list(child.files),
                "path": str(child.root),
            },
            action_type=ACQUIRE,
            frontier_target_id=action.frontier_target_id or action.target,
        ),
        make_event(
            "WorldAttested",
            {
                "id": child.id,
                "world_id": child.id,
                "digest": child.digest,
                "world_evidence_id": child.world_evidence_id,
            },
            action_type=ACQUIRE,
            frontier_target_id=action.frontier_target_id or action.target,
        ),
        make_event(
            "ArtifactAdded",
            {"world_id": child.id, "files": list(plan.expected_artifacts), "digest": child.digest},
            action_type=ACQUIRE,
            frontier_target_id=action.frontier_target_id or action.target,
        ),
    ]
    attested = is_attested(child)
    if not attested:
        events = [event for event in events if event.event_type != "WorldAttested"]
    evidence = {
        "role": "AcquisitionEvidence",
        "world_id": child.id,
        "parent_world": parent.id,
        "evidence_id": child.world_evidence_id,
        "digest": child.digest,
        "parent_digest": parent.digest,
        "epistemic": "WORLD" if attested else "GENERATED",
        "attested": attested,
        "harness_version": env.harness.version_id,
        "artifacts": list(plan.expected_artifacts),
        "plan": plan.to_dict(),
    }
    return CandidateResult(
        status="acquired",
        events=events,
        evidence=evidence,
        world_id=child.id,
        evidence_id=child.world_evidence_id,
        unlocked=True,
        extras={
            "parent_digest": parent.digest,
            "child_digest": child.digest,
            "files": list(child.files),
            "harness_version": env.harness.version_id,
            "plan": plan.to_dict(),
        },
    )


def execute_theorize(env: Any, action: ActionInstance) -> CandidateResult:
    view = env.state.view()
    qid = action.target or action.frontier_target_id
    question = next(
        (row for row in view.get("open_questions") or [] if row.get("id") == qid),
        {},
    )
    if not any("validator" in str(item).lower() for item in [question, view.get("goal")]):
        return CandidateResult(status="blocked", extras={
            "reason": "topic_theory_executor_not_connected",
            "next": "Connect the research workers through run_research with a configured model.",
        })
    evidence = list(view.get("evidence_records") or [])
    anomalies = list(view.get("anomalies") or [])
    competing = list(view.get("competing_explanations") or ["null_no_version_effect"])
    mechanism = (
        "validator-version history changes acceptance if snapshots differ"
        if any("validator" in str(item).lower() for item in [question, view.get("goal")])
        else f"mechanism for {qid or view.get('goal')}"
    )
    tid = f"T-{qid or 'goal'}"
    preds = [
        {
            "prediction": "aligned validator snapshots yield a contrast",
            "maps_to": OBSERVE,
            "competitor_differs": True,
            "competitor": competing[0] if competing else "null",
        },
        {
            "prediction": "a two-arm freeze on the same measure separates the mechanism",
            "maps_to": PROBE,
            "competitor_differs": True,
            "competitor": competing[0] if competing else "null",
        },
    ]
    theory = {
        "id": tid,
        "question_id": qid,
        "target": qid,
        "text": action.target or mechanism,
        "assumptions": ["world artifacts are versioned", "measurement is observational"],
        "mechanism": mechanism,
        "predictions": [item["prediction"] for item in preds],
        "boundary_conditions": ["same world lineage", "no harness rewrite of evidence"],
        "competing": competing,
        "discriminating_predictions": preds,
        "anomalies": anomalies[:4],
        "evidence_considered": [str(row.get("evidence_id") or "") for row in evidence[:6]],
        "epistemic": "GENERATED",
    }
    return CandidateResult(
        status="theorized",
        events=[
            make_event(
                "TheoryCreated",
                theory,
                action_type=THEORIZE,
                frontier_target_id=qid or action.frontier_target_id,
            )
        ],
        extras={"epistemic": "GENERATED", "theory": theory},
    )


def execute_verify(env: Any, action: ActionInstance) -> CandidateResult:
    view = env.state.view()
    kind = str(action.extra.get("verify_kind") or "VERIFY_REPLICATION")
    world_id = str(view.get("world_id") or "")
    world_root = resolve_world_root(env.workspace, world_id)
    candidates = [
        row
        for row in view.get("evidence_records") or []
        if row.get("evidence_id")
    ]
    target = None
    for row in candidates:
        if str(row.get("epistemic") or "") == "WORLD" and int(row.get("verify_count") or 0) < 1:
            target = dict(row)
            break
    if target is None and candidates:
        target = dict(candidates[-1])
    if target is None:
        return CandidateResult(status="verify_skipped", extras={"reason": "no_evidence"})

    records = []
    if world_root is not None:
        payload = load_json_artifact(world_root, SNAPSHOT_FILE)
        if isinstance(payload, Mapping):
            records = list(payload.get("records") or [])
    fixture = fixture_from_root(world_root, world_id) if world_root else None
    replay_ok = False
    if fixture and is_attested(fixture):
        if target.get("measurement") == "descriptives-v1":
            replay_ok = (target.get("world_digest") == fixture.digest and target.get("result") == summarize(fixture))
        elif target.get("role") == "AcquisitionEvidence":
            replay_ok = target.get("digest") == fixture.digest
    compatible = str(target.get("world_id") or "") in {"", world_id}
    if not compatible:
        kind = "VERIFY_WORLD_COMPATIBILITY"
        replay_ok = False
    updated = dict(target)
    updated["verify_count"] = int(target.get("verify_count") or 0) + 1
    updated["verify_kind"] = kind
    updated["verified_by"] = f"verify-{env.harness.version_id}-{updated['verify_count']}"
    updated["world_compatible"] = compatible
    updated["replay_ok"] = replay_ok
    if str(updated.get("epistemic") or "") != "WORLD":
        updated["epistemic"] = str(target.get("epistemic") or "GENERATED")
        updated["attested"] = False
    events = [
        make_event(
            "EvidenceAdded",
            updated,
            action_type=VERIFY,
            frontier_target_id=action.frontier_target_id or str(target.get("question_id") or ""),
        )
    ]
    return CandidateResult(
        status="verified",
        events=events,
        extras={"verify_kind": kind, "replay_ok": replay_ok, "world_compatible": compatible},
        evidence_id=str(updated.get("evidence_id") or ""),
        world_id=world_id,
        unlocked=replay_ok,
    )


def execute_probe(env: Any, action: ActionInstance) -> CandidateResult:
    extra = dict(action.extra or {})
    if not (extra.get("causal_handle") and extra.get("measure") and (extra.get("freeze") or env.state.world_id)):
        return CandidateResult(
            status="probe_not_forced",
            extras={"reason": "no_default_two_arm"},
        )
    treatment = extra.get("treatment")
    control = extra.get("control")
    if treatment is None or control is None:
        return CandidateResult(
            status="probe_not_forced",
            extras={"reason": "arms_not_provided", "reuse": "diagnose.judge_probe"},
        )
    evidence_id = f"probe-{env.state.world_id}-{action.target or 'h'}"
    evidence = {
        "role": "ProbeEvidence",
        "evidence_id": evidence_id,
        "world_id": env.state.world_id,
        "causal_handle": extra.get("causal_handle"),
        "measure": extra.get("measure"),
        "treatment": treatment,
        "control": control,
        "epistemic": "GENERATED",
        "attested": False,
        "harness_version": env.harness.version_id,
    }
    return CandidateResult(
        status="probed",
        evidence=evidence,
        evidence_id=evidence_id,
        world_id=env.state.world_id,
        extras={"reuse": "diagnose.judge_probe"},
    )


def execute_ask(env: Any, action: ActionInstance) -> CandidateResult:
    qid = f"q-{len(env.state.open_questions)+1}"
    return CandidateResult(
        status="asked",
        events=[
            make_event(
                "QuestionCreated",
                {
                    "id": qid,
                    "text": action.target or env.state.goal,
                    "epistemic": "GENERATED",
                    "lineage_id": action.lineage_id,
                },
                action_type=ASK,
                frontier_target_id=action.frontier_target_id or qid,
            )
        ],
        extras={"question_id": qid},
    )


def execute_survey(env: Any, action: ActionInstance) -> CandidateResult:
    return CandidateResult(
        status="blocked",
        extras={"citations": [], "reason": "literature_executor_not_connected",
                "gaps": list(env.state.missing_capabilities)},
    )


def execute_reflect(env: Any, action: ActionInstance) -> CandidateResult:
    from .evolve import pathology_ready

    ready = pathology_ready(env.pathologies)
    return CandidateResult(
        status="reflected",
        extras={"pathologies": [row.to_dict() for row in ready]},
    )


def execute_synthesize(env: Any, action: ActionInstance) -> CandidateResult:
    snapshot = frontier_from_state(env.state)
    snapshot.update(
        {
            "known": [row.get("evidence_id") for row in env.state.evidence_records if row.get("attested")],
            "weaker": [row.get("id") for row in env.state.theories if row.get("status") == "weakened"],
            "unresolved": [row.get("id") for row in env.state.open_questions],
            "surviving_theories": [row.get("id") for row in env.state.theories],
            "conflicts": list(env.state.contradictions),
            "capabilities": list(env.state.available_capabilities),
            "stale_worlds": [
                row.get("id") or row.get("world_id")
                for row in env.state.world_versions
                if str(row.get("id") or row.get("world_id") or "") != env.state.world_id
            ],
            "debt": dict(env.state.debt),
        }
    )
    if env.workspace is not None:
        dest = Path(env.workspace) / "FRONTIER_SNAPSHOT.json"
        dest.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return CandidateResult(
        status="synthesized",
        events=[
            make_event(
                "FrontierChanged",
                {"frontier": snapshot},
                action_type=SYNTHESIZE,
                frontier_target_id=action.frontier_target_id,
            )
        ],
        extras={"snapshot": snapshot},
    )


def execute_archive(env: Any, action: ActionInstance) -> CandidateResult:
    return CandidateResult(status="archived")


def execute_fork(env: Any, action: ActionInstance) -> CandidateResult:
    branches = fork_trajectories(env.state, harness_version=env.harness.version_id)
    return CandidateResult(
        status="forked",
        events=[
            make_event(
                "LineageForked",
                {"lineages": [row.lane for row in branches]},
                action_type=FORK_LINEAGE,
            )
        ],
        extras={
            "lanes": [row.lane for row in branches],
            "next_actions": [row.next_action for row in branches],
        },
    )


def maybe_gc(env: Any) -> None:
    if env.state.ticks and env.state.ticks % 20 == 0:
        env.harness, deprecated = gc_harness(env.harness, tick=env.state.ticks)
        for item in deprecated:
            if item.name in env.capabilities.items:
                env.capabilities.deprecate(item.name)


DEFAULT_EXECUTORS = {
    OBSERVE: execute_observe,
    ACQUIRE: execute_acquire,
    THEORIZE: execute_theorize,
    VERIFY: execute_verify,
    PROBE: execute_probe,
    ASK: execute_ask,
    SURVEY: execute_survey,
    REFLECT: execute_reflect,
    SYNTHESIZE: execute_synthesize,
    ARCHIVE: execute_archive,
    FORK_LINEAGE: execute_fork,
}
