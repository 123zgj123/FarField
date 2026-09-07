"""Trusted StateReducer: the only writer of canonical ScientificState."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from .events import EventLog, ScientificEvent
from .frontier import debt_from_state, frontier_from_state, graph_from_state
from .kernel import refuse_generated_promotion
from .progress import update_metrics
from .state import ScientificState


def state_digest(state: ScientificState) -> str:
    raw = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def reduce_event(
    state: ScientificState,
    event: ScientificEvent,
    *,
    action: Any = None,
) -> ScientificState:
    current = ScientificState.from_dict(state.to_dict())
    payload = dict(event.payload or {})
    kind = event.event_type
    current.event_seq = max(current.event_seq, int(event.seq or 0))

    if kind == "GoalSet":
        current.goal = str(payload.get("goal") or current.goal)
    elif kind == "QuestionCreated":
        qid = str(payload.get("id") or payload.get("question_id") or "")
        if qid and not any(row.get("id") == qid for row in current.open_questions):
            current.add_question(
                question_id=qid,
                text=str(payload.get("text") or ""),
                required_observable=str(payload.get("required_observable") or ""),
                required_capability=str(payload.get("required_capability") or ""),
                lineage_id=str(payload.get("lineage_id") or ""),
                epistemic=str(payload.get("epistemic") or "GENERATED"),
            )
        gap = str(payload.get("required_capability") or "")
        if gap and gap not in current.available_capabilities:
            current.note_missing(gap)
    elif kind == "QuestionResolved":
        current.resolve_question(
            str(payload.get("id") or payload.get("question_id") or ""),
            evidence_id=str(payload.get("evidence_id") or ""),
            partial=False,
        )
    elif kind == "QuestionPartiallyResolved":
        current.resolve_question(
            str(payload.get("id") or payload.get("question_id") or ""),
            evidence_id=str(payload.get("evidence_id") or ""),
            partial=True,
        )
    elif kind == "QuestionBlocked":
        current.block_question(
            str(payload.get("id") or payload.get("question_id") or ""),
            reason=str(payload.get("reason") or ""),
        )
    elif kind == "TheoryCreated":
        row = dict(payload)
        row.setdefault("epistemic", "GENERATED")
        if refuse_generated_promotion(row):
            row["epistemic"] = "GENERATED"
        tid = str(row.get("id") or row.get("text") or "")
        if tid and not any(
            str(item.get("id") or item.get("text") or "") == tid for item in current.theories
        ):
            current.theories.append(row)
        for item in row.get("competing") or []:
            text = str(item)
            if text and text not in current.competing_explanations:
                current.competing_explanations.append(text)
    elif kind == "TheoryWeakened":
        _touch_theory(current, payload, status="weakened")
    elif kind == "TheorySupported":
        _touch_theory(current, payload, status="supported")
    elif kind == "TheoryContradicted":
        _touch_theory(current, payload, status="contradicted")
    elif kind == "EvidenceAdded":
        row = dict(payload)
        if refuse_generated_promotion(row) and str(row.get("epistemic") or "") == "WORLD":
            row["epistemic"] = "GENERATED"
            row["promotion_refused"] = refuse_generated_promotion(row)
        eid = str(row.get("evidence_id") or "")
        existing = next(
            (
                item
                for item in current.evidence_records
                if eid and str(item.get("evidence_id") or "") == eid
            ),
            None,
        )
        if existing is not None:
            if row.get("verify_count") is not None or row.get("verify_kind"):
                existing.update(
                    {
                        key: row[key]
                        for key in (
                            "verify_count",
                            "verify_kind",
                            "verified_by",
                            "world_compatible",
                            "replay_ok",
                        )
                        if key in row
                    }
                )
            # do not duplicate the same EvidenceID
        else:
            current.evidence_records.append(row)
        if eid and eid not in current.archive_ids:
            current.archive_ids.append(eid)
    elif kind == "EvidenceRejected":
        current.rejected_ideas.append(dict(payload))
    elif kind == "WorldCreated":
        row = dict(payload)
        wid = str(row.get("id") or row.get("world_id") or "")
        if wid:
            current.world_id = wid
            if not any(
                str(item.get("id") or item.get("world_id") or "") == wid
                for item in current.world_versions
            ):
                current.world_versions.append(row)
        obs = str(row.get("observables") or "")
        if obs and obs in current.missing_capabilities:
            current.missing_capabilities.remove(obs)
    elif kind == "WorldAttested":
        wid = str(payload.get("id") or payload.get("world_id") or "")
        for item in current.world_versions:
            if str(item.get("id") or item.get("world_id") or "") == wid:
                item["attested"] = True
                item["digest"] = payload.get("digest") or item.get("digest")
                item["world_evidence_id"] = payload.get("world_evidence_id") or item.get(
                    "world_evidence_id"
                )
        if wid:
            current.world_id = wid
        for row in list(current.blocked_questions):
            current.reopen_question(str(row.get("id") or ""))
    elif kind == "WorldDeprecated":
        wid = str(payload.get("id") or payload.get("world_id") or "")
        for item in current.world_versions:
            if str(item.get("id") or item.get("world_id") or "") == wid:
                item["deprecated"] = True
    elif kind == "CapabilityProposed":
        name = str(payload.get("name") or "")
        if name and name not in current.missing_capabilities and name not in current.available_capabilities:
            current.note_missing(name)
    elif kind == "CapabilityValidated":
        name = str(payload.get("name") or "")
        if name:
            current.note_available(name)
    elif kind == "CapabilityTrusted":
        name = str(payload.get("name") or "")
        if name:
            current.note_available(name)
    elif kind == "CapabilityRevoked":
        name = str(payload.get("name") or "")
        if name in current.available_capabilities:
            current.available_capabilities.remove(name)
        current.note_missing(name)
    elif kind == "CapabilityDeprecated":
        name = str(payload.get("name") or "")
        if name in current.available_capabilities:
            current.available_capabilities.remove(name)
    elif kind == "HarnessAdmitted":
        current.harness_version = str(payload.get("harness_version") or current.harness_version)
        cap = str(payload.get("capability") or "")
        if cap:
            current.note_available(cap)
    elif kind == "HarnessPatchProposed":
        pass
    elif kind == "ActionFailed":
        current.failed_designs.append(dict(payload))
    elif kind == "ActionBlocked":
        qid = str(payload.get("question_id") or payload.get("target") or "")
        if qid and any(row.get("id") == qid for row in current.open_questions):
            current.block_question(qid, reason=str(payload.get("reason") or ""))
        elif payload.get("gap"):
            current.note_missing(str(payload.get("gap")))
    elif kind == "ContradictionDetected":
        row = dict(payload)
        if row not in current.contradictions:
            current.contradictions.append(row)
    elif kind == "LineageForked":
        for lid in payload.get("lineages") or []:
            text = str(lid)
            if text and text not in current.active_lineages:
                current.active_lineages.append(text)
    elif kind == "LineageMerged":
        for lid in payload.get("retired") or []:
            text = str(lid)
            if text in current.active_lineages and len(current.active_lineages) > 1:
                current.active_lineages.remove(text)
    elif kind == "FrontierChanged":
        current.frontier = dict(payload.get("frontier") or payload)
        current.metrics["last_synthesis_events"] = int(current.metrics.get("meaningful_events") or 0)
        current.metrics["meaningful_events"] = int(current.metrics.get("meaningful_events") or 0) + 1
    elif kind == "DebtUpdated":
        merged = dict(current.debt)
        merged.update({str(k): int(v) for k, v in dict(payload.get("debt") or payload).items()})
        current.debt = merged
    elif kind == "ArtifactAdded":
        row = dict(payload)
        if row not in current.artifacts:
            current.artifacts.append(row)
    elif kind == "TickAdvanced":
        current.ticks = int(current.ticks) + 1
        if payload.get("meaningful"):
            current.last_progress_tick = current.ticks
            current.metrics["meaningful_events"] = int(current.metrics.get("meaningful_events") or 0) + 1
        if payload.get("goal_drift"):
            current.metrics["goal_drift"] = int(current.metrics.get("goal_drift") or 0) + 1
        current.debt = debt_from_state(current)
        current.frontier = frontier_from_state(current)
        current.graph = graph_from_state(current)
        update_metrics(
            current,
            action=action,
            action_type=str(payload.get("action_type") or event.action_type or ""),
            scripted=bool(payload.get("scripted")),
            unlock=bool(payload.get("unlock")),
            integrity_gap=str(payload.get("integrity_gap") or ""),
            archive_hit=bool(payload.get("archive_hit")),
            harness_general=bool(payload.get("harness_general")),
            regression=bool(payload.get("regression")),
        )
        return current
    current.debt = debt_from_state(current)
    current.frontier = frontier_from_state(current)
    current.graph = graph_from_state(current)
    return current


def apply_events(
    state: ScientificState,
    events: Iterable[ScientificEvent],
    *,
    log: EventLog | None = None,
    action: Any = None,
    tick: int | None = None,
) -> ScientificState:
    current = ScientificState.from_dict(state.to_dict())
    now = int(current.ticks if tick is None else tick)
    for event in events:
        if log is not None:
            event = log.append(event, tick=now)
        current = reduce_event(current, event, action=action)
    current.debt = debt_from_state(current)
    current.frontier = frontier_from_state(current)
    current.graph = graph_from_state(current)
    return current


def rebuild_state(log: EventLog) -> ScientificState:
    state = ScientificState.from_dict(log.baseline)
    for event in log.events:
        state = reduce_event(state, event)
    state.debt = debt_from_state(state)
    state.frontier = frontier_from_state(state)
    state.graph = graph_from_state(state)
    return state


def _touch_theory(state: ScientificState, payload: Mapping[str, Any], *, status: str) -> None:
    tid = str(payload.get("id") or payload.get("theory_id") or payload.get("text") or "")
    for row in state.theories:
        if str(row.get("id") or row.get("text") or "") == tid:
            row["status"] = status
            if payload.get("evidence_id"):
                row.setdefault("evidence_ids", [])
                if isinstance(row["evidence_ids"], list):
                    eid = str(payload.get("evidence_id"))
                    if eid not in row["evidence_ids"]:
                        row["evidence_ids"].append(eid)
            return
    row = dict(payload)
    row["status"] = status
    row.setdefault("epistemic", "GENERATED")
    state.theories.append(row)
