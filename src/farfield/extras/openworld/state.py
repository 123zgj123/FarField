"""Persistent scientific control state. Not evidence, not a WorldID.

Notebook and researchspace.ResearchState are not this object.
This is the only control state the scheduler reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .kernel import refuse_generated_promotion


SCIENTIFIC_STATE_FILE = "SCIENTIFIC_STATE.json"


def evidence_reliability(row: Mapping[str, Any]) -> float:
    """Versioned semi-quantitative accounting rule, not a learned probability.

    Evidence about execution failure, descriptive observations and generated
    proposals contributes zero. Reproduction improves reliability, not sample size.
    """
    if (row.get("epistemic") != "WORLD" or not row.get("attested")
            or not row.get("promotion_ok") or refuse_generated_promotion(row)
            or row.get("cannot_corroborate") or row.get("execution_status") != "ran"
            or not row.get("experiment_digest") or not row.get("world_digest")
            or not row.get("claim_consistent")):
        return 0.0
    if row.get("verify_count") and (row.get("reproduction_ok") is False or row.get("replay_ok") is False):
        return 0.0
    return 1.0 if row.get("reproduction_ok") and row.get("identity_ok") else 0.5


def refresh_research_views(state: "ScientificState") -> None:
    """Derived event-replay views; workers cannot supply posterior confidence.

    Beta(2,2) pseudo-counts with reliability-weighted positive/negative evidence.
    This is deliberately a bounded bookkeeping scale, not calibrated Bayesian
    inference. Correlated reruns of the same source on the same freeze count once.
    """
    beliefs: dict[str, dict[str, Any]] = {}
    verified: dict[str, dict[str, Any]] = {}
    for theory in state.theories:
        tid = str(theory.get("id") or "")
        if not tid:
            continue
        belief: dict[str, Any] = {
            "rule": "reliability-beta-counts-v1", "prior_confidence": 0.5,
            "posterior_confidence": 0.5, "supporting_evidence": [],
            "contradicting_evidence": [], "evidence_reliability": {},
            "competing_theories": list(theory.get("competing") or theory.get("competing_explanations") or []),
            "unresolved_assumptions": list(theory.get("assumptions") or []),
            "update_provenance": [],
        }
        positive, negative = 2.0, 2.0
        seen: set[tuple[str, str]] = set()
        rows = [r for r in state.evidence_records if r.get("theory_id") == tid]
        for row in rows:
            reliability = evidence_reliability(row)
            identity = (str(row.get("world_digest")), str(row.get("experiment_digest")))
            eid = str(row.get("evidence_id") or "")
            outcome = row.get("outcome") or row.get("verdict")
            if not eid or not reliability or identity in seen or outcome not in {"supports", "weakens", "negative", "contradicted"}:
                continue
            seen.add(identity)
            before = positive / (positive + negative)
            supports = outcome == "supports"
            if supports:
                positive += reliability
                belief["supporting_evidence"].append(eid)
            else:
                negative += reliability
                belief["contradicting_evidence"].append(eid)
            belief["evidence_reliability"][eid] = reliability
            after = positive / (positive + negative)
            belief["update_provenance"].append({
                "evidence_id": eid, "event_id": row.get("scientific_event_id", ""),
                "verification_event_id": row.get("verification_event_id", ""),
                "prior": round(before, 8), "evidence_contribution": reliability if supports else -reliability,
                "posterior": round(after, 8), "world_digest": identity[0], "experiment_digest": identity[1],
            })
        belief["posterior_confidence"] = round(positive / (positive + negative), 8)
        theory["prior_confidence"] = belief["prior_confidence"]
        theory["posterior_confidence"] = belief["posterior_confidence"]
        theory["belief_state_id"] = tid
        if belief["contradicting_evidence"]:
            theory["status"] = "weakened"
        elif belief["supporting_evidence"]:
            theory["status"] = "supported"
        beliefs[tid] = belief
        # A corroborated execution is not automatically a verified mechanism.
        for row in rows:
            if not (evidence_reliability(row) and row.get("outcome") == "supports"
                    and all(row.get(k) for k in ("literature_checked", "reproduction_ok", "identity_ok",
                                                 "executable_ok", "mechanism_validated"))):
                continue
            replication = next((r for r in rows if r.get("replication_of") == row.get("evidence_id")
                and r.get("world_digest") != row.get("world_digest") and evidence_reliability(r)
                and r.get("outcome") == "supports" and r.get("identity_ok") and r.get("reproduction_ok")), None)
            if row.get("replication_required", True) and replication is None:
                continue
            if belief["contradicting_evidence"]:
                continue  # unresolved contradictions keep the mechanism speculative
            verified[tid] = {"theory_id": tid, "evidence_id": row["evidence_id"],
                             "replication_evidence_id": replication["evidence_id"] if replication else "",
                             "claim_strength": "verified_within_tested_conditions",
                             "world_id": row.get("world_id"), "world_digest": row.get("world_digest")}
            break
    state.belief_states = beliefs
    state.verified_research_state = verified
    state.speculative_frontier = [tid for tid in beliefs if tid not in verified]


def _copy_rows(rows: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in rows or []:
        if isinstance(item, Mapping):
            out.append(dict(item))
        elif str(item).strip():
            out.append({"text": str(item).strip()})
    return out


def _copy_text(rows: Any) -> list[str]:
    seen: list[str] = []
    for item in rows or []:
        if isinstance(item, Mapping):
            text = str(item.get("text") or item.get("id") or item.get("target") or "").strip()
        else:
            text = str(item).strip()
        if text and text not in seen:
            seen.append(text)
    return seen


@dataclass
class ScientificState:
    """Control state for the open-world scheduler.

    GENERATED rows may be listed. Listing is not attestation.
    Mutation of these fields in production goes through the reducer.
    """

    goal: str = ""
    open_questions: list[dict[str, Any]] = field(default_factory=list)
    resolved_questions: list[dict[str, Any]] = field(default_factory=list)
    partial_questions: list[dict[str, Any]] = field(default_factory=list)
    blocked_questions: list[dict[str, Any]] = field(default_factory=list)
    theories: list[dict[str, Any]] = field(default_factory=list)
    competing_explanations: list[str] = field(default_factory=list)
    world_versions: list[dict[str, Any]] = field(default_factory=list)
    evidence_records: list[dict[str, Any]] = field(default_factory=list)
    accepted_ideas: list[dict[str, Any]] = field(default_factory=list)
    rejected_ideas: list[dict[str, Any]] = field(default_factory=list)
    missing_capabilities: list[str] = field(default_factory=list)
    available_capabilities: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    failed_designs: list[dict[str, Any]] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    archive_ids: list[str] = field(default_factory=list)
    active_lineages: list[str] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    harness_version: str = "H0"
    world_id: str = ""
    ticks: int = 0
    last_progress_tick: int = 0
    event_seq: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    frontier: dict[str, Any] = field(default_factory=dict)
    debt: dict[str, int] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)
    belief_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    speculative_frontier: list[str] = field(default_factory=list)
    verified_research_state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "open_questions": list(self.open_questions),
            "resolved_questions": list(self.resolved_questions),
            "partial_questions": list(self.partial_questions),
            "blocked_questions": list(self.blocked_questions),
            "theories": list(self.theories),
            "competing_explanations": list(self.competing_explanations),
            "world_versions": list(self.world_versions),
            "evidence_records": list(self.evidence_records),
            "accepted_ideas": list(self.accepted_ideas),
            "rejected_ideas": list(self.rejected_ideas),
            "missing_capabilities": list(self.missing_capabilities),
            "available_capabilities": list(self.available_capabilities),
            "artifacts": list(self.artifacts),
            "failed_designs": list(self.failed_designs),
            "anomalies": list(self.anomalies),
            "archive_ids": list(self.archive_ids),
            "active_lineages": list(self.active_lineages),
            "contradictions": list(self.contradictions),
            "harness_version": self.harness_version,
            "world_id": self.world_id,
            "ticks": self.ticks,
            "last_progress_tick": self.last_progress_tick,
            "event_seq": self.event_seq,
            "metrics": dict(self.metrics),
            "frontier": dict(self.frontier),
            "debt": dict(self.debt),
            "graph": dict(self.graph),
            "belief_states": dict(self.belief_states),
            "speculative_frontier": list(self.speculative_frontier),
            "verified_research_state": dict(self.verified_research_state),
        }

    def view(self) -> Mapping[str, Any]:
        """Read-only snapshot. Executors must not write through this."""
        return MappingProxyType(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ScientificState":
        row = dict(payload or {})
        return cls(
            goal=str(row.get("goal") or ""),
            open_questions=_copy_rows(row.get("open_questions")),
            resolved_questions=_copy_rows(row.get("resolved_questions")),
            partial_questions=_copy_rows(row.get("partial_questions")),
            blocked_questions=_copy_rows(row.get("blocked_questions")),
            theories=_copy_rows(row.get("theories")),
            competing_explanations=_copy_text(row.get("competing_explanations")),
            world_versions=_copy_rows(row.get("world_versions")),
            evidence_records=_copy_rows(row.get("evidence_records")),
            accepted_ideas=_copy_rows(row.get("accepted_ideas")),
            rejected_ideas=_copy_rows(row.get("rejected_ideas")),
            missing_capabilities=_copy_text(row.get("missing_capabilities")),
            available_capabilities=_copy_text(row.get("available_capabilities")),
            artifacts=_copy_rows(row.get("artifacts")),
            failed_designs=_copy_rows(row.get("failed_designs")),
            anomalies=_copy_text(row.get("anomalies")),
            archive_ids=_copy_text(row.get("archive_ids")),
            active_lineages=_copy_text(row.get("active_lineages")),
            contradictions=_copy_rows(row.get("contradictions")),
            harness_version=str(row.get("harness_version") or "H0"),
            world_id=str(row.get("world_id") or ""),
            ticks=int(row.get("ticks") or 0),
            last_progress_tick=int(row.get("last_progress_tick") or 0),
            event_seq=int(row.get("event_seq") or 0),
            metrics=dict(row.get("metrics") or {}),
            frontier=dict(row.get("frontier") or {}),
            debt={str(k): int(v) for k, v in dict(row.get("debt") or {}).items()},
            graph=dict(row.get("graph") or {}),
            belief_states=dict(row.get("belief_states") or {}),
            speculative_frontier=_copy_text(row.get("speculative_frontier")),
            verified_research_state=dict(row.get("verified_research_state") or {}),
        )

    def add_question(
        self,
        *,
        question_id: str,
        text: str,
        required_observable: str = "",
        required_capability: str = "",
        lineage_id: str = "",
        epistemic: str = "GENERATED",
    ) -> None:
        """Test / bootstrap helper. Production writes go through QuestionCreated."""
        row = {
            "id": question_id,
            "text": text,
            "required_observable": required_observable,
            "required_capability": required_capability,
            "lineage_id": lineage_id,
            "epistemic": epistemic or "GENERATED",
            "status": "open",
        }
        if refuse_generated_promotion(row):
            row["epistemic"] = "GENERATED"
        if not any(item.get("id") == question_id for item in self.open_questions):
            self.open_questions.append(row)
        if lineage_id and lineage_id not in self.active_lineages:
            self.active_lineages.append(lineage_id)

    def note_missing(self, name: str) -> None:
        text = str(name or "").strip()
        if text and text not in self.missing_capabilities:
            self.missing_capabilities.append(text)
        if text in self.available_capabilities:
            self.available_capabilities.remove(text)

    def note_available(self, name: str) -> None:
        text = str(name or "").strip()
        if text and text not in self.available_capabilities:
            self.available_capabilities.append(text)
        if text in self.missing_capabilities:
            self.missing_capabilities.remove(text)

    def record_evidence(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        if refuse_generated_promotion(payload) and str(payload.get("epistemic") or "") == "WORLD":
            payload["epistemic"] = "GENERATED"
            payload["promotion_refused"] = refuse_generated_promotion(payload)
        self.evidence_records.append(payload)
        return payload

    def resolve_question(self, question_id: str, *, evidence_id: str = "", partial: bool = False) -> None:
        kept: list[dict[str, Any]] = []
        moved = None
        for row in self.open_questions:
            if row.get("id") == question_id:
                moved = dict(row)
            else:
                kept.append(row)
        self.open_questions = kept
        if moved is None:
            for row in self.blocked_questions:
                if row.get("id") == question_id:
                    moved = dict(row)
                    self.blocked_questions = [
                        item for item in self.blocked_questions if item.get("id") != question_id
                    ]
                    break
        if moved is None:
            return
        moved["status"] = "partial" if partial else "resolved"
        if evidence_id:
            moved["evidence_id"] = evidence_id
        dest = self.partial_questions if partial else self.resolved_questions
        dest.append(moved)

    def block_question(self, question_id: str, *, reason: str = "") -> None:
        kept: list[dict[str, Any]] = []
        moved = None
        for row in self.open_questions:
            if row.get("id") == question_id:
                moved = dict(row)
            else:
                kept.append(row)
        self.open_questions = kept
        if moved is None:
            return
        moved["status"] = "blocked"
        if reason:
            moved["blocked_reason"] = reason
        if not any(item.get("id") == question_id for item in self.blocked_questions):
            self.blocked_questions.append(moved)

    def reopen_question(self, question_id: str) -> None:
        kept: list[dict[str, Any]] = []
        moved = None
        for row in self.blocked_questions:
            if row.get("id") == question_id:
                moved = dict(row)
            else:
                kept.append(row)
        self.blocked_questions = kept
        if moved is None:
            return
        moved["status"] = "open"
        if not any(item.get("id") == question_id for item in self.open_questions):
            self.open_questions.append(moved)


def load_scientific_state(workspace: Path | None) -> ScientificState:
    if workspace is None:
        return ScientificState()
    path = Path(workspace) / SCIENTIFIC_STATE_FILE
    if not path.is_file():
        return ScientificState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ScientificState()
    if not isinstance(payload, dict):
        return ScientificState()
    return ScientificState.from_dict(payload)


def save_scientific_state(workspace: Path | None, state: ScientificState) -> Path | None:
    if workspace is None:
        return None
    dest = Path(workspace)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / SCIENTIFIC_STATE_FILE
    path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def state_is_empty(state: ScientificState) -> bool:
    return not (
        state.goal
        or state.open_questions
        or state.resolved_questions
        or state.evidence_records
        or state.world_id
        or state.theories
    )


def sync_from_notebook(state: ScientificState, notebook: Any) -> ScientificState:
    """Bootstrap only. Notebook is not an authority once ScientificState exists."""
    if notebook is None:
        return state
    for text in getattr(notebook, "unresolved_questions", ()) or []:
        if not any(row.get("text") == text for row in state.open_questions):
            state.add_question(
                question_id=f"q-{len(state.open_questions)+1}",
                text=str(text),
                epistemic="GENERATED",
            )
    for gap in getattr(notebook, "missing_capabilities", ()) or []:
        state.note_missing(str(gap))
    for row in getattr(notebook, "world_versions", ()) or []:
        if isinstance(row, Mapping) and row not in state.world_versions:
            state.world_versions.append(dict(row))
            if not state.world_id:
                state.world_id = str(row.get("world_id") or "")
    for row in getattr(notebook, "confirmed_evidence", ()) or []:
        if isinstance(row, Mapping):
            stamped = dict(row)
            stamped.setdefault("epistemic", stamped.get("epistemic") or "GENERATED")
            if stamped not in state.evidence_records:
                state.record_evidence(stamped)
    for text in getattr(notebook, "open_mechanisms", ()) or []:
        if not any(row.get("text") == text for row in state.theories):
            state.theories.append({"text": str(text), "epistemic": "GENERATED"})
    for row in getattr(notebook, "failed_designs", ()) or []:
        if isinstance(row, Mapping) and row not in state.failed_designs:
            state.failed_designs.append(dict(row))
    for lineage in getattr(notebook, "ideas", ()) or []:
        if isinstance(lineage, Mapping):
            lid = str(lineage.get("lineage_id") or "")
            if lid and lid not in state.active_lineages:
                state.active_lineages.append(lid)
    return state


def notebook_projection(state: ScientificState) -> dict[str, Any]:
    """ScientificState → prompt projection. Not an independent truth."""
    return {
        "confirmed_evidence": [
            dict(row)
            for row in state.evidence_records
            if str(row.get("epistemic") or "") == "WORLD"
        ],
        "ruled_out": list(state.anomalies),
        "unresolved_questions": [
            str(row.get("text") or row.get("id") or "") for row in state.open_questions
        ],
        "missing_capabilities": list(state.missing_capabilities),
        "world_versions": list(state.world_versions),
        "failed_designs": list(state.failed_designs),
        "open_mechanisms": [
            str(row.get("text") or row.get("id") or "")
            for row in state.theories
            if str(row.get("epistemic") or "") != "WORLD"
        ],
        "next_actions": list((state.frontier or {}).get("current_frontier_nodes") or []),
        "ideas": list(state.accepted_ideas),
        "projection_of": "ScientificState",
    }
