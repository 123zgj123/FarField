"""Research frontier, debt, and a thin relation graph over ScientificState.

These are projections of the canonical state, not a second truth.
"""

from __future__ import annotations

from typing import Any, Mapping

from .state import ScientificState


SCIENTIFIC_ACTIONS = frozenset(
    {
        "SURVEY",
        "ASK",
        "OBSERVE",
        "ACQUIRE",
        "THEORIZE",
        "PROBE",
        "VERIFY",
    }
)

MAINTENANCE_ACTIONS = frozenset(
    {
        "REFLECT",
        "ARCHIVE",
        "EVOLVE_HARNESS",
        "SYNTHESIZE",
        "FORK_LINEAGE",
    }
)


def _ids(rows: list[Mapping[str, Any]] | list[dict[str, Any]], key: str = "id") -> list[str]:
    out: list[str] = []
    for row in rows or []:
        if isinstance(row, Mapping):
            text = str(row.get(key) or row.get("text") or "").strip()
        else:
            text = str(row).strip()
        if text and text not in out:
            out.append(text)
    return out


def frontier_from_state(state: ScientificState) -> dict[str, Any]:
    competing = [
        str(row.get("id") or row.get("text") or "")
        for row in state.theories
        if isinstance(row, Mapping) and row.get("competing")
    ]
    nodes = []
    if state.goal:
        nodes.append({"id": f"goal:{_goal_id(state)}", "kind": "Goal", "status": "active"})
    for row in state.open_questions:
        nodes.append({"id": str(row.get("id") or ""), "kind": "Question", "status": "active"})
    for row in state.resolved_questions:
        nodes.append({"id": str(row.get("id") or ""), "kind": "Question", "status": "resolved"})
    for row in state.blocked_questions:
        nodes.append({"id": str(row.get("id") or ""), "kind": "Question", "status": "blocked"})
    for row in state.theories:
        nodes.append(
            {
                "id": str(row.get("id") or row.get("text") or "")[:80],
                "kind": "Theory",
                "status": str(row.get("status") or "open"),
            }
        )
    return {
        "root_goal": state.goal,
        "active_questions": _ids(state.open_questions),
        "resolved_questions": _ids(state.resolved_questions),
        "blocked_questions": _ids(state.blocked_questions),
        "open_theories": _ids(state.theories),
        "competing_theories": [item for item in competing if item],
        "required_capabilities": list(state.missing_capabilities),
        "verification_debt": int((state.debt or {}).get("verification_debt") or 0),
        "contradictions": [dict(item) for item in state.contradictions],
        "current_frontier_nodes": [item for item in nodes if item.get("status") in {"active", "blocked", "open"}],
        "nodes": nodes,
    }


def debt_from_state(state: ScientificState) -> dict[str, int]:
    world_ev = [
        row
        for row in state.evidence_records
        if str(row.get("epistemic") or "") == "WORLD" and row.get("evidence_id")
    ]
    unverified = [
        row
        for row in world_ev
        if int(row.get("verify_count") or 0) < 1
        or (row.get("high_impact") and int(row.get("source_count") or 1) <= 1)
    ]
    stale_worlds = max(0, len(state.world_versions) - 1) if state.world_versions else 0
    unused_caps = [
        name
        for name in state.available_capabilities
        if not any(str(row.get("capability") or "") == name for row in state.evidence_records)
    ]
    orphan_theories = [
        row
        for row in state.theories
        if not row.get("discriminating_predictions")
        or not any(pred.get("competitor_differs") for pred in row.get("discriminating_predictions") or [])
    ]
    archive_working = int((state.metrics or {}).get("working_archive") or 0)
    return {
        "verification_debt": len(unverified),
        "contradiction_debt": len(state.contradictions),
        "world_staleness": stale_worlds,
        "capability_debt": len(state.missing_capabilities) + len(unused_caps),
        "unresolved_dependency": len(state.blocked_questions)
        + sum(1 for row in state.open_questions if row.get("required_observable") or row.get("required_capability")),
        "archive_debt": max(0, archive_working - len(world_ev)),
        "orphan_theory_debt": len(orphan_theories),
    }


def graph_from_state(state: ScientificState) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    seen: set[str] = set()

    def node(kind: str, node_id: str, **attrs: Any) -> None:
        key = f"{kind}:{node_id}"
        if not node_id or key in seen:
            return
        seen.add(key)
        nodes.append({"kind": kind, "id": node_id, **attrs})

    def edge(src: str, rel: str, dst: str) -> None:
        if src and rel and dst:
            edges.append({"src": src, "rel": rel, "dst": dst})

    goal_id = f"goal:{_goal_id(state)}"
    if state.goal:
        node("Goal", goal_id, text=state.goal)
    if state.world_id:
        node("World", state.world_id, current=True)
    for row in state.world_versions:
        wid = str(row.get("id") or row.get("world_id") or "")
        node("World", wid)
        parent = str(row.get("parent_id") or "")
        if parent:
            edge(parent, "derived_from", wid)
    for row in state.open_questions + state.resolved_questions + state.blocked_questions:
        qid = str(row.get("id") or "")
        node("Question", qid, status=str(row.get("status") or "open"))
        if state.goal:
            edge(goal_id, "motivates", qid)
        cap = str(row.get("required_capability") or "")
        if cap:
            node("Capability", cap)
            edge(qid, "requires", cap)
        obs = str(row.get("required_observable") or "")
        if obs:
            node("Artifact", obs)
            edge(qid, "requires", obs)
    for row in state.theories:
        tid = str(row.get("id") or row.get("text") or "")[:80]
        node("Theory", tid, epistemic=str(row.get("epistemic") or "GENERATED"))
        qid = str(row.get("question_id") or row.get("target") or "")
        if qid:
            edge(qid, "motivates", tid)
        for pred in row.get("discriminating_predictions") or []:
            maps = str(pred.get("maps_to") or "")
            if maps and maps != "untestable":
                edge(tid, "tests", maps)
    for row in state.evidence_records:
        eid = str(row.get("evidence_id") or "")
        node(
            "Evidence",
            eid,
            epistemic=str(row.get("epistemic") or ""),
            world_id=str(row.get("world_id") or ""),
        )
        qid = str(row.get("question") or row.get("question_id") or row.get("target") or "")
        if qid:
            edge(eid, "answers", qid)
        wid = str(row.get("world_id") or "")
        if wid:
            edge(eid, "valid_in_world", wid)
        tid = str(row.get("theory_id") or "")
        relation = str(row.get("theory_relation") or row.get("verdict") or "")
        if tid and relation in {"supports", "weakens", "contradicts"}:
            edge(eid, relation, tid)
        produced = str(row.get("produced_by") or "")
        if produced:
            edge(produced, "produced_by", eid)
    for name in state.available_capabilities:
        node("Capability", name, status="available")
    for name in state.missing_capabilities:
        node("Capability", name, status="missing")
    return {"nodes": nodes, "edges": edges}


def _goal_id(state: ScientificState) -> str:
    return "G0" if state.goal else ""


def traces_to_root(state: ScientificState, frontier_target_id: str) -> bool:
    """True when the target is the goal, a question, or a node linked from the goal."""
    target = str(frontier_target_id or "").strip()
    if not target:
        return False
    if target == state.goal or target == f"goal:{_goal_id(state)}" or target == "G0":
        return True
    known = set(_ids(state.open_questions + state.resolved_questions + state.blocked_questions))
    known.update(_ids(state.theories))
    known.update(str(row.get("evidence_id") or "") for row in state.evidence_records)
    known.update(state.active_lineages)
    if target in known:
        return True
    graph = state.graph if state.graph.get("edges") else graph_from_state(state)
    goal_id = f"goal:{_goal_id(state)}"
    seen = {goal_id, state.goal}
    queue = [goal_id, state.goal]
    edges = list(graph.get("edges") or [])
    while queue:
        cur = queue.pop(0)
        for edge in edges:
            if edge.get("src") == cur and edge.get("dst") not in seen:
                seen.add(str(edge.get("dst") or ""))
                queue.append(str(edge.get("dst") or ""))
    return target in seen


def blocked_descendants(state: ScientificState, action_type: str, target: str) -> int:
    """How many frontier nodes this action would unlock."""
    if action_type == "ACQUIRE":
        n = 0
        for row in state.open_questions:
            qid = str(row.get("id") or "")
            if target and qid != target and target not in {
                str(row.get("required_observable") or ""),
                str(row.get("required_capability") or ""),
            }:
                continue
            if row.get("required_observable") or row.get("required_capability"):
                n += 1
                n += 1  # OBSERVE
                n += 1  # possible VERIFY
        return n or 1
    if action_type == "EVOLVE_HARNESS":
        return max(1, len(state.missing_capabilities) + len(state.open_questions))
    if action_type == "OBSERVE":
        return 1
    if action_type == "VERIFY":
        return max(1, len(state.contradictions))
    return 0
