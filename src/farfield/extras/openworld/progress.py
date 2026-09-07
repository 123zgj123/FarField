"""Research progress, stagnation, and trajectory-level multistart."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .actions import FORK_LINEAGE, ActionInstance
from .state import ScientificState


LANES = (
    "exploration",
    "exploitation",
    "revival",
    "understanding",
    "infrastructure",
)

STAGNATION_IDLE = 4


@dataclass
class ResearchProgress:
    question_resolution: int = 0
    validated_new_evidence: int = 0
    capability_unlock: int = 0
    theory_discrimination: int = 0
    frontier_advancement: int = 0
    high_value_negative: int = 0
    archive_reuse: int = 0
    contradiction_resolution: int = 0
    unused_capability: int = 0
    equivalent_theory: int = 0
    duplicate_evidence: int = 0

    @property
    def new_evidence(self) -> int:
        return self.validated_new_evidence

    @property
    def frontier_improvement(self) -> int:
        return self.frontier_advancement

    @property
    def meaningful(self) -> bool:
        gained = (
            self.question_resolution
            + self.validated_new_evidence
            + self.capability_unlock
            + self.theory_discrimination
            + self.frontier_advancement
            + self.high_value_negative
            + self.archive_reuse
            + self.contradiction_resolution
        )
        penalty = self.unused_capability + self.equivalent_theory + self.duplicate_evidence
        return gained > penalty

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_resolution": self.question_resolution,
            "validated_new_evidence": self.validated_new_evidence,
            "new_evidence": self.validated_new_evidence,
            "capability_unlock": self.capability_unlock,
            "theory_discrimination": self.theory_discrimination,
            "frontier_advancement": self.frontier_advancement,
            "frontier_improvement": self.frontier_advancement,
            "high_value_negative": self.high_value_negative,
            "archive_reuse": self.archive_reuse,
            "contradiction_resolution": self.contradiction_resolution,
            "unused_capability": self.unused_capability,
            "equivalent_theory": self.equivalent_theory,
            "duplicate_evidence": self.duplicate_evidence,
            "meaningful": self.meaningful,
        }


def _discriminating(theories: list[dict[str, Any]]) -> int:
    n = 0
    for row in theories:
        preds = list(row.get("discriminating_predictions") or [])
        if any(item.get("competitor_differs") for item in preds if isinstance(item, Mapping)):
            n += 1
    return n


def _world_evidence(state: ScientificState) -> list[dict[str, Any]]:
    return [
        row
        for row in state.evidence_records
        if str(row.get("epistemic") or "") == "WORLD" and row.get("evidence_id")
    ]


def measure_progress(before: ScientificState, after: ScientificState) -> ResearchProgress:
    before_ids = {
        (str(row.get("evidence_id") or ""), str(row.get("role") or ""))
        for row in before.evidence_records
    }
    new_rows = [
        row
        for row in after.evidence_records
        if (str(row.get("evidence_id") or ""), str(row.get("role") or "")) not in before_ids
    ]
    validated = [
        row
        for row in new_rows
        if str(row.get("epistemic") or "") == "WORLD" and row.get("attested")
    ]
    duplicate = max(0, len(new_rows) - len({str(row.get("evidence_id") or "") for row in new_rows}))
    unused = [
        name
        for name in after.available_capabilities
        if name not in before.available_capabilities
        and not any(str(row.get("capability") or "") == name for row in after.evidence_records)
    ]
    return ResearchProgress(
        question_resolution=max(0, len(after.resolved_questions) - len(before.resolved_questions)),
        validated_new_evidence=len(validated),
        capability_unlock=max(0, len(after.available_capabilities) - len(before.available_capabilities)),
        theory_discrimination=max(0, _discriminating(after.theories) - _discriminating(before.theories)),
        frontier_advancement=max(
            0,
            len(after.resolved_questions) - len(before.resolved_questions),
        )
        + max(0, len(_world_evidence(after)) - len(_world_evidence(before))),
        high_value_negative=max(
            0,
            sum(1 for row in after.evidence_records if row.get("outcome") in {"negative", "weakens"})
            - sum(1 for row in before.evidence_records if row.get("outcome") in {"negative", "weakens"}),
        ),
        archive_reuse=max(0, len(after.archive_ids) - len(before.archive_ids)),
        contradiction_resolution=max(0, len(before.contradictions) - len(after.contradictions)),
        unused_capability=len(unused),
        equivalent_theory=max(
            0,
            (len(after.theories) - _discriminating(after.theories))
            - (len(before.theories) - _discriminating(before.theories)),
        ),
        duplicate_evidence=duplicate,
    )


@dataclass
class StagnationEvent:
    idle_ticks: int
    lanes: tuple[str, ...] = LANES
    reason: str = "no_meaningful_progress"
    forbid: str = "far_jump_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "idle_ticks": self.idle_ticks,
            "lanes": list(self.lanes),
            "reason": self.reason,
            "forbid": self.forbid,
        }


def maybe_stagnation(state: ScientificState, *, idle: int = STAGNATION_IDLE) -> StagnationEvent | None:
    gap = int(state.ticks) - int(state.last_progress_tick)
    if gap >= idle and state.ticks > 0:
        return StagnationEvent(idle_ticks=gap)
    return None


@dataclass
class TrajectoryBranch:
    lane: str
    harness_version: str
    next_action: str
    snapshot: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    world_evidence: list[dict[str, Any]] = field(default_factory=list)
    generated: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "harness_version": self.harness_version,
            "next_action": self.next_action,
            "score": self.score,
            "world_evidence": list(self.world_evidence),
            "generated": list(self.generated),
        }


def fork_trajectories(
    state: ScientificState,
    *,
    harness_version: str,
    n: int = 3,
    preferred_actions: tuple[str, ...] = (),
) -> list[TrajectoryBranch]:
    """Snapshot state + harness and open short lanes. Not N idea cards."""
    snapshot = state.to_dict()
    lanes = list(LANES[: max(1, int(n))])
    actions = list(preferred_actions) or [
        "SURVEY",
        "THEORIZE",
        "EVOLVE_HARNESS",
        "ACQUIRE",
        "REFLECT",
    ]
    branches: list[TrajectoryBranch] = []
    for index, lane in enumerate(lanes):
        action = actions[index % len(actions)]
        if lane == "infrastructure" and "EVOLVE_HARNESS" in actions:
            action = "EVOLVE_HARNESS"
        if lane == "understanding":
            action = "THEORIZE"
        branches.append(
            TrajectoryBranch(
                lane=lane,
                harness_version=harness_version,
                next_action=action,
                snapshot=snapshot,
            )
        )
    return branches


def judge_branches(branches: list[TrajectoryBranch]) -> TrajectoryBranch:
    ranked = sorted(branches, key=lambda row: (-row.score, row.lane))
    return ranked[0]


def _evidence_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("question_id") or row.get("question") or row.get("target") or ""),
        str(row.get("claim") or row.get("contrast") or row.get("result") or ""),
        str(row.get("verdict") or row.get("outcome") or row.get("theory_relation") or ""),
    )


def classify_branch_evidence(
    rows_a: list[Mapping[str, Any]],
    rows_b: list[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """compatible / independent / conflicting. GENERATED never becomes WORLD."""
    compatible: list[dict[str, Any]] = []
    independent: list[dict[str, Any]] = []
    conflicting: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in list(rows_a) + list(rows_b):
        payload = dict(row)
        if str(payload.get("epistemic") or "") == "GENERATED":
            continue
        key = _evidence_key(payload)
        if key in seen:
            continue
        seen.add(key)
        match = None
        pool = rows_b if row in rows_a else rows_a
        for other in pool:
            other_key = _evidence_key(other)
            if other_key[0] == key[0] and other_key[0]:
                if other_key[2] and key[2] and other_key[2] != key[2]:
                    match = "conflict"
                    break
                if other_key == key:
                    match = "compatible"
                    break
        if match == "conflict":
            conflicting.append(payload)
        elif match == "compatible":
            compatible.append(payload)
        else:
            independent.append(payload)
    return {
        "compatible": compatible,
        "independent": independent,
        "conflicting": conflicting,
    }


def merge_branches(
    branches: list[TrajectoryBranch],
    state: ScientificState,
) -> dict[str, Any]:
    """Preserve all valid WORLD evidence. Score does not discard a branch."""
    world_rows: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    for branch in branches:
        for row in branch.world_evidence:
            if str(row.get("epistemic") or "") == "WORLD" and row.get("evidence_id"):
                world_rows.append(dict(row))
        for row in branch.generated:
            generated.append(dict(row))
    half = max(1, len(world_rows) // 2)
    split = classify_branch_evidence(world_rows[:half], world_rows[half:])
    if len(branches) >= 2:
        split = classify_branch_evidence(
            [dict(row) for row in branches[0].world_evidence],
            [dict(row) for row in branches[1].world_evidence],
        )
    merged: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    for row in split["compatible"] + split["independent"]:
        if str(row.get("epistemic") or "") != "WORLD":
            continue
        if not any(
            item.get("evidence_id") == row.get("evidence_id") for item in state.evidence_records
        ):
            state.record_evidence(row)
        merged.append(row)
    for row in split["conflicting"]:
        if str(row.get("epistemic") or "") != "WORLD":
            continue
        if not any(
            item.get("evidence_id") == row.get("evidence_id") for item in state.evidence_records
        ):
            state.record_evidence(row)
        merged.append(row)
        contradictions.append(
            {
                "question_id": str(row.get("question_id") or row.get("question") or ""),
                "evidence_id": str(row.get("evidence_id") or ""),
                "theory_id": str(row.get("theory_id") or ""),
                "verdict": str(row.get("verdict") or row.get("theory_relation") or ""),
            }
        )
        if contradictions[-1] not in state.contradictions:
            state.contradictions.append(dict(contradictions[-1]))
    return {
        "merged": merged,
        "contradictions": contradictions,
        "generated_kept_local": generated,
        "promoted_generated": False,
    }


def merge_branch(winner: TrajectoryBranch, state: ScientificState) -> list[dict[str, Any]]:
    """Backward-compatible wrapper. Prefer merge_branches for conflicts."""
    result = merge_branches([winner], state)
    return list(result["merged"])


def default_metrics() -> dict[str, Any]:
    return {
        "action_diversity": [],
        "non_scripted_transitions": 0,
        "scripted_transitions": 0,
        "question_resolution_rate": 0.0,
        "capability_growth": 0,
        "unlock_rate": 0,
        "harness_generalization": 0,
        "regression_rate": 0,
        "archive_reuse": 0,
        "lineage_progress": 0,
        "epistemic_integrity": 0,
    }


def update_metrics(
    state: ScientificState,
    *,
    action: ActionInstance | None = None,
    action_type: str = "",
    scripted: bool = False,
    unlock: bool = False,
    integrity_gap: str = "",
    archive_hit: bool = False,
    harness_general: bool = False,
    regression: bool = False,
) -> None:
    metrics = state.metrics if state.metrics else default_metrics()
    kind = (action.action_type if action is not None else "") or action_type
    if kind:
        seen = list(metrics.get("action_diversity") or [])
        if kind not in seen:
            seen.append(kind)
        metrics["action_diversity"] = seen
    if scripted:
        metrics["scripted_transitions"] = int(metrics.get("scripted_transitions") or 0) + 1
    else:
        metrics["non_scripted_transitions"] = int(metrics.get("non_scripted_transitions") or 0) + 1
    opened = max(1, len(state.open_questions) + len(state.resolved_questions))
    metrics["question_resolution_rate"] = len(state.resolved_questions) / opened
    metrics["capability_growth"] = len(state.available_capabilities)
    if unlock:
        metrics["unlock_rate"] = int(metrics.get("unlock_rate") or 0) + 1
    if archive_hit:
        metrics["archive_reuse"] = int(metrics.get("archive_reuse") or 0) + 1
    if harness_general:
        metrics["harness_generalization"] = int(metrics.get("harness_generalization") or 0) + 1
    if regression:
        metrics["regression_rate"] = int(metrics.get("regression_rate") or 0) + 1
    if integrity_gap:
        metrics["epistemic_integrity"] = int(metrics.get("epistemic_integrity") or 0) + 1
    else:
        metrics.setdefault("epistemic_integrity", 0)
    if state.resolved_questions and state.evidence_records:
        metrics["lineage_progress"] = int(metrics.get("lineage_progress") or 0) + 1
    opened = max(1, len(state.open_questions) + len(state.resolved_questions))
    world_ev = [
        row
        for row in state.evidence_records
        if str(row.get("epistemic") or "") == "WORLD" and row.get("evidence_id")
    ]
    verified = [row for row in world_ev if int(row.get("verify_count") or 0) >= 1]
    metrics["verified_evidence_yield"] = len(verified)
    metrics["theory_discrimination_rate"] = _discriminating(state.theories) / max(
        1, len(state.theories)
    )
    metrics["verification_depth"] = sum(int(row.get("verify_count") or 0) for row in world_ev)
    metrics["research_debt"] = dict(state.debt)
    metrics["contradiction_resolution_rate"] = (
        0.0
        if not (state.contradictions or verified)
        else (1.0 if not state.contradictions and verified else 0.0)
    )
    metrics["goal_drift_rate"] = float(metrics.get("goal_drift") or 0) / max(
        1, int(metrics.get("scientific_actions") or 1)
    )
    if kind in {"OBSERVE", "ACQUIRE", "THEORIZE", "PROBE", "VERIFY", "ASK", "SURVEY"}:
        metrics["scientific_actions"] = int(metrics.get("scientific_actions") or 0) + 1
    metrics.setdefault("meaningful_events", 0)
    state.metrics = metrics


def should_terminate(
    state: ScientificState,
    *,
    horizon: int,
    min_expected_value: float = 0.15,
) -> str | None:
    """Hard budget is a safety cap. Scientific exhaustion can stop earlier."""
    if int(state.ticks) >= int(horizon):
        return "budget"
    debt = state.debt or {}
    open_high = [
        row
        for row in state.open_questions
        if row.get("priority") in {None, "", "high"} or not row.get("priority")
    ]
    if (
        not open_high
        and not state.blocked_questions
        and int(debt.get("verification_debt") or 0) == 0
        and int(debt.get("contradiction_debt") or 0) == 0
        and state.resolved_questions
    ):
        return "frontier_exhausted"
    idle = int(state.ticks) - int(state.last_progress_tick)
    if idle >= STAGNATION_IDLE * 3 and state.resolved_questions and min_expected_value >= 0:
        return "stagnation"
    return None


def fork_action_reason() -> str:
    return FORK_LINEAGE
