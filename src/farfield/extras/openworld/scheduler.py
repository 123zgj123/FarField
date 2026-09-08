"""State-driven scheduler: what can be done now, not what the pipeline says next.

Same scheduler as before — extended with filters and dependency-aware value.
This is not a third control plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Iterable

from .actions import (
    ACQUIRE,
    ASK,
    EVOLVE_HARNESS,
    FORK_LINEAGE,
    OBSERVE,
    PROBE,
    THEORIZE,
    VERIFY,
    ActionInstance,
)
from .frontier import MAINTENANCE_ACTIONS, SCIENTIFIC_ACTIONS, blocked_descendants, traces_to_root
from .state import ScientificState


WEIGHTS = {
    "heuristic_information_value": 1.2,
    "expected_unlock_value": 1.6,
    "scientific_relevance": 1.0,
    "novelty": 0.6,
    "execution_cost": -0.8,
    "uncertainty": 0.4,
    "downstream_branching": 0.7,
    "evidence_quality": 1.1,
    "failure_recoverability": 0.5,
    "verification_value": 1.3,
    "debt_pressure": 1.4,
}


@dataclass(frozen=True)
class ScoredAction:
    action: ActionInstance
    scores: dict[str, float]
    total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "scores": dict(self.scores),
            "total": self.total,
        }


def epistemic_safe(action: ActionInstance, state: ScientificState) -> bool:
    kind = action.action_type
    if kind == PROBE and not (
        action.extra.get("causal_handle") and action.extra.get("measure") and (
            action.extra.get("freeze") or state.world_id
        )
    ):
        return False
    if kind in SCIENTIFIC_ACTIONS:
        target = action.frontier_target_id or action.target or state.goal
        if not target:
            return False
        if target and not traces_to_root(state, target) and target != state.goal:
            # still allow a brand-new question id that is already on the state
            known = {str(row.get("id") or "") for row in state.open_questions + state.resolved_questions}
            if target not in known and target not in {state.goal, "goal:G0", "G0"}:
                return False
    return True


def budget_ok(action: ActionInstance, state: ScientificState) -> bool:
    spent = float((state.metrics or {}).get("spent_cost") or 0)
    cap = float((state.metrics or {}).get("budget_cap") or 1e9)
    return spent + float(action.spec.estimated_cost) <= cap


def expected_frontier_advancement(action: ActionInstance, state: ScientificState) -> float:
    kind = action.action_type
    unlock = blocked_descendants(state, kind, action.target)
    immediate = 0.9 if kind in {OBSERVE, PROBE, VERIFY} else (0.25 if kind == ACQUIRE else 0.15)
    if kind == ACQUIRE and unlock >= 2:
        return immediate + 1.2 * unlock
    if kind == EVOLVE_HARNESS and state.missing_capabilities:
        return immediate + 0.8 * unlock
    if kind == VERIFY and (state.contradictions or (state.debt or {}).get("verification_debt")):
        return immediate + 0.9 * max(1, unlock)
    return immediate + 0.25 * unlock


def _score(action: ActionInstance, state: ScientificState) -> dict[str, float]:
    kind = action.action_type
    blocked = kind == ACQUIRE and action.reason == "blocked_on_capability_or_world"
    unlocks_observe = blocked or kind == EVOLVE_HARNESS
    observe = kind == OBSERVE
    probe = kind == PROBE
    fork = kind == FORK_LINEAGE
    debt = state.debt or {}
    verify_debt = float(debt.get("verification_debt") or 0)
    contra = float(debt.get("contradiction_debt") or 0)
    dep = float(debt.get("unresolved_dependency") or 0)
    orphan = float(debt.get("orphan_theory_debt") or 0)
    advancement = expected_frontier_advancement(action, state)
    debt_pressure = 0.1
    if kind == VERIFY and (verify_debt or contra):
        debt_pressure = 1.0
    elif kind == ACQUIRE and dep:
        debt_pressure = 0.9
    elif kind == THEORIZE and (
        contra or orphan or (state.resolved_questions and not state.theories)
    ):
        debt_pressure = 0.85
    elif kind == FORK_LINEAGE and state.resolved_questions and not state.theories:
        debt_pressure = 0.05
    elif kind == ASK:
        debt_pressure = 0.05
    verification_value = 0.9 if kind == VERIFY else (0.4 if observe or probe else 0.1)
    if observe and state.open_questions:
        debt_pressure = max(debt_pressure, 0.85)
    if kind == VERIFY and state.open_questions:
        debt_pressure = min(debt_pressure, 0.25)
        verification_value = 0.35
    return {
        "heuristic_information_value": advancement if kind == ACQUIRE else (
            0.9 if observe or probe else (0.5 if kind == "SURVEY" else 0.35)
        ),
        "expected_unlock_value": 1.0 if unlocks_observe else (0.2 if observe else 0.1 * advancement),
        "scientific_relevance": 0.8 if action.target or action.frontier_target_id else 0.4,
        "novelty": 0.15 if kind == ASK else (0.3 if fork else 0.2),
        "execution_cost": float(action.spec.estimated_cost),
        "uncertainty": 0.6 if kind == EVOLVE_HARNESS else 0.3,
        "downstream_branching": 0.8 if unlocks_observe or fork else min(0.9, 0.2 * advancement),
        "evidence_quality": 0.9 if observe or probe or kind == VERIFY else 0.2,
        "failure_recoverability": 0.7 if kind in {ACQUIRE, EVOLVE_HARNESS, "REFLECT"} else 0.4,
        "verification_value": verification_value,
        "debt_pressure": debt_pressure,
    }


def hypothesis_resolution_priority(proposal: dict[str, Any], state: ScientificState) -> dict[str, Any]:
    """Estimate discriminating value, never evidence or calibrated information gain.

    Resolution is the posterior-weighted mass of competing pairs separated by
    the declared intervention: sum(2*w_i*w_j), with weights normalized over the
    currently known theories. Belief confidence is a replayable bookkeeping
    scale, not a joint probability distribution. This pair-disagreement proxy
    does not assume a likelihood model that the runtime has not measured.

    Explicit competitor edges restrict eligible pairs; with no recorded edges,
    a proposal may declare a comparison between known theories. Different
    observables or uncertain predictions do not establish discrimination.
    All gain/cost factors are proposal estimates, not measured world values.
    """
    def invalid(reason: str) -> dict[str, Any]:
        return {"valid": False, "reason": reason, "priority": 0.0,
                "score_basis": "proposal_estimate_not_evidence"}

    def number(value: Any, upper: float | None = None) -> bool:
        try:
            return (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0
                    and (upper is None or value <= upper))
        except OverflowError:
            return False

    if not isinstance(proposal, dict):
        return invalid("experiment_proposal_must_be_object")
    known = {row["id"]: row for row in state.theories
             if isinstance(row.get("id"), str) and row["id"]}
    compared = proposal.get("hypotheses_compared")
    if (not isinstance(compared, list) or len(compared) < 2
            or any(not isinstance(hid, str) or hid not in known for hid in compared)
            or len(set(compared)) != len(compared)):
        return invalid("comparison_requires_distinct_known_hypotheses")
    predictions = proposal.get("expected_outcomes")
    if not isinstance(predictions, dict) or set(predictions) != set(compared):
        return invalid("predictions_must_match_compared_hypotheses")
    for prediction in predictions.values():
        if (not isinstance(prediction, dict)
                or not isinstance(prediction.get("observable"), str)
                or not prediction["observable"].strip()
                or prediction.get("direction") not in ("increase", "decrease", "unchanged")):
            return invalid("prediction_requires_observable_and_determinate_direction")
    if len({p["observable"].strip().casefold() for p in predictions.values()}) != 1:
        return invalid("comparison_requires_same_observable")
    nuisance = proposal.get("nuisance_factors")
    if not isinstance(nuisance, list) or any(not isinstance(x, str) or not x.strip() for x in nuisance):
        return invalid("nuisance_factors_must_be_explicit_list")
    intervention = proposal.get("minimal_discriminating_intervention")
    if not isinstance(intervention, str) or not intervention.strip():
        return invalid("minimal_discriminating_intervention_required")
    interpretations = proposal.get("interpretations", {})
    if not isinstance(interpretations, dict):
        return invalid("interpretations_must_be_object")
    for outcome in ("positive", "negative", "inconclusive"):
        value = proposal.get("interpretation_if_" + outcome, interpretations.get(outcome))
        if not isinstance(value, str) or not value.strip():
            return invalid("interpretation_required_for_" + outcome)
    for name in ("expected_scientific_gain", "transfer_value", "frontier_unlock", "cost", "risk"):
        if not number(proposal.get(name), None if name in {"cost", "risk"} else 1.0):
            return invalid("invalid_estimate_" + name)
    weights: dict[str, float] = {}
    competing_pairs: set[frozenset[str]] = set()
    for hid, theory in known.items():
        belief = state.belief_states.get(hid, {})
        confidence = belief.get("posterior_confidence") if isinstance(belief, dict) else None
        if not number(confidence, 1.0):
            return invalid("missing_or_invalid_replayed_belief_" + hid)
        weights[hid] = float(confidence)
        competitors = belief.get("competing_theories") or theory.get("competing") or []
        for other in competitors if isinstance(competitors, list) else []:
            other_id = other.get("id") if isinstance(other, dict) else other
            if isinstance(other_id, str) and other_id in known and other_id != hid:
                competing_pairs.add(frozenset((hid, other_id)))
    mass = sum(weights.values())
    resolution = 0.0
    separated: list[list[str]] = []
    for left, right in combinations(compared, 2):
        if competing_pairs and frozenset((left, right)) not in competing_pairs:
            continue
        if predictions[left]["direction"] != predictions[right]["direction"]:
            separated.append([left, right])
            if mass:
                resolution += 2.0 * (weights[left] / mass) * (weights[right] / mass)
    terms = {name: float(proposal[name]) for name in
             ("expected_scientific_gain", "transfer_value", "frontier_unlock", "cost", "risk")}
    priority = (resolution * terms["expected_scientific_gain"] * terms["transfer_value"]
                * terms["frontier_unlock"] - terms["cost"] - terms["risk"])
    if not math.isfinite(priority):
        return invalid("nonfinite_priority")
    return {"valid": True, "reason": "", "score_basis": "proposal_estimate_not_evidence",
            "expected_hypothesis_resolution": resolution, "separated_pairs": separated,
            "belief_weights": weights, **terms, "priority": priority}


def score_action(action: ActionInstance, state: ScientificState) -> ScoredAction:
    scores = _score(action, state)
    total = sum(WEIGHTS[name] * scores[name] for name in WEIGHTS)
    if "experiment_proposal" in action.extra:
        priority = hypothesis_resolution_priority(action.extra["experiment_proposal"], state)
        if not priority["valid"]:
            raise ValueError("invalid experiment proposal: " + priority["reason"])
        scores.update({name: priority[name] for name in (
            "expected_hypothesis_resolution", "expected_scientific_gain", "transfer_value",
            "frontier_unlock", "cost", "risk")})
        scores["hypothesis_resolution_priority"] = priority["priority"]
        scores["proposal_estimate_not_evidence"] = 1.0
        total += priority["priority"]
    return ScoredAction(action=action, scores=scores, total=total)


def _valid_experiment(action: ActionInstance, state: ScientificState) -> bool:
    return ("experiment_proposal" not in action.extra
            or hypothesis_resolution_priority(action.extra["experiment_proposal"], state)["valid"])


def schedule(
    state: ScientificState,
    actions: Iterable[ActionInstance],
) -> ScoredAction | None:
    """Heuristic research-value scheduling. Not immediate-score-only."""
    filtered = [
        item
        for item in actions
        if (item.action_type in MAINTENANCE_ACTIONS or epistemic_safe(item, state))
        and budget_ok(item, state)
        and _valid_experiment(item, state)
    ]
    ranked = sorted(
        (score_action(item, state) for item in filtered),
        key=lambda row: (-row.total, row.action.action_type, row.action.target),
    )
    return ranked[0] if ranked else None


def schedule_all(
    state: ScientificState,
    actions: Iterable[ActionInstance],
) -> list[ScoredAction]:
    filtered = [
        item
        for item in actions
        if (item.action_type in MAINTENANCE_ACTIONS or epistemic_safe(item, state))
        and budget_ok(item, state)
        and _valid_experiment(item, state)
    ]
    return sorted(
        (score_action(item, state) for item in filtered),
        key=lambda row: (-row.total, row.action.action_type, row.action.target),
    )
