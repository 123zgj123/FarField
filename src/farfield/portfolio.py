"""Diversity-preserving scheduler for speculative research candidates."""

from __future__ import annotations

from .models import Candidate, CandidateMode


def _dominates(left: Candidate, right: Candidate) -> bool:
    beneficial_left = (
        left.importance,
        left.expected_information_gain,
        left.falsifiability,
        left.plausibility,
    )
    beneficial_right = (
        right.importance,
        right.expected_information_gain,
        right.falsifiability,
        right.plausibility,
    )
    no_worse = all(a >= b for a, b in zip(beneficial_left, beneficial_right))
    no_worse = no_worse and left.cost <= right.cost
    strictly_better = any(a > b for a, b in zip(beneficial_left, beneficial_right))
    strictly_better = strictly_better or left.cost < right.cost
    return no_worse and strictly_better


def pareto_front(candidates: list[Candidate]) -> list[Candidate]:
    for candidate in candidates:
        candidate.validate()
    return [
        candidate
        for candidate in candidates
        if not any(
            _dominates(other, candidate)
            for other in candidates
            if other.id != candidate.id
        )
    ]


def _priority(candidate: Candidate) -> float:
    # Novelty is intentionally absent. "Unknown" novelty is an audit obligation,
    # not free value. This score only orders a Pareto layer after dominance tests.
    epistemic_value = (
        0.35 * candidate.importance
        + 0.30 * candidate.expected_information_gain
        + 0.25 * candidate.falsifiability
        + 0.10 * candidate.plausibility
    )
    return epistemic_value / candidate.cost


def select_portfolio(candidates: list[Candidate], budget: float) -> list[Candidate]:
    """Select a bounded, diverse portfolio without collapsing value to one metric.

    The scheduler first uses Pareto dominance, then preserves distinct mechanism
    signatures and tries to reserve a slot for an explicit falsifier. It is a
    transparent baseline policy, not an oracle for scientific importance.
    """

    if budget <= 0:
        raise ValueError("budget must be positive")
    for candidate in candidates:
        candidate.validate()

    layers: list[Candidate] = pareto_front(candidates)
    remainder = [candidate for candidate in candidates if candidate not in layers]
    ordered = sorted(layers, key=_priority, reverse=True) + sorted(
        remainder, key=_priority, reverse=True
    )

    chosen: list[Candidate] = []
    signatures: set[str] = set()
    spent = 0.0

    falsifiers = [
        candidate for candidate in ordered if candidate.mode is CandidateMode.FALSIFIER
    ]
    if falsifiers and falsifiers[0].cost <= budget:
        first = falsifiers[0]
        chosen.append(first)
        signatures.add(first.independence_signature)
        spent += first.cost

    for candidate in ordered:
        if candidate in chosen:
            continue
        if spent + candidate.cost > budget:
            continue
        if candidate.independence_signature in signatures:
            continue
        chosen.append(candidate)
        signatures.add(candidate.independence_signature)
        spent += candidate.cost

    return chosen
