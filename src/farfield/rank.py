"""Two-term scoring of candidate directions: plausibility and alienness.

Farfield's drift arm has only ever had the second term. `_distant_paths` hard
filters to paths longer than the k-hop, `anti_neighborhood` removes what is left
inside it, and `proximity_invert` prefers the least neighbourly survivor. Nothing
in that chain asks whether a direction is *likely*, so the arm sits at one fixed
corner of the design space and every measurement taken on it is a measurement of
that corner, not of far-field search.

Sourati and Evans (Nature Human Behaviour 2023) score a direction as

    score = T(plausibility) + beta * T(alienness)

and sweep beta. Their result is that both terms are needed. This module makes
the same score computable here so the sweep can actually be run, which turns
"drift picks no better than chance" from a claim about far-field search into a
claim about one point on a curve that can then be checked at the other points.

Every feature below reads the walker graph only: degrees, shared neighbours and
hop counts on `G_<=T`. None reads `year`, per the INV-16 note that the generator
treats the snapshot as untimed, and none can see `confirmed`, which lives in the
evaluator graph the walker is barred from.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Sequence

from .graph import GraphSnapshot, path_length

# The incumbent arm, kept nameable so a report can say which corner it sat in.
ALIEN_ONLY = "alien_only"

# Negative values are not a formality: if plausibility alone selects better than
# any mixture, the sweep should be able to say so.
BETA_GRID: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)


def neighbours(graph: GraphSnapshot, node: str) -> set[str]:
    """Undirected neighbours: citation direction does not matter for overlap."""
    return set(graph.outgoing.get(node, ())) | set(graph.incoming.get(node, ()))


def common_neighbours(graph: GraphSnapshot, u: str, v: str) -> set[str]:
    return neighbours(graph, u) & neighbours(graph, v)


def adamic_adar(graph: GraphSnapshot, u: str, v: str) -> float:
    """Shared neighbours, discounted by how promiscuous each one is."""
    total = 0.0
    for node in common_neighbours(graph, u, v):
        degree = len(neighbours(graph, node))
        if degree > 1:
            total += 1.0 / math.log(degree)
    return total


def jaccard(graph: GraphSnapshot, u: str, v: str) -> float:
    left, right = neighbours(graph, u), neighbours(graph, v)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def hop_counts(graph: GraphSnapshot, source: str) -> dict[str, int]:
    """Undirected hop counts, so a candidate two citations away is two hops.

    The directed count is what drift walks, but alienness is about how far apart
    two bodies of work sit, and a work that cites the seed is not alien merely
    because the arrow points the other way.
    """
    if source not in graph.nodes:
        return {}
    seen = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for other in neighbours(graph, node):
            if other not in seen:
                seen[other] = seen[node] + 1
                queue.append(other)
    return seen


def plausibility(
    graph: GraphSnapshot, seed_node: str, endpoint: str, hops: int
) -> float:
    """How much the pre-T structure already argues for this pairing.

    These are the link-prediction features that won Science4Cast, restricted to
    the ones computable from a single static snapshot: shared neighbours,
    discounted shared neighbours, endpoint prominence, and closeness. All four
    are properties of `G_<=T`, so using them does not leak the answer.
    """
    shared = len(common_neighbours(graph, seed_node, endpoint))
    return (
        math.log1p(shared)
        + adamic_adar(graph, seed_node, endpoint)
        + 0.5 * math.log1p(len(neighbours(graph, endpoint)))
        + (1.0 / hops if hops > 0 else 0.0)
    )


def alienness(
    graph: GraphSnapshot, seed_node: str, endpoint: str, hops: int
) -> float:
    """How far outside the seed's existing world the endpoint sits.

    Distance carries most of it, obscurity the rest: a famous work five hops away
    is a less alien proposal than an ignored one at the same distance, because
    the famous one is already in everybody's field of view.
    """
    degree = len(neighbours(graph, endpoint))
    return (
        float(hops)
        + (1.0 - jaccard(graph, seed_node, endpoint))
        + 1.0 / (1.0 + math.log1p(degree))
    )


def rank_transform(values: Sequence[float]) -> list[float]:
    """The `T` of the score: rank percentile in [0, 1], ties averaged.

    A rank transform rather than a z-score because the two terms are on
    unrelated scales, and beta is only interpretable if adding one unit of
    alienness is comparable to adding one unit of plausibility.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = mean_rank / (n - 1)
        i = j + 1
    return out


def score_paths(
    graph: GraphSnapshot,
    seed_node: str,
    paths: Sequence[Sequence[str]],
    beta: float,
) -> list[tuple[float, list[str]]]:
    """`T(plausibility) + beta * T(alienness)`, highest first.

    Ties break on path length then endpoint id, so two runs on the same snapshot
    return the same order and a beta sweep compares selections rather than
    shuffles.
    """
    hops = hop_counts(graph, seed_node)
    rows = [list(path) for path in paths]
    plaus_raw = []
    alien_raw = []
    for path in rows:
        endpoint = path[-1]
        distance = hops.get(endpoint, path_length(path))
        plaus_raw.append(plausibility(graph, seed_node, endpoint, distance))
        alien_raw.append(alienness(graph, seed_node, endpoint, distance))
    plaus = rank_transform(plaus_raw)
    alien = rank_transform(alien_raw)

    scored = [
        (plaus[i] + beta * alien[i], rows[i]) for i in range(len(rows))
    ]

    scored.sort(key=lambda item: (-item[0], path_length(item[1]), item[1][-1]))
    return scored


def order_by_beta(beta: float):
    """An ordering operator for `_drift_cards`, so the sweep reuses one pipeline."""

    def order(paths: Iterable[Sequence[str]], graph: GraphSnapshot) -> list[list[str]]:
        rows = [list(path) for path in paths]
        if not rows:
            return []
        seed_node = rows[0][0]
        return [path for _, path in score_paths(graph, seed_node, rows, beta)]

    return order
