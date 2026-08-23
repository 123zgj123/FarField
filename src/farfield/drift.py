"""Deterministic graph-walk card generator — the v2 control arm.

Under DESIGN_V2_ZH.md this module is no longer the system's idea generator;
it is the baseline the v2 generator (LLM far-field jumps over the embedding
space) must beat under the same anchor and budget. It stays exactly as
measured — drift vs random reach 0.359 vs 0.394, novelty hits 6.9% vs 7.9%
expected by chance — because a control arm that quietly improves stops
being a control.

``dump_seed_refs`` is an evaluation arm, not user-dossier content.
The generator does not read node year.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from .graph import (
    GraphSnapshot,
    PINNED_K,
    in_neighborhood,
    independence_signature,
    node_degree,
    path_edges,
    path_from_tree,
    path_length,
    seed_neighborhood,
    shortest_path,
    shortest_path_tree,
)
from .models import WAVE_A_PLACEHOLDERS, ConjectureCard, new_id

GenerateMode = Literal["drift", "dump_seed_refs", "random_far", "scored"]

# Four per mode, so a mode that finds nothing is visible as an empty column
# rather than as a bundle that silently borrowed another mode's cards.
CARDS_PER_MODE = 4
MAX_BRIDGES = 3


def _take(
    paths: list[list[str]],
    predicate: Any,
    used_endpoints: set[str],
    limit: int,
) -> list[list[str]]:
    """Top `limit` paths matching `predicate`, one per endpoint, in given order."""
    picked = []
    for path in paths:
        endpoint = path[-1]
        if endpoint in used_endpoints or not predicate(path):
            continue
        used_endpoints.add(endpoint)
        picked.append(path)
        if len(picked) >= limit:
            break
    return picked


@dataclass(frozen=True)
class DriftCard:
    card: ConjectureCard
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    mode: str
    prediction_core: str
    parent_signatures: tuple[str, ...] = ()
    probed: bool = False

    def to_conjecture_dict(self, seed: str) -> dict[str, object]:
        payload = self.card.to_dict()
        payload.update(
            {
                "dv": WAVE_A_PLACEHOLDERS["dv"],
                "seed_excerpt": seed,
                "nodes": list(self.nodes),
                "edges": [list(edge) for edge in self.edges],
                "mode": self.mode,
                "prediction_core": self.prediction_core,
                "parent_signatures": list(self.parent_signatures),
                "probed": self.probed,
            }
        )
        return payload


@dataclass(frozen=True)
class DriftBundle:
    cards: tuple[DriftCard, ...]
    neighborhood: frozenset[str]
    seed_node_id: str
    snapshot_id: str
    memory_text: str | None = None
    operator_digest: str | None = None
    operator_calls: int = 0
    strategies_run: int = 0


def generate(
    seed: str,
    graph: GraphSnapshot,
    *,
    mode: GenerateMode,
    seed_node_id: str,
    installed_operator_digest: str | None = None,
    memory_text: str | None = None,
    beta: float | None = None,
) -> DriftBundle:
    seed = seed.strip()
    if not seed:
        raise ValueError("seed must be non-empty")
    if seed_node_id not in graph.nodes:
        raise ValueError(f"seed node {seed_node_id!r} is not in snapshot {graph.snapshot_id}")
    if installed_operator_digest and memory_text:
        raise ValueError("dump-M and an installed operator cannot run in the same arm")
    if (mode == "scored") != (beta is not None):
        raise ValueError("beta belongs to the scored mode and only to it")
    neighborhood = seed_neighborhood(graph, seed_node_id, PINNED_K)
    operator_calls = 0
    strategies_run = 0
    if mode == "dump_seed_refs":
        cards = _dump_cards(seed, graph, seed_node_id, neighborhood)
    elif mode == "random_far":
        cards = _random_far_cards(seed, graph, seed_node_id, neighborhood)
    elif mode == "scored":
        cards = _scored_cards(seed, graph, seed_node_id, float(beta))
    elif mode == "drift":
        order = None
        if installed_operator_digest:
            from .operators import get_operator

            order = get_operator(installed_operator_digest)
            before = _operator_invocations()
            cards, strategies_run = _drift_cards(
                seed, graph, seed_node_id, neighborhood, order_distant=order
            )
            operator_calls = _operator_invocations() - before
        else:
            cards, strategies_run = _drift_cards(seed, graph, seed_node_id, neighborhood)
    else:
        raise ValueError(f"unknown generate mode: {mode}")
    return DriftBundle(
        cards=cards,
        neighborhood=frozenset(neighborhood),
        seed_node_id=seed_node_id,
        snapshot_id=graph.snapshot_id,
        memory_text=memory_text,
        operator_digest=installed_operator_digest,
        operator_calls=operator_calls,
        strategies_run=strategies_run,
    )


def _operator_invocations() -> int:
    from .operators import INVOCATIONS

    return sum(INVOCATIONS.values())


def prediction_core(prediction: str, nodes: Iterable[str], snapshot_id: str) -> str:
    text = " ".join(prediction.lower().split())
    text = text.replace(snapshot_id.lower(), " ")
    for node in sorted(nodes, key=len, reverse=True):
        text = text.replace(node.lower(), " ")
    text = re.sub("[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def anti_neighborhood(nodes: Iterable[str], neighborhood: set[str]) -> bool:
    """True iff the path is entirely inside the seed k-hop and must be dropped."""
    return in_neighborhood(nodes, neighborhood)


def proximity_score(path: list[str], neighborhood: set[str]) -> float:
    return sum(1 for node in path if node in neighborhood) / max(1, len(path))


def proximity_invert(
    paths: list[list[str]], neighborhood: set[str]
) -> list[list[str]]:
    """Cluster by endpoint; keep the lowest-proximity survivor."""
    best = {}
    for path in paths:
        end = path[-1]
        current = best.get(end)
        if current is None:
            best[end] = path
            continue
        left = (proximity_score(path, neighborhood), -path_length(path), path[-1])
        right = (
            proximity_score(current, neighborhood),
            -path_length(current),
            current[-1],
        )
        if left < right:
            best[end] = path
    return sorted(best.values(), key=lambda path: (path_length(path), path[-1]))


def multi_strategy_distant(distant: list[list[str]]) -> tuple[list[list[str]], int]:
    """Run at least two strategies before a proposal (nearest-distant and farthest)."""
    nearest = sorted(distant, key=lambda path: (path_length(path), path[-1]))
    farthest = sorted(distant, key=lambda path: (-path_length(path), path[-1]))
    merged = list({tuple(path): path for path in nearest + farthest}.values())
    return merged, 2


def is_neighborhood_replay(
    dump_card: DriftCard, drift_card: DriftCard, neighborhood: set[str]
) -> bool:
    return (
        drift_card.prediction_core == dump_card.prediction_core
        and anti_neighborhood(drift_card.nodes, neighborhood)
    )


def _dump_cards(
    seed: str,
    graph: GraphSnapshot,
    seed_node_id: str,
    neighborhood: set[str],
) -> tuple[DriftCard, ...]:
    cards = []
    seen = set()
    paths = [[seed_node_id]]
    for neighbor in graph.outgoing.get(seed_node_id, ()):
        paths.append([seed_node_id, neighbor])
    for path in paths:
        if not in_neighborhood(path, neighborhood):
            raise ValueError("dump_seed_refs walked outside the seed neighborhood")
        card = _card_from_path(
            seed,
            graph,
            path,
            mode="dump_seed_refs",
            operator="dump_seed_refs",
            distance=str(path_length(path)) if path_length(path) >= 1 else "1",
        )
        if card.card.independence_signature in seen:
            continue
        seen.add(card.card.independence_signature)
        cards.append(card)
    if not cards:
        raise ValueError("dump_seed_refs arm produced no inspectable conjecture")
    return tuple(cards)


def _endpoint_mode(graph: GraphSnapshot, endpoint: str) -> str:
    if endpoint in graph.counterexample_nodes:
        return "falsifier"
    if node_degree(graph, endpoint) == 1:
        return "anomaly"
    return "mechanism"


def _distant_paths(
    graph: GraphSnapshot, seed_node_id: str, neighborhood: set[str]
) -> list[list[str]]:
    parents = shortest_path_tree(graph, seed_node_id)
    paths = []
    for node_id in graph.nodes:
        if node_id == seed_node_id:
            continue
        path = path_from_tree(parents, seed_node_id, node_id)
        if path is None or path_length(path) <= PINNED_K:
            continue
        if anti_neighborhood(path, neighborhood):
            continue
        paths.append(path)
    return paths


def has_far_field(graph: GraphSnapshot, seed_node_id: str) -> bool:
    """Is there anything outside this seed's k-hop for drift to reach?

    Callers freeze a judgment clause before generating, and a frozen clause is a
    promise, so the promise has to be checkable up front.
    """
    if seed_node_id not in graph.nodes:
        return False
    neighborhood = seed_neighborhood(graph, seed_node_id, PINNED_K)
    return bool(_distant_paths(graph, seed_node_id, neighborhood))


def _random_far_cards(
    seed: str,
    graph: GraphSnapshot,
    seed_node_id: str,
    neighborhood: set[str],
) -> tuple[DriftCard, ...]:
    """The control arm for reach: distant, but chosen by coin flip.

    `dump_seed_refs` cannot leave the seed k-hop, so comparing reach against it
    compares two definitions rather than two behaviours, and the audit had to
    drop the column. This arm draws from exactly the same pool of distant paths
    and labels each card by what its endpoint actually is, so both arms face the
    same executable predicates. What is left between them is the only thing
    drift claims: which distant endpoints are worth proposing.
    """
    import random

    candidates = _distant_paths(graph, seed_node_id, neighborhood)
    if not candidates:
        raise ValueError("random_far arm produced no inspectable conjecture")
    candidates.sort(key=lambda path: (path_length(path), path[-1]))

    rng = random.Random(f"{graph.snapshot_id}:{seed_node_id}:random_far")
    budget = min(len(candidates), CARDS_PER_MODE * 3)
    picked = rng.sample(candidates, budget)
    cards = []
    seen = set()
    used = set()
    for path in picked:
        endpoint = path[-1]
        if endpoint in used:
            continue
        used.add(endpoint)
        card = _card_from_path(
            seed,
            graph,
            path,
            mode=_endpoint_mode(graph, endpoint),
            operator="random_far",
            distance=str(path_length(path)),
        )
        if card.card.independence_signature in seen:
            continue
        seen.add(card.card.independence_signature)
        cards.append(card)
    if not cards:
        raise ValueError("random_far arm produced no inspectable conjecture")
    return tuple(cards)


def _reachable_paths(graph: GraphSnapshot, seed_node_id: str) -> list[list[str]]:
    """Every node the seed can reach, near ones included.

    The drift pool is filtered to beyond the k-hop before anything is ranked, so
    an ordering rule applied to it can only reshuffle alien candidates. A beta
    sweep has to be able to choose a near candidate when beta is low, otherwise
    the low-beta end of the curve is unreachable by construction and the sweep
    measures nothing.
    """
    parents = shortest_path_tree(graph, seed_node_id)
    paths = []
    for node_id in graph.nodes:
        if node_id == seed_node_id:
            continue
        path = path_from_tree(parents, seed_node_id, node_id)
        if path is None:
            continue
        paths.append(path)
    return paths


def _scored_cards(
    seed: str,
    graph: GraphSnapshot,
    seed_node_id: str,
    beta: float,
) -> tuple[DriftCard, ...]:
    """`T(plausibility) + beta * T(alienness)` over the whole reachable pool.

    Cards are labelled by what their endpoint actually is, so every arm on the
    sweep faces the same executable predicates as the drift arm does.
    """
    from .rank import score_paths

    candidates = _reachable_paths(graph, seed_node_id)
    if not candidates:
        raise ValueError("scored arm produced no inspectable conjecture")
    cards = []
    seen = set()
    used = set()
    budget = CARDS_PER_MODE * 3
    for _, path in score_paths(graph, seed_node_id, candidates, beta):
        endpoint = path[-1]
        if endpoint in used:
            continue
        card = _card_from_path(
            seed,
            graph,
            path,
            mode=_endpoint_mode(graph, endpoint),
            operator=f"scored_beta_{beta:+.2f}",
            distance=str(path_length(path)),
        )
        if card.card.independence_signature in seen:
            continue
        used.add(endpoint)
        seen.add(card.card.independence_signature)
        cards.append(card)
        if len(cards) >= budget:
            break
    if not cards:
        raise ValueError("scored arm produced no inspectable conjecture")
    return tuple(cards)


def _drift_cards(
    seed: str,
    graph: GraphSnapshot,
    seed_node_id: str,
    neighborhood: set[str],
    *,
    order_distant: Any = None,
) -> tuple[tuple[DriftCard, ...], int]:
    distant = _distant_paths(graph, seed_node_id, neighborhood)
    merged, strategies_run = multi_strategy_distant(distant)
    inverted = proximity_invert(merged, neighborhood)
    if order_distant is not None:
        inverted = list(order_distant(inverted, graph))
    else:
        inverted.sort(key=lambda path: (path_length(path), path[-1]))

    used = set()
    chosen = []
    for mode, predicate in (
        ("mechanism", lambda path: node_degree(graph, path[-1]) >= 2),
        ("anomaly", lambda path: node_degree(graph, path[-1]) == 1),
        ("falsifier", lambda path: path[-1] in graph.counterexample_nodes),
    ):
        for path in _take(inverted, predicate, used, CARDS_PER_MODE):
            chosen.append((mode, path, "distant_path", str(path_length(path))))

    cards = []
    seen = set()
    for mode, path, operator, distance in chosen:
        card = _card_from_path(seed, graph, path, mode=mode, operator=operator, distance=distance)
        if card.card.independence_signature in seen:
            continue
        seen.add(card.card.independence_signature)
        cards.append(card)

    parents = [card for card in cards if card.mode in {"mechanism", "anomaly"}]
    for index in range(0, min(len(parents) - 1, MAX_BRIDGES * 2), 2):
        left, right = parents[index], parents[index + 1]
        if left.card.independence_signature == right.card.independence_signature:
            continue
        bridge = _crossover_card(seed, graph, left, right)

        if anti_neighborhood(bridge.nodes, neighborhood):
            continue
        if bridge.card.independence_signature not in seen:
            seen.add(bridge.card.independence_signature)
            cards.append(bridge)
    if not cards:
        raise ValueError("drift arm produced no inspectable conjecture")
    return tuple(cards), strategies_run


FALSIFIER_COST_BY_MODE = {
    "mechanism": "graph inspection of onward edges leaving the seed k-hop",
    "anomaly": "graph inspection of endpoint in/out degree",
    "falsifier": "graph inspection of counterexample parents inside the k-hop",
    "dump_seed_refs": "graph inspection of the seed bibliography",
    "bridge": "graph inspection of contact between the two parent endpoints",
}


def _card_from_path(
    seed: str,
    graph: GraphSnapshot,
    path: list[str],
    *,
    mode: str,
    operator: str,
    distance: str,
) -> DriftCard:
    endpoint = path[-1]
    title = str(graph.nodes[endpoint]["title"])
    nodes = tuple(path)
    edges = path_edges(path)
    signature = independence_signature(nodes, edges)
    source_id = f"{graph.snapshot_id}:{endpoint}"
    if mode == "falsifier":
        prediction = (
            f"the distant counterexample node {endpoint} should falsify the seed mechanism rather than support it. seed: "
            f"{seed}"
        )
        falsifier = (
            f"check that {endpoint} is a declared counterexample node and that no node inside the seed "
            f"{PINNED_K}-hop cites it directly; a near-implied counterexample is not a distant falsifier. do not treat the near-field sandbox as confirmation of this path"
        )
    elif mode == "dump_seed_refs":
        prediction = (
            f"reading seed bibliography node {endpoint} ({title}) is sufficient to restate the seed task. seed: "
            f"{seed}"
        )
        falsifier = "replace the node with a distant path; the restated seed task should change"
    elif mode == "anomaly":
        prediction = (
            f"an anomalous distant leaf {endpoint} ({title}) is inspectable from seed "
            f"{seed} and should not be treated as a mechanism"
        )
        falsifier = (
            f"check that {endpoint} is a sink with at least one citing parent; a total degree of 1 alone does not make a leaf. the near-field sandbox does not probe this path"
        )
    elif mode == "mechanism":
        prediction = (
            f"a mechanism bridge through distant node {endpoint} ({title}) is inspectable from seed "
            f"{seed}"
        )
        falsifier = (
            f"check that {endpoint} has an outgoing edge leaving the seed "
            f"{PINNED_K}-hop; a bridge that goes nowhere is not a mechanism bridge. the near-field sandbox does not probe this path"
        )
    else:
        raise ValueError(f"unknown path card mode: {mode}")
    core = prediction_core(prediction, nodes, graph.snapshot_id)
    card_id = new_id("conj")
    card = ConjectureCard(
        statement=f"{mode} conjecture over path {' -> '.join(path)}",
        prediction=prediction,
        cheapest_falsifier=falsifier,
        falsifier_cost=FALSIFIER_COST_BY_MODE[mode],
        source_id=source_id,
        span_start=0,
        span_end=len(title),
        independence_signature=signature,
        graph_distance=distance,
        drift_operator=operator,
        id=card_id,
    )
    return DriftCard(
        card=card,
        nodes=nodes,
        edges=edges,
        mode=mode,
        prediction_core=core,
    )


def _crossover_card(seed: str, graph: GraphSnapshot, left: DriftCard, right: DriftCard) -> DriftCard:
    nodes = tuple(dict.fromkeys(left.nodes + right.nodes))
    edges = tuple(dict.fromkeys(left.edges + right.edges))
    signature = independence_signature(nodes, edges)
    endpoint = right.nodes[-1]
    title = str(graph.nodes[endpoint]["title"])
    source_id = f"{graph.snapshot_id}:{endpoint}"
    distance = f"crossover:{_numeric_distance(left.card.graph_distance)}+{_numeric_distance(right.card.graph_distance)}"
    prediction = (
        f"crossing parent paths ending at {left.nodes[-1]} and {right.nodes[-1]} yields an inspectable bridge. seed: "
        f"{seed}"
    )
    core = prediction_core(prediction, nodes, graph.snapshot_id)
    card = ConjectureCard(
        statement=f"bridge conjecture crossing {left.nodes[-1]} and {right.nodes[-1]}",
        prediction=prediction,
        cheapest_falsifier=(
            f"check that {left.nodes[-1]} and {right.nodes[-1]} have structural contact beyond the shared prefix, i.e. a connecting path or a common descendant; otherwise the node union is a stapling, not a bridge"
        ),
        falsifier_cost=FALSIFIER_COST_BY_MODE["bridge"],
        source_id=source_id,
        span_start=0,
        span_end=len(title),
        independence_signature=signature,
        graph_distance=distance,
        drift_operator="crossover",
        id=new_id("conj"),
    )
    return DriftCard(
        card=card,
        nodes=nodes,
        edges=edges,
        mode="bridge",
        prediction_core=core,
        parent_signatures=(
            left.card.independence_signature,
            right.card.independence_signature,
        ),
    )


def _numeric_distance(value: str) -> int:
    if value.isdigit():
        return int(value)
    return 0
