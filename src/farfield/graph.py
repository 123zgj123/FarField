"""Kernel-managed citation/concept graph snapshots (INV-16).

The generator must not use node year to choose paths. Year exists so Wave C can
later split G_<=T; Wave B treats the snapshot as an untimed topology.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .ledger import content_digest

PINNED_K = 2


@dataclass(frozen=True)
class GraphSnapshot:
    snapshot_id: str
    k: int
    nodes: dict[str, dict[str, Any]]
    edges: tuple[tuple[str, str], ...]
    outgoing: dict[str, tuple[str, ...]]
    incoming: dict[str, tuple[str, ...]]
    counterexample_nodes: frozenset[str]


def load_graph(path: Path | str) -> GraphSnapshot:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    snapshot_id = str(payload.get("snapshot_id") or "").strip()
    if not snapshot_id:
        raise ValueError("graph snapshot_id must be non-empty")
    k = payload.get("k")
    if k != PINNED_K:
        raise ValueError(f"graph k must be pinned at {PINNED_K}, got {k!r}")
    nodes = {item["id"]: dict(item) for item in payload["nodes"]}
    if len(nodes) != len(payload["nodes"]):
        raise ValueError("graph node ids must be unique")
    edges: list[tuple[str, str]] = []
    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)
    for src, dst in payload["edges"]:
        if src not in nodes or dst not in nodes:
            raise ValueError(f"edge ({src}, {dst}) references a missing node")
        edges.append((src, dst))
        outgoing[src].append(dst)
        incoming[dst].append(src)
    counter = frozenset(payload.get("counterexample_nodes") or [])
    unknown = counter - set(nodes)
    if unknown:
        raise ValueError(f"counterexample nodes not in graph: {sorted(unknown)}")
    return GraphSnapshot(
        snapshot_id=snapshot_id,
        k=PINNED_K,
        nodes=nodes,
        edges=tuple(edges),
        outgoing={key: tuple(values) for key, values in outgoing.items()},
        incoming={key: tuple(values) for key, values in incoming.items()},
        counterexample_nodes=counter,
    )


def seed_neighborhood(
    graph: GraphSnapshot, seed_node_id: str, k: int = PINNED_K
) -> set[str]:
    if seed_node_id not in graph.nodes:
        return set()
    seen = {seed_node_id}
    frontier = {seed_node_id}
    for _ in range(k):
        nxt = set()
        for node in frontier:
            nxt.update(graph.outgoing.get(node, ()))
        nxt -= seen
        seen |= nxt
        frontier = nxt
    return seen


def shortest_path_tree(graph: GraphSnapshot, src: str) -> dict[str, str]:
    """Parent pointers for every node reachable from `src`, one BFS.

    Callers that want a path to every node were running one BFS per node, which
    is quadratic and became the wall on concept graphs with thousands of nodes.
    The enqueue order and the first-parent-wins rule are the same as
    `shortest_path`, so the reconstructed paths are the same paths.
    """
    parents: dict[str, str] = {}
    if src not in graph.nodes:
        return parents
    seen = {src}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for child in graph.outgoing.get(node, ()):
            if child in seen:
                continue
            seen.add(child)
            parents[child] = node
            queue.append(child)
    return parents


def path_from_tree(
    parents: dict[str, str], src: str, dst: str
) -> list[str] | None:
    if dst == src:
        return [src]
    if dst not in parents:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(parents[path[-1]])
    path.reverse()
    return path


def shortest_path(graph: GraphSnapshot, src: str, dst: str) -> list[str] | None:
    if src not in graph.nodes or dst not in graph.nodes:
        return None
    queue = deque([[src]])
    seen = {src}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == dst:
            return path
        for child in graph.outgoing.get(node, ()):
            if child in seen:
                continue
            seen.add(child)
            queue.append(path + [child])
    return None


def path_length(path: list[str]) -> int:
    if len(path) < 1:
        raise ValueError("path must contain at least one node")
    return len(path) - 1


def path_edges(path: list[str]) -> tuple[tuple[str, str], ...]:
    return tuple(zip(path, path[1:]))


def in_neighborhood(nodes: Iterable[str], neighborhood: set[str]) -> bool:
    return set(nodes) <= neighborhood


def node_degree(graph: GraphSnapshot, node_id: str) -> int:
    return len(graph.outgoing.get(node_id, ())) + len(graph.incoming.get(node_id, ()))


def independence_signature(
    nodes: Iterable[str], edges: Iterable[tuple[str, str]]
) -> str:
    return content_digest(
        {"nodes": sorted(nodes), "edges": sorted(tuple(edge) for edge in edges)}
    )[:16]
