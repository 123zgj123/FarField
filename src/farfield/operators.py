"""Registered drift/judge operators. A skill is a digest → callable, not a prompt."""

from __future__ import annotations

from typing import Any, Callable

from .graph import GraphSnapshot, node_degree, path_length
from .ledger import content_digest

DistantPaths = list[list[str]]
OperatorFn = Callable[[DistantPaths, GraphSnapshot], DistantPaths]

INVOCATIONS: dict[str, int] = {}

PREFER_DEGREE_SPEC: dict[str, Any] = {
    "name": "prefer_endpoint_degree_ge_2",
    "prefer_degree_at_least": 2,
    "kind": "drift_skill",
}

PREFER_LENGTH_SPEC: dict[str, Any] = {
    "name": "prefer_longest_distant_path",
    "prefer": "path_length_desc",
    "kind": "drift_skill",
}

PREFER_OUT_DEGREE_SPEC: dict[str, Any] = {
    "name": "prefer_endpoint_onward_edges",
    "prefer": "endpoint_out_degree_desc",
    "kind": "drift_skill",
}


def spec_digest(spec: dict[str, Any]) -> str:
    return content_digest(spec)


def endpoint_out_degree(graph: GraphSnapshot, node_id: str) -> int:
    return len(graph.outgoing.get(node_id, ()))


def _ordering(name: str, key: Any) -> OperatorFn:
    """A registered ordering operator. The ranking key is the whole policy."""

    def operator(distant: DistantPaths, graph: GraphSnapshot) -> DistantPaths:
        INVOCATIONS[name] = INVOCATIONS.get(name, 0) + 1
        return sorted(distant, key=lambda path: key(path, graph))

    operator.__name__ = name
    return operator


prefer_endpoint_degree_ge_2 = _ordering(
    PREFER_DEGREE_SPEC["name"],
    lambda path, graph: (-node_degree(graph, path[-1]), path_length(path), path[-1]),
)

prefer_longest_distant_path = _ordering(
    PREFER_LENGTH_SPEC["name"],
    lambda path, graph: (-path_length(path), path[-1]),
)

prefer_endpoint_onward_edges = _ordering(
    PREFER_OUT_DEGREE_SPEC["name"],
    lambda path, graph: (
        -endpoint_out_degree(graph, path[-1]),
        path_length(path),
        path[-1],
    ),
)


REGISTRY: dict[str, OperatorFn] = {
    spec_digest(PREFER_DEGREE_SPEC): prefer_endpoint_degree_ge_2,
    spec_digest(PREFER_LENGTH_SPEC): prefer_longest_distant_path,
    spec_digest(PREFER_OUT_DEGREE_SPEC): prefer_endpoint_onward_edges,
}


def get_operator(digest: str) -> OperatorFn:
    try:
        return REGISTRY[digest]
    except KeyError as exc:
        raise KeyError(f"unknown operator digest: {digest}") from exc
