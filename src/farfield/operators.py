"""Registered drift/judge operators. A skill is a digest → callable, not a prompt."""

from __future__ import annotations

import json
from pathlib import Path
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


def reset_invocations() -> None:
    INVOCATIONS.clear()


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


CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    PREFER_DEGREE_SPEC,
    PREFER_LENGTH_SPEC,
    PREFER_OUT_DEGREE_SPEC,
)

REGISTRY: dict[str, OperatorFn] = {
    spec_digest(PREFER_DEGREE_SPEC): prefer_endpoint_degree_ge_2,
    spec_digest(PREFER_LENGTH_SPEC): prefer_longest_distant_path,
    spec_digest(PREFER_OUT_DEGREE_SPEC): prefer_endpoint_onward_edges,
}

PREFER_DEGREE_DIGEST = spec_digest(PREFER_DEGREE_SPEC)
INSTALL_PATH = Path(".farfield") / "installed_operator.json"
HOLD_PATH = Path(".farfield") / "value_anchor_held.json"


def get_operator(digest: str) -> OperatorFn:
    try:
        return REGISTRY[digest]
    except KeyError as exc:
        raise KeyError(f"unknown operator digest: {digest}") from exc


def install_operator(
    project_dir: Path | str,
    digest: str,
    *,
    spec: dict[str, Any] | None = None,
) -> None:
    """Install a registry operator.

    ``digest`` keys the callable, so it cannot depend on the extracted spec.
    The extracted spec is recorded separately under its own digest, otherwise
    two campaigns with different evidence leave identical install records.
    """
    if digest not in REGISTRY:
        raise KeyError(f"unknown operator digest: {digest}")
    body = dict(spec or PREFER_DEGREE_SPEC)
    path = Path(project_dir) / INSTALL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "digest": digest,
                "spec": body,
                "spec_digest": spec_digest(body),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def hold_value_anchor(
    project_dir: Path | str, digest: str, reasons: tuple[str, ...]
) -> None:
    path = Path(project_dir) / HOLD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"digest": digest, "state": "VALUE_ANCHOR", "reasons": list(reasons)},
            indent=2,
        ),
        encoding="utf-8",
    )


def load_installed_digest(project_dir: Path | str) -> str | None:
    path = Path(project_dir) / INSTALL_PATH
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = str(payload.get("digest") or "")
    return digest or None
