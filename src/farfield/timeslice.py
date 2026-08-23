"""Time-slice corpus split (Wave C). Walker never loads held-out labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .graph import GraphSnapshot, load_graph

PINNED_T = 2017
MIN_TIMESLICE_SEEDS = 3


class HeldOutAccessError(PermissionError):
    """Walker attempted to resolve a post-T confirmed node."""


@dataclass(frozen=True)
class HeldOutSet:
    T: int
    confirmed: frozenset[str]
    path: str

    def contains(self, node_id: str) -> bool:
        return node_id in self.confirmed


def load_heldout(path: Path | str, *, t: int = PINNED_T) -> HeldOutSet:
    """Load a held-out set, refusing one cut at a different year than declared.

    `t` defaults to the kernel pin and callers only pass something else when
    the corpus itself declares another cut (a concept slice harvested from
    mid-2015 cannot support 2017). The protection is unchanged either way:
    the expected cut comes from a declaration, never from the file being
    loaded, so a file cannot smuggle in its own more permissive year.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    t_found = payload.get("T")
    if t_found != t:
        raise ValueError(f"held-out T must be pinned at {t}, got {t_found!r}")
    confirmed = payload.get("confirmed")
    if not isinstance(confirmed, list) or not confirmed:
        raise ValueError("held-out confirmed must be a non-empty list of node ids")
    ids = [str(item).strip() for item in confirmed]
    if any(not item for item in ids):
        raise ValueError("held-out confirmed ids must be non-empty")
    if len(set(ids)) != len(ids):
        raise ValueError("held-out confirmed ids must be unique")
    return HeldOutSet(T=t, confirmed=frozenset(ids), path=str(Path(path)))


def derive_g_le_t(payload: dict[str, Any], t: int = PINNED_T) -> dict[str, Any]:
    nodes = [dict(node) for node in payload["nodes"] if int(node["year"]) <= t]
    kept = {node["id"] for node in nodes}
    edges = [list(edge) for edge in payload["edges"] if edge[0] in kept and edge[1] in kept]
    counter = [node for node in payload.get("counterexample_nodes") or [] if node in kept]
    return {
        "snapshot_id": f"{payload['snapshot_id']}-le-{t}",
        "k": payload["k"],
        "T": t,
        "derived_from": payload["snapshot_id"],
        "nodes": nodes,
        "edges": edges,
        "counterexample_nodes": counter,
    }


def load_walker_graph(path: Path | str, *, t: int = PINNED_T) -> GraphSnapshot:
    """Load G_<=T. This function does not accept a held-out file.

    `t` is the corpus's declared cut, defaulting to the kernel pin. Passing
    it does not weaken the check: the expected year comes from the caller's
    declaration, never from the file being loaded.
    """
    graph = load_graph(path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for node in payload["nodes"]:
        year = node.get("year")
        if year is None:
            raise ValueError(f"walker graph node {node['id']!r} missing year")
        if int(year) > t:
            raise ValueError(
                f"walker graph must not contain year > {t}: {node['id']}"
            )
    return graph


def walker_lookup(
    graph: GraphSnapshot,
    node_id: str,
    *,
    heldout: HeldOutSet | None = None,
) -> dict[str, Any]:
    if heldout is not None and heldout.contains(node_id):
        raise HeldOutAccessError(
            f"walker cannot access held-out confirmed node {node_id!r}"
        )
    if node_id not in graph.nodes:
        raise KeyError(node_id)
    return graph.nodes[node_id]


def load_evaluator_graph(path: Path | str) -> GraphSnapshot:
    """Load a full graph that may contain post-T nodes.

    Walker code must not call this. Confirmed labels stay in the held-out file.
    """
    return load_graph(path)


def successor_hits_confirmed(
    full_graph: GraphSnapshot, endpoint: str, confirmed: frozenset[str]
) -> bool:
    if endpoint not in full_graph.nodes:
        return False
    return any(child in confirmed for child in full_graph.outgoing.get(endpoint, ()))


def _card_endpoint(card: Any) -> str | None:
    nodes = getattr(card, "nodes", None)
    if nodes is None and isinstance(card, dict):
        nodes = card.get("nodes") or []
    if not nodes:
        return None
    return str(nodes[-1])


def measure_timeslice_value(
    far_cards: tuple[Any, ...] | list[Any],
    dump_cards: tuple[Any, ...] | list[Any],
    full_graph: GraphSnapshot,
    confirmed: frozenset[str],
    *,
    seed_node_id: str = "",
) -> dict[str, Any]:
    """One (snapshot, seed) rate over an equal number of shots per arm.

    Whether far cards beat the dump arm on a single layout is decided by where
    the post-T edges were placed, not by drift quality: hang the confirmed nodes
    under the dump arm's endpoints instead and the comparison inverts. A value
    anchor therefore needs `certify_timeslice_value` over independent seeds.

    Both arms are truncated to the same number of cards first. On a real corpus
    the dump arm emits one card per reference (dozens to a hundred) while drift
    emits a handful, and a hit *rate* over unequal card counts rewards whichever
    arm said less: one lucky card out of two beats fifty hits out of a hundred.
    Equal shots makes the column read "given the same number of guesses, which arm
    pointed at a confirmed direction more often", which is the question.
    """
    if not confirmed:
        return {
            "measured": False,
            "certifiable": False,
            "value_far": None,
            "value_near": None,
            "beats_baseline": False,
            "reason": "held-out confirmed is empty",
        }
    if not any(node_id in full_graph.nodes for node_id in confirmed):
        return {
            "measured": False,
            "certifiable": False,
            "value_far": None,
            "value_near": None,
            "beats_baseline": False,
            "reason": "confirmed nodes are absent from the evaluator graph",
        }

    shots = min(len(far_cards), len(dump_cards))
    if shots == 0:
        silent = "far" if not far_cards else "near"
        return {
            "measured": False,
            "certifiable": False,
            "value_far": None,
            "value_near": None,
            "beats_baseline": False,
            "n_far": len(far_cards),
            "n_near": len(dump_cards),
            "equal_shots": 0,
            "silent_arm": silent,
            "snapshot_id": full_graph.snapshot_id,
            "seed_node_id": seed_node_id,
            "reason": (
                f"the {silent} arm produced no cards on this seed, so there is no equal-shot comparison to make here"
            ),
        }

    def rate(cards: tuple[Any, ...] | list[Any]) -> float:
        hits = 0
        for card in list(cards)[:shots]:
            endpoint = _card_endpoint(card)
            if endpoint and successor_hits_confirmed(full_graph, endpoint, confirmed):
                hits += 1
        return hits / shots

    value_far = rate(far_cards)
    value_near = rate(dump_cards)
    return {
        "measured": False,
        "certifiable": False,
        "value_far": value_far,
        "value_near": value_near,
        "beats_baseline": False,
        "arm_beats_baseline": value_far > value_near,
        "n_far": len(far_cards),
        "n_near": len(dump_cards),
        "equal_shots": shots,
        "snapshot_id": full_graph.snapshot_id,
        "seed_node_id": seed_node_id,
        "reason": (
            "a single (snapshot, seed) layout cannot certify a value anchor; the "
            "post-T edges are placed by the corpus author. need >= "
            f"{MIN_TIMESLICE_SEEDS} independent seeds through certify_timeslice_value"
        ),
    }


def certify_timeslice_value(
    measurements: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate independent (snapshot, seed) measurements into a value anchor.

    Independence is by distinct (snapshot_id, seed_node_id). A majority of the
    seeds must favour the drift arm, so one favourable layout cannot carry the
    anchor.
    """
    usable = [item for item in measurements if item.get("value_far") is not None]
    seeds = {
        (str(item.get("snapshot_id") or ""), str(item.get("seed_node_id") or ""))
        for item in usable
    }
    seeds.discard(("", ""))
    if len(seeds) < MIN_TIMESLICE_SEEDS:
        return {
            "measured": False,
            "certifiable": False,
            "beats_baseline": False,
            "n_seeds": len(seeds),
            "reason": (
                f"{len(seeds)} independent seed(s) measured; a value anchor needs >= "
                f"{MIN_TIMESLICE_SEEDS}"
            ),
        }
    far = sum(float(item["value_far"]) for item in usable) / len(usable)
    near = sum(float(item["value_near"]) for item in usable) / len(usable)
    wins = sum(1 for item in usable if item.get("arm_beats_baseline"))
    return {
        "measured": True,
        "certifiable": True,
        "value_far": far,
        "value_near": near,
        "beats_baseline": far > near and wins * 2 > len(usable),
        "n_seeds": len(seeds),
        "n_measurements": len(usable),
        "wins": wins,
    }
