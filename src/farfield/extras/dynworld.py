"""A frozen world, awakened: runtime-owned dynamics over attested bytes.

The catalog freeze made worlds *real* — retrieved, hashed, bound before
the claim. It did not make them *analyzable*: a fixture was a byte blob
a model-written script read however it liked, so a token stream from a
novel could dress up as the world of any sequence-shaped claim. Schema
match is not scientific match, and nothing downstream could tell.

This module wraps each schema family with a dynamics bridge, in the
spirit of EnvHarness's "wrapping, not authoring": the frozen bytes and
their digest are untouched; what is added is a deterministic interface
the *trusted runtime* owns —

- **levers**: named interventions with a bounded intensity (edge
  dropout, token dropout, base mutation, row noise, transition
  dropout, ...). A lever is the only way a mechanism can act on this
  world; a claim whose mechanism corresponds to no lever has no handle
  here and must say so.
- **observables**: named measurements the runtime computes from the
  world state. An observable is a verifier the model did not write.
- **forward simulation**: roll the frozen payload under a lever along
  an intensity ramp and record every observable per step. Deterministic
  given (world digest, lever, seed): any auditor can recompute the
  trajectory bit for bit.

The model never writes a transition rule. A dynamics authored after
seeing the claim would just be a larger invented world — the same hole
the network ban closed for data. Bridges live here, versioned with the
runtime, written before any particular hypothesis exists.

Epistemic status: everything this module produces is **scout evidence**.
It may inform a diagnosis, expose an inert lever, sharpen a redesign,
or unbind a surrogate world. It may never climb the ladder — `supports`
still comes only from the registered two-arm probe and its host/heavy
reproductions.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .world import WorldFixture, digest_files, schema_family

# Loading caps keep a scout pass cheap on the biggest frozen slices.
_MAX_EDGES = 20_000
_MAX_TOKENS = 20_000
_MAX_BASES = 40_000
_MAX_ROWS = 4_000
_MAX_TRANSITIONS = 10_000

# Ramp defaults: t=0 is the untouched world, later steps raise the
# lever intensity linearly up to `intensity`.
DEFAULT_INTENSITY = 0.6
DEFAULT_HORIZON = 4

# Below this relative change over the full ramp a lever/observable pair
# is reported inert — pulling it would probably yield an uninformative
# probe.
RESPONSE_FLOOR = 0.03

LEVER_NONE = "none"


class DynamicsUnavailable(Exception):
    """No runtime bridge exists for this fixture (schema or payload)."""


# ---------------------------------------------------------------------------
# payload loading (frozen bytes -> in-memory world state)


def _load_graph(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "graph.json").read_text(encoding="utf-8"))
    edges = [tuple(edge) for edge in payload["edges"][:_MAX_EDGES]]
    nodes = set(payload.get("nodes") or [])
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)
    return {"nodes": sorted(nodes), "edges": edges}


def _load_stream(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "stream.json").read_text(encoding="utf-8"))
    return {"tokens": [str(t) for t in payload["tokens"][:_MAX_TOKENS]]}


def _load_fasta(root: Path) -> dict[str, Any]:
    name = next(
        (n for n in ("sequence.fasta", "genome.fasta") if (root / n).is_file()),
        None,
    )
    if name is None:
        candidates = sorted(root.glob("*.fasta"))
        if not candidates:
            raise DynamicsUnavailable("no fasta file in fixture")
        name = candidates[0].name
    seq = "".join(
        line.strip()
        for line in (root / name).read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(">")
    )
    return {"sequence": seq[:_MAX_BASES]}


def _load_table(root: Path) -> dict[str, Any]:
    name = next(
        (n for n in ("table.json", "iris.json") if (root / n).is_file()), None
    )
    if name is None:
        raise DynamicsUnavailable("no table payload in fixture")
    payload = json.loads((root / name).read_text(encoding="utf-8"))
    rows = payload["rows"][:_MAX_ROWS]
    columns = [
        str(col)
        for col in (payload.get("columns") or [])
        if rows and isinstance(rows[0].get(str(col)), (int, float))
        and not isinstance(rows[0].get(str(col)), bool)
    ]
    if not columns and rows:
        columns = [
            key
            for key, value in rows[0].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
    vectors = [
        tuple(float(row.get(col) or 0.0) for col in columns) for row in rows
    ]
    return {"columns": columns, "vectors": vectors}


def _load_trace(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "world.json").read_text(encoding="utf-8"))
    states = [
        str(s.get("id") if isinstance(s, dict) else s)
        for s in payload["states"]
    ]
    transitions = [
        (str(t["src"]), str(t["dst"]))
        for t in payload["transitions"][:_MAX_TRANSITIONS]
    ]
    initial = str(payload.get("initial") or (states[0] if states else ""))
    return {"states": states, "transitions": transitions, "initial": initial}


# ---------------------------------------------------------------------------
# observables (runtime-owned verifiers)


def _observe_graph(state: dict[str, Any]) -> dict[str, float]:
    nodes = state["nodes"]
    edges = state["edges"]
    adjacency: dict[Any, list[Any]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    seen: set[Any] = set()
    largest = 0
    for start in nodes:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for peer in adjacency.get(node, ()):
                if peer not in seen:
                    seen.add(peer)
                    stack.append(peer)
        largest = max(largest, size)
    total = max(1, len(nodes))
    isolated = sum(1 for node in nodes if not adjacency.get(node))
    return {
        "largest_component_fraction": largest / total,
        "mean_degree": 2.0 * len(edges) / total,
        "isolated_fraction": isolated / total,
    }


def _observe_stream(state: dict[str, Any]) -> dict[str, float]:
    tokens = state["tokens"]
    total = max(1, len(tokens))
    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    counts = Counter(tokens)
    top = counts.most_common(1)[0][1] if counts else 0
    return {
        "distinct_fraction": len(counts) / total,
        "adjacent_repeat_rate": repeats / max(1, total - 1),
        "top_token_share": top / total,
    }


def _observe_fasta(state: dict[str, Any]) -> dict[str, float]:
    seq = state["sequence"]
    total = max(1, len(seq))
    k = 8
    kmers = {seq[i : i + k] for i in range(0, max(0, len(seq) - k + 1))}
    gc = sum(1 for base in seq if base in "GCgc")
    return {
        "kmer8_diversity": len(kmers) / max(1, total - k + 1),
        "gc_fraction": gc / total,
    }


def _observe_table(state: dict[str, Any]) -> dict[str, float]:
    vectors = state["vectors"]
    if not vectors:
        return {"mean_pairwise_distance": 0.0, "mean_column_spread": 0.0}
    dims = len(vectors[0])
    # Deterministic pairwise sample: consecutive pairs wrap around.
    dist = 0.0
    count = 0
    for i in range(len(vectors)):
        a = vectors[i]
        b = vectors[(i + 1) % len(vectors)]
        dist += sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
        count += 1
    spreads = []
    for d in range(dims):
        column = [vec[d] for vec in vectors]
        spreads.append(max(column) - min(column))
    return {
        "mean_pairwise_distance": dist / max(1, count),
        "mean_column_spread": sum(spreads) / max(1, dims),
    }


def _observe_trace(state: dict[str, Any]) -> dict[str, float]:
    states = state["states"]
    transitions = state["transitions"]
    adjacency: dict[str, list[str]] = {}
    for src, dst in transitions:
        adjacency.setdefault(src, []).append(dst)
    seen = {state["initial"]} if state["initial"] else set()
    stack = list(seen)
    while stack:
        node = stack.pop()
        for peer in adjacency.get(node, ()):
            if peer not in seen:
                seen.add(peer)
                stack.append(peer)
    total = max(1, len(states))
    dead = sum(1 for s in states if not adjacency.get(s))
    return {
        "reachable_fraction": len(seen) / total,
        "dead_end_fraction": dead / total,
        "mean_out_degree": len(transitions) / total,
    }


# ---------------------------------------------------------------------------
# levers (the only ways a mechanism can act on the world)


def _keep(items: list, fraction_removed: float, rng: random.Random) -> list:
    if fraction_removed <= 0:
        return list(items)
    kept = [item for item in items if rng.random() >= fraction_removed]
    return kept if kept else list(items[:1])


def _lever_graph(
    state: dict[str, Any], lever: str, intensity: float, rng: random.Random
) -> dict[str, Any]:
    nodes, edges = state["nodes"], list(state["edges"])
    if lever == "dropout":
        return {"nodes": nodes, "edges": _keep(edges, intensity, rng)}
    if lever == "hub_removal":
        degree: Counter = Counter()
        for a, b in edges:
            degree[a] += 1
            degree[b] += 1
        cut = max(0, int(len(nodes) * intensity * 0.2))
        hubs = {node for node, _ in degree.most_common(cut)}
        return {
            "nodes": nodes,
            "edges": [e for e in edges if e[0] not in hubs and e[1] not in hubs],
        }
    if lever == "rewire":
        out = []
        pool = list({n for e in edges for n in e})
        for a, b in edges:
            if rng.random() < intensity and pool:
                out.append((a, rng.choice(pool)))
            else:
                out.append((a, b))
        return {"nodes": nodes, "edges": out}
    raise DynamicsUnavailable(f"unknown graph lever {lever!r}")


def _lever_stream(
    state: dict[str, Any], lever: str, intensity: float, rng: random.Random
) -> dict[str, Any]:
    tokens = list(state["tokens"])
    if lever == "dropout":
        return {"tokens": _keep(tokens, intensity, rng)}
    if lever == "window_shuffle":
        window = max(2, int(4 + intensity * 60))
        out = []
        for i in range(0, len(tokens), window):
            chunk = tokens[i : i + window]
            if rng.random() < intensity:
                rng.shuffle(chunk)
            out.extend(chunk)
        return {"tokens": out}
    raise DynamicsUnavailable(f"unknown stream lever {lever!r}")


def _lever_fasta(
    state: dict[str, Any], lever: str, intensity: float, rng: random.Random
) -> dict[str, Any]:
    seq = state["sequence"]
    if lever == "dropout":
        return {
            "sequence": "".join(
                base for base in seq if rng.random() >= intensity
            )
            or seq[:1]
        }
    if lever == "point_mutation":
        alphabet = "ACGT"
        out = [
            rng.choice(alphabet) if rng.random() < intensity else base
            for base in seq
        ]
        return {"sequence": "".join(out)}
    raise DynamicsUnavailable(f"unknown fasta lever {lever!r}")


def _lever_table(
    state: dict[str, Any], lever: str, intensity: float, rng: random.Random
) -> dict[str, Any]:
    vectors = list(state["vectors"])
    if lever == "dropout":
        return {"columns": state["columns"], "vectors": _keep(vectors, intensity, rng)}
    if lever == "noise":
        if not vectors:
            return dict(state)
        dims = len(vectors[0])
        spans = []
        for d in range(dims):
            column = [vec[d] for vec in vectors]
            spans.append((max(column) - min(column)) or 1.0)
        out = [
            tuple(
                value + rng.uniform(-1, 1) * intensity * spans[d]
                for d, value in enumerate(vec)
            )
            for vec in vectors
        ]
        return {"columns": state["columns"], "vectors": out}
    raise DynamicsUnavailable(f"unknown table lever {lever!r}")


def _lever_trace(
    state: dict[str, Any], lever: str, intensity: float, rng: random.Random
) -> dict[str, Any]:
    transitions = list(state["transitions"])
    if lever == "dropout":
        return {**state, "transitions": _keep(transitions, intensity, rng)}
    if lever == "rewire":
        out = []
        for src, dst in transitions:
            if rng.random() < intensity and state["states"]:
                out.append((src, rng.choice(state["states"])))
            else:
                out.append((src, dst))
        return {**state, "transitions": out}
    raise DynamicsUnavailable(f"unknown trace lever {lever!r}")


# ---------------------------------------------------------------------------
# the bridge registry

_BRIDGES: dict[str, dict[str, Any]] = {
    "undirected_graph": {
        "load": _load_graph,
        "observe": _observe_graph,
        "apply": _lever_graph,
        "levers": ("dropout", "hub_removal", "rewire"),
    },
    "text_stream": {
        "load": _load_stream,
        "observe": _observe_stream,
        "apply": _lever_stream,
        "levers": ("dropout", "window_shuffle"),
    },
    "fasta": {
        "load": _load_fasta,
        "observe": _observe_fasta,
        "apply": _lever_fasta,
        "levers": ("dropout", "point_mutation"),
    },
    "numeric_table": {
        "load": _load_table,
        "observe": _observe_table,
        "apply": _lever_table,
        "levers": ("dropout", "noise"),
    },
    "symbolic_trace": {
        "load": _load_trace,
        "observe": _observe_trace,
        "apply": _lever_trace,
        "levers": ("dropout", "rewire"),
    },
}
_BRIDGES["undirected_named_graph"] = _BRIDGES["undirected_graph"]


def _bridge_of(fixture: WorldFixture) -> dict[str, Any] | None:
    return _BRIDGES.get(schema_family(fixture.schema)) or _BRIDGES.get(fixture.schema)


def has_dynamics(fixture: WorldFixture | None) -> bool:
    return fixture is not None and _bridge_of(fixture) is not None


def levers_of(fixture: WorldFixture) -> tuple[str, ...]:
    bridge = _bridge_of(fixture)
    return tuple(bridge["levers"]) if bridge else ()


def observables_of(fixture: WorldFixture) -> tuple[str, ...]:
    bridge = _bridge_of(fixture)
    if not bridge:
        return ()
    state = bridge["load"](fixture.root)
    return tuple(sorted(bridge["observe"](state)))


def _rng(fixture: WorldFixture, lever: str, seed: int, step: int) -> random.Random:
    key = f"{fixture.digest}:{lever}:{seed}:{step}"
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def forward_simulate(
    fixture: WorldFixture,
    lever: str,
    *,
    intensity: float = DEFAULT_INTENSITY,
    seed: int = 0,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Roll the frozen world under `lever` along an intensity ramp.

    Step 0 is the untouched world. Each later step re-applies the lever
    to the *frozen* payload at a higher intensity — a ramp, not a random
    walk, so a trajectory reads as a dose-response curve. Deterministic
    given (world digest, lever, seed, horizon): recompute and compare.
    """
    bridge = _bridge_of(fixture)
    if bridge is None:
        raise DynamicsUnavailable(f"no bridge for schema {fixture.schema!r}")
    if lever not in bridge["levers"]:
        raise DynamicsUnavailable(f"unknown lever {lever!r} for {fixture.schema}")
    frozen = bridge["load"](fixture.root)
    steps: list[dict[str, Any]] = []
    for t in range(horizon + 1):
        level = intensity * t / max(1, horizon)
        state = (
            frozen
            if t == 0
            else bridge["apply"](frozen, lever, level, _rng(fixture, lever, seed, t))
        )
        steps.append(
            {
                "t": t,
                "intensity": round(level, 6),
                "observables": {
                    key: round(value, 6)
                    for key, value in bridge["observe"](state).items()
                },
            }
        )
    response: dict[str, float] = {}
    first, last = steps[0]["observables"], steps[-1]["observables"]
    for key, base in first.items():
        end = last.get(key, base)
        response[key] = round(
            (end - base) / abs(base) if abs(base) > 1e-12 else end - base, 6
        )
    trajectory = {
        "world_id": fixture.id,
        "world_digest": fixture.digest,
        "schema": fixture.schema,
        "lever": lever,
        "intensity": intensity,
        "seed": seed,
        "horizon": horizon,
        "steps": steps,
        "response": response,
    }
    trajectory["report_digest"] = hashlib.sha256(
        json.dumps(trajectory, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return trajectory


def response_card(
    fixture: WorldFixture, *, seed: int = 0, horizon: int = 3
) -> dict[str, Any]:
    """One forward simulation per lever: the world's measured dynamics.

    The card is what a diagnosis gets to read — which levers exist,
    which observables they move, and by how much. `inert` collects the
    lever/observable pairs whose response stayed under the floor; an
    experiment built on an inert pair is pre-announced uninformative.
    """
    if not has_dynamics(fixture):
        raise DynamicsUnavailable(f"no bridge for schema {fixture.schema!r}")
    levers: dict[str, dict[str, float]] = {}
    for lever in levers_of(fixture):
        trajectory = forward_simulate(
            fixture, lever, seed=seed, horizon=horizon
        )
        levers[lever] = trajectory["response"]
    observables = sorted(next(iter(levers.values()), {}))
    card = {
        "world_id": fixture.id,
        "world_digest": fixture.digest,
        "schema": fixture.schema,
        "seed": seed,
        "horizon": horizon,
        "levers": levers,
        "observables": observables,
        "inert": sorted(
            f"{lever}/{obs}"
            for lever, resp in levers.items()
            for obs, delta in resp.items()
            if abs(delta) < RESPONSE_FLOOR
        ),
    }
    card["report_digest"] = hashlib.sha256(
        json.dumps(card, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return card


def card_lines(card: dict[str, Any]) -> str:
    """Render the response card for a prompt: measured, not asserted."""
    lines = []
    for lever, response in sorted(card.get("levers", {}).items()):
        moved = ", ".join(
            f"{obs} {delta:+.1%}"
            for obs, delta in sorted(response.items())
            if abs(delta) >= RESPONSE_FLOOR
        )
        lines.append(f"- {lever}: {moved or 'inert (no observable moved)'}")
    return "\n".join(lines)


def scout_summary(card: dict[str, Any], lever: str = "") -> str:
    """A compact world-response quote for a redesign prompt.

    Quoting this is safe where treatment/control numbers are not: these
    are properties of the frozen world under runtime levers, computed
    with no reference to any claim's verdict.
    """
    parts = [f"forward simulation of world {card.get('world_id')}:"]
    parts.append(card_lines(card))
    chosen = card.get("levers", {}).get(lever)
    if lever and chosen is not None:
        strongest = max(chosen.items(), key=lambda kv: abs(kv[1]), default=None)
        if strongest and abs(strongest[1]) < RESPONSE_FLOOR:
            parts.append(
                f"the lever you registered ({lever}) is measured inert in this"
                " world — choose a lever the simulation shows responsive"
            )
    return "\n".join(parts)


def arm_separation(treatment: float, control: float) -> float:
    base = abs(control)
    return (treatment - control) / base if base > 0 else treatment - control


def world_consumed(
    sep_real: float, sep_placebo: float, margin: float
) -> bool:
    """True when the registered experiment scientifically used the world.

    Displacement vs the pre-registered margin, not a z-test: tiny
    placebo variance would otherwise call a scenery k-mer window
    "world-consuming".
    """
    try:
        gap = abs(float(sep_real) - float(sep_placebo))
        floor = float(margin)
    except (TypeError, ValueError):
        return False
    if not (gap >= 0) or floor <= 0:
        return False
    return gap >= floor


def materialize_placebo(
    fixture: WorldFixture, dest: Path, *, seed: int = 0
) -> WorldFixture:
    """Write a structure-destroying copy of the freeze. Same filenames.

    Schema-preserving: a graph stays a graph, a stream stays tokens.
    The scientific object is ruined (rewired, shuffled, column-permuted)
    so a script that only uses schema shape produces the same numbers.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))
    family = schema_family(fixture.schema)
    for name in fixture.files:
        src = fixture.root / name
        out = dest / name
        if not src.is_file():
            continue
        if family == "undirected_graph" and name.endswith(".json"):
            payload = json.loads(src.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "edges" in payload:
                nodes = list(payload.get("nodes") or [])
                edges = list(payload.get("edges") or [])
                if nodes and edges:
                    payload["edges"] = [
                        [rng.choice(nodes), rng.choice(nodes)] for _ in edges
                    ]
                out.write_text(
                    json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                continue
        if family == "text_stream" and name.endswith(".json"):
            payload = json.loads(src.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("tokens"), list):
                tokens = list(payload["tokens"])
                rng.shuffle(tokens)
                payload["tokens"] = tokens
                out.write_text(
                    json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                continue
        if family == "text_stream" and name.endswith(".txt"):
            words = src.read_text(encoding="utf-8").split()
            rng.shuffle(words)
            out.write_text(" ".join(words), encoding="utf-8")
            continue
        if family == "fasta" and name.endswith(".fasta"):
            lines = src.read_text(encoding="utf-8").splitlines()
            header = next((line for line in lines if line.startswith(">")), ">placebo")
            seq = list(
                "".join(line.strip() for line in lines if line and not line.startswith(">"))
            )
            rng.shuffle(seq)
            body = "".join(seq)
            wrapped = "\n".join(body[i : i + 70] for i in range(0, len(body), 70))
            out.write_text(header + "\n" + wrapped + "\n", encoding="utf-8")
            continue
        if family == "numeric_table" and name.endswith(".json"):
            payload = json.loads(src.read_text(encoding="utf-8"))
            rows = list(payload.get("rows") or []) if isinstance(payload, dict) else []
            if rows and isinstance(rows[0], dict):
                keys = [
                    key
                    for key, value in rows[0].items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ]
                for key in keys:
                    column = [row.get(key) for row in rows]
                    rng.shuffle(column)
                    for row, value in zip(rows, column):
                        row[key] = value
                payload["rows"] = rows
                out.write_text(
                    json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                continue
        if family == "symbolic_trace" and name == "world.json":
            payload = json.loads(src.read_text(encoding="utf-8"))
            states = [
                str(item.get("id") if isinstance(item, dict) else item)
                for item in (payload.get("states") or [])
            ]
            for edge in payload.get("transitions") or []:
                if isinstance(edge, dict) and states:
                    edge["dst"] = rng.choice(states)
            out.write_text(
                json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            continue
        shutil.copy2(src, out)
    files = list(fixture.files)
    digest = digest_files(dest, files)
    return WorldFixture(
        id=f"{fixture.id}-placebo",
        title=f"placebo of {fixture.id}",
        source="placebo",
        retrieved_at=fixture.retrieved_at,
        digest=digest,
        files=tuple(files),
        root=dest,
        domains=fixture.domains,
        role="placebo",
        schema=fixture.schema,
        load_hint=fixture.load_hint,
    )
