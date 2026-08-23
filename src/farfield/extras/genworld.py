"""Construct the executable world this iteration's experiment needs.

Construction is `resolve(requirement)` for the claim's own schema family.
A graph experiment gets a graph, a trace experiment gets traces, a
formula experiment gets an automaton — never a substitute family.
`io` still has no constructor. Origin is `generated`. Host freeze of a
matching schema remains the only WORLD climb.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import OBJECT_SCHEMA, SCHEMAS
from .world import (
    WorldFixture,
    WorldRequirement,
    digest_files,
    in_mission_constructable,
    refine_requirement,
    schema_family,
)

KIND_GENERATED = "GENERATED"

_TRACE = SCHEMAS["symbolic_trace"]

HONESTY = (
    "constructed this iteration from the claim and registered experiment; "
    "not a freeze; cannot corroborate; farfield freeze of a matching "
    "schema is required for WORLD"
)

INCOMPLETE_HONESTY = (
    "acquisition incomplete: this object has no in-mission constructor; "
    "refusing to substitute a formula automaton; freeze a matching "
    "schema for WORLD; cannot corroborate"
)

_FORBIDDEN_KEYS = ("treatment", "control", "verdict")


def as_requirement(
    requirement: WorldRequirement | dict[str, Any] | None,
) -> WorldRequirement:
    if isinstance(requirement, WorldRequirement):
        return requirement
    payload = requirement or {}
    return WorldRequirement(
        object_type=str(payload.get("object_type") or ""),
        schema=str(payload.get("schema") or ""),
        operations=tuple(payload.get("operations") or ()),
        environment=tuple(payload.get("environment") or ()),
        anchors=tuple(payload.get("anchors") or ()),
    )


def construct_world(
    client: Any,
    card: Any,
    requirement: WorldRequirement | dict[str, Any],
    *,
    dest: Path,
    diagnosis: Any = None,
    previous: dict[str, Any] | None = None,
    errors: list[str] | tuple[str, ...] | None = None,
    lineage: Any = None,
) -> WorldFixture:
    """Write the experimental world for this card and this registration.

    `previous` + `errors` adapts the same dest in place. A new `diagnosis`
    without errors constructs the world that experiment needs. Always
    returns a fixture the probe can bind.
    """
    req = refine_requirement(
        as_requirement(requirement),
        getattr(card, "claim", ""),
        getattr(card, "prediction", ""),
        getattr(diagnosis, "experiment", "") if diagnosis is not None else "",
        getattr(diagnosis, "treatment_arm", "") if diagnosis is not None else "",
        getattr(diagnosis, "control_arm", "") if diagnosis is not None else "",
    ) or as_requirement(requirement)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if previous and previous.get("acquisition") == "incomplete":
        previous = None
    if not in_mission_constructable(req):
        return _seal_incomplete(dest, card, req, lineage)
    schema = _schema_of(req)
    payload = None
    source = "stub"
    error_list = [str(item).strip() for item in (errors or []) if str(item or "").strip()]
    if client is not None and hasattr(client, "complete"):
        try:
            if previous is not None and error_list:
                payload = _ask_adapt(client, card, req, diagnosis, previous, error_list)
                source = "adapt"
            else:
                payload = _ask_construct(client, card, req, diagnosis, lineage)
                source = "model"
        except Exception:
            payload = None
    if payload is None or not _valid_world(payload, schema):
        if previous is not None and _valid_world(previous, schema) and not error_list:
            payload = dict(previous)
            source = "kept"
        else:
            payload = _stub_world(card, req, diagnosis)
            source = "stub"
    if lineage is not None and hasattr(lineage, "to_dict"):
        payload = dict(payload)
        payload["lineage"] = lineage.to_dict()
    return _seal(dest, payload, card, req, source)


def execute_world(payload: dict[str, Any] | None, *, max_steps: int = 64) -> list[str]:
    """Walk the constructed instance. Empty means it can host a probe."""
    if not isinstance(payload, dict):
        return ["payload is not an object"]
    if payload.get("acquisition") == "incomplete":
        return ["acquisition incomplete"]
    errors: list[str] = []
    if any(key in payload for key in _FORBIDDEN_KEYS):
        errors.append("world must not pre-register treatment/control/verdict")
    schema = schema_family(str(payload.get("schema") or "")) or str(
        payload.get("schema") or ""
    )
    if schema == "undirected_graph" or (
        not schema and (payload.get("nodes") or payload.get("edges"))
    ):
        return errors + _execute_graph(payload)
    if schema == "labeled_traces" or (
        not schema and isinstance(payload.get("traces"), list)
    ):
        return errors + _execute_traces(payload)
    if schema == "numeric_table" or (
        not schema and isinstance(payload.get("rows"), list)
    ):
        return errors + _execute_table(payload)
    if schema == "text_stream" or (
        not schema and (payload.get("tokens") or payload.get("text"))
    ):
        return errors + _execute_stream(payload)
    if schema == "fasta" or (
        not schema and isinstance(payload.get("sequence"), str)
    ):
        return errors + _execute_fasta(payload)
    return errors + _execute_trace_automaton(payload, max_steps=max_steps)


def load_world_payload(root: Path) -> dict[str, Any]:
    path = Path(root) / "world.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def world_excerpt(payload: dict[str, Any] | None, *, limit: int = 8) -> str:
    data = payload or {}
    if isinstance(data.get("traces"), list):
        lines = [f"traces: {len(data.get('traces') or [])}"]
        for trace in list(data.get("traces") or [])[:limit]:
            if isinstance(trace, dict):
                steps = trace.get("steps") or []
                lines.append(
                    f"trace {trace.get('id')}: label={trace.get('label')} "
                    f"steps={len(steps) if isinstance(steps, list) else 0}"
                )
        return "\n".join(lines)
    if data.get("nodes") or data.get("edges"):
        nodes = data.get("nodes") or []
        edges = data.get("edges") or []
        return f"nodes: {len(nodes) if isinstance(nodes, list) else 0}\nedges: {len(edges) if isinstance(edges, list) else 0}"
    lines = [f"invariant: {data.get('invariant') or ''}"]
    for state in list(data.get("states") or [])[:limit]:
        if isinstance(state, dict):
            lines.append(f"state {state.get('id')}: {state.get('label') or ''}")
    for edge in list(data.get("transitions") or [])[:limit]:
        if isinstance(edge, dict):
            lines.append(
                f"{edge.get('src')} -[{edge.get('action')}]→ {edge.get('dst')}"
            )
    return "\n".join(lines)


def _schema_of(req: WorldRequirement) -> str:
    return schema_family(req.schema) or OBJECT_SCHEMA.get(req.object_type, "") or req.schema


def _spec_of(req: WorldRequirement):
    name = _schema_of(req)
    return SCHEMAS.get(name) or _TRACE


def _seal(
    dest: Path,
    payload: dict[str, Any],
    card: Any,
    req: WorldRequirement,
    source: str,
) -> WorldFixture:
    spec = _spec_of(req)
    sealed = dict(payload)
    sealed["origin"] = "generated"
    sealed["honesty"] = HONESTY
    sealed["source"] = source
    sealed["object_type"] = req.object_type or spec.object_type
    sealed["schema"] = spec.name
    for key in _FORBIDDEN_KEYS:
        sealed.pop(key, None)
    (dest / "world.json").write_text(
        json.dumps(sealed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    files = ["world.json"]
    files.extend(_write_sidecars(dest, sealed, spec.name))
    digest = digest_files(dest, files)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    object_type = req.object_type or spec.object_type
    manifest = {
        "id": dest.name,
        "title": f"{object_type} world for {getattr(card, 'card_id', dest.name)}",
        "source": "generated",
        "retrieved_at": stamp,
        "digest": digest,
        "files": files,
        "domains": list(spec.default_domains) or [object_type],
        "role": "generated",
        "schema": spec.name,
        "load_hint": spec.load_hint,
        "origin": "generated",
        "honesty": HONESTY,
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return WorldFixture(
        id=str(manifest["id"]),
        title=str(manifest["title"]),
        source="generated",
        retrieved_at=stamp,
        digest=digest,
        files=tuple(files),
        root=dest,
        domains=tuple(manifest["domains"]),
        role="generated",
        schema=spec.name,
        load_hint=str(manifest["load_hint"]),
        source_digest=digest,
        slice_rule={"kind": "generated", "object_type": object_type},
    )


def _write_sidecars(dest: Path, payload: dict[str, Any], schema: str) -> list[str]:
    extra: list[str] = []
    if schema == "undirected_graph":
        nodes = list(payload.get("nodes") or [])
        edges = [list(edge)[:2] for edge in (payload.get("edges") or [])]
        graph = {"n": len(nodes), "m": len(edges), "nodes": nodes, "edges": edges}
        (dest / "graph.json").write_text(
            json.dumps(graph, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        extra.append("graph.json")
    elif schema == "text_stream":
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            text = str(payload.get("text") or "")
            tokens = text.split()
        (dest / "stream.json").write_text(
            json.dumps({"tokens": tokens}, ensure_ascii=False),
            encoding="utf-8",
        )
        extra.append("stream.json")
    elif schema == "numeric_table":
        table = {
            "columns": payload.get("columns") or [],
            "rows": payload.get("rows") or [],
        }
        (dest / "table.json").write_text(
            json.dumps(table, ensure_ascii=False),
            encoding="utf-8",
        )
        extra.append("table.json")
    elif schema == "fasta":
        seq = str(payload.get("sequence") or "ACGT")
        (dest / "sequence.fasta").write_text(f">generated\n{seq}\n", encoding="utf-8")
        extra.append("sequence.fasta")
    elif schema == "labeled_traces":
        (dest / "traces.json").write_text(
            json.dumps({"traces": payload.get("traces") or []}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        extra.append("traces.json")
    return extra


def _seal_incomplete(
    dest: Path,
    card: Any,
    req: WorldRequirement,
    lineage: Any = None,
) -> WorldFixture:
    """Refuse to invent a substitute object. Still bindable as GENERATED."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "object_type": req.object_type,
        "schema": req.schema,
        "acquisition": "incomplete",
        "origin": "generated",
        "honesty": INCOMPLETE_HONESTY,
        "source": "incomplete",
    }
    if lineage is not None and hasattr(lineage, "to_dict"):
        payload["lineage"] = lineage.to_dict()
    (dest / "world.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    files = ["world.json"]
    digest = digest_files(dest, files)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    object_type = req.object_type or "unknown"
    manifest = {
        "id": dest.name,
        "title": f"incomplete {object_type} for {getattr(card, 'card_id', dest.name)}",
        "source": "generated",
        "retrieved_at": stamp,
        "digest": digest,
        "files": files,
        "domains": [object_type],
        "role": "generated",
        "schema": req.schema,
        "load_hint": "",
        "origin": "generated",
        "honesty": INCOMPLETE_HONESTY,
        "acquisition": "incomplete",
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return WorldFixture(
        id=str(manifest["id"]),
        title=str(manifest["title"]),
        source="generated",
        retrieved_at=stamp,
        digest=digest,
        files=tuple(files),
        root=dest,
        domains=(object_type,),
        role="generated",
        schema=req.schema,
        load_hint="",
        source_digest=digest,
        slice_rule={"kind": "incomplete", "object_type": object_type},
    )


def _card_fields(card: Any) -> str:
    return (
        f"claim: {getattr(card, 'claim', '')}\n"
        f"mechanism: {getattr(card, 'mechanism', '')}\n"
        f"prediction: {getattr(card, 'prediction', '')}\n"
    )


def _diagnosis_fields(diagnosis: Any) -> str:
    if diagnosis is None:
        return ""
    return (
        f"experiment: {getattr(diagnosis, 'experiment', '')}\n"
        f"treatment_arm: {getattr(diagnosis, 'treatment_arm', '')}\n"
        f"control_arm: {getattr(diagnosis, 'control_arm', '')}\n"
        f"mechanism_flag: {getattr(diagnosis, 'mechanism_flag', '')}\n"
        f"alternative: {getattr(diagnosis, 'alternative', '')}\n"
    )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    blob = str(text or "").strip()
    start = blob.find("{")
    end = blob.rfind("}")
    if start < 0 or end <= start:
        return None
    payload = json.loads(blob[start : end + 1])
    return payload if isinstance(payload, dict) else None


def _lineage_fields(lineage: Any) -> str:
    if lineage is None:
        return ""
    cites = getattr(lineage, "cite_ids", ()) or ()
    return (
        "Literature lineage (retrieved papers only; construct THIS object,"
        " not a toy that makes the idea win):\n"
        f"named_instance: {getattr(lineage, 'named_instance', '')}\n"
        f"lineage: {getattr(lineage, 'lineage', '')}\n"
        f"why_this_object: {getattr(lineage, 'why_this_object', '')}\n"
        f"cite_ids: {', '.join(str(item) for item in cites)}\n"
    )


def _shape_prompt(schema: str) -> str:
    if schema == "undirected_graph":
        return (
            "Keys: object_type, schema, invariant, nodes (list), "
            "edges (list of [u, v] or {source, target}). "
            "Need at least two nodes and one edge. Connected enough that "
            "some shortest paths have length >= 2."
        )
    if schema == "labeled_traces":
        return (
            "Keys: object_type, schema, invariant, traces (list of "
            "{id, label, steps}). steps is an ordered list of objects "
            "with at least token or node. Need at least two traces and "
            "two labels (authorized vs bypass)."
        )
    if schema == "numeric_table":
        return "Keys: object_type, schema, columns (list), rows (list of lists). Need two rows."
    if schema == "text_stream":
        return "Keys: object_type, schema, tokens (list of strings). Need at least eight tokens."
    if schema == "fasta":
        return "Keys: object_type, schema, sequence (ACGT string). Need at least 16 bases."
    return (
        "Keys: object_type, invariant, states (list of {id, label}), "
        "transitions (list of {src, dst, action}). Every src/dst must be a "
        "state id. From start, a transition must fire."
    )


def _ask_construct(
    client: Any,
    card: Any,
    req: WorldRequirement,
    diagnosis: Any,
    lineage: Any = None,
) -> dict[str, Any] | None:
    schema = _schema_of(req)
    prompt = (
        "Write one JSON object that is the executable world THIS registered "
        "experiment will measure. Same scientific object as the claim. "
        "Not a sidecar. Not a paper. No markdown. "
        f"{_shape_prompt(schema)} "
        "Do not include treatment, control, or verdict — the probe computes "
        "its own measure.\n"
        f"object_type: {req.object_type}\n"
        f"schema: {schema}\n"
        f"{_card_fields(card)}"
        f"{_diagnosis_fields(diagnosis)}"
        f"{_lineage_fields(lineage)}"
    )
    completion = client.complete(
        prompt,
        purpose="world_construction",
        system="You construct executable world models as JSON. Not papers.",
    )
    completion.assert_usable()
    return _parse_json_object(str(completion.text or ""))


def _ask_adapt(
    client: Any,
    card: Any,
    req: WorldRequirement,
    diagnosis: Any,
    previous: dict[str, Any],
    errors: list[str],
) -> dict[str, Any] | None:
    schema = _schema_of(req)
    prior = json.dumps(
        {key: previous[key] for key in previous if key not in _FORBIDDEN_KEYS},
        ensure_ascii=False,
        indent=2,
    )[:4000]
    prompt = (
        "The bound experimental world failed execution. Adapt THIS same world "
        "in place so the registered experiment can run. Do not invent a second "
        "world. Do not change the claim. Do not write treatment/control/"
        f"verdict. {_shape_prompt(schema)}\n"
        f"object_type: {req.object_type}\n"
        f"schema: {schema}\n"
        f"{_card_fields(card)}"
        f"{_diagnosis_fields(diagnosis)}"
        "Execution errors:\n"
        + "\n".join(f"- {item}" for item in errors[:12])
        + "\nPrevious world.json:\n"
        + prior
    )
    completion = client.complete(
        prompt,
        purpose="world_adapt",
        system="You adapt an executable world in place. Not a new fixture.",
    )
    completion.assert_usable()
    return _parse_json_object(str(completion.text or ""))


def _valid_world(payload: dict[str, Any], schema: str = "") -> bool:
    if not isinstance(payload, dict) or payload.get("acquisition") == "incomplete":
        return False
    if any(key in payload for key in _FORBIDDEN_KEYS):
        return False
    guessed = schema or schema_family(str(payload.get("schema") or ""))
    if guessed == "undirected_graph" or payload.get("nodes") or payload.get("edges"):
        return not _execute_graph(payload)
    if guessed == "labeled_traces" or isinstance(payload.get("traces"), list):
        return not _execute_traces(payload)
    if guessed == "numeric_table" or isinstance(payload.get("rows"), list):
        return not _execute_table(payload)
    if guessed == "text_stream" or payload.get("tokens") or payload.get("text"):
        return not _execute_stream(payload)
    if guessed == "fasta" or isinstance(payload.get("sequence"), str):
        return not _execute_fasta(payload)
    return (
        all(key in payload for key in ("object_type", "states", "transitions", "invariant"))
        and isinstance(payload.get("states"), list)
        and len(payload.get("states") or []) >= 2
        and isinstance(payload.get("transitions"), list)
        and len(payload.get("transitions") or []) >= 1
    )


def _stub_world(
    card: Any,
    req: WorldRequirement,
    diagnosis: Any = None,
) -> dict[str, Any]:
    digest = hashlib.sha256(
        f"{getattr(card, 'card_id', '')}|{getattr(card, 'claim', '')}|{getattr(diagnosis, 'experiment', '')}".encode()
    ).hexdigest()
    invariant = str(
        getattr(card, "prediction", "")
        or getattr(diagnosis, "experiment", "")
        or "sampler invariant"
    )[:240]
    schema = _schema_of(req)
    if schema == "undirected_graph":
        nodes = [str(i) for i in range(8)]
        edges = [[str(i), str((i + 1) % 8)] for i in range(8)]
        edges.extend([["0", "4"], ["2", "6"]])
        return {
            "object_type": "graph",
            "schema": "undirected_graph",
            "invariant": invariant,
            "nodes": nodes,
            "edges": edges,
            "seed": digest[:16],
        }
    if schema == "labeled_traces":
        auth = [{"t": i, "token": f"call_{i}", "score": 0.9 - 0.05 * i} for i in range(6)]
        bypass = list(reversed(auth))
        return {
            "object_type": "trace",
            "schema": "labeled_traces",
            "invariant": invariant,
            "traces": [
                {"id": "authorized_0", "label": "authorized", "steps": auth},
                {"id": "authorized_1", "label": "authorized", "steps": auth[:5] + auth[4:5]},
                {"id": "bypass_0", "label": "bypass", "steps": bypass},
                {"id": "bypass_1", "label": "bypass", "steps": auth[::2] + auth[1::2]},
            ],
            "seed": digest[:16],
        }
    if schema == "numeric_table":
        return {
            "object_type": "table",
            "schema": "numeric_table",
            "columns": ["x", "y"],
            "rows": [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
            "seed": digest[:16],
        }
    if schema == "text_stream":
        return {
            "object_type": "stream",
            "schema": "text_stream",
            "tokens": ["the", "same", "object", "or", "it", "cannot", "corroborate", "yet"],
            "seed": digest[:16],
        }
    if schema == "fasta":
        return {
            "object_type": "sequence",
            "schema": "fasta",
            "sequence": "ACGTACGTACGTACGT",
            "seed": digest[:16],
        }
    flag = str(getattr(diagnosis, "mechanism_flag", "") or "mechanism_enabled")
    return {
        "object_type": req.object_type or "formula",
        "schema": "symbolic_trace",
        "invariant": invariant,
        "states": [
            {"id": "start", "label": "initial"},
            {"id": "safe", "label": "invariant holds"},
            {"id": "unsafe", "label": "invariant broken"},
        ],
        "transitions": [
            {"src": "start", "dst": "safe", "action": f"{flag}_off"},
            {"src": "start", "dst": "unsafe", "action": f"{flag}_on"},
            {"src": "safe", "dst": "safe", "action": "stay"},
            {"src": "unsafe", "dst": "unsafe", "action": "stay"},
        ],
        "seed": digest[:16],
    }


def _execute_graph(payload: dict[str, Any]) -> list[str]:
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or len(nodes) < 2:
        return ["need at least two nodes"]
    if not isinstance(edges, list) or not edges:
        return ["need at least one edge"]
    return []


def _execute_traces(payload: dict[str, Any]) -> list[str]:
    traces = payload.get("traces")
    if not isinstance(traces, list) or len(traces) < 2:
        return ["need at least two traces"]
    labels = set()
    for index, trace in enumerate(traces):
        if not isinstance(trace, dict):
            return [f"trace {index} is not an object"]
        steps = trace.get("steps")
        if not isinstance(steps, list) or len(steps) < 2:
            return [f"trace {index} needs at least two ordered steps"]
        labels.add(str(trace.get("label") or ""))
    if len(labels) < 2:
        return ["need at least two labels (authorized vs bypass)"]
    return []


def _execute_table(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return ["need at least two rows"]
    return []


def _execute_stream(payload: dict[str, Any]) -> list[str]:
    tokens = payload.get("tokens")
    if isinstance(tokens, list) and len(tokens) >= 4:
        return []
    text = str(payload.get("text") or "")
    if len(text.split()) >= 4:
        return []
    return ["need a token stream"]


def _execute_fasta(payload: dict[str, Any]) -> list[str]:
    seq = str(payload.get("sequence") or "")
    if len(seq) < 8:
        return ["need a sequence of at least 8 bases"]
    return []


def _execute_trace_automaton(
    payload: dict[str, Any], *, max_steps: int = 64
) -> list[str]:
    errors: list[str] = []
    invariant = str(payload.get("invariant") or "").strip()
    if not invariant:
        errors.append("invariant is empty")
    states = payload.get("states")
    transitions = payload.get("transitions")
    if not isinstance(states, list) or len(states) < 2:
        errors.append("need at least two states")
        return errors
    ids: list[str] = []
    for index, state in enumerate(states):
        if not isinstance(state, dict) or not str(state.get("id") or "").strip():
            errors.append(f"state {index} missing id")
            continue
        ids.append(str(state["id"]))
    if len(set(ids)) != len(ids):
        errors.append("duplicate state ids")
    id_set = set(ids)
    if not isinstance(transitions, list) or not transitions:
        errors.append("need at least one transition")
        return errors
    adj: dict[str, list[tuple[str, str]]] = {sid: [] for sid in ids}
    fired_ok = False
    for index, edge in enumerate(transitions):
        if not isinstance(edge, dict):
            errors.append(f"transition {index} not an object")
            continue
        src = str(edge.get("src") or "")
        dst = str(edge.get("dst") or "")
        action = str(edge.get("action") or "").strip()
        if src not in id_set:
            errors.append(f"transition {index} src {src!r} is not a state")
            continue
        if dst not in id_set:
            errors.append(f"transition {index} dst {dst!r} is not a state")
            continue
        if not action:
            errors.append(f"transition {index} missing action")
        adj[src].append((dst, action))
        fired_ok = True
    start = ids[0]
    for sid in ids:
        if sid in {"start", "init", "initial"}:
            start = sid
            break
    if not adj.get(start):
        errors.append(f"start state {start!r} has no outgoing transition")
    seen = {start}
    queue = [start]
    steps = 0
    while queue and steps < max_steps:
        current = queue.pop(0)
        for dst, _action in adj.get(current, []):
            steps += 1
            if dst not in seen:
                seen.add(dst)
                queue.append(dst)
    if not fired_ok:
        errors.append("no transition fired from start")
    unreachable = [sid for sid in ids if sid not in seen]
    if unreachable:
        errors.append("unreachable states: " + ", ".join(unreachable[:12]))
    return errors
