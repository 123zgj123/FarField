"""Construct the executable world this iteration's experiment needs.

Construction is `resolve(requirement)` for the *task topic's* schema
family, not a later card's painted pair. A graph experiment gets a
graph, a trace experiment gets traces, a formula experiment gets an
automaton — never a substitute family. `io` still has no constructor.
Origin is `generated`. Host freeze of a matching schema remains the
only WORLD climb. A GENERATED world is scoped to one mission: its id
carries a topic digest so auto cannot pick it for the next task.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from types import SimpleNamespace

from .domain import topic_object_phrases
from .dynworld import graph_nodes_edges
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

HONESTY = (
    "constructed this iteration from the task topic and registered "
    "experiment; not a freeze; cannot corroborate; farfield freeze of a "
    "matching schema is required for WORLD"
)

INCOMPLETE_HONESTY = (
    "acquisition incomplete: this object has no in-mission constructor; "
    "refusing to substitute a formula automaton; freeze a matching "
    "schema for WORLD; cannot corroborate"
)

_FORBIDDEN_KEYS = ("treatment", "control", "verdict")


def topic_lock_card(topic: str) -> SimpleNamespace:
    """A stand-in card so construction can freeze the task object first.

    Later landings reuse this world. Their far-field pair must not rename
    the object (balanced tree must not become the world for post-training).
    """
    text = str(topic or "").strip()
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    return SimpleNamespace(
        card_id=f"task_{digest}",
        claim=text,
        mechanism="",
        prediction="",
        pair=(),
    )


def task_world_id(schema: str, topic: str) -> str:
    digest = hashlib.sha256(str(topic or "").strip().encode()).hexdigest()[:12]
    family = schema_family(schema) or schema or "world"
    return f"generated-{family}-{digest}"


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
    topic: str = "",
    idea_world: Any = None,
) -> WorldFixture:
    """Write the experimental world for this task and this registration.

    `topic` is the mission object. The card's far pair does not vote.
    `previous` + `errors` adapts the same dest in place. `idea_world`
    is analysis only: it names quantities the stub/model must store.
    Always returns a fixture the probe can bind. Cannot corroborate.
    """
    task_topic = str(topic or "").strip() or str(
        getattr(card, "claim", "") or ""
    ).strip()
    req = refine_requirement(
        as_requirement(requirement),
        task_topic,
        getattr(card, "prediction", "") if not str(topic or "").strip() else "",
        getattr(diagnosis, "experiment", "") if diagnosis is not None else "",
        getattr(diagnosis, "treatment_arm", "") if diagnosis is not None else "",
        getattr(diagnosis, "control_arm", "") if diagnosis is not None else "",
    ) or as_requirement(requirement)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if previous and previous.get("acquisition") == "incomplete":
        previous = None
    if not in_mission_constructable(req):
        return _seal_incomplete(dest, card, req, lineage, topic=task_topic)
    schema = _schema_of(req)
    payload = None
    source = "stub"
    error_list = [str(item).strip() for item in (errors or []) if str(item or "").strip()]
    if client is not None and hasattr(client, "complete"):
        try:
            if previous is not None and error_list:
                payload = _ask_adapt(
                    client,
                    card,
                    req,
                    diagnosis,
                    previous,
                    error_list,
                    topic=task_topic,
                    idea_world=idea_world,
                )
                source = "adapt"
            else:
                payload = _ask_construct(
                    client,
                    card,
                    req,
                    diagnosis,
                    lineage,
                    topic=task_topic,
                    idea_world=idea_world,
                )
                source = "model"
        except Exception:
            payload = None
    if payload is None or not _valid_world(payload, schema):
        if previous is not None and _valid_world(previous, schema) and not error_list:
            payload = dict(previous)
            source = "kept"
        else:
            payload = _stub_world(
                card, req, diagnosis, topic=task_topic, idea_world=idea_world
            )
            source = "stub"
    if lineage is not None and hasattr(lineage, "to_dict"):
        payload = dict(payload)
        payload["lineage"] = lineage.to_dict()
    return _seal(dest, payload, card, req, source, topic=task_topic)


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
    if schema == "program_state" or (
        not schema and isinstance(payload.get("updates"), list)
    ):
        return errors + _execute_program(payload)
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
    if isinstance(data.get("updates"), list):
        updates = data.get("updates") or []
        accepted = sum(1 for row in updates if isinstance(row, dict) and row.get("accepted"))
        lines = [
            f"cells: {len(data.get('cells') or [])}",
            f"validators: {len(data.get('validators') or [])}",
            f"updates: {len(updates)} (accepted {accepted})",
        ]
        for row in updates[:limit]:
            if isinstance(row, dict):
                lines.append(
                    f"update {row.get('id')}: writes={row.get('writes')} "
                    f"validator_writes={row.get('validator_writes')} "
                    f"accepted={row.get('accepted')} divergent={row.get('divergent')}"
                )
        return "\n".join(lines)
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
    """The registered spec for this requirement — never a substitute.

    Falling back to the automaton spec is how an unregistered family
    used to be built as a protocol state machine. `construct_world`
    already routes those to `_seal_incomplete`; reaching here without a
    registered family is a programming error, not a construction.
    """
    name = _schema_of(req)
    spec = SCHEMAS.get(name)
    if spec is None:
        raise ValueError(
            f"no registered world schema for {req.object_type!r}/{req.schema!r}; "
            "refusing to substitute a formula automaton"
        )
    return spec


def _seal(
    dest: Path,
    payload: dict[str, Any],
    card: Any,
    req: WorldRequirement,
    source: str,
    *,
    topic: str = "",
) -> WorldFixture:
    spec = _spec_of(req)
    sealed = dict(payload)
    sealed["origin"] = "generated"
    sealed["honesty"] = HONESTY
    sealed["source"] = source
    sealed["object_type"] = req.object_type or spec.object_type
    sealed["schema"] = spec.name
    task_topic = str(topic or "").strip()
    if task_topic:
        sealed["task_topic"] = task_topic
        sealed["task_digest"] = hashlib.sha256(task_topic.encode()).hexdigest()[:16]
        sealed["far_concept_must_not_rename_object"] = True
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
    world_id = (
        task_world_id(spec.name, task_topic) if task_topic else dest.name
    )
    phrases = topic_object_phrases(task_topic)[:4] if task_topic else ()
    title_object = phrases[0] if phrases else object_type
    manifest = {
        "id": world_id,
        "title": f"{object_type} world for {title_object}",
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
        "task_topic": task_topic,
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
        nodes, pairs = graph_nodes_edges(payload)
        edges = [list(pair) for pair in pairs]
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
    *,
    topic: str = "",
) -> WorldFixture:
    """Refuse to invent a substitute object. Still bindable as GENERATED."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    task_topic = str(topic or "").strip()
    payload: dict[str, Any] = {
        "object_type": req.object_type,
        "schema": req.schema,
        "acquisition": "incomplete",
        "origin": "generated",
        "honesty": INCOMPLETE_HONESTY,
        "source": "incomplete",
    }
    if task_topic:
        payload["task_topic"] = task_topic
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
    world_id = (
        task_world_id(req.schema or object_type, task_topic)
        if task_topic
        else dest.name
    )
    manifest = {
        "id": world_id,
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
        "task_topic": task_topic,
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


def _card_fields(card: Any, topic: str = "") -> str:
    phrases = topic_object_phrases(topic) if topic else ()
    return (
        f"task_topic: {topic}\n"
        f"task_object_phrases: {', '.join(phrases)}\n"
        "Construct THIS topic's scientific object. The far-field pair "
        "member is a mechanism, not a license to build a different world.\n"
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


def _idea_world_fields(idea_world: Any) -> str:
    notes = ""
    if idea_world is not None and hasattr(idea_world, "construct_notes"):
        notes = str(idea_world.construct_notes() or "").strip()
    elif isinstance(idea_world, dict):
        objects = [str(item) for item in (idea_world.get("objects") or []) if str(item)]
        missing = [str(item) for item in (idea_world.get("missing") or []) if str(item)]
        if objects:
            notes = "Must store countable fields for: " + ", ".join(objects[:6]) + "."
        if missing:
            notes += (" " if notes else "") + "Do not drop: " + ", ".join(missing[:6]) + "."
    if not notes:
        return ""
    return (
        "Idea-world analysis (diagnostic; construct these quantities, "
        "do not invent a win):\n"
        f"{notes}\n"
    )


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
    if schema == "program_state":
        return (
            "Keys: object_type, schema, invariant, cells (list of state "
            "cell ids), validators (list of {id, reads} where reads names "
            "cells), updates (ordered list of {id, epoch, writes, reads, "
            "validator_writes, accepted, divergent, task_return}). writes/"
            "reads name cells; validator_writes names validators this "
            "update mutates. Need at least three cells, one validator, six "
            "updates, at least one update on a write–read cycle (it writes "
            "a validator that reads one of its own written cells), and a "
            "mix of accepted and rejected updates. divergent records "
            "whether replay against the pre-update state diverges — it is "
            "ground truth about the instance, not a verdict."
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
    *,
    topic: str = "",
    idea_world: Any = None,
) -> dict[str, Any] | None:
    schema = _schema_of(req)
    prompt = (
        "Write one JSON object that is the executable world THIS task "
        "will measure. Same scientific object as the task topic, not the "
        "far-field pair member. Not a sidecar. Not a paper. No markdown. "
        f"{_shape_prompt(schema)} "
        "Do not include treatment, control, or verdict — the probe computes "
        "its own measure.\n"
        f"object_type: {req.object_type}\n"
        f"schema: {schema}\n"
        f"{_card_fields(card, topic)}"
        f"{_diagnosis_fields(diagnosis)}"
        f"{_lineage_fields(lineage)}"
        f"{_idea_world_fields(idea_world)}"
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
    *,
    topic: str = "",
    idea_world: Any = None,
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
        "world. Do not change the task object. Do not write treatment/control/"
        f"verdict. {_shape_prompt(schema)}\n"
        f"object_type: {req.object_type}\n"
        f"schema: {schema}\n"
        f"{_card_fields(card, topic)}"
        f"{_diagnosis_fields(diagnosis)}"
        f"{_idea_world_fields(idea_world)}"
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


def _guess_schema(payload: dict[str, Any]) -> str:
    """Family from the payload's own keys when nothing was required."""
    if payload.get("nodes") or payload.get("edges"):
        return "undirected_graph"
    if isinstance(payload.get("traces"), list):
        return "labeled_traces"
    if isinstance(payload.get("updates"), list):
        return "program_state"
    if isinstance(payload.get("rows"), list):
        return "numeric_table"
    if payload.get("tokens") or payload.get("text"):
        return "text_stream"
    if isinstance(payload.get("sequence"), str):
        return "fasta"
    if isinstance(payload.get("states"), list):
        return "symbolic_trace"
    return ""


def _valid_world(payload: dict[str, Any], schema: str = "") -> bool:
    """A payload is valid only as the family that was required of it.

    Dispatch is schema-first. The old key-first order accepted a graph
    for a program_state requirement because `nodes` happened to be
    present — the model answered a different question and the world
    passed. When a schema is required, the payload's declared schema
    (if any) and its executable shape must both be that family.
    """
    if not isinstance(payload, dict) or payload.get("acquisition") == "incomplete":
        return False
    if any(key in payload for key in _FORBIDDEN_KEYS):
        return False
    declared = schema_family(str(payload.get("schema") or ""))
    wanted = schema_family(schema) if schema else (declared or _guess_schema(payload))
    if schema and declared and declared != wanted:
        return False
    if wanted == "undirected_graph":
        return not _execute_graph(payload)
    if wanted == "labeled_traces":
        return not _execute_traces(payload)
    if wanted == "program_state":
        return not _execute_program(payload)
    if wanted == "numeric_table":
        return not _execute_table(payload)
    if wanted == "text_stream":
        return not _execute_stream(payload)
    if wanted == "fasta":
        return not _execute_fasta(payload)
    if wanted == "symbolic_trace":
        return (
            all(key in payload for key in ("object_type", "states", "transitions", "invariant"))
            and isinstance(payload.get("states"), list)
            and len(payload.get("states") or []) >= 2
            and isinstance(payload.get("transitions"), list)
            and len(payload.get("transitions") or []) >= 1
        )
    return False


def _stub_world(
    card: Any,
    req: WorldRequirement,
    diagnosis: Any = None,
    *,
    topic: str = "",
    idea_world: Any = None,
) -> dict[str, Any]:
    digest = hashlib.sha256(
        f"{topic}|{getattr(card, 'card_id', '')}|{getattr(card, 'claim', '')}|{getattr(diagnosis, 'experiment', '')}".encode()
    ).hexdigest()
    phrases = topic_object_phrases(topic) if topic else ()
    invariant = str(
        phrases[0]
        if phrases
        else (
            getattr(card, "prediction", "")
            or getattr(diagnosis, "experiment", "")
            or "sampler invariant"
        )
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
        extra = _idea_world_objects(idea_world)
        auth = [
            {
                "t": i,
                "token": f"call_{i}",
                "score": 0.9 - 0.05 * i,
                **({name: 0 for name in extra} if extra else {}),
            }
            for i in range(6)
        ]
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
    if schema == "program_state":
        cells = [f"cell_{i}" for i in range(4)]
        validators = [
            {"id": "v_guard", "reads": ["cell_1", "cell_2"]},
            {"id": "v_frozen", "reads": ["cell_0"]},
        ]
        updates: list[dict[str, Any]] = []
        for i in range(10):
            on_cycle = i % 3 == 1
            divergent = i % 3 != 0
            update: dict[str, Any] = {
                "id": f"u{i}",
                "epoch": i,
                "writes": ["cell_1" if on_cycle else f"cell_{i % 4}"],
                "reads": [f"cell_{(i + 1) % 4}"],
                # On-cycle updates mutate the very validator that reads
                # their written cell: the self-green-light the claim is
                # about exists in the instance, so a gating lever has
                # something real to intercept. Off-cycle divergent
                # updates are caught by the untouched validator.
                "validator_writes": ["v_guard"] if on_cycle else [],
                "accepted": bool(on_cycle or not divergent),
                "divergent": divergent,
                "task_return": round(0.9 - 0.07 * (i % 5), 4),
            }
            updates.append(update)
        return {
            "object_type": "executable",
            "schema": "program_state",
            "invariant": invariant,
            "cells": cells,
            "validators": validators,
            "updates": updates,
            "seed": digest[:16],
        }
    if schema == "numeric_table":
        return {
            "object_type": "table",
            "schema": "numeric_table",
            "columns": ["x", "y"],
            "rows": [
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 0.5, "y": 0.5},
            ],
            "seed": digest[:16],
        }
    if schema == "text_stream":
        words = [part for phrase in phrases for part in phrase.split() if part]
        tokens = (words + ["the", "same", "object", "or", "it", "cannot", "corroborate", "yet"])[:16]
        if len(tokens) < 8:
            tokens = ["the", "same", "object", "or", "it", "cannot", "corroborate", "yet"]
        return {
            "object_type": "stream",
            "schema": "text_stream",
            "tokens": tokens,
            "seed": digest[:16],
        }
    if schema == "fasta":
        return {
            "object_type": "sequence",
            "schema": "fasta",
            "sequence": "ACGTACGTACGTACGT",
            "seed": digest[:16],
        }
    if schema != "symbolic_trace":
        # Every registered family has its own stub above. Anything else
        # must have been routed to `_seal_incomplete` before this point.
        raise ValueError(
            f"no stub for world schema {schema!r}; refusing to substitute "
            "a formula automaton"
        )
    flag = str(getattr(diagnosis, "mechanism_flag", "") or "mechanism_enabled")
    quantities = _idea_world_objects(idea_world)[:4]
    payload = {
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
    if quantities:
        payload["quantities"] = quantities
    return payload


def _idea_world_objects(idea_world: Any) -> list[str]:
    if idea_world is None:
        return []
    raw = getattr(idea_world, "objects", None)
    if raw is None and isinstance(idea_world, dict):
        raw = idea_world.get("objects")
    out: list[str] = []
    seen: set[str] = set()
    for item in list(raw or []):
        text = " ".join(str(item or "").split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text[:40])
    return out


def _execute_graph(payload: dict[str, Any]) -> list[str]:
    nodes, edges = graph_nodes_edges(payload)
    if len(nodes) < 2:
        return ["need at least two nodes"]
    if not edges:
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


def execute_program_state(payload: dict[str, Any]) -> list[str]:
    """Public name for the program_state executor (used by `freeze`)."""
    return _execute_program(payload)


def _execute_program(payload: dict[str, Any]) -> list[str]:
    """Walk the update history: every reference must resolve.

    This is execution, not judging: it checks the instance is a runnable
    object (cells exist, validators read real cells, updates reference
    real cells/validators, acceptance outcomes vary). It never reads
    treatment or control.
    """
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) < 2:
        return ["need at least two state cells"]
    cell_ids = {str(item) for item in cells}
    validators = payload.get("validators")
    if not isinstance(validators, list) or not validators:
        return ["need at least one validator"]
    validator_ids: set[str] = set()
    errors: list[str] = []
    for index, validator in enumerate(validators):
        if not isinstance(validator, dict) or not str(validator.get("id") or "").strip():
            errors.append(f"validator {index} missing id")
            continue
        validator_ids.add(str(validator["id"]))
        reads = validator.get("reads")
        if not isinstance(reads, list) or not reads:
            errors.append(f"validator {validator['id']} reads no cells")
            continue
        for cell in reads:
            if str(cell) not in cell_ids:
                errors.append(
                    f"validator {validator['id']} reads unknown cell {cell!r}"
                )
    updates = payload.get("updates")
    if not isinstance(updates, list) or len(updates) < 4:
        errors.append("need at least four ordered updates")
        return errors
    accepted_states: set[bool] = set()
    for index, update in enumerate(updates):
        if not isinstance(update, dict):
            errors.append(f"update {index} is not an object")
            continue
        for field in ("writes", "reads"):
            for cell in update.get(field) or []:
                if str(cell) not in cell_ids:
                    errors.append(
                        f"update {update.get('id') or index} {field} "
                        f"unknown cell {cell!r}"
                    )
        for vid in update.get("validator_writes") or []:
            if str(vid) not in validator_ids:
                errors.append(
                    f"update {update.get('id') or index} mutates unknown "
                    f"validator {vid!r}"
                )
        if not isinstance(update.get("accepted"), bool):
            errors.append(f"update {update.get('id') or index} missing accepted bool")
        else:
            accepted_states.add(bool(update["accepted"]))
        if not isinstance(update.get("divergent"), bool):
            errors.append(f"update {update.get('id') or index} missing divergent bool")
    if len(accepted_states) < 2:
        errors.append(
            "acceptance never varies: need both accepted and rejected updates"
        )
    return errors


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
