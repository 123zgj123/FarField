"""Derive one attested schema from another, deterministically.

A catalog freeze is bytes plus a pre-registered rule. Some attested
bytes carry a second object inside them: a Live-SWE-agent trajectory
(`labeled_traces`) records the agent writing tool files into its own
harness at runtime and then calling them — a self-modification history
of executable program state. That is the object a `program_state` claim
is about, and it is already attested; only its encoding is wrong.

A derivation here is a pure function of the parent payload. It reads
nothing but the frozen bytes, invents no field, orders by what the
parent already ordered (trace order, then step index), and records the
rule name so a reader can replay it. It is not GENERATED: no model and
no topic ever touch it. It is not a new observation either: origin
keeps the parent digest, and the manifest says `provenance: derived`.

Registry: `DERIVATIONS[(parent_schema, child_schema)]`.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# A tool file the agent wrote to its own harness during the episode.
_WRITE = r"(?:cat\s*<<[^\n]*>\s*|cat\s*>\s*|tee\s+(?:-a\s+)?|>\s*)(?:\S*/)?{name}(?=\s|$|['\"])"
_RUN = r"(?:python3?|bash|sh|\./)\s*(?:\S*/)?{name}(?=\s|$|['\"])"
# Tools whose *name* says they judge the agent's own work: an update
# that writes them is the agent mutating one of its validators.
_VALIDATOR_NAME = re.compile(
    r"(review|verify|verif|validat|check|test|assert|lint|audit)",
    re.IGNORECASE,
)

TRACES_RULE = "labeled_traces.created_tools→program_state.v1"


def program_state_from_traces(payload: dict[str, Any]) -> dict[str, Any]:
    """Live-SWE-agent runtime tool creation as a program_state history.

    Mapping (every field is read off the parent, nothing is inferred
    from outside it):

    - cells: `workspace`, `submission`, and every tool file name the
      traces created (the harness files the agent may rewrite);
    - validators: `tests:<instance_id>` for each episode with tools —
      the hidden SWE-bench tests that labelled the episode — reading
      that episode's tools and `submission`; plus `self:<tool>` for each
      tool whose name says it reviews/verifies/tests, the agent's own
      validator, reading `submission`;
    - updates: one per tool creation, ordered by trace order then the
      step that wrote it; `writes` the tool, `reads` the tools already
      present in that episode and `workspace`; `validator_writes` names
      `self:<tool>` when the tool is a self-validator; `accepted` is
      whether a later step of the same episode ran the tool (the agent
      kept its own modification); `divergent` is whether that harness
      cell had already been written earlier in the history with a
      different sha256 (the cell's content diverges from its previous
      version); `task_return` is 1.0 when the episode was labelled
      resolved, else 0.0.

    Episodes without created tools contribute nothing: they did not
    modify the program. Nothing is filtered by outcome.
    """
    traces = payload.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("parent payload has no traces")
    cells: list[str] = ["workspace", "submission"]
    seen_cells = set(cells)
    validators: list[dict[str, Any]] = []
    validator_ids: set[str] = set()
    updates: list[dict[str, Any]] = []
    episodes_with_tools = 0
    # Last content written to each harness cell, across the whole history.
    shas: dict[str, str] = {}
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        tools = [
            item
            for item in (trace.get("created_tools") or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
        if not tools:
            continue
        steps = [s for s in (trace.get("steps") or []) if isinstance(s, dict)]
        instance = str(trace.get("instance_id") or trace.get("id") or "").strip()
        label = str(trace.get("label") or "")
        task_return = 1.0 if label == "resolved" else 0.0
        episodes_with_tools += 1
        events: list[tuple[int, int, dict[str, Any]]] = []
        for order, tool in enumerate(tools):
            name = str(tool["name"]).strip()
            write_at = _first_step(steps, _WRITE.format(name=re.escape(name)))
            events.append((write_at if write_at is not None else len(steps), order, tool))
        events.sort(key=lambda item: (item[0], item[1]))
        present: list[str] = []
        episode_tools: list[str] = []
        for write_at, order, tool in events:
            name = str(tool["name"]).strip()
            sha = str(tool.get("sha256") or "")
            if name not in seen_cells:
                seen_cells.add(name)
                cells.append(name)
            self_validator = bool(_VALIDATOR_NAME.search(name))
            validator_writes: list[str] = []
            if self_validator:
                vid = f"self:{name}"
                if vid not in validator_ids:
                    validator_ids.add(vid)
                    validators.append({"id": vid, "reads": ["submission"]})
                validator_writes.append(vid)
            ran_later = _first_step(
                steps,
                _RUN.format(name=re.escape(name)),
                after=write_at,
            )
            divergent = name in shas and shas[name] != sha
            updates.append(
                {
                    "id": f"{instance}#{len(updates)}",
                    "epoch": len(updates),
                    "episode": instance,
                    "step": write_at if write_at < len(steps) else None,
                    "writes": [name],
                    "reads": ["workspace", *present],
                    "validator_writes": validator_writes,
                    "accepted": ran_later is not None,
                    "divergent": divergent,
                    "task_return": task_return,
                    "sha256": sha,
                }
            )
            shas[name] = sha
            if name not in present:
                present.append(name)
            if name not in episode_tools:
                episode_tools.append(name)
        vid = f"tests:{instance}"
        if vid not in validator_ids:
            validator_ids.add(vid)
            validators.append({"id": vid, "reads": [*episode_tools, "submission"]})
    if not updates:
        raise ValueError("no trace created a tool; nothing to derive")
    return {
        "cells": cells,
        "validators": validators,
        "invariant": (
            "each update writes one tool file the agent added to its own "
            "harness; accepted means a later step of the same episode ran it"
        ),
        "updates": updates,
        # task_return is the hidden-test label; the gate (accepted = the
        # agent ran its own tool) never saw it. That independence is what
        # makes false acceptance definable on this world.
        "oracle": {"field": "task_return", "independent_of_gate": True, "source": "swe-bench hidden tests"},
        "derivation": {
            "rule": TRACES_RULE,
            "parent_schema": "labeled_traces",
            "episodes": episodes_with_tools,
            "parent_traces": len(traces),
        },
    }


def _first_step(
    steps: list[dict[str, Any]],
    pattern: str,
    *,
    after: int | None = None,
) -> int | None:
    regex = re.compile(pattern, re.MULTILINE)
    for index, step in enumerate(steps):
        if after is not None and index <= after:
            continue
        if regex.search(str(step.get("action") or "")):
            return index
    return None


Derivation = Callable[[dict[str, Any]], dict[str, Any]]

DERIVATIONS: dict[tuple[str, str], tuple[str, Derivation]] = {
    ("labeled_traces", "program_state"): (TRACES_RULE, program_state_from_traces),
}


def derivation_for(parent_schema: str, child_schema: str) -> tuple[str, Derivation] | None:
    return DERIVATIONS.get((str(parent_schema or ""), str(child_schema or "")))
