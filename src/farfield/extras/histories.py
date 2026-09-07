"""Import a published self-improvement run directory as program_state.

Open-source harness-evolution frameworks already write the object a
`program_state` claim is about — an ordered history of a program
modifying itself under validators — but each in its own layout:

- Darwin Gödel Machine (`jennyzzt/dgm`, `output_dgm/<run>/`):
  `dgm_metadata.jsonl` (one indented JSON object per generation with
  `selfimprove_entries`, `children`, `children_compiled`, `archive`)
  and one `<child>/metadata.json` per self-improvement attempt
  (`run_id`, `parent_commit`, `entry`, `is_compiled`,
  `overall_performance.accuracy_score`) next to `model_patch.diff`;
- OpenEvolve (`algorithmicsuperintelligence/openevolve`,
  `checkpoint_<n>/`): `programs/<id>.json` per evaluated program
  (`id`, `parent_id`, `generation`, `iteration_found`, `metrics`,
  `metadata.island`) and `metadata.json` with the MAP-Elites `archive`,
  `islands`, `best_program_id`.

An importer is a pure function of the files under the run directory.
It reads nothing else, orders updates by the harness's own order
(generation / iteration, then id), and keeps every attempt — failed
compiles and rejected programs included. The source digest is over the
sorted (relative path, sha256) pairs of the files actually read, so a
freeze from a run directory is as replayable as one from a URL.

Field mapping to program_state updates:

    id               the child / program id
    epoch            position in the harness order
    writes           files the patch touched (dgm) or ["program"]
    reads            [parent id]
    validator_writes files the patch touched that belong to the agent's
                     own validation / review path (dgm); [] for a fixed
                     external evaluator (openevolve)
    accepted         entered the archive (dgm: `children_compiled` and
                     archive; openevolve: id in `archive` or an island)
    divergent        the attempt did not compile / had no metrics
    task_return      accuracy_score (dgm) or combined_score (openevolve)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

FORMATS = ("dgm", "openevolve")

_DIFF_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)
_PLUS_FILE = re.compile(r"^\+\+\+ b/(\S+)$", re.MULTILINE)
# Paths in a coding-agent repository that judge the agent's own output.
_SELF_VALIDATION = re.compile(
    r"(review|verif|validat|check|test|lint|audit|judge|evaluat)", re.IGNORECASE
)


class HistoryError(ValueError):
    """The run directory is not a complete history of the named format."""


def import_history(fmt: str, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (program_state payload, source manifest) for a run directory."""
    kind = str(fmt or "").strip().lower()
    importer = _IMPORTERS.get(kind)
    if importer is None:
        raise HistoryError(f"unknown history format {fmt!r}; use {list(FORMATS)}")
    root = Path(root)
    if not root.is_dir():
        raise HistoryError(f"run directory missing: {root}")
    reader = _Reader(root)
    payload = importer(reader)
    return payload, reader.manifest(kind)


class _Reader:
    """Reads files under one root and remembers what it read."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.read: dict[str, str] = {}

    def text(self, relative: str) -> str:
        path = self.root / relative
        data = path.read_bytes()
        self.read[relative] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8", errors="replace")

    def exists(self, relative: str) -> bool:
        return (self.root / relative).is_file()

    def manifest(self, fmt: str) -> dict[str, Any]:
        rows = sorted(self.read.items())
        digest = hashlib.sha256(
            "\n".join(f"{name} {sha}" for name, sha in rows).encode("utf-8")
        ).hexdigest()
        return {
            "format": fmt,
            "root": str(self.root),
            "files": [name for name, _ in rows],
            "source_digest": digest,
        }

    def source_bytes(self) -> bytes:
        rows = sorted(self.read.items())
        return "\n".join(f"{name} {sha}" for name, sha in rows).encode("utf-8")


def _concatenated_json(text: str) -> list[dict[str, Any]]:
    """`dgm_metadata.jsonl` is indented objects appended one after another."""
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            obj, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise HistoryError(f"dgm_metadata.jsonl is not JSON at offset {index}") from exc
        if isinstance(obj, dict):
            rows.append(obj)
        index = end
    return rows


def _patch_files(diff_text: str) -> list[str]:
    names: list[str] = []
    for match in _DIFF_FILE.finditer(diff_text):
        if match.group(2) not in names:
            names.append(match.group(2))
    if not names:
        for match in _PLUS_FILE.finditer(diff_text):
            if match.group(1) not in names:
                names.append(match.group(1))
    return names


def import_dgm(reader: _Reader) -> dict[str, Any]:
    if not reader.exists("dgm_metadata.jsonl"):
        raise HistoryError("dgm run has no dgm_metadata.jsonl")
    generations = _concatenated_json(reader.text("dgm_metadata.jsonl"))
    if not generations:
        raise HistoryError("dgm_metadata.jsonl has no generations")
    cells: list[str] = ["coding_agent.py"]
    validators: list[dict[str, Any]] = [
        {"id": "swe_bench", "reads": ["coding_agent.py"]},
    ]
    validator_ids = {"swe_bench"}
    updates: list[dict[str, Any]] = []
    seen: set[str] = set()
    parent_ids: list[str] = ["initial"]
    for generation in generations:
        gen = int(generation.get("generation") or 0)
        children = [str(c) for c in (generation.get("children") or [])]
        compiled = {str(c) for c in (generation.get("children_compiled") or [])}
        archive = {str(c) for c in (generation.get("archive") or [])}
        entries = generation.get("selfimprove_entries") or []
        entry_parent: dict[str, str] = {}
        for pair in entries:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                entry_parent[str(pair[1])] = str(pair[0])
        for child in children:
            if child in seen:
                continue
            seen.add(child)
            meta: dict[str, Any] = {}
            if reader.exists(f"{child}/metadata.json"):
                try:
                    meta = json.loads(reader.text(f"{child}/metadata.json"))
                except json.JSONDecodeError as exc:
                    raise HistoryError(f"{child}/metadata.json is not JSON") from exc
            parent = str(meta.get("parent_commit") or "").strip()
            if not parent:
                # Fall back to the generation's entry list order.
                parent = parent_ids[-1] if parent_ids else "initial"
            touched: list[str] = []
            if reader.exists(f"{child}/model_patch.diff"):
                touched = _patch_files(reader.text(f"{child}/model_patch.diff"))
            if not touched:
                touched = ["coding_agent.py"]
            for name in touched:
                if name not in cells:
                    cells.append(name)
            validator_writes: list[str] = []
            for name in touched:
                if _SELF_VALIDATION.search(name):
                    vid = f"self:{name}"
                    if vid not in validator_ids:
                        validator_ids.add(vid)
                        validators.append({"id": vid, "reads": [name]})
                    validator_writes.append(vid)
            performance = meta.get("overall_performance") or {}
            score = performance.get("accuracy_score") if isinstance(performance, dict) else None
            is_compiled = meta.get("is_compiled")
            if is_compiled is None:
                is_compiled = child in compiled
            updates.append(
                {
                    "id": child,
                    "epoch": len(updates),
                    "generation": gen,
                    "entry": meta.get("entry") or entry_parent.get(child, ""),
                    "writes": touched,
                    "reads": [parent],
                    "validator_writes": validator_writes,
                    "accepted": child in archive,
                    "divergent": not bool(is_compiled),
                    "task_return": float(score) if isinstance(score, (int, float)) else 0.0,
                }
            )
            parent_ids.append(child)
    if not updates:
        raise HistoryError("dgm run recorded no self-improvement attempts")
    # Parents are program versions, so every referenced id is a cell.
    for update in updates:
        for parent in update["reads"]:
            if parent not in cells:
                cells.append(parent)
        if update["id"] not in cells:
            cells.append(update["id"])
    validators[0]["reads"] = list(cells[:1])
    return {
        "cells": cells,
        "validators": validators,
        "invariant": (
            "each update is one self-improvement attempt on the coding agent; "
            "accepted means the attempt entered the archive"
        ),
        "updates": updates,
        # accuracy_score on the small evaluation set is what decided archive
        # entry; the deeper evaluation (when present) is the independent one.
        "oracle": {"field": "task_return", "independent_of_gate": False, "source": "swe-bench subset accuracy_score"},
        "derivation": {"rule": "dgm.output_dgm→program_state.v1", "generations": len(generations)},
    }


def import_openevolve(reader: _Reader) -> dict[str, Any]:
    programs_dir = reader.root / "programs"
    if not programs_dir.is_dir():
        raise HistoryError("openevolve checkpoint has no programs/ directory")
    meta: dict[str, Any] = {}
    if reader.exists("metadata.json"):
        try:
            meta = json.loads(reader.text("metadata.json"))
        except json.JSONDecodeError as exc:
            raise HistoryError("metadata.json is not JSON") from exc
    # Acceptance is the elite archive. Island membership is the population
    # — with population_size above the iteration count every evaluated
    # program is still "in an island", and a 40-iteration harvest read
    # that way had no rejected update at all. When a checkpoint carries
    # no archive, the occupied MAP-Elites cells stand in.
    archive = {str(item) for item in (meta.get("archive") or [])}
    if not archive:
        for feature_map in meta.get("island_feature_maps") or [meta.get("feature_map") or {}]:
            for item in (feature_map or {}).values():
                archive.add(str(item))
    islands: set[str] = set()
    for island in meta.get("islands") or []:
        for item in island or []:
            islands.add(str(item))
    rows: list[dict[str, Any]] = []
    for path in sorted(programs_dir.glob("*.json")):
        try:
            row = json.loads(reader.text(f"programs/{path.name}"))
        except json.JSONDecodeError as exc:
            raise HistoryError(f"{path.name} is not JSON") from exc
        if isinstance(row, dict) and row.get("id"):
            rows.append(row)
    if not rows:
        raise HistoryError("openevolve checkpoint has no program records")
    rows.sort(
        key=lambda r: (
            int(r.get("iteration_found") or 0),
            int(r.get("generation") or 0),
            str(r.get("id")),
        )
    )
    cells: list[str] = ["program"]
    validators = [{"id": "evaluator", "reads": ["program"]}]
    updates: list[dict[str, Any]] = []
    for row in rows:
        ident = str(row["id"])
        parent = str(row.get("parent_id") or "").strip()
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        score = metrics.get("combined_score")
        if not isinstance(score, (int, float)):
            numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
            score = sum(numeric) / len(numeric) if numeric else None
        if ident not in cells:
            cells.append(ident)
        if parent and parent not in cells:
            cells.append(parent)
        updates.append(
            {
                "id": ident,
                "epoch": len(updates),
                "generation": int(row.get("generation") or 0),
                "iteration_found": int(row.get("iteration_found") or 0),
                "island": (row.get("metadata") or {}).get("island") if isinstance(row.get("metadata"), dict) else None,
                "writes": ["program"],
                "reads": [parent] if parent else ["program"],
                "validator_writes": [],
                "accepted": ident in archive,
                "in_population": ident in islands,
                "divergent": score is None,
                "task_return": float(score) if score is not None else 0.0,
            }
        )
    return {
        "cells": cells,
        "validators": validators,
        "invariant": (
            "each update is one evaluated program; the evaluator is fixed, so "
            "validator_writes is empty throughout; accepted means the program "
            "held an elite-archive slot at checkpoint time (in_population is "
            "island membership, not acceptance)"
        ),
        "updates": updates,
        # combined_score is what decided archive membership: the gate and the
        # outcome are one quantity, so this world has no independent oracle.
        "oracle": {"field": "task_return", "independent_of_gate": False, "source": "evaluator combined_score"},
        "derivation": {
            "rule": "openevolve.checkpoint→program_state.v1",
            "best_program_id": meta.get("best_program_id"),
        },
    }


_IMPORTERS: dict[str, Callable[[_Reader], dict[str, Any]]] = {
    "dgm": import_dgm,
    "openevolve": import_openevolve,
}
