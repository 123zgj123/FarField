"""Versioned evolvable harness. Orthogonal to WorldVersion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .actions import ACTION_TYPES
from .kernel import MUTABLE_MODULES, assert_mutable_module, assert_patch_files


HARNESS_STATE_FILE = "HARNESS_STATE.json"
HARNESS_DIR = "harness"


@dataclass
class HarnessArtifact:
    name: str
    module: str
    scope: str
    activation: str
    body: str
    source_trajectories: tuple[str, ...] = ()
    supported_pathology: str = ""
    version: str = "1"
    dependencies: tuple[str, ...] = ()
    validation: dict[str, Any] = field(default_factory=dict)
    activations: int = 0
    benefit: float = 0.0
    success_delta: float = 0.0
    failure_delta: float = 0.0
    context_cost: int = 0
    runtime_cost: float = 0.0
    last_used: int = 0
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "scope": self.scope,
            "activation": self.activation,
            "body": self.body,
            "source_trajectories": list(self.source_trajectories),
            "supported_pathology": self.supported_pathology,
            "version": self.version,
            "dependencies": list(self.dependencies),
            "validation": dict(self.validation),
            "activations": self.activations,
            "benefit": self.benefit,
            "success_delta": self.success_delta,
            "failure_delta": self.failure_delta,
            "context_cost": self.context_cost,
            "runtime_cost": self.runtime_cost,
            "last_used": self.last_used,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HarnessArtifact":
        row = dict(payload)
        return cls(
            name=str(row.get("name") or ""),
            module=str(row.get("module") or ""),
            scope=str(row.get("scope") or "task"),
            activation=str(row.get("activation") or ""),
            body=str(row.get("body") or ""),
            source_trajectories=tuple(row.get("source_trajectories") or ()),
            supported_pathology=str(row.get("supported_pathology") or ""),
            version=str(row.get("version") or "1"),
            dependencies=tuple(row.get("dependencies") or ()),
            validation=dict(row.get("validation") or {}),
            activations=int(row.get("activations") or 0),
            benefit=float(row.get("benefit") or 0),
            success_delta=float(row.get("success_delta") or 0),
            failure_delta=float(row.get("failure_delta") or 0),
            context_cost=int(row.get("context_cost") or 0),
            runtime_cost=float(row.get("runtime_cost") or 0),
            last_used=int(row.get("last_used") or 0),
            status=str(row.get("status") or "active"),
        )


@dataclass
class HarnessVersion:
    version_id: str
    parent_id: str = ""
    artifacts: list[HarnessArtifact] = field(default_factory=list)
    overlays: dict[str, list[str]] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_id": self.parent_id,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "overlays": {key: list(val) for key, val in self.overlays.items()},
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "HarnessVersion":
        row = dict(payload or {})
        arts = [
            HarnessArtifact.from_dict(item)
            for item in row.get("artifacts") or []
            if isinstance(item, Mapping)
        ]
        return cls(
            version_id=str(row.get("version_id") or "H0"),
            parent_id=str(row.get("parent_id") or ""),
            artifacts=arts,
            overlays={
                str(key): list(val)
                for key, val in dict(row.get("overlays") or {}).items()
            },
            notes=str(row.get("notes") or ""),
        )


def next_harness_label(parent: str = "H0") -> str:
    text = str(parent or "H0").strip() or "H0"
    if text.startswith("H") and text[1:].isdigit():
        return f"H{int(text[1:]) + 1}"
    return "H1"


def artifact_digest(artifact: HarnessArtifact) -> str:
    raw = json.dumps(artifact.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def project_harness(
    harness: HarnessVersion,
    *,
    action_type: str = "",
    lineage_id: str = "",
    task: str = "",
    limit: int = 4,
) -> list[HarnessArtifact]:
    """Activate only the overlay slice this action needs.

    Never dump the full harness history into a prompt.
    """
    wanted: list[HarnessArtifact] = []
    action = str(action_type or "").strip().upper()
    for item in harness.artifacts:
        cond = str(item.activation or "").strip().lower()
        scope = str(item.scope or "").strip().lower()
        if scope == "core":
            wanted.append(item)
            continue
        if action and (action.lower() in cond or item.module.endswith(f"{action.lower()}_executor")):
            wanted.append(item)
            continue
        if lineage_id and lineage_id in cond:
            wanted.append(item)
            continue
        if task and task.lower() in cond:
            wanted.append(item)
            continue
        if scope == "action-kind" and action and action.lower() in item.name.lower():
            wanted.append(item)
            continue
    # Prefer artifacts that declare this action; drop unrelated recipes.
    if action:
        focused = [
            item
            for item in wanted
            if action.lower() in (item.activation + " " + item.module + " " + item.name).lower()
            or item.scope == "core"
        ]
        if focused:
            wanted = focused
    seen: set[str] = set()
    unique: list[HarnessArtifact] = []
    for item in wanted:
        if item.name in seen:
            continue
        seen.add(item.name)
        unique.append(item)
    return unique[: max(1, int(limit))]


def projection_lines(
    harness: HarnessVersion,
    *,
    action_type: str,
    lineage_id: str = "",
    task: str = "",
) -> tuple[str, ...]:
    slice_ = project_harness(
        harness, action_type=action_type, lineage_id=lineage_id, task=task
    )
    if not slice_:
        return (
            f"Harness {harness.version_id}: no overlay for {action_type}. "
            "Do not load unrelated recipes.",
        )
    lines = [
        f"Harness {harness.version_id} projection for {action_type} "
        f"({len(slice_)} artifacts; history is not dumped):"
    ]
    for item in slice_:
        item.activations += 1
        lines.append(f"- [{item.module}/{item.scope}] {item.name}: {item.body[:160]}")
    return tuple(lines)


def admit_artifact(
    harness: HarnessVersion,
    artifact: HarnessArtifact,
    *,
    files: Iterable[str] = (),
) -> HarnessVersion:
    assert_mutable_module(artifact.module)
    assert_patch_files(tuple(files))
    child = HarnessVersion(
        version_id=next_harness_label(harness.version_id),
        parent_id=harness.version_id,
        artifacts=list(harness.artifacts) + [artifact],
        overlays=dict(harness.overlays),
        notes=f"admitted {artifact.name} for {artifact.supported_pathology}",
    )
    kind = artifact.activation.upper()
    if kind in ACTION_TYPES:
        child.overlays.setdefault("action_kind", [])
        if artifact.name not in child.overlays["action_kind"]:
            child.overlays["action_kind"].append(artifact.name)
    return child


def gc_harness(
    harness: HarnessVersion,
    *,
    min_activations: int = 3,
    min_benefit: float = 0.0,
    max_idle_cost: int = 200,
    tick: int = 0,
) -> tuple[HarnessVersion, list[HarnessArtifact]]:
    """Prune unused high-cost artifacts without inventing a new research goal."""
    kept: list[HarnessArtifact] = []
    deprecated: list[HarnessArtifact] = []
    for item in harness.artifacts:
        idle = (item.activations >= min_activations and item.benefit <= min_benefit) or (
            item.context_cost >= max_idle_cost and item.success_delta <= 0
        )
        unused = item.activations == 0 and item.context_cost >= max_idle_cost
        if item.status == "active" and (idle or unused):
            item.status = "DEPRECATED"
            deprecated.append(item)
        elif item.status != "REVOKED":
            kept.append(item)
    child = HarnessVersion(
        version_id=harness.version_id,
        parent_id=harness.parent_id,
        artifacts=kept,
        overlays=dict(harness.overlays),
        notes=harness.notes
        + (f"; gc deprecated {len(deprecated)} at t={tick}" if deprecated else ""),
    )
    return child, deprecated


def load_harness(workspace: Path | None) -> HarnessVersion:
    if workspace is None:
        return HarnessVersion(version_id="H0")
    path = Path(workspace) / HARNESS_STATE_FILE
    if not path.is_file():
        return HarnessVersion(version_id="H0")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return HarnessVersion(version_id="H0")
    if not isinstance(payload, dict):
        return HarnessVersion(version_id="H0")
    return HarnessVersion.from_dict(payload)


def save_harness(workspace: Path | None, harness: HarnessVersion) -> Path | None:
    if workspace is None:
        return None
    dest = Path(workspace)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / HARNESS_STATE_FILE
    path.write_text(
        json.dumps(harness.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    overlay = dest / HARNESS_DIR / "overlays"
    overlay.mkdir(parents=True, exist_ok=True)
    for item in harness.artifacts:
        if item.module in MUTABLE_MODULES:
            (overlay / f"{item.module}.json").write_text(
                json.dumps(item.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return path
