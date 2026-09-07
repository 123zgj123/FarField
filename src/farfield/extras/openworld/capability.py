"""Capability registry: verified helpers that expand affordances."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


CAPABILITY_FILE = "CAPABILITY_REGISTRY.json"

TRUST_PROPOSED = "PROPOSED"
TRUST_SANDBOXED = "SANDBOXED"
TRUST_VALIDATED = "VALIDATED"
TRUST_SHADOW = "SHADOW"
TRUST_TRUSTED = "TRUSTED"
TRUST_DEPRECATED = "DEPRECATED"
TRUST_REVOKED = "REVOKED"

TRUST_LEVELS = frozenset(
    {
        "untrusted",
        "validated",
        "kernel",
        TRUST_PROPOSED,
        TRUST_SANDBOXED,
        TRUST_VALIDATED,
        TRUST_SHADOW,
        TRUST_TRUSTED,
        TRUST_DEPRECATED,
        TRUST_REVOKED,
    }
)

USABLE_TRUST = frozenset(
    {"validated", "kernel", TRUST_VALIDATED, TRUST_SHADOW, TRUST_TRUSTED}
)

# Built-in deterministic helpers. Never exec LLM Python.
EXECUTORS: dict[str, Callable[..., Any]] = {}


def register_executor(name: str, fn: Callable[..., Any]) -> None:
    EXECUTORS[str(name)] = fn


@dataclass
class Capability:
    name: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    executor: str
    validator: str
    sandbox: str = "offline"
    version: str = "1"
    provenance: str = ""
    trust_level: str = TRUST_PROPOSED
    evidence_permission: str = "diagnostic"
    validated_scope: str = ""
    known_failure_modes: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    activation_count: int = 0
    success_delta: int = 0
    failure_delta: int = 0
    context_cost: int = 0
    last_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "executor": self.executor,
            "validator": self.validator,
            "sandbox": self.sandbox,
            "version": self.version,
            "provenance": self.provenance,
            "trust_level": self.trust_level,
            "evidence_permission": self.evidence_permission,
            "validated_scope": self.validated_scope,
            "known_failure_modes": list(self.known_failure_modes),
            "dependencies": list(self.dependencies),
            "activation_count": self.activation_count,
            "success_delta": self.success_delta,
            "failure_delta": self.failure_delta,
            "context_cost": self.context_cost,
            "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Capability":
        row = dict(payload)
        return cls(
            name=str(row.get("name") or ""),
            input_schema=dict(row.get("input_schema") or {}),
            output_schema=dict(row.get("output_schema") or {}),
            executor=str(row.get("executor") or ""),
            validator=str(row.get("validator") or ""),
            sandbox=str(row.get("sandbox") or "offline"),
            version=str(row.get("version") or "1"),
            provenance=str(row.get("provenance") or ""),
            trust_level=str(row.get("trust_level") or TRUST_PROPOSED),
            evidence_permission=str(row.get("evidence_permission") or "diagnostic"),
            validated_scope=str(row.get("validated_scope") or ""),
            known_failure_modes=tuple(row.get("known_failure_modes") or ()),
            dependencies=tuple(row.get("dependencies") or ()),
            activation_count=int(row.get("activation_count") or 0),
            success_delta=int(row.get("success_delta") or 0),
            failure_delta=int(row.get("failure_delta") or 0),
            context_cost=int(row.get("context_cost") or 0),
            last_used=int(row.get("last_used") or 0),
        )


@dataclass
class CapabilityRegistry:
    items: dict[str, Capability] = field(default_factory=dict)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.items))

    def has(self, name: str) -> bool:
        return str(name) in self.items

    def usable(self, name: str) -> bool:
        cap = self.items.get(str(name))
        return cap is not None and cap.trust_level in USABLE_TRUST

    def add(self, cap: Capability) -> None:
        if not cap.name:
            raise ValueError("capability needs a name")
        if cap.trust_level not in TRUST_LEVELS:
            raise ValueError("unknown trust level")
        self.items[cap.name] = cap

    def record_use(self, name: str, *, success: bool, tick: int = 0, cost: int = 0) -> Capability | None:
        cap = self.items.get(str(name))
        if cap is None:
            return None
        cap.activation_count += 1
        cap.context_cost += int(cost)
        cap.last_used = int(tick)
        if success:
            cap.success_delta += 1
        else:
            cap.failure_delta += 1
        if cap.trust_level in {"validated", TRUST_VALIDATED} and cap.activation_count >= 1 and success:
            cap.trust_level = TRUST_SHADOW
        elif cap.trust_level == TRUST_SHADOW and cap.activation_count >= 2 and cap.success_delta >= 2:
            cap.trust_level = TRUST_TRUSTED
        return cap

    def deprecate(self, name: str) -> None:
        cap = self.items.get(str(name))
        if cap is not None:
            cap.trust_level = TRUST_DEPRECATED

    def revoke(self, name: str) -> None:
        cap = self.items.get(str(name))
        if cap is not None:
            cap.trust_level = TRUST_REVOKED

    def to_dict(self) -> dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items.values()]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "CapabilityRegistry":
        row = dict(payload or {})
        items = {}
        for item in row.get("items") or []:
            if isinstance(item, Mapping) and item.get("name"):
                cap = Capability.from_dict(item)
                items[cap.name] = cap
        return cls(items=items)


def compare_validator_snapshots(records: list[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Align versioned validator records. This is a research capability, not a claim."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in records or []:
        row = dict(raw)
        version = str(row.get("validator_version") or row.get("version") or "").strip()
        if not version:
            continue
        grouped.setdefault(version, []).append(row)
    versions = tuple(sorted(grouped))
    contrast = ""
    if len(versions) >= 2:
        contrast = f"{versions[0]} vs {versions[-1]}"
    return {
        "versions": list(versions),
        "counts": {key: len(grouped[key]) for key in versions},
        "aligned": bool(len(versions) >= 2),
        "contrast": contrast,
    }


def validate_validator_snapshots(result: Mapping[str, Any] | None) -> bool:
    row = dict(result or {})
    return bool(row.get("aligned") and row.get("contrast") and row.get("versions"))


register_executor("compare_validator_snapshots", compare_validator_snapshots)


SNAPSHOT_CAPABILITY = Capability(
    name="compare_validator_snapshots",
    input_schema={"records": "list of {validator_version, ...}"},
    output_schema={"versions": "list", "counts": "object", "aligned": "bool", "contrast": "str"},
    executor="compare_validator_snapshots",
    validator="validate_validator_snapshots",
    sandbox="offline",
    version="1",
    provenance="harness_evolution",
    trust_level="validated",
    evidence_permission="evidence-producing",
    validated_scope="validator_version alignment",
)


def run_capability(name: str, **kwargs: Any) -> Any:
    fn = EXECUTORS.get(str(name))
    if fn is None:
        raise KeyError(f"capability executor {name!r} is not registered")
    return fn(**kwargs)


def load_capabilities(workspace: Path | None) -> CapabilityRegistry:
    if workspace is None:
        return CapabilityRegistry()
    path = Path(workspace) / CAPABILITY_FILE
    if not path.is_file():
        return CapabilityRegistry()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CapabilityRegistry()
    if not isinstance(payload, dict):
        return CapabilityRegistry()
    return CapabilityRegistry.from_dict(payload)


def save_capabilities(workspace: Path | None, registry: CapabilityRegistry) -> Path | None:
    if workspace is None:
        return None
    dest = Path(workspace)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / CAPABILITY_FILE
    path.write_text(
        json.dumps(registry.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
