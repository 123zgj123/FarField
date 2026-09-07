"""Immutable trusted epistemic kernel for the open-world controller.

This module does not invent promotion rules. It names the boundary and
calls the existing honesty functions. Harness evolution may not rewrite
these names, these files, or these checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ..evidence import host_confirmation_gaps, world_attested
from ..lifecycle import CrossWorldEvidenceError, assert_evidence_world


TRUSTED_KERNEL = frozenset(
    {
        "world_generated_separation",
        "evidence_id_check",
        "world_id_compatibility",
        "attestation",
        "evidence_promotion",
        "object_domain_identity",
        "prior_novelty_provenance",
        "probe_causal_fairness",
        "evaluator_oracle",
        "sandbox_boundary",
        "harness_credit_gate",
        "harness_rollback",
        "state_reducer",
        "event_log",
        "kernel_mutation_api",
    }
)

TRUSTED_KERNEL_FILES = frozenset(
    {
        "src/farfield/extras/evidence.py",
        "src/farfield/extras/world.py",
        "src/farfield/extras/claimspec.py",
        "src/farfield/extras/prior.py",
        "src/farfield/extras/plugins.py",
        "src/farfield/extras/openworld/kernel.py",
        "src/farfield/extras/openworld/events.py",
        "src/farfield/extras/openworld/reducer.py",
    }
)

MUTABLE_MODULES = frozenset(
    {
        "context",
        "retrieval",
        "memory_projection",
        "question_executor",
        "acquire_executor",
        "theory_discriminator",
        "probe_executor",
        "scheduler_policy",
        "subagent_policy",
        "skill_registry",
    }
)

SOCIAL_PROMOTION_KEYS = (
    "sibling_agreement",
    "citation_count",
    "harness_repetition",
    "archive_discussion",
    "memory_frequency",
    "lineage_references",
)


class KernelViolation(ValueError):
    """A harness patch or state write tried to cross the kernel."""


@dataclass(frozen=True)
class RunContext:
    """Reproducible execution identity. World and harness stay orthogonal."""

    world_id: str
    evidence_id: str
    harness_version: str
    tool_versions: tuple[tuple[str, str], ...] = ()
    evaluator_version: str = "kernel-v1"
    capability_versions: tuple[tuple[str, str], ...] = ()

    @property
    def run_context_id(self) -> str:
        payload = {
            "world_id": self.world_id,
            "evidence_id": self.evidence_id,
            "harness_version": self.harness_version,
            "tools": dict(self.tool_versions),
            "evaluator": self.evaluator_version,
            "capabilities": dict(self.capability_versions),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "evidence_id": self.evidence_id,
            "harness_version": self.harness_version,
            "tool_versions": dict(self.tool_versions),
            "evaluator_version": self.evaluator_version,
            "capability_versions": dict(self.capability_versions),
            "run_context_id": self.run_context_id,
        }


def make_run_context(
    *,
    world_id: str,
    evidence_id: str,
    harness_version: str,
    tool_versions: Mapping[str, str] | None = None,
    evaluator_version: str = "kernel-v1",
    capability_versions: Mapping[str, str] | None = None,
) -> RunContext:
    tools = tuple(sorted((str(k), str(v)) for k, v in dict(tool_versions or {}).items()))
    caps = tuple(
        sorted((str(k), str(v)) for k, v in dict(capability_versions or {}).items())
    )
    return RunContext(
        world_id=str(world_id or ""),
        evidence_id=str(evidence_id or ""),
        harness_version=str(harness_version or ""),
        tool_versions=tools,
        evaluator_version=str(evaluator_version or "kernel-v1"),
        capability_versions=caps,
    )


def assert_mutable_module(module: str) -> str:
    name = str(module or "").strip()
    if name not in MUTABLE_MODULES:
        raise KernelViolation(
            f"{name!r} is not a mutable harness module; "
            f"trusted kernel surfaces cannot be patched here"
        )
    return name


def assert_patch_files(files: list[str] | tuple[str, ...] | None) -> None:
    for raw in files or ():
        path = str(raw or "").replace("\\", "/")
        if path in TRUSTED_KERNEL_FILES or path.endswith("/openworld/kernel.py"):
            raise KernelViolation(
                f"refusing harness patch of trusted kernel file {path}"
            )
        if path.endswith("mission.py"):
            raise KernelViolation(
                "harness evolution may not rewrite mission.py; "
                "that is pipeline code, not a mutable overlay"
            )


def social_consensus_is_not_evidence(row: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Sibling agreement, citations, and memory frequency never promote."""
    payload = dict(row or {})
    hits = [
        key
        for key in SOCIAL_PROMOTION_KEYS
        if payload.get(key) not in (None, "", 0, False, [], {})
    ]
    if hits:
        return tuple(f"social signal {name} is not an evidence promotion condition" for name in hits)
    return ()


def refuse_generated_promotion(row: Mapping[str, Any] | None) -> str:
    payload = dict(row or {})
    epistemic = str(payload.get("epistemic") or payload.get("probe_kind") or "").strip()
    if epistemic.upper() in {"GENERATED", "SYNTHETIC", "E0_SYNTHETIC"} or epistemic == "GENERATED":
        return "GENERATED hypothesis cannot become WORLD by entering state, archive, or memory"
    if str(payload.get("probe_kind") or "") == "GENERATED":
        return "GENERATED probe cannot corroborate"
    return ""


def can_promote_to_world(row: Mapping[str, Any] | None, world: Any = None) -> tuple[bool, tuple[str, ...]]:
    """Closed kernel: only attested WORLD evidence with a matching EvidenceID."""
    payload = dict(row or {})
    gaps: list[str] = []
    generated = refuse_generated_promotion(payload)
    if generated:
        gaps.append(generated)
    gaps.extend(social_consensus_is_not_evidence(payload))
    if world is not None:
        try:
            assert_evidence_world(payload, world)
        except CrossWorldEvidenceError as exc:
            gaps.append(str(exc))
    role = str(payload.get("role") or "")
    if role == "QuestionEvidence":
        if not (
            payload.get("attested")
            and payload.get("evidence_id")
            and payload.get("world_id")
            and str(payload.get("epistemic") or "") == "WORLD"
        ):
            gaps.append(
                "observational WORLD evidence needs attested world_id and EvidenceID"
            )
        return (not gaps, tuple(gaps))
    if payload.get("host_ok") is True:
        gaps.extend(host_confirmation_gaps(payload))
    if str(payload.get("epistemic") or "") != "WORLD" and not world_attested(payload):
        if not any("GENERATED" in item for item in gaps):
            gaps.append("only attested WORLD evidence may promote")
    if str(payload.get("epistemic") or "") == "WORLD" and not world_attested(payload):
        if not payload.get("attested"):
            gaps.append("WORLD label without attestation is not promotion")
    return (not gaps, tuple(gaps))


def attach_run_context(row: Mapping[str, Any], context: RunContext) -> dict[str, Any]:
    """Stamp execution identity without rewriting prior provenance."""
    payload = dict(row)
    existing = payload.get("run_context")
    if isinstance(existing, Mapping) and existing.get("run_context_id"):
        payload["historical_run_context"] = dict(existing)
    payload["run_context"] = context.to_dict()
    payload["harness_version"] = context.harness_version
    payload["world_id"] = payload.get("world_id") or context.world_id
    payload["evidence_id"] = payload.get("evidence_id") or context.evidence_id
    return payload


def preserve_evidence_provenance(
    original: Mapping[str, Any], *, new_harness: str = ""
) -> dict[str, Any]:
    """A later harness version must not mutate historical evidence stamps."""
    payload = dict(original)
    payload.pop("rewrite_harness_version", None)
    if new_harness and str(payload.get("harness_version") or "") != str(new_harness):
        payload["later_harness_seen"] = str(new_harness)
    return payload


@dataclass
class CandidateResult:
    """What a mutable executor may return. It is not a state write."""

    status: str
    events: list[Any] = field(default_factory=list)
    evidence: dict[str, Any] | None = None
    world_id: str = ""
    evidence_id: str = ""
    unlocked: bool = False
    integrity_gap: str = ""
    harness_general: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "world_id": self.world_id,
            "evidence_id": self.evidence_id,
            "unlocked": self.unlocked,
            "integrity_gap": self.integrity_gap,
            "harness_general": self.harness_general,
        }
        if self.evidence is not None:
            payload["evidence"] = dict(self.evidence)
        payload.update(dict(self.extras))
        return payload


def readonly_view(state: Any) -> Mapping[str, Any]:
    if hasattr(state, "to_dict"):
        return MappingProxyType(state.to_dict())
    if isinstance(state, Mapping):
        return MappingProxyType(dict(state))
    return MappingProxyType({})


def as_candidate(raw: Any) -> CandidateResult:
    if isinstance(raw, CandidateResult):
        return raw
    if isinstance(raw, Mapping):
        extras = dict(raw)
        return CandidateResult(
            status=str(raw.get("status") or "recorded"),
            events=list(raw.get("events") or []),
            evidence=dict(raw["evidence"]) if isinstance(raw.get("evidence"), Mapping) else None,
            world_id=str(raw.get("world_id") or ""),
            evidence_id=str(raw.get("evidence_id") or ""),
            unlocked=bool(raw.get("unlocked")),
            integrity_gap=str(raw.get("integrity_gap") or ""),
            harness_general=bool(raw.get("harness_general")),
            extras=extras,
        )
    return CandidateResult(status="recorded")


def sanitize_evidence_candidate(
    evidence: Mapping[str, Any] | None,
    *,
    world: Any = None,
) -> tuple[dict[str, Any] | None, str]:
    """Mutable code may propose evidence. Only the kernel labels WORLD."""
    if evidence is None:
        return None, ""
    payload = dict(evidence)
    generated = refuse_generated_promotion(payload)
    if generated and str(payload.get("epistemic") or "") == "WORLD":
        payload["epistemic"] = "GENERATED"
        payload["attested"] = False
        payload["promotion_refused"] = generated
        return payload, generated
    ok, gaps = can_promote_to_world(payload, world=world)
    payload["promotion_ok"] = ok
    payload["promotion_gaps"] = list(gaps)
    if not ok and str(payload.get("epistemic") or "") == "WORLD":
        payload["epistemic"] = "GENERATED"
        payload["attested"] = False
        return payload, "; ".join(gaps)
    return payload, ""


def executor_may_not_write_world(evidence: Mapping[str, Any] | None) -> str:
    """Callers that set WORLD without attestation are rejected, not coerced silently only."""
    if evidence is None:
        return ""
    if str(evidence.get("epistemic") or "") == "WORLD" and not evidence.get("attested"):
        return "executor cannot mark WORLD without kernel attestation"
    if evidence.get("promote_generated"):
        return "executor cannot promote GENERATED"
    return ""
