"""First-class research-object identity.

Claim identity is not a natural-language string. A far-field noun, a
world handle, a measured field, and a probe verdict are different
objects; prose cannot promote one into another. This module is the
type checker for that rule. Lexical weld checks in `farcompile` remain
presentation lint.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "claimspec-1"
SCIENTIFIC_PROMPT_VERSION = "3"

# Presentation lint only. The type matrix is the real constraint.
_CAUSAL_SURFACE = (
    "cause",
    "causes",
    "causing",
    "drive",
    "drives",
    "produce",
    "produces",
    "reduce",
    "reduces",
    "increase because",
    "acts as",
    "leads to",
)


class ContractViolation(ValueError):
    """Caller dropped object identity. Not a scientific finding."""


class MechanismStatus(str, Enum):
    ATTESTED = "attested"
    DERIVED = "derived"
    HYPOTHETICAL = "hypothetical"
    ABSENT = "absent"


class HandleKind(str, Enum):
    OBSERVATIONAL = "observational"
    INTERVENTIONAL = "interventional"


class ClaimType(str, Enum):
    ASSOCIATION = "association"
    PREDICTIVE = "predictive"
    INTERVENTION = "intervention"
    CAUSAL = "causal"
    MECHANISM = "mechanism"


class CandidateState(str, Enum):
    GENERATED = "generated"
    OBJECT_VALIDATED = "object_validated"
    LITERATURE_VALIDATED = "literature_validated"
    REVIEW_PASSED = "review_passed"
    DIAGNOSED = "diagnosed"
    PROBE_REGISTERED = "probe_registered"
    PROBED = "probed"
    EVIDENCE_VALIDATED = "evidence_validated"
    WRITTEN = "written"
    REJECTED_OBJECT = "rejected_object"
    REJECTED_HANDLE = "rejected_handle"
    REJECTED_LITERATURE = "rejected_literature"
    REVIEW_FAILED = "review_failed"
    REVIEW_UNAVAILABLE = "review_unavailable"
    PROBE_FAILED = "probe_failed"
    EVIDENCE_INVALID = "evidence_invalid"
    NEEDS_WORLD_EXPANSION = "needs_world_expansion"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class CompileDisposition(str, Enum):
    OBJECT_VALIDATED = "object_validated"
    NEEDS_WORLD_EXPANSION = "needs_world_expansion"
    REJECTED_OBJECT = "rejected_object"
    REJECTED_HANDLE = "rejected_handle"
    HANDLE_CLAIM_TYPE_MISMATCH = "handle_claim_type_mismatch"
    MEASUREMENT_UNGROUNDED = "measurement_ungrounded"


class ScopedVerdict(str, Enum):
    SUPPORTS_ASSOCIATION = "supports_association"
    SUPPORTS_PREDICTION = "supports_prediction"
    SUPPORTS_INTERVENTION = "supports_intervention"
    SUPPORTS_MECHANISM = "supports_mechanism"
    WEAKENS_ASSOCIATION = "weakens_association"
    WEAKENS_MECHANISM = "weakens_mechanism"
    INCONCLUSIVE = "inconclusive"


# Scientific pipeline is fail-closed. Review skipped because idea_budget==0
# is the only documented escape, and only for the unit-test one-shot path.
FAIL_OPEN_ALLOWED: frozenset[str] = frozenset()

ALLOWED_TRANSITIONS: dict[CandidateState, frozenset[CandidateState]] = {
    CandidateState.GENERATED: frozenset(
        {
            CandidateState.OBJECT_VALIDATED,
            CandidateState.REJECTED_OBJECT,
            CandidateState.REJECTED_HANDLE,
            CandidateState.NEEDS_WORLD_EXPANSION,
        }
    ),
    CandidateState.OBJECT_VALIDATED: frozenset(
        {
            CandidateState.LITERATURE_VALIDATED,
            CandidateState.REJECTED_LITERATURE,
        }
    ),
    CandidateState.LITERATURE_VALIDATED: frozenset(
        {
            CandidateState.REVIEW_PASSED,
            CandidateState.REVIEW_FAILED,
            CandidateState.REVIEW_UNAVAILABLE,
            CandidateState.NEEDS_HUMAN_REVIEW,
        }
    ),
    CandidateState.REVIEW_PASSED: frozenset({CandidateState.DIAGNOSED}),
    CandidateState.DIAGNOSED: frozenset(
        {CandidateState.PROBE_REGISTERED, CandidateState.PROBE_FAILED}
    ),
    CandidateState.PROBE_REGISTERED: frozenset(
        {CandidateState.PROBED, CandidateState.PROBE_FAILED}
    ),
    CandidateState.PROBED: frozenset(
        {CandidateState.EVIDENCE_VALIDATED, CandidateState.EVIDENCE_INVALID}
    ),
    CandidateState.EVIDENCE_VALIDATED: frozenset({CandidateState.WRITTEN}),
}

_CLAIM_HANDLE_MATRIX: dict[ClaimType, frozenset[HandleKind]] = {
    ClaimType.ASSOCIATION: frozenset(
        {HandleKind.OBSERVATIONAL, HandleKind.INTERVENTIONAL}
    ),
    ClaimType.PREDICTIVE: frozenset(
        {HandleKind.OBSERVATIONAL, HandleKind.INTERVENTIONAL}
    ),
    ClaimType.INTERVENTION: frozenset({HandleKind.INTERVENTIONAL}),
    ClaimType.CAUSAL: frozenset({HandleKind.INTERVENTIONAL}),
    ClaimType.MECHANISM: frozenset({HandleKind.INTERVENTIONAL}),
}


def handle_kind_of(lever: str) -> HandleKind:
    name = str(lever or "").strip()
    if name.startswith("contrast:"):
        return HandleKind.OBSERVATIONAL
    return HandleKind.INTERVENTIONAL


@dataclass(frozen=True)
class WorldObjectRegistry:
    """Exact object ids this world can name. Not a bag of keywords."""

    objects: frozenset[str] = field(default_factory=frozenset)

    def exists(self, ref: str) -> bool:
        name = str(ref or "").strip()
        return bool(name) and name in self.objects

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "WorldObjectRegistry":
        return cls(
            frozenset(
                str(item).strip()
                for item in names
                if str(item).strip() and str(item).strip() != "none"
            )
        )

    @classmethod
    def from_menu(cls, menu: Any) -> "WorldObjectRegistry":
        names: list[str] = []
        for attr in ("levers", "observables", "attested_fields"):
            names.extend(getattr(menu, attr, ()) or ())
        for lever in getattr(menu, "levers", ()) or ():
            text = str(lever or "").strip()
            if text.startswith("contrast:") and ":" in text:
                names.append(text.split(":", 1)[1])
        return cls.from_names(names)


@dataclass(frozen=True)
class MechanismSpec:
    name: str
    status: MechanismStatus = MechanismStatus.ABSENT
    world_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "world_refs": list(self.world_refs),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any] | None) -> "MechanismSpec":
        data = dict(row or {})
        refs = tuple(
            str(item).strip()
            for item in (data.get("world_refs") or ())
            if str(item).strip()
        )
        raw = str(data.get("status") or MechanismStatus.ABSENT.value)
        try:
            status = MechanismStatus(raw)
        except ValueError:
            status = MechanismStatus.ABSENT
        return cls(name=str(data.get("name") or ""), status=status, world_refs=refs)


@dataclass(frozen=True)
class HandleSpec:
    id: str
    kind: HandleKind
    world_refs: tuple[str, ...] = ()
    source_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "world_refs": list(self.world_refs),
            "source_fields": list(self.source_fields),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any] | None) -> "HandleSpec":
        data = dict(row or {})
        raw = str(data.get("kind") or HandleKind.INTERVENTIONAL.value)
        try:
            kind = HandleKind(raw)
        except ValueError:
            kind = handle_kind_of(str(data.get("id") or ""))
        return cls(
            id=str(data.get("id") or ""),
            kind=kind,
            world_refs=tuple(
                str(item).strip()
                for item in (data.get("world_refs") or ())
                if str(item).strip()
            ),
            source_fields=tuple(
                str(item).strip()
                for item in (data.get("source_fields") or ())
                if str(item).strip()
            ),
        )


@dataclass(frozen=True)
class MeasurementSpec:
    dv: str = ""
    source_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"dv": self.dv, "source_fields": list(self.source_fields)}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any] | None) -> "MeasurementSpec":
        data = dict(row or {})
        return cls(
            dv=str(data.get("dv") or ""),
            source_fields=tuple(
                str(item).strip()
                for item in (data.get("source_fields") or ())
                if str(item).strip()
            ),
        )


@dataclass(frozen=True)
class ClaimSpec:
    """Registered claim identity. `statement` is presentation only."""

    claim_id: str
    target_object: str
    mechanism: MechanismSpec
    handle: HandleSpec
    measurement: MeasurementSpec
    claim_type: ClaimType
    evidence_scope: tuple[str, ...] = ()
    statement: str = ""
    requested_claim_type: ClaimType | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "claim_id": self.claim_id,
            "target_object": self.target_object,
            "mechanism": self.mechanism.to_dict(),
            "handle": self.handle.to_dict(),
            "measurement": self.measurement.to_dict(),
            "claim_type": self.claim_type.value,
            "evidence_scope": list(self.evidence_scope),
            "statement": self.statement,
            "schema_version": self.schema_version,
        }
        if self.requested_claim_type is not None:
            payload["requested_claim_type"] = self.requested_claim_type.value
        return payload

    @classmethod
    def from_dict(cls, row: Mapping[str, Any] | None) -> "ClaimSpec":
        if not isinstance(row, Mapping):
            raise ContractViolation("ClaimSpec.from_dict requires a mapping")
        requested = row.get("requested_claim_type")
        return cls(
            claim_id=str(row.get("claim_id") or ""),
            target_object=str(row.get("target_object") or ""),
            mechanism=MechanismSpec.from_dict(row.get("mechanism")),
            handle=HandleSpec.from_dict(row.get("handle")),
            measurement=MeasurementSpec.from_dict(row.get("measurement")),
            claim_type=_claim_type(row.get("claim_type")),
            evidence_scope=tuple(
                str(item)
                for item in (row.get("evidence_scope") or ())
                if str(item).strip()
            ),
            statement=str(row.get("statement") or ""),
            requested_claim_type=_claim_type(requested) if requested else None,
            schema_version=str(row.get("schema_version") or SCHEMA_VERSION),
        )


def _claim_type(raw: Any) -> ClaimType:
    text = str(raw or ClaimType.ASSOCIATION.value).strip().lower()
    aliases = {
        "associational": ClaimType.ASSOCIATION,
        "predictive": ClaimType.PREDICTIVE,
        "intervention": ClaimType.INTERVENTION,
        "interventional": ClaimType.INTERVENTION,
        "causal": ClaimType.CAUSAL,
        "mechanism": ClaimType.MECHANISM,
        "mechanistic": ClaimType.MECHANISM,
    }
    if text in aliases:
        return aliases[text]
    try:
        return ClaimType(text)
    except ValueError as exc:
        raise ContractViolation(f"unknown claim_type {raw!r}") from exc


@dataclass(frozen=True)
class TypecheckResult:
    disposition: CompileDisposition
    spec: ClaimSpec
    reasons: tuple[str, ...] = ()
    missing_observables: tuple[str, ...] = ()
    required_interventions: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.disposition == CompileDisposition.OBJECT_VALIDATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "spec": self.spec.to_dict(),
            "reasons": list(self.reasons),
            "missing_observables": list(self.missing_observables),
            "required_interventions": list(self.required_interventions),
        }


def resolve_mechanism(
    name: str,
    world: WorldObjectRegistry,
    world_refs: Iterable[str] = (),
) -> MechanismSpec:
    """Registry membership only. Prose cannot create world_refs."""
    refs = tuple(
        str(item).strip()
        for item in world_refs
        if str(item).strip() and world.exists(str(item).strip())
    )
    label = str(name or "").strip()
    if not refs and label and world.exists(label):
        refs = (label,)
    if refs:
        return MechanismSpec(name=label, status=MechanismStatus.ATTESTED, world_refs=refs)
    return MechanismSpec(
        name=label, status=MechanismStatus.ABSENT, world_refs=()
    )


def resolve_handle(lever: str, world: WorldObjectRegistry) -> HandleSpec:
    name = str(lever or "").strip()
    kind = handle_kind_of(name)
    refs = (name,) if world.exists(name) else ()
    fields = ()
    if name.startswith("contrast:") and ":" in name:
        stratum = name.split(":", 1)[1]
        fields = (stratum,) if stratum else ()
    return HandleSpec(id=name, kind=kind, world_refs=refs, source_fields=fields)


def typecheck_claim(
    spec: ClaimSpec,
    world: WorldObjectRegistry,
) -> TypecheckResult:
    """Decide object identity. Does not read mapping sentences."""
    reasons: list[str] = []
    if spec.handle.id and not world.exists(spec.handle.id):
        return TypecheckResult(
            CompileDisposition.REJECTED_HANDLE,
            spec,
            reasons=(f"handle {spec.handle.id!r} is not in the world registry",),
        )
    if spec.claim_type in {ClaimType.CAUSAL, ClaimType.MECHANISM}:
        if spec.mechanism.status == MechanismStatus.ABSENT or not spec.mechanism.world_refs:
            missing = spec.mechanism.name or "unnamed mechanism"
            return TypecheckResult(
                CompileDisposition.NEEDS_WORLD_EXPANSION,
                spec,
                reasons=(
                    f"mechanism {missing!r} is not represented in the current world",
                ),
                missing_observables=(missing,),
                required_interventions=(f"intervene on {missing}",),
            )
        for ref in spec.mechanism.world_refs:
            if not world.exists(ref):
                return TypecheckResult(
                    CompileDisposition.NEEDS_WORLD_EXPANSION,
                    spec,
                    reasons=(f"mechanism world_ref {ref!r} is not in the world",),
                    missing_observables=(ref,),
                )
    if spec.handle.kind not in _CLAIM_HANDLE_MATRIX[spec.claim_type]:
        return TypecheckResult(
            CompileDisposition.HANDLE_CLAIM_TYPE_MISMATCH,
            spec,
            reasons=(
                f"{spec.claim_type.value} is not permitted on a "
                f"{spec.handle.kind.value} handle",
            ),
        )
    if spec.measurement.dv and not world.exists(spec.measurement.dv):
        return TypecheckResult(
            CompileDisposition.MEASUREMENT_UNGROUNDED,
            spec,
            reasons=(
                f"measurement {spec.measurement.dv!r} is not an attested field",
            ),
        )
    if spec.handle.kind == HandleKind.OBSERVATIONAL:
        surface = f"{spec.statement} {spec.mechanism.name}".lower()
        hits = [term for term in _CAUSAL_SURFACE if term in surface]
        if hits:
            reasons.append(
                "presentation lint: observational claim uses "
                + ", ".join(hits)
            )
    scope = _default_scope(spec)
    admitted = ClaimSpec(
        claim_id=spec.claim_id,
        target_object=spec.target_object,
        mechanism=spec.mechanism,
        handle=spec.handle,
        measurement=spec.measurement,
        claim_type=spec.claim_type,
        evidence_scope=spec.evidence_scope or scope,
        statement=spec.statement,
        requested_claim_type=spec.requested_claim_type,
    )
    return TypecheckResult(
        CompileDisposition.OBJECT_VALIDATED,
        admitted,
        reasons=tuple(reasons),
    )


def _default_scope(spec: ClaimSpec) -> tuple[str, ...]:
    dv = spec.measurement.dv or "dv"
    handle = spec.handle.id or "handle"
    if spec.claim_type == ClaimType.MECHANISM and spec.mechanism.world_refs:
        return (f"mechanism({spec.mechanism.name},{dv})",)
    if spec.claim_type in {ClaimType.CAUSAL, ClaimType.INTERVENTION}:
        return (f"intervention({handle},{dv})",)
    if spec.claim_type == ClaimType.PREDICTIVE:
        return (f"prediction({handle},{dv})",)
    return (f"association({handle},{dv})",)


def admit_generated_card(
    *,
    claim_id: str,
    statement: str,
    mechanism_name: str,
    handle_id: str,
    measurement_dv: str,
    target_object: str,
    world: WorldObjectRegistry,
    requested_claim_type: ClaimType | None = None,
) -> TypecheckResult:
    """Adapter for today's cards. Prose cannot request a stronger type.

    A far noun is inspiration. The admitted claim is the strongest type
    the handle and registry allow — never mechanism unless the mechanism
    already has world_refs.
    """
    handle = resolve_handle(handle_id, world)
    mechanism = resolve_mechanism(mechanism_name, world)
    if mechanism.status == MechanismStatus.ABSENT and mechanism.name:
        mechanism = MechanismSpec(
            name=mechanism.name,
            status=MechanismStatus.HYPOTHETICAL,
            world_refs=(),
        )
    measurement = MeasurementSpec(
        dv=str(measurement_dv or "").strip(),
        source_fields=(str(measurement_dv or "").strip(),)
        if str(measurement_dv or "").strip()
        else (),
    )
    requested = requested_claim_type
    if requested is None:
        admitted_type = (
            ClaimType.ASSOCIATION
            if handle.kind == HandleKind.OBSERVATIONAL
            else ClaimType.INTERVENTION
        )
        if (
            mechanism.status == MechanismStatus.ATTESTED
            and handle.kind == HandleKind.INTERVENTIONAL
        ):
            admitted_type = ClaimType.INTERVENTION
    else:
        admitted_type = requested
    spec = ClaimSpec(
        claim_id=claim_id,
        target_object=target_object,
        mechanism=mechanism,
        handle=handle,
        measurement=measurement,
        claim_type=admitted_type,
        statement=statement,
        requested_claim_type=requested,
    )
    checked = typecheck_claim(spec, world)
    if (
        checked.disposition == CompileDisposition.NEEDS_WORLD_EXPANSION
        and requested is None
        and handle.world_refs
        and (not measurement.dv or world.exists(measurement.dv))
    ):
        # Far-field idea: keep the card as an in-world handle test.
        downgraded = ClaimSpec(
            claim_id=spec.claim_id,
            target_object=spec.target_object,
            mechanism=MechanismSpec(
                name=mechanism.name,
                status=MechanismStatus.HYPOTHETICAL,
                world_refs=(),
            ),
            handle=handle,
            measurement=measurement,
            claim_type=(
                ClaimType.ASSOCIATION
                if handle.kind == HandleKind.OBSERVATIONAL
                else ClaimType.INTERVENTION
            ),
            statement=statement,
            requested_claim_type=requested,
        )
        retry = typecheck_claim(downgraded, world)
        if retry.ok:
            return TypecheckResult(
                CompileDisposition.OBJECT_VALIDATED,
                ClaimSpec(
                    claim_id=retry.spec.claim_id,
                    target_object=retry.spec.target_object,
                    mechanism=downgraded.mechanism,
                    handle=retry.spec.handle,
                    measurement=retry.spec.measurement,
                    claim_type=retry.spec.claim_type,
                    evidence_scope=retry.spec.evidence_scope,
                    statement=retry.spec.statement,
                    requested_claim_type=requested,
                ),
                reasons=retry.reasons
                + (
                    "far mechanism is hypothetical; admitted as a handle test",
                    "NEEDS_WORLD_EXPANSION remains for the unattested mechanism",
                ),
                missing_observables=(mechanism.name,) if mechanism.name else (),
            )
    return checked


@dataclass(frozen=True)
class EvidenceRecord:
    claim_id: str
    handle_id: str
    handle_kind: HandleKind
    measured_fields: tuple[str, ...]
    metric: str
    observed_effect: dict[str, Any]
    supports_scope: tuple[str, ...]
    prohibited_inferences: tuple[str, ...]
    verdict: ScopedVerdict
    arithmetic_verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "handle_id": self.handle_id,
            "handle_kind": self.handle_kind.value,
            "measured_fields": list(self.measured_fields),
            "metric": self.metric,
            "observed_effect": dict(self.observed_effect),
            "supports_scope": list(self.supports_scope),
            "prohibited_inferences": list(self.prohibited_inferences),
            "verdict": self.verdict.value,
            "arithmetic_verdict": self.arithmetic_verdict,
        }


@dataclass(frozen=True)
class EvidenceProjection:
    record: EvidenceRecord
    fact: str
    caveat: str

    def scientific_text(self) -> str:
        parts = [self.fact]
        if self.caveat:
            parts.append(self.caveat)
        return "\n".join(parts)


def _ceiling_verdict(spec: ClaimSpec) -> ScopedVerdict:
    if spec.claim_type == ClaimType.MECHANISM and spec.mechanism.status == MechanismStatus.ATTESTED:
        return ScopedVerdict.SUPPORTS_MECHANISM
    if spec.claim_type in {ClaimType.CAUSAL, ClaimType.INTERVENTION}:
        if spec.handle.kind == HandleKind.INTERVENTIONAL:
            return ScopedVerdict.SUPPORTS_INTERVENTION
        return ScopedVerdict.SUPPORTS_ASSOCIATION
    if spec.claim_type == ClaimType.PREDICTIVE:
        return ScopedVerdict.SUPPORTS_PREDICTION
    return ScopedVerdict.SUPPORTS_ASSOCIATION


def project_evidence_to_claim(
    spec: ClaimSpec,
    *,
    arithmetic_verdict: str,
    observed_effect: Mapping[str, Any] | None = None,
    metric: str = "",
) -> EvidenceProjection:
    """Pure function. Evidence cannot rename or strengthen the claim."""
    effect = dict(observed_effect or {})
    raw = str(arithmetic_verdict or "").strip().lower()
    ceiling = _ceiling_verdict(spec)
    prohibited: list[str] = []
    if spec.mechanism.status in {
        MechanismStatus.ABSENT,
        MechanismStatus.HYPOTHETICAL,
    }:
        prohibited.extend(("mechanism", spec.mechanism.name or "unnamed_mechanism"))
    if spec.handle.kind == HandleKind.OBSERVATIONAL:
        prohibited.extend(("causal", "intervention", "mechanism"))
    if spec.claim_type != ClaimType.MECHANISM:
        prohibited.append("mechanism")
    if raw in {"supports", "support"}:
        verdict = ceiling
        if verdict == ScopedVerdict.SUPPORTS_MECHANISM and "mechanism" in prohibited:
            verdict = (
                ScopedVerdict.SUPPORTS_INTERVENTION
                if spec.handle.kind == HandleKind.INTERVENTIONAL
                else ScopedVerdict.SUPPORTS_ASSOCIATION
            )
    elif raw in {"weakens", "weaken"}:
        verdict = (
            ScopedVerdict.WEAKENS_MECHANISM
            if spec.claim_type == ClaimType.MECHANISM
            and spec.mechanism.status == MechanismStatus.ATTESTED
            else ScopedVerdict.WEAKENS_ASSOCIATION
        )
    else:
        verdict = ScopedVerdict.INCONCLUSIVE
    if verdict == ScopedVerdict.SUPPORTS_MECHANISM and spec.handle.kind == HandleKind.OBSERVATIONAL:
        verdict = ScopedVerdict.SUPPORTS_ASSOCIATION
    scope = spec.evidence_scope or _default_scope(spec)
    if verdict in {
        ScopedVerdict.SUPPORTS_ASSOCIATION,
        ScopedVerdict.WEAKENS_ASSOCIATION,
        ScopedVerdict.INCONCLUSIVE,
    }:
        scope = tuple(
            item for item in scope if item.startswith("association(")
        ) or (f"association({spec.handle.id or 'handle'},{spec.measurement.dv or 'dv'})",)
    delta = effect.get("delta")
    if delta is None and "treatment" in effect and "control" in effect:
        try:
            delta = float(effect["treatment"]) - float(effect["control"])
        except (TypeError, ValueError):
            delta = None
    dv = spec.measurement.dv or metric or "the measured field"
    handle = spec.handle.id or "the registered handle"
    if verdict == ScopedVerdict.INCONCLUSIVE:
        fact = (
            f"The registered comparison on {handle} did not separate {dv} "
            "beyond the pre-registered margin."
        )
    elif delta is not None:
        fact = (
            f"Runs under {handle} differed from the comparison stratum on "
            f"{dv} (Δ = {delta})."
        )
    else:
        fact = f"The registered comparison on {handle} produced a {verdict.value} reading on {dv}."
    caveat = ""
    if spec.mechanism.status in {
        MechanismStatus.ABSENT,
        MechanismStatus.HYPOTHETICAL,
    } and spec.mechanism.name:
        caveat = (
            f"This observation does not establish the proposed mechanism "
            f"{spec.mechanism.name!r} because that mechanism is not "
            "represented or manipulated in the current world."
        )
    if spec.handle.kind == HandleKind.OBSERVATIONAL and verdict == ScopedVerdict.SUPPORTS_ASSOCIATION:
        extra = (
            "The handle is observational; the result is an association, "
            "not a causal or mechanistic test."
        )
        caveat = f"{caveat} {extra}".strip() if caveat else extra
    record = EvidenceRecord(
        claim_id=spec.claim_id,
        handle_id=spec.handle.id,
        handle_kind=spec.handle.kind,
        measured_fields=spec.measurement.source_fields
        or ((spec.measurement.dv,) if spec.measurement.dv else ()),
        metric=metric or spec.measurement.dv,
        observed_effect=effect,
        supports_scope=scope,
        prohibited_inferences=tuple(dict.fromkeys(item for item in prohibited if item)),
        verdict=verdict,
        arithmetic_verdict=raw,
    )
    return EvidenceProjection(record=record, fact=fact, caveat=caveat)


def transition(current: CandidateState, dest: CandidateState) -> CandidateState:
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    if dest not in allowed:
        raise ContractViolation(
            f"illegal candidate transition {current.value} → {dest.value}"
        )
    return dest


def can_emit_evidence(state: CandidateState) -> bool:
    return state == CandidateState.REVIEW_PASSED


def cards_eligible_for_evidence(
    batch: list[Any],
    review_events: Iterable[Mapping[str, Any]],
    *,
    review_required: bool,
) -> list[Any]:
    """Fail-closed: a refused or missing mandatory review cannot proceed."""
    events = list(review_events)
    refused = {
        str(event.get("card_id") or "")
        for event in events
        if event.get("stage") == "idea_review_refused"
    }
    passed = {
        str(event.get("card_id") or "")
        for event in events
        if event.get("stage") in {"idea_review", "review_admitted"}
    }
    if not review_required:
        return [
            item
            for item in batch
            if _card_id(item) not in refused
        ]
    eligible = []
    for item in batch:
        card_id = _card_id(item)
        if card_id in refused:
            continue
        if card_id not in passed:
            continue
        eligible.append(item)
    return eligible


def _card_id(item: Any) -> str:
    if isinstance(item, tuple) and item:
        return str(getattr(item[0], "card_id", "") or "")
    return str(getattr(item, "card_id", "") or item or "")


@dataclass(frozen=True)
class LiteratureSet:
    claim_object_strong: tuple[Any, ...] = ()
    claim_object_weak: tuple[Any, ...] = ()
    inspiration: tuple[Any, ...] = ()
    background: tuple[Any, ...] = ()
    rejected: tuple[Any, ...] = ()

    def related_work(self) -> tuple[Any, ...]:
        return self.claim_object_strong

    def related_work_ids(self) -> frozenset[str]:
        return frozenset(_work_id(item) for item in self.claim_object_strong if _work_id(item))


def _work_id(work: Any) -> str:
    if hasattr(work, "cite_id"):
        try:
            return str(work.cite_id() or "")
        except Exception:
            pass
    if isinstance(work, Mapping):
        return str(
            work.get("arxiv_id")
            or work.get("cite_id")
            or work.get("work_id")
            or work.get("id")
            or ""
        )
    return ""


def assert_related_work_subset(
    paper_ids: Iterable[str],
    literature: LiteratureSet,
) -> None:
    extra = {str(item) for item in paper_ids if str(item).strip()} - literature.related_work_ids()
    if extra:
        raise ContractViolation(
            "related work may only cite claim_object_strong papers; "
            f"smuggled {sorted(extra)}"
        )


def compile_payload(result: TypecheckResult) -> dict[str, Any]:
    """Persist ClaimSpec plus compile metadata. Not a TypecheckResult wrapper."""
    payload = result.spec.to_dict()
    payload["disposition"] = result.disposition.value
    payload["compile_reasons"] = list(result.reasons)
    payload["missing_observables"] = list(result.missing_observables)
    payload["required_interventions"] = list(result.required_interventions)
    return payload


def claim_spec_from_payload(row: Mapping[str, Any] | None) -> ClaimSpec:
    """Accept ClaimSpec dicts and the older TypecheckResult wrapper."""
    if not isinstance(row, Mapping):
        raise ContractViolation("claim spec payload must be a mapping")
    inner = row.get("spec")
    if isinstance(inner, Mapping) and "mechanism" in inner:
        return ClaimSpec.from_dict(inner)
    return ClaimSpec.from_dict(row)


def compile_disposition_of(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    return str(row.get("disposition") or "")


def in_world_testable(row: Mapping[str, Any] | None) -> bool:
    """True when a handle test may run. Empty payload keeps legacy cards live."""
    disposition = compile_disposition_of(row)
    return disposition in {"", CompileDisposition.OBJECT_VALIDATED.value}


def should_queue_world_expansion(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    if compile_disposition_of(row) == CompileDisposition.NEEDS_WORLD_EXPANSION.value:
        return True
    if row.get("missing_observables"):
        return True
    try:
        spec = claim_spec_from_payload(row)
    except ContractViolation:
        return False
    return spec.mechanism.status in {
        MechanismStatus.ABSENT,
        MechanismStatus.HYPOTHETICAL,
    } and bool(spec.mechanism.name)


def world_expansion_record(
    card: Any, row: Mapping[str, Any] | None
) -> dict[str, Any]:
    spec = None
    if isinstance(row, Mapping):
        try:
            spec = claim_spec_from_payload(row)
        except ContractViolation:
            spec = None
    missing = [
        str(item)
        for item in ((row or {}).get("missing_observables") or ())
        if str(item).strip()
    ]
    if (
        spec
        and spec.mechanism.name
        and spec.mechanism.name not in missing
        and spec.mechanism.status
        in {MechanismStatus.ABSENT, MechanismStatus.HYPOTHETICAL}
    ):
        missing.append(spec.mechanism.name)
    interventions = [
        str(item)
        for item in ((row or {}).get("required_interventions") or ())
        if str(item).strip()
    ]
    return {
        "card_id": str(getattr(card, "card_id", "") or ""),
        "status": CandidateState.NEEDS_WORLD_EXPANSION.value,
        "mechanism": (
            spec.mechanism.name
            if spec
            else str(getattr(card, "mechanism", "") or "")
        ),
        "handle_id": (
            spec.handle.id if spec else str(getattr(card, "world_lever", "") or "")
        ),
        "claim_type": spec.claim_type.value if spec else "",
        "missing_observables": missing,
        "required_interventions": (
            interventions
            if interventions
            else (
                [f"instrument or intervene on {missing[0]}"] if missing else []
            )
        ),
        "in_world_testable": in_world_testable(row),
        "statement": (
            spec.statement if spec else str(getattr(card, "claim", "") or "")
        ),
    }


def scientific_for(
    *,
    card: Any = None,
    world: Any = None,
    workspace: Any = None,
    claim: str = "",
) -> dict[str, Any]:
    workspace_id = Path(str(workspace)).name if workspace is not None else ""
    world_digest = str(getattr(world, "digest", "") or "")
    text = str(claim or getattr(card, "claim", "") or "")
    claim_digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
    return scientific_cache_fields(
        workspace_id=workspace_id,
        world_digest=world_digest,
        claim_digest=claim_digest,
    )


def scientific_cache_fields(
    *,
    prompt_version: str = SCIENTIFIC_PROMPT_VERSION,
    workspace_id: str = "",
    world_digest: str = "",
    claim_digest: str = "",
    artifact_digests: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "cache_namespace": "scientific",
        "prompt_version": str(prompt_version or SCIENTIFIC_PROMPT_VERSION),
        "workspace_id": str(workspace_id or ""),
        "world_digest": str(world_digest or ""),
        "claim_digest": str(claim_digest or ""),
        "artifact_digests": list(artifact_digests),
    }


@dataclass(frozen=True)
class MissionManifest:
    mission_id: str
    created_at: str
    topic: str
    world_digest: str = ""
    corpus_digest: str = ""
    schema_version: str = SCHEMA_VERSION
    code_revision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "created_at": self.created_at,
            "topic": self.topic,
            "world_digest": self.world_digest,
            "corpus_digest": self.corpus_digest,
            "schema_version": self.schema_version,
            "code_revision": self.code_revision,
        }
