"""Shared scientific kernel for typed research ideas.

`idea_kind` is a first-class control type, not a generator tag. This
module is the policy: opportunity, identity, kind transition, world
versions, acquire lifecycle, notebook, attractor keys, and typed
evidence roles. Pipelines call these functions; they do not grow a
private `if idea_kind == acquire: skip_x` tree.

Honesty invariants stay elsewhere (EvidenceID, WORLD attestation,
object lock, two-arm fairness, prior-kill). This file changes what a
legal idea *is*, not whether evidence can be forged.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ideakind import ACQUIRE, PROBE, QUESTION, THEORY, parse_idea_kind


ORIGIN_LITERATURE = "literature_far"
ORIGIN_LOCAL = "local_problem_structure"
LOCAL_ORIGIN_LABEL = "local problem structure"
LOCAL_NODE_PREFIX = "local:"

MISSING_WORLD = "missing_world"
MISSING_OBSERVABLE = "missing_observable"
CAUSAL_IDENTIFICATION_READY = "causal_identification_ready"
HARVEST_SUCCESS = "harvest_success"
PREDICTION_OBSERVABLE = "prediction_observable"
PREDICTION_MANIPULABLE = "prediction_manipulable"

TRANSITION_REASONS = frozenset(
    {
        MISSING_WORLD,
        MISSING_OBSERVABLE,
        CAUSAL_IDENTIFICATION_READY,
        HARVEST_SUCCESS,
        PREDICTION_OBSERVABLE,
        PREDICTION_MANIPULABLE,
    }
)

ALLOWED_KIND_TRANSITIONS: dict[tuple[str, str], frozenset[str]] = {
    (QUESTION, ACQUIRE): frozenset({MISSING_WORLD, MISSING_OBSERVABLE}),
    (QUESTION, PROBE): frozenset({CAUSAL_IDENTIFICATION_READY}),
    (PROBE, ACQUIRE): frozenset({MISSING_WORLD, MISSING_OBSERVABLE}),
    (ACQUIRE, QUESTION): frozenset({HARVEST_SUCCESS}),
    (ACQUIRE, PROBE): frozenset({HARVEST_SUCCESS, CAUSAL_IDENTIFICATION_READY}),
    (THEORY, QUESTION): frozenset({PREDICTION_OBSERVABLE}),
    (THEORY, PROBE): frozenset({PREDICTION_MANIPULABLE}),
}

WRONG_KIND = "wrong_kind"
MISSING_VALIDATOR = "missing_validator"
CAUSAL_OVERCLAIM = "causal_overclaim"
INSUFFICIENT_CONTRAST = "insufficient_contrast"
RECIPE_NOT_ATTESTABLE = "recipe_not_attestable"
THEORY_NOT_DISCRIMINATIVE = "theory_not_discriminative"
PROBE_UNFAIR = "probe_unfair"
MEASURE_MISMATCH = "measure_mismatch"

FAILURE_TYPES = frozenset(
    {
        WRONG_KIND,
        MISSING_WORLD,
        MISSING_OBSERVABLE,
        MISSING_VALIDATOR,
        CAUSAL_OVERCLAIM,
        INSUFFICIENT_CONTRAST,
        RECIPE_NOT_ATTESTABLE,
        THEORY_NOT_DISCRIMINATIVE,
        PROBE_UNFAIR,
        MEASURE_MISMATCH,
    }
)

ACQUIRE_PROPOSED = "PROPOSED"
ACQUIRE_RECIPE_VALIDATED = "RECIPE_VALIDATED"
ACQUIRE_HARVESTING = "HARVESTING"
ACQUIRE_ATTESTED = "ATTESTED"
ACQUIRE_BOUND = "BOUND"
ACQUIRE_PARTIAL = "PARTIAL"
ACQUIRE_FAILED = "FAILED"

ACQUIRE_STATES = frozenset(
    {
        ACQUIRE_PROPOSED,
        ACQUIRE_RECIPE_VALIDATED,
        ACQUIRE_HARVESTING,
        ACQUIRE_ATTESTED,
        ACQUIRE_BOUND,
        ACQUIRE_PARTIAL,
        ACQUIRE_FAILED,
    }
)

QUESTION_EVIDENCE = "QuestionEvidence"
ACQUISITION_EVIDENCE = "AcquisitionEvidence"
THEORY_EVIDENCE = "TheoryEvidence"
PROBE_EVIDENCE = "ProbeEvidence"

EVIDENCE_ROLES = {
    QUESTION: QUESTION_EVIDENCE,
    ACQUIRE: ACQUISITION_EVIDENCE,
    THEORY: THEORY_EVIDENCE,
    PROBE: PROBE_EVIDENCE,
}

PAPER_SECTIONS = {
    QUESTION: "research question / motivating analysis",
    ACQUIRE: "methods / infrastructure / data-world construction",
    THEORY: "model / explanation / hypotheses",
    PROBE: "experiment",
}

KIND_CORE_FIELDS = {
    QUESTION: (
        "research_question",
        "observables",
        "contrast",
        "why_answer_matters",
        "possible_outcomes",
        "decision_change",
        "required_world_state",
    ),
    ACQUIRE: (
        "capability_gap",
        "missing_artifact",
        "recipe",
        "validators",
        "readiness_criteria",
        "expected_affordances",
    ),
    THEORY: (
        "phenomenon",
        "assumptions",
        "mechanism",
        "derived_predictions",
        "boundary_conditions",
        "disconfirmation",
    ),
    PROBE: (
        "hypothesis",
        "lever",
        "measure",
        "arm_a",
        "arm_b",
        "frozen_world",
        "expected_direction",
    ),
}

_CAUSAL_SURFACE = (
    "cause",
    "causes",
    "causing",
    "causal",
    "intervention",
    "intervene",
    "treatment arm",
    "two-arm",
    "two arm",
    "effect of",
    "drives",
    "produces",
)

RESEARCH_STATE_FILE = "RESEARCH_STATE.json"
WORLD_LINEAGE_FILE = "world_lineage.json"
ACQUIRE_DIR = "acquire"


class KindTransitionError(ValueError):
    """A kind change was not on the admitted table."""


class CrossWorldEvidenceError(ValueError):
    """Evidence from one world version was offered for another."""


def idea_id_of(row: Any) -> str:
    if row is None:
        return ""
    if isinstance(row, Mapping):
        return str(row.get("idea_id") or row.get("card_id") or "").strip()
    return str(
        getattr(row, "idea_id", "") or getattr(row, "card_id", "") or ""
    ).strip()


def lineage_id_for(
    pair: tuple[str, ...] | list[str] | None,
    *,
    origin: str = ORIGIN_LITERATURE,
    explicit: str = "",
) -> str:
    if str(explicit or "").strip():
        return str(explicit).strip()
    items = [str(item).strip() for item in (pair or [])[:2]]
    while len(items) < 2:
        items.append("")
    raw = f"{origin}::{items[0].lower()}::{items[1].lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def origin_of(row: Any) -> str:
    if row is None:
        return ORIGIN_LITERATURE
    raw = ""
    if isinstance(row, Mapping):
        raw = str(row.get("origin") or "")
    else:
        raw = str(getattr(row, "origin", "") or "")
    text = raw.strip().lower()
    if text in {ORIGIN_LOCAL, "local", "local_problem", "problem_structure"}:
        return ORIGIN_LOCAL
    return ORIGIN_LITERATURE


def is_local_origin_label(label: str) -> bool:
    return str(label or "").strip().lower() == LOCAL_ORIGIN_LABEL


def local_node_id(label: str = LOCAL_ORIGIN_LABEL) -> str:
    slug = "-".join(str(label or LOCAL_ORIGIN_LABEL).lower().split()) or "problem"
    return f"{LOCAL_NODE_PREFIX}{slug}"


def is_nongraph_endpoint(node: str) -> bool:
    text = str(node or "")
    return (
        not text
        or text.startswith("literature:")
        or text.startswith(LOCAL_NODE_PREFIX)
    )


def parse_origin(raw: Any, *, far_labels: Iterable[str] = ()) -> str:
    text = str(raw or "").strip().lower()
    if text in {ORIGIN_LOCAL, "local", "local_problem", "problem_structure"}:
        return ORIGIN_LOCAL
    labels = [str(item).strip() for item in far_labels if str(item).strip()]
    if labels and all(is_local_origin_label(item) for item in labels):
        return ORIGIN_LOCAL
    if not labels:
        return ORIGIN_LOCAL
    return ORIGIN_LITERATURE


def parse_kind_fields(kind: str, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Pull typed core fields. Missing keys stay empty; never invented."""
    kind = parse_idea_kind(kind)
    src = dict(payload or {})
    nested = src.get("kind_fields")
    if isinstance(nested, Mapping):
        src = {**src, **dict(nested)}
    out: dict[str, Any] = {}
    for name in KIND_CORE_FIELDS.get(kind, ()):
        value = src.get(name)
        if isinstance(value, (list, tuple)):
            out[name] = [str(item).strip() for item in value if str(item).strip()]
        else:
            out[name] = str(value or "").strip()
    if kind == QUESTION and not out.get("research_question"):
        out["research_question"] = str(src.get("claim") or "").strip()
    if kind == QUESTION and not out.get("why_answer_matters"):
        out["why_answer_matters"] = str(
            src.get("why_it_matters") or src.get("mechanism") or ""
        ).strip()
    if kind == ACQUIRE and not out.get("capability_gap"):
        out["capability_gap"] = str(src.get("claim") or "").strip()
    if kind == ACQUIRE and not out.get("recipe"):
        out["recipe"] = str(src.get("prediction") or src.get("mechanism") or "").strip()
    if kind == THEORY and not out.get("phenomenon"):
        out["phenomenon"] = str(src.get("claim") or "").strip()
    if kind == THEORY and not out.get("derived_predictions"):
        out["derived_predictions"] = str(src.get("prediction") or "").strip()
    if kind == PROBE and not out.get("hypothesis"):
        out["hypothesis"] = str(src.get("claim") or "").strip()
    if kind == PROBE and not out.get("lever"):
        out["lever"] = str(src.get("world_lever") or "").strip()
    if kind == PROBE and not out.get("measure"):
        out["measure"] = str(src.get("world_observable") or "").strip()
    return out


def acquire_recipe_complete(fields: Mapping[str, Any]) -> bool:
    needed = ("recipe", "validators", "readiness_criteria")
    return all(str(fields.get(name) or "").strip() for name in needed)


def question_has_contrast(fields: Mapping[str, Any], *, lever: str = "") -> bool:
    if str(lever or "").startswith("contrast:"):
        return True
    if str(fields.get("contrast") or "").strip():
        return True
    observables = fields.get("observables")
    if isinstance(observables, (list, tuple)) and observables:
        return True
    return bool(str(observables or "").strip())


def causal_overclaim_reason(
    kind: str,
    *,
    claim: str = "",
    lever: str = "",
    prediction: str = "",
) -> str:
    """Observational questions may not be rewritten as causal probes."""
    kind = parse_idea_kind(kind)
    if kind != QUESTION:
        return ""
    lever = str(lever or "").strip()
    if lever and lever != "none" and not lever.startswith("contrast:"):
        return (
            "an observational question cannot compile onto an interventional "
            f"lever ({lever}); that is a causal overclaim, not a tighter question"
        )
    blob = f"{claim} {prediction}".lower()
    if any(token in blob for token in _CAUSAL_SURFACE) and (
        "margin" in blob or "mde" in blob or "effect size" in blob
    ):
        return (
            "an observational question cannot be causalized by adding a "
            "margin, MDE, or intervention vocabulary"
        )
    return ""


def compile_required(kind: str) -> bool:
    """Probe and question must weld a handle when a menu exists."""
    return parse_idea_kind(kind) in {PROBE, QUESTION}


@dataclass(frozen=True)
class WorldOpportunity:
    """What the bound world can actually support. Not a spray kill switch.

    `probe_room` is the old `room_to_move`: a lever that moves an
    observable. That is probe eligibility, not research closure.
    """

    manipulable_handles: tuple[str, ...] = ()
    observable_contrasts: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    unresolved_mechanisms: tuple[str, ...] = ()
    world_id: str = ""
    probe_room: bool = False

    def eligible_kinds(self, *, local_problem: bool = False) -> frozenset[str]:
        kinds: set[str] = set()
        if self.probe_room and self.manipulable_handles:
            kinds.add(PROBE)
        if self.observable_contrasts or local_problem:
            kinds.add(QUESTION)
        if self.missing_capabilities or local_problem or not self.world_id:
            kinds.add(ACQUIRE)
        if self.unresolved_mechanisms or self.observable_contrasts or local_problem:
            kinds.add(THEORY)
        if not kinds:
            kinds.update({QUESTION, ACQUIRE})
        return frozenset(kinds)

    def acquire_priority(self) -> int:
        """Missing world/capability raises acquire, never lowers it."""
        return len(self.missing_capabilities) + (0 if self.world_id else 2)

    def research_open(self) -> bool:
        """A closed probe room does not close the research task."""
        return bool(self.eligible_kinds(local_problem=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "manipulable_handles": list(self.manipulable_handles),
            "observable_contrasts": list(self.observable_contrasts),
            "missing_capabilities": list(self.missing_capabilities),
            "unresolved_mechanisms": list(self.unresolved_mechanisms),
            "world_id": self.world_id,
            "probe_room": self.probe_room,
            "eligible_kinds": sorted(self.eligible_kinds()),
            "acquire_priority": self.acquire_priority(),
            "research_open": self.research_open(),
        }


def opportunity_from_menu(
    menu: Any = None,
    *,
    notebook: "ResearchNotebook | None" = None,
    missing_capabilities: Iterable[str] = (),
    unresolved_mechanisms: Iterable[str] = (),
) -> WorldOpportunity:
    from .dynworld import is_contrast

    levers = tuple(getattr(menu, "levers", ()) or ()) if menu is not None else ()
    observables = (
        tuple(getattr(menu, "observables", ()) or ()) if menu is not None else ()
    )
    inert = set(getattr(menu, "inert", ()) or ()) if menu is not None else set()
    world_id = str(getattr(menu, "world_id", "") or "") if menu is not None else ""
    contrasts: list[str] = []
    manipulable: list[str] = []
    missing: list[str] = [str(item).strip() for item in missing_capabilities if str(item).strip()]
    unresolved: list[str] = [
        str(item).strip() for item in unresolved_mechanisms if str(item).strip()
    ]
    for lever in levers:
        for obs in observables:
            handle = f"{lever}/{obs}"
            if handle in inert:
                if not is_contrast(lever):
                    line = f"{handle} inert on this freeze"
                    if line not in missing:
                        missing.append(line)
                continue
            if is_contrast(lever):
                contrasts.append(handle)
            else:
                manipulable.append(handle)
    if notebook is not None:
        for item in notebook.missing_capabilities:
            if item not in missing:
                missing.append(item)
        for item in notebook.open_mechanisms:
            if item not in unresolved:
                unresolved.append(item)
    if not world_id and "attested world" not in missing:
        missing.append("attested world")
    probe_room = bool(getattr(menu, "room_to_move", False)) if menu is not None else False
    if not probe_room:
        probe_room = bool(manipulable)
    return WorldOpportunity(
        manipulable_handles=tuple(manipulable),
        observable_contrasts=tuple(contrasts),
        missing_capabilities=tuple(missing),
        unresolved_mechanisms=tuple(unresolved),
        world_id=world_id,
        probe_room=probe_room,
    )


def kind_eligible(kind: str, opportunity: WorldOpportunity, *, origin: str = "") -> bool:
    kind = parse_idea_kind(kind)
    local = origin == ORIGIN_LOCAL or not opportunity.world_id
    return kind in opportunity.eligible_kinds(local_problem=local)


def admit_kind_transition(
    parent_kind: str,
    child_kind: str,
    reason: str,
) -> bool:
    parent = parse_idea_kind(parent_kind)
    child = parse_idea_kind(child_kind)
    if parent == child:
        return True
    allowed = ALLOWED_KIND_TRANSITIONS.get((parent, child))
    return bool(allowed and str(reason or "").strip() in allowed)


def refuse_kind_transition(
    parent_kind: str,
    child_kind: str,
    reason: str,
) -> str:
    parent = parse_idea_kind(parent_kind)
    child = parse_idea_kind(child_kind)
    if parent == child:
        return ""
    allowed = ALLOWED_KIND_TRANSITIONS.get((parent, child))
    if allowed is None:
        return (
            f"{parent} cannot become {child}; that is not an admitted "
            "scientific transition"
        )
    if str(reason or "").strip() not in allowed:
        return (
            f"{parent} -> {child} requires one of {sorted(allowed)}; "
            f"got {reason or '(none)'}. Weak results or low feasibility "
            "are not a reason to change kind"
        )
    return ""


def infer_transition_reason(
    parent_kind: str,
    child_kind: str,
    *,
    failure_types: Iterable[str] = (),
    killed_by: Iterable[str] = (),
) -> str:
    parent = parse_idea_kind(parent_kind)
    child = parse_idea_kind(child_kind)
    tokens = {str(item).strip() for item in failure_types if str(item).strip()}
    tokens.update(str(item).strip() for item in killed_by if str(item).strip())
    lowered = {item.lower() for item in tokens}
    if child == ACQUIRE and (
        MISSING_WORLD in tokens
        or MISSING_OBSERVABLE in tokens
        or "no_handle" in lowered
        or "needs_world" in lowered
        or "missing_world" in lowered
    ):
        if MISSING_OBSERVABLE in tokens or "missing_observable" in lowered:
            return MISSING_OBSERVABLE
        return MISSING_WORLD
    if child == PROBE and (
        CAUSAL_IDENTIFICATION_READY in tokens
        or PREDICTION_MANIPULABLE in tokens
        or HARVEST_SUCCESS in tokens
    ):
        if parent == THEORY:
            return PREDICTION_MANIPULABLE
        if parent == ACQUIRE:
            return HARVEST_SUCCESS
        return CAUSAL_IDENTIFICATION_READY
    if child == QUESTION and HARVEST_SUCCESS in tokens:
        return HARVEST_SUCCESS
    if child == QUESTION and PREDICTION_OBSERVABLE in tokens:
        return PREDICTION_OBSERVABLE
    return ""


def parse_failure_types(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        return ()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        head = text.split(":", 1)[0].strip().lower().replace(" ", "_")
        if head in FAILURE_TYPES and head not in out:
            out.append(head)
        elif text in FAILURE_TYPES and text not in out:
            out.append(text)
    return tuple(out)


def refine_action_for_failures(
    failure_types: Iterable[str],
    kind: str,
) -> str:
    """What the refine controller should do. Not a free kind rewrite."""
    kinds = set(parse_failure_types(list(failure_types)))
    if WRONG_KIND in kinds or MISSING_WORLD in kinds or MISSING_OBSERVABLE in kinds:
        return "transition_kind"
    if RECIPE_NOT_ATTESTABLE in kinds or PROBE_UNFAIR in kinds or MEASURE_MISMATCH in kinds:
        return "revise_design"
    if CAUSAL_OVERCLAIM in kinds or INSUFFICIENT_CONTRAST in kinds:
        return "revise_text"
    if THEORY_NOT_DISCRIMINATIVE in kinds:
        return "revise_design"
    if MISSING_VALIDATOR in kinds:
        return "enter_acquire" if parse_idea_kind(kind) != ACQUIRE else "revise_design"
    return "revise_text"


def feasibility_rubric(kind: str) -> str:
    """Kind-specific feasibility. Probe is the only two-arm rubric."""
    kind = parse_idea_kind(kind)
    return {
        QUESTION: (
            "1 = the question is unclear or the contrast is not observable, "
            "5 = the contrast is real, answers discriminate, and would change "
            "a research decision. Do not require a two-arm intervention."
        ),
        ACQUIRE: (
            "1 = the gap is vague or the recipe cannot be attested, "
            "5 = the artifact/world is constructible, validators can attest "
            "it, and success would add research capability. Do not require "
            "a two-arm test or an effect."
        ),
        THEORY: (
            "1 = assumptions or mechanism are missing, "
            "5 = predictions are derived from the mechanism and would "
            "discriminate a competing explanation. Do not require a two-arm."
        ),
        PROBE: (
            "1 = the prediction has no measurable handle, "
            "5 = a fair two-arm test on one freeze and one measure is obvious."
        ),
    }[kind]


def feasibility_requires_two_arm(kind: str) -> bool:
    return parse_idea_kind(kind) == PROBE


def two_arm_feasibility_leak(text: str, kind: str) -> bool:
    """True when a non-probe rubric still demands two-arm obviousness."""
    if feasibility_requires_two_arm(kind):
        return False
    blob = str(text or "").lower()
    return "two-arm test is obvious" in blob or "a two-arm test is obvious" in blob


def design_signature(
    *,
    lever: str = "",
    observable: str = "",
    treatment: str = "",
    control: str = "",
    measure: str = "",
    mechanism: str = "",
    prediction: str = "",
) -> str:
    payload = {
        "lever": str(lever or "").strip().lower(),
        "observable": str(observable or "").strip().lower(),
        "treatment": " ".join(str(treatment or "").lower().split()),
        "control": " ".join(str(control or "").lower().split()),
        "measure": str(measure or observable or "").strip().lower(),
        "mechanism": " ".join(str(mechanism or "").lower().split())[:120],
        "prediction": " ".join(str(prediction or "").lower().split())[:120],
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def attractor_key(
    *,
    world_id: str,
    handle: str,
    failure_mode: str,
    design_sig: str,
) -> str:
    return "|".join(
        (
            str(world_id or "").strip() or "no-world",
            str(handle or "").strip(),
            str(failure_mode or "").strip() or "unknown",
            str(design_sig or "").strip() or "unsigned",
        )
    )


def failure_mode_of(event: Mapping[str, Any] | None) -> str:
    row = dict(event or {})
    if row.get("object_absent"):
        return "object_absent"
    if row.get("dv_blind"):
        return "dv_blind"
    reason = str(row.get("reason") or row.get("error") or "").lower()
    if "placebo" in reason or "did not consume the bound world" in reason:
        return "placebo_pathway"
    if "matching" in reason:
        return "poor_matching"
    if str(row.get("verdict") or "") == "uninformative":
        return "insufficient_power"
    if str(row.get("status") or "") in {"crashed", "error"}:
        return "implementation_failure"
    if "invariant" in reason:
        return "genuinely_invariant"
    return str(row.get("failure_mode") or "unknown").strip() or "unknown"


def handle_is_blacklisted(
    blocked: Iterable[str],
    handle: str,
    *,
    world_id: str = "",
    failure_mode: str = "",
    design_sig: str = "",
) -> bool:
    taken = {str(item).strip() for item in blocked if str(item).strip()}
    if not taken:
        return False
    key = attractor_key(
        world_id=world_id,
        handle=handle,
        failure_mode=failure_mode,
        design_sig=design_sig,
    )
    if key in taken:
        return True
    # Legacy bare-handle death only applies when the new design is unsigned.
    if not design_sig and (handle in taken or handle.split("/", 1)[0] in taken):
        return True
    return False


def research_target_of(row: Any) -> str:
    if row is None:
        return ""
    if isinstance(row, Mapping):
        target = str(row.get("research_target") or "").strip()
        if target:
            return target
        fields = row.get("kind_fields") if isinstance(row.get("kind_fields"), dict) else {}
        for name in (
            "research_question",
            "capability_gap",
            "phenomenon",
            "hypothesis",
            "claim",
        ):
            text = str(fields.get(name) or row.get(name) or "").strip()
            if text:
                return text
        return ""
    target = str(getattr(row, "research_target", "") or "").strip()
    if target:
        return target
    fields = getattr(row, "kind_fields", None) or {}
    if isinstance(fields, dict):
        for name in (
            "research_question",
            "capability_gap",
            "phenomenon",
            "hypothesis",
        ):
            text = str(fields.get(name) or "").strip()
            if text:
                return text
    return str(getattr(row, "claim", "") or "").strip()


def same_research_target(left: Any, right: Any) -> bool:
    a = " ".join(research_target_of(left).lower().split())
    b = " ".join(research_target_of(right).lower().split())
    return bool(a) and a == b


def same_idea(left: Any, right: Any) -> bool:
    """Identity is idea_id, not pair."""
    a = idea_id_of(left)
    b = idea_id_of(right)
    if a and b:
        return a == b
    return False


@dataclass
class ResearchNotebook:
    """Task-local structured research state. GENERATED is not WORLD."""

    confirmed_evidence: list[dict[str, Any]] = field(default_factory=list)
    ruled_out: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    missing_capabilities: list[str] = field(default_factory=list)
    world_versions: list[dict[str, Any]] = field(default_factory=list)
    failed_designs: list[dict[str, Any]] = field(default_factory=list)
    open_mechanisms: list[str] = field(default_factory=list)
    next_actions: list[dict[str, Any]] = field(default_factory=list)
    ideas: list[dict[str, Any]] = field(default_factory=list)

    def record_idea(
        self,
        card: Any,
        *,
        status: str,
        epistemic: str = "GENERATED",
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        kind = parse_idea_kind(
            getattr(card, "idea_kind", None)
            if not isinstance(card, Mapping)
            else card.get("idea_kind")
        )
        claim = (
            str(card.get("claim") or "")
            if isinstance(card, Mapping)
            else str(getattr(card, "claim", "") or "")
        )
        row = {
            "idea_id": idea_id_of(card),
            "lineage_id": (
                str(card.get("lineage_id") or "")
                if isinstance(card, Mapping)
                else str(getattr(card, "lineage_id", "") or "")
            ),
            "idea_kind": kind,
            "pair": (
                list(card.get("pair") or [])
                if isinstance(card, Mapping)
                else list(getattr(card, "pair", ()) or [])
            ),
            "origin": origin_of(card),
            "research_target": research_target_of(card),
            "status": status,
            "epistemic": epistemic,
            "claim": claim,
        }
        if extra:
            row.update(dict(extra))
        self.ideas.append(row)
        target = research_target_of(card) or claim
        if status in {"rejected", "killed", "failed"}:
            design = {
                "idea_id": row["idea_id"],
                "idea_kind": kind,
                "target": target,
                "why": str((extra or {}).get("why") or status),
            }
            if design not in self.failed_designs:
                self.failed_designs.append(design)
        if epistemic == "WORLD" and status in {"attested", "supports", "confirmed"}:
            ev = {
                "idea_id": row["idea_id"],
                "world_id": str((extra or {}).get("world_id") or ""),
                "evidence_id": str((extra or {}).get("evidence_id") or ""),
                "role": EVIDENCE_ROLES.get(kind, PROBE_EVIDENCE),
            }
            if ev not in self.confirmed_evidence:
                self.confirmed_evidence.append(ev)
        elif kind == QUESTION and status in {"accepted", "entered", "registered"}:
            if target and target not in self.unresolved_questions:
                self.unresolved_questions.append(target)
        elif kind == ACQUIRE and status in {"accepted", "entered", "proposed"}:
            gap = str((extra or {}).get("capability_gap") or target)
            if gap and gap not in self.missing_capabilities:
                self.missing_capabilities.append(gap)
        elif kind == THEORY and epistemic != "WORLD":
            if target and target not in self.open_mechanisms:
                self.open_mechanisms.append(target)
        if status in {"accepted", "entered", "attested", "bound"}:
            action = _next_action_for(kind, status, row, extra or {})
            if action and action not in self.next_actions:
                self.next_actions.append(action)

    def prompt_lines(self) -> tuple[str, ...]:
        """Ground the next generation. Hypotheses stay labelled GENERATED."""
        if not (
            self.ideas
            or self.confirmed_evidence
            or self.missing_capabilities
            or self.world_versions
        ):
            return ()
        lines = [
            "Task-local research state (epistemic labels are binding: "
            "GENERATED is not WORLD truth; sibling claims do not become facts):"
        ]
        if self.world_versions:
            versions = ", ".join(
                f"{row.get('world_id')}[{row.get('version') or row.get('world_id')}]"
                for row in self.world_versions[:6]
            )
            lines.append(f"- available world versions: {versions}")
        if self.confirmed_evidence:
            lines.append(
                "- confirmed WORLD evidence: "
                + "; ".join(
                    f"{row.get('role')}@{row.get('world_id')}"
                    for row in self.confirmed_evidence[:6]
                )
            )
        if self.unresolved_questions:
            lines.append(
                "- unresolved questions: " + "; ".join(self.unresolved_questions[:6])
            )
        if self.missing_capabilities:
            lines.append(
                "- missing capabilities: " + "; ".join(self.missing_capabilities[:6])
            )
        if self.ruled_out:
            lines.append("- ruled out: " + "; ".join(self.ruled_out[:6]))
        if self.failed_designs:
            lines.append(
                "- failed designs (not a permanent object ban): "
                + "; ".join(
                    f"{row.get('idea_kind')}:{row.get('target')}"[:80]
                    for row in self.failed_designs[:6]
                )
            )
        if self.open_mechanisms:
            lines.append(
                "- open mechanisms (GENERATED, not WORLD): "
                + "; ".join(self.open_mechanisms[:6])
            )
        if self.next_actions:
            lines.append(
                "- candidate next actions: "
                + "; ".join(
                    f"{row.get('kind')} ({row.get('reason')})"
                    for row in self.next_actions[:6]
                )
            )
        return tuple(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed_evidence": list(self.confirmed_evidence),
            "ruled_out": list(self.ruled_out),
            "unresolved_questions": list(self.unresolved_questions),
            "missing_capabilities": list(self.missing_capabilities),
            "world_versions": list(self.world_versions),
            "failed_designs": list(self.failed_designs),
            "open_mechanisms": list(self.open_mechanisms),
            "next_actions": list(self.next_actions),
            "ideas": list(self.ideas),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ResearchNotebook":
        row = dict(payload or {})
        return cls(
            confirmed_evidence=list(row.get("confirmed_evidence") or []),
            ruled_out=list(row.get("ruled_out") or []),
            unresolved_questions=list(row.get("unresolved_questions") or []),
            missing_capabilities=list(row.get("missing_capabilities") or []),
            world_versions=list(row.get("world_versions") or []),
            failed_designs=list(row.get("failed_designs") or []),
            open_mechanisms=list(row.get("open_mechanisms") or []),
            next_actions=list(row.get("next_actions") or []),
            ideas=list(row.get("ideas") or []),
        )


def _next_action_for(
    kind: str, status: str, row: Mapping[str, Any], extra: Mapping[str, Any]
) -> dict[str, Any] | None:
    if kind == QUESTION and status in {"accepted", "entered", "registered"}:
        if extra.get("missing_world") or extra.get("capability_gap"):
            return {
                "kind": ACQUIRE,
                "reason": MISSING_WORLD,
                "lineage_id": row.get("lineage_id"),
                "parent_idea_id": row.get("idea_id"),
            }
        return {
            "kind": QUESTION,
            "reason": "observe",
            "lineage_id": row.get("lineage_id"),
            "parent_idea_id": row.get("idea_id"),
        }
    if kind == ACQUIRE and status in {"attested", "bound"}:
        return {
            "kind": PROBE if extra.get("causal_ready") else QUESTION,
            "reason": HARVEST_SUCCESS,
            "lineage_id": row.get("lineage_id"),
            "parent_idea_id": row.get("idea_id"),
            "world_id": extra.get("world_id"),
        }
    return None


def load_notebook(workspace: Path | None) -> ResearchNotebook:
    if workspace is None:
        return ResearchNotebook()
    path = Path(workspace) / RESEARCH_STATE_FILE
    if not path.is_file():
        return ResearchNotebook()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ResearchNotebook()
    if not isinstance(payload, dict):
        return ResearchNotebook()
    return ResearchNotebook.from_dict(payload)


def save_notebook(workspace: Path | None, notebook: ResearchNotebook) -> Path | None:
    if workspace is None:
        return None
    dest = Path(workspace)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / RESEARCH_STATE_FILE
    path.write_text(
        json.dumps(notebook.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True)
class WorldVersion:
    world_id: str
    evidence_id: str
    digest: str
    parent_world: str = ""
    acquisition_event: str = ""
    attestation: str = "WORLD"
    provenance: str = ""
    version: str = "W0"
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "evidence_id": self.evidence_id,
            "digest": self.digest,
            "parent_world": self.parent_world,
            "acquisition_event": self.acquisition_event,
            "attestation": self.attestation,
            "provenance": self.provenance,
            "version": self.version,
            "path": self.path,
        }


def world_attestation_id(
    *,
    world_id: str,
    digest: str,
    parent_world: str = "",
    acquisition_event: str = "",
) -> str:
    payload = {
        "world_id": str(world_id or ""),
        "digest": str(digest or ""),
        "parent_world": str(parent_world or ""),
        "acquisition_event": str(acquisition_event or ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def next_world_label(parent_version: str = "W0") -> str:
    text = str(parent_version or "W0").strip() or "W0"
    if text.startswith("W") and text[1:].isdigit():
        return f"W{int(text[1:]) + 1}"
    return "W1"


def derive_world_version(
    parent: Any,
    dest: Path,
    *,
    new_id: str,
    acquisition_event: str,
    idea_id: str = "",
    parent_version: str = "W0",
    extra_artifacts: Mapping[str, str] | None = None,
) -> Any:
    """Copy parent bytes into a new version. Never mutates the parent root.

    `extra_artifacts` writes real new files so W1 is not a metadata-only copy.
    """
    from .world import WorldFixture, digest_files

    generated = (
        str(getattr(parent, "provenance", "")).lower() in {"generated", "synthetic"}
        or str(getattr(parent, "role", "")).lower() in {"generated", "synthetic", "placebo"}
    )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    parent_root = Path(parent.root)
    files = [str(name) for name in parent.files]
    for name in files:
        src = parent_root / name
        if src.is_file():
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    event = {
        "parent_id": str(parent.id),
        "parent_digest": str(parent.digest),
        "acquisition_event": acquisition_event,
        "idea_id": idea_id,
        "child_id": new_id,
    }
    (dest / "acquisition.json").write_text(
        json.dumps(event, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if "acquisition.json" not in files:
        files.append("acquisition.json")
    for rel, content in dict(extra_artifacts or {}).items():
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if rel not in files:
            files.append(rel)
    files = sorted(files)
    digest = digest_files(dest, files)
    version = next_world_label(parent_version)
    evidence = world_attestation_id(
        world_id=new_id,
        digest=digest,
        parent_world=str(parent.id),
        acquisition_event=acquisition_event,
    )
    child = WorldFixture(
        id=new_id,
        title=str(parent.title or new_id),
        source=str(parent.source or ""),
        retrieved_at=str(parent.retrieved_at or ""),
        digest=digest,
        files=tuple(files),
        root=dest,
        domains=tuple(parent.domains or ()),
        role="generated" if generated else str(parent.role or "world"),
        schema=str(parent.schema or ""),
        load_hint=str(parent.load_hint or ""),
        source_url=str(getattr(parent, "source_url", "") or ""),
        source_digest=str(getattr(parent, "source_digest", "") or ""),
        slice_rule=getattr(parent, "slice_rule", None),
        provenance="generated" if generated else "derived",
        object_properties=dict(parent.object_properties or {})
        if getattr(parent, "object_properties", None)
        else None,
        parent_id=str(parent.id),
        version=version,
        acquisition_event=acquisition_event,
        world_evidence_id=evidence,
    )
    manifest = child.to_dict()
    manifest["files"] = files
    (dest / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return child


def version_from_fixture(fixture: Any, *, path: str = "") -> WorldVersion:
    parent = str(getattr(fixture, "parent_id", "") or "")
    digest = str(getattr(fixture, "digest", "") or "")
    world_id = str(getattr(fixture, "id", "") or "")
    event = str(getattr(fixture, "acquisition_event", "") or "")
    evidence = str(getattr(fixture, "world_evidence_id", "") or "") or world_attestation_id(
        world_id=world_id,
        digest=digest,
        parent_world=parent,
        acquisition_event=event,
    )
    return WorldVersion(
        world_id=world_id,
        evidence_id=evidence,
        digest=digest,
        parent_world=parent,
        acquisition_event=event,
        attestation="WORLD",
        provenance=str(getattr(fixture, "provenance", "") or ""),
        version=str(getattr(fixture, "version", "") or ("W1" if parent else "W0")),
        path=path or str(getattr(fixture, "root", "") or ""),
    )


def load_world_lineage(workspace: Path | None) -> list[dict[str, Any]]:
    if workspace is None:
        return []
    path = Path(workspace) / WORLD_LINEAGE_FILE
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("versions") if isinstance(payload, dict) else payload
    return [dict(row) for row in (rows or []) if isinstance(row, dict)]


def save_world_lineage(
    workspace: Path | None, versions: list[dict[str, Any]] | list[WorldVersion]
) -> Path | None:
    if workspace is None:
        return None
    dest = Path(workspace)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / WORLD_LINEAGE_FILE
    rows = [
        row.to_dict() if isinstance(row, WorldVersion) else dict(row)
        for row in versions
    ]
    path.write_text(
        json.dumps({"versions": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def append_world_version(workspace: Path | None, version: WorldVersion | dict[str, Any]) -> list[dict[str, Any]]:
    rows = load_world_lineage(workspace)
    payload = version.to_dict() if isinstance(version, WorldVersion) else dict(version)
    rows.append(payload)
    save_world_lineage(workspace, rows)
    return rows


def freeze_dir_for(workspace: Path, world_id: str, *, version: str = "") -> Path:
    """W0 stays at workspace/world/. Later versions live under worlds/<id>/."""
    workspace = Path(workspace)
    if not version or version == "W0":
        w0 = workspace / "world"
        manifest = w0 / "world.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            if str(data.get("id") or "") in {"", world_id}:
                return w0
        if not world_id or world_id == str(data.get("id") or ""):
            return w0
    return workspace / "worlds" / world_id


def evidence_belongs_to_world(
    evidence: Mapping[str, Any] | None,
    world: Any,
) -> bool:
    row = dict(evidence or {})
    if isinstance(world, Mapping):
        world_id = str(world.get("id") or "")
        digest = str(world.get("digest") or "")
    else:
        world_id = str(getattr(world, "id", "") or "")
        digest = str(getattr(world, "digest", "") or "")
    if str(row.get("world_id") or "") and world_id and str(row.get("world_id")) != world_id:
        return False
    if str(row.get("world_digest") or "") and digest and str(row.get("world_digest")) != digest:
        return False
    expected = str(row.get("expected_evidence_id") or "")
    if expected and str(row.get("evidence_id") or "") and expected != str(row.get("evidence_id")):
        return False
    return True


def assert_evidence_world(
    evidence: Mapping[str, Any] | None,
    world: Any,
) -> None:
    if evidence_belongs_to_world(evidence, world):
        return
    row = dict(evidence or {})
    if isinstance(world, Mapping):
        world_id = str(world.get("id") or "")
    else:
        world_id = str(getattr(world, "id", "") or "")
    raise CrossWorldEvidenceError(
        f"evidence {row.get('evidence_id') or '(missing)'} from world "
        f"{row.get('world_id') or row.get('world_digest') or '(unknown)'} "
        f"cannot promote a claim on {world_id or '(unbound)'}; "
        "EvidenceID / world freeze mismatch is fail-closed"
    )


@dataclass
class AcquireRecord:
    idea_id: str
    lineage_id: str
    state: str = ACQUIRE_PROPOSED
    recipe: str = ""
    validators: str = ""
    readiness_criteria: str = ""
    failure_criteria: str = ""
    expected_affordances: str = ""
    capability_gap: str = ""
    world_id: str = ""
    evidence_id: str = ""
    parent_world: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea_id": self.idea_id,
            "lineage_id": self.lineage_id,
            "state": self.state,
            "recipe": self.recipe,
            "validators": self.validators,
            "readiness_criteria": self.readiness_criteria,
            "failure_criteria": self.failure_criteria,
            "expected_affordances": self.expected_affordances,
            "capability_gap": self.capability_gap,
            "world_id": self.world_id,
            "evidence_id": self.evidence_id,
            "parent_world": self.parent_world,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AcquireRecord":
        state = str(payload.get("state") or ACQUIRE_PROPOSED)
        if state not in ACQUIRE_STATES:
            state = ACQUIRE_PROPOSED
        return cls(
            idea_id=str(payload.get("idea_id") or ""),
            lineage_id=str(payload.get("lineage_id") or ""),
            state=state,
            recipe=str(payload.get("recipe") or ""),
            validators=str(payload.get("validators") or ""),
            readiness_criteria=str(payload.get("readiness_criteria") or ""),
            failure_criteria=str(payload.get("failure_criteria") or ""),
            expected_affordances=str(payload.get("expected_affordances") or ""),
            capability_gap=str(payload.get("capability_gap") or ""),
            world_id=str(payload.get("world_id") or ""),
            evidence_id=str(payload.get("evidence_id") or ""),
            parent_world=str(payload.get("parent_world") or ""),
            note=str(payload.get("note") or ""),
        )


def propose_acquire(card: Any) -> AcquireRecord:
    fields = parse_kind_fields(
        ACQUIRE,
        card if isinstance(card, Mapping) else getattr(card, "kind_fields", None) or {},
    )
    if not isinstance(card, Mapping):
        extra = {
            "recipe": fields.get("recipe") or getattr(card, "prediction", ""),
            "claim": getattr(card, "claim", ""),
            "prediction": getattr(card, "prediction", ""),
            "mechanism": getattr(card, "mechanism", ""),
            **(getattr(card, "kind_fields", None) or {}),
        }
        fields = parse_kind_fields(ACQUIRE, extra)
    return AcquireRecord(
        idea_id=idea_id_of(card),
        lineage_id=(
            str(card.get("lineage_id") or "")
            if isinstance(card, Mapping)
            else str(getattr(card, "lineage_id", "") or "")
        )
        or lineage_id_for(
            card.get("pair") if isinstance(card, Mapping) else getattr(card, "pair", ()),
            origin=origin_of(card),
        ),
        state=ACQUIRE_PROPOSED,
        recipe=str(fields.get("recipe") or ""),
        validators=str(fields.get("validators") or ""),
        readiness_criteria=str(fields.get("readiness_criteria") or ""),
        expected_affordances=str(fields.get("expected_affordances") or ""),
        capability_gap=str(fields.get("capability_gap") or research_target_of(card)),
        parent_world=str(
            (card.get("world_version") if isinstance(card, Mapping) else getattr(card, "world_version", ""))
            or ""
        ),
    )


def validate_acquire_recipe(record: AcquireRecord) -> AcquireRecord:
    if acquire_recipe_complete(record.to_dict()):
        return replace(record, state=ACQUIRE_RECIPE_VALIDATED)
    return replace(
        record,
        state=ACQUIRE_PROPOSED,
        note="recipe needs validators and a readiness criterion before harvest",
    )


def persist_acquire(workspace: Path | None, record: AcquireRecord) -> Path | None:
    if workspace is None:
        return None
    dest = Path(workspace) / ACQUIRE_DIR
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{record.idea_id or 'unknown'}.json"
    path.write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_acquire(workspace: Path | None, idea_id: str) -> AcquireRecord | None:
    if workspace is None or not idea_id:
        return None
    path = Path(workspace) / ACQUIRE_DIR / f"{idea_id}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return AcquireRecord.from_dict(payload)


def harvest_acquired_world(
    parent: Any,
    dest: Path,
    record: AcquireRecord,
    *,
    new_id: str = "",
    parent_version: str = "W0",
) -> tuple[Any, AcquireRecord]:
    """Deterministic acquire success: new version, new EvidenceID, parent intact."""
    child_id = new_id or f"{str(getattr(parent, 'id', 'world') or 'world')}-acq-{record.idea_id[:8] or 'x'}"
    child = derive_world_version(
        parent,
        dest,
        new_id=child_id,
        acquisition_event=f"acquire:{record.idea_id}",
        idea_id=record.idea_id,
        parent_version=parent_version,
    )
    attested = replace(
        record,
        state=ACQUIRE_ATTESTED,
        world_id=str(child.id),
        evidence_id=str(child.world_evidence_id),
        parent_world=str(parent.id),
        note="harvest attested a new world version; W0 was not rewritten",
    )
    return child, attested


def bind_acquired_world(
    workspace: Path,
    child: Any,
    record: AcquireRecord,
) -> AcquireRecord:
    from .world import bind_world

    dest = Path(workspace) / "worlds" / str(child.id)
    dest.mkdir(parents=True, exist_ok=True)
    bind_world(dest, child)
    bound = replace(record, state=ACQUIRE_BOUND)
    persist_acquire(workspace, bound)
    append_world_version(workspace, version_from_fixture(child, path=str(dest)))
    return bound


def child_probe_freeze(card: Any, versions: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The probe must name the world version it freezes. No silent W0 reuse."""
    wanted = ""
    if isinstance(card, Mapping):
        wanted = str(card.get("world_version") or card.get("world_id") or "")
    else:
        wanted = str(
            getattr(card, "world_version", "") or getattr(card, "world_id", "") or ""
        )
    rows = [dict(row) for row in versions]
    if wanted:
        for row in rows:
            if str(row.get("world_id") or "") == wanted or str(row.get("version") or "") == wanted:
                return row
        return None
    return None


def typed_evidence_role(kind: str) -> str:
    return EVIDENCE_ROLES[parse_idea_kind(kind)]


def paper_section_for(kind: str) -> str:
    return PAPER_SECTIONS[parse_idea_kind(kind)]


def diagnose_goal(kind: str) -> str:
    return {
        QUESTION: "the smallest discriminating observation/contrast",
        ACQUIRE: "the smallest capability-building / harvesting plan",
        THEORY: "the smallest prediction derivation / discrimination plan",
        PROBE: "the smallest fair two-arm design on one freeze and one measure",
    }[parse_idea_kind(kind)]


def runs_two_arm(kind: str) -> bool:
    """Only probe spends a causal two-arm. Question is observational."""
    return parse_idea_kind(kind) == PROBE
