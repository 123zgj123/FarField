"""Typed records for the hardened Farfield alpha control plane."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

CANONICAL_ID = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9][a-z0-9-]*)+$")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def require_canonical_id(value: str, name: str) -> str:
    if value != value.strip().lower() or not CANONICAL_ID.fullmatch(value):
        raise ValueError(
            f"{name} must be a canonical lowercase ID with a typed prefix: {value!r}"
        )
    return value


def require_finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


class ObjectiveLevel(str, Enum):
    RESEARCH_QUESTION = "research_question"
    HYPOTHESIS = "hypothesis"
    MISSION = "mission"


class UpdateKind(str, Enum):
    MEMORY = "memory"
    SKILL = "skill"
    VERIFIER = "verifier"
    ROUTER = "router"
    UPDATER = "updater"


class CandidateMode(str, Enum):
    ANOMALY = "anomaly"
    MECHANISM = "mechanism"
    BRIDGE = "bridge"
    FALSIFIER = "falsifier"


class TrustClass(str, Enum):
    UNTRUSTED = "untrusted"
    ATTESTED = "attested"
    BOUND = "bound"
    ADMITTED = "admitted"
    PERSISTENT = "persistent"


ENVELOPE_KEYS = (
    "wall_clock_minutes",
    "token_cost",
    "accelerator_hours",
    "api_calls",
    "literature_fetches",
    "human_rating_minutes",
)

WAVE_A_PLACEHOLDERS = {
    "dv": "not_computed",
    "graph_distance": "unknown",
    "drift_operator": "none_wave_a",
    "rating": "UNRATED",
    "taint": "clear",
    "mpf_digest": "none",
    "grounding": "span_cited_not_method_extracted",
}


@dataclass(frozen=True)
class Evidence:
    """Mission-bound evidence.

    ``artifact_uri`` must be a local ``file://`` URI in the hardened alpha so
    the runtime can bind the evidence to actual bytes. A sealed record may only
    be ingested under the ``protected_vault`` actor label.
    """

    claim: str
    artifact_uri: str
    verifier_id: str
    outcome: str
    scope: str
    mission_id: str
    clause_ids: tuple[str, ...]
    sealed: bool = False
    trust_class: TrustClass = TrustClass.BOUND
    source_id: str = ""
    span_start: int = 0
    span_end: int = 0
    id: str = field(default_factory=lambda: new_id("ev"))

    def validate(self) -> None:
        require_canonical_id(self.id, "evidence.id")
        require_canonical_id(self.mission_id, "evidence.mission_id")
        for clause_id in self.clause_ids:
            require_canonical_id(clause_id, "evidence.clause_id")
        if not self.clause_ids:
            raise ValueError("evidence.clause_ids must be non-empty")
        for name in ("claim", "artifact_uri", "verifier_id", "outcome", "scope"):
            if not getattr(self, name).strip():
                raise ValueError(f"evidence.{name} must be non-empty")
        if not self.artifact_uri.startswith("file://"):
            raise ValueError(
                "hardened alpha evidence requires a local file:// artifact"
            )
        if not isinstance(self.trust_class, TrustClass):
            raise ValueError("evidence.trust_class must be a TrustClass")
        if isinstance(self.span_start, bool) or not isinstance(self.span_start, int):
            raise ValueError("evidence.span_start must be an integer")
        if isinstance(self.span_end, bool) or not isinstance(self.span_end, int):
            raise ValueError("evidence.span_end must be an integer")
        if self.span_start < 0 or self.span_end < self.span_start:
            raise ValueError("evidence span offsets must satisfy 0 <= start <= end")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["clause_ids"] = list(self.clause_ids)
        payload["trust_class"] = self.trust_class.value
        return payload


@dataclass(frozen=True)
class PairedTrial:
    """One protected candidate-vs-frozen-baseline comparison."""

    task_family: str
    candidate_score: float
    baseline_score: float
    candidate_compute: float
    baseline_compute: float
    protected_regression: float

    def validate(self) -> None:
        if not self.task_family.strip():
            raise ValueError("trial.task_family must be non-empty")
        for name in (
            "candidate_score",
            "baseline_score",
            "candidate_compute",
            "baseline_compute",
            "protected_regression",
        ):
            require_finite_number(getattr(self, name), f"trial.{name}")
        if self.candidate_compute < 0 or self.baseline_compute < 0:
            raise ValueError("trial compute must be non-negative")
        if self.protected_regression < 0:
            raise ValueError("trial.protected_regression must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PromotionEvidence:
    """Statistics derived from an ingested protected-evaluation receipt.

    Callers do not pass this object directly to ``evaluate_update``. The runtime
    derives it from immutable paired-trial records in a receipt event.
    """

    paired_trials: int
    distinct_task_families: int
    effect_lower_bound: float
    worst_regression: float
    matched_compute: bool
    frozen_baseline: bool
    sealed_holdout: bool
    independent_verifier: bool
    rollback_ref: str
    protected_eval_id: str
    evidence_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        if isinstance(self.paired_trials, bool) or not isinstance(
            self.paired_trials, int
        ):
            raise ValueError("paired_trials must be an integer")
        if isinstance(self.distinct_task_families, bool) or not isinstance(
            self.distinct_task_families, int
        ):
            raise ValueError("distinct_task_families must be an integer")
        if self.paired_trials < 0 or self.distinct_task_families < 0:
            raise ValueError("trial and task-family counts must be non-negative")
        require_finite_number(self.effect_lower_bound, "effect_lower_bound")
        require_finite_number(self.worst_regression, "worst_regression")
        if self.worst_regression < 0:
            raise ValueError("worst_regression must be non-negative")
        if not self.rollback_ref.strip():
            raise ValueError("rollback_ref must be non-empty")
        require_canonical_id(self.protected_eval_id, "protected_eval_id")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("promotion evidence IDs must be unique")
        for evidence_id in self.evidence_ids:
            require_canonical_id(evidence_id, "promotion evidence ID")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["evidence_ids"] = list(self.evidence_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromotionEvidence":
        return cls(
            paired_trials=payload["paired_trials"],
            distinct_task_families=payload["distinct_task_families"],
            effect_lower_bound=payload["effect_lower_bound"],
            worst_regression=payload["worst_regression"],
            matched_compute=payload["matched_compute"],
            frozen_baseline=payload["frozen_baseline"],
            sealed_holdout=payload["sealed_holdout"],
            independent_verifier=payload["independent_verifier"],
            rollback_ref=payload["rollback_ref"],
            protected_eval_id=payload["protected_eval_id"],
            evidence_ids=tuple(payload["evidence_ids"]),
        )


@dataclass(frozen=True)
class Candidate:
    """A speculative research direction, not a verified knowledge claim."""

    title: str
    problem: str
    mode: CandidateMode
    importance: float
    expected_information_gain: float
    falsifiability: float
    plausibility: float
    cost: float
    independence_signature: str
    novelty_status: str = "unknown"
    id: str = field(default_factory=lambda: new_id("cand"))

    def validate(self) -> None:
        require_canonical_id(self.id, "candidate.id")
        for name in (
            "importance",
            "expected_information_gain",
            "falsifiability",
            "plausibility",
        ):
            value = require_finite_number(getattr(self, name), f"candidate.{name}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"candidate.{name} must be in [0, 1]")
        cost = require_finite_number(self.cost, "candidate.cost")
        if cost <= 0:
            raise ValueError("candidate.cost must be positive")
        if not self.title.strip() or not self.problem.strip():
            raise ValueError("candidate title and problem must be non-empty")
        if not self.independence_signature.strip():
            raise ValueError("independence_signature must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


def _legal_graph_distance(value: str) -> bool:
    if value == WAVE_A_PLACEHOLDERS["graph_distance"]:
        return True
    if isinstance(value, str) and value.isdigit() and int(value) >= 1:
        return True
    if isinstance(value, str) and value.startswith("crossover:"):
        parts = value.removeprefix("crossover:").split("+")
        return len(parts) == 2 and all(part.isdigit() and int(part) >= 1 for part in parts)
    return False


@dataclass(frozen=True)
class BudgetEnvelope:
    """INV-15 campaign envelope. Unused dimensions are zero here, not in mission budget."""

    wall_clock_minutes: float = 10.0
    token_cost: float = 0.0
    accelerator_hours: float = 0.0
    api_calls: float = 0.0
    literature_fetches: float = 3.0
    human_rating_minutes: float = 0.0

    def validate(self) -> None:
        for key in ENVELOPE_KEYS:
            value = require_finite_number(getattr(self, key), f"envelope.{key}")
            if value < 0:
                raise ValueError(f"envelope.{key} must be non-negative")
        if all(getattr(self, key) == 0.0 for key in ENVELOPE_KEYS):
            raise ValueError("envelope must bound at least one positive resource")

    def to_dict(self) -> dict[str, float]:
        self.validate()
        return {key: float(getattr(self, key)) for key in ENVELOPE_KEYS}

    def mission_budget(self) -> dict[str, float]:
        self.validate()
        positive = {key: value for key, value in self.to_dict().items() if value > 0}
        if not positive:
            raise ValueError("mission budget requires at least one positive resource")
        return positive

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BudgetEnvelope":
        data = {key: 0.0 for key in ENVELOPE_KEYS}
        if payload:
            for key, value in payload.items():
                if key not in ENVELOPE_KEYS:
                    raise ValueError(f"unknown envelope key: {key}")
                data[key] = require_finite_number(value, f"envelope.{key}")
        envelope = cls(**data)
        envelope.validate()
        return envelope


@dataclass(frozen=True)
class BlockedRecord:
    missing_capability: str
    attempted: str
    unlock_condition: str

    def validate(self) -> None:
        for name in ("missing_capability", "attempted", "unlock_condition"):
            if not getattr(self, name).strip():
                raise ValueError(f"blocked.{name} must be non-empty")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class AttestedSpan:
    source_id: str
    start: int
    end: int
    text: str
    artifact_uri: str
    title: str = ""

    def validate(self) -> None:
        if not self.source_id.strip() or not self.text.strip():
            raise ValueError("attested span requires source_id and text")
        if not self.artifact_uri.startswith("file://"):
            raise ValueError("attested span artifact must be a local file:// URI")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("attested span offsets are out of range")
        if self.end - self.start != len(self.text.encode("utf-8")):
            # Offsets are byte offsets into the stored artifact. A span whose
            # length does not match its own text cannot be re-read and checked.
            raise ValueError(
                "attested span offsets must span exactly the span text in bytes"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ConjectureCard:
    statement: str
    prediction: str
    cheapest_falsifier: str
    falsifier_cost: str
    source_id: str
    span_start: int
    span_end: int
    independence_signature: str
    graph_distance: str = WAVE_A_PLACEHOLDERS["graph_distance"]
    drift_operator: str = WAVE_A_PLACEHOLDERS["drift_operator"]
    id: str = field(default_factory=lambda: new_id("conj"))

    def validate(self) -> None:
        require_canonical_id(self.id, "conjecture.id")
        for name in (
            "statement",
            "prediction",
            "cheapest_falsifier",
            "falsifier_cost",
            "source_id",
            "independence_signature",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"conjecture.{name} must be non-empty")
        if self.graph_distance in {"0", "0.0", "0.00"} or self.graph_distance == 0:
            raise ValueError("graph_distance placeholder must be 'unknown', not 0.0")
        if not _legal_graph_distance(self.graph_distance):
            raise ValueError(
                "graph_distance must be 'unknown', a positive integer, or crossover:N+M"
            )
        if not self.source_id.strip():
            raise ValueError("conjecture must cite a literature source_id")
        if self.span_end < self.span_start:
            raise ValueError("conjecture span offsets are invalid")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
