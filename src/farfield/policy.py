"""Genesis-pinned promotion policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .models import PromotionEvidence, UpdateKind, require_finite_number


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PromotionPolicy:
    version: str = "alpha-2"
    min_paired_trials: int = 6
    min_task_families: int = 2
    min_effect_lower_bound: float = 0.0
    max_worst_regression: float = 0.02
    allow_verifier_promotion: bool = False

    def validate(self) -> None:
        if not self.version.strip():
            raise ValueError("policy.version must be non-empty")
        for name in ("min_paired_trials", "min_task_families"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"policy.{name} must be a positive integer")
        require_finite_number(self.min_effect_lower_bound, "min_effect_lower_bound")
        maximum = require_finite_number(
            self.max_worst_regression, "max_worst_regression"
        )

        if maximum < 0:
            raise ValueError("max_worst_regression must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromotionPolicy":
        policy = cls(**payload)
        policy.validate()
        return policy

    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def evaluate(
        self,
        kind: UpdateKind,
        evidence: PromotionEvidence,
    ) -> PromotionDecision:
        self.validate()
        evidence.validate()
        reasons: list[str] = []

        if evidence.paired_trials < self.min_paired_trials:
            reasons.append(
                f"paired_trials={evidence.paired_trials} < {self.min_paired_trials}"
            )
        if evidence.distinct_task_families < self.min_task_families:
            reasons.append(
                "insufficient cross-task transport: "
                f"{evidence.distinct_task_families} < {self.min_task_families} families"
            )
        if evidence.effect_lower_bound <= self.min_effect_lower_bound:
            reasons.append(
                "improvement lower bound is not strictly above the pinned threshold: "
                f"{evidence.effect_lower_bound}"
            )
        if evidence.worst_regression > self.max_worst_regression:
            reasons.append(
                f"worst regression {evidence.worst_regression} exceeds "
                f"{self.max_worst_regression}"
            )
        if not evidence.matched_compute:
            reasons.append("candidate and frozen baseline did not use matched compute")
        if not evidence.frozen_baseline:
            reasons.append("no frozen-state counterfactual baseline")
        if not evidence.sealed_holdout:
            reasons.append("protected receipt is not marked as a sealed holdout")
        if not evidence.independent_verifier:
            reasons.append("protected receipt lacks an independent-verifier assertion")
        if not evidence.evidence_ids:
            reasons.append("no sealed ledger evidence IDs attached")
        if kind is UpdateKind.VERIFIER:
            if not self.allow_verifier_promotion:
                reasons.append(
                    "verifier updates are shadow-only in the pinned alpha policy"
                )
            elif evidence.distinct_task_families < 3:
                reasons.append(
                    "verifier updates require calibration on at least 3 task families"
                )

        return PromotionDecision(accepted=not reasons, reasons=tuple(reasons))


VALUE_ANCHOR = "VALUE_ANCHOR"
MIN_VALUE_PAIRS = 3
MAX_STRAWMAN_RATE = 0.25


@dataclass(frozen=True)
class WaveDExtras:
    dump_m_passed: bool
    value_anchor: str
    strawman_rate: float
    update_kind: str
    dump_seed_refs_passed: bool | None = None
    min_value_pairs: int = 0
    pairwise_channel_certified: bool = False
    falsekill_measured: bool | None = None
    falsekill_rose: bool | None = None
    timeslice_measured: bool | None = None
    timeslice_beats_baseline: bool | None = None

    def validate(self) -> None:
        if self.value_anchor not in {"none", "timeslice", "pairwise"}:
            raise ValueError("value_anchor must be none, timeslice, or pairwise")
        if self.update_kind not in {"memory", "drift_skill", "judge_skill"}:
            raise ValueError("update_kind must be memory, drift_skill, or judge_skill")
        if self.strawman_rate < 0:
            raise ValueError("strawman_rate must be non-negative")


@dataclass(frozen=True)
class WaveDDecision:
    accepted: bool
    held: str | None
    reasons: tuple[str, ...]
    install: bool


def evaluate_wave_d(
    kind: UpdateKind | str,
    alpha2_decision: PromotionDecision,
    extras: WaveDExtras,
) -> WaveDDecision:
    """Additional Wave D gate. Does not change alpha-2 ``evaluate()``."""
    extras.validate()
    reasons = list(alpha2_decision.reasons)
    if not alpha2_decision.accepted:
        return WaveDDecision(
            accepted=False, held=None, reasons=tuple(reasons), install=False
        )

    if not extras.dump_m_passed:
        reasons.append("dump-M arm did not pass")
    if extras.update_kind == "drift_skill" and extras.dump_seed_refs_passed is not True:
        reasons.append("drift skill requires dump-seed-refs to pass")
    if extras.strawman_rate > MAX_STRAWMAN_RATE:
        reasons.append(
            f"strawman_rate={extras.strawman_rate} > {MAX_STRAWMAN_RATE}"
        )

    if extras.update_kind == "judge_skill":
        if extras.falsekill_measured is not True:
            reasons.append("judge skill requires a measured false-kill result")
        elif extras.falsekill_rose:
            reasons.append("judge skill false-kill rose on the time-slice suite")
    if extras.value_anchor == "pairwise":
        if extras.min_value_pairs < MIN_VALUE_PAIRS:
            reasons.append(
                f"pairwise value arm requires min_value_pairs>={MIN_VALUE_PAIRS}"
            )
        if not extras.pairwise_channel_certified:
            reasons.append("pairwise channel is not certified")
    if extras.value_anchor == "timeslice":
        if extras.timeslice_measured is not True:
            reasons.append(
                "timeslice value is unmeasured; cannot use as a value anchor"
            )
        elif extras.timeslice_beats_baseline is not True:
            reasons.append("timeslice value did not beat dump-seed-refs baseline")
    if extras.value_anchor == "none":
        hold_reasons = tuple(reasons + ["value_unanchored"])
        return WaveDDecision(
            accepted=False,
            held=VALUE_ANCHOR,
            reasons=hold_reasons,
            install=False,
        )

    if reasons:
        return WaveDDecision(
            accepted=False, held=None, reasons=tuple(reasons), install=False
        )

    return WaveDDecision(accepted=True, held=None, reasons=(), install=True)
