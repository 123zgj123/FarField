"""Gated install of an F0 update. Worker campaigns do not auto-install."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .extract import TypedUpdate, extract_f0
from .memory import append_capsules
from .operators import hold_value_anchor, install_operator
from .policy import (
    VALUE_ANCHOR,
    PromotionDecision,
    WaveDExtras,
    WaveDDecision,
    evaluate_wave_d,
)


def gated_install(
    project_dir: Path | str,
    update: TypedUpdate,
    extras: WaveDExtras,
    alpha2_decision: PromotionDecision,
) -> WaveDDecision:
    decision = evaluate_wave_d(update.kind, alpha2_decision, extras)
    if decision.install:
        if update.kind == "drift_skill":
            install_operator(project_dir, update.digest, spec=update.spec)
        elif update.kind == "memory":
            failures = list(update.spec.get("failures") or [])
            if failures:
                append_capsules(project_dir, failures)
        return decision
    if decision.held == VALUE_ANCHOR:
        hold_value_anchor(project_dir, update.digest, decision.reasons)
    return decision


def extract_and_gate(
    project_dir: Path | str,
    dossier,
    *,
    settlement_status: str,
    evidence_ids: Sequence[str],
    neighborhood,
    extras: WaveDExtras,
    alpha2_decision: PromotionDecision,
) -> tuple[TypedUpdate | None, WaveDDecision]:
    update = extract_f0(
        dossier,
        settlement_status=settlement_status,
        evidence_ids=evidence_ids,
        neighborhood=neighborhood,
    )
    return update, gated_install(project_dir, update, extras, alpha2_decision)
