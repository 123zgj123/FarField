"""Deterministic F0 extractor. Not a trained updater. Not an RSI claim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .dossier import Dossier
from .graph import PINNED_K, in_neighborhood
from .ledger import content_digest
from .operators import PREFER_DEGREE_SPEC, spec_digest


class ExtractError(ValueError):
    pass


@dataclass(frozen=True)
class TypedUpdate:
    kind: str
    spec: dict[str, Any]
    digest: str
    rationale: str


def extract_f0(
    dossier: Dossier,
    *,
    settlement_status: str,
    evidence_ids: Sequence[str],
    neighborhood: Iterable[str] | None = None,
) -> TypedUpdate:
    dossier.validate()
    if dossier.state.get("pre_judgment", True):
        raise ExtractError("pre_judgment trajectories cannot be F0 positives")
    if str(dossier.state.get("near_control") or "absent").startswith("absent"):
        raise ExtractError(
            "no near-field control ran in this campaign, so a drift gain has "
            "nothing to be measured against"
        )

    if settlement_status == "blocked" or dossier.kind == "survey":
        raise ExtractError("blocked/survey campaigns cannot be F0 positives")
    if not evidence_ids:
        raise ExtractError("zero-evidence endorse is illegal")
    if dossier.state.get("rating") == "UNRATED" and _endorsed_as_rated(dossier):
        raise ExtractError("UNRATED cannot be used as a rated positive example")

    scoped_failures = [
        item for item in dossier.failures if str(item.get("scope") or "").strip()
    ]
    if scoped_failures:
        spec = {
            "kind": "memory",
            "failures": list(scoped_failures),
            "evidence_ids": list(evidence_ids),
        }
        return TypedUpdate(
            kind="memory",
            spec=spec,
            digest=content_digest(spec),
            rationale="scoped failure capsule",
        )

    hood = set(neighborhood or ())
    live_far = live_far_cards(dossier, hood)
    if not live_far:
        if _unprobed_far_cards(dossier, hood):
            raise ExtractError("unprobed far-field cards cannot be F0 positives")
        raise ExtractError("no live far-field cards survive judgment")
    spec = dict(PREFER_DEGREE_SPEC)
    spec["source_signatures"] = sorted(
        str(card.get("independence_signature")) for card in live_far
    )
    spec["seed_k_hop"] = PINNED_K
    return TypedUpdate(
        kind="drift_skill",
        spec=spec,
        digest=spec_digest(PREFER_DEGREE_SPEC),
        rationale="prefer shortest distant paths whose endpoint degree is at least 2",
    )


def _unprobed_far_cards(
    dossier: Dossier, neighborhood: set[str]
) -> list[dict[str, Any]]:
    rows = []
    for card in dossier.conjectures:
        nodes = card.get("nodes") or []
        if not nodes or card.get("killed") or in_neighborhood(nodes, neighborhood):
            continue
        if card.get("probed") is not True:
            rows.append(card)
    return rows


def live_far_cards(
    dossier: Dossier, neighborhood: set[str]
) -> list[dict[str, Any]]:
    live = []
    for card in dossier.conjectures:
        if card.get("killed"):
            continue
        nodes = card.get("nodes") or []
        if not nodes:
            continue
        if in_neighborhood(nodes, neighborhood):
            continue
        if card.get("probed") is not True:
            continue
        live.append(card)
    return live


def _endorsed_as_rated(dossier: Dossier) -> bool:
    return bool(dossier.state.get("endorsed_as_rated"))
