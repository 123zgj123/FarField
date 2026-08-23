"""INV-4 human pairwise channel. Fail-closed in this alpha.

A role label is not authentication. An unauthenticated channel cannot
unlock a value-anchor install.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

CHANNEL_PATH = Path(".farfield") / "pairwise_channel.json"
ROLE_LABELS = {
    "human",
    "manager",
    "planner",
    "engineer",
    "reviewer",
    "promotion_reviewer",
    "verifier",
    "updater",
    "protected_vault",
}

MAX_RATINGS_PER_CAMPAIGN = 12
MAX_RATINGS_PER_RATER = 8
CARD_LENGTH_BOUND = 2000
SECTION_ORDER = (
    "statement",
    "prediction",
    "cheapest_falsifier",
    "falsifier_cost",
    "source_id",
    "independence_signature",
)


class PairwiseError(ValueError):
    pass


def channel_certified(project_dir: Path | str) -> bool:
    path = Path(project_dir) / CHANNEL_PATH
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("authenticated") is not True:
        return False
    principal = str(payload.get("authenticated_principal") or "").strip()
    if not principal:
        return False
    if principal in ROLE_LABELS:
        return False
    return True


def normalize_card(card: dict[str, Any]) -> dict[str, str]:
    rendered = {key: str(card.get(key, ""))[:CARD_LENGTH_BOUND] for key in SECTION_ORDER}
    return rendered


def record_pair(
    project_dir: Path | str,
    *,
    left: dict[str, Any],
    right: dict[str, Any],
    preference: str,
    opened_evidence_ids: Sequence[str],
    rater_id: str,
    campaign_id: str,
    existing_pairs: int = 0,
    rater_pairs: int = 0,
) -> dict[str, Any]:
    if not channel_certified(project_dir):
        raise PairwiseError("pairwise channel is not certified")
    if preference not in {"left", "right", "endorse_left", "endorse_right"}:
        raise PairwiseError("preference must name a side")
    if existing_pairs >= MAX_RATINGS_PER_CAMPAIGN:
        raise PairwiseError("campaign rating cap exceeded; excess is invalid")
    if rater_pairs >= MAX_RATINGS_PER_RATER:
        raise PairwiseError("rater cap exceeded; excess is invalid")
    if not opened_evidence_ids:
        raise PairwiseError("zero-evidence preferences are excluded from every H")
    if preference.startswith("endorse") and not opened_evidence_ids:
        raise PairwiseError("endorse must cite at least one opened evidence ID")
    return {
        "campaign_id": campaign_id,
        "rater_id": rater_id,
        "left": normalize_card(left),
        "right": normalize_card(right),
        "preference": preference,
        "opened_evidence_ids": list(opened_evidence_ids),
    }
