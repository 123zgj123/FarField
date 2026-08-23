"""F1: an updater fitted on frozen T*. Extras, never kernel, never an RSI claim.

F0 (`farfield.extract.extract_f0`) returns one fixed spec no matter what the
trajectories say. F1 is the same interface with the constant removed: it ranks the
registered proposal space against the outcomes recorded in T* and emits the
winning digest.

What it may learn from
----------------------
Only walker-visible fields: each card's endpoint degree, endpoint out-degree,
path length, and whether the card survived **its own declared falsifier**. Post-T
confirmation never enters T*, so it cannot enter the fit; it is reserved for
judging F1's proposal against F0's.

Why it refuses
--------------
A fit over three candidates on a handful of card records is not evidence that one
policy is better. F1 fails closed unless T* is large enough to contain a signal
and the winner separates survivors from kills by a margin over F0's fixed choice.
A refusal is the honest output of a small T*, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..operators import (
    CANDIDATE_SPECS,
    PREFER_DEGREE_SPEC,
    endpoint_out_degree,
    spec_digest,
)


MIN_TRAJECTORIES = 3
MIN_DISCORDANT_PAIRS = 4

MIN_MARGIN = 0.1




RANKERS: dict[str, Callable[[dict[str, Any]], tuple[Any, ...]]] = {
    "prefer_endpoint_degree_ge_2": lambda row: (
        -int(row.get("endpoint_degree") or 0),
        int(row.get("path_length") or 0),
        str(row.get("endpoint") or ""),
    ),
    "prefer_longest_distant_path": lambda row: (
        -int(row.get("path_length") or 0),
        str(row.get("endpoint") or ""),
    ),
    "prefer_endpoint_onward_edges": lambda row: (
        -int(row.get("endpoint_out_degree") or 0),
        int(row.get("path_length") or 0),
        str(row.get("endpoint") or ""),
    ),
}


@dataclass(frozen=True)
class Fit:
    proposed: bool
    kind: str
    spec: dict[str, Any]
    digest: str
    rationale: str
    reason: str
    scores: tuple[dict[str, Any], ...] = ()
    n_trajectories: int = 0
    n_cards: int = 0
    n_pairs: int = 0
    baseline_score: float | None = None
    winner_score: float | None = None
    differs_from_f0: bool = False
    features: tuple[str, ...] = field(
        default=("endpoint_degree", "endpoint_out_degree", "path_length")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed": self.proposed,
            "kind": self.kind,
            "spec": dict(self.spec),
            "digest": self.digest,
            "rationale": self.rationale,
            "reason": self.reason,
            "scores": [dict(item) for item in self.scores],
            "n_trajectories": self.n_trajectories,
            "n_cards": self.n_cards,
            "n_pairs": self.n_pairs,
            "baseline_score": self.baseline_score,
            "winner_score": self.winner_score,
            "differs_from_f0": self.differs_from_f0,
            "trained_on": list(self.features),
            "label": "survived its own declared falsifier (walker-visible)",
        }


def card_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        for card in row.get("cards") or []:
            if card.get("probed"):
                records.append(card)
    return records


def concordance(records: Sequence[dict[str, Any]], name: str) -> tuple[float, int]:
    """Fraction of (survivor, killed) pairs the ranking puts in the right order.

    Ties score 0.5, so a policy that cannot tell two cards apart earns nothing.
    """
    rank = RANKERS[name]
    survivors = [row for row in records if row.get("survived")]
    kills = [row for row in records if not row.get("survived")]
    pairs = len(survivors) * len(kills)
    if not pairs:
        return 0.0, 0
    score = 0.0
    for good in survivors:
        for bad in kills:
            left, right = rank(good), rank(bad)
            if left < right:
                score += 1.0
            elif left == right:
                score += 0.5
    return score / pairs, pairs


def fit_f1(t_star: dict[str, Any]) -> Fit:
    rows = list(t_star.get("rows") or [])
    records = card_records(rows)
    baseline_name = str(PREFER_DEGREE_SPEC["name"])
    scores = []
    for spec in CANDIDATE_SPECS:
        name = str(spec["name"])
        value, pairs = concordance(records, name)
        scores.append(
            {"name": name, "concordance": value, "n_pairs": pairs, "digest": spec_digest(spec)}
        )
    n_pairs = scores[0]["n_pairs"] if scores else 0
    baseline = next(
        (item["concordance"] for item in scores if item["name"] == baseline_name), None
    )


    ranked = sorted(
        scores,
        key=lambda item: (
            -item["concordance"],
            0 if item["name"] == baseline_name else 1,
            item["name"],
        ),
    )
    winner = ranked[0] if ranked else None

    def refuse(reason: str) -> Fit:
        return Fit(
            proposed=False,
            kind="drift_skill",
            spec=dict(PREFER_DEGREE_SPEC),
            digest=spec_digest(PREFER_DEGREE_SPEC),
            rationale="F1 declined to propose; F0's fixed spec stands",
            reason=reason,
            scores=tuple(scores),
            n_trajectories=len(rows),
            n_cards=len(records),
            n_pairs=n_pairs,
            baseline_score=baseline,
            winner_score=winner["concordance"] if winner else None,
        )

    if len(rows) < MIN_TRAJECTORIES:
        return refuse(
            f"T* holds {len(rows)} trajectory(ies); a fit needs >= {MIN_TRAJECTORIES}"
        )
    if n_pairs < MIN_DISCORDANT_PAIRS:
        return refuse(
            f"T* yields {n_pairs} (survivor, killed) pair(s); a fit needs >= "
            f"{MIN_DISCORDANT_PAIRS}. With no kills or no survivors there is nothing to separate"
        )

    if winner is None or baseline is None:
        return refuse("no candidate could be scored")
    if winner["name"] == baseline_name:
        return refuse(
            "the fit selected F0's own spec; F1 has nothing to add on this T*"
        )
    if winner["concordance"] - baseline < MIN_MARGIN:
        return refuse(
            f"best candidate {winner['name']} scores "
            f"{winner['concordance']:.3f} against F0's {baseline:.3f}; the margin is under "
            f"{MIN_MARGIN}"
        )
    spec = next(
        dict(item) for item in CANDIDATE_SPECS if item["name"] == winner["name"]
    )
    fitted = dict(spec)
    fitted["fitted_on_t_star"] = t_star.get("digest")
    fitted["n_trajectories"] = len(rows)
    return Fit(
        proposed=True,
        kind="drift_skill",
        spec=fitted,
        digest=spec_digest(spec),
        rationale=(
            f"{winner['name']} ranks surviving far cards above killed ones on T* with concordance "
            f"{winner['concordance']:.3f} against F0's {baseline:.3f}"
        ),
        reason="fitted on frozen T*",
        scores=tuple(scores),
        n_trajectories=len(rows),
        n_cards=len(records),
        n_pairs=n_pairs,
        baseline_score=baseline,
        winner_score=winner["concordance"],
        differs_from_f0=True,
    )
