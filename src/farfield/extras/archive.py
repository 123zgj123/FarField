"""QD archive, tournament, failure memory, and the generational trend read (v2 P3).

DESIGN_V2_ZH.md §3 steps [3], [5], [6]. The loop this module powers is:
survivors enter a quality-diversity archive; elites debate in a recorded
tournament; winners recombine in latent space; killed cards leave failure
capsules that the next generation's prompts carry as negative examples.
The exit criterion for the phase is a *trend test across generations*,
archived whether it goes up, down, or nowhere.

Boundaries the design pins and this module enforces by construction:

- The archive admits only cards that survived the executable oracle; a
  debate verdict can rank live cards but can never resurrect a killed one,
  because killed cards are never offered to the tournament.
- Novelty is structural: a pair already archived is refused as a duplicate
  no matter its score, and a cell's incumbent is displaced only by a
  strictly higher score — ties keep the incumbent, so insertion order
  cannot smuggle in churn.
- The judge sees two anonymous hypotheses in an order decided by a seeded
  coin, not by which card came first — position is a known judge bias, so
  it is randomized and the coin's seed is in the artifact trail.

The archive cell is (nearest registry seed to the card's far concept, in
embedding space) x (alienness band). Both axes are readable off recorded
state, so an admit decision replays byte-for-byte. This is MAP-Elites at
its smallest useful size, not a tuned instance of it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..models import BlockedRecord
from .embed import EmbeddingSpace, cosine_distance
from .llm import Completion

ALIENNESS_BANDS = 5
ELO_INITIAL = 1000.0
ELO_K = 32.0

JUDGE_SYSTEM = (
    "You are judging two research hypotheses."
    " Answer with one JSON object and nothing else."
)

JUDGE_TEMPLATE = """Two anonymous research hypotheses. Decide which is the stronger falsifiable research bet.

Prefer the one whose two-arm test would separate on an attested world, that aims at a named literature-gap sentence without declaring the limitation solved, and whose either-outcome would discriminate the mechanism from the strongest alternative. Specificity and checkability still beat vagueness. Do not prefer a card because it sounds more novel, is longer, or is more confident.

Hypothesis A:
  claim: {claim_a}
  mechanism: {mechanism_a}
  prediction: {prediction_a}

Hypothesis B:
  claim: {claim_b}
  mechanism: {mechanism_b}
  prediction: {prediction_b}

Answer with JSON: {{"winner": "A" or "B", "reason": "<=40 words"}}
"""


class DebateRefused(RuntimeError):
    """The judge answered off-schema. Recorded, never repaired."""

    def __init__(self, record: BlockedRecord):
        super().__init__(record.unlock_condition)
        self.record = record


# --------------------------------------------------------------------------
# QD archive


@dataclass(frozen=True)
class AdmitDecision:
    card_id: str
    cell: tuple[str, int]
    outcome: str  # "new_cell" | "displaced_incumbent" | "kept_incumbent" | "duplicate_pair"
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "cell": list(self.cell),
            "outcome": self.outcome,
            "score": round(self.score, 6),
        }


class QDArchive:
    """One elite per cell, structural dedupe before any score is consulted."""

    def __init__(self, space: EmbeddingSpace, seed_nodes: Sequence[str]) -> None:
        if not seed_nodes:
            raise ValueError("an archive needs at least one registry seed")
        self._space = space
        self._seeds = tuple(sorted(seed_nodes))
        self._cells: dict[tuple[str, int], dict[str, Any]] = {}
        self._pairs: set[tuple[str, str]] = set()

    def cell_of(self, far_node: str, alienness: float) -> tuple[str, int]:
        nearest = min(
            self._seeds,
            key=lambda seed: (
                cosine_distance(
                    self._space.vectors[far_node], self._space.vectors[seed]
                ),
                seed,
            ),
        )
        band = min(int(alienness * ALIENNESS_BANDS), ALIENNESS_BANDS - 1)
        return (nearest, max(band, 0))

    def admit(self, row: Mapping[str, Any], score: float) -> AdmitDecision:
        """Offer one *surviving* card. Killed cards must never reach here."""
        if row.get("killed"):
            raise ValueError(
                f"a killed card cannot enter the archive: {row.get('card_id')}"
            )
        pair = tuple(sorted(row["pair_nodes"]))
        far_node = str(row["pair_nodes"][1])
        cell = self.cell_of(far_node, float(row["alienness"]))
        card_id = str(row["card_id"])
        if pair in self._pairs:
            return AdmitDecision(card_id, cell, "duplicate_pair", score)
        incumbent = self._cells.get(cell)
        if incumbent is not None and score <= incumbent["score"]:
            return AdmitDecision(card_id, cell, "kept_incumbent", score)
        self._cells[cell] = {"row": dict(row), "score": score}
        self._pairs.add(pair)
        return AdmitDecision(
            card_id,
            cell,
            "new_cell" if incumbent is None else "displaced_incumbent",
            score,
        )

    def elites(self) -> list[dict[str, Any]]:
        """Occupied cells' cards, best score first, ties broken by card_id."""
        ranked = sorted(
            self._cells.values(),
            key=lambda item: (-item["score"], str(item["row"]["card_id"])),
        )
        return [dict(item["row"]) | {"archive_score": item["score"]} for item in ranked]

    def occupancy(self) -> dict[str, int]:
        return {
            "cells": len(self._cells),
            "pairs": len(self._pairs),
        }


# --------------------------------------------------------------------------
# Tournament: recorded pairwise debates, Elo over live cards only.


@dataclass(frozen=True)
class DebateVerdict:
    winner_card: str
    loser_card: str
    presented_first: str
    reason: str
    artifact_digest: str
    replay_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner_card": self.winner_card,
            "loser_card": self.loser_card,
            "presented_first": self.presented_first,
            "reason": self.reason,
            "artifact_digest": self.artifact_digest,
            "replay_mode": self.replay_mode,
        }


def _judge_refuse(attempted: str, unlock: str) -> DebateRefused:
    return DebateRefused(
        BlockedRecord(
            missing_capability="judge_verdict_schema",
            attempted=attempted,
            unlock_condition=unlock,
        )
    )


def _parse_verdict(completion: Completion) -> dict[str, Any]:
    text = completion.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _judge_refuse(
            "parse a JSON verdict",
            f"the judge answers with one JSON object; it answered {text[:80]!r} ({exc})",
        ) from exc
    if not isinstance(payload, dict):
        raise _judge_refuse(
            "read a JSON object verdict",
            f"the top-level value is an object, not {type(payload).__name__}",
        )
    return payload


def debate(
    client: Any,
    card_a: Mapping[str, Any],
    card_b: Mapping[str, Any],
    *,
    rng_seed: str,
) -> DebateVerdict:
    """One recorded pairwise debate. The judge never sees card ids or scores.

    A seeded coin decides which card is presented as "A": position bias is a
    measured judge failure, and randomizing it per pairing (rather than
    always putting the higher-scored card first) keeps the tournament from
    laundering the archive's own ranking through the judge.
    """
    first_is_a = random.Random(
        f"{rng_seed}:{card_a['card_id']}:{card_b['card_id']}"
    ).random() < 0.5
    first, second = (card_a, card_b) if first_is_a else (card_b, card_a)
    prompt = JUDGE_TEMPLATE.format(
        claim_a=first["claim"],
        mechanism_a=first["mechanism"],
        prediction_a=first["prediction"],
        claim_b=second["claim"],
        mechanism_b=second["mechanism"],
        prediction_b=second["prediction"],
    )
    completion = client.complete(
        prompt,
        purpose=f"tournament_debate:{first['card_id']}:{second['card_id']}",
        system=JUDGE_SYSTEM,
    )
    completion.assert_usable()
    payload = _parse_verdict(completion)
    winner_label = str(payload.get("winner") or "").strip().upper()
    if winner_label not in {"A", "B"}:
        raise _judge_refuse(
            f"map {winner_label!r} onto a hypothesis",
            'the winner field is exactly "A" or "B"',
        )
    winner, loser = (first, second) if winner_label == "A" else (second, first)
    return DebateVerdict(
        winner_card=str(winner["card_id"]),
        loser_card=str(loser["card_id"]),
        presented_first=str(first["card_id"]),
        reason=str(payload.get("reason") or "").strip(),
        artifact_digest=completion.digest,
        replay_mode=completion.mode,
    )


@dataclass
class EloTable:
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, card_id: str) -> float:
        return self.ratings.get(card_id, ELO_INITIAL)

    def record(self, winner: str, loser: str) -> None:
        expected = 1.0 / (
            1.0 + 10.0 ** ((self.rating(loser) - self.rating(winner)) / 400.0)
        )
        delta = ELO_K * (1.0 - expected)
        self.ratings[winner] = self.rating(winner) + delta
        self.ratings[loser] = self.rating(loser) - delta

    def to_dict(self) -> dict[str, float]:
        return {card: round(value, 2) for card, value in sorted(self.ratings.items())}


# --------------------------------------------------------------------------
# Failure memory


def failure_capsule(row: Mapping[str, Any]) -> dict[str, str]:
    """One killed card, compressed to what the next prompt needs to avoid it."""
    pair = row.get("pair") or row.get("pair_nodes") or ("?", "?")
    killed_by = ", ".join(row.get("killed_by") or []) or "an executable check"
    return {
        "context": f"combining '{pair[0]}' with '{pair[1]}'",
        "mechanism": f"killed by {killed_by}",
    }


# --------------------------------------------------------------------------
# The generational trend read


def trend_test(
    groups: Sequence[tuple[int, int]],
    *,
    rng_seed: str,
    resamples: int = 10000,
) -> dict[str, Any]:
    """Cochran-Armitage-style permutation trend across ordered generations.

    `groups` is (n, successes) per generation in order. The statistic is the
    generation-index-weighted success count; the null shuffles individual
    outcomes across generations with group sizes fixed, so the p-value asks
    "could a loop that learned nothing produce a slope this steep?". Seeded,
    hence byte-stable. Two-sided, because a loop that gets *worse* with
    generations is as much a finding as one that improves.
    """
    if len(groups) < 2:
        return {"measurable": False, "reason": "a trend needs at least two generations"}
    if any(k > n or n < 0 or k < 0 for n, k in groups):
        raise ValueError(f"impossible group counts: {groups}")
    total = sum(n for n, _ in groups)
    if total == 0:
        return {"measurable": False, "reason": "no cards in any generation"}

    def statistic(counts: Sequence[int]) -> float:
        return float(sum(index * k for index, k in enumerate(counts)))

    observed = statistic([k for _, k in groups])
    # Center against the expectation under exchangeability so "steep" means
    # steep relative to the same outcomes spread evenly.
    outcomes = [1] * sum(k for _, k in groups) + [0] * (
        total - sum(k for _, k in groups)
    )
    rng = random.Random(rng_seed)
    sizes = [n for n, _ in groups]
    at_least = 0
    draws: list[float] = []
    for _ in range(resamples):
        rng.shuffle(outcomes)
        counts = []
        cursor = 0
        for size in sizes:
            counts.append(sum(outcomes[cursor : cursor + size]))
            cursor += size
        draws.append(statistic(counts))
    mean_draw = sum(draws) / len(draws)
    at_least = sum(
        1 for value in draws if abs(value - mean_draw) >= abs(observed - mean_draw)
    )
    return {
        "measurable": True,
        "groups": [{"n": n, "successes": k} for n, k in groups],
        "statistic": observed,
        "null_mean": round(mean_draw, 4),
        "p_value_two_sided": round(at_least / resamples, 6),
        "resamples": resamples,
    }
