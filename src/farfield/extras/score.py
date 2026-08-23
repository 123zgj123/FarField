"""Score generated cards as T(plausibility) + β·T(alienness).

Sourati & Evans found that machine-generated hypotheses land best when an
implausibility ("alien") term is *traded against* plausibility rather than
maximized alone. The v1 drift scorer was, in these terms, fixed at zero
plausibility and β = +1 — pure distance — and its value verdict was "no
better than chance". This module makes the trade-off explicit and sweepable
so that claim can be tested rather than assumed.

Both raw signals arrive from elsewhere and are only combined here:

- plausibility: the pair's neighborhood Jaccard on the pre-T concept graph —
  indirect connectivity strength, the same executable quantity the oracle's
  implication check reads, used below its kill ceiling as a graded signal.
  The design also names NLL surprise as a second term; the adapter does not
  request token logprobs, so that term is absent and the pool report carries
  a blocked record saying so instead of approximating it.
- alienness: embedding distance from the seed's neighborhood, recorded per
  jump by the P0 sampler.

T is the midrank percentile within the pool being selected from. Raw Jaccard
and raw cosine distance live on incomparable scales; ranks make β mean the
same thing at every β and on every corpus. Ties get the midrank so that a
pool of identical values scores 0.5 everywhere instead of rewarding input
order, and all ordering ties break on card_id, never on dict order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = ["ScoredCard", "percentile_ranks", "score_pool", "select"]


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """Midrank percentile of each value within `values`, in (0, 1)."""
    if not values:
        return []
    n = len(values)
    ordered = sorted(values)
    first: dict[float, int] = {}
    count: dict[float, int] = {}
    for position, value in enumerate(ordered):
        first.setdefault(value, position)
        count[value] = count.get(value, 0) + 1
    return [
        (first[value] + (count[value] - 1) / 2 + 0.5) / n for value in values
    ]


@dataclass(frozen=True)
class ScoredCard:
    card_id: str
    pair_nodes: tuple[str, str]
    plausibility: float
    alienness: float
    t_plausibility: float
    t_alienness: float

    def score(self, beta: float) -> float:
        return self.t_plausibility + beta * self.t_alienness

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "pair_nodes": list(self.pair_nodes),
            "plausibility": self.plausibility,
            "alienness": self.alienness,
            "t_plausibility": round(self.t_plausibility, 6),
            "t_alienness": round(self.t_alienness, 6),
        }


def score_pool(rows: Sequence[Mapping[str, Any]]) -> list[ScoredCard]:
    """Rank-transform a pool of card rows into scoreable cards.

    The percentiles are computed within exactly the rows passed in, so the
    caller decides the reference population — selection happens among live
    cards, so passing survivors only is the honest default.

    Plausibility has two named components in the design: the executable
    graph signal (neighborhood Jaccard) and NLL surprise. When *every* row
    in the pool carries `mean_nll`, T(plausibility) is the mean of the two
    component percentiles, with low NLL ranking as more plausible — the
    model found its own card unsurprising. A pool where any row lacks the
    NLL reading falls back to the graph signal alone for the whole pool:
    mixing one-component and two-component ranks inside one selection would
    make the score mean different things for different cards.
    """
    rows = sorted(rows, key=lambda row: str(row["card_id"]))
    t_graph = percentile_ranks([float(row["plausibility_signal"]) for row in rows])
    t_alien = percentile_ranks([float(row["alienness"]) for row in rows])
    if rows and all(row.get("mean_nll") is not None for row in rows):
        t_nll = percentile_ranks([-float(row["mean_nll"]) for row in rows])
        t_plaus = [(g + n) / 2 for g, n in zip(t_graph, t_nll)]
    else:
        t_plaus = t_graph
    return [
        ScoredCard(
            card_id=str(row["card_id"]),
            pair_nodes=(str(row["pair_nodes"][0]), str(row["pair_nodes"][1])),
            plausibility=float(row["plausibility_signal"]),
            alienness=float(row["alienness"]),
            t_plausibility=tp,
            t_alienness=ta,
        )
        for row, tp, ta in zip(rows, t_plaus, t_alien)
    ]


def select(
    scored: Sequence[ScoredCard], beta: float, k: int
) -> list[ScoredCard]:
    """Top-k by score, deduplicated by pair, deterministic under ties.

    Two cards proposing the same pair count once — realization is a property
    of the pair, and letting a repeated pair occupy two selection slots would
    double-count its hit or miss.
    """
    picked: list[ScoredCard] = []
    seen: set[tuple[str, str]] = set()
    for card in sorted(scored, key=lambda c: (-c.score(beta), c.card_id)):
        pair = tuple(sorted(card.pair_nodes))
        if pair in seen:
            continue
        seen.add(pair)
        picked.append(card)
        if len(picked) == k:
            break
    return picked
