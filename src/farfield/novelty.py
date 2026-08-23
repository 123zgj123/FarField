"""Evaluator-side novelty scoring: did an arm propose combinations nobody had made?

The timeslice value anchor asks whether a card's endpoint grew a highly cited
post-T successor. That is a popularity question, and a bibliography dump of a
famous seed wins it by construction while far-field drift, which is supposed to
walk somewhere cold, loses it by construction. It is the wrong target for a search
whose point is to surface directions that have not formed into a field yet, where
most attempts are expected to be wrong.

What far-field search claims is narrower and testable: among pairs of works that
nobody had put in one bibliography, it picks the ones that later get put together.
So a pair is scored on two questions.

Novel before T: no pre-T work cites both. A long citation chain can connect two
works that never once appeared in the same reference list, and it is that
never-combined status, not graph distance, that makes a combination new.

Realized after T: some post-T work cites both. That is the combination actually
being made by someone, independently, later.

The arm's novelty rate is a description, not a win: the dump arm proposes pairs
that are already combined, so its novelty rate is near zero by construction. The
falsifiable claim lives in the realization rate *among novel pairs*, against a
control that draws from the same novel pool at the same graph distance. That
control can beat drift, which is the point of having it.

Nothing here may be imported by walker-side code: it reads post-T years.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .graph import GraphSnapshot
from .timeslice import PINNED_T

PINNED_RESAMPLES = 200
MIN_NOVEL_PAIRS = 8


def _children(graph: GraphSnapshot, node: str) -> set[str]:
    return set(graph.outgoing.get(node, ()))


def co_citers(graph: GraphSnapshot, u: str, v: str) -> set[str]:
    """Works citing both. Edges point older -> newer, so children are citers."""
    return _children(graph, u) & _children(graph, v)


def novel_pre_t(walker: GraphSnapshot, u: str, v: str) -> bool:
    """Never combined pre-T: not co-cited, and not citing one another either.

    A direct citation is the strongest existing connection two works can have, so
    a pair joined by one is not a new combination no matter how few bibliographies
    list them together. Without this clause the dump arm scores its own references
    as novel combinations and out-proposes drift on a column it should be empty on.
    """
    if u == v or u not in walker.nodes or v not in walker.nodes:
        return False
    if v in _children(walker, u) or u in _children(walker, v):
        return False
    return not co_citers(walker, u, v)


def realized_post_t(
    full_graph: GraphSnapshot, u: str, v: str, t: int = PINNED_T
) -> tuple[str, ...]:
    return tuple(
        sorted(
            node
            for node in co_citers(full_graph, u, v)
            if int(full_graph.nodes[node].get("year", t)) > t
        )
    )


@dataclass(frozen=True)
class PairOracle:
    """What "already combined" and "combined later" mean for a corpus kind.

    Citation corpora answer both by co-citation. Concept corpora answer both by
    co-occurrence of two labels on one work, which is read off that work alone
    and therefore does not inherit the root-expansion bias that makes a bridge
    pair almost unconfirmable in a citation corpus.

    The statistics below — matched shells, resampling, the permutation read —
    are the same either way, so they take the oracle rather than being written
    twice and drifting apart.
    """

    kind: str
    novel: Any
    realized: Any

    def is_novel(self, walker: GraphSnapshot, u: str, v: str) -> bool:
        return bool(self.novel(walker, u, v))

    def is_realized(self, full_graph: GraphSnapshot, u: str, v: str, t: int) -> bool:
        return bool(self.realized(full_graph, u, v, t))


CITATION_PAIRS = PairOracle(
    kind="co_citation",
    novel=novel_pre_t,
    realized=lambda full_graph, u, v, t: realized_post_t(full_graph, u, v, t=t),
)


def _linked(graph: GraphSnapshot, u: str, v: str) -> bool:
    return v in _children(graph, u) or u in _children(graph, v)


def _concept_novel(walker: GraphSnapshot, u: str, v: str) -> bool:
    if u == v or u not in walker.nodes or v not in walker.nodes:
        return False
    return not _linked(walker, u, v)


def _concept_realized(full_graph: GraphSnapshot, u: str, v: str, t: int) -> bool:
    del t
    return _linked(full_graph, u, v)


CONCEPT_PAIRS = PairOracle(
    kind="co_occurrence",
    novel=_concept_novel,
    realized=_concept_realized,
)


def distances_from(graph: GraphSnapshot, source: str) -> dict[str, int]:
    """Directed hop counts along citation edges, which is how drift walks."""
    if source not in graph.nodes:
        return {}
    seen = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for child in graph.outgoing.get(node, ()):
            if child not in seen:
                seen[child] = seen[node] + 1
                queue.append(child)
    return seen


def card_pairs(cards: Iterable[Any], seed_node: str) -> tuple[tuple[str, str], ...]:
    """The combinations an arm put on the table: seed with each card endpoint.

    A bridge card also proposes its two parent endpoints to each other, which is
    the only pair in the whole system that no single citation path connects.
    """
    pairs: list[tuple[str, str]] = []
    for card in cards:
        nodes = tuple(getattr(card, "nodes", ()) or ())
        if not nodes:
            continue
        if seed_node and nodes[-1] != seed_node:
            pairs.append((seed_node, nodes[-1]))
        if str(getattr(card, "mode", "")) == "bridge":
            targets = _card_terminals(card)
            for left in range(len(targets)):
                for right in range(left + 1, len(targets)):
                    pairs.append((targets[left], targets[right]))
    return tuple(dict.fromkeys(pairs))


def _card_terminals(card: Any) -> tuple[str, ...]:
    nodes = tuple(getattr(card, "nodes", ()) or ())
    edges = tuple(getattr(card, "edges", ()) or ())
    if not edges:
        return nodes[-1:]
    has_outgoing = {src for src, _ in edges}
    return tuple(node for node in nodes if node not in has_outgoing)


@dataclass(frozen=True)
class ArmNovelty:
    arm: str
    n_pairs: int
    n_novel: int
    n_realized: int
    novel_pairs: tuple[tuple[str, str], ...]
    realized_pairs: tuple[tuple[str, str], ...]

    @property
    def novelty_rate(self) -> float:
        return self.n_novel / self.n_pairs if self.n_pairs else 0.0

    @property
    def realization_rate(self) -> float | None:
        """None when the arm proposed nothing novel: there is no rate to report."""
        return self.n_realized / self.n_novel if self.n_novel else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "n_pairs": self.n_pairs,
            "n_novel": self.n_novel,
            "n_realized": self.n_realized,
            "novelty_rate": self.novelty_rate,
            "realization_rate": self.realization_rate,
            "realized_pairs": [list(pair) for pair in self.realized_pairs],
        }


def score_arm_novelty(
    arm: str,
    cards: Sequence[Any],
    seed_node: str,
    walker: GraphSnapshot,
    full_graph: GraphSnapshot,
    *,
    t: int = PINNED_T,
    oracle: PairOracle = CITATION_PAIRS,
) -> ArmNovelty:
    pairs = card_pairs(cards, seed_node)
    novel = tuple(pair for pair in pairs if oracle.is_novel(walker, *pair))
    realized = tuple(
        pair for pair in novel if oracle.is_realized(full_graph, *pair, t)
    )
    return ArmNovelty(
        arm=arm,
        n_pairs=len(pairs),
        n_novel=len(novel),
        n_realized=len(realized),
        novel_pairs=novel,
        realized_pairs=realized,
    )


def matched_chance(
    novel_pairs: Sequence[tuple[str, str]],
    walker: GraphSnapshot,
    full_graph: GraphSnapshot,
    *,
    rng_seed: str,
    resamples: int = PINNED_RESAMPLES,
    t: int = PINNED_T,
    oracle: PairOracle = CITATION_PAIRS,
) -> dict[str, Any]:
    """Realization rate of random novel pairs drawn at the same hop distance.

    Matching on distance matters: drift lands three or four hops out, and a pair
    drawn from anywhere in the corpus would fail to be realized for reasons of
    topical distance rather than anything about drift. Drawing from the same
    distance shell asks the only question worth asking, which is whether drift
    picks better than chance *among the combinations equally available to it*.
    """
    if not novel_pairs:
        return {
            "measurable": False,
            "reason": "the arm proposed no novel pair, so there is nothing to match",
        }
    rng = random.Random(rng_seed)
    shells: dict[str, dict[int, list[str]]] = {}
    distance_cache: dict[str, dict[str, int]] = {}
    draws: list[int] = []
    unmatched = 0
    for _ in range(resamples):
        hits = 0
        for anchor, target in novel_pairs:
            if anchor not in shells:
                distances = distance_cache.setdefault(
                    anchor, distances_from(walker, anchor)
                )
                by_shell: dict[int, list[str]] = {}
                for node in walker.nodes:
                    if not oracle.is_novel(walker, anchor, node):
                        continue
                    by_shell.setdefault(distances.get(node, -1), []).append(node)
                shells[anchor] = {
                    key: sorted(value) for key, value in by_shell.items()
                }
            distances = distance_cache[anchor]
            shell = shells[anchor].get(distances.get(target, -1), [])
            if not shell:
                unmatched += 1
                continue
            pick = rng.choice(shell)
            if oracle.is_realized(full_graph, anchor, pick, t):
                hits += 1
        draws.append(hits)
    return {
        "measurable": True,
        "resamples": resamples,
        "n_pairs": len(novel_pairs),
        "unmatched_draws": unmatched,
        "mean_hits": sum(draws) / len(draws),
        "chance_rate": sum(draws) / (len(draws) * len(novel_pairs)),
        "draws": draws,
    }


def beats_chance(
    arm: ArmNovelty, chance: dict[str, Any], *, alpha: float = 0.05
) -> dict[str, Any]:
    """A permutation-style read: how often chance matched or beat the arm.

    With a handful of novel pairs per seed this will usually refuse, and that is
    the honest report. Far-field search is a high-variance bet; a metric that
    declared victory from eight shots would be measuring noise.
    """
    if not chance.get("measurable"):
        return {"decided": False, "reason": chance.get("reason", "chance unmeasurable")}
    draws = chance["draws"]
    at_least = sum(1 for value in draws if value >= arm.n_realized)
    p_value = at_least / len(draws)
    enough = arm.n_novel >= MIN_NOVEL_PAIRS
    return {
        "decided": enough and p_value < alpha,
        "p_value": p_value,
        "arm_hits": arm.n_realized,
        "chance_mean_hits": chance["mean_hits"],
        "n_novel": arm.n_novel,
        "reason": (
            f"{arm.n_realized} of {arm.n_novel} novel pairs were realized; matched chance realized "
            f"{chance['mean_hits']:.2f} on average, and drew at least as many in "
            f"{p_value:.0%} of {len(draws)} resamples"
        )
        + (""
           if enough
           else f"; below the {MIN_NOVEL_PAIRS}-pair floor, so no verdict"),
    }
