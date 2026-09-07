"""Seed-field anchors and the jump record — not a vector jump sampler.

Product far-field exploration lives in `researchspace.py`: recovered
research trajectories, an informed first jump, then feedback-directed
later jumps whose far labels come from verified literature. This module
keeps the two things that path still needs from the concept embedding:

- `select_near_anchors`: the seed-field concepts the topic itself names,
  ranked by topic-term coverage before cosine distance;
- `Jump`: the replayable record every landing writes (operator, params,
  retrieved neighbours, alienness), so a walk you cannot replay is not
  provenance.

The old vector operators (directional / interpolate / low-density /
analogy), their near-control sampler, and the KS separation stake were
the graph-walk product of an earlier design. They chose `pair[1]` from
cosine neighbours of a cs.DS co-occurrence graph and were the source of
`feedback edge set` welded onto a code-world claim. They are gone; the
concept graph now only answers "has this pair been combined" for pairs
that are both graph nodes. Nothing here calls a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ledger import content_digest
from .embed import EmbedError, EmbeddingSpace, cosine_distance
from .domain import (
    folded_terms,
    label_covers_topic,
    label_topic_overlap,
    topic_object_phrases,
)

# Anchor selection looks past the cosine-nearest handful: a long topic
# sentence otherwise collapses onto generic neighbours. Phrase hits recover
# the object the researcher named.
ANCHOR_POOL = 64
PHRASE_NEIGHBORS = 4


class FarSpaceError(ValueError):
    pass


def select_near_anchors(
    space: EmbeddingSpace,
    topic: str,
    label_of: dict[str, str],
    *,
    k: int = 6,
    pool: int = ANCHOR_POOL,
    phrase_neighbors: int = PHRASE_NEIGHBORS,
) -> tuple[tuple[float, str], ...]:
    """Pick the seed-field concepts for generation.

    Rank by how many of the topic's own terms the label carries, then by
    distance to the topic vector. Phrase-level retrieve votes for nodes
    the whole-sentence embedding would skip. When no label overlaps the
    topic, this is the cosine-nearest handful — the previous behaviour.
    """
    if k <= 0 or not topic.strip():
        return ()
    topic_vector = space.embed(topic)
    scored: dict[str, tuple[int, float]] = {}

    def consider(nid: str, dist: float, bonus: int = 0) -> None:
        if nid not in space.vectors:
            return
        label = label_of.get(nid) or nid
        overlap = label_topic_overlap(label, topic) + bonus
        prev = scored.get(nid)
        if prev is None or (overlap, -dist) > (prev[0], -prev[1]):
            scored[nid] = (overlap, dist)

    by_topic = sorted(
        (cosine_distance(topic_vector, vector), nid)
        for nid, vector in space.vectors.items()
    )
    for dist, nid in by_topic[: max(pool, k)]:
        consider(nid, dist)

    for phrase in topic_object_phrases(topic)[:12]:
        try:
            phrase_vector = space.embed(phrase)
        except EmbedError:
            continue
        nearby = sorted(
            (cosine_distance(phrase_vector, vector), nid)
            for nid, vector in space.vectors.items()
        )
        phrase_terms = folded_terms(phrase)
        for _, nid in nearby[:phrase_neighbors]:
            label = label_of.get(nid) or nid
            dist = cosine_distance(topic_vector, space.vectors[nid])
            bonus = 1 if phrase_terms & folded_terms(label) else 0
            consider(nid, dist, bonus=bonus)

    items = [
        (overlap, dist, nid) for nid, (overlap, dist) in scored.items()
    ]
    # Coverage is the strict standard: a topic bigram carried whole, or
    # two distinctive unigrams. One stray shared token (`program` on
    # `definite program`) is a cosine neighbour wearing the topic's
    # clothes, and it must not displace the researcher's own phrases.
    covering = [
        row
        for row in items
        if label_covers_topic(label_of.get(row[2]) or row[2], topic)
    ]
    # Covering labels are the scientific object. Padding them with
    # cosine-nearest graph nodes (finite metric, balanced tree, …) is
    # how a coding-agent topic becomes a z-score-distance card.
    if covering:
        covering.sort(key=lambda row: (-row[0], row[1], row[2]))
        return tuple((dist, nid) for _, dist, nid in covering[:k])
    ranked_items = items
    ranked_items.sort(key=lambda row: (-row[0], row[1], row[2]))
    chosen = ranked_items[:k]
    if len(chosen) < k:
        seen = {nid for _, _, nid in chosen}
        rest = sorted(
            (dist, nid)
            for dist, nid in by_topic
            if nid not in seen
        )
        for dist, nid in rest:
            if len(chosen) >= k:
                break
            overlap = scored.get(nid, (0, dist))[0]
            chosen.append((overlap, dist, nid))
    return tuple((dist, nid) for _, dist, nid in chosen[:k])


@dataclass(frozen=True)
class Jump:
    operator: str
    seed_id: str
    params: dict[str, Any]
    landing: tuple[float, ...]
    retrieved: tuple[tuple[str, float], ...]
    alienness: float

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "operator": self.operator,
            "seed_id": self.seed_id,
            "params": self.params,
            "landing": [round(value, 10) for value in self.landing],
            "retrieved": [
                {"node_id": node_id, "distance": round(distance, 10)}
                for node_id, distance in self.retrieved
            ],
            "alienness": round(self.alienness, 10),
        }
        payload["digest"] = content_digest(payload)[:16]
        return payload
