"""Same-anchor control stake: the graph-walk arm the product deleted.

The design claim behind far-field jumps is "beats a random walk at the
same distance". When `farfield run` was removed, the product lost the
measurement stake for that claim. This module puts the stake back as a
pure measurement inside every mission: both arms call the same measure
(`gate_survival`, the graph gates against the same anchor) and differ
only in where the far node came from — the LLM-directed jump, or a
matched-alienness random draw from the same corpus.

It is a stake, not a ladder. The number is emitted as a `baseline`
event and banked in the trajectory log; it never enters ranking, QD
admission, routing, promotion, or the briefing. Only pre-T facts are
read: today's graph, today's embeddings.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

from .embed import EmbeddingSpace, cosine_distance
from .generate import ConceptOracle, check_pair

PER_ARM = 8
ALIENNESS_BAND = 0.05

STAKE_NOTE = (
    "measurement stake: graph-gate survival at matched alienness, same "
    "anchor, same measure both arms. Not a verdict; enters no ranking, "
    "no routing, no state."
)


def anchor_alienness(
    space: EmbeddingSpace, near_ids: Sequence[str], node_id: str
) -> float:
    """Distance from a corpus node to the nearest anchor concept.

    One ruler for both arms: the arm nodes are re-measured with this
    same function, so neither arm inherits the jump machinery's own
    alienness bookkeeping.
    """
    vector = space.vectors[node_id]
    return min(
        cosine_distance(vector, space.vectors[near_id]) for near_id in near_ids
    )


def gate_survival(oracle: ConceptOracle, near_node: str, node: str) -> bool:
    """The shared measure: does this pair pass every graph gate?"""
    return all(check.passed for check in check_pair(oracle, near_node, node))


def control_stake(
    space: EmbeddingSpace,
    oracle: ConceptOracle,
    near_ids: Sequence[str],
    arm_nodes: Sequence[str],
    *,
    per_arm: int = PER_ARM,
    band: float = ALIENNESS_BAND,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Both arms through `gate_survival` against `near_ids[0]`.

    For each far node the mission actually proposed, up to `per_arm`
    corpus nodes within `band` of its alienness are drawn with a seeded
    rng and gated identically. Returns None when the mission proposed
    nothing or no matched controls exist — a missing stake is reported
    as missing, never invented.
    """
    if not arm_nodes or not near_ids:
        return None
    anchor = near_ids[0]
    excluded = set(near_ids) | set(arm_nodes)
    pool = [
        (node_id, anchor_alienness(space, near_ids, node_id))
        for node_id in sorted(space.vectors)
        if node_id not in excluded
    ]
    rng = random.Random(seed)
    arm_pass = 0
    control_nodes: list[str] = []
    for node in arm_nodes:
        arm_pass += gate_survival(oracle, anchor, node)
        alienness = anchor_alienness(space, near_ids, node)
        matched = [
            node_id for node_id, a in pool if abs(a - alienness) <= band
        ]
        take = min(per_arm, len(matched))
        if take:
            control_nodes.extend(rng.sample(matched, take))
    if not control_nodes:
        return None
    control_pass = sum(
        gate_survival(oracle, anchor, node) for node in control_nodes
    )
    return {
        "anchor": anchor,
        "band": band,
        "arm": {
            "n": len(arm_nodes),
            "passed": arm_pass,
            "rate": round(arm_pass / len(arm_nodes), 4),
        },
        "control": {
            "n": len(control_nodes),
            "passed": control_pass,
            "rate": round(control_pass / len(control_nodes), 4),
        },
        "note": STAKE_NOTE,
    }
