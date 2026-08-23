"""Far-field sampler: seeded jumps in embedding space (v2 P0).

DESIGN_V2_ZH.md §3 step [1]. The four operators are the vector form of the
jump taxonomy the design names — directional, interpolation toward a far
centroid, lowest-density structural holes, and analogy offsets. Every jump
records its landing vector, the real nodes retrieved around the landing,
and its alienness (distance from the landing to the seed's neighborhood),
which generalizes the v1 path provenance: a jump you cannot replay is not
provenance.

The sampler is generation-free on purpose. It proposes *where to look*, not
what to claim; claims are P1's job (LLM generation) and killing them is the
kernel's. Nothing here calls a model, so the whole stage costs zero API
budget — the cost structure the design bakes in (§4).

P0's falsifiable exit criterion lives in `ks_statistic`: the alienness
distribution of far jumps must be separable from a near-field control drawn
by the same machinery. If it is not, the sampler is a random number
generator with provenance, and the design says so would be a failure.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from ..ledger import content_digest
from .embed import EmbeddingSpace, cosine_distance

JUMP_OPERATORS = ("directional", "interpolate", "low_density", "analogy")

# Fraction of the corpus, by distance from the seed, that counts as the
# seed's own neighborhood. Alienness is measured against this set, so the
# quantile is part of the metric's definition and is recorded on every jump.
NEAR_QUANTILE = 0.2

TOP_M = 5

# Cost caps for the low-density scan; pure-Python kNN over a whole concept
# corpus would be quadratic. Both samples are seeded, so the cap changes
# resolution, never replayability.
DENSITY_CANDIDATES = 256
DENSITY_REFERENCES = 512
DENSITY_K = 4


class FarSpaceError(ValueError):
    pass


def _normalized(vector: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise FarSpaceError("jump landed on the zero vector")
    return tuple(value / norm for value in vector)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise FarSpaceError("cannot take a quantile of nothing")
    index = min(int(q * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


@dataclass(frozen=True)
class Neighborhood:
    seed_id: str
    seed_vector: tuple[float, ...]
    near_quantile: float
    radius: float
    near: tuple[str, ...]
    far: tuple[str, ...]

    def alienness(self, landing: tuple[float, ...], space: EmbeddingSpace) -> float:
        """Distance from a landing to the nearest point of the neighborhood."""
        best = cosine_distance(landing, self.seed_vector)
        for node_id in self.near:
            best = min(best, cosine_distance(landing, space.vectors[node_id]))
        return best


def neighborhood(
    space: EmbeddingSpace,
    seed: str,
    *,
    near_quantile: float = NEAR_QUANTILE,
) -> Neighborhood:
    """The seed's near set: the closest `near_quantile` of the corpus.

    `seed` is a node id when it names one, otherwise it is embedded as text,
    so a registry seed node and a one-sentence mission seed go through the
    same door.
    """
    if seed in space.vectors:
        seed_vector = space.vectors[seed]
    else:
        seed_vector = space.embed(seed)
    others = sorted(node_id for node_id in space.vectors if node_id != seed)
    if not others:
        raise FarSpaceError("a one-node corpus has no far field")
    distances = {
        node_id: cosine_distance(seed_vector, space.vectors[node_id])
        for node_id in others
    }
    radius = _quantile(sorted(distances.values()), near_quantile)
    near = tuple(n for n in others if distances[n] <= radius)
    far = tuple(n for n in others if distances[n] > radius)
    if not far:
        raise FarSpaceError("every node is inside the seed neighborhood")
    return Neighborhood(
        seed_id=seed,
        seed_vector=seed_vector,
        near_quantile=near_quantile,
        radius=radius,
        near=near,
        far=far,
    )


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


def domain_band(
    space: EmbeddingSpace,
    seed_vector: tuple[float, ...],
    *,
    near_quantile: float,
    domain_quantile: float,
) -> tuple[str, ...]:
    """Nodes farther than the seed neighborhood but inside the topic ball.

    Far-field search is supposed to leave the seed's *near set*, not the
    researcher's object domain. Restricting retrieve to this band keeps a
    jump about agent safety from landing on an unrelated graph-theory node
    just because the landing vector wandered there.
    """
    if not 0.0 < near_quantile < domain_quantile <= 1.0:
        raise FarSpaceError(
            "domain_quantile must sit strictly above near_quantile and at most 1"
        )
    others = sorted(node_id for node_id in space.vectors)
    distances = sorted(
        cosine_distance(seed_vector, space.vectors[node_id]) for node_id in others
    )
    near_r = _quantile(distances, near_quantile)
    domain_r = _quantile(distances, domain_quantile)
    band = tuple(
        node_id
        for node_id in others
        if near_r < cosine_distance(seed_vector, space.vectors[node_id]) <= domain_r
    )
    return band


def _retrieve(
    space: EmbeddingSpace,
    landing: tuple[float, ...],
    top_m: int,
    allowed: tuple[str, ...] | None = None,
) -> tuple[tuple[str, float], ...]:
    pool = (
        ((nid, space.vectors[nid]) for nid in allowed if nid in space.vectors)
        if allowed is not None
        else space.vectors.items()
    )
    ranked = sorted(
        (cosine_distance(landing, vector), node_id) for node_id, vector in pool
    )
    return tuple((node_id, distance) for distance, node_id in ranked[:top_m])


def _retrieve_in_domain(
    space: EmbeddingSpace,
    hood: Neighborhood,
    landing: tuple[float, ...],
    top_m: int,
    domain_quantile: float | None,
) -> tuple[tuple[str, float], ...]:
    """Default None: identical to unrestricted retrieve (P0 replay)."""
    if domain_quantile is None:
        return _retrieve(space, landing, top_m)
    band = domain_band(
        space,
        hood.seed_vector,
        near_quantile=hood.near_quantile,
        domain_quantile=domain_quantile,
    )
    retrieved = _retrieve(space, landing, top_m, allowed=band) if band else ()
    if len(retrieved) >= min(top_m, 1) and retrieved:
        return retrieved
    # Empty band or a landing with nothing in-domain nearby: fall back to
    # the far set so the jump still proposes somewhere to look.
    return _retrieve(space, landing, top_m, allowed=hood.far)


def _gaussian_direction(rng: random.Random, dimensions: int) -> tuple[float, ...]:
    return _normalized([rng.gauss(0.0, 1.0) for _ in range(dimensions)])


def _directional(
    space: EmbeddingSpace, hood: Neighborhood, rng: random.Random
) -> tuple[tuple[float, ...], dict[str, Any]]:
    far_distances = sorted(
        cosine_distance(hood.seed_vector, space.vectors[node_id])
        for node_id in hood.far
    )
    radius = rng.uniform(_quantile(far_distances, 0.5), far_distances[-1])
    direction = _gaussian_direction(rng, space.dimensions)
    landing = _normalized(
        [s + radius * u for s, u in zip(hood.seed_vector, direction)]
    )
    return landing, {"radius": round(radius, 10)}


def _interpolate(
    space: EmbeddingSpace, hood: Neighborhood, rng: random.Random
) -> tuple[tuple[float, ...], dict[str, Any]]:
    ranked = sorted(
        hood.far,
        key=lambda n: cosine_distance(hood.seed_vector, space.vectors[n]),
        reverse=True,
    )
    quarter = ranked[: max(1, len(ranked) // 4)]
    centroid = [0.0] * space.dimensions
    for node_id in quarter:
        for axis, value in enumerate(space.vectors[node_id]):
            centroid[axis] += value
    centroid = _normalized(centroid)
    # lambda > 0.5 by design: interpolation that stays in the seed's half is
    # a near-field move wearing the far operator's name.
    lam = rng.uniform(0.6, 0.95)
    landing = _normalized(
        [(1 - lam) * s + lam * c for s, c in zip(hood.seed_vector, centroid)]
    )
    return landing, {"lambda": round(lam, 10), "centroid_of": len(quarter)}


def _low_density(
    space: EmbeddingSpace, hood: Neighborhood, rng: random.Random
) -> tuple[tuple[float, ...], dict[str, Any]]:
    candidates = sorted(hood.far)
    if len(candidates) > DENSITY_CANDIDATES:
        candidates = rng.sample(candidates, DENSITY_CANDIDATES)
    references = sorted(space.vectors)
    if len(references) > DENSITY_REFERENCES:
        references = rng.sample(references, DENSITY_REFERENCES)
    sparsest, sparsest_gap = None, -1.0
    for node_id in candidates:
        vector = space.vectors[node_id]
        nearest = sorted(
            cosine_distance(vector, space.vectors[other])
            for other in references
            if other != node_id
        )[:DENSITY_K]
        gap = sum(nearest) / len(nearest)
        if gap > sparsest_gap:
            sparsest, sparsest_gap = node_id, gap
    assert sparsest is not None
    jitter = _gaussian_direction(rng, space.dimensions)
    landing = _normalized(
        [v + 0.1 * j for v, j in zip(space.vectors[sparsest], jitter)]
    )
    return landing, {"hole_node": sparsest, "knn_gap": round(sparsest_gap, 10)}


def _analogy(
    space: EmbeddingSpace, hood: Neighborhood, rng: random.Random
) -> tuple[tuple[float, ...], dict[str, Any]]:
    a, b = rng.sample(sorted(hood.far), 2)
    landing = _normalized(
        [
            s + x - y
            for s, x, y in zip(
                hood.seed_vector, space.vectors[a], space.vectors[b]
            )
        ]
    )
    return landing, {"plus": a, "minus": b}


_OPERATORS = {
    "directional": _directional,
    "interpolate": _interpolate,
    "low_density": _low_density,
    "analogy": _analogy,
}


def sample_jumps(
    space: EmbeddingSpace,
    seed: str,
    *,
    rng_seed: str,
    count: int,
    top_m: int = TOP_M,
    near_quantile: float = NEAR_QUANTILE,
    operators: tuple[str, ...] | None = None,
    domain_quantile: float | None = None,
) -> list[Jump]:
    """`count` far-field jumps, cycling through the operator menu.

    Replay contract: (space digest, seed, rng_seed, count) fully determine
    the output when `operators` is left at its default and `domain_quantile`
    is left at None. Each jump gets its own child rng keyed by index *and
    operator*, so an explicit cycle (the research policy's routing) changes
    only which operators run, never how a given operator lands — and old
    campaigns replay byte-identically. `domain_quantile` only changes which
    nodes retrieve may return; landings stay the same.
    """
    cycle = operators or JUMP_OPERATORS
    hood = neighborhood(space, seed, near_quantile=near_quantile)
    jumps: list[Jump] = []
    for index in range(count):
        operator = cycle[index % len(cycle)]
        rng = random.Random(f"{rng_seed}:{index}:{operator}")
        landing, params = _OPERATORS[operator](space, hood, rng)
        recorded = {**params, "near_quantile": near_quantile}
        if domain_quantile is not None:
            recorded["domain_quantile"] = domain_quantile
        jumps.append(
            Jump(
                operator=operator,
                seed_id=hood.seed_id,
                params=recorded,
                landing=landing,
                retrieved=_retrieve_in_domain(
                    space, hood, landing, top_m, domain_quantile
                ),
                alienness=hood.alienness(landing, space),
            )
        )
    return jumps


def crossover_jump(
    space: EmbeddingSpace,
    hood: Neighborhood,
    parent_a: str,
    parent_b: str,
    *,
    top_m: int = TOP_M,
    lam: float = 0.5,
    domain_quantile: float | None = None,
) -> Jump:
    """Semantic crossover (DESIGN_V2_ZH.md §3 step [6]): land between parents.

    The evolution loop's recombination is done in latent space, not in text:
    the landing is the normalized midpoint of two elite cards' far concepts,
    and what the child generation sees is whatever real nodes live near that
    midpoint — the same retrieve-then-generate door every other jump uses.
    Deterministic by construction: no rng, so a rerun recombines identically.
    """
    if parent_a not in space.vectors or parent_b not in space.vectors:
        raise FarSpaceError(
            f"crossover parents must be corpus nodes: {parent_a!r}, {parent_b!r}"
        )
    if parent_a == parent_b:
        raise FarSpaceError("crossover needs two distinct parents")
    landing = _normalized(
        [
            (1.0 - lam) * a + lam * b
            for a, b in zip(space.vectors[parent_a], space.vectors[parent_b])
        ]
    )
    return Jump(
        operator="crossover",
        seed_id=hood.seed_id,
        params={
            "parent_a": parent_a,
            "parent_b": parent_b,
            "lambda": lam,
            "near_quantile": hood.near_quantile,
            **({"domain_quantile": domain_quantile} if domain_quantile is not None else {}),
        },
        landing=landing,
        retrieved=_retrieve_in_domain(
            space, hood, landing, top_m, domain_quantile
        ),
        alienness=hood.alienness(landing, space),
    )


def near_control(
    space: EmbeddingSpace,
    seed: str,
    *,
    rng_seed: str,
    count: int,
    near_quantile: float = NEAR_QUANTILE,
) -> list[float]:
    """Alienness of points sampled *inside* the neighborhood.

    Same jitter machinery as the far operators, applied to near nodes. This
    is the control distribution the KS criterion compares against; using
    different machinery for the control would make the separation an
    artefact of the sampler, not of the geometry.
    """
    hood = neighborhood(space, seed, near_quantile=near_quantile)
    anchors = (hood.seed_id,) + hood.near
    values: list[float] = []
    for index in range(count):
        rng = random.Random(f"{rng_seed}:near:{index}")
        anchor = rng.choice(sorted(anchors))
        base = (
            hood.seed_vector
            if anchor == hood.seed_id
            else space.vectors[anchor]
        )
        jitter = _gaussian_direction(rng, space.dimensions)
        landing = _normalized([v + 0.1 * j for v, j in zip(base, jitter)])
        values.append(hood.alienness(landing, space))
    return values


def ks_statistic(sample_a: list[float], sample_b: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic, no p-value machinery.

    P0's criterion is stated on the statistic itself: D near 1 means the far
    and near alienness distributions barely overlap; D near 0 means the
    sampler failed to leave the neighborhood.
    """
    if not sample_a or not sample_b:
        raise FarSpaceError("KS needs two non-empty samples")
    a = sorted(sample_a)
    b = sorted(sample_b)
    i = j = 0
    d = 0.0
    while i < len(a) and j < len(b):
        # Ties advance both sides together; stepping one side at a time
        # would report separation between two identical samples.
        value = min(a[i], b[j])
        while i < len(a) and a[i] == value:
            i += 1
        while j < len(b) and b[j] == value:
            j += 1
        d = max(d, abs(i / len(a) - j / len(b)))
    return d
