"""Deterministic local embeddings for corpus nodes (v2 P0).

DESIGN_V2_ZH.md §3.2 requires the embedding tier to be local, deterministic,
and free of any API dependence, with the model name and the vector digest
recorded in the manifest. This module implements the fallback tier named
there — hashed TF-IDF with a seeded random projection — because it is the
only tier that keeps the repository at zero third-party dependencies while
still giving the far-field sampler a metric space to jump in.

Two disciplines carry over from the LLM adapter:

- Replay: the same corpus embedded twice must produce byte-identical vectors
  and therefore the same digest. Nothing here reads a clock, the process
  hash seed, or the network. Feature hashing uses sha256, not `hash()`,
  because `hash()` is salted per process.
- Honesty about resolution: this is a lexical space. Two titles sharing no
  content words land far apart even if a language model would call them
  close. The far-field sampler treats that as the P0 floor to beat, not as
  ground truth about meaning.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..graph import GraphSnapshot
from ..ledger import content_digest
from .arxiv import phrases

MODEL = "hashed-idf-rp:1"

BUCKETS = 4096

DIMENSIONS = 64

# Vectors are rounded before hashing so the digest is a statement about the
# geometry rather than about the last bits of float arithmetic.
DIGEST_DECIMALS = 10


class EmbedError(ValueError):
    pass


def features(text: str) -> set[str]:
    """Unigrams and bigrams under the same tokenizer the concept builder uses.

    Reusing `arxiv.phrases` means the embedding space and the concept network
    disagree about nothing lexical: a phrase that is a concept node is also
    an embedding feature.
    """
    return phrases(text, min_gram=1, max_gram=2)


def _bucket(feature: str) -> int:
    return int.from_bytes(
        hashlib.sha256(feature.encode("utf-8")).digest()[:8], "big"
    ) % BUCKETS


@lru_cache(maxsize=BUCKETS)
def _projection_row(bucket: int) -> tuple[float, ...]:
    rng = random.Random(f"{MODEL}:{bucket}")
    return tuple(rng.gauss(0.0, 1.0) for _ in range(DIMENSIONS))


def _normalized(vector: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise EmbedError("cannot normalize a zero vector")
    return tuple(value / norm for value in vector)


def cosine_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """1 - cosine similarity, assuming unit vectors (everything here is)."""
    return 1.0 - sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class EmbeddingSpace:
    model: str
    dimensions: int
    vectors: dict[str, tuple[float, ...]]
    idf: dict[str, float]
    # Nodes whose titles yield no features under this tokenizer (in the real
    # corpora: one non-English title). They get no vector — a lexical space
    # has no honest location for text it cannot read — but they are named
    # here and hashed into the digest, so the exclusion is a recorded fact
    # rather than a silent shrink of the far pool.
    unreadable: tuple[str, ...]
    digest: str

    def embed(self, text: str) -> tuple[float, ...]:
        """Embed out-of-corpus text (a seed sentence) into this space.

        Features unseen in the corpus get the maximum idf rather than being
        dropped. Their projection rows are hash noise relative to the
        corpus, so out-of-vocabulary mass pushes the seed *away from
        everything* — which is the honest reading: a seed the corpus has no
        words for should not pretend to sit inside one of its clusters.
        """
        found = features(text)
        if not found:
            raise EmbedError(f"no content-word features in {text!r}")
        ceiling = max(self.idf.values(), default=1.0)
        dense = [0.0] * self.dimensions
        for feature in sorted(found):
            weight = self.idf.get(feature, ceiling)
            row = _projection_row(_bucket(feature))
            for axis in range(self.dimensions):
                dense[axis] += weight * row[axis]
        return _normalized(dense)


def embed_nodes(nodes: dict[str, dict[str, Any]]) -> EmbeddingSpace:
    """Embed every readable node title.

    Unreadable nodes are excluded *on the record* (see
    `EmbeddingSpace.unreadable`), never silently: a quietly missing vector
    would make the sampler's neighborhood a function of which nodes happened
    to embed, which is the kind of degradation INV-2 forbids. A corpus with
    nothing readable at all is refused outright.
    """
    if not nodes:
        raise EmbedError("cannot embed an empty node set")
    node_features: dict[str, set[str]] = {}
    unreadable: list[str] = []
    for node_id, node in nodes.items():
        found = features(str(node.get("title") or ""))
        if found:
            node_features[node_id] = found
        else:
            unreadable.append(node_id)
    if not node_features:
        raise EmbedError("no node title yields any content-word feature")
    df: dict[str, int] = {}
    for found in node_features.values():
        for feature in found:
            df[feature] = df.get(feature, 0) + 1
    total = len(node_features)
    idf = {
        feature: math.log((1 + total) / (1 + count)) + 1.0
        for feature, count in df.items()
    }
    vectors: dict[str, tuple[float, ...]] = {}
    for node_id in sorted(node_features):
        dense = [0.0] * DIMENSIONS
        for feature in sorted(node_features[node_id]):
            weight = idf[feature]
            row = _projection_row(_bucket(feature))
            for axis in range(DIMENSIONS):
                dense[axis] += weight * row[axis]
        vectors[node_id] = _normalized(dense)
    digest = content_digest(
        {
            "model": MODEL,
            "dimensions": DIMENSIONS,
            "unreadable": sorted(unreadable),
            "vectors": {
                node_id: [round(value, DIGEST_DECIMALS) for value in vector]
                for node_id, vector in vectors.items()
            },
        }
    )
    return EmbeddingSpace(
        model=MODEL,
        dimensions=DIMENSIONS,
        vectors=vectors,
        idf=idf,
        unreadable=tuple(sorted(unreadable)),
        digest=digest,
    )


def embed_graph(graph: GraphSnapshot) -> EmbeddingSpace:
    return embed_nodes(graph.nodes)
