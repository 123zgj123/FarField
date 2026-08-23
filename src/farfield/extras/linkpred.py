"""A task family derived from the seed topic's own citation corpus.

The two frozen toy families only cover topics that happen to mention linear
regression or thresholds, so every real topic routed to `unmatched` and lost its
near-field control. This family covers any topic that has a corpus, because the
task is built out of the corpus itself: hide a slice of the pre-`T` citation
edges and try to predict them back.

What it can and cannot say. A win here is a statement about the shape of the
topic's literature — that co-citation structure predicts citations better than
raw popularity does, or the reverse. It is not a statement about the topic's
science, and the dossier says so in `topic_relation`.

Nothing post-`T` is visible: the mask is drawn from `G_le_T`, which is the same
graph the walker sees. The evaluator's post-`T` edges are never loaded here.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

HOLDOUT_FRACTION = 0.2
MASK_SEED = 20170101
MIN_EDGES = 40
TOP_K = 10


class CorpusTooSmall(ValueError):
    pass


def edge_rows(graph: Any) -> list[list[str]]:
    """Every pre-`T` edge, in an order that does not depend on dict iteration."""
    return [[str(src), str(dst)] for src, dst in sorted(graph.edges)]


def data_script(graph: Any) -> str:
    """A `make_data.py` carrying the corpus inline, so the split is auditable.

    Embedding the edges rather than reading the graph at run time keeps the
    sandbox a closed subprocess: the frozen `data.json` is reproducible from the
    script alone, and a reviewer can diff the script to see exactly which edges
    the run was allowed to see.
    """
    rows = edge_rows(graph)
    if len(rows) < MIN_EDGES:
        raise CorpusTooSmall(
            f"corpus has {len(rows)} pre-T edges, fewer than the {MIN_EDGES} "
            "needed for a holdout split worth scoring"
        )
    return textwrap.dedent(
        """\
        import json, random
        from pathlib import Path

        EDGES = {edges}
        rng = random.Random({seed})
        order = list(range(len(EDGES)))
        rng.shuffle(order)
        cut = max(1, int(len(EDGES) * {fraction}))
        holdout_ix = sorted(order[:cut])
        train_ix = sorted(order[cut:])
        Path("data.json").write_text(
            json.dumps(
                {{
                    "train": [EDGES[i] for i in train_ix],
                    "holdout": [EDGES[i] for i in holdout_ix],
                }}
            ),
            encoding="utf-8",
        )
        """
    ).format(
        edges=json.dumps(rows),
        seed=MASK_SEED,
        fraction=HOLDOUT_FRACTION,
    )


_SCORING_PREAMBLE = textwrap.dedent(
    """\
    import json
    from pathlib import Path

    data = json.loads(Path("data.json").read_text(encoding="utf-8"))
    train, holdout = data["train"], data["holdout"]
    nodes = sorted({n for edge in train + holdout for n in edge})
    out_edges = {}
    in_edges = {}
    for u, v in train:
        out_edges.setdefault(u, set()).add(v)
        in_edges.setdefault(v, set()).add(u)
    """
)

_SCORING_EPILOGUE = textwrap.dedent(
    """\
    misses = 0
    for u, v in holdout:
        seen = out_edges.get(u, set())
        ranked = sorted(
            (c for c in nodes if c != u and c not in seen),
            key=lambda c: (-score(u, c), c),
        )
        if v not in ranked[:{top_k}]:
            misses += 1
    Path("metrics.json").write_text(
        json.dumps(
            {{
                "holdout_miss_rate": misses / len(holdout),
                "variant": "{variant}",
                "n_holdout": len(holdout),
                "top_k": {top_k},
            }}
        ),
        encoding="utf-8",
    )
    """
)


def _variant(variant: str, score_body: str) -> str:
    return (
        _SCORING_PREAMBLE
        + textwrap.dedent(score_body)
        + _SCORING_EPILOGUE.format(top_k=TOP_K, variant=variant)
    )


POPULARITY_TRAIN = _variant(
    "popularity_baseline",
    """\
    def score(u, c):
        return len(in_edges.get(c, ()))
    """,
)

COMMON_NEIGHBOUR_TRAIN = _variant(
    "common_neighbours",
    """\
    def score(u, c):
        # How many works that u cites are themselves cited by c's citers: the
        # standard co-citation signal, with popularity only as a tiebreak.
        u_out = out_edges.get(u, set())
        c_in = in_edges.get(c, set())
        shared = len(u_out & c_in)
        for w in u_out:
            if c in out_edges.get(w, ()):  # u -> w -> c
                shared += 1
        return shared
    """,
)

VARIANTS: tuple[tuple[str, str], ...] = (
    ("popularity_baseline", POPULARITY_TRAIN),
    ("common_neighbours", COMMON_NEIGHBOUR_TRAIN),
)
