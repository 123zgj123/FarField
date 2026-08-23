"""Concept co-occurrence corpora built live from OpenAlex.

Why this exists alongside `corpus.py`. The citation corpus answers "was this pair
of *works* later cited together", and it is built by expanding outward from a
handful of roots. That expansion is also its blind spot: a post-T work only
enters the corpus if it cites a root, so a bridge card proposing two mutually
unconnected works can almost never be confirmed, and the measured 0 hits on
bridge pairs is a property of the sampling rather than of the cards.

A concept corpus does not have that shape. Whether two concepts were combined is
decided by whether one work carries both labels, which is read off that work
alone. Nothing has to cite anything for a combination to be observable, so the
bridge case becomes measurable.

Where the concepts come from depends on the catalogue, so the fetching and the
labelling sit behind a source. `OpenAlexSource` reads curated `keywords` and
`topics`; `arxiv.ArxivSource` lifts phrases out of titles and abstracts. Both
hand back the same `Work` record, and everything downstream of that -- the
vocabulary band, the T split, confirmation, counterexamples, seeds -- is
identical, so the two catalogues are comparable rather than merely analogous.
Whichever source ran is hashed into the corpus id, because two corpora with the
same policy numbers over different catalogues are different corpora.

The T split is *not* the one `derive_g_le_t` performs. That helper filters nodes
by year and keeps every edge between survivors, which is right for citations,
where an edge is dated by its endpoints. A co-occurrence edge carries its own
date: two pre-T concepts can be combined for the first time in 2021, and that
edge is precisely the thing the walker must not see. So the walker payload is
built from pre-T co-occurrences only, and `_assert_no_future_edges` checks it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from ..graph import PINNED_K, load_graph
from ..ledger import content_digest
from ..models import BlockedRecord
from ..timeslice import PINNED_T, load_heldout, load_walker_graph
from .corpus import Fetcher, CorpusError, _write

OPENALEX = "https://api.openalex.org/works"
SELECT = "id,title,publication_year,keywords,topics"
PER_PAGE = 200

# Identifies the caller so the endpoint answers from its polite pool. It does not
# change which works come back, which is why `corpus._replay_key` drops it and a
# build recorded before it existed still replays.
MAILTO = "dev@example.com"

# The vocabulary band. The ceiling is a share of the pre-T works rather than a
# count, so it means the same thing in a 1k-work slice and a 20k one; a label
# carried by more of them than this pairs with everything and the pairing says
# nothing.
GENERIC_DF = 0.15
MIN_DF = 2

# The counterexample metric. It is a per-work *rate* over the last STALE_YEARS
# before T, not a count, because the slice itself grows over time and a raw count
# rises for a concept the field is abandoning. Both bounds have to hold: the
# quantile is relative to this corpus, and the ceiling is absolute, so a corpus
# where nothing actually collapsed cannot manufacture counterexamples out of its
# slowest-growing concepts. Every number here is read from pre-T data only, so a
# counterexample is something the walker could have seen for itself.
STALE_YEARS = 3
STALE_DECLINE_QUANTILE = 0.15
STALE_DECLINE_CEILING = 0.5
STALE_QUANTILE = 0.75

MIN_COUNTEREXAMPLES = 20

CONFIRMED_QUANTILE = 0.9
CONFIRMED_MIN_DF = 5

MAX_VOCABULARY = 20000
MAX_LABELS_PER_WORK = 16


@dataclass(frozen=True)
class Work:
    """One catalogue record, reduced to what a concept corpus needs.

    `year` is the date the work entered the record, never a revision date: it
    is the only thing the T split may read.
    """

    id: str
    year: int
    labels: frozenset[str]


class ConceptSource(Protocol):
    """A catalogue, reduced to "give me both sides of T, and say what you did"."""

    def describe(self) -> dict[str, Any]: ...

    def harvest(self, fetcher: Fetcher, T: int) -> tuple[list[Work], list[Work]]: ...


@dataclass(frozen=True)
class ConceptSpec:
    topic: str
    corpus_id: str
    T: int = PINNED_T
    pre_works: int = 10000
    post_works: int = 10000
    generic_df: float = GENERIC_DF
    min_df: int = MIN_DF
    stale_years: int = STALE_YEARS
    stale_decline_quantile: float = STALE_DECLINE_QUANTILE
    stale_decline_ceiling: float = STALE_DECLINE_CEILING
    stale_quantile: float = STALE_QUANTILE
    confirmed_quantile: float = CONFIRMED_QUANTILE
    confirmed_min_df: int = CONFIRMED_MIN_DF
    max_vocabulary: int = MAX_VOCABULARY
    max_labels_per_work: int = MAX_LABELS_PER_WORK

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "concept_cooccurrence",
            "topic": self.topic,
            "corpus_id": self.corpus_id,
            "T": self.T,
            "pre_works": self.pre_works,
            "post_works": self.post_works,
            "generic_df": self.generic_df,
            "min_df": self.min_df,
            "stale_years": self.stale_years,
            "stale_decline_quantile": self.stale_decline_quantile,
            "stale_decline_ceiling": self.stale_decline_ceiling,
            "stale_quantile": self.stale_quantile,
            "confirmed_quantile": self.confirmed_quantile,
            "confirmed_min_df": self.confirmed_min_df,
            "max_vocabulary": self.max_vocabulary,
            "max_labels_per_work": self.max_labels_per_work,
            "vocabulary_rule": (
                f"a label carried by at least {self.min_df} pre-{self.T} works and by no more than "
                f"{self.generic_df * 100:g}% of them: frequent enough to pair, specific enough for the pairing to mean something; at most "
                f"{self.max_vocabulary} survive, the rarest first, and a work contributes at most "
                f"{self.max_labels_per_work} of them"
            ),
            "edge_rule": (
                f"two vocabulary concepts share an edge when some work carries both labels; the walker graph counts only pre-"
                f"{self.T} works, so an edge there means the combination had already been made"
            ),
            "confirmed_rule": (
                f"a concept absent from every pre-{self.T} work, carried by at least "
                f"{self.confirmed_min_df} post-{self.T} works, and in the top "
                f"{(1.0 - self.confirmed_quantile) * 100:g}% of post-{self.T} document frequency: a direction that did not exist at T and then became one"
            ),
            "counterexample_rule": (
                f"a vocabulary concept used by at least {self.min_df} works up to "
                f"{self.T - self.stale_years}, whose per-work usage rate over the last "
                f"{self.stale_years} years before T fell into the steepest-declining "
                f"{self.stale_decline_quantile * 100:g}% of such concepts and to no more than "
                f"{self.stale_decline_ceiling * 100:g}% of its earlier rate, with a pre-"
                f"{self.T} degree in the top "
                f"{(1.0 - self.stale_quantile) * 100:g}% of those: it was well connected and then the field moved off it. Read entirely from pre-"
                f"{self.T} data, so the walker could see it too"
            ),
            "seed_rule": (
                f"the most connected concepts whose pre-{self.T} document frequency sits between the "
                f"{SEED_BAND[0] * 100:g}th and "
                f"{SEED_BAND[1] * 100:g}th percentile of the vocabulary: above that band a concept is generic residue with no topic, below it there is too little structure to walk"
            ),
        }


def _slug(name: str) -> str:
    keep = [ch.lower() if ch.isalnum() else "-" for ch in name.strip()]
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return "concept:" + out.strip("-")


def _labels(work: dict[str, Any]) -> set[str]:
    names = {
        str(item.get("display_name") or "").strip()
        for item in work.get("keywords") or []
    }
    names |= {
        str(item.get("display_name") or "").strip()
        for item in work.get("topics") or []
    }
    return {name for name in names if name}


def _pages(
    fetcher: Fetcher, topic: str, year_filter: str, limit: int
) -> list[dict[str, Any]]:
    works: list[dict[str, Any]] = []
    cursor = "*"
    while cursor and len(works) < limit:
        url = (
            f"{OPENALEX}?"
            + urllib.parse.urlencode(
                {
                    "search": topic,
                    "filter": year_filter,
                    "per-page": str(PER_PAGE),
                    "cursor": cursor,
                    "select": SELECT,
                    "mailto": MAILTO,
                }
            )
        )
        payload = fetcher.get(url)
        results = payload.get("results") or []
        if not results:
            break
        works.extend(results)
        cursor = (payload.get("meta") or {}).get("next_cursor")
    return works[:limit]


@dataclass(frozen=True)
class OpenAlexSource:
    """The original source: a relevance-ranked topic slice with curated labels.

    Kept because the citation corpora and the first concept corpus were built
    this way and their numbers have to stay re-derivable. Its selection is a
    relevance query, so unlike an OAI subject slice it does have an expansion
    cone, and `selection_rule` says so.
    """

    topic: str
    pre_works: int = 10000
    post_works: int = 10000

    def describe(self) -> dict[str, Any]:
        return {
            "catalogue": "openalex",
            "endpoint": OPENALEX,
            "topic": self.topic,
            "select": SELECT,
            "pre_works": self.pre_works,
            "post_works": self.post_works,
            "selection_rule": (
                f"the first {self.pre_works} pre-T and {self.post_works} post-T works OpenAlex ranks as relevant to "
                f"{self.topic!r}; membership is decided by that ranking, so a combination made outside the topic slice is not observable here"
            ),
            "label_rule": "OpenAlex `keywords` and `topics` display names",
        }

    def harvest(self, fetcher: Fetcher, T: int) -> tuple[list[Work], list[Work]]:
        pre = _pages(fetcher, self.topic, f"publication_year:<{T + 1}", self.pre_works)
        post = _pages(fetcher, self.topic, f"publication_year:>{T}", self.post_works)
        return [_work(row, T) for row in pre], [_work(row, T) for row in post]


def _work(row: dict[str, Any], T: int) -> Work:
    return Work(
        id=str(row.get("id") or ""),
        year=int(row.get("publication_year") or T),
        labels=frozenset(_labels(row)),
    )


def _pairs(labels: Iterable[str]) -> list[tuple[str, str]]:
    ordered = sorted(set(labels))
    return [
        (ordered[i], ordered[j])
        for i in range(len(ordered))
        for j in range(i + 1, len(ordered))
    ]


def _quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[index]


def _assert_distinct_slugs(vocabulary: Iterable[str]) -> None:
    """Two labels that share a node id would be silently merged into one concept.

    `_slug` folds punctuation to hyphens, so a source whose labels differ only
    in punctuation collapses them. Downstream this surfaces as a duplicate node
    id far from its cause, and if the loader ever stopped checking it would
    surface as an edge between a concept and itself.
    """
    seen: dict[str, str] = {}
    clashes: list[tuple[str, str]] = []
    for name in sorted(vocabulary):
        node_id = _slug(name)
        if node_id in seen:
            clashes.append((seen[node_id], name))
        seen[node_id] = name
    if clashes:
        raise CorpusError(
            BlockedRecord(
                missing_capability="distinct_concept_ids",
                attempted="give every vocabulary concept its own node",
                unlock_condition=(
                    f"{len(clashes)} pairs of labels share a node id, e.g. "
                    f"{clashes[:3]}; normalise them in the source so the two spellings arrive as one label, rather than letting the id decide which one wins"
                ),
            )
        )


def _assert_no_future_edges(
    walker_edges: set[tuple[str, str]], pre_edges: set[tuple[str, str]]
) -> None:
    leaked = walker_edges - pre_edges
    if leaked:
        raise CorpusError(
            BlockedRecord(
                missing_capability="concept_corpus_t_split",
                attempted="build a walker graph from pre-T co-occurrence only",
                unlock_condition=(
                    f"{len(leaked)} walker edges were first made after T, e.g. "
                    f"{sorted(leaked)[:3]}; a co-occurrence edge is dated by the work that made it, not by its endpoints, so it cannot be carried over by a node-year filter"
                ),
            )
        )


SEED_BAND = (0.4, 0.8)
SEED_COUNT = 6


def _pick_seeds(walker, df: Counter, spec: ConceptSpec) -> list[str]:
    """Seeds from the middle of the frequency band, not the top of it.

    Taking the highest-degree concepts returns the generic residue that sits
    just under the vocabulary ceiling: `business`, `electrical engineering`. A
    seed like that has no topic, so a direction proposed from it cannot be
    judged as a direction for anything. The bottom of the band has the opposite
    problem, too little structure to walk. The middle is where a concept is
    specific enough to mean something and connected enough to leave.
    """
    by_id = {_slug(name): count for name, count in df.items()}
    counts = sorted(by_id[node_id] for node_id in walker.nodes if node_id in by_id)
    if not counts:
        return []
    low = counts[min(len(counts) - 1, int(SEED_BAND[0] * (len(counts) - 1)))]
    high = counts[min(len(counts) - 1, int(SEED_BAND[1] * (len(counts) - 1)))]
    band = [
        node_id
        for node_id in walker.nodes
        if low <= by_id.get(node_id, 0) <= high
    ]
    band.sort(key=lambda node_id: (-len(walker.outgoing.get(node_id, ())), node_id))
    return band[:SEED_COUNT]


def verify_concepts(dest: Path | str) -> dict[str, Any]:
    """Same integrity contract as the citation corpora: ids pin content."""
    from .corpus import verify_corpus

    return verify_corpus(dest)


def realization_years(
    dest: Path | str,
    spec: ConceptSpec,
    *,
    source: ConceptSource,
) -> dict[tuple[str, str], int]:
    """When each novel pair was first realized: the year of the earliest
    post-T work carrying both labels.

    The corpus artifacts only say *whether* a pair was made post-T -- an edge in
    `G_full` -- because realization was originally a hit count. Turning the
    value scale toward creativity needs the *when*: a pair the field made the
    year after T was lying on the table, one it took five years to make was
    ahead of the field. The year is not in any artifact, so this replays the
    recorded harvest offline and re-derives the vocabulary and the per-work
    label cap exactly as `build_concept_corpus` does, then cross-checks its
    pair counts against the manifest. A mismatch means this function and the
    build have drifted apart, and the reading would be about the drift.

    Returns slug pairs (sorted within the pair) for exactly the post-T-new
    combinations: pairs already made pre-T are absent, because they have no
    realization year, only a repetition.
    """
    from .corpus import replay_transport

    dest = Path(dest)
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    fetcher = Fetcher(dest / "raw", transport=replay_transport(dest))
    pre_works, post_works = source.harvest(fetcher, spec.T)

    df: Counter[str] = Counter()
    for work in pre_works:
        for name in work.labels:
            df[name] += 1
    ceiling = max(spec.min_df, int(spec.generic_df * len(pre_works)))
    banded = [
        (count, name)
        for name, count in df.items()
        if spec.min_df <= count <= ceiling
    ]
    banded.sort(key=lambda item: (-item[0], item[1]))
    vocabulary = {name for _, name in banded[: spec.max_vocabulary]}

    def carried(labels: frozenset[str]) -> set[str]:
        kept = labels & vocabulary
        if len(kept) <= spec.max_labels_per_work:
            return kept
        ranked = sorted(kept, key=lambda name: (-df[name], name))
        return set(ranked[: spec.max_labels_per_work])

    pre_edges: set[tuple[str, str]] = set()
    for work in pre_works:
        pre_edges.update(_pairs(carried(work.labels)))

    years: dict[tuple[str, str], int] = {}
    post_edges: set[tuple[str, str]] = set()
    for work in post_works:
        for pair in _pairs(carried(work.labels)):
            post_edges.add(pair)
            if pair in pre_edges:
                continue
            slugged = tuple(sorted((_slug(pair[0]), _slug(pair[1]))))
            years[slugged] = min(years.get(slugged, work.year), work.year)

    counts = manifest["counts"]
    if (
        len(pre_edges) != counts["pre_t_pairs"]
        or len(post_edges - pre_edges) != counts["post_t_new_pairs"]
    ):
        raise CorpusError(
            BlockedRecord(
                missing_capability="realization_year_consistency",
                attempted=f"re-derive the pair space of {manifest['corpus_uid']}",
                unlock_condition=(
                    f"the replay found {len(pre_edges)} pre-T and "
                    f"{len(post_edges - pre_edges)} new post-T pairs where the manifest"
                    f" recorded {counts['pre_t_pairs']} and {counts['post_t_new_pairs']};"
                    " this function has drifted from build_concept_corpus and must be"
                    " realigned before any lag is read off it"
                ),
            )
        )
    return years


def build_concept_corpus(
    spec: ConceptSpec,
    dest: Path | str,
    *,
    source: ConceptSource | None = None,
    transport: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Fetch both sides of T, build the vocabulary pre-T, write the artifacts."""
    dest = Path(dest)
    source = source or OpenAlexSource(
        topic=spec.topic, pre_works=spec.pre_works, post_works=spec.post_works
    )
    fetcher = Fetcher(dest / "raw", transport=transport)
    pre_works, post_works = source.harvest(fetcher, spec.T)
    if len(pre_works) < 200:
        raise CorpusError(
            BlockedRecord(
                missing_capability="pre_t_works",
                attempted=f"fetch at least 200 pre-{spec.T} works for {spec.topic!r}",
                unlock_condition=(
                    f"the catalogue returned {len(pre_works)}; widen the slice or pick a later T"
                ),
            )
        )

    df: Counter[str] = Counter()
    for work in pre_works:
        for name in work.labels:
            df[name] += 1
    ceiling = max(spec.min_df, int(spec.generic_df * len(pre_works)))
    banded = [
        (count, name)
        for name, count in df.items()
        if spec.min_df <= count <= ceiling
    ]

    # Frequency-first, so the cap keeps the most used labels inside the band.
    # `carried` ranks a work's own labels the same way, and the two have to
    # agree; its docstring says what disagreeing costs.
    banded.sort(key=lambda item: (-item[0], item[1]))
    vocabulary = {name for _, name in banded[: spec.max_vocabulary]}
    _assert_distinct_slugs(vocabulary)
    if len(vocabulary) < 50:
        raise CorpusError(
            BlockedRecord(
                missing_capability="concept_vocabulary",
                attempted="keep at least 50 usable concepts",
                unlock_condition=(
                    f"only {len(vocabulary)} labels sat between df>={spec.min_df} and df<="
                    f"{ceiling}; fetch more pre-T works"
                ),
            )
        )

    def carried(labels: frozenset[str]) -> set[str]:
        """The vocabulary a work contributes, capped at its most frequent few.

        Without the cap one abstract can carry hundreds of concepts and assert
        tens of thousands of pairs, which both stalls the build and lets a single
        verbose work dominate the edge set.

        Frequency-first, matching how the vocabulary itself was chosen, and the
        two have to agree. Keeping the rarest few instead -- which this did at
        first, on the theory that they are the most characteristic -- selects the
        vocabulary by frequency and then systematically drops those same
        frequent concepts from most of the works that contain them. The graph's
        nodes end up being a subject's established terms while its edges come
        from whichever works happened to mention them alongside nothing rarer,
        and a well-used concept can fail an "was it used" test in the very corpus
        that selected it for being used.
        """
        kept = labels & vocabulary
        if len(kept) <= spec.max_labels_per_work:
            return kept
        ranked = sorted(kept, key=lambda name: (-df[name], name))
        return set(ranked[: spec.max_labels_per_work])

    first_year: dict[str, int] = {}
    last_year: dict[str, int] = {}
    carried_df: Counter[str] = Counter()
    recent_df: Counter[str] = Counter()
    recent_from = spec.T - spec.stale_years + 1
    pre_edges: set[tuple[str, str]] = set()
    for work in pre_works:
        kept = carried(work.labels)
        for name in kept:
            first_year[name] = min(first_year.get(name, work.year), work.year)
            last_year[name] = max(last_year.get(name, work.year), work.year)
            carried_df[name] += 1
            if work.year >= recent_from:
                recent_df[name] += 1
        pre_edges.update(_pairs(kept))

    post_edges: set[tuple[str, str]] = set()
    post_only_df: Counter[str] = Counter()
    for work in post_works:
        for name in work.labels:
            if name not in df:
                post_only_df[name] += 1
        post_edges.update(_pairs(carried(work.labels)))

    candidates = [
        count for count in post_only_df.values() if count >= spec.confirmed_min_df
    ]
    floor = max(
        spec.confirmed_min_df, _quantile(candidates, spec.confirmed_quantile)
    )
    confirmed = sorted(
        _slug(name) for name, count in post_only_df.items() if count >= floor
    )
    if not confirmed:
        raise CorpusError(
            BlockedRecord(
                missing_capability="confirmed_concepts",
                attempted=f"find post-{spec.T} concepts above df {floor}",
                unlock_condition="fetch more post-T works, or lower confirmed_min_df",
            )
        )

    degree: defaultdict[str, int] = defaultdict(int)
    for left, right in pre_edges:
        degree[left] += 1
        degree[right] += 1

    early_works = sum(1 for work in pre_works if work.year < recent_from)
    recent_works = len(pre_works) - early_works
    declines: list[tuple[float, str]] = []
    for name in vocabulary:
        early = carried_df[name] - recent_df[name]
        if early < spec.min_df or not early_works or not recent_works:
            continue
        early_rate = early / early_works
        recent_rate = recent_df[name] / recent_works
        declines.append((recent_rate / early_rate, name))
    declines.sort()
    cut = int(spec.stale_decline_quantile * len(declines))
    stale = [
        name
        for ratio, name in declines[:cut]
        if ratio <= spec.stale_decline_ceiling
    ]
    stale_floor = _quantile([degree[name] for name in stale], spec.stale_quantile)
    counterexamples = sorted(
        _slug(name) for name in stale if degree[name] >= max(1, stale_floor)
    )
    if len(counterexamples) < MIN_COUNTEREXAMPLES:
        oldest = min((work.year for work in pre_works), default=spec.T)
        raise CorpusError(
            BlockedRecord(
                missing_capability="counterexample_concepts",
                attempted=(
                    f"find at least {MIN_COUNTEREXAMPLES} concepts whose usage collapsed before "
                    f"{spec.T}"
                ),
                unlock_condition=(
                    f"only {len(counterexamples)} qualified out of {len(stale)} declining concepts, themselves drawn from "
                    f"{len(declines)} with enough early usage; the pre-T window here runs from "
                    f"{oldest} to {spec.T}, and a collapse is only visible with years on both sides of "
                    f"{recent_from}, so harvest from further back or widen stale_decline_quantile"
                ),
            )
        )

    nodes = [
        {
            "id": _slug(name),
            "title": name,
            "year": first_year.get(name, spec.T),
            "last_pre_t_year": last_year.get(name, spec.T),
            "pre_t_df": df[name],
        }
        for name in sorted(vocabulary)
    ]
    post_only = sorted({name for name in post_only_df if _slug(name) in set(confirmed)})
    full_nodes = nodes + [
        {"id": _slug(name), "title": name, "year": spec.T + 1, "post_t_df": post_only_df[name]}
        for name in post_only
    ]

    confirmed_edges: set[tuple[str, str]] = set()
    confirmed_names = {name for name in post_only}
    for work in post_works:
        new = work.labels & confirmed_names
        old = carried(work.labels)
        for a in old:
            for b in new:
                confirmed_edges.add((a, b))

    def directed(pairs: Iterable[tuple[str, str]]) -> list[list[str]]:
        """Co-occurrence has no direction, so both arrows are written."""
        out: set[tuple[str, str]] = set()
        for left, right in pairs:
            out.add((_slug(left), _slug(right)))
            out.add((_slug(right), _slug(left)))
        return [list(edge) for edge in sorted(out)]

    walker_edges = set(pre_edges)
    _assert_no_future_edges(walker_edges, pre_edges)

    content = content_digest(
        {
            "spec": spec.to_dict(),
            "source": source.describe(),
            "nodes": [node["id"] for node in full_nodes],
            "pre_edges": sorted(pre_edges),
            "post_edges": sorted(post_edges),
            "confirmed": confirmed,
            "counterexample_nodes": counterexamples,
        }
    )
    corpus_uid = f"{spec.corpus_id}-{content[:12]}"

    full_payload = {
        "snapshot_id": corpus_uid,
        "k": PINNED_K,
        "T": spec.T,
        "nodes": full_nodes,
        "edges": directed(pre_edges | post_edges) + directed(confirmed_edges),
        "counterexample_nodes": counterexamples,
    }
    walker_payload = {
        "snapshot_id": f"{corpus_uid}-le-{spec.T}",
        "k": PINNED_K,
        "T": spec.T,
        "derived_from": corpus_uid,
        "nodes": nodes,
        "edges": directed(walker_edges),
        "counterexample_nodes": counterexamples,
    }
    heldout_payload = {"T": spec.T, "confirmed": confirmed}

    dest.mkdir(parents=True, exist_ok=True)
    existing = dest / "manifest.json"
    if existing.is_file():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous.get("content_digest") not in (None, content):
            raise CorpusError(
                BlockedRecord(
                    missing_capability="corpus_immutability",
                    attempted=f"write a different corpus into {dest}",
                    unlock_condition=(
                        f"{dest.name} already holds "
                        f"{previous.get('corpus_uid', 'another corpus')}; write the new fetch under a new directory rather than redefining the one results cite"
                    ),
                )
            )

    paths = {
        "G_full": dest / "G_full.json",
        "G_le_T": dest / "G_le_T.json",
        "heldout": dest / "heldout_post_T.json",
    }
    _write(paths["G_full"], full_payload)
    _write(paths["G_le_T"], walker_payload)
    _write(paths["heldout"], heldout_payload)

    walker = load_walker_graph(paths["G_le_T"], t=spec.T)
    evaluator = load_graph(paths["G_full"])
    held = load_heldout(paths["heldout"], t=spec.T)
    if held.confirmed & set(walker.nodes):
        raise CorpusError(
            BlockedRecord(
                missing_capability="confirmed_isolation",
                attempted="keep confirmed concepts out of the walker graph",
                unlock_condition="confirmed is drawn from post-T-only concepts",
            )
        )
    seed_nodes = _pick_seeds(walker, df, spec)

    manifest = {
        "spec": spec.to_dict(),
        "source": source.describe(),
        "corpus_uid": corpus_uid,
        "content_digest": content,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pages": fetcher.pages,
        "seed_nodes": seed_nodes,
        "counts": {
            "pre_works": len(pre_works),
            "post_works": len(post_works),
            "vocabulary": len(vocabulary),
            "nodes_full": len(evaluator.nodes),
            "nodes_walker": len(walker.nodes),
            "pre_t_pairs": len(pre_edges),
            "post_t_new_pairs": len(post_edges - pre_edges),
            "confirmed": len(held.confirmed),
            "confirmed_df_floor": floor,
            "counterexample_nodes": len(counterexamples),
        },
        "digests": {
            name: content_digest(json.loads(path.read_text(encoding="utf-8")))
            for name, path in paths.items()
        },
        "known_limitation": (
            f"the vocabulary is {source.describe()['catalogue']}'s labelling of this"
            " slice, so a combination the labels do not distinguish is invisible"
            " here; both arms are scored against the same labelling"
        ),
    }
    _write(dest / "manifest.json", manifest)
    return manifest
