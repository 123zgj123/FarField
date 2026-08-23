"""Build a real citation corpus from OpenAlex, split at the pinned T.

Until now the only graph the far-field machinery ever ran on was a hand-written
six-node fixture, so "far drift finds value" was never tested against a citation
structure anyone else produced. This module fetches real works and their real
reference lists and writes the three artifacts the kernel already knows how to
read: the walker graph G_<=T, the evaluator graph G_full, and the held-out set of
post-T confirmed nodes.

Two directional details matter.

Edges point forward in time. OpenAlex gives `referenced_works` (newer -> older),
so every edge is reversed on the way in: `(reference, work)`. Walking `outgoing`
from a pre-T seed therefore moves toward later work, which is what makes a
post-T confirmation reachable and `successor_hits_confirmed` meaningful.

The walker graph must not know the future. Today's `cited_by_count` includes
post-T citations, so it is dropped from the walker graph, and which pre-T nodes
enter the corpus is decided by OpenAlex relevance and by id, never by how cited a
work eventually became. Selection by eventual citations would hand the walker a
ranking of the answer.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..graph import PINNED_K, load_graph
from ..ledger import content_digest
from ..models import BlockedRecord
from ..timeslice import PINNED_T, derive_g_le_t, load_heldout, load_walker_graph

OPENALEX = "https://api.openalex.org/works"
SELECT = "id,title,publication_year,referenced_works,cited_by_count"
USER_AGENT = "farfield/0.4 (https://github.com/farfield; mailto:dev@example.com)"
PAUSE_SECONDS = 0.15

CONFIRMED_QUANTILE = 0.9
CONFIRMED_MIN_CITATIONS = 20

COUNTEREXAMPLE_MIN_IN_DEGREE = 2
COUNTEREXAMPLE_QUANTILE = 0.75
COUNTEREXAMPLE_MIN_AGE = 3


class CorpusError(RuntimeError):
    def __init__(self, record: BlockedRecord):
        super().__init__(record.unlock_condition)
        self.record = record


@dataclass(frozen=True)
class CorpusSpec:
    """Everything that decides the corpus, hashed into its id."""

    topic: str
    corpus_id: str
    T: int = PINNED_T
    roots: int = 6
    per_root: int = 100
    max_nodes: int = 600
    confirmed_quantile: float = CONFIRMED_QUANTILE
    confirmed_min_citations: int = CONFIRMED_MIN_CITATIONS
    counterexample_min_in_degree: int = COUNTEREXAMPLE_MIN_IN_DEGREE
    counterexample_quantile: float = COUNTEREXAMPLE_QUANTILE
    counterexample_min_age: int = COUNTEREXAMPLE_MIN_AGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "corpus_id": self.corpus_id,
            "T": self.T,
            "roots": self.roots,
            "per_root": self.per_root,
            "max_nodes": self.max_nodes,
            "confirmed_quantile": self.confirmed_quantile,
            "confirmed_min_citations": self.confirmed_min_citations,
            "counterexample_min_in_degree": self.counterexample_min_in_degree,
            "counterexample_quantile": self.counterexample_quantile,
            "counterexample_min_age": self.counterexample_min_age,
            "confirmed_rule": (
                f"post-{self.T} node in the top "
                f"{(1.0 - self.confirmed_quantile) * 100:g}% of post-{self.T} citation counts in this corpus, and cited at least "
                f"{self.confirmed_min_citations} times: the direction it took was itself taken up by others"
            ),
            "counterexample_rule": (
                f"pre-{self.T} node published on or before {self.T - self.counterexample_min_age} with no outgoing edge inside G_<=T, at least "
                f"{self.counterexample_min_in_degree} incoming ones, and an in-degree in the top "
                f"{(1.0 - self.counterexample_quantile) * 100:g}% of such dead ends: well cited for this corpus, old enough for a successor, and nothing in the corpus built forward from it"
            ),
        }


@dataclass
class Fetcher:
    """Records every page it pulls, so a corpus can be re-derived without the API.

    `api` and `suffix` exist because this now serves two catalogues. They only
    change what a failure is called and what the recorded page is named; the
    record-and-replay contract is identical for both, and a page recorded
    before those fields existed still replays, because the filename falls back
    to the old `.json` when a page record does not carry one.
    """

    raw_dir: Path
    transport: Callable[[str], bytes] | None = None
    pages: list[dict[str, str]] = field(default_factory=list)
    api: str = "openalex_api"
    host: str = "api.openalex.org"
    suffix: str = "json"
    pause: float = PAUSE_SECONDS

    def get_bytes(self, url: str) -> bytes:
        transport = self.transport or _http_get
        try:
            raw = transport(url)
        except urllib.error.HTTPError as exc:
            raise CorpusError(
                BlockedRecord(
                    missing_capability=self.api,
                    attempted=f"GET {url}",
                    unlock_condition=f"{self.host} returned HTTP {exc.code}",
                )
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CorpusError(
                BlockedRecord(
                    missing_capability=self.api,
                    attempted=f"GET {url}",
                    unlock_condition=f"{self.host} is reachable: {exc}",
                )
            ) from exc
        digest = content_digest(raw.decode("utf-8", "replace"))
        name = f"{digest[:16]}.{self.suffix}"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_dir / name).write_bytes(raw)
        self.pages.append({"url": url, "digest": digest, "file": name})
        time.sleep(self.pause)
        return raw

    def get(self, url: str) -> dict[str, Any]:
        return json.loads(self.get_bytes(url))


def _replay_key(url: str) -> str:
    """The part of a request that decides the answer.

    `mailto` identifies the caller so the API can put it in the polite pool. It
    does not change which works come back, so a build recorded before it was
    added still answers a build that sends it. Everything else is compared
    literally, because everything else does change the answer.
    """
    parts = urllib.parse.urlsplit(url)
    query = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if key != "mailto"
    ]
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(sorted(query)), "")
    )


def replay_transport(dest: Path | str) -> Callable[[str], bytes]:
    """Serve a rebuild from the pages an earlier build recorded.

    Without this, re-deriving a corpus means re-fetching, and OpenAlex answers
    the same query differently months later, so a rule change and a data change
    arrive mixed together and neither can be attributed. Replaying pins the data
    and lets a policy change be measured on its own.
    """
    dest = Path(dest)
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    by_url = {
        _replay_key(page["url"]): page.get("file") or f"{page['digest'][:16]}.json"
        for page in manifest.get("pages") or []
    }

    def transport(url: str) -> bytes:
        name = by_url.get(_replay_key(url))
        if name is None:
            raise CorpusError(
                BlockedRecord(
                    missing_capability="recorded_page",
                    attempted=f"replay {url}",
                    unlock_condition="the recorded build asked for this url; it did not, so the rebuild is not the same query and must go to the network",
                )
            )
        return (dest / "raw" / name).read_bytes()

    return transport


RETRY_STATUS = (429, 500, 502, 503, 504)
MAX_RETRIES = 5

MAX_BACKOFF_SECONDS = 120.0


def _http_get(url: str) -> bytes:
    """GET with backoff on the statuses that mean "later", not "no".

    A build is dozens to hundreds of pages, and one 429 partway through used to
    abort the whole corpus. Retrying the retryable statuses is not politeness
    theatre: without it, corpus size is capped by whatever burst the API happens
    to tolerate that minute, which would silently become a policy number.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            wait = float(exc.headers.get("Retry-After") or delay)
            if exc.code not in RETRY_STATUS or attempt == MAX_RETRIES - 1:
                raise
            if wait > MAX_BACKOFF_SECONDS:
                raise CorpusError(
                    BlockedRecord(
                        missing_capability="openalex_quota",
                        attempted=f"GET {url}",
                        unlock_condition=f"the endpoint asked for {wait:.0f}s before the next request ("
                        f"{wait / 3600:.1f}h), which is a quota reset rather than a burst; re-run the build after it, or point --replay at a recorded build",
                    )
                ) from exc
            time.sleep(wait)
            delay = min(delay * 2, MAX_BACKOFF_SECONDS)
    raise RuntimeError("unreachable: the retry loop returns or raises")


def _short(openalex_id: str) -> str:
    return "openalex:" + str(openalex_id).rstrip("/").rsplit("/", 1)[-1]


def _record(work: dict[str, Any]) -> dict[str, Any] | None:
    year = work.get("publication_year")
    title = (work.get("title") or "").strip()
    if year is None or not title:
        return None
    return {
        "id": _short(work["id"]),
        "title": title,
        "year": int(year),
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "references": [_short(item) for item in work.get("referenced_works") or []],
    }


def verify_corpus(dest: Path | str) -> dict[str, Any]:
    """Recompute the artifact digests and compare them to the manifest.

    A result cites a corpus by id. If the files under that id can drift without
    anything noticing, the citation means nothing, so every reader of a corpus
    should call this before measuring on it.
    """
    dest = Path(dest)
    manifest_path = dest / "manifest.json"
    if not manifest_path.is_file():
        raise CorpusError(
            BlockedRecord(
                missing_capability="corpus_manifest",
                attempted=f"verify the corpus at {dest}",
                unlock_condition="build the corpus with build_corpus, which writes manifest.json",
            )
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recomputed = {
        name: content_digest(json.loads((dest / f"{stem}.json").read_text(encoding="utf-8")))
        for name, stem in (
            ("G_full", "G_full"),
            ("G_le_T", "G_le_T"),
            ("heldout", "heldout_post_T"),
        )
    }
    drifted = sorted(
        name
        for name, digest in recomputed.items()
        if manifest.get("digests", {}).get(name) != digest
    )
    if drifted:
        raise CorpusError(
            BlockedRecord(
                missing_capability="corpus_integrity",
                attempted=f"verify {manifest.get('corpus_uid', dest.name)}",
                unlock_condition=f"{', '.join(drifted)} no longer match the manifest; rebuild under a new corpus id rather than editing a corpus results already cite",
            )
        )
    return manifest


def build_corpus(
    spec: CorpusSpec,
    dest: Path | str,
    *,
    transport: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Fetch, split at T, write the three artifacts, then load them back."""
    dest = Path(dest)
    fetcher = Fetcher(dest / "raw", transport=transport)
    query = urllib.parse.quote(spec.topic)

    root_page = fetcher.get(
        f"{OPENALEX}?search={query}"
        f"&filter=publication_year:1900-{spec.T}"
        f"&per-page={spec.roots}&select={SELECT}"
    )
    roots = [row for row in (_record(w) for w in root_page.get("results") or []) if row]
    if len(roots) < 3:
        raise CorpusError(
            BlockedRecord(
                missing_capability="openalex_roots",
                attempted=f"find at least 3 pre-{spec.T} works for {spec.topic!r}",
                unlock_condition=f"a topic with at least 3 works published on or before {spec.T}; OpenAlex returned "
                f"{len(roots)}",
            )
        )

    collected = {row["id"]: row for row in roots}
    for root in roots:
        page = fetcher.get(
            f"{OPENALEX}?filter=cites:{root['id'].split(':')[1]}"
            f"&per-page={spec.per_root}&select={SELECT}"
        )
        rows = [row for row in (_record(w) for w in page.get("results") or []) if row]

        for row in sorted(rows, key=lambda item: item["id"]):
            if len(collected) >= spec.max_nodes:
                break
            collected.setdefault(row["id"], row)

    edges = sorted(
        {
            (reference, node_id)
            for node_id, row in collected.items()
            for reference in row["references"]
            if reference in collected and reference != node_id
        }
    )
    if not edges:
        raise CorpusError(
            BlockedRecord(
                missing_capability="openalex_reference_lists",
                attempted="reverse referenced_works into forward citation edges",
                unlock_condition="the fetched works expose referenced_works that land inside the collected node set; none did, so the graph has no edges",
            )
        )

    post_t = {
        node_id for node_id, row in collected.items() if row["year"] > spec.T
    }
    cutoff = _quantile(
        sorted(collected[node_id]["cited_by_count"] for node_id in post_t),
        spec.confirmed_quantile,
    )
    floor = max(cutoff, spec.confirmed_min_citations)
    confirmed = sorted(
        node_id
        for node_id in post_t
        if collected[node_id]["cited_by_count"] >= floor
    )
    if not confirmed:
        raise CorpusError(
            BlockedRecord(
                missing_capability="openalex_confirmed_successors",
                attempted=f"find post-{spec.T} works cited at least "
                f"{spec.confirmed_min_citations} times",
                unlock_condition="a topic whose post-T successors were themselves taken up; without one the value anchor has nothing to score against",
            )
        )

    pre_t_edges = [
        (src, dst) for src, dst in edges if src not in post_t and dst not in post_t
    ]
    out_degree = {}
    in_degree = {}
    for src, dst in pre_t_edges:
        out_degree[src] = out_degree.get(src, 0) + 1
        in_degree[dst] = in_degree.get(dst, 0) + 1

    dead_ends = [
        node_id
        for node_id in collected
        if node_id not in post_t
        and out_degree.get(node_id, 0) == 0
        and in_degree.get(node_id, 0) >= spec.counterexample_min_in_degree
        and collected[node_id]["year"] <= spec.T - spec.counterexample_min_age
    ]
    in_cutoff = _quantile(
        sorted(in_degree.get(node_id, 0) for node_id in dead_ends),
        spec.counterexample_quantile,
    )
    counterexamples = sorted(
        node_id for node_id in dead_ends if in_degree.get(node_id, 0) >= in_cutoff
    )

    full_nodes = [
        {
            "id": row["id"],
            "title": row["title"],
            "year": row["year"],
            "cited_by_count": row["cited_by_count"],
        }
        for row in sorted(collected.values(), key=lambda item: item["id"])
    ]

    content = content_digest(
        {
            "spec": spec.to_dict(),
            "nodes": full_nodes,
            "edges": [list(edge) for edge in edges],
            "confirmed": confirmed,
            "counterexample_nodes": counterexamples,
        }
    )[:12]
    corpus_uid = f"{spec.corpus_id}-{content}"
    full_payload = {
        "snapshot_id": f"{corpus_uid}-full",
        "k": PINNED_K,
        "nodes": full_nodes,
        "edges": [list(edge) for edge in edges],
        "counterexample_nodes": counterexamples,
    }
    walker_payload = derive_g_le_t(
        {
            **full_payload,
            "snapshot_id": corpus_uid,
            "nodes": [
                {key: value for key, value in node.items() if key != "cited_by_count"}
                for node in full_nodes
            ],
        },
        spec.T,
    )
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
                    unlock_condition=f"{dest.name} already holds "
                    f"{previous.get('corpus_uid', 'another corpus')}; OpenAlex has moved since, so write the new fetch under a new directory instead of redefining the one results cite",
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

    walker = load_walker_graph(paths["G_le_T"])
    evaluator = load_graph(paths["G_full"])
    heldout = load_heldout(paths["heldout"])
    seed_nodes = [row["id"] for row in roots if row["id"] in walker.nodes]
    if len(seed_nodes) < 3:
        raise CorpusError(
            BlockedRecord(
                missing_capability="walker_visible_seeds",
                attempted="keep at least 3 roots inside G_<=T",
                unlock_condition="the value anchor needs at least 3 independent seeds per snapshot; this corpus kept "
                f"{len(seed_nodes)}",
            )
        )

    manifest = {
        "spec": spec.to_dict(),
        "corpus_uid": corpus_uid,
        "content_digest": content,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pages": fetcher.pages,
        "seed_nodes": seed_nodes,
        "counts": {
            "nodes_full": len(evaluator.nodes),
            "nodes_walker": len(walker.nodes),
            "edges_full": len(evaluator.edges),
            "edges_walker": len(walker.edges),
            "post_T_nodes": len(post_t),
            "confirmed": len(heldout.confirmed),
            "confirmed_citation_floor": floor,
            "counterexample_nodes": len(counterexamples),
        },
        "digests": {
            name: content_digest(json.loads(path.read_text(encoding="utf-8")))
            for name, path in paths.items()
        },
        "known_limitation": "which pre-T works enter the corpus depends on OpenAlex relevance for the topic string, so the corpus is a sample of the literature, not the literature; both arms are scored on the same sample",
    }
    _write(dest / "manifest.json", manifest)
    return manifest


def _quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[index]


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
