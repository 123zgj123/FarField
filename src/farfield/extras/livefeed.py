"""Live arXiv feed: the freshest knowledge, fetched at mission time.

The concept corpus pins what "already combined" means up to its harvest end
— that pin is what makes the oracle's verdicts replayable and its quantiles
meaningful. But a production mission runs *today*, and the field kept
publishing after the harvest. This module fills exactly that gap, live and
narrow, with two queries against the public arXiv API:

- `recent_in_field`: the newest submissions near the mission's anchor
  concepts, fed to the generator as grounding context (titles only; context,
  never evidence).
- `pair_recently_combined`: for a card that survived every judge, one last
  freshness probe — has any submission newer than the corpus already used
  both concepts? The answer downgrades nothing by itself; it is reported
  next to the verdict so the caller knows whether the combination is still
  open *as of right now* or was taken in the weeks the corpus cannot see.

Network is a privilege, not an assumption: every failure (timeout, HTTP
error, malformed feed) returns a `FeedBlocked` record naming what was
attempted, and the mission reports it instead of pretending the check ran.
Nothing here writes to the corpus, and nothing here can kill a card — a
live web query is not a replayable fact, so it stays advisory.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from .parallel import map_parallel

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
TIMEOUT_S = 20
USER_AGENT = "farfield-livefeed/1.0 (research tool; single query per mission stage)"


@dataclass(frozen=True)
class FeedBlocked:
    attempted: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"attempted": self.attempted, "reason": self.reason}


@dataclass(frozen=True)
class FreshWork:
    title: str
    published: str  # YYYY-MM-DD, or a year when that is all a source gives
    arxiv_id: str = ""
    abstract: str = ""
    source: str = "arxiv"
    work_id: str = ""
    url: str = ""
    venue: str = ""

    def cite_id(self) -> str:
        """The only id a briefing may copy into read_first."""
        return self.arxiv_id or self.work_id

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "FreshWork":
        return cls(
            title=str(row.get("title") or ""),
            published=str(row.get("published") or ""),
            arxiv_id=str(row.get("arxiv_id") or ""),
            abstract=str(row.get("abstract") or ""),
            source=str(row.get("source") or "arxiv"),
            work_id=str(row.get("work_id") or ""),
            url=str(row.get("url") or ""),
            venue=str(row.get("venue") or ""),
        )

    def href(self) -> str:
        if self.arxiv_id:
            bare = re.sub(r"v\d+$", "", self.arxiv_id)
            return f"https://arxiv.org/abs/{bare}"
        return self.url

    def line(self) -> str:
        return f"{self.title} ({self.published})"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "title": self.title,
            "published": self.published,
            "arxiv_id": self.arxiv_id,
            "source": self.source,
            "work_id": self.cite_id(),
        }
        if self.abstract:
            payload["abstract"] = self.abstract
        if self.url:
            payload["url"] = self.url
        if self.venue:
            payload["venue"] = self.venue
        return payload


def _fetch(query: str, max_results: int) -> list[FreshWork] | FeedBlocked:
    url = ARXIV_API + "?" + urllib.parse.urlencode(
        {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(max_results),
        }
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body = response.read()
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        return FeedBlocked(attempted=query, reason=str(error))
    try:
        root = ET.fromstring(body)
    except ET.ParseError as error:
        return FeedBlocked(attempted=query, reason=f"malformed feed: {error}")
    works: list[FreshWork] = []
    for entry in root.findall("a:entry", ATOM):
        title = " ".join((entry.findtext("a:title", "", ATOM) or "").split())
        published = (entry.findtext("a:published", "", ATOM) or "")[:10]
        arxiv_id = (entry.findtext("a:id", "", ATOM) or "").rsplit("/", 1)[-1]
        abstract = " ".join((entry.findtext("a:summary", "", ATOM) or "").split())
        if title:
            works.append(
                FreshWork(
                    title=title,
                    published=published,
                    arxiv_id=arxiv_id,
                    abstract=abstract,
                )
            )
    return works


def _phrase(concept: str) -> str:
    return '"' + concept.replace('"', "") + '"'


class ArxivFeed:
    """The live queries a mission is allowed to make, and nothing else."""

    def recent_in_field(
        self, concepts: tuple[str, ...], *, max_results: int = 6
    ) -> list[FreshWork] | FeedBlocked:
        """Newest submissions mentioning any of the anchor concepts."""
        terms = " OR ".join(f"all:{_phrase(c)}" for c in concepts[:3])
        return _fetch(terms, max_results)

    def pair_recently_combined(
        self, concept_a: str, concept_b: str, *, max_results: int = 5
    ) -> dict[str, Any] | FeedBlocked:
        """Has any submission used both concepts, as of right now?

        Advisory only: a hit means the newest literature already carries
        both phrases somewhere in title/abstract — worth reading before
        investing, not a kill. A miss means the live index has no such
        paper either, so the combination is open as far as today's arXiv
        can tell.
        """
        query = f"all:{_phrase(concept_a)} AND all:{_phrase(concept_b)}"
        works = _fetch(query, max_results)
        if isinstance(works, FeedBlocked):
            return works
        return {
            "combined": bool(works),
            "evidence": [work.to_dict() for work in works],
        }

    def survey_around(
        self,
        concept_a: str,
        concept_b: str,
        *,
        per_side: int = 4,
        claim: str = "",
        topic: str = "",
    ) -> list[FreshWork] | FeedBlocked:
        """Papers on each side of the pair, and on both, for a real briefing.

        Titles alone are not a survey. This pull is what a surviving
        hypothesis is written against before it is offered as something
        a researcher can continue. When `topic` is set, one extra query
        keeps the survey inside the researcher's object domain so a
        drifted pair cannot be briefed only against the distant field.
        """
        queries = (
            " OR ".join(f"all:{_phrase(c)}" for c in (concept_a,)),
            " OR ".join(f"all:{_phrase(c)}" for c in (concept_b,)),
            f"all:{_phrase(concept_a)} AND all:{_phrase(concept_b)}",
        )
        if topic.strip():
            queries = queries + (
                f"all:{_phrase(topic.strip()[:80])} AND all:{_phrase(concept_a)}",
            )
        merged: dict[str, FreshWork] = {}
        last_block: FeedBlocked | None = None
        results = map_parallel(lambda q: _fetch(q, per_side), queries, workers=len(queries))
        for result in results:
            if isinstance(result, FeedBlocked):
                last_block = result
                continue
            for work in result:
                merged.setdefault(work.cite_id() or work.title, work)
        if not merged:
            return last_block or FeedBlocked(
                attempted="survey_around", reason="empty feed"
            )
        return sorted(merged.values(), key=lambda w: w.published, reverse=True)


class CompositeFeed:
    """arXiv plus Semantic Scholar plus OpenAlex, same methods as ArxivFeed.

    Freshness (`recent_in_field`) stays on arXiv: it is dated. Pair probes
    and surveys ask the other indexes too, so a combination that only
    exists in a journal is not reported as open.
    """

    def __init__(self, *, arxiv: ArxivFeed | None = None) -> None:
        self.arxiv = arxiv or ArxivFeed()

    def recent_in_field(
        self, concepts: tuple[str, ...], *, max_results: int = 6
    ) -> list[FreshWork] | FeedBlocked:
        return self.arxiv.recent_in_field(concepts, max_results=max_results)

    def pair_recently_combined(
        self, concept_a: str, concept_b: str, *, max_results: int = 5
    ) -> dict[str, Any] | FeedBlocked:
        from .sources import merge_works, search_openalex, search_scholar

        def _arxiv():
            return self.arxiv.pair_recently_combined(
                concept_a, concept_b, max_results=max_results
            )

        query = f"{concept_a} {concept_b}"
        jobs = (_arxiv, lambda: search_scholar(query, max_results=max_results),
                lambda: search_openalex(query, max_results=max_results))
        fetched = map_parallel(lambda job: job(), jobs, workers=3)
        evidence: list[FreshWork] = []
        blocked: list[dict[str, Any]] = []
        arxiv = fetched[0]
        if isinstance(arxiv, FeedBlocked):
            blocked.append(arxiv.to_dict())
        else:
            evidence.extend(
                FreshWork(
                    title=str(row.get("title") or ""),
                    published=str(row.get("published") or ""),
                    arxiv_id=str(row.get("arxiv_id") or ""),
                    abstract=str(row.get("abstract") or ""),
                    source="arxiv",
                )
                for row in arxiv.get("evidence") or []
                if isinstance(row, dict)
            )
        for result in fetched[1:]:
            if isinstance(result, FeedBlocked):
                blocked.append(result.to_dict())
            else:
                evidence.extend(result)
        works = merge_works(evidence)
        if not works and blocked and not evidence:
            return FeedBlocked(
                attempted=query,
                reason="; ".join(item.get("reason") or "" for item in blocked),
            )
        return {
            "combined": bool(works),
            "evidence": [work.to_dict() for work in works[:max_results]],
            "sources_blocked": blocked,
        }

    def survey_around(
        self,
        concept_a: str,
        concept_b: str,
        *,
        per_side: int = 4,
        claim: str = "",
        topic: str = "",
    ) -> list[FreshWork] | FeedBlocked:
        from .sources import merge_works, search_openalex, search_scholar

        groups: list[list[FreshWork]] = []
        last_block: FeedBlocked | None = None
        pair_query = f'"{concept_a}" "{concept_b}"'
        jobs = [
            lambda: self.arxiv.survey_around(
                concept_a, concept_b, per_side=per_side, topic=topic
            ),
            lambda: search_scholar(pair_query, max_results=per_side),
            lambda: search_openalex(f"{concept_a} {concept_b}", max_results=per_side),
        ]
        if topic.strip():
            jobs.append(
                lambda: search_scholar(
                    f"{topic.strip()[:80]} {concept_a}", max_results=per_side
                )
            )
        if claim.strip():
            snippet = claim.strip()[:180]
            jobs.append(lambda: search_scholar(snippet, max_results=per_side))
        fetched = map_parallel(lambda job: job(), jobs, workers=len(jobs))
        for result in fetched:
            if isinstance(result, FeedBlocked):
                last_block = result
            else:
                groups.append(result)
        merged = merge_works(*groups)
        if not merged:
            return last_block or FeedBlocked(
                attempted="survey_around", reason="empty feed"
            )
        return merged
