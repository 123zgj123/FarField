"""Live literature retrieve: today's indexes, frozen per mission.

The concept corpus pins what "already combined" means up to its harvest
end. Production still has to see papers published after that pin.
`ArxivFeed` is the dated arXiv Export API. `CompositeFeed` asks arXiv,
Semantic Scholar, and OpenAlex in parallel so a 429 on one index does
not empty the field.

Retrieve is not admit. Rows from this module are candidates;
`verifypapers` (arXiv id_list / CrossRef DOI / Scholar title) decides
what may enter the Research Wiki. `verify_pending` (429 / 5xx) is
reported, never cited.
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
    references: tuple[str, ...] = ()

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
            references=tuple(
                str(item).strip()
                for item in (row.get("references") or ())
                if str(item).strip()
            ),
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
        if self.references:
            payload["references"] = list(self.references)
        return payload


def as_work(item: Any) -> FreshWork | None:
    """Accept FreshWork or a retrieved dict. Other shapes are dropped."""
    if isinstance(item, FreshWork):
        return item
    if isinstance(item, dict) and (
        item.get("title") or item.get("cite_id") or item.get("arxiv_id")
    ):
        return FreshWork.from_dict(item)
    return None


def is_signature_typeerror(exc: BaseException) -> bool:
    """True only for a call-signature TypeError, not 'unhashable type: dict'."""
    if not isinstance(exc, TypeError):
        return False
    msg = str(exc).lower()
    return (
        "unexpected keyword" in msg
        or "required positional argument" in msg
        or "takes from" in msg
        or "got an unexpected" in msg
        or ("positional argument" in msg and "given" in msg)
    )


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
    except urllib.error.HTTPError as error:
        reason = f"HTTP {error.code}"
        if error.code == 429 or error.code >= 500:
            reason += " (retryable)"
        return FeedBlocked(attempted=query, reason=reason)
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


def object_survey_phrases(topic: str) -> tuple[str, ...]:
    """Raw topic phrases for retrieval, not stems used by semantic matching.

    Stemming corrupts names (Iris -> iri, species -> specie); external indexes
    must see the researcher's actual terms. Keep scientific hyphens and short
    names such as RNA. Do not bridge prepositions into invented noun phrases.
    """
    words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", topic, flags=re.UNICODE)
    stop = {"a", "an", "the", "and", "or", "of", "on", "in", "for", "to",
            "with", "under", "over", "from", "by", "into", "using", "versus",
            "vs", "is", "are", "research", "study", "investigate"}
    grams = [f"{left} {right}" for left, right in zip(words, words[1:])
             if left.casefold() not in stop and right.casefold() not in stop]
    choices = grams or [word for word in words if word.casefold() not in stop]
    seen: set[str] = set()
    phrases: list[str] = []
    for phrase in choices:
        if phrase.casefold() not in seen:
            seen.add(phrase.casefold())
            phrases.append(phrase)
    return tuple(phrases[:3])


def survey_queries(
    concept_a: str,
    concept_b: str,
    *,
    topic: str = "",
) -> tuple[str, ...]:
    """arXiv query strings for a pair survey.

    With a topic, the far concept is never queried alone — that is how
    a mechanism noun like 'stochastic reward' pulls knapsack papers.
    """
    a = " ".join(str(concept_a or "").split())
    b = " ".join(str(concept_b or "").split())
    if topic.strip():
        phrases = object_survey_phrases(topic)
        queries: list[str] = []
        if phrases:
            queries.append(" OR ".join(f"all:{_phrase(item)}" for item in phrases[:2]))
            obj = phrases[0]
            redundant = {obj.casefold(), " ".join(topic.split()).casefold()}
            for concept in (a, b):
                if concept and concept.casefold() not in redundant:
                    queries.append(f"all:{_phrase(obj)} AND all:{_phrase(concept)}")
                    redundant.add(concept.casefold())
        return tuple(queries) if queries else (f"all:{_phrase(topic.strip()[:80])}",)
    queries = []
    if a:
        queries.append(" OR ".join(f"all:{_phrase(item)}" for item in (a,)))
    if b and b.casefold() != a.casefold():
        queries.append(" OR ".join(f"all:{_phrase(item)}" for item in (b,)))
    if a and b and b.casefold() != a.casefold():
        queries.append(f"all:{_phrase(a)} AND all:{_phrase(b)}")
    return tuple(queries)


class ArxivFeed:
    """Dated arXiv Export API. Live rows still need verify_papers."""

    verify_literature = True

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
        queries = survey_queries(concept_a, concept_b, topic=topic)
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

    `recent_in_field` is multi-source: an arXiv 429 must not hide a
    Semantic Scholar or OpenAlex hit on the same topic phrases. Pair
    probes and surveys already asked the other indexes; freshness now
    does too. Tests inject `scholar` / `openalex` callables and never
    open a socket.
    """

    verify_literature = True

    def __init__(
        self,
        *,
        arxiv: ArxivFeed | None = None,
        scholar: Any = None,
        openalex: Any = None,
    ) -> None:
        self.arxiv = arxiv or ArxivFeed()
        self._search_scholar = scholar
        self._search_openalex = openalex
        self.last_sources_blocked: list[dict[str, Any]] = []

    def _scholar(self, query: str, max_results: int) -> list[FreshWork] | FeedBlocked:
        from .sources import search_scholar

        fn = self._search_scholar or search_scholar
        try:
            return fn(query, max_results=max_results)
        except TypeError as exc:
            if not is_signature_typeerror(exc):
                raise
            return fn(query)

    def _openalex(self, query: str, max_results: int) -> list[FreshWork] | FeedBlocked:
        from .sources import search_openalex

        fn = self._search_openalex or search_openalex
        try:
            return fn(query, max_results=max_results)
        except TypeError as exc:
            if not is_signature_typeerror(exc):
                raise
            return fn(query)

    def recent_in_field(
        self, concepts: tuple[str, ...], *, max_results: int = 6
    ) -> list[FreshWork] | FeedBlocked:
        from .sources import merge_works, quoted_query

        query = " ".join(str(c) for c in concepts[:3] if str(c).strip())
        phrased = quoted_query(tuple(concepts)[:3])
        jobs = (
            lambda: self.arxiv.recent_in_field(concepts, max_results=max_results),
            lambda: self._scholar(query, max_results=max_results),
            lambda: self._openalex(phrased or query, max_results=max_results),
        )
        fetched = map_parallel(lambda job: job(), jobs, workers=3)
        groups: list[list[FreshWork]] = []
        blocked: list[dict[str, Any]] = []
        last_block: FeedBlocked | None = None
        for result in fetched:
            if isinstance(result, FeedBlocked):
                last_block = result
                blocked.append(result.to_dict())
                continue
            if isinstance(result, list):
                group = [work for item in result if (work := as_work(item)) is not None]
                if group:
                    groups.append(group)
        self.last_sources_blocked = blocked
        merged = merge_works(*groups)
        if not merged:
            return last_block or FeedBlocked(
                attempted=query or "recent_in_field",
                reason="empty feed",
            )
        return merged[:max_results]

    def pair_recently_combined(
        self, concept_a: str, concept_b: str, *, max_results: int = 5
    ) -> dict[str, Any] | FeedBlocked:
        from .sources import merge_works

        def _arxiv():
            return self.arxiv.pair_recently_combined(
                concept_a, concept_b, max_results=max_results
            )

        query = f"{concept_a} {concept_b}"
        jobs = (
            _arxiv,
            lambda: self._scholar(query, max_results=max_results),
            lambda: self._openalex(query, max_results=max_results),
        )
        fetched = map_parallel(lambda job: job(), jobs, workers=3)
        evidence: list[FreshWork] = []
        blocked: list[dict[str, Any]] = []
        arxiv = fetched[0]
        if isinstance(arxiv, FeedBlocked):
            blocked.append(arxiv.to_dict())
        elif isinstance(arxiv, dict):
            evidence.extend(
                work
                for row in arxiv.get("evidence") or []
                if (work := as_work(row) if isinstance(row, dict) else None) is not None
            )
        for result in fetched[1:]:
            if isinstance(result, FeedBlocked):
                blocked.append(result.to_dict())
                continue
            if isinstance(result, list):
                evidence.extend(work for item in result if (work := as_work(item)) is not None)
            elif isinstance(result, dict):
                rows = result.get("evidence") or result.get("works") or []
                evidence.extend(
                    work
                    for row in rows
                    if (work := as_work(row)) is not None
                )
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
        from .sources import merge_works, quoted_query

        groups: list[list[FreshWork]] = []
        last_block: FeedBlocked | None = None
        phrases = object_survey_phrases(topic) if topic.strip() else ()
        if phrases:
            # Overlapping bigrams are alternative retrieval anchors, not two
            # mandatory exact phrases that every paper must contain together.
            object_q = phrases[0]
            jobs = [
                lambda: self.arxiv.survey_around(
                    concept_a, concept_b, per_side=per_side, topic=topic
                ),
                lambda: self._scholar(object_q, max_results=per_side),
                lambda: self._openalex(
                    quoted_query(phrases[:1]) or object_q, max_results=per_side
                ),
            ]
            far_term = " ".join(str(concept_b or "").split())
            if far_term and far_term.casefold() not in {
                object_q.casefold(), " ".join(topic.split()).casefold()
            }:
                jobs.append(
                    lambda: self._scholar(
                        f"{phrases[0]} {far_term}", max_results=per_side
                    )
                )
        else:
            pair_query = f'"{concept_a}" "{concept_b}"'
            jobs = [
                lambda: self.arxiv.survey_around(
                    concept_a, concept_b, per_side=per_side, topic=topic
                ),
                lambda: self._scholar(pair_query, max_results=per_side),
                lambda: self._openalex(f"{concept_a} {concept_b}", max_results=per_side),
            ]
            if claim.strip():
                snippet = claim.strip()[:180]
                jobs.append(lambda: self._scholar(snippet, max_results=per_side))
        fetched = map_parallel(lambda job: job(), jobs, workers=len(jobs))
        blocked: list[dict[str, Any]] = []
        for result in fetched:
            if isinstance(result, FeedBlocked):
                last_block = result
                blocked.append(result.to_dict())
            elif isinstance(result, list):
                group = [work for item in result if (work := as_work(item)) is not None]
                if group:
                    groups.append(group)
        self.last_sources_blocked = blocked
        merged = merge_works(*groups)
        if not merged:
            return last_block or FeedBlocked(
                attempted="survey_around", reason="empty feed"
            )
        return merged
