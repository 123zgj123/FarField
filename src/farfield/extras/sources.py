"""Multi-source literature search used at mission time.

arXiv, Semantic Scholar, and OpenAlex all run at `fresh` and at survey.
A 429 on one index is `FeedBlocked` for that source, not an empty field.
Admission is a separate tooth (`verifypapers`): CrossRef is a verify
layer, not a third search. Failures stay advisory. Tests pass recorded
JSON and never open a socket.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from .livefeed import FeedBlocked, FreshWork, as_work

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API = "https://api.openalex.org/works"
TIMEOUT_S = 20
# One polite retry after a 429/5xx before the source reports FeedBlocked.
# A rate-limited index is a delay, not an empty field; more than one
# retry would stall the mission on a source that is genuinely down.
RETRY_WAIT_S = 2.0
USER_AGENT = (
    "farfield-sources/1.0 (research tool; mailto:farfield-dev@localhost)"
)
Fetcher = Callable[[str], bytes]


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.read()


def _get(url: str, fetcher: Fetcher | None) -> bytes | FeedBlocked:
    blocked: FeedBlocked | None = None
    for attempt in range(2):
        try:
            return (fetcher or _http_get)(url)
        except urllib.error.HTTPError as error:
            reason = f"HTTP {error.code}"
            retryable = error.code == 429 or error.code >= 500
            if retryable:
                reason += " (retryable)"
            blocked = FeedBlocked(attempted=url, reason=reason)
            if not retryable or attempt:
                return blocked
            time.sleep(RETRY_WAIT_S)
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            return FeedBlocked(attempted=url, reason=str(error))
    return blocked or FeedBlocked(attempted=url, reason="unreachable")


def quoted_query(concepts: tuple[str, ...] | list[str]) -> str:
    """Multiword concepts as quoted phrases for indexes that honor them.

    Joining three phrases into one bag of words is how `code world models
    of executable program state` retrieved drought indexes: every generic
    token matched something highly cited. OpenAlex honors quoted phrases;
    a phrase the author actually wrote is the object, the soup is not.
    """
    parts: list[str] = []
    for concept in concepts:
        text = " ".join(str(concept or "").replace('"', "").split())
        if not text:
            continue
        parts.append(f'"{text}"' if " " in text else text)
    return " ".join(parts)


def search_scholar(
    query: str,
    *,
    max_results: int = 5,
    fetcher: Fetcher | None = None,
) -> list[FreshWork] | FeedBlocked:
    """Title + abstract search. `fetcher` is the test seam."""
    query = query.strip()
    if not query:
        return FeedBlocked(attempted="scholar", reason="empty query")
    url = S2_API + "?" + urllib.parse.urlencode(
        {
            "query": query,
            "limit": str(max_results),
            "fields": "title,abstract,year,venue,url,externalIds,references.externalIds",
        }
    )
    raw = _get(url, fetcher)
    if isinstance(raw, FeedBlocked):
        return raw
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return FeedBlocked(attempted=url, reason=f"malformed scholar json: {error}")
    works: list[FreshWork] = []
    for row in payload.get("data") or []:
        if not isinstance(row, dict):
            continue
        title = " ".join(str(row.get("title") or "").split())
        if not title:
            continue
        ids = row.get("externalIds") if isinstance(row.get("externalIds"), dict) else {}
        arxiv = str(ids.get("ArXiv") or ids.get("ARXIV") or "").strip()
        doi = str(ids.get("DOI") or "").strip()
        paper_id = str(row.get("paperId") or "").strip()
        year = row.get("year")
        works.append(
            FreshWork(
                title=title,
                published=str(year) if year else "",
                arxiv_id=arxiv,
                abstract=" ".join(str(row.get("abstract") or "").split()),
                source="s2",
                work_id=arxiv or (f"doi:{doi}" if doi else f"s2:{paper_id}"),
                url=str(row.get("url") or ""),
                venue=str(row.get("venue") or ""),
                references=_reference_keys(row),
            )
        )
    return works


def _reference_keys(row: dict[str, Any]) -> tuple[str, ...]:
    """Attested citation ids only. Titles are not invented edges."""
    keys: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        text = str(value or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        keys.append(text)

    for item in row.get("references") or []:
        if not isinstance(item, dict):
            continue
        ids = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
        arxiv = str(ids.get("ArXiv") or ids.get("ARXIV") or "").strip()
        doi = str(ids.get("DOI") or "").strip()
        paper_id = str(item.get("paperId") or "").strip()
        if arxiv:
            _add(arxiv)
        elif doi:
            _add(f"doi:{doi}")
        elif paper_id:
            _add(f"s2:{paper_id}")
    for item in row.get("referenced_works") or []:
        _add(str(item).rsplit("/", 1)[-1])
    return tuple(keys[:48])


def _union_fresh(keep: FreshWork, other: FreshWork) -> FreshWork:
    refs = tuple(dict.fromkeys([*keep.references, *other.references]))[:48]
    arxiv = keep.arxiv_id or other.arxiv_id
    if refs == keep.references and arxiv == keep.arxiv_id:
        return keep
    return FreshWork(
        title=keep.title,
        published=keep.published,
        arxiv_id=arxiv,
        abstract=keep.abstract,
        source=keep.source,
        work_id=keep.work_id,
        url=keep.url or other.url,
        venue=keep.venue or other.venue,
        references=refs,
    )


def _openalex_abstract(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    placed: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                placed.append((pos, str(word)))
    placed.sort()
    return " ".join(word for _, word in placed)


def search_openalex(
    query: str,
    *,
    max_results: int = 5,
    fetcher: Fetcher | None = None,
) -> list[FreshWork] | FeedBlocked:
    query = query.strip()
    if not query:
        return FeedBlocked(attempted="openalex", reason="empty query")
    url = OPENALEX_API + "?" + urllib.parse.urlencode(
        {
            "search": query,
            "per_page": str(max_results),
            "select": "id,display_name,publication_year,doi,abstract_inverted_index,primary_location,referenced_works,ids",
        }
    )
    raw = _get(url, fetcher)
    if isinstance(raw, FeedBlocked):
        return raw
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return FeedBlocked(attempted=url, reason=f"malformed openalex json: {error}")
    works: list[FreshWork] = []
    for row in payload.get("results") or []:
        if not isinstance(row, dict):
            continue
        title = " ".join(str(row.get("display_name") or "").split())
        if not title:
            continue
        oa_id = str(row.get("id") or "").rsplit("/", 1)[-1]
        doi = str(row.get("doi") or "").replace("https://doi.org/", "").strip()
        loc = row.get("primary_location") if isinstance(row.get("primary_location"), dict) else {}
        source = loc.get("source") if isinstance(loc.get("source"), dict) else {}
        year = row.get("publication_year")
        works.append(
            FreshWork(
                title=title,
                published=str(year) if year else "",
                arxiv_id="",
                abstract=_openalex_abstract(row.get("abstract_inverted_index")),
                source="openalex",
                work_id=f"doi:{doi}" if doi else f"openalex:{oa_id}",
                url=str(row.get("doi") or row.get("id") or ""),
                venue=str(source.get("display_name") or ""),
                references=_reference_keys(row),
            )
        )
    return works


def merge_works(*groups: list[FreshWork]) -> list[FreshWork]:
    """Dedup by cite id, then by normalized title. Keep the richer abstract."""
    by_key: dict[str, FreshWork] = {}
    title_to_key: dict[str, str] = {}
    for group in groups:
        for raw in group or []:
            work = as_work(raw)
            if work is None:
                continue
            title_key = " ".join(work.title.lower().split())
            key = title_to_key.get(title_key) or work.cite_id() or title_key
            current = by_key.get(key)
            if current is None:
                by_key[key] = work
            elif len(work.abstract) > len(current.abstract):
                by_key[key] = _union_fresh(work, current)
            else:
                by_key[key] = _union_fresh(current, work)
            title_to_key[title_key] = key
    return sorted(
        by_key.values(),
        key=lambda w: (w.published, w.title),
        reverse=True,
    )
