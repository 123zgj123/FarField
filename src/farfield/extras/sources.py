"""Multi-source literature search used at mission time.

arXiv is still the freshness probe: it is dated, public, and already in
the feed. Semantic Scholar and OpenAlex add the rest of the published
record — venues the concept harvest never saw, and wording that does not
match our noun-phrase labels. Failures stay advisory (`FeedBlocked`); a
dead index does not invent an empty field.

Tests pass recorded JSON and never open a socket.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from .livefeed import FeedBlocked, FreshWork

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API = "https://api.openalex.org/works"
TIMEOUT_S = 20
USER_AGENT = (
    "farfield-sources/1.0 (research tool; mailto:farfield-dev@localhost)"
)
Fetcher = Callable[[str], bytes]


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.read()


def _get(url: str, fetcher: Fetcher | None) -> bytes | FeedBlocked:
    try:
        return (fetcher or _http_get)(url)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        return FeedBlocked(attempted=url, reason=str(error))


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
            "fields": "title,abstract,year,venue,url,externalIds",
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
            )
        )
    return works


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
            "select": "id,display_name,publication_year,doi,abstract_inverted_index,primary_location",
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
            )
        )
    return works


def merge_works(*groups: list[FreshWork]) -> list[FreshWork]:
    """Dedup by cite id, then by normalized title. Keep the richer abstract."""
    by_key: dict[str, FreshWork] = {}
    title_to_key: dict[str, str] = {}
    for group in groups:
        for work in group:
            title_key = " ".join(work.title.lower().split())
            key = title_to_key.get(title_key) or work.cite_id() or title_key
            current = by_key.get(key)
            if current is None or len(work.abstract) > len(current.abstract):
                by_key[key] = work
            title_to_key[title_key] = key
    return sorted(
        by_key.values(),
        key=lambda w: (w.published, w.title),
        reverse=True,
    )
