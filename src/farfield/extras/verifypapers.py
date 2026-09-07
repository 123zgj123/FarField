"""Three-layer literature verification: arXiv, CrossRef, Semantic Scholar.

A retrieved row is not yet a paper FarField may cite. arXiv 429, a
hallucinated id, and a real SODA paper must not look the same. This
module is the tooth: host-side, no LLM, the same kind of admission as a
fair two-arm.

Layers, in order:

1. arXiv `id_list` when the row names an arXiv id.
2. CrossRef `/works/{doi}` when the row names a DOI.
3. Semantic Scholar title search; a short token overlap (0.6) on the
   best hit counts as the same work.

Statuses:

- `verified` — an index opened the id or matched the title.
- `unverified` — the index answered and did not have this work (do not
  cite; this is how fabricated references die).
- `verify_pending` — 429 / 5xx / timeout. Not a hallucination, not a
  citation.
- `error` — malformed payload. Same as unverified for admission.

Tests inject a fetcher and never open a socket.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .livefeed import FeedBlocked, FreshWork
from .prior import content_tokens

ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works/"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
TIMEOUT_S = 20
USER_AGENT = (
    "farfield-verifypapers/1.0 (research tool; mailto:farfield-dev@localhost)"
)
TITLE_OVERLAP = 0.6
Fetcher = Callable[[str], bytes]

VERIFIED = "verified"
UNVERIFIED = "unverified"
PENDING = "verify_pending"
ERROR = "error"

_ARXIV_VERSION = re.compile(r"v\d+$", re.I)


def bare_arxiv(arxiv_id: str) -> str:
    return _ARXIV_VERSION.sub("", str(arxiv_id or "").strip())


def doi_of(work: FreshWork) -> str:
    wid = str(work.work_id or "")
    if wid.lower().startswith("doi:"):
        return wid.split(":", 1)[-1].strip()
    url = str(work.url or "")
    if "doi.org/" in url:
        return url.split("doi.org/", 1)[-1].strip()
    return ""


def title_overlap(left: str, right: str) -> float:
    a = content_tokens(left)
    b = content_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.read()


def _get(url: str, fetcher: Fetcher | None) -> bytes | FeedBlocked:
    try:
        return (fetcher or _http_get)(url)
    except urllib.error.HTTPError as error:
        retryable = error.code == 429 or error.code >= 500
        reason = f"HTTP {error.code}"
        if retryable:
            reason += " (retryable)"
        return FeedBlocked(attempted=url, reason=reason)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        return FeedBlocked(attempted=url, reason=str(error))


def _pending(block: FeedBlocked) -> bool:
    reason = str(block.reason or "").lower()
    return (
        "429" in reason
        or "retryable" in reason
        or "timed out" in reason
        or "timeout" in reason
        or "http 5" in reason
    )


@dataclass(frozen=True)
class VerifyReport:
    work: FreshWork
    status: str
    layer: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cite_id": self.work.cite_id(),
            "title": self.work.title,
            "status": self.status,
            "layer": self.layer,
            "detail": self.detail,
        }


def _arxiv_layer(work: FreshWork, fetcher: Fetcher | None) -> VerifyReport | None:
    arxiv_id = bare_arxiv(work.arxiv_id)
    if not arxiv_id:
        return None
    url = ARXIV_API + "?" + urllib.parse.urlencode({"id_list": arxiv_id})
    raw = _get(url, fetcher)
    if isinstance(raw, FeedBlocked):
        status = PENDING if _pending(raw) else UNVERIFIED
        return VerifyReport(work, status, "arxiv", raw.reason)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return VerifyReport(work, ERROR, "arxiv", f"malformed atom: {exc}")
    titles = [
        " ".join((entry.findtext("a:title", "", ATOM) or "").split())
        for entry in root.findall("a:entry", ATOM)
    ]
    titles = [title for title in titles if title]
    if not titles:
        return VerifyReport(
            work, UNVERIFIED, "arxiv", f"id_list {arxiv_id} returned no entry"
        )
    if work.title and title_overlap(work.title, titles[0]) < TITLE_OVERLAP:
        return VerifyReport(
            work,
            UNVERIFIED,
            "arxiv",
            "arxiv id opened a different title",
        )
    return VerifyReport(work, VERIFIED, "arxiv", arxiv_id)


def _crossref_layer(work: FreshWork, fetcher: Fetcher | None) -> VerifyReport | None:
    doi = doi_of(work)
    if not doi:
        return None
    url = CROSSREF_API + urllib.parse.quote(doi)
    raw = _get(url, fetcher)
    if isinstance(raw, FeedBlocked):
        status = PENDING if _pending(raw) else UNVERIFIED
        return VerifyReport(work, status, "crossref", raw.reason)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return VerifyReport(work, ERROR, "crossref", f"malformed json: {exc}")
    message = payload.get("message") if isinstance(payload, dict) else None
    titles = message.get("title") if isinstance(message, dict) else None
    title = " ".join(str(titles[0]).split()) if isinstance(titles, list) and titles else ""
    if not title:
        return VerifyReport(work, UNVERIFIED, "crossref", f"doi {doi} has no title")
    if work.title and title_overlap(work.title, title) < TITLE_OVERLAP:
        return VerifyReport(
            work, UNVERIFIED, "crossref", "doi opened a different title"
        )
    return VerifyReport(work, VERIFIED, "crossref", doi)


def _scholar_layer(work: FreshWork, fetcher: Fetcher | None) -> VerifyReport:
    query = (work.title or work.cite_id() or "").strip()
    if not query:
        return VerifyReport(work, UNVERIFIED, "scholar", "empty title")
    url = S2_API + "?" + urllib.parse.urlencode(
        {
            "query": query,
            "limit": "5",
            "fields": "title,externalIds,year,url",
        }
    )
    raw = _get(url, fetcher)
    if isinstance(raw, FeedBlocked):
        status = PENDING if _pending(raw) else UNVERIFIED
        return VerifyReport(work, status, "scholar", raw.reason)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return VerifyReport(work, ERROR, "scholar", f"malformed json: {exc}")
    best = 0.0
    hit = ""
    for row in payload.get("data") or []:
        if not isinstance(row, dict):
            continue
        title = " ".join(str(row.get("title") or "").split())
        score = title_overlap(work.title, title)
        if score > best:
            best = score
            hit = title
    if best >= TITLE_OVERLAP:
        return VerifyReport(
            work, VERIFIED, "scholar", f"title overlap {best:.2f} with {hit[:80]}"
        )
    return VerifyReport(
        work,
        UNVERIFIED,
        "scholar",
        f"best title overlap {best:.2f} below {TITLE_OVERLAP}",
    )


def verify_paper(work: FreshWork, *, fetcher: Fetcher | None = None) -> VerifyReport:
    """Admit one retrieved row, or name why it cannot be cited."""
    for layer in (_arxiv_layer, _crossref_layer):
        report = layer(work, fetcher)
        if report is None:
            continue
        if report.status == VERIFIED:
            return report
        if report.status == PENDING:
            return report
        # unverified/error on this id: still try the next named id, then scholar.
    return _scholar_layer(work, fetcher)


def verify_works(
    works: Iterable[FreshWork], *, fetcher: Fetcher | None = None
) -> list[VerifyReport]:
    return [verify_paper(work, fetcher=fetcher) for work in works]


def admitted(reports: Iterable[VerifyReport]) -> list[FreshWork]:
    """Only verified rows may enter the wiki, a briefing, or a diagnosis."""
    return [row.work for row in reports if row.status == VERIFIED]
