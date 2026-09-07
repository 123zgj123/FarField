"""Host literature harvest: retrieve, verify, bank.

FarField's probe sandbox stays offline. Knowledge arrives on the host,
is frozen into the mission snapshot, and only verified rows may enter
the neighbourhood wiki or a briefing. This is the same shape as a
SkyPilot GPU run: an external index does the retrieval; FarField admits
the rows.

Pipeline:

    multi-source retrieve (arXiv + Semantic Scholar + OpenAlex)
    → verify_papers (arXiv id_list / CrossRef DOI / Scholar title)
    → Research Wiki (verified only)

`verify_pending` (429, 5xx) is reported, never cited, never counted as
a fabricated reference. Limitation sentences on verified abstracts
become agenda targets, not solved claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .livefeed import FeedBlocked, FreshWork, is_signature_typeerror
from .snapshot import ReplayFeed
from .verifypapers import (
    PENDING,
    UNVERIFIED,
    VERIFIED,
    VerifyReport,
    admitted,
    verify_works,
)


def feed_verifies(feed: Any) -> bool:
    """True when this feed's rows still need an index check.

    Canned test feeds and snapshot replay already hold attested rows.
    A live CompositeFeed does not.
    """
    inner = getattr(feed, "inner", feed)
    if isinstance(inner, ReplayFeed) or isinstance(feed, ReplayFeed):
        return False
    return bool(getattr(inner, "verify_literature", True))


@dataclass
class Harvest:
    works: list[FreshWork] = field(default_factory=list)
    reports: list[VerifyReport] = field(default_factory=list)
    blocked: FeedBlocked | None = None
    sources_blocked: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "works": [work.to_dict() for work in self.works],
            "verification": [row.to_dict() for row in self.reports],
            "admitted": len(self.works),
            "pending": sum(1 for row in self.reports if row.status == PENDING),
            "unverified": sum(1 for row in self.reports if row.status == UNVERIFIED),
        }
        if self.blocked is not None:
            payload["blocked"] = self.blocked.to_dict()
        if self.sources_blocked:
            payload["sources_blocked"] = list(self.sources_blocked)
        return payload


def _trusted(works: list[FreshWork]) -> list[VerifyReport]:
    return [
        VerifyReport(work, VERIFIED, "trusted_feed", "test or replay feed")
        for work in works
    ]


def admit_works(
    works: list[FreshWork] | FeedBlocked | None,
    *,
    verify: bool = True,
    fetcher: Any = None,
) -> Harvest:
    """Filter retrieved rows through verify_papers. Empty is not a block."""
    if isinstance(works, FeedBlocked):
        return Harvest(blocked=works)
    rows: list[FreshWork] = []
    for item in list(works or []):
        if isinstance(item, FreshWork):
            rows.append(item)
        elif isinstance(item, dict):
            rows.append(FreshWork.from_dict(item))
    if not rows:
        return Harvest()
    reports = (
        verify_works(rows, fetcher=fetcher) if verify else _trusted(rows)
    )
    return Harvest(works=admitted(reports), reports=reports)


def _verify_fetcher(feed: Any, fetcher: Any) -> Any:
    if fetcher is not None:
        return fetcher
    inner = getattr(feed, "inner", feed)
    return getattr(inner, "verify_fetcher", None) or getattr(
        feed, "verify_fetcher", None
    )


def _sources_blocked(feed: Any) -> list[dict[str, Any]]:
    inner = getattr(feed, "inner", feed)
    extra = getattr(inner, "last_sources_blocked", None)
    if extra is None:
        extra = getattr(feed, "last_sources_blocked", None)
    return list(extra or [])


def harvest_recent(
    feed: Any,
    concepts: tuple[str, ...],
    *,
    max_results: int = 6,
    fetcher: Any = None,
) -> Harvest:
    """`fresh` stage: multi-source retrieve, then admit verified rows."""
    if feed is None:
        return Harvest(
            blocked=FeedBlocked(attempted="recent_in_field", reason="no feed")
        )
    raw = feed.recent_in_field(concepts, max_results=max_results)
    extra = _sources_blocked(feed)
    if isinstance(raw, dict) and ("works" in raw or "evidence" in raw):
        extra = list(raw.get("sources_blocked") or extra)
        raw = raw.get("works") or raw.get("evidence")
    harvest = admit_works(
        raw,
        verify=feed_verifies(feed),
        fetcher=_verify_fetcher(feed, fetcher),
    )
    harvest.sources_blocked = extra
    return harvest


def harvest_survey(
    feed: Any,
    concept_a: str,
    concept_b: str,
    *,
    per_side: int = 4,
    claim: str = "",
    topic: str = "",
    fetcher: Any = None,
) -> Harvest:
    """Survey around a surviving pair; only verified rows brief the card."""
    if feed is None or not hasattr(feed, "survey_around"):
        return Harvest()
    try:
        raw = feed.survey_around(
            concept_a, concept_b, per_side=per_side, claim=claim, topic=topic
        )
    except TypeError as exc:
        if not is_signature_typeerror(exc):
            raise
        try:
            raw = feed.survey_around(concept_a, concept_b, claim=claim)
        except TypeError as inner:
            if not is_signature_typeerror(inner):
                raise
            raw = feed.survey_around(concept_a, concept_b)
    harvest = admit_works(
        raw,
        verify=feed_verifies(feed),
        fetcher=_verify_fetcher(feed, fetcher),
    )
    harvest.sources_blocked = _sources_blocked(feed)
    return harvest
