"""Compile retrieved papers into the neighbourhood research state.

Polaris keeps a research wiki: one paper, one page, shared across the
lab. FarField's survey used to die with the mission, so the next
`generate_card` re-searched and prior-check could not see a restatement
found last time. This module compiles attested `FreshWork` rows into
the digested state. No LLM. Citations in a packet remain this-mission
only; the wiki is memory for generation and prior support, not a license
to cite unread papers.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from .livefeed import FreshWork
from .prior import APPLY_PAD, content_tokens

WIKI_CAP = 40
PROMPT_CAP = 8
ABSTRACT_CAP = 800
GAP_CAP = 8
GAP_OVERLAP = 2
_CITE_PREFIX = re.compile(r"^\[.*?\]\s*")
_GAP_PAD = frozenset(
    {
        "however",
        "limitation",
        "future",
        "remain",
        "unclear",
        "unsolved",
        "cannot",
        "address",
        "consider",
        "leave",
        "without",
        "open",
        "problem",
    }
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_LIMIT = (
    "however",
    "limitation",
    "we do not",
    "we cannot",
    "future work",
    "open problem",
    "remains unclear",
    "not address",
    "do not consider",
    "left as",
    "unsolved",
    "fails to",
    "without evaluating",
    "we leave",
)


def limitation_sentences(text: str, *, cap: int = 3) -> list[str]:
    """Deterministic gaps: sentences that name a limitation, not an LLM summary."""
    parts = [piece.strip() for piece in _SENTENCE.split(text or "") if piece.strip()]
    hits: list[str] = []
    seen: set[str] = set()
    for sentence in parts:
        lower = sentence.lower()
        if not any(marker in lower for marker in _LIMIT):
            continue
        key = " ".join(sentence.split())
        if key in seen:
            continue
        seen.add(key)
        hits.append(key[:240])
        if len(hits) >= cap:
            break
    return hits


def compile_entry(
    work: FreshWork,
    *,
    pair: tuple[str, str] | list[str] | None = None,
    seen_at: str = "",
) -> dict[str, Any]:
    cite = str(work.cite_id() or "").strip()
    title = str(work.title or "").strip()
    abstract = str(work.abstract or "")[:ABSTRACT_CAP]
    return {
        "cite_id": cite,
        "title": title,
        "published": str(work.published or "")[:10],
        "source": str(work.source or "arxiv"),
        "abstract": abstract,
        "url": str(work.href() or ""),
        "venue": str(work.venue or ""),
        "first_seen": str(seen_at or "")[:10],
        "last_seen": str(seen_at or "")[:10],
        "pairs": [list(pair)[:2]] if pair and len(list(pair)) >= 2 else [],
        "limitations": limitation_sentences(abstract),
    }


def merge_entry(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Same cite_id: keep identity, extend pairs, prefer the longer abstract."""
    pairs: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for row in list(old.get("pairs") or []) + list(new.get("pairs") or []):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        key = (str(row[0]), str(row[1]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append([key[0], key[1]])
        if len(pairs) >= 6:
            break
    old_abs = str(old.get("abstract") or "")
    new_abs = str(new.get("abstract") or "")
    first = str(old.get("first_seen") or "") or str(new.get("first_seen") or "")
    last = str(new.get("last_seen") or "") or str(old.get("last_seen") or "")
    return {
        "cite_id": str(old.get("cite_id") or new.get("cite_id") or ""),
        "title": str(new.get("title") or old.get("title") or ""),
        "published": str(old.get("published") or new.get("published") or "")[:10],
        "source": str(old.get("source") or new.get("source") or "arxiv"),
        "abstract": (new_abs if len(new_abs) >= len(old_abs) else old_abs)[:ABSTRACT_CAP],
        "url": str(new.get("url") or old.get("url") or ""),
        "venue": str(new.get("venue") or old.get("venue") or ""),
        "first_seen": first[:10],
        "last_seen": last[:10],
        "pairs": pairs,
        "limitations": _merge_limitations(old, new),
    }


def _merge_limitations(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(old.get("limitations") or []) + list(new.get("limitations") or []):
        text = " ".join(str(item or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text[:240])
        if len(out) >= 4:
            break
    return out


def key_of(entry: dict[str, Any]) -> str:
    return str(entry.get("cite_id") or entry.get("title") or "").strip()


def upsert(
    wiki: list[dict[str, Any]], entries: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Insert or merge; evict least-recently seen past WIKI_CAP. Returns (wiki, added)."""
    by_key: dict[str, dict[str, Any]] = {}
    for row in wiki:
        if isinstance(row, dict) and key_of(row):
            by_key[key_of(row)] = dict(row)
    added = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = key_of(entry)
        if not key:
            continue
        if key in by_key:
            by_key[key] = merge_entry(by_key[key], entry)
        else:
            by_key[key] = dict(entry)
            added += 1
    ranked = sorted(
        by_key.values(),
        key=lambda row: str(row.get("last_seen") or row.get("first_seen") or ""),
        reverse=True,
    )
    return ranked[:WIKI_CAP], added


def as_works(wiki: Iterable[dict[str, Any]]) -> list[FreshWork]:
    works: list[FreshWork] = []
    for row in wiki:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        works.append(
            FreshWork(
                title=title,
                published=str(row.get("published") or ""),
                arxiv_id=str(row.get("cite_id") or "") if row.get("source") == "arxiv" else "",
                abstract=str(row.get("abstract") or ""),
                source=str(row.get("source") or "arxiv"),
                work_id="" if row.get("source") == "arxiv" else str(row.get("cite_id") or ""),
                url=str(row.get("url") or ""),
                venue=str(row.get("venue") or ""),
            )
        )
    return works


def merge_works(
    live: Iterable[FreshWork], wiki: Iterable[FreshWork]
) -> list[FreshWork]:
    """Live survey wins on the same cite_id; wiki fills papers this mission did not pull."""
    by_id: dict[str, FreshWork] = {}
    for work in wiki:
        key = work.cite_id() or work.title
        if key:
            by_id[key] = work
    for work in live:
        key = work.cite_id() or work.title
        if key:
            by_id[key] = work
    return list(by_id.values())


def aimed_gap(
    claim: str,
    mechanism: str,
    gaps: Sequence[str],
    *,
    min_overlap: int = GAP_OVERLAP,
) -> str:
    """First limitation sentence the card aims at, or empty.

    Overlap is distinctive content words, not span support: hitting a gap
    is not a license to declare it solved. Prior-kill still closes a
    restatement. Empty when nothing distinctive overlaps.
    """
    body = (content_tokens(claim) | content_tokens(mechanism)) - APPLY_PAD - _GAP_PAD
    if not body:
        return ""
    for raw in gaps:
        text = _CITE_PREFIX.sub("", str(raw or "")).strip()
        if not text:
            continue
        overlap = body & (content_tokens(text) - APPLY_PAD - _GAP_PAD)
        if len(overlap) >= min_overlap:
            return text
    return ""


def gap_lines(wiki: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    """Limitation sentences from the wiki, for agenda targets. Empty if none."""
    lines: list[str] = []
    for row in wiki:
        if not isinstance(row, dict):
            continue
        cite = str(row.get("cite_id") or row.get("title") or "").strip()
        for item in row.get("limitations") or []:
            text = " ".join(str(item or "").split())
            if not text:
                continue
            head = f"[{cite}] {text}" if cite else text
            if head not in lines:
                lines.append(head)
            if len(lines) >= GAP_CAP:
                return tuple(lines)
    return tuple(lines)


def prompt_lines(wiki: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    """Shown to the generator. Empty keeps cached prompts byte-identical."""
    rows = [row for row in wiki if isinstance(row, dict)]
    lines: list[str] = []
    for row in rows[:PROMPT_CAP]:
        cite = str(row.get("cite_id") or "").strip()
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        year = str(row.get("published") or "")[:4]
        snippet = " ".join(str(row.get("abstract") or "").split())[:160]
        head = f"[{cite}] {title}" if cite else title
        if year:
            head += f" ({year})"
        if snippet:
            head += f" — {snippet}"
        lines.append(head)
    for gap in gap_lines(rows)[:4]:
        lines.append(
            "Literature gap (not a license to declare it solved): " + gap
        )
    return tuple(lines)


def block(lines: tuple[str, ...]) -> str:
    if not lines:
        return ""
    return (
        "Literature wiki for this neighbourhood (papers earlier missions "
        "actually retrieved; not evidence and not a license to restate them "
        "as new). A claim already supported by a span in one of these papers "
        "is closed_by_prior, not a discovery:\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n"
    )
