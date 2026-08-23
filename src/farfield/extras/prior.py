"""Executable false-novelty checks that do not ask the model to vote.

The graph oracle answers "have these two *labels* co-occurred?" That is
replayable and strict, and it is also exactly the hole SciAgents-style
systems fall into: the same idea under different words is invisible, and
"apply A to B" looks like an uncombined pair. These checks sit next to
the oracle, not instead of it.

- `apply_x_to_y` reads only the card. If stripping the two concept phrases
  leaves nothing but juxtaposition verbs and padding, the card is the
  named failure mode and it dies. No network, so it may kill.
- `claim_already_stated` reads papers this neighbourhood has actually
  retrieved (this mission's survey plus the research-state wiki). Support
  is "does a short span restating the claim's distinctive phrases exist",
  not bag-of-words recall against a long abstract. Scattered field
  vocabulary is not a restatement. A high hit can close the idea as
  already stated. It cannot rewrite the graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .livefeed import FreshWork

APPLY_PAD = frozenset(
    {
        "a",
        "an",
        "and",
        "application",
        "applied",
        "apply",
        "applying",
        "approach",
        "as",
        "based",
        "be",
        "better",
        "by",
        "combination",
        "combine",
        "combining",
        "domain",
        "effective",
        "efficiently",
        "extend",
        "extending",
        "for",
        "framework",
        "from",
        "improve",
        "improved",
        "improving",
        "in",
        "into",
        "is",
        "leverage",
        "leveraging",
        "method",
        "model",
        "new",
        "novel",
        "of",
        "on",
        "onto",
        "or",
        "our",
        "performance",
        "problem",
        "proposed",
        "result",
        "results",
        "setting",
        "task",
        "technique",
        "the",
        "this",
        "through",
        "to",
        "use",
        "used",
        "using",
        "via",
        "we",
        "with",
    }
)

# Phrase-span support that means "this paper restates the claim".
PRIOR_FLAG = 0.50
PRIOR_KILL = 0.70
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def content_tokens(text: str) -> frozenset[str]:
    """Content words, lowercased, light plural fold. Same spirit as sandbox."""
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    tokens = set()
    for word in cleaned.split():
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if len(word) >= 3:
            tokens.add(word)
    return frozenset(tokens)


def apply_x_to_y(claim: str, mechanism: str, pair: tuple[str, str]) -> bool:
    """True when the text is just using A as a method for B."""
    leftover = (content_tokens(claim) | content_tokens(mechanism)) - content_tokens(
        pair[0]
    ) - content_tokens(pair[1]) - APPLY_PAD
    return len(leftover) < 2


def claim_coverage(claim: str, paper_text: str) -> float:
    """Share of the claim's content words that appear in the paper text.

    Kept for topic-overlap checks. Prior-kill uses `claim_support`: a long
    abstract that happens to mention the same field words is not a restatement.
    """
    wanted = content_tokens(claim) - APPLY_PAD
    if not wanted:
        return 0.0
    have = content_tokens(paper_text)
    return len(wanted & have) / len(wanted)


def ordered_content(text: str) -> list[str]:
    """Content words in order, same folding as `content_tokens`."""
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    tokens: list[str] = []
    for word in cleaned.split():
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if len(word) >= 3 and word not in APPLY_PAD:
            tokens.append(word)
    return tokens


def claim_phrases(text: str) -> tuple[str, ...]:
    """Distinctive bigrams and trigrams; the unit a supporting span must carry."""
    tokens = ordered_content(text)
    phrases: list[str] = []
    seen: set[str] = set()
    for n in (2, 3):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            if gram not in seen:
                seen.add(gram)
                phrases.append(gram)
    return tuple(phrases)


def _windows(text: str) -> list[str]:
    parts = [piece.strip() for piece in _SENTENCE.split(text or "") if piece.strip()]
    if not parts:
        return [text or ""]
    windows = list(parts)
    for i in range(len(parts) - 1):
        windows.append(parts[i] + " " + parts[i + 1])
    return windows


def claim_support(claim: str, paper_text: str) -> tuple[float, str]:
    """Phrase recall inside the best 1–2 sentence span, plus that span.

    A paper supports a claim when a short contiguous span carries the
    claim's distinctive phrases. Token soup across the whole abstract is
    not support — that was the overlap hole.
    """
    wanted = claim_phrases(claim)
    if not wanted:
        tokens = ordered_content(claim)
        if not tokens:
            return 0.0, ""
        wanted_set = set(tokens)
        best = 0.0
        span = ""
        for window in _windows(paper_text):
            have = set(ordered_content(window))
            score = len(wanted_set & have) / len(wanted_set)
            if score > best:
                best, span = score, window
        return best, span[:280]
    best = 0.0
    span = ""
    for window in _windows(paper_text):
        haystack = " ".join(ordered_content(window))
        if not haystack:
            continue
        hit = sum(1 for phrase in wanted if phrase in haystack)
        score = hit / len(wanted)
        if score > best:
            best, span = score, window
    return best, span[:280]


@dataclass(frozen=True)
class PriorHit:
    coverage: float
    work: FreshWork
    kills: bool
    support: float = 0.0
    span: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": round(self.coverage, 3),
            "support": round(self.support, 3),
            "span": self.span,
            "kills": self.kills,
            "title": self.work.title,
            "cite_id": self.work.cite_id(),
            "source": self.work.source,
        }


def strongest_prior(
    claim: str, works: Iterable[FreshWork]
) -> PriorHit | None:
    """The retrieved paper whose best span most completely restates the claim."""
    best: PriorHit | None = None
    for work in works:
        paper = f"{work.title}. {work.abstract}"
        support, span = claim_support(claim, paper)
        if support < PRIOR_FLAG:
            continue
        hit = PriorHit(
            coverage=support,
            support=support,
            span=span,
            work=work,
            kills=support >= PRIOR_KILL,
        )
        if best is None or hit.support > best.support:
            best = hit
    return best
