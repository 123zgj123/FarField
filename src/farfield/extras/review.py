"""A model reviewer scores a value vector and a revision loop polishes the idea.

The graph oracle remains the executable novelty check: it can kill a pair.
This module is a colleague's opinion after that check has already passed.
The opinion is a *vector*, never collapsed into one number: novelty,
clarity, feasibility, and the three questions a score sheet usually skips —
importance (if true, what changes), information gain (what would testing
it teach us), transfer potential (does it survive outside this setting).
No dimension kills, no weighted sum is computed anywhere; a reader ranks
by whichever dimension their resources care about, and the executable
promotion ladder (state.py) never reads these numbers at all.

The concept pair does not move during revision. Changing the pair would
silently void the graph verdict. What gets rewritten is the gap, the
program, the first steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .brief import (
    ResearchBrief,
    _paper_block,
    _parse,
    brief_from_payload,
)
from .domain import claim_home
from .generate import GeneratedCard, GenerationRefused
from .livefeed import FreshWork

REVIEW_SYSTEM = (
    "You are a critical but fair reviewer of a research idea."
    " Answer with one JSON object and nothing else."
)

REVIEW_TEMPLATE = """Score this research idea as a reviewer. Novelty here is your opinion of the research program given the papers — not a substitute for the structural check that already passed.

Researcher's topic: {topic}
Concept pair (fixed): {pair_a} × {pair_b}
Claim: {claim}

Current idea:
  title: {title}
  gap: {gap}
  idea: {idea}
  approach: {approach}
  first_steps: {steps}
  baseline: {baseline}
  risks: {risks}

Papers retrieved today:
{papers}

Answer with JSON:
{{"novelty": 1,
 "novelty_note": "<=40 words, why this score; name a paper id if any were given",
 "clarity": 1,
 "feasibility": 1,
 "importance": 1,
 "information_gain": 1,
 "transfer_potential": 1,
 "keep_going": true,
 "critique": "<=60 words: the single most useful change for the next draft",
 "must_change": ["one concrete fix"]}}

Scores are integers 1 to 5, each answering its own question — do not average them in your head:
- novelty: 1 = obvious given these papers, 5 = these papers leave this program open.
- importance: 1 = even if true, nothing downstream changes; 5 = if true, methods or beliefs in this field change.
- information_gain: 1 = testing it teaches nothing beyond the result itself; 5 = either outcome settles a question the field actually has.
- transfer_potential: 1 = a trick of this exact setting; 5 = the mechanism, if real, applies well beyond it.
keep_going is false only when another draft would not help. must_change has 1 to 3 items.
"""

REFINE_SYSTEM = (
    "You are revising a research idea after a review. Answer with one JSON"
    " object and nothing else. Do not change the concept pair."
)

REFINE_TEMPLATE = """Revise the research-ready idea. Keep the same concept pair. Address the review. Cite only the papers below.

Researcher's topic: {topic}
Concept pair (do not change): {pair_a} × {pair_b}
Claim: {claim}

Previous draft:
  title: {title}
  gap: {gap}
  idea: {idea}
  approach: {approach}
  first_steps: {steps}
  baseline: {baseline}
  risks: {risks}
  read_first: {reads}

Reviewer (opinion, not a kill):
  novelty {novelty}/5 — {novelty_note}
  critique: {critique}
  must_change: {must_change}

Papers retrieved today (cite only these ids):
{papers}

Answer with the same briefing JSON as before:
{{"title": "...", "gap": "...", "idea": "...", "approach": "...",
 "first_steps": ["...", "..."], "baseline": "...", "risks": "...",
 "read_first": ["..."]}}

Change at least the idea or the first_steps. gap must still name a retrieved id if papers were given. Do not retitle the project into a field the claim does not occupy.
"""


@dataclass(frozen=True)
class Review:
    card_id: str
    novelty: int
    novelty_note: str
    clarity: int
    feasibility: int
    importance: int
    information_gain: int
    transfer_potential: int
    keep_going: bool
    critique: str
    must_change: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "novelty": self.novelty,
            "novelty_note": self.novelty_note,
            "clarity": self.clarity,
            "feasibility": self.feasibility,
            "importance": self.importance,
            "information_gain": self.information_gain,
            "transfer_potential": self.transfer_potential,
            "value_vector": {
                "novelty": self.novelty,
                "clarity": self.clarity,
                "feasibility": self.feasibility,
                "importance": self.importance,
                "information_gain": self.information_gain,
                "transfer_potential": self.transfer_potential,
            },
            "keep_going": self.keep_going,
            "critique": self.critique,
            "must_change": list(self.must_change),
            "status": (
                "every score is a reviewer's opinion given the retrieved"
                " papers; the vector is never collapsed into one number,"
                " does not replace the executable checks, and cannot kill"
            ),
        }


def review_idea(
    client: Any,
    card: GeneratedCard,
    brief: ResearchBrief,
    topic: str,
    works: list[FreshWork],
) -> Review:
    from ..models import BlockedRecord
    from .generate import GenerationRefused as Refused

    prompt = REVIEW_TEMPLATE.format(
        topic=topic,
        pair_a=card.pair[0],
        pair_b=card.pair[1],
        claim=card.claim,
        title=brief.title,
        gap=brief.gap,
        idea=brief.idea,
        approach=brief.approach,
        steps="; ".join(brief.first_steps),
        baseline=brief.baseline,
        risks=brief.risks,
        papers=_paper_block(works),
    )
    completion = client.complete(
        prompt,
        purpose=f"review:{card.card_id}",
        system=REVIEW_SYSTEM,
    )
    completion.assert_usable()
    try:
        payload = _parse(completion)
    except Refused as exc:
        raise Refused(
            BlockedRecord(
                missing_capability="model_review_schema",
                attempted=exc.record.attempted,
                unlock_condition=exc.record.unlock_condition,
            )
        ) from exc

    def score(name: str) -> int:
        raw = payload.get(name)
        try:
            value = int(raw)
        except (TypeError, ValueError) as error:
            raise Refused(
                BlockedRecord(
                    missing_capability="model_review_schema",
                    attempted=f"read {name}",
                    unlock_condition=f"{name} is an integer from 1 to 5",
                )
            ) from error
        if value not in (1, 2, 3, 4, 5):
            raise Refused(
                BlockedRecord(
                    missing_capability="model_review_schema",
                    attempted=f"read {name}",
                    unlock_condition=f"{name} is an integer from 1 to 5",
                )
            )
        return value

    note = str(payload.get("novelty_note") or "").strip()
    critique = str(payload.get("critique") or "").strip()
    if not note or not critique:
        raise Refused(
            BlockedRecord(
                missing_capability="model_review_schema",
                attempted="read novelty_note and critique",
                unlock_condition="the review names why the novelty score and what to change",
            )
        )
    if works and not any(work.cite_id() in note for work in works):
        titled = any(work.title[:24] in note for work in works)
        if not titled:
            raise Refused(
                BlockedRecord(
                    missing_capability="model_review_schema",
                    attempted="ground novelty_note in a retrieved paper",
                    unlock_condition="novelty_note names a retrieved paper id"
                    " when papers were given",
                )
            )
    raw_fix = payload.get("must_change")
    if not isinstance(raw_fix, list) or not 1 <= len(raw_fix) <= 3:
        raise Refused(
            BlockedRecord(
                missing_capability="model_review_schema",
                attempted="read must_change",
                unlock_condition="must_change is a list of 1 to 3 concrete fixes",
            )
        )
    fixes = tuple(str(item).strip() for item in raw_fix if str(item).strip())
    keep = payload.get("keep_going")
    if not isinstance(keep, bool):
        raise Refused(
            BlockedRecord(
                missing_capability="model_review_schema",
                attempted="read keep_going",
                unlock_condition="keep_going is true or false",
            )
        )
    return Review(
        card_id=card.card_id,
        novelty=score("novelty"),
        novelty_note=note,
        clarity=score("clarity"),
        feasibility=score("feasibility"),
        importance=score("importance"),
        information_gain=score("information_gain"),
        transfer_potential=score("transfer_potential"),
        keep_going=keep,
        critique=critique,
        must_change=fixes,
    )


def refine_brief(
    client: Any,
    card: GeneratedCard,
    brief: ResearchBrief,
    review: Review,
    topic: str,
    works: list[FreshWork],
) -> ResearchBrief:
    prompt = REFINE_TEMPLATE.format(
        topic=topic,
        pair_a=card.pair[0],
        pair_b=card.pair[1],
        claim=card.claim,
        title=brief.title,
        gap=brief.gap,
        idea=brief.idea,
        approach=brief.approach,
        steps="; ".join(brief.first_steps),
        baseline=brief.baseline,
        risks=brief.risks,
        reads=", ".join(brief.read_first),
        novelty=review.novelty,
        novelty_note=review.novelty_note,
        critique=review.critique,
        must_change="; ".join(review.must_change),
        papers=_paper_block(works),
    )
    completion = client.complete(
        prompt,
        purpose=f"refine:{card.card_id}",
        system=REFINE_SYSTEM,
    )
    completion.assert_usable()
    revised = brief_from_payload(
        card.card_id,
        _parse(completion),
        works,
        topic=topic,
        home=claim_home(card.claim, card.mechanism, card.pair),
        seed_label=card.pair[0],
    )
    if revised.idea == brief.idea and revised.first_steps == brief.first_steps:
        from ..models import BlockedRecord
        from .generate import GenerationRefused as Refused

        raise Refused(
            BlockedRecord(
                missing_capability="model_brief_schema",
                attempted="read a revised idea",
                unlock_condition="a revision changes the idea or the first_steps",
            )
        )
    return revised
