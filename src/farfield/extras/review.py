"""A model reviewer scores a value vector and a revision loop polishes the idea.

The graph oracle remains the executable novelty check: it can kill a
*graph* pair. This module is a colleague's opinion after that check has
already passed — or after the oracle stayed silent on a literature
mechanism. The opinion is a *vector*, never collapsed into one number:
novelty, clarity, feasibility, and the three questions a score sheet
usually skips — importance (if true, what changes), information gain
(what would testing it teach us), transfer potential (does it survive
outside this setting). No dimension kills, no weighted sum is computed
anywhere; a reader ranks by whichever dimension their resources care
about, and the executable promotion ladder (state.py) never reads these
numbers at all.

The reviewer expands a draft. It cannot fail the idea, cannot climb the
ladder, cannot change the pair, and cannot vote the next far concept off
the menu. Changing the pair would silently void the graph verdict.
What gets rewritten is the claim wording, the mechanism, the prediction.
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
from .generate import GeneratedCard
from .ideakind import idea_kind_of, parse_idea_kind
from .lifecycle import feasibility_rubric, parse_failure_types

KIND_REVIEW_BRIEFS = {
    "question": (
        "Score clarity of the question, whether the contrast is observable, "
        "whether answers discriminate, whether the claim stays observational, "
        "and whether an answer would change a research decision. "
        "Do not demand a two-arm intervention."
    ),
    "acquire": (
        "Score whether the capability gap is real, the recipe is concrete, "
        "the artifact is constructible, validators can attest it, and "
        "success would add research capability. "
        "Do not demand a two-arm effect."
    ),
    "theory": (
        "Score whether assumptions are named, the mechanism explains the "
        "phenomenon, predictions are derived, they discriminate a competitor, "
        "and a disconfirmation condition exists. "
        "Do not demand a two-arm."
    ),
    "probe": (
        "Score whether the lever is real, the freeze is shared, the arms "
        "are fair, the measure is the same, and the effect is not overclaimed."
    ),
}
from .livefeed import FreshWork
from .llm import complete_call

REVIEW_SYSTEM = (
    "You are a critical but fair reviewer of a research idea."
    " You may demand a rewrite of the same pair; you cannot fail the"
    " idea, change the distant concept, or pick a new field."
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
  plain_title: {plain_title}
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
{{"title": "...",
 "plain_title": "中文，<=20 字，外行能看懂的项目名，不逐词翻译 title",
 "one_liner": "中文一句话：这张卡在问什么问题",
 "why_it_matters": "中文 1-2 句：答案改变什么决策",
 "gap": "...", "idea": "...", "approach": "...",
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
    failure_types: tuple[str, ...] = ()

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
            "failure_types": list(self.failure_types),
            "status": (
                "every score is a reviewer's opinion given the retrieved"
                " papers; the vector is never collapsed into one number,"
                " does not replace the executable checks, and cannot kill"
            ),
        }


def _strong_review_works(
    card: GeneratedCard, topic: str, works: list[FreshWork] | None
) -> list[FreshWork]:
    """Review citations may only name claim-object-strong papers."""
    from .worldfields import papers_on_claim_object

    claim = str(getattr(card, "claim", "") or "").strip()
    if not claim:
        return []
    from .domain import topic_object_phrases

    topic_text = str(topic or "").strip()
    topic_phrases = [g for g in topic_object_phrases(topic_text) if " " in g]
    identity = topic_text if topic_phrases else claim
    return papers_on_claim_object(
        list(works or []), claim=claim, topic=identity, minimum="strong"
    )


def review_idea(
    client: Any,
    card: GeneratedCard,
    brief: ResearchBrief,
    topic: str,
    works: list[FreshWork],
    *,
    scientific: dict[str, Any] | None = None,
) -> Review:
    works = _strong_review_works(card, topic, works)
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
    completion = complete_call(
        client,
        prompt,
        purpose=f"review:{card.card_id}",
        system=REVIEW_SYSTEM,
        scientific=scientific,
    )
    return _review_from_completion(card.card_id, completion, works)


CARD_REVIEW_TEMPLATE = """Attack this {idea_kind} as an adversarial reviewer, before any experiment is registered. Your job is to find the weakest joint, not to be nice. Novelty here is your opinion given the papers — the structural check already passed.

Judge this idea as a {idea_kind}, not as a two-arm probe unless it is a probe.
{kind_brief}

Researcher's topic: {topic}
Concept pair (search provenance, not identity): {pair_a} × {pair_b}

Card under review:
  claim: {claim}
  mechanism: {mechanism}
  prediction: {prediction}
  preregistered_falsifier: {falsifier}
  strongest_objection_on_card: {objection}

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
 "critique": "<=60 words: the single most damaging objection a careful colleague would raise",
 "must_change": ["one concrete fix"],
 "failure_types": ["wrong_kind|missing_world|missing_observable|missing_validator|causal_overclaim|insufficient_contrast|recipe_not_attestable|theory_not_discriminative|probe_unfair|measure_mismatch"]}}

Scores are integers 1 to 5, each answering its own question — do not average them:
- novelty: 1 = these papers already state this claim, 5 = these papers leave it open.
- clarity: 1 = the claim's quantities are not named, 5 = an intern could restate the claim exactly.
- feasibility: {feasibility}
- importance: 1 = even if true nothing changes, 5 = methods or beliefs in this field change.
- information_gain: 1 = testing teaches nothing beyond the result, 5 = either outcome settles a live question.
- transfer_potential: 1 = a trick of this exact setting, 5 = the mechanism applies well beyond it.
keep_going is false only when another draft of this same pair would not help — that is not a fail that deletes the idea. must_change has 1 to 3 items, each a rewrite of claim, mechanism, or prediction — never "run more experiments", never a new concept pair, and never a new distant concept.
"""


def critique_card(
    client: Any,
    card: GeneratedCard,
    topic: str,
    works: list[FreshWork],
    *,
    scientific: dict[str, Any] | None = None,
) -> Review:
    """Adversarial review of an entered card, before registration.

    The gates admit; they do not measure quality. This is the colleague
    who reads the card and names the weakest joint while a rewrite is
    still legal — the same claim-object, the same pair, no experiment
    registered yet, so nothing can be rewritten to fit a result. The
    vector never kills, never climbs the ladder, and never replaces the
    distant concept. keep_going false means another draft of this pair
    would not help; it is not a fail that deletes the idea.
    """
    works = _strong_review_works(card, topic, works)
    kind = parse_idea_kind(idea_kind_of(card))
    prompt = CARD_REVIEW_TEMPLATE.format(
        idea_kind=kind,
        kind_brief=KIND_REVIEW_BRIEFS[kind],
        feasibility=feasibility_rubric(kind),
        topic=topic,
        pair_a=card.pair[0],
        pair_b=card.pair[1],
        claim=card.claim,
        mechanism=card.mechanism,
        prediction=card.prediction,
        falsifier=card.falsifier,
        objection=getattr(card, "objection", "") or "",
        papers=_paper_block(works),
    )
    completion = complete_call(
        client,
        prompt,
        purpose=f"critique:{card.card_id}",
        system=REVIEW_SYSTEM,
        scientific=scientific,
    )
    return _review_from_completion(card.card_id, completion, works)


def _review_from_completion(
    card_id: str, completion: Any, works: list[FreshWork]
) -> Review:
    from ..models import BlockedRecord
    from .generate import GenerationRefused as Refused

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
    failures = parse_failure_types(payload.get("failure_types"))
    if not failures:
        failures = parse_failure_types(fixes)
    return Review(
        card_id=card_id,
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
        failure_types=failures,
    )


def refine_brief(
    client: Any,
    card: GeneratedCard,
    brief: ResearchBrief,
    review: Review,
    topic: str,
    works: list[FreshWork],
    *,
    scientific: dict[str, Any] | None = None,
) -> ResearchBrief:
    works = _strong_review_works(card, topic, works)
    prompt = REFINE_TEMPLATE.format(
        topic=topic,
        pair_a=card.pair[0],
        pair_b=card.pair[1],
        claim=card.claim,
        title=brief.title,
        plain_title=brief.plain_title,
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
    completion = complete_call(
        client,
        prompt,
        purpose=f"refine:{card.card_id}",
        system=REFINE_SYSTEM,
        scientific=scientific,
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
