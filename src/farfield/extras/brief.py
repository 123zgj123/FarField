"""Turn a surviving hypothesis into a research-ready idea.

A card that passed the judges is still only a claim. This module does the
work a colleague would do before saying "this is worth starting": pull the
nearby papers (already fetched by the caller), read what they actually did,
name the gap, and write a short program — approach, first experiments,
what to compare against, how it dies. The text is still model text, not
evidence; the papers listed next to it are the survey, and `read_first`
must name ids from that list so the briefing cannot invent a bibliography.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..models import BlockedRecord
from .generate import GenerationRefused, GeneratedCard
from .livefeed import FreshWork
from .llm import Completion
from .domain import claim_home
from .plugins import refuse_writeup

SYSTEM = (
    "You are a researcher writing a short briefing so a colleague can start"
    " the project next week. Answer with one JSON object and nothing else."
)

TEMPLATE = """A hypothesis survived novelty and writing checks. Write a research-ready idea grounded in the papers below.

Researcher's topic: {topic}
Hypothesis claim: {claim}
Why it might hold: {mechanism}
Concept pair: {pair_a} × {pair_b}
What later work would show: {prediction}

{struggle}Papers retrieved today (cite only these ids in read_first and when you mention a paper):
{papers}

{limitation}
Answer with JSON:
{{"title": "<=12 words, a project title a lab would put on a slide",
 "gap": "<=80 words: what these papers do and what they leave open; name at least one id if any papers were given",
 "idea": "<=90 words: the research program, not a restatement of the claim",
 "approach": "<=80 words: how to pursue it (method, data, or construction)",
 "first_steps": ["a concrete next action", "another", "another"],
 "baseline": "<=40 words: what you compare against or what would count as the obvious attempt",
 "risks": "<=50 words: how this project fails",
 "read_first": ["<paper id from the list>", "..."]}}

`read_first` must be ids copied from the list above. If the list is empty, use []. Do not invent papers. first_steps has 2 to 5 items, each a thing someone can do next week. The title, idea, and approach must stay in the hypothesis's own field (the claim and the concept pair). Do not retitle a hardness, spanner, or indexing claim as an LLM-agent or tool-use paper just because that is the researcher's topic.
"""


def _refuse(attempted: str, unlock: str) -> GenerationRefused:
    return GenerationRefused(
        BlockedRecord(
            missing_capability="model_brief_schema",
            attempted=attempted,
            unlock_condition=unlock,
        )
    )


def _parse(completion: Completion) -> dict[str, Any]:
    text = completion.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _refuse(
            "parse a JSON research brief",
            f"the model answers with one JSON object; it answered with {text[:80]!r} ({exc})",
        ) from exc
    if not isinstance(payload, dict):
        raise _refuse(
            "read a JSON object for the brief",
            f"the top-level value is an object, not {type(payload).__name__}",
        )
    return payload


def _paper_block(works: list[FreshWork]) -> str:
    if not works:
        return "(none retrieved)"
    lines = []
    for work in works:
        abstract = (work.abstract or "").strip()
        if len(abstract) > 420:
            abstract = abstract[:417] + "..."
        body = f"[{work.cite_id()}] {work.title} ({work.published})"
        if abstract:
            body += f"\n  {abstract}"
        lines.append(body)
    return "\n".join(lines)


def _struggle_brief_block(card: GeneratedCard) -> str:
    if not getattr(card, "dead_end", ""):
        return ""
    return (
        "The author's discarded approach (unverified model text, not evidence):\n"
        f"- failed approach: {card.dead_end}\n"
        f"- why it failed: {card.why_failed}\n"
        f"- reframe: {card.reframe}\n"
        f"- strongest objection: {card.objection}\n"
        "The briefing is still about the surviving claim.\n\n"
    )


@dataclass(frozen=True)
class ResearchBrief:
    card_id: str
    title: str
    gap: str
    idea: str
    approach: str
    first_steps: tuple[str, ...]
    baseline: str
    risks: str
    read_first: tuple[str, ...]
    papers: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "title": self.title,
            "gap": self.gap,
            "idea": self.idea,
            "approach": self.approach,
            "first_steps": list(self.first_steps),
            "baseline": self.baseline,
            "risks": self.risks,
            "read_first": list(self.read_first),
            "papers": list(self.papers),
        }


def write_brief(
    client: Any,
    card: GeneratedCard,
    topic: str,
    works: list[FreshWork],
    *,
    skills: str = "",
) -> ResearchBrief:
    """One briefing for one surviving card. Refuses; never repairs."""
    limitation = (
        ""
        if works
        else "No live papers were retrieved. Say so in `gap` and still write"
        " a program someone can start from the hypothesis alone.\n"
    )
    prompt = TEMPLATE.format(
        topic=topic,
        claim=card.claim,
        mechanism=card.mechanism,
        pair_a=card.pair[0],
        pair_b=card.pair[1],
        prediction=card.prediction,
        struggle=_struggle_brief_block(card),
        papers=_paper_block(works),
        limitation=limitation,
    )
    if skills:
        prompt = skills + prompt
    completion = client.complete(
        prompt,
        purpose=f"research_brief:{card.card_id}",
        system=SYSTEM,
    )
    completion.assert_usable()
    return brief_from_payload(
        card.card_id,
        _parse(completion),
        works,
        topic=topic,
        home=claim_home(card.claim, card.mechanism, card.pair),
        seed_label=card.pair[0],
    )


def brief_from_payload(
    card_id: str,
    payload: dict[str, Any],
    works: list[FreshWork],
    *,
    topic: str = "",
    home: str = "",
    seed_label: str = "",
) -> ResearchBrief:
    """Shared schema teeth for the first briefing and every later revision."""
    known = {work.cite_id() for work in works if work.cite_id()}
    fields = {}
    for name in ("title", "gap", "idea", "approach", "baseline", "risks"):
        value = str(payload.get(name) or "").strip()
        if not value:
            raise _refuse(
                f"read a non-empty {name}",
                f"the briefing fills every field; {name} was empty",
            )
        fields[name] = value

    raw_steps = payload.get("first_steps")
    if not isinstance(raw_steps, list) or not 2 <= len(raw_steps) <= 5:
        raise _refuse(
            "read first_steps",
            "first_steps is a list of 2 to 5 concrete next actions",
        )
    steps = tuple(str(item).strip() for item in raw_steps if str(item).strip())
    if len(steps) < 2:
        raise _refuse(
            "read first_steps",
            "first_steps is a list of 2 to 5 concrete next actions",
        )

    raw_reads = payload.get("read_first")
    if raw_reads is None:
        raw_reads = []
    if not isinstance(raw_reads, list):
        raise _refuse(
            "read read_first",
            "read_first is a list of paper ids copied from the paper list",
        )
    reads = tuple(str(item).strip() for item in raw_reads if str(item).strip())
    invented = [item for item in reads if item not in known]
    if invented:
        raise _refuse(
            f"map {invented[0]!r} onto a retrieved paper",
            "read_first copies ids from the retrieved list; invented citations"
            " are not a survey",
        )
    if works and not reads:
        raise _refuse(
            "name at least one paper to read first",
            "when papers were retrieved, read_first names at least one of them",
        )
    if works and not any(work.cite_id() in fields["gap"] for work in works):
        titled = any(work.title[:24] in fields["gap"] for work in works)
        if not titled:
            raise _refuse(
                "ground the gap in a retrieved paper",
                "gap names at least one retrieved paper id, so the opening"
                " is about these papers and not a generic literature claim",
            )

    if topic.strip() and home.strip():
        locked = refuse_writeup(
            fields["title"],
            fields["idea"],
            home,
            topic,
            seed_label,
        )
        if locked:
            raise _refuse(
                "keep the write-up in the claim's field",
                locked,
            )

    return ResearchBrief(
        card_id=card_id,
        first_steps=steps,
        read_first=reads,
        papers=tuple(work.to_dict() for work in works),
        **fields,
    )


def compile_brief(
    card: GeneratedCard,
    works: list[FreshWork] | None = None,
    *,
    diagnosis: Any = None,
) -> ResearchBrief:
    """Attested fallback when the briefing model refuses. No new claims."""
    diag = {}
    if diagnosis is not None:
        if hasattr(diagnosis, "to_dict"):
            diag = dict(diagnosis.to_dict() or {})
        elif isinstance(diagnosis, dict):
            diag = dict(diagnosis)
    steps: list[str] = []
    experiment = str(diag.get("experiment") or "").strip()
    if experiment:
        steps.append(experiment)
    treatment = str(diag.get("treatment_arm") or "").strip()
    control = str(diag.get("control_arm") or "").strip()
    if treatment:
        steps.append(f"Run the treatment arm: {treatment}")
    if control:
        steps.append(f"Run the control arm: {control}")
    if card.prediction:
        steps.append(f"Success if later work sees: {card.prediction}")
    steps.append(
        "Do not cite the cheap probe or metrics.json as an empirical result."
    )
    papers = []
    for work in works or []:
        if hasattr(work, "to_dict"):
            papers.append(work.to_dict())
        elif isinstance(work, dict):
            papers.append(dict(work))
    title_words = [part for part in str(card.claim or "").split() if part][:12]
    return ResearchBrief(
        card_id=card.card_id,
        title=" ".join(title_words) or "Research plan",
        gap="",
        idea=card.claim,
        approach=card.mechanism,
        first_steps=tuple(steps[:5]),
        baseline=str(
            diag.get("baseline_method") or diag.get("alternative") or ""
        ),
        risks=str(getattr(card, "falsifier", "") or ""),
        read_first=(),
        papers=tuple(papers),
    )
