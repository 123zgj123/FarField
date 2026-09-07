"""Reframe operators: hypotheses that question the field, not just combine it.

Far-field jumps ask "what distant concept could this field use?" — a
coverage device, and it stays one. High-value ideas are often a different
move: name an assumption the field makes by default and remove it, find
where two results contradict, or find where a method's guarantee breaks.
These operators produce hypothesis cards *about the anchor field itself*.

A reframe card cannot be judged by the concept-pair oracle — "have these
two labels co-occurred" is not the question it raises. So the honest
contract is different and stated on the card: the graph verdict is
skipped with a named reason, the text judges and `apply_x_to_y` still
apply, the assumption must be verbatim inside the claim or mechanism (a
challenge to nothing is not a challenge), and the card enters the same
survey → prior → diagnosis → probe path as every other survivor. It earns
promotion the same way: evidence, not vibes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..models import BlockedRecord
from .domain import topic_prompt_block
from .plugins import refuse_claim
from .generate import GeneratedCard, GenerationRefused, parse_struggle
from .llm import Completion
from .prior import content_tokens
from .program import block as program_block
from .wiki import block as wiki_block

OPERATORS: dict[str, str] = {
    "assumption_removal": (
        "Name ONE assumption this field makes by default, then propose what"
        " becomes possible or true when it is removed."
    ),
    "contradiction_search": (
        "Name ONE tension where results or common beliefs in this field pull"
        " in opposite directions, then propose a claim whose truth would"
        " resolve which side is right."
    ),
    "boundary_search": (
        "Name ONE regime where the field's standard guarantee or method"
        " should stop working, then propose a claim about what happens at"
        " that boundary."
    ),
}

GRAPH_SKIP_REASON = (
    "reframe card: the claim questions the field's own assumption, so the"
    " pair-co-occurrence oracle does not answer the question this card asks"
)

SYSTEM = (
    "You are questioning a research field's defaults, not combining it with"
    " another field. Answer with one JSON object and nothing else."
)

TEMPLATE = """{topic_block}{skills}A research field, its recent papers, and what earlier missions established.

Seed concept: {seed}
Concepts in the seed's own field: {near}
{context}{knowledge}{wiki}{program}{criteria}
{instruction}

Think on the card, not in a diary. Name the obvious approach that fails, why it fails, and the strongest objection. Those fields are unverified model text and are not evidence. At least two content words of the reframe must appear in the claim or the mechanism.

Answer with JSON:
{{"dead_end": "<=40 words: the obvious approach that keeps the assumption and fails",
 "why_failed": "<=40 words: the non-trivial reason it breaks",
 "reframe": "<=40 words: how dropping or relocating the assumption changes the question",
 "objection": "<=40 words: the strongest reviewer objection to the survivor",
 "assumption": "<=15 words: the default assumption, tension, or boundary you are targeting, as a noun phrase",
 "claim": "<=60 words, a falsifiable claim; it must contain the assumption phrase verbatim",
 "mechanism": "<=80 words on why the claim could hold",
 "prediction": "<=30 words: what later published work would show if the claim is right"}}

The claim must be about this field's own structure, not about importing a method from elsewhere. The assumption phrase must appear verbatim in the claim or the mechanism.
"""


def _refuse(attempted: str, unlock: str) -> GenerationRefused:
    return GenerationRefused(
        BlockedRecord(
            missing_capability="model_reframe_schema",
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
            "parse a JSON reframe card",
            f"the model answers with one JSON object; it answered with {text[:80]!r} ({exc})",
        ) from exc
    if not isinstance(payload, dict):
        raise _refuse(
            "read a JSON object for the reframe card",
            f"the top-level value is an object, not {type(payload).__name__}",
        )
    return payload


def card_from_payload(
    payload: dict[str, Any],
    *,
    seed_label: str,
    operator: str,
    model: str,
    artifact_digest: str,
    artifact_uri: str,
    replay_mode: str,
) -> GeneratedCard:
    fields = {}
    for name in ("assumption", "claim", "mechanism", "prediction"):
        value = str(payload.get(name) or "").strip()
        if not value:
            raise _refuse(
                f"read a non-empty {name}",
                f"a reframe card fills every field; {name} was empty",
            )
        fields[name] = value
    assumption_tokens = content_tokens(fields["assumption"])
    body_tokens = content_tokens(fields["claim"]) | content_tokens(fields["mechanism"])
    if not assumption_tokens or not assumption_tokens <= body_tokens:
        raise _refuse(
            "ground the assumption in the claim",
            "every content word of the assumption appears in the claim or the"
            " mechanism; a challenge to nothing is not a challenge",
        )
    struggle = parse_struggle(
        payload,
        claim=fields["claim"],
        mechanism=fields["mechanism"],
        refuse=_refuse,
    )
    card_id = "gen_rf" + hashlib.sha256(
        f"{operator}:{seed_label}:{fields['claim']}".encode()
    ).hexdigest()[:10]
    return GeneratedCard(
        card_id=card_id,
        operator=operator,
        claim=fields["claim"],
        mechanism=fields["mechanism"],
        prediction=fields["prediction"],
        falsifier="claim_contains_a_falsifiable_assertion",
        pair=(seed_label, fields["assumption"]),
        pair_nodes=("concept:seed", "assumption:free"),
        alienness=0.0,
        model=model,
        artifact_digest=artifact_digest,
        artifact_uri=artifact_uri,
        replay_mode=replay_mode,
        dead_end=struggle["dead_end"],
        why_failed=struggle["why_failed"],
        reframe=struggle["reframe"],
        objection=struggle["objection"],
    )


def generate_reframe(
    client: Any,
    *,
    seed_label: str,
    near_labels: tuple[str, ...],
    operator: str,
    context: tuple[str, ...] = (),
    knowledge: tuple[str, ...] = (),
    criteria: tuple[str, ...] = (),
    topic: str = "",
    skills: str = "",
    program: tuple[str, ...] = (),
    wiki: tuple[str, ...] = (),
) -> GeneratedCard:
    if operator not in OPERATORS:
        raise KeyError(f"unknown reframe operator {operator!r}; known: {sorted(OPERATORS)}")
    context_block = (
        "Recent papers near this field:\n" + "\n".join(f"- {line}" for line in context) + "\n"
        if context
        else ""
    )
    knowledge_block = (
        "From earlier missions (established findings and open questions):\n"
        + "\n".join(f"- {line}" for line in knowledge)
        + "\n"
        if knowledge
        else ""
    )
    criteria_block = (
        "The card will face these executable checks:\n"
        + "\n".join(f"- {line}" for line in criteria)
        + "\n"
        if criteria
        else ""
    )
    prompt = TEMPLATE.format(
        topic_block=topic_prompt_block(topic),
        skills=skills,
        seed=seed_label,
        near=", ".join(near_labels),
        context=context_block,
        knowledge=knowledge_block,
        wiki=wiki_block(tuple(wiki)),
        program=program_block(tuple(program)),
        criteria=criteria_block,
        instruction=OPERATORS[operator],
    )
    completion = client.complete(
        prompt,
        purpose=f"reframe:{operator}:{seed_label}",
        system=SYSTEM,
    )
    completion.assert_usable()
    try:
        card = card_from_payload(
            _parse(completion),
            seed_label=seed_label,
            operator=operator,
            model=completion.model,
            artifact_digest=completion.digest,
            artifact_uri=completion.artifact_uri,
            replay_mode=completion.mode,
        )
    except GenerationRefused as first:
        # cwm-iclr2027-v3 lost an importance-5 reframe to one empty field.
        # A schema slip gets one retry that quotes the refusal; a second
        # slip is the refusal.
        if first.record.missing_capability != "model_card_schema":
            raise
        retry_prompt = (
            prompt
            + "\nYOUR PREVIOUS ANSWER WAS REFUSED: "
            + str(first.record.unlock_condition)[:300]
            + ". Fill every field of the JSON schema; an empty claim is refused.\n"
        )
        completion = client.complete(
            retry_prompt,
            purpose=f"reframe:{operator}:{seed_label}:retry",
            system=SYSTEM,
        )
        completion.assert_usable()
        card = card_from_payload(
            _parse(completion),
            seed_label=seed_label,
            operator=operator,
            model=completion.model,
            artifact_digest=completion.digest,
            artifact_uri=completion.artifact_uri,
            replay_mode=completion.mode,
        )
    locked = refuse_claim(card.claim, card.mechanism, topic, seed_label)
    if locked:
        raise _refuse(
            "keep the reframe claim inside the researcher's topic",
            locked,
        )
    return card


def refine_reframe(
    client: Any,
    parent: GeneratedCard,
    *,
    seed_label: str,
    near_labels: tuple[str, ...],
    killed_by: list[str] | None = None,
    context: tuple[str, ...] = (),
    knowledge: tuple[str, ...] = (),
    criteria: tuple[str, ...] = (),
    topic: str = "",
    skills: str = "",
    program: tuple[str, ...] = (),
    wiki: tuple[str, ...] = (),
) -> GeneratedCard:
    """Rewrite the same field-challenge. Assumption phrase is locked.

    Reframe cards are not concept-graph pairs, so this cannot go through
    `refine_card` / `generate_card`.
    """
    killers = ", ".join(str(item) for item in (killed_by or [])[:6])
    extra = (
        f"Revise THIS idea. Keep assumption {parent.pair[1]!r} verbatim.",
        f"Previous claim: {parent.claim}",
        f"Previous prediction: {parent.prediction}",
        f"Why it died: {killers or 'unspecified bottleneck'}.",
        "Change the claim or the prediction so that failure cannot recur. "
        "Do not name a new assumption.",
    )
    card = generate_reframe(
        client,
        seed_label=seed_label,
        near_labels=near_labels,
        operator=parent.operator,
        context=context,
        knowledge=tuple(knowledge) + extra,
        criteria=criteria,
        topic=topic,
        skills=skills,
        program=program,
        wiki=wiki,
    )
    if card.pair[1] != parent.pair[1]:
        raise _refuse(
            "keep the same assumption while evolving this idea",
            "a refine that names a new assumption is a new reframe, not a rewrite",
        )
    return card
