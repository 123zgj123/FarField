"""LLM-generated hypothesis cards from a trajectory-informed landing.

Generation writes the claim; it does not choose where to look.
`researchspace` recovers how related work evolved and proposes the
distant mechanisms. The model reads that landing, the topic object,
and the failure capsules, and answers in a strong schema — claim,
mechanism, a concept pair copied verbatim from what it was shown, a
pre-registered falsifier, and a testable prediction. Anything
off-schema is refused and recorded; the refusal rate is a P1 exit
criterion, so a refusal here is a reading, not an error.

The model's text never becomes evidence.
The only parts of a card with causal consequences are executable: the pair
and the falsifier name. The graph oracle may kill a pair only when both
ends are nodes of this corpus. A distant concept taken from verified
literature is not a graph fact — novelty for those pairs is prior-kill,
not `pair_not_already_combined`.

The graph oracle is the killing floor for *graph* pairs. Its three checks
were sized on the real corpus before being adopted, because a check that
cannot pass or cannot fail is a ceremony:

- `pair_not_already_combined`: the two concepts already co-occur pre-T
  (6.9% of random pairs in `ds-arxiv-concepts`). Same definition as the
  novelty anchor's `CONCEPT_PAIRS.is_novel`, and a test holds them equal.
- `pair_not_implied_by_neighborhood`: neighborhood Jaccard above the
  corpus's own 0.9 quantile among unlinked pairs. The naive form — any
  shared neighbour kills — was measured first and kills 99.4% of unlinked
  pairs on this graph; a constant killer measures nothing.
- `endpoint_is_not_a_concept_hub`: either concept's degree above the 0.9
  degree quantile. A concept the whole corpus uses combines with anything,
  so reaching it is a fact about the graph, not about the seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from typing import Any, Sequence

from ..graph import GraphSnapshot
from ..models import BlockedRecord
from .domain import topic_prompt_block
from .ideakind import ACQUIRE, PROBE, QUESTION, THEORY, parse_idea_kind
from .lifecycle import (
    LOCAL_NODE_PREFIX,
    LOCAL_ORIGIN_LABEL,
    ORIGIN_LITERATURE,
    ORIGIN_LOCAL,
    admit_kind_transition,
    causal_overclaim_reason,
    compile_required,
    infer_transition_reason,
    is_local_origin_label,
    lineage_id_for,
    local_node_id,
    parse_kind_fields,
    parse_origin,
    refuse_kind_transition,
)
from .plugins import refuse_claim
from .program import block as program_block
from .wiki import block as wiki_block
from .llm import Completion, LLMUnavailable, complete_call
from .prior import content_tokens

CHECK_CATALOGUE = (
    "pair_not_already_combined",
    "pair_not_implied_by_neighborhood",
    "endpoint_is_not_a_concept_hub",
)

HUB_QUANTILE = 0.9
IMPLICATION_QUANTILE = 0.9
IMPLICATION_SAMPLE = 2000
IMPLICATION_RNG_SEED = "p1-implication-ceiling"
STRUGGLE_FIELDS = ("dead_end", "why_failed", "reframe", "objection")
MIN_STRUGGLE_WORDS = 8
DEAD_END_CLAIM_OVERLAP = 0.55
REFRAME_BODY_OVERLAP = 2

LITERATURE_NODE_PREFIX = "literature:"

SYSTEM = (
    "You are writing one research idea for this mission."
    " idea_kind is a scientific action type: question, acquire, theory, or probe."
    " These are different ideas, not four wordings of a two-arm test."
    " Prefer question (an observable contrast) or acquire"
    " (a missing world/capability) over inventing a numeric intervention."
    " A probe is legal only when this freeze has a manipulable handle"
    " and a measurable outcome."
    " The distant concept is search provenance, not a new field: pair[0]"
    " stays the researcher's named object."
    " If there is no verified far literature, origin is local_problem_structure;"
    " do not invent a distant mechanism."
    " Struggle fields (dead_end, why_failed, reframe, objection) are optional"
    " search provenance, not a validity ritual."
    " Answer with one JSON object and nothing else."
)

TEMPLATE = """{topic_block}{skills}A research seed and a set of distant concepts. The distant list is a trajectory-informed landing in research space (an observed development move applied to the current position), not a new field and not a random mix of catalog nouns.

Seed concept: {seed}
Concepts in the seed's own field: {near}
Distant concepts retrieved at the jump landing (operator: {operator}): {far}
{pairings}{jump_block}{context}{knowledge}{wiki}{program}{failures}{struggle_memory}{criteria}{value}{compile_block}
Propose ONE idea that combines the seed's field with exactly one of the distant concepts. Do not merely apply the first concept to the second: the mechanism must name a third technical idea (a construction, a bound, or a measurement) that would disappear if you only juxtaposed the two names. Prefer an unused declared handle and a wiki limitation sentence as the gap. Do not spray a cosine neighbour as a new topic.

Choose idea_kind:
- question: an observational contrast. Name the question, the observable, the contrast, why the answer matters, and what decision each outcome would change. Do not invent an intervention, a two-arm, or an effect margin.
- acquire: a missing world/validator/artifact/replay capability. Name the gap, the construction recipe, the validators, and the readiness/failure criteria. Success is attesting a new world, not a two-arm effect.
- theory: a mechanism that explains a named phenomenon. Name assumptions, derived predictions, boundary and disconfirmation conditions. Unchecked predictions are not WORLD facts.
- probe: a fair two-arm intervention this freeze can feel. Only when a manipulable handle and a measure exist.

Struggle fields are optional far-jump provenance, not required for a legal question/acquire/theory.

Answer with JSON:
{{"idea_kind": "question | acquire | theory | probe",
 "origin": "literature_far | local_problem_structure",
 "research_target": "<=40 words: the actual question, recipe, hypothesis, or phenomenon — not the pair",
 "dead_end": "optional <=40 words: a discarded approach, only if this is far-jump provenance",
 "why_failed": "optional",
 "reframe": "optional",
 "objection": "optional",
 "derivation": "<=80 words: required for theory; otherwise optional",
 "claim": "<=80 words: the research target in prose",
 "mechanism": "<=80 words on why this action is the right next one",
 "pair": ["<topic object, copied exactly>", "<distant concept OR 'local problem structure'>"],
 "falsifier": "<one of: {checks}>",
 "prediction": "<=40 words: question contrast / acquire readiness / theory prediction / probe measurable",
 "research_question": "question only",
 "contrast": "question only",
 "why_answer_matters": "question only",
 "possible_outcomes": "question only",
 "decision_change": "question only",
 "required_world_state": "question/acquire",
 "capability_gap": "acquire only",
 "recipe": "acquire only",
 "validators": "acquire only",
 "readiness_criteria": "acquire only",
 "expected_affordances": "acquire only",
 "assumptions": "theory only",
 "derived_predictions": "theory only",
 "boundary_conditions": "theory only",
 "disconfirmation": "theory only",
 "transition_reason": "optional: missing_world | missing_observable | causal_identification_ready | harvest_success | prediction_observable | prediction_manipulable"{compile_json}}}

`pair` must copy two concepts exactly as written above: the first from the seed's field, the second from the distant list. `falsifier` must be exactly one of the listed names; it is the executable check you judge this card most at risk of failing. Anything else voids the card.
"""


class GenerationRefused(RuntimeError):
    def __init__(self, record: BlockedRecord, failed_far: str = "", used_handle: bool = False):
        super().__init__(record.unlock_condition)
        self.record = record
        self.failed_far = failed_far
        # The weld landed on a lever/observable this mission already used:
        # the far label is fine, the handle is not — retry with the handle
        # constraint made hard instead of dropping the label.
        self.used_handle = used_handle


def _refuse(
    attempted: str,
    unlock: str,
    capability: str = "model_card_schema",
    failed_far: str = "",
    used_handle: bool = False,
) -> GenerationRefused:
    return GenerationRefused(
        BlockedRecord(
            missing_capability=capability,
            attempted=attempted,
            unlock_condition=unlock,
        ),
        failed_far=failed_far,
        used_handle=used_handle,
    )


_ATTRACTOR_REWRITE = (
    "dropout",
    "dropping",
    "independently drop",
    "independently mask",
    "mask 20%",
)


def _bind_menu(world_menu: Any, blocked_handles: tuple[str, ...] | list[str]) -> Any:
    if world_menu is None or not blocked_handles:
        return world_menu
    binder = getattr(world_menu, "with_blocked", None)
    if callable(binder):
        return binder(blocked_handles)
    return world_menu


def _without_parent_handle(
    used: tuple[str, ...] | list[str], parent: "GeneratedCard"
) -> tuple[str, ...]:
    lever = str(parent.world_lever or "").strip()
    if not lever or lever == "none":
        return tuple(used)
    handle = (
        f"{lever}/{parent.world_observable}"
        if parent.world_observable
        else lever
    )
    return tuple(
        item
        for item in used
        if item != handle
        and item != lever
        and not str(item).startswith(lever + "/")
    )


def _abandons_observational_handle(parent: "GeneratedCard", card: "GeneratedCard") -> bool:
    lever = str(parent.world_lever or "")
    if not lever.startswith("contrast:"):
        return False
    surface = f"{card.claim} {card.mechanism} {card.prediction}".lower()
    return any(term in surface for term in _ATTRACTOR_REWRITE)


@dataclass(frozen=True)
class GeneratedCard:
    card_id: str
    operator: str
    claim: str
    mechanism: str
    prediction: str
    falsifier: str
    pair: tuple[str, str]
    pair_nodes: tuple[str, str]
    alienness: float
    model: str
    artifact_digest: str
    artifact_uri: str
    replay_mode: str
    # Optional far-jump provenance. Never a universal validity ritual,
    # never evidence, never a ladder input.
    dead_end: str = ""
    why_failed: str = ""
    reframe: str = ""
    objection: str = ""
    derivation: str = ""
    world_lever: str = ""
    world_observable: str = ""
    far_maps_to_lever: str = ""
    idea_kind: str = PROBE
    # NLL surprise of the visible completion (plausibility's second term).
    # None when the call did not request logprobs; only written to the dict
    # when present, so reports from before this field existed stay
    # byte-identical on replay.
    mean_nll: float | None = None
    # Registered object identity. Optional so older fixtures stay valid.
    # Natural-language fields above are presentation; this is the type.
    claim_spec: dict[str, Any] | None = None
    # Identity is idea_id (= card_id) + lineage_id. pair is search provenance.
    lineage_id: str = ""
    origin: str = ORIGIN_LITERATURE
    research_target: str = ""
    parent_idea_id: str = ""
    transition_reason: str = ""
    world_requirements: str = ""
    world_version: str = ""
    kind_fields: dict[str, Any] | None = None

    @property
    def idea_id(self) -> str:
        return self.card_id

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        if self.mean_nll is not None:
            payload["mean_nll"] = self.mean_nll
        return payload

    def struggle_event(self) -> dict[str, str]:
        if not self.dead_end:
            return {}
        return {
            "dead_end": self.dead_end,
            "why_failed": self.why_failed,
            "reframe": self.reframe,
            "objection": self.objection,
        }

    def _base_dict(self) -> dict[str, Any]:
        payload = {
            "card_id": self.card_id,
            "operator": self.operator,
            "claim": self.claim,
            "mechanism": self.mechanism,
            "prediction": self.prediction,
            "falsifier": self.falsifier,
            "pair": list(self.pair),
            "pair_nodes": list(self.pair_nodes),
            "alienness": round(self.alienness, 10),
            "model": self.model,
            "artifact_digest": self.artifact_digest,
            "artifact_uri": self.artifact_uri,
            "replay_mode": self.replay_mode,
            "dead_end": self.dead_end,
            "why_failed": self.why_failed,
            "reframe": self.reframe,
            "objection": self.objection,
            "status": (
                "claim/mechanism/prediction are unverified model text and"
                " are not evidence; pair is search provenance; only the"
                " executable checks and the pre-registered falsifier acted"
                " on the campaign"
            ),
            "idea_kind": self.idea_kind or PROBE,
            "idea_id": self.card_id,
            "lineage_id": self.lineage_id or "",
            "origin": self.origin or ORIGIN_LITERATURE,
        }
        if self.research_target:
            payload["research_target"] = self.research_target
        if self.parent_idea_id:
            payload["parent_idea_id"] = self.parent_idea_id
        if self.transition_reason:
            payload["transition_reason"] = self.transition_reason
        if self.world_requirements:
            payload["world_requirements"] = self.world_requirements
        if self.world_version:
            payload["world_version"] = self.world_version
        if self.kind_fields:
            payload["kind_fields"] = dict(self.kind_fields)
        if self.derivation:
            payload["derivation"] = self.derivation
        if self.world_lever:
            payload["world_lever"] = self.world_lever
            payload["world_observable"] = self.world_observable
            payload["far_maps_to_lever"] = self.far_maps_to_lever
        if self.claim_spec:
            payload["claim_spec"] = dict(self.claim_spec)
        return payload


def _optional_derivation(raw: Any) -> str:
    """Theory sketch. Optional so older fixtures stay valid. Never evidence."""
    text = str(raw or "").strip()
    if len(text.split()) < 8:
        return ""
    return text


def _failures_block(capsules: list[dict[str, Any]], limit: int = 3) -> str:
    if not capsules:
        return ""
    lines = []
    for row in capsules[:limit]:
        context = str(row.get("context") or "").strip()
        mechanism = str(row.get("mechanism") or "").strip()
        why = str(row.get("why") or row.get("lesson") or "").strip()
        if context or mechanism:
            line = f"- {context}: {mechanism}"
            if why:
                line += f" — {why}"
            lines.append(line)
    if not lines:
        return ""
    return "Directions that already failed (do not repeat them):\n" + "\n".join(lines) + "\n"


def _jump_block(notes: tuple[str, ...]) -> str:
    """Reasoning chain of this landing. Empty keeps cached prompts identical."""
    lines = [str(row).strip() for row in notes if str(row).strip()]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _struggle_memory_block(rows: tuple[str, ...], limit: int = 8) -> str:
    """Routes already tried this mission. Not evidence and not another idea's H.

    Empty keeps cached prompts byte-identical.
    """
    lines = [str(row).strip() for row in rows if str(row).strip()][:limit]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _pairings_block(
    viable_pairings: dict[str, tuple[str, ...]] | None,
    far_labels: tuple[str, ...],
    object_labels: tuple[str, ...],
) -> str:
    """Only pairings the parse-time checks would accept.

    A partner outside the object labels is never offered: telling the
    model "pair with X" while the object lock refuses X is a guaranteed
    refusal, so such pairings are dropped here (the screen already
    prescreens them out on the mission path).
    """
    if not viable_pairings:
        return ""
    obj = set(object_labels)
    lines = []
    for far_label, partners in viable_pairings.items():
        if far_label not in far_labels or not partners:
            continue
        offered = tuple(item for item in partners if item in obj) if obj else tuple(partners)
        if not offered:
            continue
        lines.append(f"- {far_label}: may pair with {', '.join(offered)}")
    if not lines:
        return ""
    lock = ""
    if obj:
        lock = (
            "The first pair member MUST be one of the topic-object concepts"
            f" ({', '.join(object_labels)}). Distant concepts do not rename"
            " the object.\n"
        )
    return (
        "Allowed pairings — pair[0] stays the topic object; pair[1] is"
        " a mechanism from this jump. The pair MUST be one of these:\n"
        + "\n".join(lines) + "\n" + lock
    )


def _context_block(context: tuple[str, ...]) -> str:
    """The freshest knowledge, fetched at mission time rather than baked in.

    The corpus pins what "already combined" means; this block is different —
    it grounds the model in what the field published *this month*, so the
    card is written against today's frontier instead of the harvest end.
    Lines arrive from a live feed and are shown as plain titles; they are
    context, not evidence, and nothing in them can pass or fail a check.
    """
    if not context:
        return ""
    return (
        "The newest papers near the seed's field (live feed, for grounding"
        " only):\n" + "\n".join(f"- {line}" for line in context) + "\n"
    )


def _knowledge_block(knowledge: tuple[str, ...]) -> str:
    """What earlier missions established or left open, from the research state.

    Only corroborated findings and named gaps arrive here — a speculative
    card from a past mission never becomes ground for the next one. Gaps
    are targets: a hypothesis aimed at an open question beats another
    random distant landing.
    """
    if not knowledge:
        return ""
    return (
        "Research state for this neighbourhood (established findings and"
        " open questions from earlier missions):\n"
        + "\n".join(f"- {line}" for line in knowledge)
        + "\n"
    )


def _criteria_block(criteria: tuple[str, ...]) -> str:
    """The installed judges' standards, shown to the author before writing.

    This is the feedback loop DESIGN_V2_ZH.md leaves open after P4: the text
    predicates that will judge the card are executable and public, so hiding
    them from the generator tests the model's luck, not its judgment. A
    scientist reads the review criteria before submitting.
    """
    if not criteria:
        return ""
    return (
        "Installed reviewers will run on the card's text; a card failing any"
        " standard is killed:\n"
        + "\n".join(f"- {line}" for line in criteria)
    )


def _value_block(lines: tuple[str, ...]) -> str:
    """Process criteria cannot kill. Empty keeps the prompt byte-identical."""
    if not lines:
        return ""
    return (
        "\nAfter the executable gates, live cards are ordered by attested"
        " evidence class (WORLD discriminant, literature-gap aim, host"
        " execute), not by a novelty debate. These lines cannot kill a"
        " card. Aim at:\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n"
    )


def _parse(completion: Completion, operator: str) -> dict[str, Any]:
    text = completion.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _refuse(
            f"parse a JSON card for the {operator} jump",
            f"the model answers with one JSON object; it answered with {text[:80]!r} ({exc})",
        ) from exc
    if not isinstance(payload, dict):
        raise _refuse(
            f"read a JSON object for the {operator} jump",
            f"the top-level value is an object, not {type(payload).__name__}",
        )
    return payload


def _token_jaccard(left: str, right: str) -> float:
    a = content_tokens(left)
    b = content_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def parse_struggle(
    payload: dict[str, Any],
    *,
    claim: str,
    mechanism: str,
    refuse,
) -> dict[str, str]:
    """Optional far-jump provenance. Not a universal validity ritual.

    Absent fields are legal. Present fields still have to be honest: a
    dead end that restates the claim, or a reframe that never lands, is
    refused. That is provenance quality, not a requirement to invent a
    failed method before a question may exist.
    """
    values: dict[str, str] = {}
    present = 0
    for name in STRUGGLE_FIELDS:
        text = str(payload.get(name) or "").strip()
        words = [word for word in text.split() if word]
        if not words:
            values[name] = ""
            continue
        if len(words) < MIN_STRUGGLE_WORDS:
            values[name] = ""
            continue
        values[name] = text
        present += 1
    if present == 0:
        return values
    if values.get("dead_end") and _token_jaccard(values["dead_end"], claim) >= DEAD_END_CLAIM_OVERLAP:
        raise refuse(
            "keep the dead_end distinct from the surviving claim",
            "the dead_end names the approach that failed, not a restatement"
            " of the claim",
        )
    if values.get("dead_end") and _token_jaccard(values["dead_end"], mechanism) >= DEAD_END_CLAIM_OVERLAP:
        raise refuse(
            "keep the surviving mechanism off the dead end",
            "the mechanism must not restate the discarded approach",
        )
    if values.get("reframe"):
        landed = content_tokens(values["reframe"]) & (
            content_tokens(claim) | content_tokens(mechanism)
        )
        if len(landed) < REFRAME_BODY_OVERLAP:
            raise refuse(
                "land the reframe in the surviving claim",
                "at least two content words of the reframe appear in the claim"
                " or the mechanism; a diary that never touches the card is void",
            )
    return values


def generate_card(
    client: Any,
    *,
    seed_label: str,
    near_labels: tuple[str, ...],
    far_labels: tuple[str, ...],
    operator: str,
    alienness: float,
    label_to_node: dict[str, str],
    capsules: list[dict[str, Any]] | None = None,
    logprobs: bool = False,
    criteria: tuple[str, ...] = (),
    context: tuple[str, ...] = (),
    knowledge: tuple[str, ...] = (),
    viable_pairings: dict[str, tuple[str, ...]] | None = None,
    value_criteria: tuple[str, ...] = (),
    topic: str = "",
    skills: str = "",
    program: tuple[str, ...] = (),
    wiki: tuple[str, ...] = (),
    object_labels: tuple[str, ...] = (),
    prior_struggle: tuple[str, ...] = (),
    world_menu: Any = None,
    used_levers: tuple[str, ...] = (),
    blocked_handles: tuple[str, ...] = (),
    jump_notes: tuple[str, ...] = (),
    scientific: dict[str, Any] | None = None,
) -> GeneratedCard:
    """One card from one jump. Refuses loudly; never repairs the model's answer.

    Verbatim copying is the schema's teeth: the pair must be two strings the
    model was actually shown, one per side, so every card is executable
    against the graph by construction and a model that paraphrases is
    refused rather than fuzzily matched.

    `viable_pairings` maps each far concept to the near-side concepts that
    survive every graph gate with it — computed before this call, because
    the gates are pure graph properties. When given, the pairings are shown
    to the model and a pair outside them is refused: a live mission showed
    that offering a far concept without naming its viable partners just
    moves the waste from the menu to the model's habit of picking the seed.

    `object_labels` are the near concepts that still name the topic. When
    non-empty, the first pair member must be one of them: otherwise the
    distant landing silently becomes the scientific object.

    `prior_struggle` is the exploration notebook: pairs already tried this
    mission and how they finished. It is not another idea's program.

    When `world_menu` declares levers, the distant concept must compile
    onto one of them (`world_lever`, `world_observable`, `far_maps_to_lever`).
    A weld failure drops that far label and retries the rest of this
    landing; exhausting the table is the caller's cue to switch operator.
    """
    remaining = [str(item) for item in far_labels if str(item).strip()]
    if not remaining:
        remaining = [LOCAL_ORIGIN_LABEL]
    world_menu = _bind_menu(world_menu, blocked_handles)
    forced_handle = False
    forced_schema = False
    retry_pending = False
    last: GenerationRefused | None = None
    while remaining:
        try:
            if retry_pending:
                retry_pending = False
                try:
                    return _draft_card(
                        client,
                        seed_label=seed_label,
                        near_labels=near_labels,
                        far_labels=tuple(remaining),
                        operator=operator,
                        alienness=alienness,
                        label_to_node=label_to_node,
                        capsules=capsules,
                        logprobs=logprobs,
                        criteria=criteria,
                        context=context,
                        knowledge=knowledge,
                        viable_pairings=viable_pairings,
                        value_criteria=value_criteria,
                        topic=topic,
                        skills=skills,
                        program=program,
                        wiki=wiki,
                        object_labels=object_labels,
                        prior_struggle=prior_struggle,
                        world_menu=world_menu,
                        used_levers=used_levers,
                        jump_notes=jump_notes,
                        scientific=scientific,
                    )
                except LLMUnavailable:
                    # The one retry could not be answered; the original
                    # refusal is the truthful record, not a transport error.
                    if last is not None:
                        raise last from None
                    raise
            return _draft_card(
                client,
                seed_label=seed_label,
                near_labels=near_labels,
                far_labels=tuple(remaining),
                operator=operator,
                alienness=alienness,
                label_to_node=label_to_node,
                capsules=capsules,
                logprobs=logprobs,
                criteria=criteria,
                context=context,
                knowledge=knowledge,
                viable_pairings=viable_pairings,
                value_criteria=value_criteria,
                topic=topic,
                skills=skills,
                program=program,
                wiki=wiki,
                object_labels=object_labels,
                prior_struggle=prior_struggle,
                world_menu=world_menu,
                used_levers=used_levers,
                jump_notes=jump_notes,
                scientific=scientific,
            )
        except GenerationRefused as exc:
            last = exc
            far = str(exc.failed_far or "").strip()
            if (
                getattr(exc, "used_handle", False)
                and world_menu is not None
                and not forced_handle
            ):
                # Same labels, hard constraint: the model reused a handle
                # although the menu said which ones were free. One retry
                # with the list made non-negotiable; a second reuse drops
                # the label as before.
                forced_handle = True
                retry_pending = True
                allowed = ", ".join(world_menu.unused_handles(tuple(used_levers))) or "(none)"
                jump_notes = tuple(jump_notes) + (
                    "YOUR PREVIOUS ANSWER WAS REFUSED: it compiled onto a lever/observable "
                    f"handle already used this mission. world_lever/world_observable MUST "
                    f"be one of: {allowed}. Any other handle is refused again.",
                )
                continue
            if (
                exc.record.missing_capability == "model_card_schema"
                and not forced_schema
            ):
                # An empty or malformed field is a one-time slip, not a dead
                # landing: say what was missing and ask once more.
                forced_schema = True
                retry_pending = True
                jump_notes = tuple(jump_notes) + (
                    "YOUR PREVIOUS ANSWER WAS REFUSED: "
                    + str(exc.record.unlock_condition)[:300]
                    + ". Fill every field of the JSON schema; an empty claim is refused.",
                )
                continue
            if (
                world_menu is not None
                and getattr(world_menu, "levers", ())
                and far
                and far in remaining
                and exc.record.missing_capability == "model_far_compile"
            ):
                remaining = [item for item in remaining if item != far]
                continue
            raise
    if last is not None:
        raise last
    raise _refuse(
        f"offer the model the {operator} jump's retrieved concepts",
        "the jump or the verified papers offered at least one distant concept",
        capability="jump_retrieved_concepts",
    )


def _compiled_claim_spec(
    *,
    claim_id: str,
    claim: str,
    mechanism: str,
    world_lever: str,
    world_observable: str,
    target_object: str,
    world_menu: Any,
) -> dict[str, Any] | None:
    """Persist ClaimSpec plus compile metadata. Never a TypecheckResult wrapper."""
    if world_menu is None or not getattr(world_menu, "levers", ()):
        return None
    from .claimspec import WorldObjectRegistry, admit_generated_card, compile_payload

    return compile_payload(
        admit_generated_card(
            claim_id=claim_id,
            statement=claim,
            mechanism_name=mechanism,
            handle_id=world_lever,
            measurement_dv=world_observable,
            target_object=target_object,
            world=WorldObjectRegistry.from_menu(world_menu),
        )
    )


def _draft_card(
    client: Any,
    *,
    seed_label: str,
    near_labels: tuple[str, ...],
    far_labels: tuple[str, ...],
    operator: str,
    alienness: float,
    label_to_node: dict[str, str],
    capsules: list[dict[str, Any]] | None = None,
    logprobs: bool = False,
    criteria: tuple[str, ...] = (),
    context: tuple[str, ...] = (),
    knowledge: tuple[str, ...] = (),
    viable_pairings: dict[str, tuple[str, ...]] | None = None,
    value_criteria: tuple[str, ...] = (),
    topic: str = "",
    skills: str = "",
    program: tuple[str, ...] = (),
    wiki: tuple[str, ...] = (),
    object_labels: tuple[str, ...] = (),
    prior_struggle: tuple[str, ...] = (),
    world_menu: Any = None,
    used_levers: tuple[str, ...] = (),
    jump_notes: tuple[str, ...] = (),
    scientific: dict[str, Any] | None = None,
) -> GeneratedCard:
    if not far_labels:
        far_labels = (LOCAL_ORIGIN_LABEL,)
    pairings_block = _pairings_block(viable_pairings, far_labels, object_labels)
    compile_block = ""
    compile_json = ""
    if world_menu is not None and getattr(world_menu, "levers", ()):
        compile_block = "\n" + world_menu.prompt_block(used_levers) + "\n"
        compile_json = (
            ',\n "world_lever": "<one declared lever>",\n'
            ' "world_observable": "<one declared observable this lever moves>",\n'
            ' "far_maps_to_lever": "<=40 words: how the distant concept becomes that lever"'
        )
    prompt = TEMPLATE.format(
        topic_block=topic_prompt_block(topic),
        skills=skills,
        seed=seed_label,
        near=", ".join(near_labels) or seed_label,
        far=", ".join(far_labels),
        operator=operator,
        pairings=pairings_block,
        jump_block=_jump_block(tuple(jump_notes)),
        context=_context_block(tuple(context)),
        knowledge=_knowledge_block(tuple(knowledge)),
        wiki=wiki_block(tuple(wiki)),
        program=program_block(tuple(program)),
        failures=_failures_block(list(capsules or [])),
        struggle_memory=_struggle_memory_block(tuple(prior_struggle)),
        criteria=_criteria_block(tuple(criteria)),
        value=_value_block(tuple(value_criteria)),
        compile_block=compile_block,
        compile_json=compile_json,
        checks=", ".join(CHECK_CATALOGUE),
    )
    completion = complete_call(
        client,
        prompt,
        purpose=f"far_card_generation:{operator}",
        system=SYSTEM,
        logprobs=logprobs,
        scientific=scientific,
    )
    completion.assert_usable()
    payload = _parse(completion, operator)

    claim = str(payload.get("claim") or "").strip()
    mechanism = str(payload.get("mechanism") or "").strip()
    prediction = str(payload.get("prediction") or "").strip()
    for name, value in (("claim", claim), ("mechanism", mechanism), ("prediction", prediction)):
        if not value:
            raise _refuse(
                f"read a non-empty {name} for the {operator} jump",
                f"the model fills every schema field; {name} was empty",
            )

    raw_pair = payload.get("pair")
    if not isinstance(raw_pair, list) or len(raw_pair) != 2:
        raise _refuse(
            f"read a two-concept pair for the {operator} jump",
            "pair is a list of exactly two concepts copied from the prompt",
        )
    first, second = (str(item).strip() for item in raw_pair)
    near_side = {seed_label, *near_labels}
    if first not in near_side:
        raise _refuse(
            f"map {first!r} onto the seed's own field",
            "the first pair member is the seed or one of the near concepts,"
            " copied exactly as written",
        )
    if object_labels and first not in set(object_labels):
        raise _refuse(
            f"keep {first!r} as a topic-object concept",
            "the first pair member names the topic's scientific object,"
            " not a cosine-nearest neighbour the distant landing dragged in",
            capability="model_claim_on_topic",
        )
    allowed_far = set(far_labels) | {LOCAL_ORIGIN_LABEL}
    if second not in allowed_far:
        raise _refuse(
            f"map {second!r} onto the jump's retrieved concepts",
            "the second pair member is one of the distant concepts, copied"
            " exactly as written, or 'local problem structure' when no"
            " verified far literature exists",
        )
    if is_local_origin_label(second) and any(
        item for item in far_labels if item and not is_local_origin_label(item)
    ):
        raise _refuse(
            "keep a literature far when one was retrieved",
            "local problem structure is only legal when the jump offered no"
            " verified distant mechanism; do not drop a retrieved far concept",
        )
    if first == second:
        raise _refuse(
            "read a pair of two distinct concepts",
            "a concept combined with itself is not a combination",
        )
    if viable_pairings is not None:
        partners = viable_pairings.get(second) or ()
        if is_local_origin_label(second) and not partners:
            partners = tuple(item for item in (object_labels or near_side) if item)
        if first not in partners:
            raise _refuse(
                f"pair {first!r} with {second!r}",
                "the pair is one of the structurally viable pairings listed"
                " in the prompt; the concept graph already killed that"
                " combination",
            )

    falsifier = str(payload.get("falsifier") or "").strip()
    if falsifier not in CHECK_CATALOGUE:
        raise _refuse(
            f"map {falsifier!r} onto an executable check",
            "the falsifier is exactly one of "
            + ", ".join(CHECK_CATALOGUE)
            + "; an unrunnable falsifier is not a falsifier",
            capability="model_named_an_executable_check",
        )

    struggle = parse_struggle(
        payload, claim=claim, mechanism=mechanism, refuse=_refuse
    )

    locked = refuse_claim(claim, mechanism, topic, seed_label)
    if locked:
        raise _refuse(
            f"keep the {operator} claim inside the researcher's topic",
            locked,
            capability="model_claim_on_topic",
        )

    idea_kind = parse_idea_kind(payload.get("idea_kind"))
    origin = parse_origin(payload.get("origin"), far_labels=far_labels)
    if is_local_origin_label(second):
        origin = ORIGIN_LOCAL
    derivation = _optional_derivation(payload.get("derivation"))
    if idea_kind == THEORY and not derivation:
        raise _refuse(
            "read a derivation for a theory idea",
            "idea_kind=theory must fill derivation with a sketch locked to "
            "registered quantities; empty theory is a diary",
        )
    kind_fields = parse_kind_fields(idea_kind, payload)
    world_lever = str(payload.get("world_lever") or "").strip()
    world_observable = str(payload.get("world_observable") or "").strip()
    far_maps = str(payload.get("far_maps_to_lever") or "").strip()
    overclaim = causal_overclaim_reason(
        idea_kind, claim=claim, lever=world_lever, prediction=prediction
    )
    if overclaim:
        raise _refuse(
            "keep an observational question observational",
            overclaim,
            capability="model_causal_overclaim",
        )
    should_compile = (
        world_menu is not None
        and getattr(world_menu, "levers", ())
        and (compile_required(idea_kind) or bool(world_lever))
    )
    if should_compile:
        from .farcompile import assert_compile

        world_lever, world_observable, far_maps = assert_compile(
            far_label=second,
            world_lever=world_lever,
            world_observable=world_observable,
            far_maps_to_lever=far_maps,
            prediction=prediction,
            menu=world_menu,
            used_levers=used_levers,
            refuse=_refuse,
            idea_kind=idea_kind,
            origin=origin,
        )
    if idea_kind == ACQUIRE and not (
        kind_fields.get("recipe") and kind_fields.get("validators")
    ):
        if not kind_fields.get("recipe"):
            kind_fields["recipe"] = prediction
        if not kind_fields.get("validators"):
            kind_fields["validators"] = str(
                payload.get("required_world_state") or ""
            ).strip()

    claim_spec = _compiled_claim_spec(
        claim_id=f"gen_{completion.digest[:12]}",
        claim=claim,
        mechanism=mechanism,
        world_lever=world_lever,
        world_observable=world_observable,
        target_object=first,
        world_menu=world_menu,
    )

    return GeneratedCard(
        card_id=f"gen_{completion.digest[:12]}",
        operator=operator,
        claim=claim,
        mechanism=mechanism,
        prediction=prediction,
        falsifier=falsifier,
        pair=(first, second),
        pair_nodes=(
            pair_node(first, label_to_node),
            pair_node(second, label_to_node),
        ),
        alienness=alienness,
        model=completion.model,
        artifact_digest=completion.digest,
        artifact_uri=completion.artifact_uri,
        replay_mode=completion.mode,
        dead_end=struggle["dead_end"],
        why_failed=struggle["why_failed"],
        reframe=struggle["reframe"],
        objection=struggle["objection"],
        derivation=derivation,
        world_lever=world_lever,
        world_observable=world_observable,
        far_maps_to_lever=far_maps,
        idea_kind=idea_kind,
        mean_nll=getattr(completion, "mean_nll", None),
        claim_spec=claim_spec,
        lineage_id=lineage_id_for((first, second), origin=origin),
        origin=origin,
        research_target=str(payload.get("research_target") or claim).strip(),
        world_requirements=str(payload.get("required_world_state") or "").strip(),
        transition_reason=str(payload.get("transition_reason") or "").strip(),
        kind_fields=kind_fields,
    )


def refine_card(
    client: Any,
    parent: GeneratedCard,
    *,
    seed_label: str,
    near_labels: tuple[str, ...],
    label_to_node: dict[str, str],
    killed_by: list[str] | None = None,
    bottleneck: str = "",
    capsules: list[dict[str, Any]] | None = None,
    logprobs: bool = False,
    criteria: tuple[str, ...] = (),
    context: tuple[str, ...] = (),
    knowledge: tuple[str, ...] = (),
    value_criteria: tuple[str, ...] = (),
    topic: str = "",
    skills: str = "",
    program: tuple[str, ...] = (),
    wiki: tuple[str, ...] = (),
    object_labels: tuple[str, ...] = (),
    prior_struggle: tuple[str, ...] = (),
    world_menu: Any = None,
    used_levers: tuple[str, ...] = (),
    blocked_handles: tuple[str, ...] = (),
    duty: str = "gate",
    keep_id: bool = False,
    compile_notes: list[str] | tuple[str, ...] = (),
    scientific: dict[str, Any] | None = None,
) -> GeneratedCard:
    """Rewrite the same idea in this mission. Pair is locked.

    Text-gate kills feed this call. A world gap constructs the experiment's
    world; it does not rewrite the claim to fit a freeze catalog. A new
    distant concept is a failed refine, not exploration. A no_handle
    death recompiles the same pair onto the scouted lever table.
    `duty=compile` is the post-diagnosis weld: mechanism, lever, competing
    explanation, and prediction may move; the claim object may not.
    """
    killers = ", ".join(str(item) for item in (killed_by or [])[:6])
    compile_hint = ""
    if "no_handle" in {str(item) for item in (killed_by or [])}:
        compile_hint = (
            " The object is already bound. Compile the distant concept "
            "onto a declared lever that is not already used. Do not paste "
            "the far noun into the title and do not rebind a foreign freeze."
        )
    notes = [str(item).strip() for item in compile_notes if str(item).strip()]
    if duty == "compile":
        extra = (
            f"COMPILE THIS idea. Keep pair {parent.pair[0]} × {parent.pair[1]} exactly.",
            f"Previous claim: {parent.claim}",
            f"Previous mechanism: {parent.mechanism}",
            f"Previous prediction: {parent.prediction}",
            "Do not rename the scientific object, pick a new distant concept,"
            " or rebind the freeze. Weld the mechanism onto an unused"
            " declared lever if one exists. Name a competing explanation"
            " that is another declared lever. The prediction must stay a"
            " measurable attested field.",
            (
                "Structural holes from idea rehearsal: " + "; ".join(notes[:4])
                if notes
                else "No structural hole was named; tighten the weld and the competing explanation."
            ),
            "Change the mechanism, the lever weld, or the prediction."
            " Copying the previous draft is not an iteration.",
        )
    else:
        extra = (
            f"Revise THIS idea. Keep pair {parent.pair[0]} × {parent.pair[1]} exactly.",
            f"Previous claim: {parent.claim}",
            f"Previous prediction: {parent.prediction}",
            f"Why it died: {killers or bottleneck or 'unspecified bottleneck'}.",
            "Change the claim or the prediction so that failure cannot recur. "
            "Do not pick a new distant concept."
            + compile_hint,
        )
        if parent.world_lever and parent.world_lever != "none":
            extra = extra + (
                f"Keep world_lever={parent.world_lever} and world_observable="
                f"{parent.world_observable or '(same)'}. A gate refine tightens "
                "identification on the compiled handle; switching to dropout or "
                "another lever is refused.",
            )
    reserved = (
        used_levers
        if duty == "compile" or "no_handle" in {str(item) for item in (killed_by or [])}
        else _without_parent_handle(used_levers, parent)
    )
    card = generate_card(
        client,
        seed_label=seed_label,
        near_labels=near_labels,
        far_labels=(parent.pair[1],),
        operator=parent.operator,
        alienness=parent.alienness,
        label_to_node=label_to_node,
        capsules=capsules,
        logprobs=logprobs,
        criteria=criteria,
        context=context,
        knowledge=tuple(knowledge) + extra,
        viable_pairings={parent.pair[1]: (parent.pair[0],)},
        value_criteria=value_criteria,
        topic=topic,
        skills=skills,
        program=program,
        wiki=wiki,
        object_labels=(parent.pair[0],),
        prior_struggle=prior_struggle,
        world_menu=world_menu,
        used_levers=reserved,
        blocked_handles=blocked_handles,
        scientific=scientific,
    )
    if card.pair != parent.pair:
        raise _refuse(
            "keep the same concept pair while evolving this idea",
            "a refine that lands on a new distant concept is spray, not evolution",
        )
    same_text = (
        card.claim.strip() == parent.claim.strip()
        and card.prediction.strip() == parent.prediction.strip()
        and card.mechanism.strip() == parent.mechanism.strip()
    )
    same_lever = (card.world_lever or "") == (parent.world_lever or "")
    if same_text and same_lever:
        raise _refuse(
            "change the claim, the mechanism, or the prediction in this refine",
            "copying the previous draft is not an iteration",
        )
    # A gate refine rewrites the claim; it does not re-weld the pair. The
    # parent's compiled handle stays the registration unless the refine's
    # duty is `compile` (a lever switch). v4: a contrast card was refined
    # onto `dropout` and spent three blind probes on the wrong handle.
    if (
        duty != "compile"
        and parent.world_lever
        and parent.world_lever != "none"
        and card.world_lever
        and card.world_lever != parent.world_lever
    ):
        card = replace(
            card,
            world_lever=parent.world_lever,
            world_observable=parent.world_observable or card.world_observable,
        )
    if duty != "compile" and parent.idea_kind and card.idea_kind != parent.idea_kind:
        reason = str(getattr(card, "transition_reason", "") or "").strip()
        if not reason:
            reason = infer_transition_reason(
                parent.idea_kind,
                card.idea_kind,
                killed_by=killed_by or [],
            )
        locked = refuse_kind_transition(parent.idea_kind, card.idea_kind, reason)
        if locked:
            raise _refuse(
                "change idea_kind only on an admitted transition",
                locked,
                capability="model_kind_transition",
            )
        card = replace(
            card,
            transition_reason=reason,
            parent_idea_id=parent.card_id,
            lineage_id=parent.lineage_id or card.lineage_id,
            origin=parent.origin or card.origin,
        )
    else:
        card = replace(
            card,
            lineage_id=parent.lineage_id or card.lineage_id,
            parent_idea_id=parent.card_id,
            origin=parent.origin or card.origin,
        )
    if duty != "compile" and _abandons_observational_handle(parent, card):
        raise _refuse(
            "keep the compiled observational handle in the claim",
            "a gate refine that rewrites an observational contrast into an "
            "intervention (dropout / mask) abandons a promising direction; "
            "tighten matching, do not switch the lever",
            capability="model_far_compile",
        )
    if keep_id:
        card = replace(card, card_id=parent.card_id)
    admitted = _compiled_claim_spec(
        claim_id=card.card_id,
        claim=card.claim,
        mechanism=card.mechanism,
        world_lever=card.world_lever,
        world_observable=card.world_observable,
        target_object=card.pair[0] if card.pair else "",
        world_menu=world_menu,
    )
    if admitted is not None:
        card = replace(card, claim_spec=admitted)
    return card


# --------------------------------------------------------------------------
# The graph oracle. Pre-T structure only; nothing here reads a year.


@dataclass(frozen=True)
class PairCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class ConceptOracle:
    """Precomputed corpus facts the three checks are read against.

    Both ceilings are quantiles of *this* snapshot, computed once and pinned
    here, so the same card gets the same verdict on every replay and a
    different corpus honestly gets different ceilings.
    """

    neighbors: dict[str, frozenset[str]]
    degree_ceiling: int
    implication_ceiling: float

    @classmethod
    def from_graph(cls, graph: GraphSnapshot) -> "ConceptOracle":
        neighbors: dict[str, set[str]] = {node: set() for node in graph.nodes}
        for src, dst in graph.edges:
            neighbors[src].add(dst)
            neighbors[dst].add(src)
        frozen = {node: frozenset(peers) for node, peers in neighbors.items()}
        degrees = sorted(len(peers) for peers in frozen.values())
        degree_ceiling = degrees[int(HUB_QUANTILE * (len(degrees) - 1))]
        rng = random.Random(IMPLICATION_RNG_SEED)
        nodes = sorted(frozen)
        overlaps: list[float] = []
        while len(overlaps) < IMPLICATION_SAMPLE and len(nodes) >= 2:
            u, v = rng.sample(nodes, 2)
            if v in frozen[u]:
                continue
            union = len(frozen[u] | frozen[v])
            overlaps.append(len(frozen[u] & frozen[v]) / union if union else 0.0)
        overlaps.sort()
        implication_ceiling = (
            overlaps[int(IMPLICATION_QUANTILE * (len(overlaps) - 1))]
            if overlaps
            else 0.0
        )
        return cls(
            neighbors=frozen,
            degree_ceiling=max(degree_ceiling, 1),
            implication_ceiling=implication_ceiling,
        )

    def linked(self, u: str, v: str) -> bool:
        return v in self.neighbors.get(u, frozenset())

    def jaccard(self, u: str, v: str) -> float:
        a = self.neighbors.get(u, frozenset())
        b = self.neighbors.get(v, frozenset())
        union = len(a | b)
        return len(a & b) / union if union else 0.0


def pair_node(label: str, label_to_node: dict[str, str]) -> str:
    """Graph id when the label is a corpus node; else a literature endpoint.

    Mapping an off-graph mechanism onto the cosine-nearest graph node is
    how a paper-named method became `feedback edge set`. The oracle must
    see that this end is not a graph fact. Callers that landed via a
    research trajectory must pass `literature_labels` so a coincidental
    catalog match (`wavelet tree`) stays a literature endpoint.
    """
    if is_local_origin_label(label):
        return local_node_id(label)
    node = str(label_to_node.get(label) or "").strip()
    if node and not node.startswith(LITERATURE_NODE_PREFIX) and not node.startswith(LOCAL_NODE_PREFIX):
        return node
    slug = "-".join(str(label or "").lower().split()) or "unnamed"
    return f"{LITERATURE_NODE_PREFIX}{slug}"


def literature_labels(
    base: dict[str, str], phrases: Sequence[str]
) -> dict[str, str]:
    """Stamp trajectory far phrases as literature endpoints.

    A paper method that happens to be a graph label still is not a graph
    fact. Overwriting the catalog id is the point: otherwise a hub on
    pair[0] kills every landing whose far words already exist in cs.DS.
    """
    labels = dict(base)
    for phrase in phrases:
        text = str(phrase or "").strip()
        if text:
            labels[text] = f"{LITERATURE_NODE_PREFIX}{text}"
    return labels


def graph_endpoint(oracle: Any, node: str) -> bool:
    """True when this id is a node the oracle was built from.

    An empty neighbors table is a test stub that still answers `linked`;
    treat those ids as graph endpoints. A missing or `literature:` id is
    not — the oracle has no right to kill it.
    """
    if (
        not node
        or str(node).startswith(LITERATURE_NODE_PREFIX)
        or str(node).startswith(LOCAL_NODE_PREFIX)
    ):
        return False
    neighbors = getattr(oracle, "neighbors", None)
    if not neighbors:
        return True
    return node in neighbors


def check_pair(oracle: ConceptOracle, u: str, v: str) -> tuple[PairCheck, ...]:
    """All three checks, always all run: a card should know every way it died.

    When either end is not a graph node, every check passes with an
    explicit 'not a graph pair' detail. Silence here is the design:
    novelty for a literature mechanism is prior-kill, not a missing edge
    on a cs.DS snapshot.
    """
    if not graph_endpoint(oracle, u) or not graph_endpoint(oracle, v):
        detail = (
            "not a graph pair; the oracle is silent and novelty is prior-kill"
        )
        return (
            PairCheck("pair_not_already_combined", True, detail),
            PairCheck("pair_not_implied_by_neighborhood", True, detail),
            PairCheck("endpoint_is_not_a_concept_hub", True, detail),
        )
    linked = oracle.linked(u, v)
    overlap = oracle.jaccard(u, v)
    degree = max(
        len(oracle.neighbors.get(u, frozenset())),
        len(oracle.neighbors.get(v, frozenset())),
    )
    return (
        PairCheck(
            name="pair_not_already_combined",
            passed=not linked,
            detail=(
                "no pre-T work carries both labels"
                if not linked
                else "the pair already co-occurs pre-T; the combination exists"
            ),
        ),
        PairCheck(
            name="pair_not_implied_by_neighborhood",
            passed=overlap <= oracle.implication_ceiling,
            detail=(
                f"neighborhood Jaccard {overlap:.4f} vs ceiling"
                f" {oracle.implication_ceiling:.4f}"
                + (
                    ""
                    if overlap <= oracle.implication_ceiling
                    else "; the contexts already overlap enough that the"
                    " combination is the neighbourhood's idea, not the card's"
                )
            ),
        ),
        PairCheck(
            name="endpoint_is_not_a_concept_hub",
            passed=degree <= oracle.degree_ceiling,
            detail=(
                f"max degree {degree} vs ceiling {oracle.degree_ceiling}"
                + (
                    ""
                    if degree <= oracle.degree_ceiling
                    else "; a concept the whole corpus uses combines with"
                    " anything, so this pairing is a fact about the graph"
                )
            ),
        ),
    )


@dataclass(frozen=True)
class PairVerdict:
    card_id: str
    pair_nodes: tuple[str, str]
    checks: tuple[PairCheck, ...]
    preregistered_falsifier: str

    @property
    def killed(self) -> bool:
        return any(not check.passed for check in self.checks)

    @property
    def killed_by(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed)

    @property
    def falsifier_was_right(self) -> bool:
        """Did the card die exactly where its author pre-registered?"""
        return self.preregistered_falsifier in self.killed_by

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "pair_nodes": list(self.pair_nodes),
            "killed": self.killed,
            "killed_by": list(self.killed_by),
            "preregistered_falsifier": self.preregistered_falsifier,
            "falsifier_was_right": self.falsifier_was_right,
            "checks": [check.to_dict() for check in self.checks],
        }


def probe_generated(
    oracle: ConceptOracle, cards: list[GeneratedCard]
) -> list[PairVerdict]:
    return [
        PairVerdict(
            card_id=card.card_id,
            pair_nodes=card.pair_nodes,
            checks=check_pair(oracle, *card.pair_nodes),
            preregistered_falsifier=card.falsifier,
        )
        for card in cards
    ]
