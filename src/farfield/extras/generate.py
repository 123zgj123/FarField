"""LLM-generated hypothesis cards over the concept network (v2 P1).

DESIGN_V2_ZH.md §3 step [2]: generation is the model's job now, not an
annotation on a graph walk. The model reads the seed, the concepts a P0
far-field jump retrieved, and the failure capsules, and answers in a strong
schema — claim, mechanism, a concept pair copied verbatim from what it was
shown, a pre-registered falsifier chosen from the executable catalogue, and
a testable prediction. Anything off-schema is refused and recorded; the
refusal rate is a P1 exit criterion, so a refusal here is a reading, not an
error.

The model's text never becomes evidence.
The only parts of a card with causal consequences are executable: the pair,
which the graph oracle checks, and the falsifier name, which is recorded so
the card's author is on the hook for where it expected to die.

The graph oracle is the killing floor. Its three checks were sized on the
real corpus before being adopted, because a check that cannot pass or
cannot fail is a ceremony:

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
from dataclasses import dataclass
from typing import Any

from ..graph import GraphSnapshot
from ..models import BlockedRecord
from .domain import topic_prompt_block
from .plugins import refuse_claim
from .program import block as program_block
from .wiki import block as wiki_block
from .llm import Completion, LLMUnavailable
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

SYSTEM = (
    "You are generating one falsifiable research hypothesis."
    " Think through a failed approach before you name the survivor."
    " Answer with one JSON object and nothing else."
)

TEMPLATE = """{topic_block}{skills}A research seed and a set of distant concepts reached by a controlled jump in embedding space.

Seed concept: {seed}
Concepts in the seed's own field: {near}
Distant concepts retrieved at the jump landing (operator: {operator}): {far}
{pairings}{context}{knowledge}{wiki}{program}{failures}{criteria}{value}
Propose ONE falsifiable hypothesis that combines the seed's field with exactly one of the distant concepts. Do not merely apply the first concept to the second: the mechanism must name a third technical idea (a construction, a bound, or a measurement) that would disappear if you only juxtaposed the two names.

Think on the card, not in a diary. Name the obvious approach that fails, why it fails for a non-trivial reason, the assumption you drop, and the strongest objection. Those four fields are unverified model text and are not evidence. The surviving mechanism must not restate the dead end. At least two content words of the reframe must appear in the claim or the mechanism.

Answer with JSON:
{{"dead_end": "<=40 words: the obvious approach that fails",
 "why_failed": "<=40 words: the non-trivial reason it breaks",
 "reframe": "<=40 words: the assumption you drop or the question you replace",
 "objection": "<=40 words: the strongest reviewer objection to the survivor",
 "derivation": "<=80 words: a sketch of why the construction or bound holds (unverified model text, not evidence)",
 "claim": "<=60 words, a falsifiable claim connecting the two concepts",
 "mechanism": "<=80 words on why the connection could hold",
 "pair": ["<the seed or one concept from the seed's field, copied exactly>", "<one distant concept, copied exactly>"],
 "falsifier": "<one of: {checks}>",
 "prediction": "<=30 words: what later published work would show the combination was actually made>"}}

`pair` must copy two concepts exactly as written above: the first from the seed's field, the second from the distant list. `falsifier` must be exactly one of the listed names; it is the executable check you judge this card most at risk of failing. Anything else voids the card.
"""


class GenerationRefused(RuntimeError):
    def __init__(self, record: BlockedRecord):
        super().__init__(record.unlock_condition)
        self.record = record


def _refuse(attempted: str, unlock: str, capability: str = "model_card_schema") -> GenerationRefused:
    return GenerationRefused(
        BlockedRecord(
            missing_capability=capability,
            attempted=attempted,
            unlock_condition=unlock,
        )
    )


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
    # Discarded approach. Required at generation; never evidence, never a
    # ladder input. Empty only on cards built from older fixtures.
    dead_end: str = ""
    why_failed: str = ""
    reframe: str = ""
    objection: str = ""
    derivation: str = ""
    # NLL surprise of the visible completion (plausibility's second term).
    # None when the call did not request logprobs; only written to the dict
    # when present, so reports from before this field existed stay
    # byte-identical on replay.
    mean_nll: float | None = None

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
                "claim/mechanism/prediction and the struggle fields are"
                " unverified model text and are not evidence; only the pair"
                " (executable checks) and the pre-registered falsifier acted"
                " on the campaign"
            ),
        }
        if self.derivation:
            payload["derivation"] = self.derivation
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
        if context or mechanism:
            lines.append(f"- {context}: {mechanism}")
    if not lines:
        return ""
    return "Directions that already failed (do not repeat them):\n" + "\n".join(lines) + "\n"


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
    """Schema teeth for the discarded approach. Not evidence.

    Empty or tiny fields refuse. A dead end that restates the surviving
    claim or mechanism refuses. A reframe that never lands in the claim
    or mechanism refuses — otherwise the struggle is a diary pasted on
    an apply-X-to-Y card.
    """
    values: dict[str, str] = {}
    for name in STRUGGLE_FIELDS:
        text = str(payload.get(name) or "").strip()
        words = [word for word in text.split() if word]
        if len(words) < MIN_STRUGGLE_WORDS:
            raise refuse(
                f"read a non-empty {name}",
                f"the model fills every schema field; {name} needs at least"
                f" {MIN_STRUGGLE_WORDS} words naming a real failed approach",
            )
        values[name] = text
    if _token_jaccard(values["dead_end"], claim) >= DEAD_END_CLAIM_OVERLAP:
        raise refuse(
            "keep the dead_end distinct from the surviving claim",
            "the dead_end names the approach that failed, not a restatement"
            " of the claim",
        )
    if _token_jaccard(values["dead_end"], mechanism) >= DEAD_END_CLAIM_OVERLAP:
        raise refuse(
            "keep the surviving mechanism off the dead end",
            "the mechanism must not restate the discarded approach",
        )
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
    """
    if not far_labels:
        raise _refuse(
            f"offer the model the {operator} jump's retrieved concepts",
            "the jump retrieved at least one concept that maps to a graph node",
            capability="jump_retrieved_concepts",
        )
    pairings_block = ""
    if viable_pairings:
        lines = [
            f"- {far_label}: may pair with {', '.join(partners)}"
            for far_label, partners in viable_pairings.items()
            if far_label in far_labels and partners
        ]
        if lines:
            pairings_block = (
                "Structurally viable pairings — the concept graph has already"
                " ruled out every other combination, so the pair MUST be one"
                " of these:\n" + "\n".join(lines) + "\n"
            )
    prompt = TEMPLATE.format(
        topic_block=topic_prompt_block(topic),
        skills=skills,
        seed=seed_label,
        near=", ".join(near_labels) or seed_label,
        far=", ".join(far_labels),
        operator=operator,
        pairings=pairings_block,
        context=_context_block(tuple(context)),
        knowledge=_knowledge_block(tuple(knowledge)),
        wiki=wiki_block(tuple(wiki)),
        program=program_block(tuple(program)),
        failures=_failures_block(list(capsules or [])),
        criteria=_criteria_block(tuple(criteria)),
        value=_value_block(tuple(value_criteria)),
        checks=", ".join(CHECK_CATALOGUE),
    )
    completion = client.complete(
        prompt,
        purpose=f"far_card_generation:{operator}",
        system=SYSTEM,
        logprobs=logprobs,
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
    if second not in set(far_labels):
        raise _refuse(
            f"map {second!r} onto the jump's retrieved concepts",
            "the second pair member is one of the distant concepts, copied"
            " exactly as written",
        )
    if first == second:
        raise _refuse(
            "read a pair of two distinct concepts",
            "a concept combined with itself is not a combination",
        )
    if viable_pairings is not None:
        partners = viable_pairings.get(second) or ()
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

    return GeneratedCard(
        card_id=f"gen_{completion.digest[:12]}",
        operator=operator,
        claim=claim,
        mechanism=mechanism,
        prediction=prediction,
        falsifier=falsifier,
        pair=(first, second),
        pair_nodes=(label_to_node[first], label_to_node[second]),
        alienness=alienness,
        model=completion.model,
        artifact_digest=completion.digest,
        artifact_uri=completion.artifact_uri,
        replay_mode=completion.mode,
        dead_end=struggle["dead_end"],
        why_failed=struggle["why_failed"],
        reframe=struggle["reframe"],
        objection=struggle["objection"],
        derivation=_optional_derivation(payload.get("derivation")),
        mean_nll=getattr(completion, "mean_nll", None),
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
) -> GeneratedCard:
    """Rewrite the same idea in this mission. Pair is locked.

    Text-gate kills feed this call. A world gap constructs the experiment's
    world; it does not rewrite the claim to fit a freeze catalog. A new
    distant concept is a failed refine, not exploration.
    """
    killers = ", ".join(str(item) for item in (killed_by or [])[:6])
    extra = (
        f"Revise THIS idea. Keep pair {parent.pair[0]} × {parent.pair[1]} exactly.",
        f"Previous claim: {parent.claim}",
        f"Previous prediction: {parent.prediction}",
        f"Why it died: {killers or bottleneck or 'unspecified bottleneck'}.",
        "Change the claim or the prediction so that failure cannot recur. "
        "Do not pick a new distant concept.",
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
    )
    if card.pair != parent.pair:
        raise _refuse(
            "keep the same concept pair while evolving this idea",
            "a refine that lands on a new distant concept is spray, not evolution",
        )
    if (
        card.claim.strip() == parent.claim.strip()
        and card.prediction.strip() == parent.prediction.strip()
    ):
        raise _refuse(
            "change the claim or the prediction in this refine",
            "copying the previous draft is not an iteration",
        )
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


def check_pair(oracle: ConceptOracle, u: str, v: str) -> tuple[PairCheck, ...]:
    """All three checks, always all run: a card should know every way it died."""
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
