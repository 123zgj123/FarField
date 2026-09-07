"""Far-field exploration: trajectory-guided search in research space.

The method is not a concept graph and not a one-shot concept mix.
Verified papers and the user's own statement become development
trajectories. A jump applies an observed evolution pattern to the
current position. Evaluation of support / novelty / value /
feasibility / distance reshapes the next jump.

Embeddings or a concept catalog may store or retrieve phrases. They
do not define the search. Empty literature is an empty menu — cosine
nouns from another field are not a fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .domain import folded_terms, topic_object_phrases
from .livefeed import FreshWork
from .prior import APPLY_PAD, claim_phrases
from .wiki import (
    described_gaps,
    described_methods,
    limitation_sentences,
)

JUMP_MOVES = (
    "drop_assumption",
    "connect_problems",
    "migrate_setting",
    "reframe_limitation",
    "change_objective",
    "explicit_latent",
    "compose_trajectories",
)

DISTANCE_BANDS = ("early", "middle", "far")
FAR_CAP = 6

_METHOD_TAIL = re.compile(
    r"\b(?:with|using|via|by means of|by)\s+(.+)$",
    re.IGNORECASE,
)
_TENSION_SPLIT = re.compile(
    r"\b(?:however|but|without|instead of|rather than|fails? to|"
    r"cannot|open problem|limitation|unlike)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"^(\d{4})")


@dataclass(frozen=True)
class ResearchState:
    """Problem definition / research domain spec. Not the control state.

    Canonical control state is extras.openworld.state.ScientificState.
    This object only frames subject, methods, tensions, and constraints.
    """

    object: tuple[str, ...]
    methods: tuple[str, ...]
    tensions: tuple[str, ...]
    directions: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "object": list(self.object),
            "methods": list(self.methods),
            "tensions": list(self.tensions),
            "directions": list(self.directions),
            "summary": self.summary,
        }


ProblemDefinition = ResearchState


@dataclass(frozen=True)
class TrajectoryStep:
    """One published move along a research line."""

    problem: str
    inherited: str
    mechanism: str
    assumption_changed: str
    limitation_addressed: str
    new_problem: str
    cite_id: str
    published: str
    move: str

    def to_dict(self) -> dict[str, str]:
        return {
            "problem": self.problem,
            "inherited": self.inherited,
            "mechanism": self.mechanism,
            "assumption_changed": self.assumption_changed,
            "limitation_addressed": self.limitation_addressed,
            "new_problem": self.new_problem,
            "cite_id": self.cite_id,
            "published": self.published,
            "move": self.move,
        }


@dataclass(frozen=True)
class ResearchTrajectory:
    """An idea's evolution, not a pile of unrelated paper nodes."""

    lineage: str
    steps: tuple[TrajectoryStep, ...]
    patterns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage": self.lineage,
            "steps": [step.to_dict() for step in self.steps],
            "patterns": list(self.patterns),
        }

    @property
    def latest_mechanism(self) -> str:
        for step in reversed(self.steps):
            if step.mechanism:
                return step.mechanism
        return ""


@dataclass(frozen=True)
class JumpProposal:
    """One trajectory-informed landing in research space."""

    index: int
    move: str
    band: str
    far_labels: tuple[str, ...]
    reasoning: tuple[str, ...]
    support: str
    novelty: str
    patterns: tuple[str, ...]
    shaped_by: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "move": self.move,
            "band": self.band,
            "far": list(self.far_labels),
            "reasoning": list(self.reasoning),
            "support": self.support,
            "novelty": self.novelty,
            "patterns": list(self.patterns),
            "shaped_by": self.shaped_by,
        }

    def prompt_lines(self) -> tuple[str, ...]:
        if not self.reasoning and not self.far_labels:
            return ()
        lines = [
            f"Trajectory-informed jump {self.index} "
            f"(band={self.band}, move={self.move}).",
            "Reasoning chain (existing knowledge → pattern → jump → "
            "candidate). This is not a random concept mix:",
        ]
        lines.extend(f"- {row}" for row in self.reasoning[:8])
        if self.support:
            lines.append(f"Support prior: {self.support}.")
        if self.novelty:
            lines.append(f"Novelty prior: {self.novelty}.")
        return tuple(lines)


@dataclass(frozen=True)
class JumpFeedback:
    """Scores that steer the next jump; they are not a kill."""

    support: str
    novelty: str
    value: str
    feasibility: str
    distance: str
    policy: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        payload = {
            "support": self.support,
            "novelty": self.novelty,
            "value": self.value,
            "feasibility": self.feasibility,
            "distance": self.distance,
            "policy": self.policy,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass
class SearchLedger:
    """Accumulated search state across jumps of one mission."""

    used_far: list[str] = field(default_factory=list)
    feedback: list[dict[str, str]] = field(default_factory=list)
    policy: str = "hold"
    band: str = "early"
    dead_directions: list[str] = field(default_factory=list)

    def absorb(self, feedback: JumpFeedback, *, far: str = "") -> None:
        row = feedback.to_dict()
        if row not in self.feedback:
            self.feedback.append(row)
        self.policy = feedback.policy
        self.band = apply_policy(self.band, feedback.policy)
        label = str(far or "").strip()
        if label and label not in self.used_far:
            self.used_far.append(label)
        if feedback.feasibility in {"blocked", "failed"} and label:
            note = f"{label} [{feedback.policy}] {feedback.detail}".strip()
            if note not in self.dead_directions:
                self.dead_directions.append(note)

    def snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "policy": self.policy,
            "band": self.band,
            "used_far": list(self.used_far),
        }
        if self.feedback:
            payload["feedback"] = list(self.feedback)[:8]
        if self.dead_directions:
            payload["dead_directions"] = list(self.dead_directions)[:8]
        return payload

    def prompt_lines(self) -> tuple[str, ...]:
        if not self.feedback and not self.dead_directions:
            return ()
        lines = [
            "Previous jumps already evaluated. The next landing must use "
            "this feedback; it is not an independent draw:"
        ]
        for row in self.feedback[-4:]:
            lines.append(
                "- support={support}, novelty={novelty}, value={value}, "
                "feasibility={feasibility}, distance={distance} → {policy}".format(
                    **{key: row.get(key) or "" for key in (
                        "support", "novelty", "value", "feasibility",
                        "distance", "policy",
                    )}
                )
            )
            if row.get("detail"):
                lines.append("  " + str(row["detail"])[:200])
        if self.dead_directions:
            lines.append(
                "Directions already negated (do not regenerate): "
                + "; ".join(self.dead_directions[:4])
            )
        return tuple(lines)


def initial_state(topic: str) -> ResearchState:
    """Compile the starting position from the user's idea, before any jump."""
    text = str(topic or "").strip()
    objects = tuple(
        gram for gram in topic_object_phrases(text) if " " in gram
    )[:4]
    if not objects:
        objects = tuple(topic_object_phrases(text)[:3])
    blocked = _blocked(text)
    methods = _method_phrases(text, blocked)
    tensions = _tension_phrases(text, blocked)
    directions = tuple(
        gram
        for gram in claim_phrases(text)
        if gram not in objects
        and gram not in methods
        and gram not in tensions
        and not _blocked_phrase(gram, blocked)
    )[:4]
    bits = []
    if objects:
        bits.append("object: " + ", ".join(objects[:2]))
    if methods:
        bits.append("methods: " + ", ".join(methods[:2]))
    if tensions:
        bits.append("open: " + ", ".join(tensions[:2]))
    return ResearchState(
        object=objects,
        methods=methods,
        tensions=tensions,
        directions=directions,
        summary="; ".join(bits) or text[:160],
    )


def recover_trajectories(
    wiki: Iterable[Any],
    topic: str,
    state: ResearchState | None = None,
) -> tuple[ResearchTrajectory, ...]:
    """Turn verified on-claim papers into development lines, ordered in time.

    A paper is a step (problem → method → limitation → next problem),
    not a disconnected node. Rows that never name the topic object stay
    in the wiki as noise; they do not donate far mechanisms. Papers join
    a lineage when one cites the other, or when a later paper addresses
    an earlier limitation. Shared glue tokens (`model` / `search` /
    `training`) do not forge a history. Unrelated singletons stay
    separate.
    """
    state = state or initial_state(topic)
    papers = _paper_records(wiki, topic)
    papers = [row for row in papers if _paper_on_object(row, topic)]
    if not papers:
        return ()
    groups = _group_lineages(papers)
    trajectories: list[ResearchTrajectory] = []
    for index, group in enumerate(groups):
        steps = _chain_steps(group, state)
        if not steps:
            continue
        if not any(step.mechanism or step.new_problem for step in steps):
            # Papers that only *name* a system and state no limitation
            # carry nothing a jump could land on.
            continue
        patterns = tuple(
            dict.fromkeys(step.move for step in steps if step.move)
        )
        name = steps[-1].mechanism or steps[0].problem or f"lineage-{index}"
        trajectories.append(
            ResearchTrajectory(lineage=name, steps=steps, patterns=patterns)
        )
    return tuple(trajectories)


def observed_patterns(
    trajectories: Sequence[ResearchTrajectory],
) -> tuple[str, ...]:
    """How past work actually jumped: the prior for the first landing."""
    counts: dict[str, int] = {}
    for trajectory in trajectories:
        for move in trajectory.patterns:
            counts[move] = counts.get(move, 0) + 1
        for step in trajectory.steps:
            if step.move:
                counts[step.move] = counts.get(step.move, 0) + 1
    ranked = sorted(counts, key=lambda name: (-counts[name], name))
    return tuple(ranked)


def band_for_index(index: int, policy: str = "hold", current: str = "early") -> str:
    """Early stays near the frontier; later jumps may travel farther."""
    base = DISTANCE_BANDS[min(max(int(index), 0), len(DISTANCE_BANDS) - 1)]
    if policy in {"increase_distance", "tighten_trajectory"}:
        return apply_policy(base if index == 0 else current or base, policy)
    return base


def apply_policy(band: str, policy: str) -> str:
    order = list(DISTANCE_BANDS)
    here = band if band in order else "early"
    at = order.index(here)
    if policy == "increase_distance":
        return order[min(at + 1, len(order) - 1)]
    if policy == "tighten_trajectory":
        return order[max(at - 1, 0)]
    return here


def propose_jump(
    *,
    index: int,
    state: ResearchState,
    trajectories: Sequence[ResearchTrajectory],
    ledger: SearchLedger | None = None,
    used_far: Sequence[str] = (),
) -> JumpProposal:
    """Compile one landing from trajectories + accumulated feedback.

    Jump 0 learns the observed development moves and applies one to the
    user's current position. Later jumps read the ledger: previous
    candidates, negated directions, and the policy the last evaluation
    named. Distance grows only when the last idea was too close; a
    random mix tightens the trajectory constraint instead.
    """
    ledger = ledger or SearchLedger()
    blocked = set(used_far) | set(ledger.used_far)
    patterns = observed_patterns(trajectories)
    policy = ledger.policy or "hold"
    band = band_for_index(index, policy, ledger.band)
    move = _choose_move(index, patterns, policy, trajectories)
    attested = _attested_phrases(trajectories, state)
    latest = tuple(
        item.latest_mechanism for item in trajectories if item.latest_mechanism
    )
    far = _menu_for_band(
        band,
        move=move,
        trajectories=trajectories,
        attested=attested,
        latest=latest,
        blocked=blocked,
        state=state,
    )
    reasoning = _reasoning_chain(
        state, trajectories, move, band, far, policy, patterns
    )
    support = (
        "from_trajectory"
        if far and any(label in attested for label in far)
        else "from_user_state"
        if far
        else "none"
    )
    novelty = (
        "linear_extension"
        if far and set(far) <= set(latest)
        else "unmoored"
        if support == "none"
        else "pattern_applied"
    )
    return JumpProposal(
        index=index,
        move=move,
        band=band,
        far_labels=far,
        reasoning=reasoning,
        support=support,
        novelty=novelty,
        patterns=patterns,
        shaped_by=policy,
    )


def evaluate_landing(
    proposal: JumpProposal,
    *,
    far: str = "",
    category: str = "",
    why: str = "",
    trajectories: Sequence[ResearchTrajectory] = (),
    state: ResearchState | None = None,
) -> JumpFeedback:
    """Turn one landing's outcome into the next jump's policy."""
    label = str(far or "").strip()
    attested = _attested_phrases(trajectories, state)
    latest = {
        item.latest_mechanism for item in trajectories if item.latest_mechanism
    }
    if not label:
        support = "none"
    elif label in attested:
        support = "strong"
    elif any(_overlap(label, item) for item in attested):
        support = "weak"
    else:
        support = "none"
    if not label:
        novelty = "none"
    elif label in latest:
        novelty = "linear"
    elif support == "none":
        novelty = "unmoored"
    else:
        novelty = "deviated"
    tensions = tuple(state.tensions) if state is not None else ()
    value = (
        "worth"
        if label and (
            any(_overlap(label, item) for item in tensions)
            or any(
                _overlap(label, step.new_problem)
                for row in trajectories
                for step in row.steps
            )
        )
        else "unclear"
    )
    closed = category in {
        "graph_pair",
        "text_gate",
        "generation_refused",
        "prior",
        "world_weakens",
        "weakens",
    }
    feasibility = (
        "failed"
        if closed
        else "open"
        if category in {
            "entered",
            "world_supports",
            "synthetic_supports",
            "uninformative",
        }
        else "unknown"
    )
    if novelty == "linear" or (proposal.band == "early" and novelty != "deviated" and label in latest):
        distance = "too_close"
    elif novelty == "unmoored" or support == "none":
        distance = "too_far"
    else:
        distance = "on_band"
    if distance == "too_close":
        policy = "increase_distance"
    elif distance == "too_far" or novelty == "unmoored":
        policy = "tighten_trajectory"
    elif support == "weak":
        policy = "retrieve_support"
    elif feasibility == "failed" and support == "strong":
        policy = "reformulate"
    else:
        policy = "hold"
    detail = " ".join(
        bit for bit in (category, why.strip()[:160]) if bit
    )
    return JumpFeedback(
        support=support,
        novelty=novelty,
        value=value,
        feasibility=feasibility,
        distance=distance,
        policy=policy,
        detail=detail,
    )


def trajectory_prompt_lines(
    state: ResearchState,
    trajectories: Sequence[ResearchTrajectory],
) -> tuple[str, ...]:
    """Show recovered development lines to the generator. Not evidence."""
    lines = [
        "Current position in research space: " + (state.summary or "unspecified")
    ]
    if state.tensions:
        lines.append("User-stated open problems: " + "; ".join(state.tensions[:3]))
    if not trajectories:
        lines.append(
            "No research trajectories recovered. Do not invent a mechanism "
            "from an unrelated concept catalog."
        )
        return tuple(lines)
    lines.append(
        "Recovered research trajectories (problem → method → limitation → "
        "next problem). Learn how the field jumped, then apply that move "
        "to the current position — do not linearly extend the last step:"
    )
    for trajectory in trajectories[:3]:
        bits = []
        for step in trajectory.steps[:4]:
            piece = step.mechanism or step.problem
            if step.new_problem:
                piece += " → " + step.new_problem
            if piece:
                bits.append(piece)
        if bits:
            lines.append("- " + " ⇒ ".join(bits))
        if trajectory.patterns:
            lines.append("  observed moves: " + ", ".join(trajectory.patterns))
    return tuple(lines)


def _blocked(topic: str) -> set[str]:
    blocked = set(topic_object_phrases(topic))
    blocked |= {token for token in folded_terms(topic) if len(token) >= 3}
    return blocked


def _blocked_phrase(phrase: str, blocked: set[str]) -> bool:
    tokens = folded_terms(phrase)
    if not tokens:
        return True
    if phrase in blocked:
        return True
    return tokens <= blocked or tokens <= APPLY_PAD


def _method_phrases(topic: str, blocked: set[str]) -> tuple[str, ...]:
    match = _METHOD_TAIL.search(topic.strip())
    blob = match.group(1) if match else ""
    phrases = [
        gram
        for gram in claim_phrases(blob)
        if " " in gram and not _blocked_phrase(gram, blocked)
    ]
    return tuple(phrases[:3])


def _tension_phrases(topic: str, blocked: set[str]) -> tuple[str, ...]:
    parts = _TENSION_SPLIT.split(topic)
    if len(parts) < 2:
        return ()
    phrases: list[str] = []
    seen: set[str] = set()
    for chunk in parts[1:]:
        for gram in claim_phrases(chunk):
            if " " not in gram or gram in seen or _blocked_phrase(gram, blocked):
                continue
            seen.add(gram)
            phrases.append(gram)
            if len(phrases) >= 4:
                return tuple(phrases)
    return tuple(phrases)


def _paper_records(wiki: Iterable[Any], topic: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in wiki or []:
        title = ""
        published = ""
        cite = ""
        abstract = ""
        limitations: list[str] = []
        if isinstance(raw, FreshWork):
            title = str(raw.title or "").strip()
            published = str(raw.published or "")
            cite = str(raw.cite_id() or "")
            abstract = str(raw.abstract or "")
            limitations = limitation_sentences(abstract)
        elif isinstance(raw, dict) and raw.get("verified") is not False:
            title = str(raw.get("title") or "").strip()
            published = str(raw.get("published") or "")
            cite = str(raw.get("cite_id") or "")
            abstract = str(raw.get("abstract") or "")
            limitations = [str(item) for item in (raw.get("limitations") or []) if item]
            if not limitations:
                limitations = limitation_sentences(abstract)
        if not title:
            continue
        if isinstance(raw, FreshWork):
            references = tuple(raw.references)
        else:
            references = tuple(
                str(item).strip()
                for item in (raw.get("references") or ())
                if str(item).strip()
            )
        # Mechanisms the paper describes in lowercase prose. A title bigram
        # the abstract only capitalises (`Darwin Gödel`, `ARC-AGI`) is the
        # name of a system and never becomes a step's mechanism.
        methods = described_methods(title, abstract, topic)
        # Gaps are the object of the limitation sentence (`dynamic
        # inserts`), not a clause bigram (`unfortunately proving`).
        limits = described_gaps(limitations, topic, abstract=abstract)
        rows.append(
            {
                "title": title,
                "published": published,
                "cite_id": cite or title,
                "abstract": abstract,
                "methods": methods,
                "limits": limits,
                "limitations": tuple(limitations),
                "references": references,
            }
        )
    rows.sort(key=lambda row: (_year(row["published"]), row["cite_id"]))
    return rows


def _year(published: str) -> str:
    match = _YEAR.match(str(published or ""))
    return match.group(1) if match else "9999"


def _group_lineages(papers: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Join papers that cite each other or continue an attested limitation.

    Token overlap is not a history. `model` / `search` / `training` used
    to weld unrelated on-claim papers into one fake chronology.
    """
    if len(papers) <= 1:
        return [papers] if papers else []
    parent = list(range(len(papers)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, later in enumerate(papers):
        for j, earlier in enumerate(papers):
            if i == j:
                continue
            if not _same_lineage(later, earlier):
                continue
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a
    buckets: dict[int, list[dict[str, Any]]] = {}
    for i, row in enumerate(papers):
        buckets.setdefault(find(i), []).append(row)
    return [buckets[key] for key in sorted(buckets)]


def _same_lineage(later: dict[str, Any], earlier: dict[str, Any]) -> bool:
    if _cites(later, earlier) or _cites(earlier, later):
        return True
    if _year(later["published"]) < _year(earlier["published"]):
        return False
    return _addresses_limitation(later, earlier)


def _paper_on_object(row: dict[str, Any], topic: str) -> bool:
    from .worldfields import paper_on_claim_object

    return paper_on_claim_object(
        str(row.get("title") or ""),
        str(row.get("abstract") or ""),
        topic,
        topic,
    )


def _norm_cite(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^https?://(arxiv\.org/abs/|doi\.org/|openalex\.org/)", "", text)
    text = re.sub(r"^(arxiv:|doi:|s2:|openalex:)", "", text)
    text = re.sub(r"v\d+$", "", text)
    return text.strip()


def _paper_keys(row: dict[str, Any]) -> set[str]:
    keys = set()
    for raw in (row.get("cite_id"), row.get("title")):
        key = _norm_cite(str(raw or ""))
        if len(key) >= 4:
            keys.add(key)
    return keys


def _paper_refs(row: dict[str, Any]) -> set[str]:
    return {
        key
        for raw in (row.get("references") or ())
        if (key := _norm_cite(str(raw))) and len(key) >= 4
    }


def _cites(source: dict[str, Any], target: dict[str, Any]) -> bool:
    return bool(_paper_refs(source) & _paper_keys(target))


def _addresses_limitation(later: dict[str, Any], earlier: dict[str, Any]) -> bool:
    """True when the later paper takes up an earlier attested limitation."""
    blobs = list(later.get("methods") or []) + [
        str(later.get("title") or ""),
        str(later.get("abstract") or ""),
    ]
    for limit in earlier.get("limits") or ():
        if not _usable_phrase(limit):
            continue
        if any(blob and _takes_up(limit, blob) for blob in blobs):
            return True
    return False


def _takes_up(limit: str, text: str) -> bool:
    """The text carries the whole limitation phrase, not one of its words.

    One shared four-letter token (`agent`, `require`) welded all seven
    cwm-iclr2027 seed papers into a single lineage. A later paper takes
    up an earlier gap when every strong token of the gap phrase appears
    in it.
    """
    want = {
        token
        for token in folded_terms(limit) - APPLY_PAD
        if len(token) >= 4 and token not in _WEAK_LIMIT
    }
    if len(want) < 2:
        return False
    have = folded_terms(text) - APPLY_PAD
    return want <= have


def _chain_steps(
    papers: list[dict[str, Any]],
    state: ResearchState,
) -> tuple[TrajectoryStep, ...]:
    steps: list[TrajectoryStep] = []
    previous: dict[str, Any] | None = None
    for row in papers:
        mechanism = (row["methods"][0] if row["methods"] else "")
        new_problem = (row["limits"][0] if row["limits"] else "")
        inherited = ""
        changed = ""
        addressed = ""
        # A lone paper did not jump from anywhere: it has no observed move.
        # Booking singletons as `migrate_setting` made every one-paper line
        # in cwm-iclr2027 vote for the same fake pattern.
        move = ""
        problem = state.object[0] if state.object else state.summary
        if previous is not None:
            inherited = previous["methods"][0] if previous["methods"] else ""
            problem = (
                previous["limits"][0]
                if previous["limits"]
                else inherited or problem
            )
            prev_tokens = folded_terms(inherited)
            now_tokens = folded_terms(mechanism)
            dropped = prev_tokens - now_tokens - APPLY_PAD
            if dropped:
                changed = " ".join(sorted(dropped)[:4])
            for limit in previous.get("limits") or ():
                if mechanism and _takes_up(limit, mechanism):
                    addressed = limit
                    break
                if any(_takes_up(limit, blob) for blob in (row["title"], row["abstract"])):
                    addressed = limit
                    break
            move = _edge_move(addressed, changed, previous, row)
        elif row["limits"] and mechanism:
            move = "reframe_limitation"
        steps.append(
            TrajectoryStep(
                problem=problem,
                inherited=inherited,
                mechanism=mechanism,
                assumption_changed=changed,
                limitation_addressed=addressed,
                new_problem=new_problem,
                cite_id=str(row["cite_id"]),
                published=str(row["published"]),
                move=move,
            )
        )
        previous = row
    return tuple(steps)


def _edge_move(
    addressed: str,
    changed: str,
    previous: dict[str, Any],
    current: dict[str, Any],
) -> str:
    if addressed:
        return "reframe_limitation"
    if changed:
        return "drop_assumption"
    prev_m = set(previous.get("methods") or ())
    now_m = set(current.get("methods") or ())
    if prev_m and now_m and prev_m.isdisjoint(now_m):
        if set(previous.get("limits") or ()) & now_m:
            return "reframe_limitation"
        return "change_objective"
    if len(previous.get("methods") or ()) and len(current.get("methods") or ()):
        return "migrate_setting"
    return "explicit_latent"


def _choose_move(
    index: int,
    patterns: Sequence[str],
    policy: str,
    trajectories: Sequence[ResearchTrajectory],
) -> str:
    if policy == "tighten_trajectory" and patterns:
        return patterns[0]
    if policy == "increase_distance" and len(trajectories) >= 2:
        return "compose_trajectories"
    if index == 0:
        return patterns[0] if patterns else "reframe_limitation"
    if index == 1 and len(trajectories) >= 2:
        return "compose_trajectories" if "compose_trajectories" in JUMP_MOVES else (
            patterns[1] if len(patterns) > 1 else "connect_problems"
        )
    if patterns:
        return patterns[min(index, len(patterns) - 1)]
    return JUMP_MOVES[min(index, len(JUMP_MOVES) - 1)]


def _attested_phrases(
    trajectories: Sequence[ResearchTrajectory],
    state: ResearchState | None,
) -> tuple[str, ...]:
    seen: set[str] = set()
    phrases: list[str] = []
    for trajectory in trajectories:
        for step in trajectory.steps:
            for item in (step.mechanism, step.new_problem, step.limitation_addressed):
                if item and item not in seen:
                    seen.add(item)
                    phrases.append(item)
    if state is not None:
        for item in state.methods + state.tensions:
            if item and item not in seen:
                seen.add(item)
                phrases.append(item)
    return tuple(phrases)


def _menu_for_band(
    band: str,
    *,
    move: str,
    trajectories: Sequence[ResearchTrajectory],
    attested: Sequence[str],
    latest: Sequence[str],
    blocked: set[str],
    state: ResearchState,
) -> tuple[str, ...]:
    older: list[str] = []
    limits: list[str] = []
    newest = set(latest)
    step_count = sum(len(item.steps) for item in trajectories)
    for trajectory in trajectories:
        mechanisms = [step.mechanism for step in trajectory.steps if step.mechanism]
        for step in trajectory.steps:
            if step.new_problem:
                limits.append(step.new_problem)
            if step.limitation_addressed:
                limits.append(step.limitation_addressed)
        older.extend(item for item in mechanisms[:-1] if item)
        if move == "reframe_limitation" and trajectory.steps:
            last = trajectory.steps[-1]
            if last.new_problem:
                limits.insert(0, last.new_problem)
    limits = [item for item in limits if _usable_phrase(item)]
    older = [item for item in older if _usable_phrase(item)]
    latest = tuple(item for item in latest if _usable_phrase(item))
    newest = set(latest)
    if band == "early":
        if step_count <= 1:
            ordered = list(dict.fromkeys([*latest, *limits, *older]))
        else:
            ordered = list(dict.fromkeys([*limits, *older]))
            ordered = [item for item in ordered if item not in newest] or ordered
    elif band == "middle":
        ordered = list(dict.fromkeys([*older, *limits, *latest]))
        if move == "compose_trajectories" and len(trajectories) >= 2:
            side = [
                step.mechanism
                for step in trajectories[-1].steps
                if step.mechanism
            ]
            ordered = list(dict.fromkeys(side + ordered))
    else:
        ordered = list(dict.fromkeys([*limits, *older, *latest, *state.tensions]))
    menu: list[str] = []
    seen: set[str] = set()
    for phrase in ordered:
        if not phrase or phrase in blocked or phrase in seen:
            continue
        seen.add(phrase)
        menu.append(phrase)
        if len(menu) >= FAR_CAP:
            break
    if not menu:
        for phrase in attested:
            if phrase and phrase not in blocked and phrase not in seen:
                menu.append(phrase)
            if len(menu) >= FAR_CAP:
                break
    return tuple(menu)


def _reasoning_chain(
    state: ResearchState,
    trajectories: Sequence[ResearchTrajectory],
    move: str,
    band: str,
    far: Sequence[str],
    policy: str,
    patterns: Sequence[str],
) -> tuple[str, ...]:
    knowledge = state.summary or "user-stated research direction"
    if trajectories:
        last = trajectories[0]
        knowledge = (
            f"{last.steps[0].problem or knowledge} evolved via "
            + " → ".join(
                step.mechanism or step.new_problem for step in last.steps[:3]
                if step.mechanism or step.new_problem
            )
        )
    pattern = ", ".join(patterns[:3]) or move
    landing = ", ".join(far[:3]) or "(empty menu; no attested far phrase)"
    return (
        f"Existing knowledge: {knowledge}",
        f"Observed limitation / pattern: {pattern}",
        f"Abstraction: apply {move} at band {band} (policy {policy})",
        f"Jump operation: {move}",
        f"Candidate idea position: {landing}",
    )


_WEAK_LIMIT = frozenset(
    {
        "allow",
        "cannot",
        "do",
        "does",
        "not",
        "support",
        "test",
        "tests",
        "use",
        "used",
        "using",
        "we",
        "however",
        "also",
        "still",
        "just",
    }
)

def _usable_phrase(phrase: str) -> bool:
    """True when the phrase is a method/limitation, not a leftover clause.

    `we do not support dynamic inserts` yields both `dynamic inserts`
    (keep) and `support dynamic` (drop): a single strong token glued to
    a weak verb is not a mechanism.
    """
    tokens = folded_terms(phrase) - APPLY_PAD
    if not tokens:
        return False
    strong = [
        token
        for token in tokens
        if len(token) >= 4 and token not in _WEAK_LIMIT
    ]
    if len(strong) >= 2:
        return True
    return len(strong) == 1 and len(strong[0]) >= 6 and len(tokens) == 1


def _overlap(left: str, right: str) -> bool:
    a = folded_terms(left) - APPLY_PAD
    b = folded_terms(right) - APPLY_PAD
    shared = {token for token in (a & b) if len(token) >= 4}
    return bool(shared)
