"""Object-domain lock: a far jump may leave the seed's neighborhood, not the topic.

The live failure mode was not "too little crossover". A card about graph
hardness or a spanner was generated under an agent-safety topic, the probe
hard-coded a win, and the write-up painted the title back onto LLM tool-use.
The evaluation function that stops that is three executable checks:

1. Generation: claim + mechanism must cover a topic bigram or two
   distinctive unigrams (not a leftover `metric` / `host`). Empty payload
   (topic is just the seed) is a no-op, so unit tests that omit `topic=`
   keep their prompt bytes.
2. Retrieval: far labels come from verified on-claim literature
   (`researchspace`), never from cosine neighbours of the whole corpus;
   the concept graph only answers "already combined" for graph pairs.
3. Writing: title and idea may not import topic-field brands that the claim
   itself does not own. The lock is the claim's field, not the topic: locking
   to the topic would *license* retitling a hardness card as tool-use safety.

These checks never read Elo, never revive a dead card, and never enter the
promotion ladder. They are schema teeth, the same kind as verbatim pair copy.
"""

from __future__ import annotations

from .prior import (
    APPLY_PAD,
    claim_phrases,
    content_tokens,
    fold_plural,
    ordered_content,
    ordered_content_runs,
)

# Extra fold so "compression" meets "compress" and "verification" meets
# "verify". Plural-s is already handled by content_tokens.
_STEMS = ("ation", "ition", "ion", "ness", "ment", "ing", "ity")


def _stem(word: str) -> str:
    for suffix in _STEMS:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def folded_terms(text: str) -> frozenset[str]:
    """Content tokens minus padding, with a light suffix fold."""
    return frozenset(_stem(word) for word in content_tokens(text) if word not in APPLY_PAD)


def topic_terms(topic: str) -> frozenset[str]:
    return folded_terms(topic)


def topic_payload_terms(topic: str, seed_label: str) -> frozenset[str]:
    """Topic words the seed label does not already carry.

    These are the object-domain brands the claim must still be about after
    a far concept supplies a mechanism. If the topic is only the seed, the
    payload is empty and generation is not asked for extra coverage.
    """
    return topic_terms(topic) - folded_terms(seed_label)


# Method/infra words that never identify the scientific object. A
# coding-agent claim that only shares "metric" / "host" with the topic
# is a cosine neighbour wearing the topic's clothes.
_OBJECT_GENERIC = frozenset(
    {
        "already",
        "appear",
        "cheap",
        "clock",
        "count",
        "counting",
        "duration",
        "english",
        "field",
        "first",
        "gpu",
        "host",
        "invent",
        "later",
        "lever",
        "like",
        "metric",
        "official",
        "only",
        "prefix",
        "probe",
        "public",
        "raw",
        "stdlib",
        "stored",
        "those",
        "versus",
        "without",
    }
)


def _surface(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def carries_adjacent_phrase(text: str, phrase: str) -> bool:
    """True when `phrase` appears as neighbouring words in `text`.

    Token-set inclusion is not coverage: `reasoning` in one clause and
    `traces` in another is not the object `reasoning traces`.
    """
    needle = _surface(phrase)
    hay = _surface(text)
    if needle and needle in hay:
        return True
    wanted = [run for run in ordered_content_runs(phrase) if len(run) >= 2]
    if not wanted:
        return False
    for run in ordered_content_runs(text):
        for gram in wanted:
            span = len(gram)
            for i in range(len(run) - span + 1):
                if run[i : i + span] == gram:
                    return True
    return False


def claim_covers_topic(
    claim: str,
    mechanism: str,
    topic: str,
    seed_label: str,
) -> bool:
    """True when the hypothesis is still a hypothesis about the topic.

    A named topic bigram is the object and must appear as adjacent
    words. Two leftover unigrams (`reasoning` + `trace`, or `metric` +
    `count`) are not coverage: that was how a wavelet-tree card still
    passed a reasoning-traces lock. Unigram fallback exists only when
    the topic itself has no bigram.
    """
    body_text = f"{claim} {mechanism}"
    body = folded_terms(claim) | folded_terms(mechanism)
    bigrams = []
    for gram in topic_object_phrases(topic):
        if " " not in gram:
            continue
        terms = folded_terms(gram) - _OBJECT_GENERIC
        if not terms:
            continue
        bigrams.append(gram)
        if carries_adjacent_phrase(body_text, gram):
            return True
    if bigrams:
        return False
    payload = topic_payload_terms(topic, seed_label) - _OBJECT_GENERIC
    if not payload:
        return True
    return len(payload & body) >= 2


def claim_home(claim: str, mechanism: str, pair: tuple[str, str]) -> str:
    return " ".join((claim, mechanism, pair[0], pair[1]))


def writeup_retrofits_topic(
    title: str,
    idea: str,
    *,
    home: str,
    topic: str,
    seed_label: str,
) -> tuple[str, ...]:
    """Topic-field brands that the write-up added and the claim does not own.

    One stray word is noise (`compression` in a pivot-log title). Two or
    more in the title or idea is the live failure: an LLM-agent title on a
    hardness claim, a genomics title on a compiler claim.
    """
    extra = topic_payload_terms(topic, seed_label) - folded_terms(home)
    if not extra:
        return ()
    surface = folded_terms(title) | folded_terms(idea)
    found = tuple(sorted(extra & surface))
    if len(found) < 2:
        return ()
    return found


# Words a probe uses to name a quantity. They never identify the
# scientific object: "edge ops" is not an Agent Card claim.
_MEASURE_PAD = frozenset(
    {
        "arm",
        "both",
        "byte",
        "control",
        "cost",
        "count",
        "hop",
        "mean",
        "measure",
        "metric",
        "number",
        "ops",
        "probe",
        "quantity",
        "rate",
        "ratio",
        "score",
        "size",
        "sum",
        "time",
        "total",
        "treatment",
        "value",
        "visit",
        "work",
    }
)


def measure_stays_on_object(
    measure: str,
    claim: str,
    mechanism: str = "",
    topic: str = "",
) -> bool:
    """True when the reported quantity is still about the claim's object.

    A far-field jump may supply the intervention (min-cut, splay). It
    must not replace the measured object: counting cut-maintenance edge
    ops does not test an AUTH_REQUIRED / Agent Card claim. Empty claim
    or empty measure is a no-op so existing unit tests keep their bytes.
    """
    if not str(claim or "").strip() or not str(measure or "").strip():
        return True
    claim_terms = folded_terms(claim)
    measure_terms = folded_terms(measure) - _MEASURE_PAD
    if not measure_terms:
        return True
    if measure_terms & claim_terms:
        return True
    mech_only = folded_terms(mechanism) - claim_terms
    if measure_terms & mech_only:
        return False
    topic_hit = topic_terms(topic) & measure_terms if topic.strip() else frozenset()
    if topic_hit:
        return True
    # No shared tokens with the far mechanism: this is not the live
    # substitution (min-cut ops on an AUTH_REQUIRED claim). Generic
    # hop-counts used by unit tests and toy probes stay admissible.
    return True


def experiment_stays_on_object(
    experiment: str,
    treatment_arm: str,
    claim: str,
    mechanism: str = "",
    topic: str = "",
) -> bool:
    """Diagnosis text must measure the claim, not only the far mechanism."""
    return measure_stays_on_object(
        f"{experiment} {treatment_arm}",
        claim,
        mechanism=mechanism,
        topic=topic,
    )


def topic_object_phrases(topic: str) -> tuple[str, ...]:
    """Distinctive phrases for live literature and phrase-level anchoring.

    Whole-sentence embedding of a long topic collapses onto generic nearby
    nodes. Phrase retrieve (topic bigrams, then distinctive unigrams)
    recovers the object the researcher named. Unigrams shorter than five
    letters are too generic to query.
    """
    text = topic.strip()
    if not text:
        return ()
    phrases: list[str] = []
    seen: set[str] = set()
    for gram in claim_phrases(text):
        if gram not in seen:
            seen.add(gram)
            phrases.append(gram)
    for word in ordered_content(text):
        if len(word) >= 5 and word not in seen:
            seen.add(word)
            phrases.append(word)
    return tuple(phrases)


def label_topic_overlap(label: str, topic: str) -> int:
    """How many topic terms the concept label actually carries."""
    if not topic.strip() or not label.strip():
        return 0
    return len(folded_terms(label) & topic_terms(topic))


def label_covers_topic(label: str, topic: str) -> bool:
    """True when the label names the topic's object, not a stray token.

    One shared generic word is how `definite program` became the anchor
    for a code-world-model topic: `program` overlaps, the object does
    not. Coverage is the same standard `claim_covers_topic` applies to
    claims — a topic bigram carried whole, or two distinctive unigrams.
    """
    if not topic.strip() or not label.strip():
        return False
    label_terms = folded_terms(label)
    for gram in topic_object_phrases(topic):
        if " " not in gram:
            continue
        terms = folded_terms(gram) - _OBJECT_GENERIC
        if terms and carries_adjacent_phrase(label, gram):
            return True
    payload = topic_terms(topic) - _OBJECT_GENERIC
    return len(payload & label_terms) >= 2


def preferred_object_labels(near_labels: tuple[str, ...], topic: str) -> tuple[str, ...]:
    """Near concepts that still name the topic's object.

    Empty when no label *covers* the topic (a single shared unigram is
    not coverage): callers should then surface `topic_object_phrases` as
    the near side rather than handing the generator a cosine-nearest
    neighbour as the scientific object.
    """
    if not topic.strip():
        return ()
    return tuple(label for label in near_labels if label_covers_topic(label, topic))


def graph_supplies_mechanisms(
    near_labels: tuple[str, ...], topic: str
) -> bool:
    """True only when this corpus graph actually names the topic object.

    Cosine neighbours are a retrieval geometry, not a mechanism lexicon.
    Agent-ml / ARIS take distant mechanisms from literature and
    complementary lenses (failure modes, reusable methods), not from a
    frozen co-occurrence snapshot of another field. An empty covering
    set means the graph oracle may still judge graph-internal pairs, but
    it must not mint pair[1].
    """
    return bool(preferred_object_labels(near_labels, topic))


def surface_object_labels(
    near_labels: tuple[str, ...],
    topic: str,
    *,
    max_phrases: int = 3,
) -> tuple[str, ...]:
    """Labels generation must use as pair[0].

    Graph overlap wins. When the theory graph has no node for the named
    object, the researcher's own phrases stand in so a far concept cannot
    rename the field to `finite metric` or `balanced tree`.
    """
    kept = preferred_object_labels(near_labels, topic)
    if kept:
        return kept
    phrases = tuple(
        gram for gram in topic_object_phrases(topic) if " " in gram
    )[:max_phrases]
    if phrases:
        return phrases
    return tuple(topic_object_phrases(topic)[:max_phrases])


def without_far_terms(text: str, far_label: str, topic: str = "") -> str:
    """Drop the far concept's own words from a claim before schema inference.

    The far member supplies a mechanism, not a new field. Leaving
    `feedback edge set` inside the text lets `edge` vote a code-world
    claim into an undirected-graph world. Words the topic itself also
    carries are kept — the far concept cannot veto the topic's object.
    """
    body = str(text or "")
    far = folded_terms(far_label) - (topic_terms(topic) if topic.strip() else frozenset())
    if not far or not body.strip():
        return body

    def _fold_one(word: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in word)
        parts = cleaned.split()
        if not parts:
            return ""
        return _stem(fold_plural(parts[0]))

    kept = [word for word in body.split() if _fold_one(word) not in far]
    return " ".join(kept)


def feed_query_concepts(
    topic: str,
    object_labels: tuple[str, ...],
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    """Concepts sent to the live literature feed.

    Prefer topic bigrams so the generator is grounded in the named field,
    not in the cosine-nearest graph node.
    """
    phrases = tuple(gram for gram in topic_object_phrases(topic) if " " in gram)[:3]
    if len(phrases) >= 2:
        return phrases
    if object_labels:
        return object_labels[:3]
    return fallback[:3]


def topic_prompt_block(topic: str) -> str:
    """Empty when no topic is supplied, so cached generation prompts replay."""
    if not topic.strip():
        return ""
    return (
        f"Researcher's topic: {topic.strip()}\n"
        "The claim must remain a hypothesis about THIS topic. The distant"
        " concept supplies a mechanism, not a new field. Do not write a"
        " graph-alignment or hardness paper when the topic is agent safety,"
        " or a genomics paper when the topic is compilers.\n"
        "The first pair member is the scientific object of the topic; the"
        " distant concept is only a construction, bound, or measurement."
        " Do not let the distant name replace the object.\n"
        "If this mission already scouted levers, prefer an unused declared"
        " lever over a new graph noun. Literature limitation sentences"
        " (however, we do not, future work) are agenda, not solved claims.\n"
        "Think like a working scientist: name a real failed approach, drop"
        " one assumption, then keep the strongest objection on the card."
        " Prefer inversion or reframing over applying the distant concept"
        " to the topic as a method.\n\n"
    )
