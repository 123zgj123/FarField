"""Object-domain lock: a far jump may leave the seed's neighborhood, not the topic.

The live failure mode was not "too little crossover". A card about graph
hardness or a spanner was generated under an agent-safety topic, the probe
hard-coded a win, and the write-up painted the title back onto LLM tool-use.
The evaluation function that stops that is three executable checks:

1. Generation: claim + mechanism must carry at least one payload term from
   the researcher's topic that is not already the seed label. Empty payload
   (topic is just the seed) is a no-op, so unit tests that omit `topic=`
   keep their prompt bytes.
2. Retrieval: far-field `top_m` is taken from the topic's domain band, not
   from the whole corpus (see `farspace.domain_band`).
3. Writing: title and idea may not import topic-field brands that the claim
   itself does not own. The lock is the claim's field, not the topic: locking
   to the topic would *license* retitling a hardness card as tool-use safety.

These checks never read Elo, never revive a dead card, and never enter the
promotion ladder. They are schema teeth, the same kind as verbatim pair copy.
"""

from __future__ import annotations

from .prior import APPLY_PAD, content_tokens

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


def claim_covers_topic(
    claim: str,
    mechanism: str,
    topic: str,
    seed_label: str,
) -> bool:
    """True when the hypothesis is still a hypothesis about the topic."""
    payload = topic_payload_terms(topic, seed_label)
    if not payload:
        return True
    body = folded_terms(claim) | folded_terms(mechanism)
    return bool(payload & body)


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


def topic_prompt_block(topic: str) -> str:
    """Empty when no topic is supplied, so cached generation prompts replay."""
    if not topic.strip():
        return ""
    return (
        f"Researcher's topic: {topic.strip()}\n"
        "The claim must remain a hypothesis about THIS topic. The distant"
        " concept supplies a mechanism, not a new field. Do not write a"
        " graph-alignment or hardness paper when the topic is agent safety,"
        " or a genomics paper when the topic is compilers.\n\n"
    )
