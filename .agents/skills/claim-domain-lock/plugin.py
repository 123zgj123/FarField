"""Executable hooks for claim-domain-lock.

The distant concept supplies a mechanism, not a new field. PluginHost
calls lock_claim at generation and lock_writeup at briefing; SKILL.md
does not replace these checks.
"""

from __future__ import annotations

import json
from typing import Any

NAME = "claim-domain-lock"
HOOKS = ("lock_claim", "lock_writeup", "validate_probe", "validate_diagnosis", "execute")


def lock_claim(
    claim: str = "",
    mechanism: str = "",
    topic: str = "",
    seed_label: str = "",
    **_: Any,
) -> str | None:
    from farfield.extras.domain import claim_covers_topic

    if topic.strip() and not claim_covers_topic(claim, mechanism, topic, seed_label):
        return (
            "claim and mechanism must use at least one content word from the"
            " topic that is not already the seed label; a distant concept is"
            " a mechanism, not a license to change fields"
        )
    return None


def validate_probe(
    measure: str = "",
    claim: str = "",
    mechanism: str = "",
    topic: str = "",
    source: str = "",
    schema: str = "",
    **_: Any,
) -> str | None:
    from farfield.extras.domain import measure_stays_on_object
    from farfield.extras.worldfields import probe_reads_attested_fields

    if not measure_stays_on_object(measure, claim, mechanism=mechanism, topic=topic):
        return (
            "the probe measure names the far-field mechanism's cost, not"
            " the claim's scientific object"
        )
    return probe_reads_attested_fields(
        source, claim=claim, mechanism=mechanism, schema=schema
    )


def validate_diagnosis(
    experiment: str = "",
    treatment_arm: str = "",
    control_arm: str = "",
    claim: str = "",
    mechanism: str = "",
    topic: str = "",
    schema: str = "",
    world_lever: str = "",
    **_: Any,
) -> str | None:
    from farfield.extras.domain import experiment_stays_on_object
    from farfield.extras.worldfields import (
        diagnosis_names_claimed_fields,
        missing_attested_field,
    )

    if not experiment_stays_on_object(
        experiment, treatment_arm, claim, mechanism=mechanism, topic=topic
    ):
        return (
            "the diagnosis measures the distant mechanism instead of the"
            " claim's object"
        )
    missing = missing_attested_field(
        claim,
        mechanism,
        schema,
        world_lever,
    )
    if missing:
        return missing
    return diagnosis_names_claimed_fields(
        experiment=experiment,
        treatment=treatment_arm,
        control=control_arm,
        claim=claim,
        mechanism=mechanism,
        schema=schema,
    )


def lock_writeup(
    title: str = "",
    idea: str = "",
    home: str = "",
    topic: str = "",
    seed_label: str = "",
    **_: Any,
) -> str | None:
    from farfield.extras.domain import writeup_retrofits_topic

    if not (topic.strip() and home.strip()):
        return None
    found = writeup_retrofits_topic(
        title, idea, home=home, topic=topic, seed_label=seed_label
    )
    if found:
        return (
            "title or idea imported topic-field terms the claim does not"
            " own (" + ", ".join(found) + "); writing cannot change"
            " the domain the hypothesis belongs to"
        )
    return None


def execute(args: dict[str, Any] | None = None, workspace: Any = None, **_: Any) -> str:
    args = args or {}
    reason = lock_claim(
        claim=str(args.get("claim") or ""),
        mechanism=str(args.get("mechanism") or ""),
        topic=str(args.get("topic") or ""),
        seed_label=str(args.get("seed_label") or ""),
    )
    if reason is None:
        reason = lock_writeup(
            title=str(args.get("title") or ""),
            idea=str(args.get("idea") or ""),
            home=str(args.get("home") or ""),
            topic=str(args.get("topic") or ""),
            seed_label=str(args.get("seed_label") or ""),
        )
    return json.dumps({"ok": reason is None, "refuse": reason}, ensure_ascii=False)
