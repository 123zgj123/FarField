"""First-class idea kinds. Not every landing is a two-arm probe.

`idea_kind` is a lifecycle control type. Policy lives in `lifecycle.py`;
this module keeps the kind catalogue and the thin predicates callers
already import. Pair[0] stays the topic object. WORLD vs GENERATED stays
the ladder. Only a probe shares one freeze and one measure across arms.
"""

from __future__ import annotations

from typing import Any, Mapping

PROBE = "probe"
QUESTION = "question"
ACQUIRE = "acquire"
THEORY = "theory"

KINDS = (QUESTION, ACQUIRE, THEORY, PROBE)
_ALIASES = {
    "probe": PROBE,
    "question": QUESTION,
    "acquire": ACQUIRE,
    "theory": THEORY,
    "harvest": ACQUIRE,
    "import": ACQUIRE,
    "derive": ACQUIRE,
    "derivation": THEORY,
}


def parse_idea_kind(raw: Any, *, default: str = PROBE) -> str:
    """Unknown or empty text becomes `default`. Never invents a new kind."""
    text = str(raw or "").strip().lower()
    if not text:
        return default
    return _ALIASES.get(text, default)


def idea_kind_of(row: Any) -> str:
    if row is None:
        return PROBE
    if isinstance(row, Mapping):
        return parse_idea_kind(row.get("idea_kind"))
    return parse_idea_kind(getattr(row, "idea_kind", ""))


def runs_probe(kind: str) -> bool:
    """Only a probe spends a causal two-arm. Question is observational."""
    return parse_idea_kind(kind) == PROBE


def can_corroborate(kind: str) -> bool:
    """Acquire and theory cannot climb. A question can support an association."""
    return parse_idea_kind(kind) in {QUESTION, PROBE}


def skip_writeup_on_needs_world(kind: str) -> bool:
    """Acquire/theory are live ideas. needs_world is the product, not a miss."""
    return parse_idea_kind(kind) not in {ACQUIRE, THEORY}


def observational_first(kind: str) -> bool:
    """Only a question must stay on observational contrasts.

    A declared probe may weld a manipulable lever even when unused
    contrasts remain. Forcing every probe onto a contrast is how a
    two-arm idea gets rewritten as an observation.
    """
    return parse_idea_kind(kind) == QUESTION


def allows_interventional_while_contrasts_remain(kind: str) -> bool:
    return parse_idea_kind(kind) in {ACQUIRE, THEORY}


def prediction_tokens_for(kind: str, measurable: set[str]) -> set[str]:
    extra = {
        ACQUIRE: {
            "harvest",
            "import",
            "derive",
            "validators_mutable",
            "replay_gate",
            "freeze_validators",
            "object_properties",
        },
        THEORY: set(),
        QUESTION: set(),
        PROBE: set(),
    }
    return set(measurable) | extra.get(parse_idea_kind(kind), set())


def render_question_board(
    *,
    scorable: tuple[str, ...] | list[str] = (),
    records: list[Mapping[str, Any]] | None = None,
    interventional: tuple[str, ...] | list[str] = (),
) -> str:
    """Mission-level Question Room analog. Not a Station finding."""
    lines = [
        "# Question board",
        "",
        "Questions this freeze can score now, plus acquire targets it cannot.",
        "A row here is not a conference result and not a two-arm win.",
        "",
        "## Scorable on this freeze",
        "",
    ]
    if scorable:
        for item in scorable:
            lines.append(f"- {item}")
        lines.append("")
    else:
        lines.extend(["- (none left unused)", ""])
    questions = [
        row
        for row in (records or [])
        if idea_kind_of(row) == QUESTION and not row.get("killed")
    ]
    if questions:
        lines.extend(["## Questions written this mission", ""])
        for row in questions:
            claim = str(row.get("claim") or "").strip() or "(no claim)"
            lever = str(row.get("world_lever") or "").strip()
            suffix = f" (`{lever}`)" if lever else ""
            lines.append(f"- {claim}{suffix}")
        lines.append("")
    lines.extend(["## Acquire targets (not runnable on this freeze)", ""])
    if interventional:
        for item in interventional:
            lines.append(f"- {item} — harvest / import / derive; do not probe here")
        lines.append("")
    else:
        lines.extend(["- (no unused interventional lever)", ""])
    return "\n".join(lines)


def render_archive(records: list[Mapping[str, Any]] | None = None) -> str:
    """This mission's notes. Not Dualverse Station mathematics."""
    lines = [
        "# Archive",
        "",
        "Non-scoring notes this mission left for the next landing.",
        "Acquire recipes and theory sketches may be published here without a probe.",
        "They cannot corroborate.",
        "",
    ]
    rows = list(records or [])
    if not rows:
        lines.extend(["(empty this mission)", ""])
        return "\n".join(lines)
    for row in rows:
        kind = idea_kind_of(row)
        pair = row.get("pair") or []
        combo = f"{pair[0]} × {pair[1]}" if len(pair) >= 2 else str(row.get("card_id") or "")
        claim = str(row.get("claim") or "").strip()
        verdict = str(row.get("verdict") or "").strip()
        extra = f" — {verdict}" if verdict else ""
        lines.append(f"- `{kind}` {combo}{extra}")
        if claim:
            lines.append(f"  {claim}")
    lines.append("")
    return "\n".join(lines)
