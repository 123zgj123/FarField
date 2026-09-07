"""Declared venue review axes. Prompt colour, never a kill.

A venue is how a later writing skill will review THIS claim, not a
license to rename the scientific object. Domain lock still wins.
These lines join the value rubric: they cannot kill, revive, climb,
or rewrite expected_direction. Official LaTeX lives in
jin-s13/ai-research-writing-skill — FarField does not vendor it.
"""

from __future__ import annotations

from typing import Any


class VenueError(ValueError):
    """Unknown or empty-in-the-wrong-way venue id."""


WHEEL = "https://github.com/jin-s13/ai-research-writing-skill"

# Public review questions, not a homemade score sheet. Empty id skips.
VENUES: dict[str, dict[str, Any]] = {
    "neurips": {
        "kind": "ml_empirical",
        "questions": (
            "What ablation isolates the named mechanism from the strongest alternative?",
            "Which published baseline and metric would a later paper report?",
            "Does the mechanism transfer past this dataset or architecture?",
        ),
    },
    "icml": {
        "kind": "ml_empirical",
        "questions": (
            "What ablation isolates the named mechanism from the strongest alternative?",
            "Which published baseline and metric would a later paper report?",
            "Is the claim identified, or only a performance delta?",
        ),
    },
    "iclr": {
        "kind": "ml_empirical",
        "questions": (
            "What would falsify the mechanism rather than the implementation?",
            "Which baseline is the honest comparison, not a weak stand-in?",
        ),
    },
    "acl": {
        "kind": "nlp_empirical",
        "questions": (
            "Which linguistic phenomenon is measured, not just a leaderboard delta?",
            "Does the claim survive a different domain or language?",
        ),
    },
    "cav": {
        "kind": "verification",
        "questions": (
            "What is the soundness or completeness claim, and what would refute it?",
            "Which artifact would a reviewer rerun?",
        ),
    },
    "osdi": {
        "kind": "systems",
        "questions": (
            "Which workload and overhead would a later systems paper report?",
            "What is the implementation boundary the claim does not cross?",
        ),
    },
    "stoc": {
        "kind": "theory",
        "questions": (
            "What bound, reduction, or impossibility is claimed?",
            "Which assumption, if dropped, collapses the argument?",
        ),
    },
    "soda": {
        "kind": "theory",
        "questions": (
            "What bound, reduction, or impossibility is claimed?",
            "Which assumption, if dropped, collapses the argument?",
        ),
    },
}

_ALIASES = {
    "nips": "neurips",
    "neurips 2026": "neurips",
    "icml 2026": "icml",
    "iclr 2026": "iclr",
    "iclr 2027": "iclr",
}


def normalize_venue(name: str | None) -> str:
    """Empty stays empty. Unknown names refuse so a typo cannot silently skip."""
    text = " ".join(str(name or "").strip().lower().split())
    if not text:
        return ""
    text = _ALIASES.get(text, text)
    if text not in VENUES:
        known = ", ".join(sorted(VENUES))
        raise VenueError(
            f"venue {name!r} is not a declared profile; known: {known}. "
            "Leave --venue empty to skip. Venue lines cannot kill a card."
        )
    return text


def venue_value_lines(venue: str) -> tuple[str, ...]:
    """Shown with the value rubric. Empty keeps generation prompts identical."""
    profile = VENUES.get(str(venue or "").strip().lower()) or {}
    questions = tuple(profile.get("questions") or ())
    if not questions:
        return ()
    return (
        f"Declared venue {venue} (review questions, cannot kill, cannot rename the object):",
        *questions,
        f"Draft later with {WHEEL} and that venue's official template.",
    )


def venue_writing_note(venue: str) -> str:
    """Appended to compile-artifact WRITING.md. Empty adds nothing."""
    vid = str(venue or "").strip().lower()
    if not vid or vid not in VENUES:
        return ""
    questions = "\n".join(f"- {item}" for item in VENUES[vid]["questions"])
    return (
        f"\nDeclared venue: {vid}\n"
        f"Feed attested.json to {WHEEL} and fetch that venue's official template.\n"
        f"Review questions (already shown at generation; still cannot invent results):\n"
        f"{questions}\n"
    )
