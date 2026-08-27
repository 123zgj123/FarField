"""Two bounded loops: rewrite the same idea, or rewrite the same test.

Far jumps are a cap plus a stop rule, not a third evolution loop.
Crossover of two parents is not on the product path.

Distant exploration stops only when an attested freeze has already
discriminated (WORLD supports). A registered experiment, a briefing, or
supports on invented / constructed data is not a reason to stop jumping
— those are not verification of the idea on its scientific object.
"""

from __future__ import annotations

from typing import Any

from .world import WORLD_KINDS


DEFAULT_EXPLORE = "auto"
DEFAULT_JUMP_CAP = 4
POLISH_AUTO = -1
POLISH_AUTO_CAP = 2
IDEA_AUTO = -1
IDEA_AUTO_CAP = 2

# Graph pair deaths cannot be repaired by rewriting the claim. A new
# landing is the fallback. Text-judge deaths can: the pair is still legal.
GRAPH_PAIR_MARKS = (
    "already_combined",
    "implied_by",
    "hub",
    "shortcut",
    "graph",
    "pair_not_already_combined",
    "endpoint_is_not_a_concept_hub",
    "pair_not_implied_by_neighborhood",
)


def normalize_explore(mode: str | None) -> str:
    text = str(mode or DEFAULT_EXPLORE).strip().lower()
    if text in ("fixed", "preset", "all"):
        return "fixed"
    return "auto"


def resolve_idea_rounds(requested: int) -> tuple[int, str]:
    """Map the idea-evolution knob to (extra refines of the same card, mode).

    `-1` means auto: up to `IDEA_AUTO_CAP` rewrites of the *same pair*
    after the first draft. `0` is the one-shot used by tests. A positive
    integer is an explicit extra-attempt cap. This is not a new far jump
    and not a polish of the briefing.
    """
    n = int(requested)
    if n < 0:
        return IDEA_AUTO_CAP, "auto"
    return max(0, n), "fixed"


def pair_structurally_dead(killed_by: list[str] | None) -> bool:
    """True when the *pair* failed the graph oracle.

    Rewriting claim/prediction cannot un-hub a node. Those deaths restore
    a new far landing instead of burning idea_rounds.
    """
    for name in killed_by or []:
        lowered = str(name).lower()
        if any(mark in lowered for mark in GRAPH_PAIR_MARKS):
            return True
    return False


def decide_idea_refine(
    *,
    attempt: int,
    cap: int,
    killed_by: list[str] | None = None,
    alive: bool = False,
) -> dict[str, Any]:
    """Whether to rewrite the same idea in this mission.

    Only text-gate deaths are worth a refine: the model needs the kill
    names in H so the next draft is a better idea, not a new pair.
    World gaps construct the experiment's world; they do not rewrite the
    claim to fit a freeze catalog.
    """
    cap = max(0, int(cap))
    attempt = max(0, int(attempt))
    if alive:
        return {
            "continue": False,
            "action": "stop",
            "reason": "entered",
            "detail": "this idea entered research; further rewrites would be polish",
        }
    if pair_structurally_dead(killed_by):
        return {
            "continue": False,
            "action": "stop",
            "reason": "graph_pair",
            "detail": (
                "the pair failed the graph oracle; rewriting the claim "
                "cannot repair it — open a new landing"
            ),
        }
    if not killed_by:
        return {
            "continue": False,
            "action": "stop",
            "reason": "no_text_kill",
            "detail": "nothing textual to repair on this draft",
        }
    if attempt >= cap:
        return {
            "continue": False,
            "action": "stop",
            "reason": "hit_cap",
            "detail": f"attempt {attempt + 1} of {cap + 1} still ineligible",
        }
    killers = ", ".join(str(item) for item in (killed_by or [])[:4])
    return {
        "continue": True,
        "action": "refine_idea",
        "reason": "gate",
        "detail": (
            f"rewrite the same pair so {killers} cannot recur; "
            "do not open a new distant landing"
        ),
    }


def resolve_polish_rounds(requested: int) -> tuple[int, str]:
    """Map the user-facing polish knob to (cap, mode).

    `-1` means auto: try up to `POLISH_AUTO_CAP` reviews, then
    `keep_going` stops. `0` still means skip polish entirely.
    """
    n = int(requested)
    if n < 0:
        return POLISH_AUTO_CAP, "auto"
    return max(0, n), "fixed"


def _world_support(row: dict[str, Any]) -> bool:
    """True when supports was measured on an attested freeze the script read."""
    if row.get("prior_kills") or row.get("killed"):
        return False
    if row.get("verdict") != "supports":
        return False
    kind = str(row.get("probe_kind") or "").upper()
    return kind in WORLD_KINDS


def has_live_line(records: list[dict[str, Any]] | None) -> bool:
    """True when a WORLD `supports` is on the books.

    SYNTHETIC or GENERATED supports are coherence, not a live scientific
    line. Spray and self-evolution read this, not bare `verdict`.
    """
    return any(_world_support(row) for row in records or [])


def has_research_plan(records: list[dict[str, Any]] | None) -> bool:
    """True when another far jump would be spray, not a better idea.

    Only an attested-world support is verification deep enough to stop
    distant exploration. A registered experiment, a briefing, or supports
    on invented or constructed data must not close the jump budget.
    """
    return has_live_line(records)


def decide_next(
    *,
    opened: int,
    cap: int,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Whether to open another generation slot.

    Stop when an attested freeze already supported a line, or the cap is
    hit. Continue through synthetic plans so distant exploration stays
    complete until the object has spoken, or the budget is gone.
    """
    cap = max(0, int(cap))
    opened = max(0, int(opened))
    if has_research_plan(records):
        return {
            "continue": False,
            "action": "stop",
            "reason": "has_plan",
            "detail": (
                "an attested-world support is already on the books; "
                "more far jumps would be spray"
            ),
        }
    if opened >= cap:
        return {
            "continue": False,
            "action": "stop",
            "reason": "hit_cap",
            "detail": f"opened {opened} of cap {cap} with no attested-world support",
        }
    return {
        "continue": True,
        "action": "continue",
        "reason": "no_plan",
        "detail": (
            "no WORLD support yet; keep opening distant landings until "
            "the object discriminates or the cap is hit"
        ),
    }
