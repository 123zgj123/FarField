"""Verifier evolution with a held-out gate: shadow judges earn lethality.

Routing already earns its changes on missions it was not fitted to; the
Verifier slot gets the same discipline. A proposed text predicate enters
the bench as a *shadow* judge: it runs on every card, its flags are
logged per card, but it cannot kill. At mission end `maybe_promote`
reads the trajectory log with the same even/odd split as routing:

    a shadow judge becomes lethal only if it flagged at least one card
    that WORLD evidence later weakened, on both the nomination half and
    the held-out half, and it never flagged a WORLD-supported card or
    errored anywhere in the log.

That is τ_judge in executable form: the Verifier changes only when the
change predicted real-world failure out of sample. Elo, value_score and
SYNTHETIC verdicts do not appear anywhere in the settlement.

`tau_judge_report` is the cross-mission readout the promotion rule does
not need but the researcher does: per judge, are later missions dying
less on the same text class than earlier ones.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .predicates import (
    Predicate,
    PredicateError,
    PredicateRefused,
    compile_predicate,
    run as run_predicate,
)
from .routing import load_log, settlement_missions
from .world import WORLD_KINDS

MIN_MISSIONS = 6

# The shadow bench is cheap to enter (whitelist compile only) because a
# shadow judge has no power; τ_judge is the real gate. It is still
# capped: an unbounded bench turns settlement into a multiple-comparisons
# machine, and every seat is one more thing a reader must audit.
SHADOW_CAP = 8

# Propose only when the mission shows a repeated failure the gates
# missed: at least this many entered cards ended WORLD-weakens or died
# in the probe-method loop.
SHADOW_TRIGGER = 2


class VerifierError(Exception):
    """The judge bench refused to load or accept: digest or whitelist."""


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _worldish(kind: Any) -> bool:
    return str(kind or "").upper() in WORLD_KINDS


def load_bench(path: Path) -> dict[str, list[dict[str, Any]]]:
    """The persisted bench: `shadow` rows watch, `lethal` rows kill."""
    if not Path(path).is_file():
        return {"shadow": [], "lethal": []}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    bench = payload.get("bench")
    if not isinstance(bench, dict) or payload.get("digest") != _digest(bench):
        raise VerifierError(
            f"{path} does not match its own digest; refusing a judge bench"
            " that was edited outside this module"
        )
    return {
        "shadow": list(bench.get("shadow") or []),
        "lethal": list(bench.get("lethal") or []),
    }


def save_bench(path: Path, bench: dict[str, list[dict[str, Any]]]) -> None:
    body = {
        "shadow": list(bench.get("shadow") or []),
        "lethal": list(bench.get("lethal") or []),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"bench": body, "digest": _digest(body)},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def compile_bench(rows: Sequence[dict[str, Any]]) -> tuple[Predicate, ...]:
    """Recompile stored rows through the same whitelist as any proposal."""
    return tuple(
        compile_predicate(
            str(row["name"]),
            str(row["source"]),
            provenance=str(row.get("provenance") or "bench"),
            rationale=str(row.get("rationale") or ""),
        )
        for row in rows
    )


def add_shadow(
    path: Path,
    *,
    name: str,
    source: str,
    provenance: str,
    rationale: str = "",
) -> dict[str, Any]:
    """Admit one predicate to the shadow bench. It cannot kill from here.

    The whitelist compiler runs first; a duplicate name anywhere on the
    bench is refused, because a judge that shadows itself would settle
    its own promotion.
    """
    try:
        compile_predicate(name, source, provenance=provenance, rationale=rationale)
    except PredicateRefused as exc:
        raise VerifierError(str(exc)) from exc
    from .filelock import file_lock

    with file_lock(path):
        bench = load_bench(path)
        taken = {str(row.get("name")) for row in bench["shadow"] + bench["lethal"]}
        if name in taken:
            raise VerifierError(f"judge {name!r} is already on the bench")
        bench["shadow"].append(
            {
                "name": name,
                "source": source,
                "provenance": provenance,
                "rationale": rationale,
            }
        )
        save_bench(path, bench)
        return {"name": name, "bench": "shadow", "shadow": len(bench["shadow"])}


PROPOSE_TEMPLATE = """You are writing one executable review predicate for a research-card pipeline.

These cards passed every installed gate, entered research, and then the evidence went against them (the probe weakened the claim, or the probe method kept failing):

{failures}

These cards entered and were supported; your predicate MUST pass every one of them:

{passes}

Write ONE Python predicate that would have flagged at least one failed card before its probe ran, using only the card's own text fields. Judge the failure mechanism, not the topic words of these examples.

Rules:
- exactly one function: def check(card)
- card is a dict with keys: claim, mechanism, prediction, falsifier, concept_a, concept_b, alienness, plausibility_signal
- allowed: re, math, and plain builtins; no imports, no underscore names, no nested functions
- return True to pass a card, False to flag it

Answer with JSON only:
{{"name": "<lowercase_identifier>", "source": "<the full function source>", "rationale": "<one sentence: which failure class this catches>"}}
"""

_FENCE = re.compile(r"^```[a-z]*\n|\n```$")


def _card_lines(views: Sequence[dict[str, Any]], cap: int = 4) -> str:
    rows = []
    for view in list(views)[:cap]:
        claim = str(view.get("claim") or "")[:300]
        mechanism = str(view.get("mechanism") or "")[:300]
        rows.append(f"- claim: {claim}\n  mechanism: {mechanism}")
    return "\n".join(rows) if rows else "- (none this mission)"


def _refused(reason: str) -> dict[str, Any]:
    return {"admitted": False, "reason": reason}


def propose_shadow(
    client: Any,
    *,
    failures: Sequence[dict[str, Any]],
    passes: Sequence[dict[str, Any]],
    bench_path: Path,
    cap: int = SHADOW_CAP,
) -> dict[str, Any]:
    """One model-written predicate onto the shadow bench, or a named refusal.

    The local screen is deliberately weak — flag at least one of this
    mission's failures, flag none of its supports, never error — because
    a shadow judge cannot kill. Lethality is only earned later, by
    `maybe_promote`'s held-out settlement. Every refusal is returned as a
    record, never raised: a bad proposal must not touch the mission.
    """
    bench = load_bench(bench_path)
    if len(bench["shadow"]) >= cap:
        return _refused(
            f"the shadow bench is full ({cap}); settle or retire before proposing"
        )
    prompt = PROPOSE_TEMPLATE.format(
        failures=_card_lines(failures), passes=_card_lines(passes)
    )
    completion = client.complete(prompt, purpose="shadow-judge")
    text = _FENCE.sub("", str(completion.text).strip())
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _refused("the proposal was not JSON; no predicate was benched")
    name = str(payload.get("name") or "").strip()
    source = str(payload.get("source") or "")
    rationale = str(payload.get("rationale") or "").strip()
    try:
        predicate = compile_predicate(
            name, source, provenance="model:shadow", rationale=rationale
        )
    except PredicateRefused as exc:
        return _refused(f"whitelist compile refused: {exc}")
    flagged = 0
    for view in failures:
        try:
            if not run_predicate(predicate, view):
                flagged += 1
        except PredicateError as exc:
            return _refused(f"the predicate errored on a failed card: {exc}")
    if not flagged:
        return _refused(
            "the predicate flags none of this mission's failed cards; it"
            " would shadow nothing it was proposed for"
        )
    for view in passes:
        try:
            if not run_predicate(predicate, view):
                return _refused(
                    "the predicate flags a supported card from this mission;"
                    " that kill would already be wrong"
                )
        except PredicateError as exc:
            return _refused(f"the predicate errored on a supported card: {exc}")
    try:
        add_shadow(
            bench_path,
            name=name,
            source=source,
            provenance="model:shadow",
            rationale=rationale,
        )
    except VerifierError as exc:
        return _refused(str(exc))
    return {
        "admitted": True,
        "name": name,
        "rationale": rationale,
        "flagged_failures": flagged,
        "must_pass": len(passes),
        "shadow_total": len(bench["shadow"]) + 1,
        "note": (
            "shadowing only: this predicate cannot kill until the held-out"
            " settlement promotes it"
        ),
    }


def shadow_flags(
    judges: Sequence[Predicate], view: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """(flags, errors) for one card view. A shadow judge can never kill:
    a False is a recorded flag, a crash is the judge's own problem."""
    flags: list[str] = []
    errors: list[str] = []
    for judge in judges:
        try:
            if not run_predicate(judge, view):
                flags.append(judge.name)
        except PredicateError:
            errors.append(judge.name)
    return flags, errors


def judge_stats(
    missions: Sequence[dict[str, Any]], names: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Per shadow judge, what its flags anticipated on these missions.

    Only WORLD-ish verdicts settle anything: a flag on a SYNTHETIC card
    is neither a catch nor a false kill, because the invented dataset
    cannot corroborate or refute the judge any more than the card.
    """
    stats = {
        name: {"flagged": 0, "caught_weakens": 0, "flagged_supports": 0, "errors": 0}
        for name in names
    }
    for mission in missions:
        for name in mission.get("shadow_errors") or []:
            if name in stats:
                stats[name]["errors"] += 1
        for card in mission.get("cards") or []:
            hits = [str(item) for item in (card.get("shadow_kills") or [])]
            world = _worldish(card.get("probe_kind"))
            verdict = card.get("verdict")
            for name in hits:
                if name not in stats:
                    continue
                stats[name]["flagged"] += 1
                if world and verdict == "weakens":
                    stats[name]["caught_weakens"] += 1
                if world and verdict == "supports":
                    stats[name]["flagged_supports"] += 1
    return stats


def maybe_promote(
    log_path: Path,
    bench_path: Path,
    *,
    min_missions: int = MIN_MISSIONS,
) -> dict[str, Any] | None:
    """The Verifier meta loop, safe to run at the end of every mission.

    Nominates on the even-indexed missions, confirms on the odd-indexed
    ones the judge was not nominated on, and moves a judge from shadow
    to lethal only when both halves caught a WORLD-weakens and neither
    half flagged a WORLD-supports. Returns the decision, or None when
    there is nothing to settle yet.
    """
    from .filelock import file_lock

    with file_lock(bench_path):
        bench = load_bench(bench_path)
        if not bench["shadow"]:
            return None
        missions = settlement_missions(load_log(log_path))
        if len(missions) < min_missions:
            return None
        names = [str(row["name"]) for row in bench["shadow"]]
        fit = judge_stats(missions[0::2], names)
        heldout = judge_stats(missions[1::2], names)
        promoted: list[str] = []
        rejected: list[dict[str, Any]] = []
        for name in names:
            errors = fit[name]["errors"] + heldout[name]["errors"]
            false_kills = fit[name]["flagged_supports"] + heldout[name]["flagged_supports"]
            if errors:
                reason = "the judge errored on logged cards; a crashing judge is not promotable"
            elif false_kills:
                reason = "the judge flagged a WORLD-supported card; that kill would have been wrong"
            elif not fit[name]["caught_weakens"]:
                reason = "no WORLD-weakens caught on the nomination half"
            elif not heldout[name]["caught_weakens"]:
                reason = "no WORLD-weakens caught on the held-out half"
            else:
                promoted.append(name)
                continue
            rejected.append(
                {
                    "judge": name,
                    "reason": reason,
                    "fit": fit[name],
                    "heldout": heldout[name],
                }
            )
        if promoted:
            rows = {str(row["name"]): row for row in bench["shadow"]}
            for name in promoted:
                row = dict(rows[name])
                row["promoted_by"] = {
                    "fit": fit[name],
                    "heldout": heldout[name],
                    "missions": len(missions),
                }
                bench["lethal"].append(row)
            bench["shadow"] = [
                row for row in bench["shadow"] if str(row["name"]) not in set(promoted)
            ]
            save_bench(bench_path, bench)
        return {
            "promoted": promoted,
            "rejected": rejected,
            "shadow_remaining": len(bench["shadow"]),
            "lethal_total": len(bench["lethal"]),
            "missions": len(missions),
        }


def tau_judge_report(missions: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Are later missions dying less on the same judge than earlier ones?

    Splits the log in half by time and counts, per judge name appearing
    in `killed_by`, kills in each half. A judge whose kills fall while
    cards keep flowing is teaching the generator; one whose kills hold
    steady is a tax the generator has not learned to pay. This is a
    readout, not a gate: it moves nothing.
    """
    half = len(missions) // 2
    report: dict[str, dict[str, Any]] = {}
    for index, mission in enumerate(missions):
        late = index >= half
        for card in mission.get("cards") or []:
            for name in card.get("killed_by") or []:
                row = report.setdefault(
                    str(name), {"early_kills": 0, "late_kills": 0}
                )
                row["late_kills" if late else "early_kills"] += 1
    for row in report.values():
        row["learning"] = row["late_kills"] < row["early_kills"]
    return report
