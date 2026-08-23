"""In-loop selection among cards that already entered research.

The executable gates have already spoken. This module used to collapse
survivors into `T(elo) + β·T(alienness)` and let that number pick the
next generation's lead. P2 showed those features do not predict
post-T realization; HindSight showed LLM novelty scores anti-correlate
with future impact. The tournament still runs and is recorded as a
colleague's opinion. It cannot kill, revive, rewrite a probe, climb the
ladder, or — as of this module's selection keys — choose the lead.

Online selection is lexicographic process control, not an estimate of
scientific value V* (novel pre-T and realized post-T). Alienness stays
the QD cell axis. Elo breaks ties only.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .archive import DebateRefused, EloTable, debate
from .iterate import pipeline_bottleneck
from .parallel import map_parallel
from .score import score_pool
from .wiki import aimed_gap

DEFAULT_BETA = 1.0
WORLD_KINDS = frozenset({"WORLD", "REAL", "FIXTURE"})

# Shown to the generator (cannot kill). Empty-tuple callers keep prompt bytes.
# These are process constraints, not a novelty-as-value rubric.
VALUE_RUBRIC: tuple[str, ...] = (
    "prefer a claim whose two-arm test would separate on the bound attested world, not an invented dataset",
    "prefer aiming at a literature-gap sentence from the wiki; do not write the limitation as a solved claim",
    "prefer a prediction whose either-outcome would discriminate the mechanism from the strongest alternative",
    "do not repeat a combination or wording this mission's gates already killed",
)


def value_lines(rubric: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Prompt-ready process criteria. Separate from kill-criteria: failing
    these must not void a card."""
    return tuple(rubric or VALUE_RUBRIC)


def is_world_kind(kind: Any) -> bool:
    return str(kind or "").upper() in WORLD_KINDS


def evidence_class(row: Mapping[str, Any]) -> int:
    """Lower is better. Not an estimate of scientific value.

    WORLD-supports beat WORLD-uninformative beat SYNTHETIC-supports.
    Closed and weakened lines sort last so they cannot become the lead.
    """
    if row.get("killed") or row.get("prior_kills") or row.get("verdict") == "weakens":
        return 90
    world = is_world_kind(row.get("probe_kind"))
    verdict = str(row.get("verdict") or "")
    if world and verdict == "supports":
        return 0
    if world and verdict == "uninformative":
        return 1
    if world:
        return 2
    if verdict == "supports":
        return 3
    if verdict == "uninformative":
        return 4
    return 5


def host_rank(row: Mapping[str, Any]) -> int:
    ok = row.get("host_ok")
    if ok is True:
        return 0
    if ok is False:
        return 2
    return 1


def selection_key(row: Mapping[str, Any]) -> tuple:
    """Lexicographic process control. Elo is the last numeric tie-break."""
    openness = {True: 0, None: 1, False: 2}
    return (
        1 if row.get("prior_kills") else 0,
        evidence_class(row),
        0 if row.get("wiki_gap_hit") else 1,
        host_rank(row),
        openness.get(row.get("open_today"), 1),
        0 if row.get("has_brief") else 1,
        0 if row.get("probe_ran") else 1,
        -int(row.get("papers") or 0),
        -float(row.get("value_score") or 0),
        str(row.get("card_id") or ""),
    )


def archive_fitness(row: Mapping[str, Any]) -> float:
    """QD cell score: evidence class, not Elo. Alienness is the cell axis."""
    klass = evidence_class(row)
    if klass >= 90:
        return 0.0
    return (
        (10 - klass) * 100.0
        + (10.0 if row.get("wiki_gap_hit") else 0.0)
        + (5.0 if row.get("host_ok") is True else 0.0)
    )


def attach_observations(
    rows: Sequence[Mapping[str, Any]],
    gaps: Sequence[str],
) -> None:
    """Fill wiki_gap_hit and pipeline_bottleneck on live records. In place."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        gap = aimed_gap(
            str(row.get("claim") or ""),
            str(row.get("mechanism") or ""),
            gaps,
        )
        row["wiki_gap"] = gap
        row["wiki_gap_hit"] = bool(gap)
        row["pipeline_bottleneck"] = pipeline_bottleneck(row)


def tournament_pool(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Cards the value judge is allowed to see."""
    live = []
    for row in rows:
        if row.get("killed") or row.get("prior_kills"):
            continue
        if row.get("verdict") == "weakens":
            continue
        if not row.get("claim"):
            continue
        live.append(dict(row))
    return live


def run_tournament(
    client: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    rng_seed: str,
    workers: int = 1,
) -> tuple[EloTable, list[dict[str, Any]]]:
    """Round-robin debates over the live pool. One card: no debate, Elo 1000.

    A debate refusal is recorded on that pairing and skipped; the rest of
    the tournament still stands. The judge never sees card ids or scores.
    Pairings run concurrently; Elo is applied afterwards in pair order so
    a parallel tournament replays the same ratings as a serial one.
    """
    pool = tournament_pool(rows)
    table = EloTable()
    for card in pool:
        table.rating(str(card["card_id"]))
    pairs = [
        (card_a, card_b)
        for i, card_a in enumerate(pool)
        for card_b in pool[i + 1 :]
    ]

    def _one(pair: tuple[Mapping[str, Any], Mapping[str, Any]]) -> dict[str, Any]:
        card_a, card_b = pair
        try:
            verdict = debate(client, card_a, card_b, rng_seed=rng_seed)
        except DebateRefused as exc:
            return {
                "refused": True,
                "a": card_a["card_id"],
                "b": card_b["card_id"],
                "unlock_condition": exc.record.unlock_condition,
            }
        return verdict.to_dict()

    debates = map_parallel(_one, pairs, workers)
    for pair, payload in zip(pairs, debates):
        if payload.get("refused"):
            continue
        table.record(str(payload["winner_card"]), str(payload["loser_card"]))
    return table, debates


def score_live(
    rows: Sequence[Mapping[str, Any]],
    elo: Mapping[str, float],
    *,
    beta: float = DEFAULT_BETA,
) -> list[dict[str, Any]]:
    """Attach selection scores. Rows that did not enter the pool get 0.

    Plausibility here is the tournament Elo (LLM value among live cards),
    not neighborhood Jaccard — the gates already used Jaccard as a kill.
    Reusing it as a graded signal would double-count the oracle. Alienness
    is the far-field term the design insists must stay in the combination.
    """
    pool = tournament_pool(rows)
    if not pool:
        return []
    scored = score_pool(
        [
            {
                "card_id": row["card_id"],
                "pair_nodes": tuple(row.get("pair_nodes") or row.get("pair") or ("", "")),
                "plausibility_signal": float(elo.get(str(row["card_id"]), 1000.0)),
                "alienness": float(row.get("alienness") or 0.0),
            }
            for row in pool
        ]
    )
    by_id = {item.card_id: item for item in scored}
    ranked = []
    for row in pool:
        item = by_id[str(row["card_id"])]
        ranked.append(
            {
                "card_id": item.card_id,
                "pair": list(row.get("pair") or []),
                "elo": round(float(elo.get(str(row["card_id"]), 1000.0)), 2),
                "alienness": item.alienness,
                "t_value": round(item.t_plausibility, 6),
                "t_alienness": round(item.t_alienness, 6),
                "beta": beta,
                "score": round(item.score(beta), 6),
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["card_id"]))
    return ranked


