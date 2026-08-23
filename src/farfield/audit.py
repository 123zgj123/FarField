"""Wave E audit protocol. Writes audit_report.json, never a dossier success field."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .drift import generate
from .graph import PINNED_K, GraphSnapshot, load_graph, seed_neighborhood
from .ledger import content_digest
from .probe import probe_card
from .timeslice import (
    MIN_TIMESLICE_SEEDS,
    PINNED_T,
    HeldOutSet,
    load_evaluator_graph,
    load_heldout,
    load_walker_graph,
    successor_hits_confirmed,
    walker_lookup,
)
from .worker import CampaignError

UNMEASURED = "未测"

# The same floor the timeslice anchor uses, because it is the same question: one
# (snapshot, seed) layout tells you where that corpus author put the post-T
# edges, not whether the arm found anything.
MIN_VALUE_ROWS = MIN_TIMESLICE_SEEDS

# An arm is allowed to kill cards -- that is what a falsifier is for -- but an
# arm that kills a quarter of what later turned out to have post-T value is
# discarding the thing the campaign was run to find, so the verdict is withheld
# rather than reported as a win.
MAX_FAR_MISKILL = 0.25

SNAPSHOT_KEYS = ("snapshot_id", "graph", "full_graph", "heldout", "post_t_placement")


def load_registry(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("T") != PINNED_T:
        raise ValueError(f"registry T must be {PINNED_T}")
    if not str(payload.get("e2_status") or "").strip():
        raise ValueError("registry requires an e2_status")
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("registry requires a non-empty snapshots list")
    for snapshot in snapshots:
        for key in SNAPSHOT_KEYS:
            if not str(snapshot.get(key) or "").strip():
                raise ValueError(f"registry snapshot requires {key}")
        if not snapshot.get("seeds") or not snapshot.get("seed_nodes"):
            raise ValueError("registry snapshot requires seeds and seed_nodes")
        if len(snapshot["seeds"]) != len(snapshot["seed_nodes"]):
            raise ValueError("seeds and seed_nodes must be aligned")
    ids = [snapshot["snapshot_id"] for snapshot in snapshots]
    if len(set(ids)) != len(ids):
        raise ValueError("registry snapshot ids must be unique")
    if payload.get("T_star_campaign_ids"):
        seeds = {seed for snapshot in snapshots for seed in snapshot["seeds"]}
        if set(payload["T_star_campaign_ids"]) & seeds:
            raise ValueError("T* campaign ids must be disjoint from E1 seeds")
    return payload


def registry_seeds(payload: dict[str, Any]) -> list[tuple[dict[str, Any], str, str]]:
    """Flatten the registry into (snapshot, seed, seed_node) rows."""
    rows = []
    for snapshot in payload["snapshots"]:
        for seed, seed_node in zip(snapshot["seeds"], snapshot["seed_nodes"]):
            rows.append((snapshot, seed, seed_node))
    return rows


def registry_digest(payload: dict[str, Any]) -> str:
    return content_digest(payload)


def _inspectable(card: Any, neighborhood: set[str]) -> bool:
    nodes = set(card.nodes)
    has_prediction = bool(card.card.prediction.strip())
    has_falsifier = bool(card.card.cheapest_falsifier.strip())
    return not nodes <= neighborhood and has_prediction and has_falsifier


def score_arm(
    cards: tuple[Any, ...],
    neighborhood: set[str],
    heldout: HeldOutSet,
    full_graph: GraphSnapshot,
    *,
    cost: float,
    graph: GraphSnapshot,
    seed: str,
) -> dict[str, Any]:
    """Score one arm on the four columns.

    `reach` counts cards that are outside the seed k-hop **and survived an
    executed falsification attempt**. Merely being outside the k-hop is the
    generator's own filter: `dump_seed_refs` paths are inside by construction and
    drift paths are outside by construction, so a reach built on membership alone
    is pinned to 0 and 1 before either arm runs. `reach_definitional` marks an arm
    whose reach is pinned that way, and such a reach must not be counted as
    evidence that one arm beat the other.

    `value` is looked up on the **evaluator** graph: did the card's endpoint grow
    a post-T confirmed successor. Intersecting `confirmed` with the walker graph
    instead is empty by contract, which makes the column permanently unmeasurable
    rather than merely unmeasured.

    `falsekill` is the mis-kill rate: of the cards this arm actually killed, how
    many turn out to have post-T value. Lower is better. An arm that executed no
    kills has no mis-kill rate, and reporting 0.0 there would read as "kills
    nothing valuable" when the truth is "kills nothing".
    """
    n = max(1, len(cards))
    probed = [
        (card, probe_card(graph, card.to_conjecture_dict(seed), neighborhood))
        for card in cards
    ]
    outside = [
        (card, result) for card, result in probed if _inspectable(card, neighborhood)
    ]
    survived = [
        card for card, result in outside if result.probed and not result.killed
    ]
    reach = len(survived) / n
    reach_definitional = bool(cards) and all(
        set(card.nodes) <= neighborhood for card in cards
    )

    confirmed = heldout.confirmed
    value_measurable = any(node_id in full_graph.nodes for node_id in confirmed)

    def has_value(card: Any) -> bool:
        endpoint = card.nodes[-1] if card.nodes else ""
        return bool(endpoint) and successor_hits_confirmed(
            full_graph, endpoint, confirmed
        )

    value = (
        sum(1 for card in cards if has_value(card)) / n
        if value_measurable
        else UNMEASURED
    )

    killed_cards = [card for card, result in probed if result.probed and result.killed]

    if not value_measurable or not killed_cards:
        falsekill = UNMEASURED
    else:
        falsekill = sum(1 for card in killed_cards if has_value(card)) / len(
            killed_cards
        )

    return {
        "reach": reach,
        "reach_definitional": reach_definitional,
        "n_survived": len(survived),
        "value": value,
        "n_valued": (
            sum(1 for card in cards if has_value(card)) if value_measurable else None
        ),
        "falsekill": falsekill,
        "n_killed": len(killed_cards),
        "cost": cost,
        "n_cards": len(cards),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def reach_counted(rows: list[dict[str, Any]]) -> bool:
    """True when reach has an opponent that was allowed to leave the k-hop.

    The reach comparison is drift against the chance arm, not against the dump
    arm: the dump arm walks inside the k-hop by construction, so its reach is 0
    before it runs. A row with no chance arm, or with a chance arm whose cards
    all fell inside the k-hop anyway, still cannot count reach.
    """
    if not rows or any("chance" not in row for row in rows):
        return False
    return not any(
        row["far"]["reach_definitional"] or row["chance"]["reach_definitional"]
        for row in rows
    )


def value_certified(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Independent (snapshot, seed) pairs whose value column was measurable."""
    pairs = {
        (str(row["snapshot_id"]), str(row["seed_node"]))
        for row in rows
        if row["far"]["value"] != UNMEASURED and row["near"]["value"] != UNMEASURED
    }
    snapshots = {snapshot for snapshot, _ in pairs}
    certified = len(pairs) >= MIN_VALUE_ROWS and len(snapshots) >= 2
    return {
        "certified": certified,
        "n_pairs": len(pairs),
        "n_snapshots": len(snapshots),
        "reason": (
            "value measured on independent snapshots"
            if certified
            else (
                f"{len(pairs)} measured (snapshot, seed) pair(s) over "
                f"{len(snapshots)} snapshot(s); certification needs >= "
                f"{MIN_VALUE_ROWS} pairs over >= 2 snapshots, otherwise the column reports where one author placed the post-T edges"
            )
        ),
    }


def far_beats_near(rows: list[dict[str, Any]]) -> str:
    if any(
        row["far"]["value"] == UNMEASURED or row["near"]["value"] == UNMEASURED
        for row in rows
    ):
        return "incomplete_columns"
    if not value_certified(rows)["certified"]:
        return "value_uncertified"
    value_far = _mean([row["far"]["value"] for row in rows])
    value_near = _mean([row["near"]["value"] for row in rows])

    # An arm with no executed kills has no mis-kill rate, so it is dropped here
    # rather than folded in as a zero.
    miskills = [row["far"]["falsekill"] for row in rows if row["far"]["falsekill"] != UNMEASURED]
    if miskills and _mean(miskills) > MAX_FAR_MISKILL:
        return "far_miskill_over_bound"
    if not reach_counted(rows):
        # With no admissible reach column, value is all that is left to compare.
        return "far_better" if value_far > value_near else "not_far_better"

    # Reach against chance, and value at least not worse: going further only
    # counts as better if it did not cost anything on the column that matters.
    reach_far = _mean([row["far"]["reach"] for row in rows])
    reach_chance = _mean([row["chance"]["reach"] for row in rows])
    if reach_far > reach_chance and value_far >= value_near:
        return "far_better"
    return "not_far_better"


def run_audit(
    *,
    registry_path: Path | str,
    report_path: Path | str,
    repo_root: Path | str,
) -> dict[str, Any]:
    root = Path(repo_root)
    registry = load_registry(registry_path)
    digest = registry_digest(registry)
    envelope = registry["envelope"]
    rows = []
    for snapshot, seed, seed_node in registry_seeds(registry):
        graph = load_walker_graph(root / snapshot["graph"])
        heldout = load_heldout(root / snapshot["heldout"])
        full_graph = load_evaluator_graph(root / snapshot["full_graph"])
        for node_id in heldout.confirmed:
            try:
                walker_lookup(graph, node_id, heldout=heldout)
                raise RuntimeError("walker leaked a confirmed node")
            except PermissionError:
                pass
            if node_id in graph.nodes:
                raise ValueError(
                    f"confirmed node {node_id!r} must not be in the walker graph"
                )
        if seed_node not in graph.nodes:
            raise ValueError(f"seed node {seed_node!r} is not in G_<=T")
        neighborhood = seed_neighborhood(graph, seed_node, PINNED_K)
        near = generate(seed, graph, mode="dump_seed_refs", seed_node_id=seed_node)

        # A seed with nothing outside its k-hop is a fact about the corpus, not a
        # failed run, so the arm reports an empty column instead of aborting.
        try:
            far_cards = generate(seed, graph, mode="drift", seed_node_id=seed_node).cards
        except ValueError:
            far_cards = ()

        # Same for the chance arm, which draws from the same distant pool and so
        # falls silent on exactly the seeds drift does.
        try:
            chance_cards = generate(
                seed, graph, mode="random_far", seed_node_id=seed_node
            ).cards
        except ValueError:
            chance_cards = ()
        cost = float(envelope["literature_fetches"])
        rows.append(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "post_t_placement": snapshot["post_t_placement"],
                "seed": seed,
                "seed_node": seed_node,
                "near": score_arm(
                    near.cards,
                    neighborhood,
                    heldout,
                    full_graph,
                    cost=cost,
                    graph=graph,
                    seed=seed,
                ),
                "far": score_arm(
                    far_cards,
                    neighborhood,
                    heldout,
                    full_graph,
                    cost=cost,
                    graph=graph,
                    seed=seed,
                ),
                "chance": score_arm(
                    chance_cards,
                    neighborhood,
                    heldout,
                    full_graph,
                    cost=cost,
                    graph=graph,
                    seed=seed,
                ),
            }
        )
    counted = reach_counted(rows)
    certification = value_certified(rows)
    report = {
        "registry_digest": digest,
        "T": PINNED_T,
        "e1": rows,
        "e1_verdict": far_beats_near(rows),
        "reach_counted": counted,
        "reach_note": (
            "reach counts cards outside the seed k-hop that survived an executed "
            "falsification attempt, and is compared against the chance arm, which "
            "draws distant paths at random from the same pool"
            if counted
            else "reach is not counted: no arm in this report was both allowed outside the k-hop and independent of drift's selection rule"
        ),
        "value_certification": certification,
        "max_far_miskill": MAX_FAR_MISKILL,
        "e2": {"tau_F": "N/A", "reason": registry["e2_status"]},
        "columns": ["reach", "value", "falsekill", "cost"],
        "claim": "audit only; not a product success and not an updater improvement",
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def measure_registry_anchor(
    registry_path: Path | str, repo_root: Path | str
) -> list[dict[str, Any]]:
    """One timeslice measurement per (snapshot, seed) for the install anchor.

    A single measurement is never a value anchor, so the anchor is measured over
    the whole corpus registry and handed to `certify_timeslice_value`.
    """
    from .timeslice import measure_timeslice_value

    registry = load_registry(registry_path)
    root = Path(repo_root)
    measurements = []
    for snapshot, seed, seed_node in registry_seeds(registry):
        graph = load_walker_graph(root / snapshot["graph"])
        full_graph = load_evaluator_graph(root / snapshot["full_graph"])
        heldout = load_heldout(root / snapshot["heldout"])

        # A silent drift arm still has to appear in the anchor, as a zero rate
        # against the dump arm, or the anchor would be averaged over only the
        # seeds that happened to work.
        far_cards = ()
        try:
            far_cards = generate(seed, graph, mode="drift", seed_node_id=seed_node).cards
        except ValueError:
            far_cards = ()
        dump = generate(seed, graph, mode="dump_seed_refs", seed_node_id=seed_node)
        measurements.append(
            measure_timeslice_value(
                far_cards,
                dump.cards,
                full_graph,
                heldout.confirmed,
                seed_node_id=seed_node,
            )
        )
    return measurements


def run_tier_b_survey(project_dir: Path | str, seed: str, arxiv_fixture: Path | str):
    raise CampaignError(
        "Tier-B survey campaigns were the control arm; use `farfield research`"
    )
