"""Settled campaign trajectories: the input `F` reads and nothing else may.

`F` is defined as `已结算轨迹 → 提议改进漂移或判定的 ΔP` (DESIGN §2.4), so a
trajectory is only eligible once the campaign settled, judgment was in the loop,
and every far card was actually probed. Training on ineligible trajectories
reproduces the failure the contract warns about: a meta loop attached to hollow
runs.

A trajectory carries **no post-T labels**. `F` proposes policy for the walker, so
it may only read what the walker could see: what was declared, what an executed
falsifier killed, what survived. Held-out value stays on the evaluator side and
is used to *judge* `F0(T*)` against `F1(T*)`, never to train either.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .ledger import content_digest

TRAJECTORY_PATH = Path(".farfield") / "trajectories.jsonl"
T_STAR_PATH = Path(".farfield") / "t_star.json"


class TrajectoryError(ValueError):
    pass


@dataclass(frozen=True)
class Trajectory:
    campaign_id: str
    seed: str
    settled_status: str
    family: str
    dataset: str
    improved: bool
    focus: str
    killed: tuple[str, ...]
    hypotheses: tuple[dict[str, Any], ...]
    purchases: tuple[dict[str, Any], ...]
    cards: tuple[dict[str, Any], ...]
    k_digest: str
    m_digest: str
    near_control: str = "absent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "seed": self.seed,
            "settled_status": self.settled_status,
            "near_control": self.near_control,
            "family": self.family,
            "dataset": self.dataset,
            "improved": self.improved,
            "focus": self.focus,
            "killed": list(self.killed),
            "hypotheses": [dict(item) for item in self.hypotheses],
            "purchases": [dict(item) for item in self.purchases],
            "cards": [dict(item) for item in self.cards],
            "k_digest": self.k_digest,
            "m_digest": self.m_digest,
        }

    def digest(self) -> str:
        return content_digest(self.to_dict())


def eligibility(dossier: Any, settled_status: str) -> tuple[bool, str]:
    """Why a campaign may or may not enter `T*`."""
    if settled_status != "done":
        return False, f"settlement status is {settled_status!r}, not done"
    if dossier.kind != "research":
        return False, f"dossier kind is {dossier.kind!r}, not research"
    if dossier.state.get("pre_judgment") is not False:
        return False, "judgment was not in the loop for this campaign"
    far = [card for card in dossier.conjectures[1:]]
    if not far:
        return False, "no far-field cards were generated"
    unprobed = [card for card in far if not card.get("probed")]
    if unprobed:
        return False, f"{len(unprobed)} far card(s) were never probed"
    return True, "settled, judged, and probed"


def trajectory_from_dossier(
    dossier: Any, settled_status: str, judge: Any, world: Any, graph: Any
) -> Trajectory:
    eligible, reason = eligibility(dossier, settled_status)
    if not eligible:
        raise TrajectoryError(f"campaign is not eligible for T*: {reason}")
    experiment = dossier.experiments[0] if dossier.experiments else {}
    purchases = [
        {
            "e_id": purchase.e_id,
            "is_core_falsifier": bool(purchase.is_core_falsifier),
            "declared_pairs": [list(pair) for pair in purchase.pair_ids],
            "realized_pairs": (
                [list(pair) for pair in sorted(purchase.realized)]
                if purchase.realized is not None
                else None
            ),
            "dv": judge.dv(purchase.e_id) if purchase.realized is not None else None,
        }
        for purchase in judge.purchases
    ]
    cards = [_card_record(card, graph) for card in dossier.conjectures[1:]]
    hypotheses = [
        {"id": hyp.id, "zone": hyp.zone, "is_core": bool(hyp.is_core)}
        for hyp in world.hypotheses
    ]
    return Trajectory(
        campaign_id=dossier.campaign_id,
        seed=dossier.seed,
        settled_status=settled_status,
        near_control=str(dossier.state.get("near_control") or "absent"),
        family=str(experiment.get("family") or "unknown"),
        dataset=str(experiment.get("dataset") or "unknown"),
        improved=bool(experiment.get("improved")),
        focus=str(world.focus),
        killed=tuple(sorted(judge.killed)),
        hypotheses=tuple(hypotheses),
        purchases=tuple(purchases),
        cards=tuple(cards),
        k_digest=str(dossier.state.get("k_digest") or "none"),
        m_digest=str(dossier.state.get("m_digest") or "none"),
    )


def _card_record(card: dict[str, Any], graph: Any) -> dict[str, Any]:
    """One card as `F` may see it: walker-side features and a walker-side label.

    `survived` is the outcome of the card's own declared falsifier, which the
    walker ran. Post-T confirmation is deliberately absent: it belongs to the
    evaluator, which judges `F`'s proposal instead of feeding it.
    """
    from .graph import node_degree

    nodes = list(card.get("nodes") or [])
    endpoint = nodes[-1] if nodes else ""
    probed = bool(card.get("probed"))
    killed = bool(card.get("killed"))
    return {
        "id": card.get("id"),
        "mode": card.get("mode"),
        "drift_operator": card.get("drift_operator"),
        "graph_distance": card.get("graph_distance"),
        "endpoint": endpoint,
        "path_length": max(0, len(nodes) - 1),
        "endpoint_degree": node_degree(graph, endpoint) if endpoint else 0,
        "endpoint_out_degree": len(graph.outgoing.get(endpoint, ())) if endpoint else 0,
        "independence_signature": card.get("independence_signature"),
        "probed": probed,
        "killed": killed,
        "survived": probed and not killed,
    }


def append_trajectory(project_dir: Path | str, trajectory: Trajectory) -> str:
    path = Path(project_dir) / TRAJECTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")
    return trajectory.digest()


def load_trajectories(project_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(project_dir) / TRAJECTORY_PATH
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def freeze_t_star(
    project_dir: Path | str, rows: Sequence[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Pin the current trajectory set so two updaters can be given the same `T*`.

    INV-7 `frozen-P` compares `F0(T*)` with `F1(T*)` on identical trajectories, so
    `T*` needs a digest that changes the moment the set changes.
    """
    rows = list(rows if rows is not None else load_trajectories(project_dir))
    if not rows:
        raise TrajectoryError("cannot freeze an empty T*")
    ids = [str(row.get("campaign_id") or "") for row in rows]
    if any(not item for item in ids):
        raise TrajectoryError("every trajectory needs a campaign_id")
    if len(set(ids)) != len(ids):
        raise TrajectoryError("T* campaign ids must be unique")
    frozen = {
        "n": len(rows),
        "campaign_ids": sorted(ids),
        "digest": content_digest(rows),
        "rows": rows,
    }
    path = Path(project_dir) / T_STAR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return frozen


def load_t_star(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise TrajectoryError("T* file has no rows")
    if content_digest(rows) != payload.get("digest"):
        raise TrajectoryError("T* digest does not match its rows; it was edited")
    return payload
