"""The evidence grid: which cells a claim needs, which are filled, by what.

A single two-arm probe carried two jobs in FarField — feasibility filter
and promotion evidence — and every "uninformative" was one more rewrite
of the same design. The grid separates the jobs. A claim registers the
cells it needs to be defended (world × handle × observable × control);
each probe fills one cell with its EvidenceID and a *typed* outcome; the
packet and the paper stage read coverage, not a single verdict. Negative
results are kept with their type (blind measure, placebo-surviving
separation, object absent, under margin) because each type says
something different about what to do next. Positive results add
coverage and narrative eligibility; they do not, by themselves, mean the
mechanism was isolated — that is what the control cells are for.

Cells whose world does not exist yet say how it would be acquired
(derive / import / harvest), so "no executable plan" reads as "these
cells need these worlds", not as a dead line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ideakind import idea_kind_of, parse_idea_kind
from .lifecycle import PROBE_EVIDENCE, paper_section_for, typed_evidence_role

FAILURE_TYPES = ("blind_measure", "placebo_fail", "object_absent", "under_margin", "crashed")


def outcome_type(row: dict[str, Any]) -> str:
    """Typed reading of one attempt / evidence row."""
    verdict = str(row.get("verdict") or "")
    reason = str(row.get("reason") or row.get("error") or "")
    status = str(row.get("status") or "")
    if status and status not in {"ran", ""}:
        return "crashed"
    if row.get("dv_blind") or "dv_blind" in reason:
        return "blind_measure"
    if "did not consume the bound world" in reason or "placebo" in reason:
        return "placebo_fail"
    if row.get("object_absent"):
        return "object_absent"
    if verdict == "supports":
        return "supports"
    if verdict == "weakens":
        return "weakens"
    if verdict == "uninformative":
        return "under_margin"
    return "unknown"


def compile_grid(
    *,
    claim: str,
    world: dict[str, Any] | None,
    dynamics_card: dict[str, Any] | None,
    diagnosis: dict[str, Any] | None,
    attempts: list[dict[str, Any]] | None,
    catalog_worlds: list[dict[str, Any]] | None = None,
    required_properties: dict[str, bool] | None = None,
    idea_kind: str = "",
) -> dict[str, Any]:
    """Typed evidence cells. Probe still uses C1 two-arm cells."""
    kind = parse_idea_kind(idea_kind or (diagnosis or {}).get("idea_kind"))
    role = typed_evidence_role(kind)
    if kind != "probe":
        return _typed_grid(
            kind=kind,
            role=role,
            claim=claim,
            world=world or {},
            diagnosis=diagnosis or {},
            attempts=attempts or [],
        )
    world = world or {}
    diagnosis = diagnosis or {}
    props = dict(world.get("object_properties") or {})
    levers = list((dynamics_card or {}).get("levers") or {}) if isinstance(dynamics_card, dict) else []
    inert = set((dynamics_card or {}).get("inert") or []) if isinstance(dynamics_card, dict) else set()
    observable = str(diagnosis.get("world_observable") or "")
    lever = str(diagnosis.get("world_lever") or "")
    required = dict(required_properties or {})
    cells: list[dict[str, Any]] = []

    def live(handle: str, obs: str) -> bool:
        return bool(handle) and bool(obs) and f"{handle}/{obs}" not in inert

    # 1. The registered handle on the bound world.
    if lever and observable:
        cells.append(
            {
                "id": "C1.main",
                "kind": "observational" if lever.startswith("contrast:") else "interventional",
                "world": world.get("id"),
                "handle": lever,
                "observable": observable,
                "control": str(diagnosis.get("control_arm") or ""),
                "required": True,
                "status": "fillable" if live(lever, observable) else "inert_on_world",
            }
        )
    # 2. Observational contrasts on the same world, same observable — the
    #    cells a replayed history can actually fill.
    for handle in levers:
        if handle.startswith("contrast:") and handle != lever and observable:
            cells.append(
                {
                    "id": f"C1.contrast.{handle.split(':', 1)[1]}",
                    "kind": "observational",
                    "world": world.get("id"),
                    "handle": handle,
                    "observable": observable,
                    "control": "records outside the stratum, matched on episode template and updates per episode",
                    "required": False,
                    "status": "fillable" if live(handle, observable) else "inert_on_world",
                }
            )
    # 3. Mechanism isolation: a competing-explanation control.
    if diagnosis.get("alternative"):
        cells.append(
            {
                "id": "C1.alternative",
                "kind": "ablation",
                "world": world.get("id"),
                "handle": lever,
                "observable": observable,
                "control": str(diagnosis.get("alternative") or "")[:200],
                "required": True,
                "status": "fillable" if live(lever, observable) else "inert_on_world",
            }
        )
    # 4. Interventional validator cells need a world whose gate the agent
    #    owns and that can be re-run, not replayed.
    wants_validator_intervention = required.get("validators_mutable") is True or any(
        h in levers for h in ("freeze_validators", "replay_gate")
    )
    if wants_validator_intervention:
        dead = all(f"{h}/{observable}" in inert for h in ("freeze_validators", "replay_gate") if h in levers) if observable else True
        cells.append(
            {
                "id": "C1.intervene.validators",
                "kind": "interventional",
                "world": world.get("id") if not dead else None,
                "handle": "freeze_validators | replay_gate",
                "observable": observable or "false_accept_fraction",
                "control": "same proposal stream, gate unchanged; acceptance rate matched on a calibration split",
                "required": True,
                "status": (
                    "fillable"
                    if not dead
                    else "needs_world"
                ),
                "acquire": (
                    None
                    if not dead
                    else {
                        "how": "harvest",
                        "what": "a harness whose validators the agent may rewrite and that can be re-run under a frozen hidden oracle "
                        "(DGM-style import of output_dgm/, or an OpenEvolve variant with an evolvable evaluator cell)",
                        "properties": {"validators_mutable": True, "has_oracle": True},
                    }
                ),
            }
        )
    # 5. Replication on a second world of the same schema and compatible properties.
    siblings = [
        row
        for row in (catalog_worlds or [])
        if row.get("id") != world.get("id")
        and row.get("schema") == world.get("schema")
        and all(
            (row.get("object_properties") or {}).get(k) == v
            for k, v in required.items()
            if k in (row.get("object_properties") or {})
        )
    ]
    cells.append(
        {
            "id": "C1.replicate",
            "kind": "replication",
            "world": siblings[0]["id"] if siblings else None,
            "handle": lever,
            "observable": observable,
            "control": "same registered protocol, other harness family",
            "required": True,
            "status": "fillable" if siblings else "needs_world",
            "acquire": None
            if siblings
            else {
                "how": "derive | import",
                "what": f"a second {world.get('schema') or 'same-schema'} world with the claim's stated properties "
                "(another model's Live-SWE release derived the same way, or a published self-improvement run)",
                "properties": required,
            },
        }
    )
    # Fill from attempts: the registered handle's cell takes every attempt.
    filled: list[dict[str, Any]] = []
    for row in attempts or []:
        if not isinstance(row, dict):
            continue
        filled.append(
            {
                "cell": "C1.main",
                "attempt": row.get("attempt"),
                "exp_id": row.get("evidence_id") or row.get("experiment_digest") or "",
                "verdict": row.get("verdict"),
                "type": outcome_type(row),
                "treatment": row.get("treatment"),
                "control": row.get("control"),
            }
        )
    for cell in cells:
        mine = [f for f in filled if f["cell"] == cell["id"]]
        if mine:
            cell["fills"] = mine
            types = [f["type"] for f in mine]
            if "supports" in types:
                cell["status"] = "filled:supports"
            elif "weakens" in types:
                cell["status"] = "filled:weakens"
            else:
                cell["status"] = "filled:" + types[-1]
    required_cells = [c for c in cells if c.get("required")]
    covered = [c for c in required_cells if str(c.get("status", "")).startswith("filled:supports")]
    return {
        "claim": claim,
        "world": world.get("id"),
        "idea_kind": "probe",
        "evidence_role": PROBE_EVIDENCE,
        "paper_section": paper_section_for("probe"),
        "cells": cells,
        "coverage": {
            "required": len(required_cells),
            "supported": len(covered),
            "needs_world": sum(1 for c in required_cells if c.get("status") == "needs_world"),
            "inert": sum(1 for c in required_cells if c.get("status") == "inert_on_world"),
        },
        "reading": _reading(cells),
    }


def _typed_grid(
    *,
    kind: str,
    role: str,
    claim: str,
    world: dict[str, Any],
    diagnosis: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    cell = {
        "id": f"{kind}.main",
        "kind": kind,
        "role": role,
        "world": world.get("id"),
        "handle": diagnosis.get("world_lever") or "",
        "observable": diagnosis.get("world_observable") or "",
        "control": diagnosis.get("control_arm") or diagnosis.get("contrast") or "",
        "required": True,
        "status": "fillable" if attempts else "open",
        "paper_section": paper_section_for(kind),
    }
    fills = []
    for row in attempts:
        if not isinstance(row, dict):
            continue
        fills.append(
            {
                "cell": cell["id"],
                "exp_id": row.get("evidence_id") or row.get("world_id") or "",
                "verdict": row.get("verdict") or row.get("state") or "",
                "type": outcome_type(row) if row.get("verdict") else str(row.get("state") or "recorded"),
            }
        )
    if fills:
        cell["fills"] = fills
        cell["status"] = "filled:" + str(fills[-1].get("type") or "recorded")
    return {
        "claim": claim,
        "world": world.get("id"),
        "idea_kind": kind,
        "evidence_role": role,
        "paper_section": paper_section_for(kind),
        "cells": [cell],
        "coverage": {
            "required": 1,
            "supported": 1 if str(cell.get("status", "")).startswith("filled:") else 0,
            "needs_world": 1 if kind == "acquire" and not world.get("id") else 0,
            "inert": 0,
        },
        "reading": f"{role}: {cell['status']}",
    }


def _reading(cells: list[dict[str, Any]]) -> str:
    types = [f["type"] for c in cells for f in (c.get("fills") or [])]
    if not types:
        return "no cell filled yet"
    parts = []
    if "blind_measure" in types:
        parts.append("a measure ignored the oracle (define the DV on the oracle field)")
    if "placebo_fail" in types:
        parts.append("a separation survived structure destruction (the script, not the world, made it)")
    if "object_absent" in types:
        parts.append("the probe never met the object")
    if "under_margin" in types:
        parts.append("arms met the object but separated less than the margin")
    if "weakens" in types:
        parts.append("the pre-registered direction was contradicted")
    if "supports" in types:
        parts.append("the registered handle's cell is supported; controls and replication decide isolation")
    return "; ".join(parts)


def render_grid(grid: dict[str, Any]) -> str:
    lines = [
        "# Evidence grid",
        "",
        "Cells C1 needs to be defended; each probe fills one cell with its EvidenceID and a typed "
        "outcome. Negative results keep their type. A supported cell adds coverage, not isolation — "
        "the control and replication cells do that. `needs_world` says how the world would be acquired.",
        "",
        f"claim: {grid.get('claim')}",
        "",
        f"coverage: {json.dumps(grid.get('coverage'))} — {grid.get('reading')}",
        "",
        "| cell | kind | world | handle | observable | control | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in grid.get("cells") or []:
        lines.append(
            f"| {c.get('id')} | {c.get('kind')} | {c.get('world') or '—'} | {c.get('handle') or '—'} | "
            f"{c.get('observable') or '—'} | {str(c.get('control') or '')[:80]} | {c.get('status')} |"
        )
    lines.append("")
    for c in grid.get("cells") or []:
        if c.get("fills"):
            lines.append(f"## {c['id']} fills")
            for f in c["fills"]:
                lines.append(
                    f"- attempt {f.get('attempt')} · exp_id `{str(f.get('exp_id') or '')[:16]}` · "
                    f"{f.get('type')} ({f.get('verdict')}) · treatment {f.get('treatment')} / control {f.get('control')}"
                )
            lines.append("")
        if c.get("acquire"):
            a = c["acquire"]
            lines.append(f"## {c['id']} needs a world")
            lines.append(f"- how: {a.get('how')}")
            lines.append(f"- what: {a.get('what')}")
            lines.append(f"- properties: {json.dumps(a.get('properties'))}")
            lines.append("")
    return "\n".join(lines)


def write_grid(idea_dir: Path, grid: dict[str, Any]) -> Path:
    refine = Path(idea_dir) / "refine-logs"
    refine.mkdir(parents=True, exist_ok=True)
    (refine / "EVIDENCE_GRID.json").write_text(
        json.dumps(grid, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    path = refine / "EVIDENCE_GRID.md"
    path.write_text(render_grid(grid), encoding="utf-8")
    return path
