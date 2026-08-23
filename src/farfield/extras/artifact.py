"""Dump attested fields as an interchange pack for an existing writing skill.

This is not a paper compiler. Venue LaTeX, citation audit, and tectonic
builds already exist (jin-s13/ai-research-writing-skill,
NiuYingchun/latex-paper-skills). FarField only copies fields the mission
already attested and points the operator at those wheels. A compiled PDF
is not a discovery and cannot climb the ladder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .packet import NOT_A_PAPER, PROTOCOL_NOT_DISCOVERY


class ArtifactError(ValueError):
    """Compile refused. The candidate folder is unchanged."""


HONESTY = (
    "Compiled from attested protocol fields. Not a paper. Not a discovery. "
    "Empty results mean the host/external run has not happened. Filling them "
    "by hand does not corroborate."
)

WRITING_WHEELS = (
    "Do not write a FarField LaTeX engine. Feed `attested.json`, "
    "`protocol.md`, and the mission `RESEARCH_PACKET.md` to one of:\n"
    "- https://github.com/jin-s13/ai-research-writing-skill\n"
    "- https://github.com/NiuYingchun/latex-paper-skills "
    "(empirical-paper-writer)\n"
    "Compile the venue project with tectonic or latexmk. "
    "Polaris Paper Writer is a lab app — use it if the lab already runs it; "
    "do not vendor it. Probe metrics.json is not a results table."
)


def compile_artifact(
    folder: Path, *, plugin_host: Any = None
) -> dict[str, Any]:
    folder = Path(folder)
    protocol_path = folder / "protocol.json"
    if not protocol_path.is_file():
        raise ArtifactError(f"missing protocol.json in {folder}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ArtifactError("protocol.json must be an object")
    dest = folder / "artifact"
    dest.mkdir(parents=True, exist_ok=True)
    attested = _attested(protocol, folder)
    (dest / "attested.json").write_text(
        json.dumps(attested, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (dest / "WRITING.md").write_text(WRITING_WHEELS + "\n", encoding="utf-8")
    (dest / "paper.md").write_text(_markdown(attested), encoding="utf-8")
    (dest / "paper.tex").write_text(_tex_stub(), encoding="utf-8")
    extra: list[str] = []
    if plugin_host is not None and hasattr(plugin_host, "offer"):
        offered = plugin_host.offer(
            "compile_artifact", folder=folder, protocol=protocol, dest=dest
        )
        if isinstance(offered, dict):
            for name, body in offered.items():
                safe = Path(str(name)).name
                if not safe or safe in {
                    "paper.md",
                    "paper.tex",
                    "attested.json",
                    "WRITING.md",
                    "manifest.json",
                }:
                    continue
                (dest / safe).write_text(str(body), encoding="utf-8")
                extra.append(safe)
    manifest = {
        "kind": "attested_interchange",
        "not_a_paper": True,
        "not_a_discovery": True,
        "wheel": "jin-s13/ai-research-writing-skill",
        "files": [
            "attested.json",
            "WRITING.md",
            "paper.md",
            "paper.tex",
            *extra,
        ],
        "honesty": HONESTY,
        "card_id": str(protocol.get("card_id") or ""),
        "title": str(protocol.get("title") or protocol.get("claim") or "")[:200],
        "has_results": _results_record(folder) is not None,
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _results_record(folder: Path) -> dict[str, Any] | None:
    for name in ("host_run.json", "external_run.json"):
        path = folder / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _results_block(folder: Path) -> str:
    record = _results_record(folder)
    if record is None:
        return (
            "Results: empty until `farfield execute` or `farfield attest-run`. "
            "Do not paste probe metrics.json here."
        )
    kind = str(record.get("kind") or record.get("status") or "")
    return (
        f"Recorded run ({kind}): verdict={record.get('verdict')!r}, "
        f"treatment={record.get('treatment')}, control={record.get('control')}. "
        f"{PROTOCOL_NOT_DISCOVERY}"
    )


def _attested(protocol: dict[str, Any], folder: Path) -> dict[str, Any]:
    diag = protocol.get("diagnosis") if isinstance(protocol.get("diagnosis"), dict) else {}
    run = _results_record(folder)
    return {
        "not_a_paper": True,
        "not_a_discovery": True,
        "honesty": HONESTY,
        "card_id": str(protocol.get("card_id") or ""),
        "title": str(protocol.get("title") or protocol.get("claim") or ""),
        "claim": str(protocol.get("claim") or ""),
        "gap": str(protocol.get("gap") or protocol.get("idea") or ""),
        "mechanism": str(protocol.get("mechanism") or ""),
        "diagnosis": {
            "treatment_arm": diag.get("treatment_arm"),
            "control_arm": diag.get("control_arm"),
            "expected_direction": diag.get("expected_direction"),
            "margin": diag.get("margin"),
            "experiment": diag.get("experiment") or protocol.get("cheap_probe"),
        },
        "results": None
        if run is None
        else {
            "kind": run.get("kind") or run.get("status"),
            "verdict": run.get("verdict"),
            "treatment": run.get("treatment"),
            "control": run.get("control"),
        },
        "results_prose": _results_block(folder),
        "wheel": "jin-s13/ai-research-writing-skill",
    }


def _markdown(attested: dict[str, Any]) -> str:
    diag = attested.get("diagnosis") if isinstance(attested.get("diagnosis"), dict) else {}
    title = str(attested.get("title") or "untitled")
    return "\n".join(
        [
            f"# {title}",
            "",
            f"*{NOT_A_PAPER}*",
            "",
            f"*{HONESTY}*",
            "",
            "## Abstract",
            "",
            str(attested.get("claim") or "(no attested claim)"),
            "",
            "## Introduction",
            "",
            str(attested.get("gap") or "(gap not attested)"),
            "",
            "## Method",
            "",
            str(attested.get("mechanism") or ""),
            "",
            "Pre-registered arms:",
            "",
            f"- treatment: {diag.get('treatment_arm') or ''}",
            f"- control: {diag.get('control_arm') or ''}",
            f"- expected_direction: {diag.get('expected_direction') or ''}",
            f"- margin: {diag.get('margin')}",
            "",
            "## Experiments",
            "",
            str(diag.get("experiment") or ""),
            "",
            "## Results",
            "",
            str(attested.get("results_prose") or ""),
            "",
            "## Limitations",
            "",
            PROTOCOL_NOT_DISCOVERY,
            "",
            "## Writing wheel",
            "",
            WRITING_WHEELS,
            "",
        ]
    )


def _tex_stub() -> str:
    return (
        "% Not a paper. This stub is interchange, not a manuscript.\n"
        "% Feed artifact/attested.json to jin-s13/ai-research-writing-skill\n"
        "% or NiuYingchun/latex-paper-skills (empirical-paper-writer).\n"
        "% Compile that venue project with tectonic / latexmk.\n"
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "FarField does not compile conference PDFs. "
        "See artifact/WRITING.md.\n"
        "\\end{document}\n"
    )
