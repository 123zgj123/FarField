"""Emit a SkyPilot task YAML. Do not vendor a GPU scheduler.

SkyPilot (Apache-2.0) already submits to clouds, Kubernetes, and Slurm.
This hook writes that project's task schema next to the protocol. Launch
belongs to the SkyPilot CLI / agent skill. Arm numbers still enter through
attest-run; this plugin never returns treatment/control.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

NAME = "external-gpu-run"
HOOKS = ("run_external", "execute")

TASK_NAME = "sky_task.yaml"


def run_external(folder: Any = None, spec: Any = None, **_: Any) -> None:
    dest = Path(folder) if folder is not None else Path.cwd()
    write_sky_task(dest, spec=spec if isinstance(spec, dict) else {})
    _maybe_launch(dest)
    return None


def execute(args: dict[str, Any] | None = None, workspace: Any = None, **_: Any) -> str:
    args = args or {}
    dest = Path(str(args.get("folder") or workspace or "."))
    path = write_sky_task(dest)
    return json.dumps(
        {"ok": True, "task": str(path), "launched": False, "wheel": "skypilot"},
        ensure_ascii=False,
    )


def write_sky_task(folder: Path, *, spec: dict[str, Any] | None = None) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    protocol: dict[str, Any] = {}
    protocol_path = folder / "protocol.json"
    if protocol_path.is_file():
        try:
            loaded = json.loads(protocol_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            protocol = loaded
    card = str(protocol.get("card_id") or folder.name)
    accel = str(os.environ.get("FARFIELD_SKY_ACCELERATORS") or "").strip()
    resources = (
        f"resources:\n  accelerators: {accel}\n"
        if accel
        else (
            "# resources.accelerators: set by the SkyPilot skill / sky CLI\n"
            "# (do not invent a GPU type in FarField)\n"
        )
    )
    body = (
        f"# SkyPilot task emitted by FarField skill external-gpu-run.\n"
        f"# Launch: sky jobs launch {TASK_NAME}\n"
        f"# Skill: https://github.com/skypilot-org/skypilot/tree/master/agent/skills/skypilot\n"
        f"# After the job: farfield attest-run --metrics metrics.json\n"
        f"# externally_replicated is not corroborated.\n"
        f"name: farfield-{_slug(card)}\n"
        f"workdir: .\n"
        f"{resources}"
        f"run: |\n"
        f"  python experiment.py\n"
        f"  test -f metrics.json\n"
    )
    path = folder / TASK_NAME
    path.write_text(body, encoding="utf-8")
    return path


def _maybe_launch(folder: Path) -> None:
    if os.environ.get("FARFIELD_SKY_LAUNCH") != "1":
        return
    binary = shutil.which("sky")
    if not binary:
        return
    task = folder / TASK_NAME
    try:
        subprocess.run(
            [binary, "jobs", "launch", str(task), "-y"],
            cwd=str(folder),
            check=False,
            timeout=30,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def _slug(text: str) -> str:
    keep = [ch.lower() if ch.isalnum() else "-" for ch in str(text)]
    slug = "".join(keep).strip("-") or "card"
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:48]
