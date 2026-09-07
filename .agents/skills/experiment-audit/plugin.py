"""Audit a probe script against FarField teeth. Cannot climb the ladder."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

NAME = "experiment-audit"
HOOKS = ("execute",)


def execute(args: dict[str, Any] | None = None, workspace: Any = None, **_: Any) -> str:
    args = args or {}
    source = str(args.get("source") or "")
    path = str(args.get("path") or "experiment.py")
    if not source and workspace is not None:
        target = Path(workspace) / path
        if not target.is_file() and args.get("card_id"):
            target = Path(workspace) / "candidates" / str(args["card_id"]) / path
        if target.is_file():
            source = target.read_text(encoding="utf-8")
    if not source.strip():
        return json.dumps(
            {"ok": False, "refuse": "no source to audit"},
            ensure_ascii=False,
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return json.dumps(
            {"ok": False, "refuse": f"syntax: {exc}"},
            ensure_ascii=False,
        )
    from farfield.extras.plugins import PluginHost, repo_root

    host = PluginHost.load(repo_root())
    refuse = host.validate_probe(
        tree=tree,
        source=source,
        claim=str(args.get("claim") or ""),
        mechanism=str(args.get("mechanism") or ""),
        topic=str(args.get("topic") or ""),
        schema=str(args.get("schema") or ""),
    )
    kind = str(args.get("probe_kind") or args.get("kind") or "").upper()
    honesty = ""
    if kind in {"GENERATED", "SYNTHETIC"} and "WORLD" in source.upper():
        honesty = (
            "script or comments label this run WORLD; GENERATED/SYNTHETIC "
            "cannot corroborate"
        )
    ok = refuse is None and not honesty
    return json.dumps(
        {
            "ok": ok,
            "refuse": refuse,
            "honesty": honesty or None,
            "kind": kind,
        },
        ensure_ascii=False,
    )
