"""Executable hooks for fair-two-arm-probe.

SKILL.md is what the model reads. PluginHost calls these functions: a
probe that hard-codes an asymmetric win is refused here, not hoped away
in the prompt.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

NAME = "fair-two-arm-probe"
HOOKS = ("validate_probe", "validate_diagnosis", "execute")


def validate_probe(tree: Any = None, source: str = "", **_: Any) -> str | None:
    from farfield.extras.probeexp import ProbeRefused, assert_fair_probe

    if tree is None:
        return "probe source did not parse as a module"
    if not isinstance(tree, ast.AST):
        return "probe source did not parse as a module"
    try:
        assert_fair_probe(tree)
    except ProbeRefused as exc:
        return exc.record.unlock_condition
    return None


def validate_diagnosis(
    treatment_arm: str = "",
    control_arm: str = "",
    **_: Any,
) -> str | None:
    left = (treatment_arm or "").strip().lower()
    right = (control_arm or "").strip().lower()
    if left and right and left == right:
        return (
            "treatment and control arms are identical; a two-arm diagnosis"
            " must remove the mechanism in the control"
        )
    return None


def execute(args: dict[str, Any] | None = None, workspace: Any = None, **_: Any) -> str:
    args = args or {}
    source = str(args.get("source") or "")
    path = str(args.get("path") or "experiment.py")
    if not source and workspace is not None:
        target = Path(workspace) / path
        if target.is_file():
            source = target.read_text(encoding="utf-8")
    if not source.strip():
        return json.dumps({"ok": False, "refuse": "no source to validate"}, ensure_ascii=False)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return json.dumps({"ok": False, "refuse": f"syntax: {exc}"}, ensure_ascii=False)
    reason = validate_probe(tree, source=source)
    return json.dumps({"ok": reason is None, "refuse": reason}, ensure_ascii=False)
