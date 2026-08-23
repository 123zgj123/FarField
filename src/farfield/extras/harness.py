"""DeepSeek Harness as a FarField plugin, not a PATH peek.

Official `dsh` (https://github.com/deepseek-ai/deepseek-harness) is a
Cordis TypeScript runtime. FarField does not vendor it
(`dependencies = []`). This module is the interchange:

- Skill leaf: write `<base>/.agents/skills/<name>/SKILL.md` so a later
  `dsh` session in this repo mounts what a mission distilled (rank 200).
- Overlay: `.dsh/cordis.yml` points `customSkillDirs` at those trees.
- Plugin: `DeepSeekHarnessPlugin.execute` actually launches `dsh` when
  the binary is on PATH. Missing `dsh` is an observation, not a fake
  success; local PluginHost hooks still run.

A skill file still cannot change the agent loop, the graph gates, probe
arithmetic, or the promotion ladder. Distilled skills never write
plugin.py — LLM-authored Python is not executed.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .skills import Skill, load_skill_tree, write_skill

# Rank 200 in dsh-skill-filesystem. Rank 100 (.dsh/skills) is left for a
# human-authored dsh overlay; FarField writes here so both runtimes share it.
DSH_PROJECT_SKILLS = Path(".agents") / "skills"
DSH_OVERLAY = Path(".dsh") / "cordis.yml"
DSH_PLUGIN_NAME = "deepseek-harness"
HEADLESS_PROFILE = "headless"


def dsh_skill_root(base: Path) -> Path:
    return Path(base) / DSH_PROJECT_SKILLS


def overlay_path(base: Path | None = None) -> Path:
    root = Path(base) if base is not None else Path(__file__).resolve().parents[3]
    return root / DSH_OVERLAY


def write_dsh_skill(base: Path, skill: Skill) -> Path:
    """Write `<base>/.agents/skills/<name>/SKILL.md` — a dsh bundle entry."""
    return write_skill(dsh_skill_root(base), skill)


def load_dsh_skills(base: Path) -> tuple[Skill, ...]:
    return load_skill_tree(dsh_skill_root(base))


def dsh_on_path() -> str | None:
    return shutil.which("dsh") or shutil.which("dsh.cmd")


def persist_skill(skill: Skill, *, workspace: Path | None, catalog: Path) -> dict[str, str]:
    """Evidence-gated skill: this mission's workspace and the durable catalog.

    `catalog` is typically `var/skills` so later missions
    load it without committing a new file to git from every run. Seed skills
    that belong in the repo are authored under `<repo>/.agents/skills`.
    Distillation writes SKILL.md only — never plugin.py.
    """
    written: dict[str, str] = {
        "catalog": str(write_skill(catalog, skill)),
    }
    if workspace is not None:
        written["workspace"] = str(write_dsh_skill(workspace, skill))
    return written


@dataclass(frozen=True)
class DshResult:
    available: bool
    binary: str | None
    returncode: int | None
    output: str
    command: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "binary": self.binary,
            "returncode": self.returncode,
            "output": self.output,
            "command": list(self.command),
        }


def dsh_status(base: Path | None = None) -> dict[str, Any]:
    binary = dsh_on_path()
    overlay = overlay_path(base)
    return {
        "binary": binary,
        "available": binary is not None,
        "overlay": str(overlay) if overlay.is_file() else None,
        "role": "plugin",
        "name": DSH_PLUGIN_NAME,
    }


def run_dsh(
    instruction: str,
    *,
    cwd: Path | None = None,
    timeout: float = 120.0,
    config: Path | None = None,
) -> DshResult:
    """Launch official `dsh` as a subprocess. Never pretends it ran."""
    binary = dsh_on_path()
    if not binary:
        return DshResult(
            available=False,
            binary=None,
            returncode=None,
            output=(
                "DeepSeek Harness is not on PATH. Local PluginHost hooks still"
                " ran. Install with: npm install -g @deepseek-ai/dsh"
            ),
            command=(),
        )
    command: list[str] = [binary, "--profile", HEADLESS_PROFILE]
    overlay = config if config is not None else overlay_path(cwd)
    if overlay is not None and Path(overlay).is_file():
        command.extend(["--config", str(Path(overlay))])
    command.append(instruction)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return DshResult(
            available=False,
            binary=binary,
            returncode=None,
            output="dsh binary disappeared before launch",
            command=tuple(command),
        )
    except subprocess.TimeoutExpired:
        return DshResult(
            available=True,
            binary=binary,
            returncode=None,
            output=f"dsh timed out after {timeout:.0f}s",
            command=tuple(command),
        )
    body = (completed.stdout or "").strip()
    if completed.stderr:
        body = (body + "\n[stderr]\n" + completed.stderr.strip()).strip()
    return DshResult(
        available=True,
        binary=binary,
        returncode=completed.returncode,
        output=body or "(no output)",
        command=tuple(command),
    )


class DeepSeekHarnessPlugin:
    """FarField plugin that *is* DeepSeek Harness when the binary exists."""

    name = DSH_PLUGIN_NAME

    def has(self, hook: str) -> bool:
        return hook == "execute"

    def call(self, hook: str, **kwargs: Any) -> Any:
        if hook != "execute":
            return None
        return self.execute(**kwargs)

    def execute(
        self,
        args: dict[str, Any] | None = None,
        workspace: Any = None,
        **_: Any,
    ) -> str:
        args = args or {}
        instruction = str(args.get("instruction") or args.get("task") or "").strip()
        if not instruction:
            return "deepseek-harness execute needs an instruction"
        cwd = Path(workspace) if workspace is not None else None
        result = run_dsh(instruction, cwd=cwd)
        return result.output

    def to_dict(self) -> dict[str, Any]:
        status = dsh_status()
        return {
            "name": self.name,
            "executable": True,
            "trusted": True,
            "hooks": ["execute"],
            "source": "farfield.extras.harness.DeepSeekHarnessPlugin",
            "dsh": status,
        }
