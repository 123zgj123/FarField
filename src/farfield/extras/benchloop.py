"""A research-intern loop whose verdict is never a toy probe.

MLE-Bench Lite and AARRI-Bench both give the agent an instruction and a
workspace, then score the deliverable with an *external* grader. This
module is the shared intern: look around, pre-register what would count
as done, write files, run commands, stop. It does not invent a metric,
does not run the two-arm sandbox probe, and does not promote anything
on the research-state ladder. The official grader is the only score.

The workspace is a protocol so the same loop can drive a local directory
(MLE host-side) or a Harbor container (AARRI) without either harness
leaking into the other.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..models import BlockedRecord
from .generate import GenerationRefused
from .llm import Completion, LLMUnavailable

ACTIONS = ("diagnose", "ls", "read", "write", "run", "skill", "dsh", "done")
MAX_STEPS = 40
OUTPUT_CAP = 8000
DEFAULT_TIMEOUT = 60.0
# ls/read/mkdir in a busy Harbor container; a live AARRI batch died to 15s.
INSPECT_TIMEOUT = 60.0
FLUSH_RESERVE_SECONDS = 45.0
SECONDS_PER_STEP_FLOOR = 3.0
# Consecutive ls/read/run while the diagnosed deliverable is still missing.
INSPECT_STREAK_CAP = 8
# read pages; 240 lines forced intern onto `run`/`sed` for papers.
READ_PAGE_LINES = 800
# Wall-clock remaining (before flush reserve) when inspect is refused.
WRITE_ONLY_SECONDS = 90.0
HISTORY_TAIL = 4
HISTORY_CHARS = 300

PLACEHOLDER_MARKERS = (
    "<=40 words",
    "<=30 words",
    "the checkable condition",
    "the most likely way to look done",
    "the smallest sequence of inspections",
)

SYSTEM = (
    "You are a research intern completing one assigned task in a workspace."
    " Answer with one JSON object and nothing else."
)

TEMPLATE = """Complete the assigned task by inspecting the workspace, writing files, and running commands. The official external grader — not you, not a toy experiment — is the only verdict. Do not claim the task is solved because a small synthetic check looked good.

Task:
{instruction}

{diagnosis_block}History (most recent last):
{history}

Answer with one JSON object, one of:
{{"action": "diagnose", "criterion": "checkable condition under which this task is done", "alternative": "most likely way to look done while being wrong", "plan": "smallest sequence of inspections and edits", "deliverable": "workspace path the grader will look for"}}
{{"action": "ls", "path": "."}}
{{"action": "read", "path": "path", "offset": 0}}
{{"action": "write", "path": "path", "content": "full file contents"}}
{{"action": "run", "command": "a shell command", "timeout": 60}}
{{"action": "skill", "name": "plugin-name", "args": {{}}}}
{{"action": "dsh", "instruction": "task for DeepSeek Harness when dsh is on PATH"}}
{{"action": "done", "summary": "what you delivered and where"}}

Rules:
- Your first action MUST be diagnose. Fill real values; do not copy these field descriptions into the JSON.
- `deliverable` is the path the official grader will open. Register every existing file you must edit as that deliverable (or you cannot overwrite it).
- `done` is refused until that path exists in the workspace. A diagnosis on record is not enough.
- After several inspections with the deliverable still missing, write a draft. Do not keep listing and reading.
- Prefer the smallest change that meets the instruction. If the instruction is to refuse, stop, or report that you cannot, do that — by writing the deliverable, not by editing frozen inputs.
- `run` has no network assumption; if a command needs the network and it fails, say so and try another way.
- `skill` runs a trusted PluginHost execute() hook (not SKILL.md paste). `dsh` launches official DeepSeek Harness when installed.
- Paths are relative to the workspace root, or absolute under /app. Do not escape it.
- `read` takes `offset` in lines (0-based). Use it to continue a long file instead of `run`/`sed`.
"""

DIAGNOSED = """Registered success criterion: {criterion}
Competing failure mode: {alternative}
Plan: {plan}
Deliverable: {deliverable} — {presence}

A diagnosis is already on record. Do NOT send diagnose again. Next action must be ls, read, write, run, skill, dsh, or done.
If the deliverable is MISSING, write a draft there as soon as you know the path; inspection is capped.
"""


def agent_timeout_from_toml(text: str, default: float = 600.0) -> float:
    """Read Harbor ``[agent] timeout_sec`` from a task.toml body."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return default
    raw = (data.get("agent") or {}).get("timeout_sec")
    if raw is None:
        return default
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return default


def steps_for_timeout(
    timeout_sec: float,
    *,
    floor: int = MAX_STEPS,
    seconds_per_step: float = SECONDS_PER_STEP_FLOOR,
) -> int:
    """Step cap that cannot fire before the task's agent timeout.

    Harbor cancels the agent at ``timeout_sec``. A hard 40-step cap on a
    600s task kills the intern first; this maps timeout → steps so the
    wall clock is the real limit.
    """
    if timeout_sec <= 0 or seconds_per_step <= 0:
        return floor
    return max(floor, int(timeout_sec / seconds_per_step))


def workspace_rel(path: str) -> str:
    """Strip the AARRI `/app/` prefix so local and Harbor keys match."""
    text = (path or "").strip().replace("\\", "/")
    if text in {"/app", "/app/"}:
        return "."
    if text.startswith("/app/"):
        text = text[5:]
    return text or "."


def path_key(path: str) -> str:
    """Stable workspace-relative key for freeze / deliverable checks."""
    parts: list[str] = []
    for part in workspace_rel(path).split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            else:
                parts.append("..")
        else:
            parts.append(part)
    return "/".join(parts) or "."


def _looks_placeholder(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in PLACEHOLDER_MARKERS)


class BenchRefused(GenerationRefused):
    pass


def _refuse(attempted: str, unlock: str) -> BenchRefused:
    return BenchRefused(
        BlockedRecord(
            missing_capability="bench_intern_schema",
            attempted=attempted,
            unlock_condition=unlock,
        )
    )


@dataclass(frozen=True)
class ShellResult:
    stdout: str
    stderr: str
    return_code: int

    def clipped(self, cap: int = OUTPUT_CAP) -> str:
        body = (self.stdout or "") + (
            f"\n[stderr]\n{self.stderr}" if self.stderr else ""
        )
        body = body.strip() or "(no output)"
        if len(body) > cap:
            return body[:cap] + f"\n… [{len(body) - cap} bytes clipped]"
        return f"[exit {self.return_code}]\n{body}"


class Workspace(Protocol):
    async def ls(self, path: str = ".") -> str: ...
    async def read(
        self, path: str, *, offset: int = 0, limit: int = READ_PAGE_LINES
    ) -> str: ...
    async def write(self, path: str, content: str) -> None: ...
    async def run(self, command: str, *, timeout: float = DEFAULT_TIMEOUT) -> ShellResult: ...
    async def exists(self, path: str) -> bool: ...
    async def snapshot_files(self) -> set[str]: ...


def _parse(completion: Completion) -> dict[str, Any]:
    text = completion.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _refuse(
            "parse a JSON intern action",
            f"the intern answers with one JSON object; it answered with {text[:80]!r} ({exc})",
        ) from exc
    if not isinstance(payload, dict):
        raise _refuse(
            "read a JSON object for the intern action",
            f"the top-level value is an object, not {type(payload).__name__}",
        )
    return payload


def action_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action") or "").strip()
    if action not in ACTIONS:
        raise _refuse(
            "read action",
            "action is diagnose, ls, read, write, run, skill, dsh, or done",
        )
    if action == "diagnose":
        fields = {
            name: str(payload.get(name) or "").strip()
            for name in ("criterion", "alternative", "plan", "deliverable")
        }
        for name, value in fields.items():
            if not value:
                raise _refuse(
                    f"read a non-empty {name}",
                    "diagnose registers criterion, alternative, plan, and deliverable",
                )
            if _looks_placeholder(value):
                raise _refuse(
                    f"read a real {name}, not the schema hint",
                    f"{name} must be a concrete value, not a copy of the field description",
                )
        key = path_key(fields["deliverable"])
        if " " in fields["deliverable"] or "\n" in fields["deliverable"]:
            raise _refuse(
                "read a deliverable path",
                "deliverable is a workspace path without spaces, not a sentence",
            )
        if key in {".", ".."} or key.startswith("../"):
            raise _refuse(
                "read a deliverable path",
                "deliverable names a file or directory the grader will open, not the workspace root",
            )
        return {"action": action, **fields}
    if action in ("ls", "read"):
        path = str(payload.get("path") or "").strip() or "."
        parsed: dict[str, Any] = {"action": action, "path": path}
        if action == "read":
            raw_offset = payload.get("offset", 0)
            try:
                parsed["offset"] = max(0, int(raw_offset))
            except (TypeError, ValueError) as exc:
                raise _refuse("read read.offset", "offset is a 0-based line number") from exc
        return parsed
    if action == "write":
        path = str(payload.get("path") or "").strip()
        if not path:
            raise _refuse("read write.path", "write names a relative path")
        if payload.get("content") is None:
            raise _refuse("read write.content", "write includes the full file contents")
        return {"action": action, "path": path, "content": str(payload["content"])}
    if action == "run":
        command = str(payload.get("command") or "").strip()
        if not command:
            raise _refuse("read run.command", "run names a shell command")
        timeout = payload.get("timeout", DEFAULT_TIMEOUT)
        try:
            timeout_f = float(timeout)
        except (TypeError, ValueError) as exc:
            raise _refuse("read run.timeout", "timeout is a number of seconds") from exc
        timeout_f = min(max(timeout_f, 1.0), 600.0)
        return {"action": action, "command": command, "timeout": timeout_f}
    if action == "skill":
        name = str(payload.get("name") or "").strip()
        if not name:
            raise _refuse("read skill.name", "skill names a loaded plugin")
        args = payload.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise _refuse("read skill.args", "args is a JSON object")
        return {"action": "skill", "name": name, "args": args}
    if action == "dsh":
        instruction = str(payload.get("instruction") or payload.get("task") or "").strip()
        if not instruction:
            raise _refuse(
                "read dsh.instruction",
                "dsh sends an instruction to the DeepSeek Harness plugin",
            )
        return {"action": "dsh", "instruction": instruction}
    summary = str(payload.get("summary") or "").strip()
    if not summary:
        raise _refuse("read done.summary", "done says what was delivered and where")
    return {"action": "done", "summary": summary}


@dataclass
class InternReport:
    instruction: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    diagnosis: dict[str, str] | None = None
    done: bool = False
    summary: str = ""
    refused: str = ""
    stop: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "steps": self.steps,
            "diagnosis": self.diagnosis,
            "done": self.done,
            "summary": self.summary,
            "refused": self.refused,
            "stop": self.stop,
        }


def _seconds_left(deadline_monotonic: float | None, reserve_seconds: float) -> float | None:
    if deadline_monotonic is None:
        return None
    return deadline_monotonic - time.monotonic() - reserve_seconds


def _wall_left(deadline_monotonic: float | None) -> float | None:
    if deadline_monotonic is None:
        return None
    return deadline_monotonic - time.monotonic()


def _page_footer(offset: int, limit: int, n_lines: int) -> str:
    start = offset + 1
    end = min(offset + limit, n_lines)
    nxt = offset + limit
    if nxt < n_lines:
        return f"\n--- lines {start}-{end} of {n_lines}; next offset {nxt}"
    return f"\n--- lines {start}-{end} of {n_lines}"


async def run_intern(
    client: Any,
    workspace: Workspace,
    instruction: str,
    *,
    max_steps: int = MAX_STEPS,
    retries: int = 1,
    deadline_monotonic: float | None = None,
    reserve_seconds: float = FLUSH_RESERVE_SECONDS,
    on_progress: Callable[[InternReport], None] | None = None,
    host: Any = None,
) -> InternReport:
    """Drive one intern through `instruction` until done, refused, step cap, or wall deadline.

    Evidence in this loop is only command output the workspace actually
    returned. Promotion, medals, and 0/1 rewards are applied *after*
    this function returns, by the official grader of the host benchmark.

    ``deadline_monotonic`` is ``time.monotonic()`` of Harbor's agent
    timeout. When remaining time drops below ``reserve_seconds``, the
    loop stops so ``on_progress`` can flush intern.json before Harbor
    cancels the task. ``on_progress`` is also called after every step,
    because a later ``asyncio.wait_for`` cancel never reaches a trailing
    write.
    """
    report = InternReport(instruction=instruction.strip())
    history: list[str] = ["(none yet)"]
    from .plugins import get_host as _get_host

    plugin_host = host if host is not None else _get_host()
    try:
        frozen = set(await workspace.snapshot_files())
    except Exception:  # noqa: BLE001 — freeze is a guard, not a hard dependency
        frozen = set()
    inspect_streak = 0

    def flush() -> None:
        if on_progress is None:
            return
        try:
            on_progress(report)
        except OSError:
            pass

    def hit_deadline(step: int, remaining: float) -> InternReport:
        report.stop = "deadline"
        report.refused = (
            f"agent deadline reached with {remaining:.1f}s slack; intern.json flushed"
        )
        report.steps.append(
            {
                "step": step,
                "stop": "deadline",
                "remaining_sec": round(remaining, 2),
            }
        )
        flush()
        return report

    async def deliverable_present() -> bool:
        if not report.diagnosis:
            return False
        return await workspace.exists(report.diagnosis["deliverable"])

    flush()
    try:
        for step in range(max_steps):
            remaining = _seconds_left(deadline_monotonic, reserve_seconds)
            if remaining is not None and remaining <= 0:
                return hit_deadline(step, remaining)
            present = await deliverable_present()
            diagnosis_block = ""
            if report.diagnosis:
                diagnosis_block = DIAGNOSED.format(
                    **report.diagnosis,
                    presence="PRESENT" if present else "MISSING",
                )
            prompt = TEMPLATE.format(
                instruction=report.instruction,
                diagnosis_block=diagnosis_block,
                history="\n".join(history[-HISTORY_TAIL:]),
            )
            try:
                action = _ask(client, prompt, step, retries=retries)
            except BenchRefused as exc:
                report.stop = "refused"
                report.refused = exc.record.unlock_condition
                report.steps.append({"step": step, "error": report.refused})
                flush()
                return report
            except LLMUnavailable as exc:
                report.stop = "budget"
                report.refused = str(exc)
                report.steps.append({"step": step, "error": report.refused})
                flush()
                return report
            remaining = _seconds_left(deadline_monotonic, reserve_seconds)
            if remaining is not None and remaining <= 0:
                return hit_deadline(step, remaining)
            if action["action"] == "run" and remaining is not None:
                action["timeout"] = min(float(action["timeout"]), max(1.0, remaining))

            kind = action["action"]
            observation = ""
            applied = False

            if kind == "diagnose":
                if report.diagnosis:
                    observation = "a diagnosis is already on record; inspect or edit next"
                else:
                    report.diagnosis = {
                        "criterion": action["criterion"],
                        "alternative": action["alternative"],
                        "plan": action["plan"],
                        "deliverable": action["deliverable"],
                    }
                    observation = (
                        "diagnosis registered; deliverable "
                        f"{action['deliverable']} must exist before done; "
                        "the grader, not this text, is the score"
                    )
            elif kind != "diagnose" and report.diagnosis is None and kind != "done":
                observation = (
                    "action refused: register a diagnosis with a deliverable path "
                    "before inspecting or editing"
                )
            elif kind == "done":
                if report.diagnosis is None:
                    observation = (
                        "done refused: register a diagnosis before claiming the task is finished"
                    )
                elif not await deliverable_present():
                    observation = (
                        "done refused: deliverable "
                        f"{report.diagnosis['deliverable']} is not in the workspace; "
                        "write a draft there first"
                    )
                else:
                    report.done = True
                    report.stop = "done"
                    report.summary = action["summary"]
                    report.steps.append({**action, "step": step})
                    history.append("done")
                    flush()
                    return report
            else:
                missing = report.diagnosis is not None and not await deliverable_present()
                if kind in {"skill", "dsh"}:
                    ws_root = getattr(workspace, "root", None)
                    if kind == "skill":
                        observation = plugin_host.execute(
                            action["name"],
                            action.get("args") or {},
                            workspace=ws_root,
                        )
                    else:
                        observation = plugin_host.execute(
                            "deepseek-harness",
                            {"instruction": action["instruction"]},
                            workspace=ws_root,
                        )
                    applied = True
                else:
                    wall = _wall_left(deadline_monotonic)
                    write_only = (
                        missing
                        and wall is not None
                        and wall <= WRITE_ONLY_SECONDS
                        and kind in {"ls", "read", "run"}
                    )
                    inspect_blocked = (
                        missing
                        and kind in {"ls", "read", "run"}
                        and inspect_streak >= INSPECT_STREAK_CAP
                    )
                    if write_only:
                        observation = (
                            f"inspect refused: {wall:.0f}s left and deliverable "
                            f"{report.diagnosis['deliverable']} is still missing; "
                            "only write or done is accepted"
                        )
                    elif inspect_blocked:
                        observation = (
                            f"inspect refused: {INSPECT_STREAK_CAP} inspections without "
                            f"{report.diagnosis['deliverable']}; write a draft there first"
                        )
                    elif kind == "write" and report.diagnosis is not None:
                        target = path_key(action["path"])
                        allowed = path_key(report.diagnosis["deliverable"])
                        if target in frozen and target != allowed:
                            observation = (
                                f"write refused: {action['path']} existed at start "
                                "and is not the diagnosed deliverable "
                                f"{report.diagnosis['deliverable']}; "
                                "register that path as deliverable to edit it, "
                                "or write a new file"
                            )
                        else:
                            observation = await _apply(workspace, action)
                            applied = True
                    else:
                        observation = await _apply(workspace, action)
                        applied = True

                if applied and kind in {"ls", "read", "run"} and missing:
                    inspect_streak += 1
                if applied and report.diagnosis is not None:
                    if await deliverable_present():
                        inspect_streak = 0

            record = {**action, "step": step, "observation": observation}
            if "content" in record and len(record["content"]) > 200:
                record["content"] = record["content"][:200] + "…"
            report.steps.append(record)
            history.append(f"{kind} → {observation[:HISTORY_CHARS]}")
            flush()
        if not report.stop:
            report.stop = "step_cap"
            report.refused = f"step cap {max_steps} reached without done"
        flush()
        return report
    except BaseException:
        flush()
        raise


def _ask(client: Any, prompt: str, step: int, *, retries: int) -> dict[str, Any]:
    refusal: BenchRefused | None = None
    for attempt in range(retries + 1):
        ask = prompt
        if refusal is not None:
            ask = (
                prompt
                + "\n\nYour previous action was rejected:"
                f" {refusal.record.unlock_condition}."
                " Send one valid JSON action."
            )
        completion = client.complete(
            ask,
            purpose=f"bench_intern:step{step}"
            + (f":retry{attempt}" if attempt else ""),
            system=SYSTEM,
        )
        completion.assert_usable()
        try:
            return action_from_payload(_parse(completion))
        except BenchRefused as exc:
            refusal = exc
    assert refusal is not None
    raise refusal


async def _apply(workspace: Workspace, action: dict[str, Any]) -> str:
    kind = action["action"]
    try:
        if kind == "ls":
            return await workspace.ls(action["path"])
        if kind == "read":
            return await workspace.read(
                action["path"], offset=int(action.get("offset") or 0)
            )
        if kind == "write":
            await workspace.write(action["path"], action["content"])
            return f"wrote {action['path']} ({len(action['content'])} bytes)"
        result = await workspace.run(action["command"], timeout=action["timeout"])
        return result.clipped()
    except Exception as exc:  # noqa: BLE001 — observation, not a crash of the intern
        return f"{type(exc).__name__}: {exc}"


def _safe(root: Path, path: str) -> Path:
    rel = workspace_rel(path)
    target = (root / rel).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise PermissionError(f"{path} escapes the workspace") from exc
    return target


def _read_page(data: str, *, offset: int, limit: int) -> str:
    lines = data.splitlines(keepends=True)
    n_lines = len(lines)
    chunk = "".join(lines[offset : offset + limit])
    if not chunk and offset >= n_lines:
        return f"(no lines at offset {offset}; file has {n_lines} lines)"
    return chunk.rstrip("\n") + _page_footer(offset, limit, n_lines)


class LocalWorkspace:
    """A directory on this machine. Used by tests and the MLE host-side runner."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    async def ls(self, path: str = ".") -> str:
        target = _safe(self.root, path)
        if not target.exists():
            return f"missing: {path}"
        if target.is_file():
            return path
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return "\n".join(names) or "(empty)"

    async def read(
        self, path: str, *, offset: int = 0, limit: int = READ_PAGE_LINES
    ) -> str:
        target = _safe(self.root, path)
        if target.is_dir():
            names = sorted(p.name for p in target.iterdir())
            return "directory:\n" + ("\n".join(names) or "(empty)")
        data = target.read_text(encoding="utf-8", errors="replace")
        return _read_page(data, offset=offset, limit=limit)

    async def write(self, path: str, content: str) -> None:
        target = _safe(self.root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def run(self, command: str, *, timeout: float = DEFAULT_TIMEOUT) -> ShellResult:
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                timeout=timeout,
                capture_output=True,
                text=True,
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key in {"PATH", "HOME", "LANG", "LC_ALL", "VIRTUAL_ENV"}
                },
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ShellResult(stdout="", stderr=f"timed out after {timeout}s", return_code=124)
        return ShellResult(
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            return_code=completed.returncode,
        )

    async def exists(self, path: str) -> bool:
        try:
            return _safe(self.root, path).exists()
        except PermissionError:
            return False

    async def snapshot_files(self) -> set[str]:
        root = self.root.resolve()
        found: set[str] = set()
        for item in root.rglob("*"):
            if item.is_file():
                found.add(str(item.relative_to(root)).replace("\\", "/"))
        return found


class HarborWorkspace:
    """A Harbor environment. `environment.exec` is the only way in.

    Docker rejects a relative `cwd` (`Cwd must be an absolute path`); a live
    AARRI trial burned 40 steps on that OCI error. Relative or empty cwd is
    therefore omitted so the container's WORKDIR (/app on AARRI) is used.
    """

    def __init__(self, environment: Any, *, cwd: str | None = None) -> None:
        self.environment = environment
        self.cwd = cwd if cwd and cwd.startswith("/") else None

    def _exec_kw(self) -> dict[str, Any]:
        return {"cwd": self.cwd} if self.cwd else {}

    async def ls(self, path: str = ".") -> str:
        result = await self.environment.exec(
            f"ls -la {shlex.quote(path)}",
            timeout_sec=int(INSPECT_TIMEOUT),
            **self._exec_kw(),
        )
        return ShellResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            return_code=result.return_code,
        ).clipped()

    async def read(
        self, path: str, *, offset: int = 0, limit: int = READ_PAGE_LINES
    ) -> str:
        start = max(int(offset), 0) + 1
        end = start + max(int(limit), 1) - 1
        quoted = shlex.quote(path)
        command = (
            f"if [ ! -e {quoted} ]; then echo MISSING; exit 1; fi; "
            f"if [ -d {quoted} ]; then ls -la {quoted}; exit 0; fi; "
            f"n=$(wc -l < {quoted}); "
            f"sed -n '{start},{end}p' {quoted}; "
            f"echo; echo \"--- lines {start}-{end} of $n; next offset {end} if {end} < $n\""
        )
        result = await self.environment.exec(
            command,
            timeout_sec=int(INSPECT_TIMEOUT),
            **self._exec_kw(),
        )
        return ShellResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            return_code=result.return_code,
        ).clipped()

    async def write(self, path: str, content: str) -> None:
        # Write on the host, then upload, so the content never has to survive
        # a shell-quoting round trip.
        with tempfile.TemporaryDirectory(prefix="ffbench-") as tmp:
            local = Path(tmp) / "blob"
            local.write_text(content, encoding="utf-8")
            if path.startswith("/"):
                remote = path
            elif self.cwd:
                remote = f"{self.cwd.rstrip('/')}/{path}"
            else:
                remote = f"/app/{path}"
            parent = str(Path(remote).parent)
            await self.environment.exec(
                f"mkdir -p {shlex.quote(parent)}",
                timeout_sec=int(INSPECT_TIMEOUT),
                **self._exec_kw(),
            )
            await self.environment.upload_file(local, remote)

    async def run(self, command: str, *, timeout: float = DEFAULT_TIMEOUT) -> ShellResult:
        result = await self.environment.exec(
            command, timeout_sec=int(timeout), **self._exec_kw()
        )
        return ShellResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            return_code=result.return_code,
        )

    async def exists(self, path: str) -> bool:
        result = await self.environment.exec(
            f"test -e {shlex.quote(path)} && echo EXISTS || echo MISSING",
            timeout_sec=int(INSPECT_TIMEOUT),
            **self._exec_kw(),
        )
        return "EXISTS" in (result.stdout or "")

    async def snapshot_files(self) -> set[str]:
        result = await self.environment.exec(
            "find . -type f 2>/dev/null | head -n 8000",
            timeout_sec=int(INSPECT_TIMEOUT),
            **self._exec_kw(),
        )
        found: set[str] = set()
        for line in (result.stdout or "").splitlines():
            text = line.strip()
            if text:
                found.add(path_key(text))
        return found
