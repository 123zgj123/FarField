"""World by execution: run a harness under a pre-registered recipe, freeze
what it produced.

The gap this closes. A `program_state` claim needs an attested history
of a program modifying itself under validators. Until now the only way
in was a host operator fetching a published archive before any claim
existed; when none matched, the mission fell to a GENERATED world,
which by rule cannot corroborate. A harvest is the third way: the host
*runs* an open-source harness-evolution framework (OpenEvolve,
ShinkaEvolve, DGM, a local loop) for a bounded number of iterations
and freezes the exported history. That history is a real observation
of a real self-modifying process — nothing about it was written by a
model asked to imagine data.

What keeps it honest (each is a check here, not a convention):

1. The recipe — harness, command, working directory, seed, iteration
   budget, output layout, world id — is written to the chain
   (`REGISTER_HARVEST`) *before* the command runs. The freeze carries the
   recipe digest; a freeze whose digest has no earlier registration is
   refused.
2. The command is a fixed argv list. No shell, no network flag added by
   FarField, no probe script anywhere near it. Timeout is the recipe's.
3. The frozen world says `provenance: harvested` and its origin carries
   `harness_run` (argv, seed, exit code, duration, stdout/stderr
   digests). A reader can tell it from a fetched archive.
4. One harvest is one seed. It can support; the confirmation is a
   second harvest of the same recipe with a different seed — never a
   rerun of the probe on the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import chain
from .freeze import FreezeError, freeze_history
from .histories import FORMATS

_REQUIRED = ("world_id", "harness", "command", "format", "output")
DEFAULT_TIMEOUT = 1800.0


class HarvestError(ValueError):
    """The recipe or the run was refused; the catalog is unchanged."""


@dataclass(frozen=True)
class HarvestRecipe:
    world_id: str
    harness: str
    command: tuple[str, ...]
    fmt: str
    output: str
    workdir: str = "."
    seed: int = 0
    iterations: int = 0
    timeout: float = DEFAULT_TIMEOUT
    title: str = ""
    slice_rule: str = "all"
    env: dict[str, str] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "harness": self.harness,
            "command": list(self.command),
            "format": self.fmt,
            "output": self.output,
            "workdir": self.workdir,
            "seed": self.seed,
            "iterations": self.iterations,
            "timeout": self.timeout,
            "title": self.title,
            "slice_rule": self.slice_rule,
            "env": dict(sorted(self.env.items())),
            "note": self.note,
        }

    def digest(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def body_digest(self) -> str:
        """The recipe with seed, world id, and title removed.

        Two harvests confirm each other when this matches and the seed
        does not."""
        body = {
            k: v for k, v in self.to_dict().items() if k not in {"seed", "world_id", "title"}
        }
        text = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_recipe(payload: dict[str, Any]) -> HarvestRecipe:
    """Validate a recipe object. Nothing runs here."""
    if not isinstance(payload, dict):
        raise HarvestError("recipe must be a JSON object")
    missing = [key for key in _REQUIRED if not payload.get(key)]
    if missing:
        raise HarvestError(f"recipe is missing {', '.join(missing)}")
    command = payload["command"]
    if isinstance(command, str):
        command = shlex.split(command)
    if not isinstance(command, (list, tuple)) or not all(isinstance(c, str) and c for c in command):
        raise HarvestError("recipe command must be a non-empty argv list")
    fmt = str(payload["format"]).strip().lower()
    if fmt not in FORMATS:
        raise HarvestError(f"recipe format {fmt!r} is not importable; use {list(FORMATS)}")
    env = payload.get("env") or {}
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise HarvestError("recipe env must map strings to strings")
    try:
        seed = int(payload.get("seed") or 0)
        iterations = int(payload.get("iterations") or 0)
        timeout = float(payload.get("timeout") or DEFAULT_TIMEOUT)
    except (TypeError, ValueError) as exc:
        raise HarvestError("seed and iterations must be integers, timeout a number") from exc
    if timeout <= 0:
        raise HarvestError("timeout must be positive")
    return HarvestRecipe(
        world_id=str(payload["world_id"]).strip(),
        harness=str(payload["harness"]).strip(),
        command=tuple(str(c) for c in command),
        fmt=fmt,
        output=str(payload["output"]).strip(),
        workdir=str(payload.get("workdir") or ".").strip() or ".",
        seed=seed,
        iterations=iterations,
        timeout=timeout,
        title=str(payload.get("title") or "").strip(),
        slice_rule=str(payload.get("slice_rule") or "all").strip() or "all",
        env=dict(env),
        note=str(payload.get("note") or "").strip(),
    )


def load_recipe(path: Path) -> HarvestRecipe:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarvestError(f"cannot read recipe {path}: {exc}") from exc
    return parse_recipe(payload)


def register_recipe(recipe: HarvestRecipe, *, catalog: Path) -> dict[str, Any]:
    """Write the recipe to the catalog chain before anything runs."""
    catalog = Path(catalog)
    catalog.mkdir(parents=True, exist_ok=True)
    return chain.append_event(
        catalog / "chain.jsonl",
        chain.REGISTER_HARVEST,
        {"recipe_digest": recipe.digest(), "recipe": recipe.to_dict()},
    )


def registered(recipe: HarvestRecipe, *, catalog: Path) -> bool:
    path = Path(catalog) / "chain.jsonl"
    if not path.is_file():
        return False
    digest = recipe.digest()
    for event in chain.read_events(path):
        if event.get("kind") == chain.REGISTER_HARVEST and (
            (event.get("payload") or {}).get("recipe_digest") == digest
        ):
            return True
    return False


def last_successful_run(recipe: HarvestRecipe, *, catalog: Path) -> dict[str, Any] | None:
    """The chain's most recent EXECUTE_HARVEST with exit 0 for this recipe."""
    path = Path(catalog) / "chain.jsonl"
    if not path.is_file():
        return None
    digest = recipe.digest()
    found: dict[str, Any] | None = None
    for event in chain.read_events(path):
        payload = event.get("payload") or {}
        if (
            event.get("kind") == chain.EXECUTE_HARVEST
            and payload.get("recipe_digest") == digest
            and payload.get("exit_code") == 0
        ):
            found = {**payload, "chain_seq": event.get("seq"), "chain_at": event.get("at")}
    return found


def run_harvest(
    recipe: HarvestRecipe,
    *,
    catalog: Path,
    force: bool = False,
    runner: Any = None,
    reuse_run: bool = False,
) -> dict[str, Any]:
    """Register (if not already), run the harness, import, freeze.

    `runner(argv, cwd, env, timeout) -> CompletedProcess` exists for
    tests; the default is `subprocess.run` with no shell. `reuse_run`
    skips the harness when the chain already records a successful run of
    this exact recipe and its output directory is still there — an
    importer fix must not cost another paid run; the freeze cites the
    recorded run.
    """
    catalog = Path(catalog)
    if not registered(recipe, catalog=catalog):
        register_recipe(recipe, catalog=catalog)
    workdir = Path(recipe.workdir)
    if not workdir.is_dir():
        raise HarvestError(f"recipe workdir missing: {workdir}")
    if reuse_run:
        previous = last_successful_run(recipe, catalog=catalog)
        output = Path(recipe.output)
        if not output.is_absolute():
            output = workdir / output
        if previous is None or not output.is_dir():
            raise HarvestError(
                "no successful recorded run of this recipe with its output still "
                "present; run without reuse"
            )
        return _freeze_harvest(recipe, output, previous, catalog=catalog, force=force)
    import os

    env = {**os.environ, **recipe.env, "FARFIELD_HARVEST_SEED": str(recipe.seed)}
    argv = [
        part.replace("{seed}", str(recipe.seed)).replace("{iterations}", str(recipe.iterations))
        for part in recipe.command
    ]
    started = time.monotonic()
    call = runner or _default_runner
    try:
        proc = call(argv, str(workdir), env, recipe.timeout)
    except subprocess.TimeoutExpired as exc:
        chain.append_event(
            catalog / "chain.jsonl",
            chain.EXECUTE_HARVEST,
            {"recipe_digest": recipe.digest(), "status": "timeout", "seconds": recipe.timeout},
        )
        raise HarvestError(f"harness timed out after {recipe.timeout:.0f}s") from exc
    duration = time.monotonic() - started
    stdout = _bytes(getattr(proc, "stdout", b""))
    stderr = _bytes(getattr(proc, "stderr", b""))
    run_record = {
        "recipe_digest": recipe.digest(),
        "harness": recipe.harness,
        "argv": argv,
        "workdir": str(workdir),
        "seed": recipe.seed,
        "iterations": recipe.iterations,
        "exit_code": int(getattr(proc, "returncode", -1)),
        "seconds": round(duration, 3),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_tail": stderr[-2000:].decode("utf-8", errors="replace"),
    }
    chain.append_event(
        catalog / "chain.jsonl",
        chain.EXECUTE_HARVEST,
        {k: v for k, v in run_record.items() if k != "stderr_tail"},
    )
    if run_record["exit_code"] != 0:
        raise HarvestError(
            f"harness exited {run_record['exit_code']}; nothing frozen. "
            f"stderr tail: {run_record['stderr_tail'][-400:]}"
        )
    output = Path(recipe.output)
    if not output.is_absolute():
        output = workdir / output
    if not output.is_dir():
        raise HarvestError(f"harness did not produce {output}")
    return _freeze_harvest(recipe, output, run_record, catalog=catalog, force=force)


def _freeze_harvest(
    recipe: HarvestRecipe,
    output: Path,
    run_record: dict[str, Any],
    *,
    catalog: Path,
    force: bool,
) -> dict[str, Any]:
    try:
        fixture = freeze_history(
            world_id=recipe.world_id,
            run_dir=output,
            fmt=recipe.fmt,
            slice_rule=recipe.slice_rule,
            catalog=catalog,
            title=recipe.title or f"{recipe.harness} harvest seed {recipe.seed}",
            source=(
                f"harvested: {recipe.harness} run under recipe {recipe.digest()[:12]}… "
                f"(seed {recipe.seed}, {recipe.iterations} iterations)"
            ),
            force=force,
            provenance="harvested",
            extra_origin={"harness_run": run_record},
            extra_manifest={
                "recipe_digest": recipe.digest(),
                "recipe_body": recipe.body_digest(),
                "harness": recipe.harness,
                "seed": recipe.seed,
            },
        )
    except FreezeError as exc:
        raise HarvestError(str(exc)) from exc
    return {"ok": True, **fixture.to_dict(), "root": str(fixture.root), "harness_run": run_record}


def sibling_seeds(world_id: str, *, catalog: Path) -> list[dict[str, Any]]:
    """Other harvested worlds frozen from the same recipe body but another seed.

    The confirmation of a harvested WORLD is one of these, not a rerun on
    the same bytes.
    """
    from .world import load_catalog

    root = Path(catalog)
    fixtures = load_catalog(root.parent if root.name == "worlds" else root)
    me = fixtures.get(world_id)
    if me is None:
        return []
    mine = _manifest(me.root)
    body = mine.get("recipe_body")
    if not body:
        return []
    out: list[dict[str, Any]] = []
    for fixture in fixtures.values():
        if fixture.id == world_id or fixture.provenance != "harvested":
            continue
        other = _manifest(fixture.root)
        if other.get("recipe_body") == body and other.get("seed") != mine.get("seed"):
            out.append({"world_id": fixture.id, "seed": other.get("seed"), "digest": fixture.digest})
    return out


def _manifest(root: Path) -> dict[str, Any]:
    try:
        return json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _default_runner(argv: list[str], cwd: str, env: dict[str, str], timeout: float):
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        timeout=timeout,
        check=False,
        shell=False,
    )


def _bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if value is None:
        return b""
    return str(value).encode("utf-8", errors="replace")
