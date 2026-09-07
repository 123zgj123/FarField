"""FarField plugin host: skills with executable hooks, not prompt paste.

DeepSeek Harness's official plugins are Cordis TypeScript
(`apply(ctx)` + `ctx.tools.register`). FarField stays stdlib-only
(`dependencies = []`), so this module is the product plugin runtime:

- A skill directory is `.agents/skills/<name>/SKILL.md` (the DSH leaf).
- If that directory also has a trusted `plugin.py`, PluginHost *calls*
  its hooks. The model reading SKILL.md is optional colour; the hook is
  the evaluation function.
- Distilled skills are SKILL.md only. LLM-written Python is never exec'd.
- DeepSeek Harness itself is one plugin: when `dsh` is on PATH the host
  actually launches it; when it is not, local hooks still run.

Hooks cannot kill a live card after the probe, rewrite expected_direction,
or climb the promotion ladder. They can only refuse a payload before it
is admitted, or return an intern observation.
"""

from __future__ import annotations

import importlib.util
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from .skills import Skill, load_catalog

KNOWN_HOOKS = (
    "validate_probe",
    "validate_diagnosis",
    "lock_claim",
    "lock_writeup",
    "after_evidence",
    "execute",
    # Extension points: a plugin may launch an external job or add
    # artifact files. Neither hook may set a scientific verdict.
    "run_external",
    "compile_artifact",
)


class PluginError(RuntimeError):
    """A plugin file could not be loaded or a hook exploded."""


class Plugin(Protocol):
    name: str

    def has(self, hook: str) -> bool: ...
    def call(self, hook: str, **kwargs: Any) -> Any: ...
    def to_dict(self) -> dict[str, Any]: ...


def repo_root() -> Path:
    here = Path(__file__).resolve()
    # src/farfield/extras/plugins.py → repo
    if len(here.parents) >= 4 and (here.parents[3] / ".agents" / "skills").is_dir():
        return here.parents[3]
    cwd = Path.cwd()
    if (cwd / ".agents" / "skills").is_dir():
        return cwd
    return here.parents[3] if len(here.parents) >= 4 else cwd


def trusted_plugin_roots(repo: Path) -> tuple[Path, ...]:
    """Only these trees may supply plugin.py. Workspace distillations cannot."""
    repo = Path(repo)
    return (repo / ".agents" / "skills", repo / ".dsh" / "skills")


def _is_trusted(path: Path, roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    for root in roots:
        if not root.exists():
            continue
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


class BuiltinPlugin:
    """Fail-closed teeth so a missing catalog cannot naked-run a probe."""

    name = "builtin"

    def has(self, hook: str) -> bool:
        return hook in {
            "validate_probe",
            "lock_claim",
            "lock_writeup",
            "validate_diagnosis",
        }

    def call(self, hook: str, **kwargs: Any) -> Any:
        return getattr(self, hook)(**kwargs)

    def validate_probe(
        self,
        tree: Any = None,
        source: str = "",
        measure: str = "",
        claim: str = "",
        mechanism: str = "",
        topic: str = "",
        **kwargs: Any,
    ) -> str | None:
        from .domain import measure_stays_on_object
        from .probeexp import ProbeRefused, assert_fair_probe

        if tree is None:
            return "probe source did not parse"
        try:
            assert_fair_probe(tree)
        except ProbeRefused as exc:
            return exc.record.unlock_condition
        if not measure_stays_on_object(
            measure, claim, mechanism=mechanism, topic=topic
        ):
            return (
                "the probe measure names the far-field mechanism's cost, not"
                " the claim's scientific object; count the object (protocol"
                " field, invariant, index) — not cut-maintenance hops"
            )
        from .worldfields import probe_reads_attested_fields

        schema = str(kwargs.get("schema") or "")
        source = str(source or "")
        return probe_reads_attested_fields(
            source, claim=claim, mechanism=mechanism, schema=schema
        )

    def lock_claim(
        self,
        claim: str = "",
        mechanism: str = "",
        topic: str = "",
        seed_label: str = "",
        **_: Any,
    ) -> str | None:
        from .domain import claim_covers_topic

        if topic.strip() and not claim_covers_topic(claim, mechanism, topic, seed_label):
            return (
                "claim and mechanism must use at least one content word from the"
                " topic that is not already the seed label; a distant concept is"
                " a mechanism, not a license to change fields"
            )
        return None

    def lock_writeup(
        self,
        title: str = "",
        idea: str = "",
        home: str = "",
        topic: str = "",
        seed_label: str = "",
        **_: Any,
    ) -> str | None:
        from .domain import writeup_retrofits_topic

        if not (topic.strip() and home.strip()):
            return None
        found = writeup_retrofits_topic(
            title, idea, home=home, topic=topic, seed_label=seed_label
        )
        if found:
            return (
                "title or idea imported topic-field terms the claim does not"
                " own (" + ", ".join(found) + "); writing cannot change"
                " the domain the hypothesis belongs to"
            )
        return None

    def validate_diagnosis(
        self,
        treatment_arm: str = "",
        control_arm: str = "",
        experiment: str = "",
        claim: str = "",
        mechanism: str = "",
        topic: str = "",
        **kwargs: Any,
    ) -> str | None:
        from .domain import experiment_stays_on_object

        left = (treatment_arm or "").strip().lower()
        right = (control_arm or "").strip().lower()
        if left and right and left == right:
            return (
                "treatment and control arms are identical; a two-arm diagnosis"
                " must remove the mechanism in the control"
            )
        if not experiment_stays_on_object(
            experiment,
            treatment_arm,
            claim,
            mechanism=mechanism,
            topic=topic,
        ):
            return (
                "the diagnosis measures the distant mechanism instead of the"
                " claim's object; the two arms must test the topic artefact"
            )
        from .worldfields import (
            diagnosis_names_claimed_fields,
            missing_attested_field,
        )

        schema = str(kwargs.get("schema") or "")
        missing = missing_attested_field(
            claim,
            mechanism,
            schema,
            str(kwargs.get("world_lever") or ""),
        )
        if missing:
            return missing
        return diagnosis_names_claimed_fields(
            experiment=experiment,
            treatment=treatment_arm,
            control=control_arm,
            claim=claim,
            mechanism=mechanism,
            schema=schema,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "executable": True,
            "trusted": True,
            "hooks": [hook for hook in KNOWN_HOOKS if self.has(hook)],
            "source": "farfield.extras.plugins.BuiltinPlugin",
        }


@dataclass
class FilePlugin:
    name: str
    path: Path
    module: Any
    skill: Skill | None
    trusted: bool = True

    @classmethod
    def load(cls, path: Path, skill: Skill | None = None) -> FilePlugin:
        path = Path(path)
        slug = (skill.name if skill is not None else path.parent.name).replace("-", "_")
        mod_name = f"farfield_plugin_{slug}_{abs(hash(str(path.resolve())))}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise PluginError(f"cannot load plugin {path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 — load failure is a plugin error
            raise PluginError(f"plugin {path} failed to import: {exc}") from exc
        name = str(getattr(module, "NAME", "") or (skill.name if skill else path.parent.name))
        return cls(name=name, path=path, module=module, skill=skill, trusted=True)

    def has(self, hook: str) -> bool:
        return hook in KNOWN_HOOKS and callable(getattr(self.module, hook, None))

    def call(self, hook: str, **kwargs: Any) -> Any:
        fn: Callable[..., Any] = getattr(self.module, hook)
        return fn(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "executable": True,
            "trusted": self.trusted,
            "hooks": [hook for hook in KNOWN_HOOKS if self.has(hook)],
            "source": str(self.path),
            "skill": self.skill.name if self.skill is not None else None,
        }


class PluginHost:
    """Discovers trusted plugin.py files and runs their hooks."""

    def __init__(
        self,
        plugins: tuple[Plugin, ...] = (),
        catalog: tuple[Skill, ...] = (),
    ) -> None:
        self.plugins = plugins
        self.catalog = catalog

    @classmethod
    def load(
        cls,
        repo: Path,
        workspace: Path | None = None,
        extra: tuple[Path, ...] = (),
        *,
        include_dsh: bool = True,
        include_builtin: bool = True,
    ) -> PluginHost:
        catalog = load_catalog(repo, workspace, extra)
        loaded: list[Plugin] = []
        if include_builtin:
            loaded.append(BuiltinPlugin())
        trusted = trusted_plugin_roots(Path(repo))
        seen: set[Path] = set()
        for skill in catalog:
            if not skill.source:
                continue
            plugin_py = Path(skill.source).parent / "plugin.py"
            if not plugin_py.is_file():
                continue
            resolved = plugin_py.resolve()
            if resolved in seen or not _is_trusted(plugin_py, trusted):
                continue
            seen.add(resolved)
            try:
                loaded.append(FilePlugin.load(plugin_py, skill))
            except PluginError:
                continue
        if include_dsh:
            from .harness import DeepSeekHarnessPlugin

            loaded.append(DeepSeekHarnessPlugin())
        return cls(tuple(loaded), catalog=catalog)

    def by_name(self, name: str) -> Plugin | None:
        for plugin in self.plugins:
            if plugin.name == name:
                return plugin
        return None

    def run(self, hook: str, **kwargs: Any) -> str | None:
        """First refuse reason wins. None means every plugin passed."""
        for plugin in self.plugins:
            if not plugin.has(hook) or hook in {
                "execute",
                "run_external",
                "compile_artifact",
            }:
                continue
            try:
                reason = plugin.call(hook, **kwargs)
            except Exception as exc:  # noqa: BLE001 — a broken hook is a refuse
                return f"{plugin.name} hook {hook} crashed: {exc}"
            if reason:
                return str(reason)
        return None

    def notify(self, hook: str, **kwargs: Any) -> None:
        for plugin in self.plugins:
            if not plugin.has(hook) or hook in {
                "execute",
                "run_external",
                "compile_artifact",
            }:
                continue
            try:
                plugin.call(hook, **kwargs)
            except Exception:
                continue

    def offer(self, hook: str, **kwargs: Any) -> Any:
        """First plugin that returns a non-None payload handles the extension.

        Used by `run_external` and `compile_artifact`. A refuse-style hook
        still goes through `run()`. Returning a verdict is ignored by the
        callers — they only accept treatment/control numbers or extra files.
        """
        if hook not in {"run_external", "compile_artifact"}:
            return None
        for plugin in self.plugins:
            if not plugin.has(hook):
                continue
            try:
                result = plugin.call(hook, **kwargs)
            except Exception:
                continue
            if result is not None:
                return result
        return None

    def execute(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        workspace: Path | None = None,
    ) -> str:
        plugin = self.by_name(name)
        if plugin is None:
            return f"unknown plugin {name!r}; loaded: {self.executable_names()}"
        if not plugin.has("execute"):
            return (
                f"{name} has SKILL.md but no execute() hook; a prompt-only"
                " skill cannot be invoked as a tool"
            )
        try:
            result = plugin.call("execute", args=args or {}, workspace=workspace)
        except Exception as exc:  # noqa: BLE001 — intern observation
            return f"{name} execute crashed: {exc}"
        return str(result)

    def executable_names(self) -> list[str]:
        return [
            plugin.name
            for plugin in self.plugins
            if plugin.has("execute") or plugin.name == "builtin"
        ]

    def summary(self) -> list[dict[str, Any]]:
        rows = [plugin.to_dict() for plugin in self.plugins]
        prompt_only = []
        executable = {row.get("skill") or row["name"] for row in rows}
        for skill in self.catalog:
            if skill.name not in executable:
                prompt_only.append(
                    {
                        "name": skill.name,
                        "executable": False,
                        "trusted": False,
                        "hooks": [],
                        "source": skill.source,
                        "skill": skill.name,
                    }
                )
        return rows + prompt_only

    def dsh_status(self) -> dict[str, Any]:
        from .harness import dsh_status

        return dsh_status()

    def tool_lines(self) -> str:
        names = [
            plugin.name
            for plugin in self.plugins
            if plugin.has("execute") and plugin.name != "builtin"
        ]
        if not names:
            return "(none loaded)"
        return ", ".join(names)


_tls = threading.local()
_default_lock = threading.Lock()
_default_host: PluginHost | None = None


def get_host() -> PluginHost:
    bound = getattr(_tls, "host", None)
    if bound is not None:
        return bound
    global _default_host
    with _default_lock:
        if _default_host is None:
            _default_host = PluginHost.load(repo_root())
        return _default_host


def reset_default_host() -> None:
    global _default_host
    with _default_lock:
        _default_host = None


@contextmanager
def bind_host(host: PluginHost) -> Iterator[PluginHost]:
    previous = getattr(_tls, "host", None)
    _tls.host = host
    try:
        yield host
    finally:
        _tls.host = previous


def refuse_probe(tree: Any, source: str = "", **kwargs: Any) -> str | None:
    return get_host().run("validate_probe", tree=tree, source=source, **kwargs)


def refuse_claim(
    claim: str, mechanism: str, topic: str, seed_label: str
) -> str | None:
    return get_host().run(
        "lock_claim",
        claim=claim,
        mechanism=mechanism,
        topic=topic,
        seed_label=seed_label,
    )


def refuse_writeup(
    title: str,
    idea: str,
    home: str,
    topic: str,
    seed_label: str,
) -> str | None:
    return get_host().run(
        "lock_writeup",
        title=title,
        idea=idea,
        home=home,
        topic=topic,
        seed_label=seed_label,
    )


def refuse_diagnosis(**kwargs: Any) -> str | None:
    return get_host().run("validate_diagnosis", **kwargs)
