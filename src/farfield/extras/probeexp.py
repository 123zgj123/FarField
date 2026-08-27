"""A bounded, discriminative experiment probe for a surviving idea.

The probe implements the diagnosis's two-arm experiment (see
`diagnose.py`): a treatment arm that isolates the proposed mechanism and
a control arm that removes exactly it. The script must be stdlib-only,
pass the whitelist compiler, run in an isolated cwd under a timeout, and
write both numbers into `metrics.json` as `treatment` and `control`.

The probe writer never declares what outcome would be good — the
expected direction was pre-registered in the diagnosis before this call,
so the model cannot pick the experiment its idea is guaranteed to win.
A crash or a refused source is the probe's failure, never a reason to
revive a dead card. The numbers are a synthetic check on a constructed
dataset, not a claim that the science is done. The writeup says so.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import BlockedRecord
from .brief import ResearchBrief
from .generate import GeneratedCard, GenerationRefused
from .llm import Completion
from .world import reads_world_data

ALLOWED_MODULES = frozenset(
    {
        "array",
        "bisect",
        "collections",
        "copy",
        "dataclasses",
        "decimal",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "pathlib",
        "random",
        "statistics",
        "string",
        "textwrap",
    }
)


@dataclass(frozen=True)
class ComputeTier:
    """A pre-registered compute budget for one experiment.

    The tier is chosen at diagnosis time — before any probe runs — so a
    bigger budget can never be a reaction to a result. `extra_modules`
    widens the import whitelist in both stages (probe and host run the
    same digest-matched script); `timeout_seconds` is the host budget;
    the in-loop probe filter keeps its own 20s. `memory_bytes` caps the
    process address space so a heavy run cannot take the host down.
    """

    name: str
    timeout_seconds: float
    extra_modules: frozenset[str] = frozenset()
    memory_bytes: int | None = None

    @property
    def allowed_modules(self) -> frozenset[str]:
        return ALLOWED_MODULES | self.extra_modules


COMPUTE_TIERS: dict[str, ComputeTier] = {
    "sandbox": ComputeTier(name="sandbox", timeout_seconds=20.0),
    "host": ComputeTier(name="host", timeout_seconds=600.0),
    "host-heavy": ComputeTier(
        name="host-heavy",
        timeout_seconds=3600.0,
        extra_modules=frozenset({"numpy"}),
        memory_bytes=8 * 1024**3,
    ),
}
DEFAULT_TIER = "host"


def resolve_tier(name: str | None) -> ComputeTier:
    """Named tier, or the default host budget. Unknown names do not
    invent a bigger budget."""
    return COMPUTE_TIERS.get(str(name or "").strip().lower(), COMPUTE_TIERS[DEFAULT_TIER])

_ALLOWED_NODES = frozenset(
    {
        ast.Module,
        ast.Import,
        ast.ImportFrom,
        ast.alias,
        ast.FunctionDef,
        # A toy data structure is naturally a class; under the import
        # whitelist and the underscore/name bans a ClassDef adds no
        # capability a FunctionDef does not already have.
        ast.ClassDef,
        ast.arguments,
        ast.arg,
        ast.Lambda,
        ast.Expr,
        ast.Return,
        ast.Pass,
        # `with open(...) as f:` is how real model code writes metrics.json;
        # under the import whitelist and name bans it adds no capability.
        ast.With,
        ast.withitem,
        ast.If,
        ast.For,
        ast.While,
        ast.Break,
        ast.Continue,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Delete,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Del,
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.Set,
        ast.Dict,
        ast.Subscript,
        ast.Slice,
        ast.Attribute,
        ast.Call,
        ast.keyword,
        ast.IfExp,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.USub,
        ast.UAdd,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        # Bit ops are the mother tongue of succinct-structure and hashing
        # probes (shifts, masks, packing); under the import whitelist they
        # add no capability. Two live probes died to `>>` before this.
        ast.LShift,
        ast.RShift,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.Invert,
        # The natural way to write a counter closure.
        ast.Nonlocal,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        ast.ListComp,
        ast.SetComp,
        ast.GeneratorExp,
        ast.DictComp,
        ast.comprehension,
        ast.JoinedStr,
        ast.FormattedValue,
    }
)

_FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "breakpoint",
        "input",
        "exit",
        "quit",
        "help",
        "memoryview",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    }
)

SYSTEM = (
    "You write a tiny, self-contained Python probe. Answer with one JSON"
    " object and nothing else."
)

TEMPLATE = """Implement this pre-registered two-arm experiment as a small Python script. The script must run with the standard library only, finish in a few seconds, and write metrics.json with exactly two finite numbers: "treatment" and "control".

Researcher's topic: {topic}
Claim: {claim}
Proposed mechanism: {mechanism}
Strongest competing explanation: {alternative}
Experiment to implement: {experiment}
Treatment arm: {treatment_arm}
Control arm: {control_arm}

The two arms must differ ONLY in the mechanism under test. If no frozen world is bound, this run is a coherence check on invented data: it can weaken, it cannot corroborate, it cannot stop distant exploration, and it cannot be distilled. Prefer not to invent. Use the same data and the same measurement in both arms. Both arms MUST call one shared `measure(...)` (or equivalent); the only difference is a boolean or config that turns the mechanism on or off. Do not hard-code different numbers for the two arms. Do not `pass` in one arm and count in the other. Do not increment a counter unconditionally in treatment and under a predicate in control. Do not touch the network. Do not decide which outcome is good — the expected direction was pre-registered before this call.

The sandbox is deterministic: only these modules may be imported — {allowed_modules}. There is no `time` module, so never measure wall-clock speed; count the work instead (comparisons, node visits, pointer hops, bytes written) with an explicit counter, and report the counts. If the experiment uses any randomness, sampling, or arbitrary ordering, derive ALL of it from the integer in data/seed.json when that file exists (fall back to 0 when it does not) — the runtime replays this same script under seeds it chooses to test whether the effect survives variation, and a script that ignores its seed cannot be confirmed at that tier. Dunder names (`__import__`, `__class__`, `__globals__`) are forbidden; ordinary helpers like `_load_fixture` are allowed. print/eval/exec/open-for-anything-but-metrics are unavailable. Use this skeleton:

```
def setup(world):
    ...
def measure(world, mechanism_enabled):
    ...
def run():
    control = measure(world, False)
    treatment = measure(world, True)
```

The two `measure` calls must share every argument except `mechanism_enabled`.

Answer with JSON:
{{"measure": "<=20 words: what single quantity both arms report",
 "source": "complete experiment.py that writes metrics.json as {{\\"treatment\\": <number>, \\"control\\": <number>}}"}}
"""

IMPLEMENTATION_FAILURE = """

The previous implementation of this SAME registered experiment failed before producing two arms.
Failure: {status}: {error}

This is an implementation bug, not a scientific verdict. Rewrite the whole script to implement the same treatment and control design. Do not change the experiment. Do not invent a new metric to chase a win. Do not look at treatment/control numbers — there were none.
"""

WORLD_BINDING = """

A frozen experimental world is already in the probe cwd as data/. Files: {files}.
World: {world_id} — {title}
Schema: {schema}
How to load (stdlib): {load_hint}
Source (attested; do not fetch): {source}
world.json repeats this metadata.

Do NOT invent a dataset. Load the instance from data/ with pathlib as in the load hint. Both arms must measure that SAME loaded instance and differ only by the mechanism flag. A script that never names data/ is still SYNTHETIC and will be refused. Report a quantity of the CLAIM's object, not the distant mechanism's internal cost.
"""

CONSTRUCTED_BINDING = """

The experimental world for THIS registered experiment is in data/. Files: {files}.
World: {world_id} — {title}
Schema: {schema}
How to load (stdlib): {load_hint}
Excerpt:
{excerpt}

This world was constructed this iteration. It is not a freeze. The run is GENERATED: it can weaken, it cannot corroborate. Do NOT invent a different graph. Load data/world.json. Both arms measure that SAME instance and differ only by the mechanism flag. Do not write treatment/control into the world.
"""

PAPER_CONTROL = """

The control arm implements a method from a paper retrieved this mission, not a restatement of related work.
Cite id: {cite_id}
Method: {method}
Treatment still turns the proposed mechanism on. Control instantiates that paper method on the SAME bound bytes.
"""

# The confirmation tier does not ask the model for a second, "heavier"
# script. The model may generate computation; it may not generate the
# evidence that audits its own computation. Replication is owned by the
# trusted executor: it reruns the *registered* script (same digest the
# probe and the host reproduction used) under seeds it derives itself —
# see `replication_seeds` and `mission._heavy_confirmation`.

HEAVY_TIMEOUT_SECONDS = 120.0


class ProbeRefused(GenerationRefused):
    pass


def _refuse(attempted: str, unlock: str) -> ProbeRefused:
    return ProbeRefused(
        BlockedRecord(
            missing_capability="model_probe_schema",
            attempted=attempted,
            unlock_condition=unlock,
        )
    )


def _parse(completion: Completion) -> dict[str, Any]:
    text = completion.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _refuse(
            "parse a JSON experiment probe",
            f"the model answers with one JSON object; it answered with {text[:80]!r} ({exc})",
        ) from exc
    if not isinstance(payload, dict):
        raise _refuse(
            "read a JSON object for the probe",
            f"the top-level value is an object, not {type(payload).__name__}",
        )
    return payload


def validate_source(source: str, *, tier: ComputeTier | None = None) -> ast.Module:
    """Whitelist-validate probe source, or raise ProbeRefused.

    `tier` widens the import whitelist for a pre-registered heavy budget;
    every other rule (node subset, dunder bans, name bans) is identical.
    """
    allowed_modules = tier.allowed_modules if tier is not None else ALLOWED_MODULES
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise _refuse("parse experiment.py", f"not parseable python: {error}") from error
    if not isinstance(tree, ast.Module) or not tree.body:
        raise _refuse("read experiment.py", "the source must be a non-empty module")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES and not isinstance(node, ast.expr_context):
            raise _refuse(
                "compile experiment.py",
                f"{type(node).__name__} is not in the allowed subset",
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise _refuse(
                "compile experiment.py",
                f"dunder attribute {node.attr!r} is not allowed",
            )
        if isinstance(node, ast.Name) and (
            node.id.startswith("__") or node.id in _FORBIDDEN_NAMES
        ):
            raise _refuse(
                "compile experiment.py",
                f"name {node.id!r} is not allowed",
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_modules:
                    raise _refuse(
                        "compile experiment.py",
                        f"import {alias.name!r} is not on the module whitelist"
                        " for this compute tier",
                    )
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in allowed_modules:
                raise _refuse(
                    "compile experiment.py",
                    f"from {node.module!r} is not on the module whitelist"
                    " for this compute tier",
                )
    return tree


def _is_pass_only(body: list[ast.stmt]) -> bool:
    return bool(body) and all(isinstance(stmt, ast.Pass) for stmt in body)


def _has_counter(body: list[ast.stmt]) -> bool:
    for stmt in body:
        if isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
            return True
        if isinstance(stmt, ast.Assign) and any(
            isinstance(target, ast.Name) for target in stmt.targets
        ):
            # `detected = detected + 1` counts; `flag = True` is not a score.
            if isinstance(stmt.value, ast.BinOp):
                return True
        if isinstance(stmt, ast.If) and (
            _has_counter(stmt.body) or _has_counter(stmt.orelse)
        ):
            return True
        if isinstance(stmt, ast.For) and _has_counter(stmt.body):
            return True
        if isinstance(stmt, ast.While) and _has_counter(stmt.body):
            return True
    return False


def _unconditional_counter(body: list[ast.stmt]) -> bool:
    return any(
        isinstance(stmt, ast.AugAssign)
        or (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.BinOp)
        )
        for stmt in body
    )


def _gated_counter(body: list[ast.stmt]) -> bool:
    return any(isinstance(stmt, ast.If) and _has_counter(stmt.body) for stmt in body)


def _literal_number(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
        node.value, bool
    )


def assert_fair_probe(tree: ast.Module) -> None:
    """Refuse two-arm scripts that hard-code an asymmetric win.

    The evaluation function for a probe is "same measure, one mechanism
    flag". A treatment arm that always increments while the control is
    gated, a `pass` versus a counter, or metrics.json filled with two
    numeric literals, is not an experiment — it is the author picking the
    outcome. Caught here, before the process runs, so a live `supports`
    cannot be laundered from `treatment: 80, control: 120`.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            arms: dict[str, ast.AST | None] = {}
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in {"treatment", "control"}:
                    arms[str(key.value)] = value
            if "treatment" in arms and "control" in arms:
                t_lit = _literal_number(arms["treatment"])
                c_lit = _literal_number(arms["control"])
                if t_lit and c_lit:
                    raise _refuse(
                        "write a computed two-arm metric",
                        "treatment and control in metrics.json are both numeric"
                        " literals; a probe that does not measure cannot support"
                        " a claim",
                    )
                if t_lit != c_lit:
                    raise _refuse(
                        "write a symmetric two-arm metric",
                        "one arm's metric is a hardcoded number and the other"
                        " is computed; both arms must call the same measure",
                    )
        if isinstance(node, ast.If):
            then, otherwise = node.body, node.orelse
            if (_is_pass_only(then) and _has_counter(otherwise)) or (
                _is_pass_only(otherwise) and _has_counter(then)
            ):
                raise _refuse(
                    "write two live arms",
                    "one branch is `pass` and the other counts; that is an"
                    " asymmetric hardcoded win, not a two-arm experiment",
                )
            if (_unconditional_counter(then) and _gated_counter(otherwise)) or (
                _unconditional_counter(otherwise) and _gated_counter(then)
            ):
                raise _refuse(
                    "gate both arms the same way",
                    "one arm increments unconditionally and the other only"
                    " under a predicate; both arms must share the same"
                    " measure and differ by a mechanism flag",
                )


@dataclass(frozen=True)
class ProbeSpec:
    measure: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "source": self.source,
        }


def spec_from_payload(
    payload: dict[str, Any],
    *,
    tier: ComputeTier | None = None,
    claim: str = "",
    mechanism: str = "",
    topic: str = "",
) -> ProbeSpec:
    measure = str(payload.get("measure") or "").strip()
    source = str(payload.get("source") or "").strip()
    if source.startswith("```"):
        source = source.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if not measure or not source:
        raise _refuse(
            "read measure and source",
            "the probe names the quantity both arms report and includes experiment.py",
        )
    validate_source(source, tier=tier)
    tree = ast.parse(source)
    from .plugins import refuse_probe

    locked = refuse_probe(
        tree,
        source=source,
        measure=measure,
        claim=claim or str(payload.get("claim") or ""),
        mechanism=mechanism or str(payload.get("mechanism") or ""),
        topic=topic or str(payload.get("topic") or ""),
    )
    if locked:
        raise _refuse("write a fair two-arm probe", locked)
    return ProbeSpec(measure=measure, source=source)


def write_probe(
    client: Any,
    card: GeneratedCard,
    diagnosis: Any,
    topic: str,
    *,
    retries: int = 1,
    skills: str = "",
    prior_failure: dict[str, Any] | None = None,
    world: Any = None,
) -> ProbeSpec:
    """The probe implements the diagnosis; it does not pick its own test.

    A sandbox refusal is a compiler error, not a scientific verdict, so the
    writer gets `retries` more attempts with the exact rejection quoted back.
    Only the final refusal propagates.

    `prior_failure` is an earlier in-mission implementation crash or
    sandbox refusal of this same registration. Quote status and error,
    never two-arm numbers: the redesign must fix the script, not chase a
    sign.

    `world` is an attested fixture already copied into the probe cwd as
    `data/`. When set, the script must read those files; inventing a
    stand-in dataset is refused so it cannot be laundered as WORLD.

    There is deliberately no "heavy" variant of this call: the
    confirmation tier reruns this same registered script under
    executor-derived seeds instead of asking the model for new code.
    """
    tier = resolve_tier(getattr(diagnosis, "compute_tier", ""))
    prompt = TEMPLATE.format(
        topic=topic,
        claim=card.claim,
        mechanism=card.mechanism,
        alternative=diagnosis.alternative,
        experiment=diagnosis.experiment,
        treatment_arm=diagnosis.treatment_arm,
        control_arm=diagnosis.control_arm,
        allowed_modules=", ".join(sorted(tier.allowed_modules)),
    )
    if skills:
        prompt = skills + prompt
    if prior_failure:
        prompt += IMPLEMENTATION_FAILURE.format(
            status=str(prior_failure.get("status") or "failed"),
            error=str(prior_failure.get("error") or "no detail")[:400],
        )
    if world is not None:
        files = list(getattr(world, "files", ()) or [])
        load_hint = (
            getattr(world, "load_hint", "")
            or f"read Path('data/{files[0] if files else 'instance.json'}')"
        )
        if str(getattr(world, "role", "") or "") == "generated":
            excerpt = ""
            try:
                from .genworld import load_world_payload, world_excerpt

                excerpt = world_excerpt(load_world_payload(world.root))
            except Exception:
                excerpt = ""
            prompt += CONSTRUCTED_BINDING.format(
                files=", ".join(files) or "world.json",
                world_id=getattr(world, "id", "world"),
                title=getattr(world, "title", ""),
                schema=getattr(world, "schema", "") or "symbolic_trace",
                load_hint=load_hint,
                excerpt=excerpt or "(see data/world.json)",
            )
        else:
            prompt += WORLD_BINDING.format(
                files=", ".join(files) or "(see data/)",
                world_id=getattr(world, "id", "world"),
                title=getattr(world, "title", ""),
                schema=getattr(world, "schema", "") or "attested files",
                load_hint=load_hint,
                source=getattr(world, "source", "") or "frozen local fixture",
            )
    cite = str(getattr(diagnosis, "baseline_cite_id", "") or "")
    method = str(getattr(diagnosis, "baseline_method", "") or "")
    if cite:
        prompt += PAPER_CONTROL.format(cite_id=cite, method=method or "the paper's named method")
    refusal: ProbeRefused | None = None
    for attempt in range(retries + 1):
        ask = prompt
        if refusal is not None:
            ask = (
                prompt
                + "\n\nYour previous script was rejected by the sandbox"
                f" compiler: {refusal.record.unlock_condition}."
                " Rewrite the whole script so this cannot happen again,"
                " staying inside the module whitelist and the identifier"
                " rules above."
            )
        completion = client.complete(
            ask,
            purpose=f"research_probe:{card.card_id}"
            + (f":retry{attempt}" if attempt else ""),
            system=SYSTEM,
        )
        completion.assert_usable()
        try:
            spec = spec_from_payload(
                _parse(completion),
                tier=tier,
                claim=card.claim,
                mechanism=card.mechanism,
                topic=topic,
            )
            if world is not None and not reads_world_data(spec.source, world):
                raise _refuse(
                    "read the bound world",
                    "data/ is bound; a probe that invents its own dataset"
                    " is still SYNTHETIC and cannot corroborate",
                )
            return spec
        except ProbeRefused as exc:
            refusal = exc
    assert refusal is not None
    raise refusal


@dataclass(frozen=True)
class ProbeResult:
    status: str
    spec: ProbeSpec
    metrics: dict[str, Any]
    treatment: float | None
    control: float | None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "measure": self.spec.measure,
            "treatment": self.treatment,
            "control": self.control,
            "metrics": self.metrics,
            "error": self.error,
            "source": self.spec.source,
        }


def replication_seeds(
    experiment_digest: str, world_digest: str, count: int
) -> tuple[int, ...]:
    """Seeds for the confirmation tier, derived — never chosen.

    `seed_i = H(experiment_digest ‖ world_digest ‖ i)`. The experiment
    digest covers the script's own bytes, so the model cannot know at
    writing time which seeds its script will be replayed under — a
    `if seed == 17` special case is ruled out by self-reference, and the
    executor has no seed-fishing freedom either: the set is a recorded,
    recomputable fact of the registered evidence object.
    """
    seeds = []
    for index in range(count):
        material = f"{experiment_digest}:{world_digest}:{index}".encode("utf-8")
        seeds.append(int.from_bytes(hashlib.sha256(material).digest()[:4], "big"))
    return tuple(seeds)


def run_probe(
    spec: ProbeSpec,
    work_dir: Path,
    *,
    timeout_seconds: float = 20.0,
    tier: ComputeTier | None = None,
    seed: int | None = None,
) -> ProbeResult:
    """Write, run, and read metrics.json. Isolation is the cwd plus the whitelist.

    A tier with `memory_bytes` caps the child's address space (POSIX);
    the wall clock stays `timeout_seconds` either way. `seed`, when set,
    is written to `data/seed.json` before the run — the runtime-owned
    variation input the script's seed contract reads.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    script = work_dir / "experiment.py"
    script.write_text(spec.source, encoding="utf-8")
    if seed is not None:
        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "seed.json").write_text(
            json.dumps({"seed": int(seed)}), encoding="utf-8"
        )
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR"}
    }
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    preexec = None
    if tier is not None and tier.memory_bytes and os.name == "posix":
        limit = int(tier.memory_bytes)

        def _cap_memory() -> None:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        preexec = _cap_memory
    try:
        completed = subprocess.run(
            [sys.executable, script.name],
            cwd=str(work_dir),
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
            env=env,
            check=False,
            preexec_fn=preexec,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            status="timeout",
            spec=spec,
            metrics={},
            treatment=None,
            control=None,
            error=f"experiment.py did not finish in {timeout_seconds}s",
        )
    if completed.returncode != 0:
        return ProbeResult(
            status="crashed",
            spec=spec,
            metrics={},
            treatment=None,
            control=None,
            error=(completed.stderr or completed.stdout or "nonzero exit")[:400],
        )
    metrics_path = work_dir / "metrics.json"
    if not metrics_path.is_file():
        return ProbeResult(
            status="no_metrics",
            spec=spec,
            metrics={},
            treatment=None,
            control=None,
            error="experiment.py did not write metrics.json",
        )
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return ProbeResult(
            status="bad_metrics",
            spec=spec,
            metrics={},
            treatment=None,
            control=None,
            error=f"metrics.json is not JSON: {error}",
        )
    if not isinstance(metrics, dict):
        return ProbeResult(
            status="bad_metrics",
            spec=spec,
            metrics={},
            treatment=None,
            control=None,
            error="metrics.json must be an object",
        )
    arms: dict[str, float] = {}
    for arm in ("treatment", "control"):
        value = metrics.get(arm)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            return ProbeResult(
                status="bad_metric",
                spec=spec,
                metrics=metrics,
                treatment=None,
                control=None,
                error=f"{arm!r} is not a finite number in metrics.json",
            )
        arms[arm] = float(value)
    return ProbeResult(
        status="ran",
        spec=spec,
        metrics=metrics,
        treatment=arms["treatment"],
        control=arms["control"],
    )
