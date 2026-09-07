"""Exploratory analysis on attested bytes — a first-class, labelled channel.

The most informative numbers of the cwm-iclr2027 line (self-acceptance
carries no information about the oracle; self-written validators
co-occur with more oracle failures) were computed by hand, outside the
system, on the attested world. The system had no way to notice them: a
probe is a pre-registered two-arm test, a brief compiles attested
fields, and nothing in between may look at the data and describe it.

This module is that in-between, kept honest by labelling rather than by
prohibition. An analysis is a stdlib script run on a bound copy of a
catalog world under the host tier, with no network and no probe
semantics. Its output (`analysis.json`) is recorded with the world
digest and the script digest, appended to the catalog chain as
`EXPLORATORY_ANALYSIS`, and rendered to `EXPLORATORY.md` under
`var/exploratory/<world_id>/`. Every record carries
`exploratory: true` and `cannot_corroborate: true`: it is post hoc, not
pre-registered, and it can only *motivate* a hypothesis that a later
two-arm protocol tests. The paper stage reads these records into the
fact sheet under that label and nowhere else.

`builtin_descriptives(schema)` ships the schema's standard tables so the
mission itself can run them on any acquired world.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import chain
from .world import WorldFixture, bind_world, load_fixture

ANALYSIS_TIMEOUT_SECONDS = 600.0


class AnalysisError(ValueError):
    """The analysis did not run or did not write analysis.json."""


PROGRAM_STATE_DESCRIPTIVES = r'''
import json, random, collections
from pathlib import Path

payload = json.loads(Path("data/world.json").read_text(encoding="utf-8"))
U = [u for u in payload["updates"] if isinstance(u, dict)]
oracle = payload.get("oracle") or {}
out = {"n_updates": len(U), "oracle": oracle}

def fail(u):
    return float(u.get("task_return") or 0.0) == 0.0

def rate(rows):
    n = len(rows); k = sum(1 for r in rows if fail(r))
    return {"n": n, "failures": k, "rate": (k / n) if n else None}

acc = [u for u in U if u.get("accepted")]
rej = [u for u in U if not u.get("accepted")]
out["oracle_failure_given_accepted"] = rate(acc)
out["oracle_failure_given_rejected"] = rate(rej)
sv = [u for u in acc if u.get("validator_writes")]
nsv = [u for u in acc if not u.get("validator_writes")]
out["oracle_failure_given_accepted_and_validator_write"] = rate(sv)
out["oracle_failure_given_accepted_and_no_validator_write"] = rate(nsv)
div = [u for u in acc if u.get("divergent")]
ndiv = [u for u in acc if not u.get("divergent")]
out["oracle_failure_given_accepted_and_divergent"] = rate(div)
out["oracle_failure_given_accepted_and_not_divergent"] = rate(ndiv)

episodes = collections.defaultdict(list)
for u in U:
    episodes[str(u.get("episode") or u.get("id"))].append(u)
with_sv = [e for e, rows in episodes.items() if any(r.get("validator_writes") for r in rows)]
without = [e for e, rows in episodes.items() if not any(r.get("validator_writes") for r in rows)]

def ep_fail(e):
    return fail(episodes[e][0])

def ep_rate(eps):
    n = len(eps); k = sum(1 for e in eps if ep_fail(e))
    return {"n": n, "failures": k, "rate": (k / n) if n else None}

out["episode_failure_given_validator_write"] = ep_rate(with_sv)
out["episode_failure_given_no_validator_write"] = ep_rate(without)
if with_sv and without:
    rng = random.Random(0)
    diffs = []
    for _ in range(2000):
        a = [ep_fail(e) for e in rng.choices(with_sv, k=len(with_sv))]
        b = [ep_fail(e) for e in rng.choices(without, k=len(without))]
        diffs.append(sum(a) / len(a) - sum(b) / len(b))
    diffs.sort()
    out["episode_failure_difference_ci95"] = [diffs[50], diffs[1949]]
by_n = collections.defaultdict(list)
for e, rows in episodes.items():
    by_n[min(len(rows), 5)].append(0.0 if ep_fail(e) else 1.0)
out["success_rate_by_updates_per_episode"] = {
    str(k) + ("+" if k == 5 else ""): {"n": len(v), "success_rate": sum(v) / len(v)} for k, v in sorted(by_n.items())
}
out["validator_writes_total"] = sum(1 for u in U if u.get("validator_writes"))
out["accepted_total"] = len(acc)
Path("analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
'''

LABELED_TRACES_DESCRIPTIVES = r'''
import json, collections
from pathlib import Path
payload = json.loads(Path("data/world.json").read_text(encoding="utf-8"))
T = [t for t in payload["traces"] if isinstance(t, dict)]
out = {"n_traces": len(T), "labels": dict(collections.Counter(str(t.get("label")) for t in T))}
by_tools = collections.defaultdict(list)
for t in T:
    by_tools[min(len(t.get("created_tools") or []), 5)].append(1.0 if t.get("label") == "resolved" else 0.0)
out["resolved_rate_by_created_tools"] = {str(k) + ("+" if k == 5 else ""): {"n": len(v), "rate": sum(v) / len(v)} for k, v in sorted(by_tools.items())}
out["mean_steps"] = sum(len(t.get("steps") or []) for t in T) / max(1, len(T))
Path("analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
'''

_BUILTINS: dict[str, str] = {
    "program_state": PROGRAM_STATE_DESCRIPTIVES,
    "labeled_traces": LABELED_TRACES_DESCRIPTIVES,
}


def builtin_descriptives(schema: str) -> str | None:
    return _BUILTINS.get(str(schema or ""))


def exploratory_root(catalog_root: Path) -> Path:
    return Path(catalog_root).parent / "var" / "exploratory"


def run_analysis(
    fixture: WorldFixture,
    source: str,
    *,
    catalog_root: Path,
    label: str = "",
    timeout_seconds: float = ANALYSIS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run `source` on a bound copy of `fixture`; record and return the result."""
    script_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    with tempfile.TemporaryDirectory(prefix="ff-explore-") as tmp:
        work = Path(tmp)
        bind_world(work, fixture)
        (work / "analysis.py").write_text(source, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LC_ALL"}}
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, "analysis.py"],
                cwd=str(work),
                timeout=timeout_seconds,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AnalysisError(f"analysis did not finish in {timeout_seconds:.0f}s") from exc
        if proc.returncode != 0:
            raise AnalysisError((proc.stderr or proc.stdout or "nonzero exit")[:600])
        result_path = work / "analysis.json"
        if not result_path.is_file():
            raise AnalysisError("analysis.py did not write analysis.json")
        try:
            results = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"analysis.json is not JSON: {exc}") from exc
    record = {
        "kind": "EXPLORATORY",
        "exploratory": True,
        "cannot_corroborate": True,
        "label": str(label or "").strip() or "analysis",
        "world_id": fixture.id,
        "world_digest": fixture.digest,
        "world_schema": fixture.schema,
        "world_provenance": fixture.provenance or "published",
        "script_digest": script_digest,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
        "note": (
            "post hoc computation on attested bytes; not pre-registered; can motivate a "
            "hypothesis for a two-arm protocol, never stand as evidence for one"
        ),
    }
    out_dir = exploratory_root(catalog_root) / fixture.id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{script_digest[:16]}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (out_dir / f"{script_digest[:16]}.py").write_text(source, encoding="utf-8")
    chain.append_event(
        Path(catalog_root) / "chain.jsonl",
        chain.EXPLORATORY_ANALYSIS,
        {
            "world_id": fixture.id,
            "world_digest": fixture.digest,
            "script_digest": script_digest,
            "label": record["label"],
        },
    )
    _render_index(out_dir)
    return record


def fact_lines(record: dict[str, Any]) -> list[str]:
    """Compact, labelled sentences for a prompt — the world's own numbers.

    Only descriptives the builtin scripts produce are rendered; a custom
    script's results are shown as a short JSON excerpt. Every line is
    exploratory and says so through the caller's header.
    """
    results = record.get("results") if isinstance(record.get("results"), dict) else {}
    lines: list[str] = []

    def rate(key: str) -> tuple[int, float] | None:
        row = results.get(key)
        if isinstance(row, dict) and row.get("rate") is not None and row.get("n"):
            return int(row["n"]), float(row["rate"])
        return None

    acc, rej = rate("oracle_failure_given_accepted"), rate("oracle_failure_given_rejected")
    if acc and rej:
        lines.append(
            f"oracle failure rate given the gate ACCEPTED an update: {acc[1]:.3f} (n={acc[0]}); "
            f"given it REJECTED: {rej[1]:.3f} (n={rej[0]}) — the gate's acceptance carries "
            f"{'almost no' if abs(acc[1] - rej[1]) < 0.02 else 'some'} information about the oracle"
        )
    sv, nsv = rate("episode_failure_given_validator_write"), rate("episode_failure_given_no_validator_write")
    if sv and nsv:
        ci = results.get("episode_failure_difference_ci95")
        ci_text = (
            f", episode-bootstrap 95% CI of the difference [{float(ci[0]):+.3f}, {float(ci[1]):+.3f}]"
            if isinstance(ci, list) and len(ci) == 2
            else ""
        )
        lines.append(
            f"episodes where the agent rewrote one of its own validators fail the oracle at "
            f"{sv[1]:.3f} (n={sv[0]}) vs {nsv[1]:.3f} (n={nsv[0]}) without{ci_text} — "
            "confounded by task difficulty until matched"
        )
    curve = results.get("success_rate_by_updates_per_episode")
    if isinstance(curve, dict) and curve:
        parts = []
        for key, row in curve.items():
            if isinstance(row, dict) and row.get("success_rate") is not None:
                parts.append(f"{key}: {float(row['success_rate']):.2f} (n={row.get('n')})")
        if parts:
            lines.append("oracle success rate by self-modifications per episode — " + ", ".join(parts))
    div, ndiv = rate("oracle_failure_given_accepted_and_divergent"), rate("oracle_failure_given_accepted_and_not_divergent")
    if div and ndiv:
        lines.append(
            f"among accepted updates, oracle failure is {div[1]:.3f} when the cell diverged from "
            f"its previous version vs {ndiv[1]:.3f} when it did not"
        )
    if not lines and results:
        lines.append("custom analysis: " + json.dumps(results, ensure_ascii=False)[:300])
    return lines


def load_records(catalog_root: Path, world_id: str) -> list[dict[str, Any]]:
    out_dir = exploratory_root(catalog_root) / world_id
    rows: list[dict[str, Any]] = []
    if not out_dir.is_dir():
        return rows
    for path in sorted(out_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict) and row.get("exploratory"):
            rows.append(row)
    return rows


def analyze_world(
    world_id: str,
    *,
    catalog_root: Path,
    source: str | None = None,
    label: str = "",
) -> dict[str, Any]:
    """CLI entry: builtin descriptives when no script is given."""
    directory = Path(catalog_root) / world_id
    if not directory.is_dir():
        raise AnalysisError(f"world {world_id!r} is not in {catalog_root}")
    fixture = load_fixture(directory)
    text = source
    if text is None:
        text = builtin_descriptives(fixture.schema)
        if text is None:
            raise AnalysisError(f"no builtin descriptives for schema {fixture.schema!r}; pass --script")
        label = label or f"descriptives:{fixture.schema}"
    return run_analysis(fixture, text, catalog_root=catalog_root, label=label)


def _render_index(out_dir: Path) -> None:
    lines = [
        f"# Exploratory analyses on `{out_dir.name}`",
        "",
        "Post hoc computations on attested bytes. **Not pre-registered. Cannot corroborate.** "
        "Each row can motivate a hypothesis for a two-arm protocol; none is evidence for one.",
        "",
    ]
    for path in sorted(out_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        lines.append(f"## {row.get('label')} — script `{str(row.get('script_digest'))[:12]}` — {row.get('at')}")
        lines.append("")
        lines.append(f"world digest `{str(row.get('world_digest'))[:12]}`, provenance `{row.get('world_provenance')}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(row.get("results"), ensure_ascii=False, indent=1)[:6000])
        lines.append("```")
        lines.append("")
    (out_dir / "EXPLORATORY.md").write_text("\n".join(lines), encoding="utf-8")
