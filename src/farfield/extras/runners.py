"""Named experiment runners. Built-ins call existing host paths; GPU is a wheel.

The in-loop 20s probe stays the filter. `host` / `host-heavy` reuse
`execute_protocol` — not a second executor. `external` does not invent a
cluster scheduler: skill `external-gpu-run` emits a SkyPilot `sky_task.yaml`
(https://github.com/skypilot-org/skypilot). Launch belongs to `sky` / the
SkyPilot agent skill. A trusted plugin may return treatment/control; those
numbers still pass `attest_external_run`. Do not import AIDE / AI-Scientist
tree search here. Do not open the sandbox to the network by adding a name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hostexp import ExecuteError, attest_external_run, execute_protocol
from .probeexp import COMPUTE_TIERS, resolve_tier


@dataclass(frozen=True)
class RunnerSpec:
    name: str
    kind: str
    compute_tier: str
    can_corroborate: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "compute_tier": self.compute_tier,
            "can_corroborate": self.can_corroborate,
            "description": self.description,
        }


RUNNERS: dict[str, RunnerSpec] = {
    "sandbox": RunnerSpec(
        name="sandbox",
        kind="builtin",
        compute_tier="sandbox",
        can_corroborate=False,
        description="20s whitelist filter; never climbs",
    ),
    "host": RunnerSpec(
        name="host",
        kind="builtin",
        compute_tier="host",
        can_corroborate=True,
        description="stdlib host reproduction on the attested parent",
    ),
    "host-heavy": RunnerSpec(
        name="host-heavy",
        kind="builtin",
        compute_tier="host-heavy",
        can_corroborate=True,
        description="numpy + 3600s + memory cap; still no network",
    ),
    "external": RunnerSpec(
        name="external",
        kind="plugin",
        compute_tier="host-heavy",
        can_corroborate=False,
        description=(
            "GPU / cluster / Slurm via SkyPilot YAML + plugin, not a FarField "
            "scheduler. Receipt then attest-run; cannot corroborate"
        ),
    ),
}

AWAITING_METRICS = (
    "No runner produced treatment/control. Launch sky_task.yaml with the "
    "SkyPilot skill (`sky jobs launch`), or any lab runner that executes "
    "the same experiment.py, then `farfield attest-run --metrics`. "
    "externally_replicated is not corroborated."
)


def resolve_runner(name: str | None, *, compute_tier: str = "") -> RunnerSpec:
    text = str(name or "").strip().lower()
    if text in RUNNERS:
        return RUNNERS[text]
    if text:
        raise ExecuteError(
            f"unknown runner {text!r}; use {sorted(RUNNERS)} or a plugin hook"
        )
    tier = resolve_tier(compute_tier)
    if tier.name in RUNNERS:
        return RUNNERS[tier.name]
    return RUNNERS["host"]


def runner_from_protocol(protocol: dict[str, Any]) -> RunnerSpec:
    host = protocol.get("host") if isinstance(protocol.get("host"), dict) else {}
    diag = protocol.get("diagnosis") if isinstance(protocol.get("diagnosis"), dict) else {}
    named = str(host.get("runner") or diag.get("runner") or "").strip()
    tier = str(diag.get("compute_tier") or host.get("compute_tier") or "")
    return resolve_runner(named, compute_tier=tier)


def dispatch_run(
    folder: Path,
    *,
    runner: str | None = None,
    catalog_root: Path | None = None,
    timeout_seconds: float | None = None,
    prefer_parent: bool = True,
    plugin_host: Any = None,
) -> dict[str, Any]:
    """Run one compiled protocol through a named runner.

    `host` / `host-heavy` call `execute_protocol`. `external` never
    pretends to have GPU results: either a plugin returns two arm
    numbers (judged by attest-run) or a receipt tells the operator
    what to submit later.
    """
    folder = Path(folder)
    protocol_path = folder / "protocol.json"
    if not protocol_path.is_file():
        raise ExecuteError(f"missing protocol.json in {folder}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ExecuteError("protocol.json must be an object")
    spec = (
        resolve_runner(runner)
        if runner
        else runner_from_protocol(protocol)
    )
    if spec.name in {"host", "host-heavy"}:
        record = execute_protocol(
            folder,
            catalog_root=catalog_root,
            timeout_seconds=timeout_seconds,
            prefer_parent=prefer_parent,
        )
        record["runner"] = spec.name
        record["can_corroborate"] = spec.can_corroborate
        return record
    if spec.name == "sandbox":
        raise ExecuteError(
            "sandbox is the in-loop filter; it is not a host runner. "
            "Use host, host-heavy, or external"
        )
    return _run_external(folder, spec, plugin_host=plugin_host)


def _run_external(
    folder: Path, spec: RunnerSpec, *, plugin_host: Any
) -> dict[str, Any]:
    offered = None
    if plugin_host is not None and hasattr(plugin_host, "offer"):
        offered = plugin_host.offer("run_external", folder=folder, spec=spec.to_dict())
    if isinstance(offered, dict) and "treatment" in offered and "control" in offered:
        metrics_path = folder / "external_metrics.json"
        payload = {
            "treatment": offered["treatment"],
            "control": offered["control"],
        }
        metrics_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        record = attest_external_run(
            folder,
            metrics_path=metrics_path,
            environment=str(offered.get("environment") or spec.name),
            runner=str(offered.get("runner") or spec.name),
        )
        record["runner"] = spec.name
        record["can_corroborate"] = False
        record["plugin"] = True
        return record
    receipt = {
        "ok": False,
        "stage": "external_receipt",
        "kind": "EXTERNAL",
        "runner": spec.name,
        "status": "awaiting_metrics",
        "can_corroborate": False,
        "next": (
            "sky jobs launch sky_task.yaml (SkyPilot skill), or run "
            "experiment.py on the lab box; write {\"treatment\": <num>, "
            "\"control\": <num>}, then farfield attest-run --metrics"
        ),
        "honesty": AWAITING_METRICS,
        "compute_tier": spec.compute_tier,
        "timeout_hint_seconds": COMPUTE_TIERS[spec.compute_tier].timeout_seconds,
    }
    (folder / "runner_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt
