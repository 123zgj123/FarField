"""Host execution of a compiled protocol.json.

This is the missing colleague: the 20s sandbox stays the filter; this
process runs the same pre-registered experiment.py against the attested
parent (same source_digest) with a longer timeout. It still uses the
stdlib whitelist, still has no network, still judges by arithmetic, and
still cannot invent data after seeing the claim.

A successful host run is `protocol_executed`. That is not a discovery
and it does not rename the ladder.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from . import chain
from .evidence import (
    TIER_SAME_INSTANCE,
    TIER_SCALE,
    evidence_id,
    replication_group_id,
    sha256_text,
)
from .diagnose import Diagnosis, judge_probe
from .freeze import FreezeError, reconstruct_parent
from .probeexp import (
    COMPUTE_TIERS,
    ProbeRefused,
    ProbeSpec,
    resolve_tier,
    run_probe,
    validate_source,
)
from .world import bind_world, load_catalog, reads_world_data


HOST_TIMEOUT_SECONDS = COMPUTE_TIERS["host"].timeout_seconds
PROTOCOL_EXECUTED = (
    "The host ran the pre-registered protocol on an attested trace. "
    "That is protocol_executed, not a scientific discovery."
)


class ExecuteError(ValueError):
    """Host run refused. The catalog and the claim ladder are unchanged."""


def execute_protocol(
    folder: Path,
    *,
    catalog_root: Path | None = None,
    timeout_seconds: float | None = None,
    prefer_parent: bool = True,
) -> dict[str, Any]:
    """Run `folder/experiment.py` against the protocol's world.

    `prefer_parent` rebuilds the unsliced mother trace from the freeze
    cache. Missing cache is blocked rather than silently shrinking to the
    in-loop slice. The timeout and module whitelist come from the
    protocol's pre-registered `compute_tier` unless the caller passes an
    explicit `timeout_seconds`.

    Replication semantics: a run on the same slice the probe saw is an
    E1 same-instance reproduction — the only kind that can confirm the
    probe's EvidenceID. A run on the reconstructed parent reads
    different bytes, so it is an E2 scale replication with its *own*
    EvidenceID: valuable generalization evidence, never "the same
    experiment again".
    """
    folder = Path(folder)
    protocol_path = folder / "protocol.json"
    script_path = folder / "experiment.py"
    if not protocol_path.is_file():
        raise ExecuteError(f"missing protocol.json in {folder}")
    if not script_path.is_file():
        raise ExecuteError(f"missing experiment.py in {folder}")
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExecuteError(f"protocol.json is not JSON: {exc}") from exc
    if not isinstance(protocol, dict):
        raise ExecuteError("protocol.json must be an object")
    diag_meta = protocol.get("diagnosis") if isinstance(protocol.get("diagnosis"), dict) else {}
    tier = resolve_tier(str((diag_meta or {}).get("compute_tier") or ""))
    if timeout_seconds is None:
        timeout_seconds = tier.timeout_seconds
    source = script_path.read_text(encoding="utf-8")
    try:
        validate_source(source, tier=tier)
    except ProbeRefused as exc:
        raise ExecuteError(f"experiment.py failed the sandbox whitelist: {exc}") from exc
    spec = ProbeSpec(
        measure=str((protocol.get("cheap_probe") or {}).get("measure") or "host measure"),
        source=source,
    )
    world_meta = protocol.get("world") or {}
    world_id = str(world_meta.get("id") or "").strip()
    if not world_id:
        raise ExecuteError(
            "protocol.json has no world id; a host run without an attested "
            "trace would be another SYNTHETIC coherence check"
        )
    repo = Path(catalog_root) if catalog_root is not None else _repo_root()
    catalog = load_catalog(repo)
    if world_id not in catalog:
        raise ExecuteError(f"world {world_id!r} is not in worlds/")
    fixture = catalog[world_id]
    recorded_digest = str(world_meta.get("source_digest") or "").strip()
    if recorded_digest and fixture.source_digest and recorded_digest != fixture.source_digest:
        raise ExecuteError(
            f"protocol source_digest {recorded_digest[:12]}… does not match "
            f"catalog {fixture.source_digest[:12]}…; the mother trace changed"
        )
    bound = "slice"
    with tempfile.TemporaryDirectory(prefix="ffhost-") as tmp:
        work = Path(tmp)
        if prefer_parent:
            try:
                reconstruct_parent(fixture, work)
                bound = "parent"
            except FreezeError as exc:
                raise ExecuteError(str(exc)) from exc
        else:
            bind_world(work, fixture)
        if not reads_world_data(source):
            raise ExecuteError(
                "experiment.py does not name data/; a host run that invents "
                "its own integers is still SYNTHETIC"
            )
        experiment_digest = sha256_text(source)
        recorded_exp = str(protocol.get("experiment_digest") or "").strip()
        if recorded_exp and recorded_exp != experiment_digest:
            raise ExecuteError(
                "experiment.py digest does not match the probe; "
                "host cannot confirm a different experiment"
            )
        # Honest data identity: the digest of the bytes this run actually
        # read, never the slice digest standing in for the parent.
        if bound == "parent":
            bound_digest = fixture.source_digest or ""
            if not bound_digest:
                raise ExecuteError(
                    "parent run has no recorded source_digest; a run whose "
                    "bytes cannot be identified cannot become evidence"
                )
        else:
            bound_digest = fixture.digest
        replication = (
            TIER_SAME_INSTANCE if bound_digest == fixture.digest else TIER_SCALE
        )
        host_eid = evidence_id(
            claim_id=str(protocol.get("card_id") or ""),
            experiment_digest=experiment_digest,
            world_digest=fixture.digest,
            data_digest=bound_digest,
        )
        group_id = replication_group_id(
            claim_id=str(protocol.get("card_id") or ""),
            experiment_digest=experiment_digest,
            source_digest=fixture.source_digest or fixture.digest,
        )
        result = run_probe(spec, work, timeout_seconds=timeout_seconds, tier=tier)
        diagnosis = _diagnosis_from_protocol(protocol)
        verdict = None
        if (
            result.status == "ran"
            and diagnosis is not None
            and result.treatment is not None
            and result.control is not None
        ):
            verdict = judge_probe(diagnosis, result.treatment, result.control)["verdict"]
        record = {
            "ok": result.status == "ran",
            "stage": "host_execute",
            "kind": "HOST",
            "bound": bound,
            "world_id": fixture.id,
            "source_digest": fixture.source_digest,
            "schema": fixture.schema,
            "timeout_seconds": timeout_seconds,
            "compute_tier": tier.name,
            "status": result.status,
            "measure": spec.measure,
            "treatment": result.treatment,
            "control": result.control,
            "verdict": verdict,
            "error": result.error,
            "honesty": PROTOCOL_EXECUTED,
            "protocol_complete": result.status == "ran",
            "experiment_digest": experiment_digest,
            "world_digest": fixture.digest,
            "data_digest": bound_digest,
            "evidence_id": host_eid,
            "replication": replication,
            "replication_group_id": group_id,
            "chain_path": str(folder / "chain.jsonl"),
            "ladder": (
                "protocol_executed is not corroborated, not verified, and "
                "not a scientific discovery"
            ),
        }
        if replication == TIER_SCALE:
            record["ladder"] = (
                "scale replication: the parent bytes are a different evidence "
                "object with its own EvidenceID; it generalizes the probe, it "
                "does not reproduce it, and it cannot confirm the probe's claim"
            )
        (folder / "host_run.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        host_dir = folder / "host"
        host_dir.mkdir(parents=True, exist_ok=True)
        (host_dir / "host_run.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if result.metrics:
            (folder / "host_metrics.json").write_text(
                json.dumps(result.metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        chain.append_event(
            folder / "chain.jsonl",
            chain.EXECUTE_PROTOCOL,
            {
                "card_id": str(protocol.get("card_id") or ""),
                "evidence_id": host_eid,
                "replication": replication,
                "replication_group_id": group_id,
                "bound": bound,
                "world_id": fixture.id,
                "world_digest": fixture.digest,
                "data_digest": bound_digest,
                "experiment_digest": experiment_digest,
                "status": result.status,
                "verdict": verdict,
            },
        )
        return record


EXTERNAL_REPLICATION = (
    "An external environment ran the pre-registered protocol and submitted "
    "two arm numbers. The verdict is computed from the registered direction "
    "and margin — never claimed by the submitter. The environment is not an "
    "attested world, so this is externally_replicated: it does not become "
    "corroborated and it does not rename the ladder."
)


def attest_external_run(
    folder: Path,
    *,
    metrics_path: Path,
    environment: str = "",
    runner: str = "",
) -> dict[str, Any]:
    """Validate and record an external replication of a compiled protocol.

    This is the exit for experiments the sandbox and host tiers cannot run
    (GPU training, real clusters). The researcher runs the protocol's
    experiment in their own environment and submits `treatment` and
    `control`; this command applies the pre-registered arithmetic and
    writes `external_run.json` next to the protocol. Submitting a verdict
    is impossible — only the two numbers count.
    """
    folder = Path(folder)
    protocol_path = folder / "protocol.json"
    if not protocol_path.is_file():
        raise ExecuteError(f"missing protocol.json in {folder}")
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExecuteError(f"protocol.json is not JSON: {exc}") from exc
    if not isinstance(protocol, dict):
        raise ExecuteError("protocol.json must be an object")
    diagnosis = _diagnosis_from_protocol(protocol)
    if diagnosis is None:
        raise ExecuteError(
            "protocol.json has no registered diagnosis; without a "
            "pre-registered direction and margin there is nothing an "
            "external run could be judged against"
        )
    metrics_path = Path(metrics_path)
    if not metrics_path.is_file():
        raise ExecuteError(f"missing metrics file: {metrics_path}")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExecuteError(f"metrics file is not JSON: {exc}") from exc
    if not isinstance(metrics, dict):
        raise ExecuteError("metrics file must be an object")
    if "verdict" in metrics:
        raise ExecuteError(
            "the metrics file must not contain a verdict; submit the two "
            "arm numbers and let the registered arithmetic decide"
        )
    arms: dict[str, float] = {}
    for arm in ("treatment", "control"):
        value = metrics.get(arm)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ExecuteError(f"metrics must contain a finite number for {arm!r}")
        arms[arm] = float(value)
    judged = judge_probe(diagnosis, arms["treatment"], arms["control"])
    record = {
        "ok": True,
        "stage": "external_replication",
        "kind": "EXTERNAL",
        "card_id": str(protocol.get("card_id") or ""),
        "verdict": judged["verdict"],
        "treatment": arms["treatment"],
        "control": arms["control"],
        "separation": judged["separation"],
        "margin": judged["margin"],
        "expected_direction": judged["expected_direction"],
        "environment": str(environment or "").strip(),
        "runner": str(runner or "").strip(),
        "metrics_digest": sha256_text(
            json.dumps(metrics, ensure_ascii=False, sort_keys=True)
        ),
        "protocol_digest": sha256_text(
            protocol_path.read_text(encoding="utf-8")
        ),
        "status": "externally_replicated",
        "honesty": EXTERNAL_REPLICATION,
        "ladder": (
            "externally_replicated is not corroborated: the environment is "
            "not an attested world and the bytes were not frozen before the "
            "claim"
        ),
    }
    (folder / "external_run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    chain.append_event(
        folder / "chain.jsonl",
        chain.EXTERNAL_REPLICATION,
        {
            "card_id": record["card_id"],
            "verdict": record["verdict"],
            "environment": record["environment"],
            "runner": record["runner"],
            "metrics_digest": record["metrics_digest"],
            "protocol_digest": record["protocol_digest"],
            "replication": "E4_external",
        },
    )
    return record


def _diagnosis_from_protocol(protocol: dict[str, Any]) -> Diagnosis | None:
    diag = protocol.get("diagnosis")
    if not isinstance(diag, dict):
        return None
    try:
        return Diagnosis(
            card_id=str(protocol.get("card_id") or ""),
            alternative=str(diag.get("alternative") or "n/a"),
            experiment=str(diag.get("experiment") or ""),
            treatment_arm=str(diag.get("treatment_arm") or "treatment"),
            control_arm=str(diag.get("control_arm") or "control"),
            expected_direction=str(diag.get("expected_direction") or ""),
            alternative_direction=str(diag.get("alternative_direction") or ""),
            expected_if_alternative=str(diag.get("expected_if_alternative") or ""),
            margin=float(diag.get("margin") or 0.05),
            margin_reason=str(diag.get("margin_reason") or ""),
        )
    except (TypeError, ValueError):
        return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
