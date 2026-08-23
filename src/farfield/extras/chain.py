"""Append-only, digest-chained epistemic event log.

Freezing bytes proves *what* was frozen; it does not prove *when*
relative to the hypothesis. This log gives every epistemic act —
freezing a world, registering a protocol, executing it, promoting a
claim — a sequence number and a digest that commits to every event
before it. An event cannot be inserted, reordered, or rewritten after
the fact without breaking every later digest.

The chain records acts, and promotion consumes it: `audit_promotion_gaps`
is the gate `state.apply_outcomes` runs before it may write `corroborated`
or `verified` — integrity first, then the order proof that the protocol
was registered before the execution that claims to confirm it. Promotion
*rules* still live in `evidence.py` / `state.py`; this module only answers
"is the recorded history intact and in the right order". One JSONL file
per scope: a worlds catalog, a candidate folder, a research-state store.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64

FREEZE_WORLD = "FREEZE_WORLD"
REGISTER_HYPOTHESIS = "REGISTER_HYPOTHESIS"
REGISTER_PROTOCOL = "REGISTER_PROTOCOL"
EXECUTE_PROTOCOL = "EXECUTE_PROTOCOL"
EXECUTE_HEAVY_SEED = "EXECUTE_HEAVY_SEED"
AGGREGATE_HEAVY = "AGGREGATE_HEAVY"
# A runtime-owned forward simulation of the bound world (dynworld).
# Scout evidence: recorded so an audit can see what the designer saw
# before registering, but no promotion gate ever counts it.
SIMULATE_WORLD = "SIMULATE_WORLD"
# Executor-owned placebo: the registered script rerun on a
# structure-destroyed copy of the bound world. Scout of consumption,
# required before a WORLD supports may corroborate.
EXECUTE_PLACEBO = "EXECUTE_PLACEBO"
EXTERNAL_REPLICATION = "EXTERNAL_REPLICATION"
PROMOTE = "PROMOTE"


class ChainError(Exception):
    """An event log failed its audit: broken digests or a tampered order."""


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def _event_digest(
    seq: int, at: str, kind: str, payload: dict[str, Any], prev: str
) -> str:
    body = _canonical(
        {"seq": seq, "at": at, "kind": kind, "payload": payload, "prev": prev}
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def append_event(
    path: Path, kind: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Append one event under the file lock; returns the recorded row."""
    from .filelock import file_lock

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        events = read_events(path)
        prev = str(events[-1]["digest"]) if events else GENESIS
        seq = len(events)
        at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record: dict[str, Any] = {
            "seq": seq,
            "at": at,
            "kind": str(kind),
            "payload": dict(payload or {}),
            "prev": prev,
        }
        record["digest"] = _event_digest(
            seq, at, record["kind"], record["payload"], prev
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    return record


def read_events(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def verify_chain(path: Path) -> dict[str, Any]:
    """Walk the log; any edit, drop, or reorder breaks a digest."""
    events = read_events(path)
    prev = GENESIS
    for index, event in enumerate(events):
        if int(event.get("seq", -1)) != index:
            return {
                "ok": False,
                "length": index,
                "error": f"event {index}: sequence broken",
            }
        if str(event.get("prev") or "") != prev:
            return {
                "ok": False,
                "length": index,
                "error": f"event {index}: prev digest broken",
            }
        expected = _event_digest(
            index,
            str(event.get("at") or ""),
            str(event.get("kind") or ""),
            dict(event.get("payload") or {}),
            prev,
        )
        if str(event.get("digest") or "") != expected:
            return {
                "ok": False,
                "length": index,
                "error": f"event {index}: content digest broken",
            }
        prev = expected
    return {"ok": True, "length": len(events), "error": None}


def audit_promotion_gaps(
    path: Path,
    *,
    experiment_digest: str = "",
    evidence_id: str = "",
) -> tuple[str, ...]:
    """Every reason this chain cannot license a promotion. Empty means it can.

    Two checks, both fail-closed. Integrity: any edited, dropped, or
    reordered event breaks a digest and the whole chain is refused.
    Order: a REGISTER_PROTOCOL event (for `experiment_digest`, when
    given) must precede the EXECUTE_PROTOCOL event (for `evidence_id`,
    when given) — a protocol registered after its confirming run proves
    nothing about preregistration.
    """
    path = Path(path)
    if not path.is_file():
        return ("no event chain recorded for this evidence",)
    report = verify_chain(path)
    if not report["ok"]:
        return (f"event chain broken: {report['error']}",)
    register_seq: int | None = None
    execute_seq: int | None = None
    for event in read_events(path):
        kind = str(event.get("kind") or "")
        payload = event.get("payload") or {}
        if kind == REGISTER_PROTOCOL and register_seq is None:
            if (
                not experiment_digest
                or str(payload.get("experiment_digest") or "") == experiment_digest
            ):
                register_seq = int(event["seq"])
        elif kind == EXECUTE_PROTOCOL and execute_seq is None:
            if (
                not evidence_id
                or str(payload.get("evidence_id") or "") == evidence_id
            ):
                execute_seq = int(event["seq"])
    gaps: list[str] = []
    if register_seq is None:
        gaps.append("the protocol was never registered in the event chain")
    if execute_seq is None:
        gaps.append("the confirming execution is not in the event chain")
    if register_seq is not None and execute_seq is not None:
        if register_seq > execute_seq:
            gaps.append(
                "the protocol was registered after the execution it claims"
                " to pre-register"
            )
    return tuple(gaps)


def find_event(
    path: Path, kind: str, **payload_match: str
) -> dict[str, Any] | None:
    """The first intact event of `kind` whose payload matches every given
    key exactly, or None. Integrity is a precondition: a broken chain has
    no findable events."""
    path = Path(path)
    if not path.is_file() or not verify_chain(path)["ok"]:
        return None
    for event in read_events(path):
        if str(event.get("kind") or "") != kind:
            continue
        payload = event.get("payload") or {}
        if all(
            str(payload.get(key) or "") == str(value)
            for key, value in payload_match.items()
        ):
            return event
    return None


def audit_heavy_gaps(
    path: Path,
    *,
    evidence_id: str,
    experiment_digest: str = "",
    min_seeds: int = 3,
) -> tuple[str, ...]:
    """Every reason the ledger cannot prove the heavy confirmation happened.

    `verified` may not rest on a dict that says a heavy run occurred; the
    ledger must contain the acts themselves: an AGGREGATE_HEAVY event for
    this aggregate EvidenceID whose `parents` reference at least
    `min_seeds` EXECUTE_HEAVY_SEED events recorded *earlier in the same
    chain*, each rerunning the registered experiment digest under a
    distinct seed. The aggregate is bound to its executions by digest
    reference — an aggregate that cannot name its runs proves nothing.
    """
    path = Path(path)
    if not path.is_file():
        return ("no event chain recorded for the heavy confirmation",)
    report = verify_chain(path)
    if not report["ok"]:
        return (f"event chain broken: {report['error']}",)
    events = read_events(path)
    aggregate = None
    for event in events:
        if str(event.get("kind") or "") != AGGREGATE_HEAVY:
            continue
        payload = event.get("payload") or {}
        if str(payload.get("evidence_id") or "") == str(evidence_id):
            aggregate = event
            break
    if aggregate is None:
        return ("the heavy aggregation is not in the event chain",)
    parents = [str(item) for item in (aggregate["payload"].get("parents") or [])]
    gaps: list[str] = []
    if len(parents) < min_seeds:
        gaps.append(
            f"the aggregate references {len(parents)} seed executions;"
            f" at least {min_seeds} are required"
        )
    by_digest = {
        str(event.get("digest") or ""): event
        for event in events
        if str(event.get("kind") or "") == EXECUTE_HEAVY_SEED
    }
    seeds_seen: set[str] = set()
    for digest in parents:
        execution = by_digest.get(digest)
        if execution is None:
            gaps.append(
                "the aggregate references a seed execution that is not in"
                " the event chain"
            )
            continue
        if int(execution.get("seq", -1)) >= int(aggregate.get("seq", -1)):
            gaps.append(
                "a referenced seed execution does not precede the aggregate"
            )
        payload = execution.get("payload") or {}
        if experiment_digest and (
            str(payload.get("experiment_digest") or "") != experiment_digest
        ):
            gaps.append(
                "a seed execution reran a different experiment than the"
                " registered protocol"
            )
        seeds_seen.add(str(payload.get("seed")))
    if not gaps and len(seeds_seen) < min_seeds:
        gaps.append("the referenced seed executions do not carry distinct seeds")
    return tuple(gaps)
