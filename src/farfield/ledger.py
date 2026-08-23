"""Strict, transactional, hash-chained event storage.

This module provides integrity and process-local serialization. It is not an
authentication boundary against an actor who can rewrite the ledger and trust
root on disk. Runtime semantics are validated by ``FarfieldRuntime``.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, cast

try:  # POSIX deployment target.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


GENESIS_HASH = "0" * 64
Event = dict[str, Any]
EventDraft = tuple[str, str, dict[str, Any]]
Planner = Callable[[tuple[Event, ...]], Sequence[EventDraft]]
Validator = Callable[[tuple[Event, ...]], None]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _event_hash(event_without_hash: Event) -> str:
    return content_digest(event_without_hash)


class LedgerIntegrityError(RuntimeError):
    pass


class EventLedger:
    """Read-mostly ledger with a capability-guarded transactional writer.

    The capability prevents accidental/raw writes through the public runtime API;
    it is not a sandbox against Python reflection or direct filesystem access.
    """

    def __init__(self, path: Path | str, writer_capability: object):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.__writer_capability = writer_capability

    def read(self) -> list[Event]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return self._parse_lines(handle)

    def verify_hash_chain(self) -> tuple[bool, str]:
        try:
            self._verify_events_or_raise(self.read())
        except LedgerIntegrityError as exc:
            return False, str(exc)
        count = len(self.read())
        return True, f"hash chain verified for {count} events"

    def find(self, event_type: str | None = None) -> Iterable[Event]:
        for event in self.read():
            if event_type is None or event["event_type"] == event_type:
                # Return a JSON round-trip copy so callers cannot mutate state in memory.
                yield json.loads(canonical_json(event))

    def _transact(
        self,
        capability: object,
        planner: Planner,
        validator: Validator,
    ) -> list[Event]:
        if capability is not self.__writer_capability:
            raise PermissionError("invalid ledger writer capability")

        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            with os.fdopen(descriptor, "r+", encoding="utf-8", closefd=False) as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.seek(0)
                existing = self._parse_lines(handle)
                self._verify_events_or_raise(existing)
                validator(tuple(existing))

                drafts = list(planner(tuple(existing)))
                if not drafts:
                    raise ValueError("transaction must append at least one event")
                appended = self._materialize(existing, drafts)
                combined = tuple(existing + appended)
                validator(combined)

                handle.seek(0, os.SEEK_END)
                serialized = "".join(canonical_json(event) + "\n" for event in appended)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
                return cast(list[Event], json.loads(canonical_json(appended)))
        except (TypeError, ValueError) as exc:
            if "Out of range float values" in str(exc):
                raise LedgerIntegrityError("non-finite JSON value rejected") from exc
            raise
        finally:
            os.close(descriptor)

    @staticmethod
    def _parse_lines(handle: Any) -> list[Event]:
        events: list[Event] = []
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(
                    raw,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant {value}")
                    ),
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise LedgerIntegrityError(
                    f"invalid strict JSON at ledger line {line_number}"
                ) from exc
            if not isinstance(event, dict):
                raise LedgerIntegrityError(f"event {line_number} is not an object")
            events.append(event)
        return events

    @staticmethod
    def _materialize(existing: list[Event], drafts: list[EventDraft]) -> list[Event]:
        previous_hash = existing[-1]["event_hash"] if existing else GENESIS_HASH
        sequence = len(existing) + 1
        appended: list[Event] = []
        for event_type, actor, payload in drafts:
            if not event_type.strip() or not actor.strip():
                raise ValueError("event_type and actor must be non-empty")
            body: Event = {
                "sequence": sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "actor": actor,
                "payload": payload,
                "prev_hash": previous_hash,
            }
            event = {**body, "event_hash": _event_hash(body)}
            # Strict serialization happens before any bytes are written.
            canonical_json(event)
            appended.append(event)
            previous_hash = event["event_hash"]
            sequence += 1
        return appended

    @staticmethod
    def _verify_events_or_raise(events: list[Event]) -> None:
        expected_previous = GENESIS_HASH
        for index, event in enumerate(events, start=1):
            if event.get("sequence") != index:
                raise LedgerIntegrityError(f"sequence mismatch at event {index}")
            if event.get("prev_hash") != expected_previous:
                raise LedgerIntegrityError(f"previous-hash mismatch at event {index}")
            supplied_hash = event.get("event_hash")
            body = {key: value for key, value in event.items() if key != "event_hash"}
            if supplied_hash != _event_hash(body):
                raise LedgerIntegrityError(f"content-hash mismatch at event {index}")
            expected_previous = str(supplied_hash)
