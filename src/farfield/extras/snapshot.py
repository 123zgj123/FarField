"""Mission-level evidence snapshot: live retrieval, frozen per mission.

Research needs today's literature; reproduction needs yesterday's bytes.
The conflict dissolves if every live answer a mission consumes is written
down the moment it arrives: `RecordingFeed` wraps any feed and appends
each call — method, arguments, answer or block — to
`evidence_snapshot/feed_log.jsonl` in the mission workspace. Every later
judgment inside the mission (prior overlap, read_first, freshness) is a
function of exactly those recorded bytes.

`ReplayFeed` reads such a log back and answers the same calls in the same
order without a socket, so a finished mission's literature-facing verdicts
can be re-derived. A call the snapshot never saw is a `FeedBlocked`, not a
silent fresh fetch — replay must not quietly become live.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .livefeed import FeedBlocked, FreshWork

LOG_NAME = "feed_log.jsonl"
SNAPSHOT_DIR = "evidence_snapshot"


def _work(row: dict[str, Any]) -> FreshWork:
    return FreshWork(
        title=str(row.get("title") or ""),
        published=str(row.get("published") or ""),
        arxiv_id=str(row.get("arxiv_id") or ""),
        abstract=str(row.get("abstract") or ""),
        source=str(row.get("source") or "arxiv"),
        work_id=str(row.get("work_id") or ""),
        url=str(row.get("url") or ""),
        venue=str(row.get("venue") or ""),
    )


def _serialize(result: Any) -> dict[str, Any]:
    if isinstance(result, FeedBlocked):
        return {"blocked": result.to_dict()}
    if isinstance(result, list):
        return {"works": [work.to_dict() for work in result]}
    return {"payload": result}


def _deserialize(row: dict[str, Any]) -> Any:
    if "blocked" in row:
        blocked = row["blocked"]
        return FeedBlocked(
            attempted=str(blocked.get("attempted") or ""),
            reason=str(blocked.get("reason") or ""),
        )
    if "works" in row:
        return [_work(item) for item in row["works"]]
    return row.get("payload")


class RecordingFeed:
    """Every answer the inner feed gives is appended to the snapshot log."""

    def __init__(self, inner: Any, snapshot_dir: Path) -> None:
        self.inner = inner
        self.dir = Path(snapshot_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = self.dir / LOG_NAME
        self._lock = threading.Lock()

    def _record(self, method: str, args: dict[str, Any], result: Any) -> Any:
        entry = {
            "method": method,
            "args": args,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        entry.update(_serialize(result))
        with self._lock:
            with self.log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return result

    def recent_in_field(self, concepts, *, max_results: int = 6):
        result = self.inner.recent_in_field(concepts, max_results=max_results)
        return self._record(
            "recent_in_field", {"concepts": list(concepts)}, result
        )

    def pair_recently_combined(self, concept_a, concept_b, *, max_results: int = 5):
        result = self.inner.pair_recently_combined(
            concept_a, concept_b, max_results=max_results
        )
        return self._record(
            "pair_recently_combined", {"pair": [concept_a, concept_b]}, result
        )

    def survey_around(
        self, concept_a, concept_b, *, per_side: int = 4, claim: str = "", topic: str = ""
    ):
        kwargs: dict[str, Any] = {"per_side": per_side, "claim": claim}
        if topic:
            kwargs["topic"] = topic
        try:
            result = self.inner.survey_around(concept_a, concept_b, **kwargs)
        except TypeError:
            result = self.inner.survey_around(concept_a, concept_b, per_side=per_side)
        args: dict[str, Any] = {"pair": [concept_a, concept_b], "claim": claim}
        if topic:
            args["topic"] = topic
        return self._record("survey_around", args, result)


class ReplayFeed:
    """Answers from a recorded snapshot, keyed by method and args.

    Parallel surveys append in completion order, not call order, so replay
    must match the request (pair, concepts) rather than consume the log as
    a FIFO of method names.
    """

    def __init__(self, snapshot_dir: Path) -> None:
        self.entries: list[dict[str, Any]] = []
        log = Path(snapshot_dir) / LOG_NAME
        if log.is_file():
            for line in log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.entries.append(json.loads(line))
        self.used: set[int] = set()

    def _find(self, method: str, args: dict[str, Any] | None = None) -> Any:
        for index, entry in enumerate(self.entries):
            if index in self.used:
                continue
            if entry.get("method") != method:
                continue
            recorded = entry.get("args") or {}
            if args:
                matched = True
                for key, value in args.items():
                    if recorded.get(key) != value:
                        matched = False
                        break
                if not matched:
                    continue
            self.used.add(index)
            return _deserialize(entry)
        return FeedBlocked(
            attempted=method,
            reason="not in the mission's evidence snapshot; replay does not go live",
        )

    def recent_in_field(self, concepts, *, max_results: int = 6):
        return self._find("recent_in_field", {"concepts": list(concepts)})

    def pair_recently_combined(self, concept_a, concept_b, *, max_results: int = 5):
        return self._find(
            "pair_recently_combined", {"pair": [concept_a, concept_b]}
        )

    def survey_around(
        self, concept_a, concept_b, *, per_side: int = 4, claim: str = "", topic: str = ""
    ):
        requested: dict[str, Any] = {"pair": [concept_a, concept_b]}
        if claim:
            requested["claim"] = claim
        if topic:
            requested["topic"] = topic
        return self._find("survey_around", requested)
