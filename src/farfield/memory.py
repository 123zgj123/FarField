"""M: append-only long-term memory index. Capsules, not a trained updater.

DESIGN §2.4 defines `M` as failure capsules **and** an `attested` citation index.
Both kinds live in the same append-only file under a `kind` field, because the
`dump-M` control arm is handed everything `M` knows and a citation index kept
somewhere else would quietly weaken the arm it has to beat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .ledger import content_digest

INDEX_PATH = Path(".farfield") / "memory_index.json"

CAPSULE_KIND = "failure_capsule"
CITATION_KIND = "attested_citation"


def load_index(project_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(project_dir) / INDEX_PATH
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("memory index must be a list of capsules")
    return payload


def append_capsules(project_dir: Path | str, capsules: list[dict[str, Any]]) -> str:
    rows = load_index(project_dir)
    for capsule in capsules:
        if not str(capsule.get("scope") or "").strip():
            raise ValueError("memory capsule requires a non-empty scope")
        row = dict(capsule)
        row.setdefault("kind", CAPSULE_KIND)
        rows.append(row)
    return _write(project_dir, rows)


def append_citations(
    project_dir: Path | str, spans: Sequence[Any], *, campaign_id: str
) -> str:
    """Index every attested span this campaign bought.

    Deduplicated on (source_id, start, end): re-reading the same span in a later
    campaign must not grow `M`, or index growth becomes a metric that rises
    without anything being learned.
    """
    rows = load_index(project_dir)
    seen = {
        (row.get("source_id"), row.get("start"), row.get("end"))
        for row in rows
        if row.get("kind") == CITATION_KIND
    }
    for span in spans:
        key = (span.source_id, span.start, span.end)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "kind": CITATION_KIND,
                "scope": "literature",
                "source_id": span.source_id,
                "title": span.title,
                "artifact_uri": span.artifact_uri,
                "start": span.start,
                "end": span.end,
                "text": span.text,
                "first_seen_campaign": campaign_id,
            }
        )
    return _write(project_dir, rows)


def citations(project_dir: Path | str) -> list[dict[str, Any]]:
    return [
        row for row in load_index(project_dir) if row.get("kind") == CITATION_KIND
    ]


def capsules(project_dir: Path | str) -> list[dict[str, Any]]:
    return [
        row
        for row in load_index(project_dir)
        if row.get("kind", CAPSULE_KIND) == CAPSULE_KIND
    ]


def _write(project_dir: Path | str, rows: list[dict[str, Any]]) -> str:
    path = Path(project_dir) / INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_digest(project_dir)


def index_digest(project_dir: Path | str) -> str:
    rows = load_index(project_dir)
    if not rows:
        return "none"
    return content_digest(rows)
