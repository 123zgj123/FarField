"""A mission's working folder: the thing Robin keeps and Farfield used not to.

One directory per run. Papers, briefs, probe source, metrics, the compiled
note, and the executable protocol live as ordinary files a researcher can
open after the browser tab is gone. Nothing here is evidence the kernel
would accept; it is the lab notebook plus a next-week experiment pack.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def var_dir(root: Path) -> Path:
    """Gitignored runtime: state, cache, missions, distilled skills."""
    return Path(root) / "var"


MISSION_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z0-9-]+$")
EVENTS_FILE = "mission.ndjson"


def mission_dir(root: Path, topic: str, *, when: datetime | None = None) -> Path:
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:48] or "topic"
    return var_dir(root) / "missions" / f"{stamp}-{slug}"


def resolve_recorded_mission(root: Path, mission_id: str) -> Path | None:
    """A mission folder under var/missions/, or None if the id is not local."""
    if not MISSION_ID.match(str(mission_id or "")):
        return None
    base = (var_dir(root) / "missions").resolve()
    dest = (base / mission_id).resolve()
    try:
        dest.relative_to(base)
    except ValueError:
        return None
    if not dest.is_dir():
        return None
    return dest


def append_recorded_event(path: Path, event: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def iter_recorded_events(folder: Path) -> list[dict[str, Any]]:
    path = Path(folder) / EVENTS_FILE
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("stage"):
            events.append(row)
    return events


def list_recorded_missions(root: Path) -> list[dict[str, Any]]:
    """Past runs the console can replay. Newest first."""
    base = var_dir(root) / "missions"
    if not base.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for dest in sorted(base.iterdir(), reverse=True):
        if not dest.is_dir() or not MISSION_ID.match(dest.name):
            continue
        topic = dest.name
        topic_path = dest / "topic.txt"
        if topic_path.is_file():
            topic = topic_path.read_text(encoding="utf-8").strip() or topic
        summary: dict[str, Any] = {}
        summary_path = dest / "summary.json"
        if summary_path.is_file():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                summary = loaded
        events = dest / EVENTS_FILE
        rows.append(
            {
                "id": dest.name,
                "topic": topic,
                "replayable": events.is_file() and events.stat().st_size > 0,
                "has_packet": (dest / "RESEARCH_PACKET.md").is_file(),
                "cards": summary.get("cards"),
                "entered": summary.get("entered"),
                "briefs": summary.get("briefs"),
                "corroborated": summary.get("corroborated"),
                "jumps_opened": summary.get("jumps_opened"),
                "workspace": str(dest),
            }
        )
    return rows


def llm_cache_dir(root: Path) -> Path:
    return var_dir(root) / "llm_cache" / "mission"


def skill_catalog_dir(root: Path) -> Path:
    return var_dir(root) / "skills"


def candidate_dir(mission: Path, card_id: str) -> Path:
    """One card owns one tree. Parallel missions must not share experiment.py."""
    dest = Path(mission) / "candidates" / str(card_id)
    for name in (
        "proposal",
        "experiment",
        "world",
        "probe",
        "host",
        "logs",
        "verdict",
    ):
        (dest / name).mkdir(parents=True, exist_ok=True)
    return dest


def write_idea(
    dest: Path,
    *,
    card_id: str,
    brief: dict[str, Any],
    works: list[dict[str, Any]],
    note_md: str = "",
    note_tex: str = "",
    probe: dict[str, Any] | None = None,
    protocol_md: str = "",
    protocol: dict[str, Any] | None = None,
    readme_md: str = "",
) -> Path:
    folder = candidate_dir(dest, card_id)
    (folder / "brief.json").write_text(
        json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "papers.json").write_text(
        json.dumps(works, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if note_md:
        (folder / "note.md").write_text(note_md, encoding="utf-8")
    if note_tex:
        (folder / "note.tex").write_text(note_tex, encoding="utf-8")
    if protocol_md:
        (folder / "protocol.md").write_text(protocol_md, encoding="utf-8")
    if protocol:
        protocol_text = json.dumps(protocol, ensure_ascii=False, indent=2)
        (folder / "protocol.json").write_text(protocol_text, encoding="utf-8")
        from . import chain
        from .evidence import sha256_text

        # Prefer the pre-probe REGISTER_PROTOCOL. A second registration
        # here would sit after execution and prove nothing new.
        digest = str(protocol.get("experiment_digest") or "")
        already = chain.find_event(
            folder / "chain.jsonl",
            chain.REGISTER_PROTOCOL,
            experiment_digest=digest,
        ) if digest else None
        if already is None:
            chain.append_event(
                folder / "chain.jsonl",
                chain.REGISTER_PROTOCOL,
                {
                    "card_id": str(card_id),
                    "protocol_digest": sha256_text(protocol_text),
                    "experiment_digest": digest,
                    "world": dict(protocol.get("world") or {})
                    if isinstance(protocol.get("world"), dict)
                    else {},
                },
            )
    if readme_md:
        (folder / "README.md").write_text(readme_md, encoding="utf-8")
    if probe:
        source = str(probe.get("source") or "")
        if source:
            (folder / "experiment.py").write_text(source, encoding="utf-8")
            (folder / "experiment" / "experiment.py").write_text(
                source, encoding="utf-8"
            )
        metrics = probe.get("metrics")
        if isinstance(metrics, dict) and metrics:
            (folder / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        (folder / "probe.json").write_text(
            json.dumps(
                {k: v for k, v in probe.items() if k != "source"},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return folder


def seal_candidate(folder: Path) -> Path:
    """Mark a finished candidate tree. Later writers should treat it as closed."""
    folder = Path(folder)
    marker = folder / "verdict" / "COMPLETE.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"sealed": True}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return marker


def write_summary(
    dest: Path,
    topic: str,
    done: dict[str, Any],
    *,
    packet_md: str = "",
) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "topic.txt").write_text(topic.strip() + "\n", encoding="utf-8")
    if packet_md:
        (dest / "RESEARCH_PACKET.md").write_text(packet_md, encoding="utf-8")
    path = dest / "summary.json"
    path.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
