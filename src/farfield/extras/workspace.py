"""A mission's working folder: the thing Robin keeps and Farfield used not to.

One directory per run. Papers, briefs, probe source, metrics, the compiled
note, and the executable protocol live as ordinary files a researcher can
open after the browser tab is gone. Nothing here is evidence the kernel
would accept; it is the lab notebook plus a next-week experiment pack.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def jsonable(value: Any) -> Any:
    """Mission events must dump. Sets and model objects cannot stay raw."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted((jsonable(item) for item in value), key=lambda item: str(item))
    if hasattr(value, "to_dict"):
        try:
            return jsonable(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


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
        handle.write(json.dumps(jsonable(event), ensure_ascii=False) + "\n")


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
                "has_packet": (dest / "RESEARCH_PACKET.md").is_file()
                or any((dest / "ideas").glob("*/RESEARCH_PACKET.md")),
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
        "skills",
    ):
        (dest / name).mkdir(parents=True, exist_ok=True)
    return dest


def idea_packet_dir(mission: Path, folder_name: str) -> Path:
    """One idea, one folder named by core content. Sibling claims stay out."""
    dest = Path(mission) / "ideas" / str(folder_name)
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def prune_idea_packets(mission: Path, *, keep: set[str]) -> None:
    """Drop idea folders that are no longer live (weakens / no_handle / stale hashes)."""
    root = Path(mission) / "ideas"
    if not root.is_dir():
        return
    for folder in root.iterdir():
        if folder.is_dir() and folder.name not in keep:
            shutil.rmtree(folder)


def write_idea_packets(
    dest: Path,
    *,
    card_id: str,
    human_md: str,
    agent_md: str,
    experiment_md: str = "",
    human_md_en: str = "",
    experiment_md_en: str = "",
    derivation_md: str = "",
    result_md: str = "",
    ablation_md: str = "",
    kill_md: str = "",
    analysis_md: str = "",
    folder_name: str | None = None,
) -> Path:
    """Write one idea using the ARIS stage layout.

    Research brief lives in ``idea-stage/``. Experiment plan, tracker, and
    results live in ``refine-logs/``. Executable protocol stays under
    ``candidates/<card_id>/``. The idea folder is named by core content,
    not the machine digest. Sibling ideas are not mixed into this tree.
    Existing intern refine-logs are overwritten by this compiler: the
    mission owns brief/plan until it hands off an execute order.
    """
    name = str(folder_name or "").strip() or str(card_id)
    folder = idea_packet_dir(dest, name)
    idea_stage = folder / "idea-stage"
    refine = folder / "refine-logs"
    idea_stage.mkdir(parents=True, exist_ok=True)
    refine.mkdir(parents=True, exist_ok=True)
    brief = human_md
    if derivation_md.strip() and "## 领域知识" not in brief and "## Domain knowledge" not in brief:
        brief = brief.rstrip() + "\n\n## 领域知识\n\n" + derivation_md.strip() + "\n"
    if kill_md.strip() and "## 非目标" not in brief and "## Non-goals" not in brief:
        brief = brief.rstrip() + "\n\n## 非目标\n\n" + kill_md.strip() + "\n"
    (idea_stage / "RESEARCH_BRIEF.md").write_text(brief, encoding="utf-8")
    if human_md_en.strip():
        (idea_stage / "RESEARCH_BRIEF_EN.md").write_text(human_md_en, encoding="utf-8")
    plan = experiment_md
    if ablation_md.strip() and "消融" not in plan and "Ablation" not in plan:
        plan = plan.rstrip() + "\n\n" + ablation_md.strip() + "\n"
    plan_path = refine / "EXPERIMENT_PLAN.md"
    if plan.strip():
        plan_path.write_text(plan, encoding="utf-8")
    if experiment_md_en.strip():
        (refine / "EXPERIMENT_PLAN_EN.md").write_text(experiment_md_en, encoding="utf-8")
    results = "\n\n".join(
        part.strip()
        for part in (result_md, analysis_md)
        if str(part or "").strip()
    )
    results_path = refine / "EXPERIMENT_RESULTS.md"
    if results:
        results_path.write_text(results + "\n", encoding="utf-8")
    tracker_path = refine / "EXPERIMENT_TRACKER.md"
    tracker_lines = [
        "# Experiment Tracker",
        "",
        "Latest copy. Downstream skills read this filename.",
        "",
        f"- Idea: `{name}`",
        "- Plan (zh): `refine-logs/EXPERIMENT_PLAN.md`",
        "- Plan (en): `refine-logs/EXPERIMENT_PLAN_EN.md`",
        "- Brief (zh): `idea-stage/RESEARCH_BRIEF.md`",
        "- Brief (en): `idea-stage/RESEARCH_BRIEF_EN.md`",
        "- Results: `refine-logs/EXPERIMENT_RESULTS.md`",
        "",
        "The research brief already inlines the experiment design.",
        "Must-run rows in the plan are executed, not a later research generate.",
        "",
    ]
    tracker_path.write_text("\n".join(tracker_lines), encoding="utf-8")
    dashboard = "\n".join(
        [
            f"# Idea `{name}`",
            "",
            "This folder is one idea. The research plan is complete in the brief;",
            "the experiment design is also complete on its own.",
            "",
            "- 研究方案（中文，含完整实验设计）: `idea-stage/RESEARCH_BRIEF.md`",
            "- Research plan (English, experiment design inlined): `idea-stage/RESEARCH_BRIEF_EN.md`",
            "- 实验设计（中文）: `refine-logs/EXPERIMENT_PLAN.md`",
            "- Experiment design (English): `refine-logs/EXPERIMENT_PLAN_EN.md`",
            "- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`",
            "- 已有结果: `refine-logs/EXPERIMENT_RESULTS.md`",
            "- World rehearsal: `world-sim/` (imagined; cannot climb)",
            "- Execute: `AGENT_PACKET.md`",
            "",
            "本场已经迭代完。AGENT_PACKET 只下达执行。",
            "",
        ]
    )
    (folder / "README.md").write_text(dashboard, encoding="utf-8")
    (folder / "RESEARCH_PACKET.md").write_text(dashboard, encoding="utf-8")
    (folder / "AGENT_PACKET.md").write_text(agent_md, encoding="utf-8")
    return folder


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


PROGRESS_FILE = "progress.ndjson"
MANIFEST_FILE = "mission_manifest.json"
WORLD_QUEUE_FILE = "future_world_queue.jsonl"
RESEARCH_STATE_FILE = "RESEARCH_STATE.json"


def progress_path(mission: Path, card_id: str) -> Path:
    return Path(mission) / "candidates" / str(card_id) / PROGRESS_FILE


def append_progress(
    mission: Path | None,
    card_id: str,
    record: dict[str, Any],
    *,
    fsync: bool = True,
) -> Path | None:
    """Runtime progress. Not a scientific event. Visible before _do_entry returns."""
    if mission is None or not str(card_id or "").strip():
        return None
    dest = progress_path(Path(mission), card_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(jsonable(record), ensure_ascii=False)
    with dest.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    return dest


def enqueue_world_expansion(
    mission: Path | None, record: dict[str, Any], *, fsync: bool = True
) -> Path | None:
    """Append a Type-B missing-mechanism request. Not a scientific verdict."""
    if mission is None:
        return None
    dest = Path(mission) / WORLD_QUEUE_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(jsonable(record), ensure_ascii=False)
    with dest.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    return dest


def write_mission_manifest(dest: Path, payload: dict[str, Any]) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / MANIFEST_FILE
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


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


def write_idea_report(dest: Path, report_md: str) -> Path:
    """Mission-level idea landscape. One file, not per-idea memory pollution."""
    dest = Path(dest)
    folder = dest / "idea-stage"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "IDEA_REPORT.md"
    if report_md.strip():
        path.write_text(report_md, encoding="utf-8")
    return path


def write_question_board(dest: Path, board_md: str) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "QUESTION_BOARD.md"
    path.write_text(board_md if board_md.endswith("\n") else board_md + "\n", encoding="utf-8")
    return path


def write_archive(dest: Path, archive_md: str) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "ARCHIVE.md"
    path.write_text(
        archive_md if archive_md.endswith("\n") else archive_md + "\n", encoding="utf-8"
    )
    return path


def write_summary(
    dest: Path,
    topic: str,
    done: dict[str, Any],
    *,
    packet_md: str = "",
    packet_md_en: str = "",
) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "topic.txt").write_text(topic.strip() + "\n", encoding="utf-8")
    if packet_md:
        (dest / "RESEARCH_PACKET.md").write_text(packet_md, encoding="utf-8")
    if packet_md_en:
        (dest / "RESEARCH_PACKET_EN.md").write_text(packet_md_en, encoding="utf-8")
    path = dest / "summary.json"
    path.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
