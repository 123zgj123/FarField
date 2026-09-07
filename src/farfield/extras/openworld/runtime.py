"""Production research entry: OpenWorld owns the next action."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from . import OpenWorld
from .events import make_event
from .executors import fixture_from_root
from .worldio import bind_fixture
from .progress import should_terminate
from .reducer import apply_events, rebuild_state, state_digest


def _seed_world(workspace: Path, world_id: str = "W0", topic: str = "") -> Path:
    root = workspace / "worlds" / world_id
    if root.is_dir() and any(root.iterdir()):
        return root
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": world_id,
        "topic": topic,
        "provenance": "generated",
        "records": [],
    }
    (root / "world.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return root


def bootstrap_research(
    topic: str,
    workspace: Path,
    *,
    world: Any = None,
    world_id: str = "",
) -> OpenWorld:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    env = OpenWorld.load(workspace)
    if env.state.goal:
        if env.state.goal != topic:
            raise ValueError("workspace already belongs to a different research topic")
        if world is not None or world_id:
            requested = str(getattr(world, "id", "") or world_id or world)
            if requested not in {str(row.get("id")) for row in env.state.world_versions}:
                raise ValueError("workspace belongs to a different world lineage")
        return env
    env.state.goal = topic
    wid = str(world_id or getattr(world, "id", "") or "W0")
    if world is not None or world_id:
        fixture = bind_fixture(workspace, world, world_id)
        env.state.world_id = fixture.id
        env.state.world_versions.append(
            {
                "id": env.state.world_id,
                "world_id": env.state.world_id,
                "digest": fixture.digest,
                "observables": fixture.schema or "artifact_inventory",
                "path": str(fixture.root),
            }
        )
    else:
        root = _seed_world(workspace, wid, topic=topic)
        fixture = fixture_from_root(root, wid)
        env.state.world_id = wid
        env.state.world_versions.append(
            {
                "id": wid,
                "world_id": wid,
                "digest": fixture.digest,
                "observables": "",
                "path": str(root),
            }
        )
    qid = "Q1"
    observable = fixture.schema or "artifact_inventory"
    # Only an actual snapshot artifact enables the specialized observer.
    if "validator_snapshots.json" in fixture.files:
        observable = "validator_snapshots"
    env.state.world_versions[-1]["observables"] = (
        observable if fixture.schema or "validator_snapshots.json" in fixture.files else ""
    )
    env.state.add_question(
        question_id=qid,
        text=topic,
        required_observable=observable,
        lineage_id="line-q1",
        epistemic="GENERATED",
    )
    env._ensure_baseline()
    env.state = apply_events(
        env.state,
        [
            make_event("GoalSet", {"goal": topic}),
            make_event(
                "WorldCreated",
                dict(env.state.world_versions[-1]),
            ),
            make_event(
                "QuestionCreated",
                {
                    "id": qid,
                    "text": env.state.open_questions[0]["text"] if env.state.open_questions else topic,
                    "required_observable": observable,
                    "lineage_id": "line-q1",
                    "epistemic": "GENERATED",
                },
            ),
        ],
        log=env.log,
    )
    env.persist()
    return env


def run_research(
    topic: str,
    *,
    workspace: Path,
    horizon: int = 20,
    world: Any = None,
    world_id: str = "",
    stop_on_exhaust: bool = True,
    **_ignored: Any,
) -> Iterator[dict[str, Any]]:
    """Production loop. Ignores legacy stage knobs."""
    env = bootstrap_research(topic, Path(workspace), world=world, world_id=str(world_id or ""))
    yield {
        "stage": "openworld_start",
        "controller": "OpenWorld",
        "goal": env.state.goal,
        "world_id": env.state.world_id,
        "harness_version": env.harness.version_id,
        "horizon": int(horizon),
        "output_kind": "controller_trace",
        "research_complete": False,
        "capability_note": "OpenWorld runs local observations. Use --legacy-pipeline for LLM/literature and research packets.",
    }
    for _ in range(max(0, int(horizon))):
        reason = should_terminate(env.state, horizon=horizon) if stop_on_exhaust else None
        if reason and reason != "budget" and stop_on_exhaust:
            yield {"stage": "terminate", "reason": reason, "ticks": env.state.ticks}
            break
        yield env.tick()
    rebuilt = rebuild_state(env.log)
    status_text = (
        "# Research status / 研究状态\n\n"
        f"Topic: {env.state.goal}\n\n"
        "Status: incomplete. This run produced a controller trace and exploratory observations, "
        "not a completed research study. 描述性观测不能确认原始假设。\n\n"
        "Next: bind a relevant frozen dataset and implement a topic-specific measurement, "
        "or use --legacy-pipeline for literature, proposals and registered protocols.\n\n"
        f"Open questions: {[row.get('text') for row in env.state.open_questions]}\n"
        f"Blocked questions: {[row.get('text') for row in env.state.blocked_questions]}\n"
        f"Missing capabilities: {env.state.missing_capabilities}\n"
    )
    gaps = sorted({str(row.get("reason")) for row in env.history if row.get("reason")})
    status_text += f"Execution gaps: {gaps}\n"
    observations = [row for row in env.state.evidence_records if row.get("role") == "QuestionEvidence"]
    if observations:
        status_text += "\n## Exploratory observations / 探索性观测\n\n"
        status_text += "```json\n" + json.dumps(observations, ensure_ascii=False, indent=2) + "\n```\n"
    (Path(workspace) / "RESEARCH_STATUS.md").write_text(status_text, encoding="utf-8")
    yield {
        "stage": "done",
        "status": "incomplete",
        "research_complete": False,
        "output_kind": "controller_trace",
        "status_file": str(Path(workspace) / "RESEARCH_STATUS.md"),
        "controller": "OpenWorld",
        "ticks": env.state.ticks,
        "world_id": env.state.world_id,
        "harness_version": env.harness.version_id,
        "resolved": [row.get("id") for row in env.state.resolved_questions],
        "open": [row.get("id") for row in env.state.open_questions],
        "blocked": [row.get("id") for row in env.state.blocked_questions],
        "capability_gaps": gaps,
        "frontier": dict(env.state.frontier),
        "debt": dict(env.state.debt),
        "metrics": dict(env.state.metrics),
        "digest": state_digest(env.state),
        "rebuild_digest": state_digest(rebuilt),
        "digest_ok": state_digest(env.state) == state_digest(rebuilt),
    }
