"""Production research entry: OpenWorld owns the next action."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from . import OpenWorld
from .events import make_event
from .executors import fixture_from_root
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
        "traces": ["validator-write-v1", "validator-write-v2"],
        "n": 2,
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
    if env.state.goal and env.state.open_questions:
        return env
    env.state.goal = topic
    wid = str(world_id or getattr(world, "id", "") or "W0")
    if world is not None and getattr(world, "root", None):
        env.state.world_id = str(getattr(world, "id", "") or wid)
        env.state.world_versions.append(
            {
                "id": env.state.world_id,
                "world_id": env.state.world_id,
                "digest": str(getattr(world, "digest", "") or ""),
                "observables": "",
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
    env.state.add_question(
        question_id=qid,
        text=topic if topic.endswith("?") else f"what does this freeze show about: {topic}",
        required_observable="validator_snapshots",
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
                {"id": env.state.world_id, "world_id": env.state.world_id, "observables": ""},
            ),
            make_event(
                "QuestionCreated",
                {
                    "id": qid,
                    "text": env.state.open_questions[0]["text"] if env.state.open_questions else topic,
                    "required_observable": "validator_snapshots",
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
    }
    for _ in range(max(0, int(horizon))):
        reason = should_terminate(env.state, horizon=horizon) if stop_on_exhaust else None
        if reason and reason != "budget" and stop_on_exhaust:
            yield {"stage": "terminate", "reason": reason, "ticks": env.state.ticks}
            break
        yield env.tick()
    rebuilt = rebuild_state(env.log)
    yield {
        "stage": "done",
        "controller": "OpenWorld",
        "ticks": env.state.ticks,
        "world_id": env.state.world_id,
        "harness_version": env.harness.version_id,
        "resolved": [row.get("id") for row in env.state.resolved_questions],
        "open": [row.get("id") for row in env.state.open_questions],
        "frontier": dict(env.state.frontier),
        "debt": dict(env.state.debt),
        "metrics": dict(env.state.metrics),
        "digest": state_digest(env.state),
        "rebuild_digest": state_digest(rebuilt),
        "digest_ok": state_digest(env.state) == state_digest(rebuilt),
    }
