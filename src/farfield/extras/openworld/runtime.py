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
        # Snapshots are derived caches. Replaying adds new research views to
        # older runs without rewriting their immutable scientific event log.
        rebuilt = rebuild_state(env.log)
        if state_digest(env.state) != state_digest(rebuilt):
            env.state = rebuilt
            env.persist()
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
    client: Any = None,
    feed: Any = None,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    api_calls: float | None = 100,
    mode: str = "live",
    heldout_worlds: tuple[Any, ...] = (),
    policy_log: Path | None = None,
    policy_file: Path | None = None,
    policy_evaluation: Any = None,
    policy_suite: dict[str, list[str]] | None = None,
    policy_evaluator: Any = None,
    **_ignored: Any,
) -> Iterator[dict[str, Any]]:
    """One scientific controller; all research services are existing workers."""
    from ..llm import LLMClient, LLMUnavailable, TokenLedger, resolve_backend
    from .workers import LegacyResearchWorkers

    env = bootstrap_research(topic, Path(workspace), world=world, world_id=str(world_id or ""))
    client_gap = ""
    if client is None:
        try:
            backend = resolve_backend(model=model, base_url=base_url, api_key=api_key)
            call_limit = 100.0 if api_calls is None else float(api_calls)
            client = LLMClient(backend, Path(workspace) / "llm_cache", mode=mode,
                               ledger=TokenLedger(api_calls=call_limit, token_cost=call_limit * 16000))
        except LLMUnavailable as exc:
            client_gap = str(exc)
    env.connect_workers(LegacyResearchWorkers(client=client, feed=feed, workspace=Path(workspace)))
    primary_world = dict(next(row for row in env.state.world_versions if row.get("id") == env.state.world_id))
    for requested in heldout_worlds:
        fixture = bind_fixture(Path(workspace), requested, "")
        env.state = apply_events(env.state, [make_event("WorldCreated", {
            "id": fixture.id, "world_id": fixture.id, "digest": fixture.digest,
            "observables": fixture.schema, "research_role": "heldout", "role": "heldout",
            "path": str(fixture.root)}), make_event("WorldCreated", primary_world)], log=env.log)
    from ..routing import load_policy
    policy = load_policy(Path(policy_file)) if policy_file is not None else None
    backend = getattr(client, "backend", None)
    env.state = apply_events(env.state, [make_event("ArtifactAdded", {
        "kind": "research_service_session", "id": f"session-{env.log.next_seq()}",
        "model": str(getattr(backend, "model", model)), "base_url": str(getattr(backend, "base_url", base_url)),
        "mode": str(getattr(client, "mode", mode))}), make_event("ArtifactAdded", {
        "kind": "research_policy_snapshot", "policy": policy,
        "activation": "frozen_for_this_mission_segment"})], log=env.log)
    env.persist()
    yield {
        "stage": "openworld_start",
        "controller": "OpenWorld",
        "goal": env.state.goal,
        "world_id": env.state.world_id,
        "harness_version": env.harness.version_id,
        "horizon": int(horizon),
        "output_kind": "scientific_runtime",
        "research_complete": False,
        "capability_note": client_gap or "Existing research workers connected; admission remains evidence-gated.",
    }
    for _ in range(max(0, int(horizon))):
        reason = should_terminate(env.state, horizon=horizon) if stop_on_exhaust else None
        if reason and reason != "budget" and stop_on_exhaust:
            yield {"stage": "terminate", "reason": reason, "ticks": env.state.ticks}
            break
        event = env.tick()
        yield event
        if event.get("scheduled") is None:
            yield {"stage": "terminate", "reason": "no_executable_frontier_action", "ticks": env.state.ticks}
            break
    if policy_log is not None and policy_file is not None:
        from ..routing import append_mission, load_log, maybe_update, _validated_outcome, PolicyError
        import hashlib
        mission_id = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()
        cards = []
        for evidence in env.state.evidence_records:
            if evidence.get("role") != "ProbeEvidence":
                continue
            theory = next((t for t in env.state.theories if t.get("id") == evidence.get("theory_id")), {})
            row = dict(evidence, ledger="scientific", operator=theory.get("operator_reused") or "unknown")
            try:
                _validated_outcome(row)
            except (PolicyError, ValueError, TypeError):
                continue
            cards.append(row)
        # Budget exhaustion settles a run, not the scientific claims it explored.
        # Repeated resumes of the same mission never count as independent failures.
        if cards and not any(m.get("mission_id") == mission_id for m in load_log(Path(policy_log))):
            failures = [{"id": f"failure-{r['evidence_id']}", "class": "SCIENTIFIC_FAILURE",
                "signature": "non_discriminating_design" if r.get("verdict") == "uninformative" else "weakened_mechanism",
                "evidence_id": r["evidence_id"]} for r in cards if r.get("verdict") in {"weakens", "uninformative"}]
            append_mission(Path(policy_log), {"mission_id": mission_id, "status": "completed",
                "research_complete": False, "topic": topic, "cards": cards, "failures": failures,
                "scientific_state_digest": state_digest(env.state)})
        if policy_evaluation is None and policy_suite is not None:
            from ..routing import evaluate_research_policy
            policy_evaluation = evaluate_research_policy(Path(policy_log), Path(policy_file),
                suite=policy_suite, evaluator=policy_evaluator)
        decision = maybe_update(Path(policy_log), Path(policy_file), evaluation=policy_evaluation)
        yield {"stage": "research_policy", **decision}
    rebuilt = rebuild_state(env.log)
    status_text = (
        "# Research status / 研究状态\n\n"
        f"Topic: {env.state.goal}\n\n"
        "Status: open research. Run completion is not scientific success. "
        "描述性观测不能确认原始假设；验证结论只适用于已测试条件。\n\n"
        "Next actions are selected from ScientificState; missing services or frozen "
        "worlds remain explicit blockers. Speculative hypotheses are not verified knowledge.\n\n"
        f"Open questions: {[row.get('text') for row in env.state.open_questions]}\n"
        f"Blocked questions: {[row.get('text') for row in env.state.blocked_questions]}\n"
        f"Missing capabilities: {env.state.missing_capabilities}\n"
        f"Speculative frontier: {env.state.speculative_frontier}\n"
        f"Verified findings: {list(env.state.verified_research_state)}\n"
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
        "output_kind": "scientific_runtime",
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
        "speculative_frontier": list(env.state.speculative_frontier),
        "verified_research_state": dict(env.state.verified_research_state),
        "digest": state_digest(env.state),
        "rebuild_digest": state_digest(rebuilt),
        "digest_ok": state_digest(env.state) == state_digest(rebuilt),
    }
