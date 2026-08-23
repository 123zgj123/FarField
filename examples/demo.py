"""End-to-end local control-plane flow; not an empirical RSI result."""

from __future__ import annotations

import tempfile
from pathlib import Path

from farfield.adapters import render_bounded_agent_handoff
from farfield.models import Evidence, PairedTrial, UpdateKind
from farfield.runtime import FarfieldRuntime


def artifact(root: Path, name: str, content: str) -> str:
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path.resolve().as_uri()


with tempfile.TemporaryDirectory() as temporary_directory:
    root = Path(temporary_directory)
    runtime = FarfieldRuntime.create(
        root,
        "Discover important mechanisms that survive external falsification.",
    )
    objective_id = runtime.declare_initial_objective(
        text="Does the proposed mechanism explain the boundary failure?",
        success_criteria=["A preregistered intervention separates hypotheses"],
        constraints=["equal compute", "preserve negative results"],
    )

    mission_id = runtime.export_mission(
        objective_id=objective_id,
        goal="Run the separating intervention and preserve raw outputs.",
        completion_evidence=["raw output", "task-native verifier report"],
        constraints=["do not redefine success"],
        budget={"wall_time_minutes": 30, "compute_units": 10},
    )
    contract = runtime.load_mission_contract(mission_id)
    print(render_bounded_agent_handoff(contract))
    clause_ids = tuple(item["id"] for item in contract["completion_clauses"])

    evidence_id = runtime.record_evidence(
        Evidence(
            claim="The intervention separates the hypotheses in the local run.",
            artifact_uri=artifact(root, "result.json", '{"effect": 0.12}\n'),
            verifier_id="task-native-v1",
            outcome="supports",
            scope="local-demo-only",
            mission_id=mission_id,
            clause_ids=clause_ids,
            sealed=True,
        ),
        actor="protected_vault",
    )
    runtime.settle_mission(
        mission_id,
        status="done",
        evidence_ids=[evidence_id],
    )

    update_id = runtime.propose_update(
        kind=UpdateKind.SKILL,
        artifact_uri=artifact(
            root,
            "skill.md",
            "# Separating intervention planner\nUse preregistered competing predictions.\n",
        ),
        scope="local causal tasks",
        rationale="Candidate extracted only after final settlement.",
        source_mission_id=mission_id,
    )
    trials = [
        PairedTrial(
            task_family=f"family-{index % 3}",
            candidate_score=0.80 + index * 0.002,
            baseline_score=0.70,
            candidate_compute=10,
            baseline_compute=10,
            protected_regression=0.0,
        )
        for index in range(8)
    ]
    receipt_id = runtime.record_protected_evaluation(
        update_id=update_id,
        protected_eval_id="pev_local-demo-001",
        trials=trials,
        evidence_ids=[evidence_id],
        frozen_baseline=True,
        independent_verifier=True,
        rollback_ref="snapshot://before-skill-v1",
    )
    decision = runtime.evaluate_update(update_id, receipt_id)
    assert decision.accepted
    assert runtime.verify()[0]
    print(f"accepted={decision.accepted}; active={runtime.active_update_ids()}")
