"""A missing mechanism cannot start diagnose/probe. It must queue expansion."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.claimspec import (
    ClaimSpec,
    ClaimType,
    CompileDisposition,
    HandleKind,
    HandleSpec,
    MeasurementSpec,
    MechanismSpec,
    MechanismStatus,
    WorldObjectRegistry,
    compile_payload,
    typecheck_claim,
)
from farfield.extras.generate import GeneratedCard
from farfield.extras.mission import _evidence_pipeline
from farfield.extras.workspace import WORLD_QUEUE_FILE


DS_WORLD = WorldObjectRegistry.from_names(
    (
        "divergent",
        "accepted",
        "contrast:divergent",
        "false_accept_fraction",
    )
)


class BoomClient:
    def complete(self, *args, **kwargs):
        raise AssertionError("untestable mechanism must not call the LLM")


def _card(payload: dict) -> GeneratedCard:
    return GeneratedCard(
        card_id="gen_expansion01",
        operator="directional",
        claim="The accepted-state retrieval index causes reduced false acceptance",
        mechanism="accepted-state retrieval index",
        prediction="false_accept_fraction drops under the contrast handle",
        falsifier="pair_not_already_combined",
        pair=("executable program state", "retrieval index"),
        pair_nodes=("concept:a", "concept:b"),
        alienness=0.6,
        model="fake",
        artifact_digest="d",
        artifact_uri="file:///dev/null",
        replay_mode="replay",
        world_lever="contrast:divergent",
        world_observable="false_accept_fraction",
        claim_spec=payload,
    )


class WorldExpansionTests(unittest.TestCase):
    def test_needs_world_expansion_skips_evidence_generation(self) -> None:
        result = typecheck_claim(
            ClaimSpec(
                claim_id="c1",
                target_object="executable program state",
                mechanism=MechanismSpec(
                    name="accepted-state retrieval index",
                    status=MechanismStatus.ABSENT,
                ),
                handle=HandleSpec(
                    id="contrast:divergent",
                    kind=HandleKind.OBSERVATIONAL,
                    world_refs=("contrast:divergent",),
                ),
                measurement=MeasurementSpec(
                    dv="false_accept_fraction",
                    source_fields=("false_accept_fraction",),
                ),
                claim_type=ClaimType.MECHANISM,
                statement=(
                    "The accepted-state retrieval index causes reduced "
                    "false acceptance"
                ),
            ),
            DS_WORLD,
        )
        self.assertEqual(
            result.disposition, CompileDisposition.NEEDS_WORLD_EXPANSION
        )
        payload = compile_payload(result)
        with tempfile.TemporaryDirectory() as tmp:
            mission = Path(tmp)
            events = list(
                _evidence_pipeline(
                    BoomClient(),
                    None,
                    _card(payload),
                    "code world model",
                    run_probes=True,
                    workspace=mission,
                )
            )
            stages = [event.get("stage") for event in events]
            self.assertIn("world_expansion", stages)
            self.assertNotIn("diagnosing", stages)
            self.assertNotIn("probing", stages)
            self.assertNotIn("evidence", stages)
            queue = mission / WORLD_QUEUE_FILE
            self.assertTrue(queue.is_file())
            row = json.loads(queue.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["status"], "needs_world_expansion")
            self.assertIn("accepted-state retrieval index", row["missing_observables"])
            self.assertFalse(row["in_world_testable"])

    def test_acquire_queues_expansion_and_keeps_writeup(self) -> None:
        result = typecheck_claim(
            ClaimSpec(
                claim_id="c1",
                target_object="executable program state",
                mechanism=MechanismSpec(
                    name="accepted-state retrieval index",
                    status=MechanismStatus.ABSENT,
                ),
                handle=HandleSpec(
                    id="contrast:divergent",
                    kind=HandleKind.OBSERVATIONAL,
                    world_refs=("contrast:divergent",),
                ),
                measurement=MeasurementSpec(
                    dv="false_accept_fraction",
                    source_fields=("false_accept_fraction",),
                ),
                claim_type=ClaimType.MECHANISM,
                statement=(
                    "The accepted-state retrieval index causes reduced "
                    "false acceptance"
                ),
            ),
            DS_WORLD,
        )
        payload = compile_payload(result)
        card = _card(payload)
        card = GeneratedCard(**{**card.__dict__, "idea_kind": "acquire"})
        with tempfile.TemporaryDirectory() as tmp:
            mission = Path(tmp)
            bundle: dict = {"skip_writeup": False}
            events = list(
                _evidence_pipeline(
                    BoomClient(),
                    None,
                    card,
                    "code world model",
                    run_probes=True,
                    workspace=mission,
                    bundle=bundle,
                )
            )
            self.assertIn("world_expansion", [event.get("stage") for event in events])
            self.assertFalse(bundle.get("skip_writeup"))
            self.assertTrue((mission / WORLD_QUEUE_FILE).is_file())


if __name__ == "__main__":
    unittest.main()
