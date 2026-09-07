"""Claim-spec persistence is a ClaimSpec, not a TypecheckResult wrapper."""

from __future__ import annotations

import unittest

from farfield.extras.claimspec import (
    ClaimSpec,
    ClaimType,
    CompileDisposition,
    HandleKind,
    HandleSpec,
    MeasurementSpec,
    MechanismSpec,
    MechanismStatus,
    TypecheckResult,
    WorldObjectRegistry,
    admit_generated_card,
    claim_spec_from_payload,
    compile_payload,
    in_world_testable,
    should_queue_world_expansion,
    typecheck_claim,
)

DS_WORLD = WorldObjectRegistry.from_names(
    (
        "divergent",
        "accepted",
        "contrast:divergent",
        "false_accept_fraction",
    )
)


class CompilePayloadTests(unittest.TestCase):
    def test_compile_payload_is_flat_claim_spec(self) -> None:
        admitted = admit_generated_card(
            claim_id="c1",
            statement="The retrieval index partitions accepted states.",
            mechanism_name="accepted-state retrieval index",
            handle_id="contrast:divergent",
            measurement_dv="false_accept_fraction",
            target_object="executable program state",
            world=DS_WORLD,
        )
        payload = compile_payload(admitted)
        self.assertIn("mechanism", payload)
        self.assertNotIn("spec", payload)
        spec = claim_spec_from_payload(payload)
        self.assertEqual(spec.mechanism.status, MechanismStatus.HYPOTHETICAL)
        self.assertEqual(spec.handle.id, "contrast:divergent")
        self.assertEqual(payload["disposition"], CompileDisposition.OBJECT_VALIDATED.value)
        self.assertTrue(in_world_testable(payload))
        self.assertTrue(should_queue_world_expansion(payload))

    def test_legacy_typecheck_wrapper_still_loads(self) -> None:
        wrapped = TypecheckResult(
            CompileDisposition.OBJECT_VALIDATED,
            ClaimSpec(
                claim_id="c1",
                target_object="program state",
                mechanism=MechanismSpec(
                    name="accepted-state retrieval index",
                    status=MechanismStatus.HYPOTHETICAL,
                ),
                handle=HandleSpec(
                    id="contrast:divergent",
                    kind=HandleKind.OBSERVATIONAL,
                    world_refs=("contrast:divergent",),
                ),
                measurement=MeasurementSpec(dv="false_accept_fraction"),
                claim_type=ClaimType.ASSOCIATION,
                statement="retrieval index associates with false acceptance",
            ),
        ).to_dict()
        spec = claim_spec_from_payload(wrapped)
        self.assertEqual(spec.mechanism.name, "accepted-state retrieval index")
        self.assertEqual(spec.handle.id, "contrast:divergent")

    def test_untestable_mechanism_is_not_in_world_testable(self) -> None:
        result = typecheck_claim(
            ClaimSpec(
                claim_id="c1",
                target_object="program state",
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
                statement="The retrieval index causes reduced false acceptance",
            ),
            DS_WORLD,
        )
        payload = compile_payload(result)
        self.assertEqual(
            payload["disposition"], CompileDisposition.NEEDS_WORLD_EXPANSION.value
        )
        self.assertFalse(in_world_testable(payload))
        self.assertTrue(should_queue_world_expansion(payload))


if __name__ == "__main__":
    unittest.main()
