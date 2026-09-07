"""Nonexistent mechanisms cannot become testable through prose."""

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
    WorldObjectRegistry,
    admit_generated_card,
    typecheck_claim,
)


DS_WORLD = WorldObjectRegistry.from_names(
    (
        "divergent",
        "accepted",
        "task_return",
        "validator_writes",
        "contrast:divergent",
        "false_accept_fraction",
        "freeze_validators",
    )
)


class ObjectIdentityTests(unittest.TestCase):
    def test_retrieval_index_mechanism_needs_world_expansion(self) -> None:
        spec = ClaimSpec(
            claim_id="c1",
            target_object="executable program state",
            mechanism=MechanismSpec(
                name="accepted-state retrieval index",
                status=MechanismStatus.ABSENT,
                world_refs=(),
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
        )
        result = typecheck_claim(spec, DS_WORLD)
        self.assertEqual(
            result.disposition, CompileDisposition.NEEDS_WORLD_EXPANSION
        )
        self.assertNotEqual(
            result.disposition, CompileDisposition.OBJECT_VALIDATED
        )

    def test_mapping_prose_cannot_create_world_refs(self) -> None:
        admitted = admit_generated_card(
            claim_id="c1",
            statement="The retrieval index partitions accepted states.",
            mechanism_name="accepted-state retrieval index",
            handle_id="contrast:divergent",
            measurement_dv="false_accept_fraction",
            target_object="executable program state",
            world=DS_WORLD,
        )
        self.assertEqual(admitted.spec.mechanism.status, MechanismStatus.HYPOTHETICAL)
        self.assertEqual(admitted.spec.mechanism.world_refs, ())
        self.assertNotEqual(admitted.spec.claim_type, ClaimType.MECHANISM)

    def test_attested_lever_can_be_an_in_world_handle_test(self) -> None:
        world = WorldObjectRegistry.from_names(("dropout", "mean_degree"))
        admitted = admit_generated_card(
            claim_id="c1",
            statement="dropout lowers mean degree",
            mechanism_name="protein folding",
            handle_id="dropout",
            measurement_dv="mean_degree",
            target_object="graph",
            world=world,
        )
        self.assertTrue(admitted.ok)
        self.assertEqual(admitted.spec.claim_type, ClaimType.INTERVENTION)
        self.assertEqual(admitted.spec.mechanism.status, MechanismStatus.HYPOTHETICAL)
        self.assertEqual(admitted.spec.mechanism.world_refs, ())


if __name__ == "__main__":
    unittest.main()
