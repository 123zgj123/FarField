"""Observational handles cannot produce causal or mechanistic claims."""

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
    typecheck_claim,
)


WORLD = WorldObjectRegistry.from_names(
    ("contrast:divergent", "divergent", "false_accept_fraction", "freeze_validators")
)


def _spec(claim_type: ClaimType, handle: str) -> ClaimSpec:
    return ClaimSpec(
        claim_id="c1",
        target_object="program state",
        mechanism=MechanismSpec(name="stratum contrast", status=MechanismStatus.ABSENT),
        handle=HandleSpec(id=handle, kind=HandleKind.OBSERVATIONAL if handle.startswith("contrast:") else HandleKind.INTERVENTIONAL, world_refs=(handle,)),
        measurement=MeasurementSpec(dv="false_accept_fraction"),
        claim_type=claim_type,
        statement="divergent runs differ from non-divergent runs",
    )


class HandleTypingTests(unittest.TestCase):
    def test_observational_causal_is_a_type_mismatch(self) -> None:
        spec = ClaimSpec(
            claim_id="c1",
            target_object="program state",
            mechanism=MechanismSpec(
                name="freeze_validators",
                status=MechanismStatus.ATTESTED,
                world_refs=("freeze_validators",),
            ),
            handle=HandleSpec(
                id="contrast:divergent",
                kind=HandleKind.OBSERVATIONAL,
                world_refs=("contrast:divergent",),
            ),
            measurement=MeasurementSpec(dv="false_accept_fraction"),
            claim_type=ClaimType.CAUSAL,
            statement="divergent runs differ from non-divergent runs",
        )
        result = typecheck_claim(spec, WORLD)
        self.assertEqual(
            result.disposition, CompileDisposition.HANDLE_CLAIM_TYPE_MISMATCH
        )

    def test_observational_mechanism_is_a_type_mismatch(self) -> None:
        spec = ClaimSpec(
            claim_id="c1",
            target_object="program state",
            mechanism=MechanismSpec(
                name="freeze_validators",
                status=MechanismStatus.ATTESTED,
                world_refs=("freeze_validators",),
            ),
            handle=HandleSpec(
                id="contrast:divergent",
                kind=HandleKind.OBSERVATIONAL,
                world_refs=("contrast:divergent",),
            ),
            measurement=MeasurementSpec(dv="false_accept_fraction"),
            claim_type=ClaimType.MECHANISM,
            statement="divergent runs differ from non-divergent runs",
        )
        result = typecheck_claim(spec, WORLD)
        self.assertEqual(
            result.disposition, CompileDisposition.HANDLE_CLAIM_TYPE_MISMATCH
        )

    def test_observational_association_is_allowed(self) -> None:
        result = typecheck_claim(
            _spec(ClaimType.ASSOCIATION, "contrast:divergent"), WORLD
        )
        self.assertTrue(result.ok)

    def test_interventional_handle_may_be_an_intervention(self) -> None:
        spec = ClaimSpec(
            claim_id="c1",
            target_object="program state",
            mechanism=MechanismSpec(name="freeze", status=MechanismStatus.ABSENT),
            handle=HandleSpec(
                id="freeze_validators",
                kind=HandleKind.INTERVENTIONAL,
                world_refs=("freeze_validators",),
            ),
            measurement=MeasurementSpec(dv="false_accept_fraction"),
            claim_type=ClaimType.INTERVENTION,
            statement="freezing validators changes false acceptance",
        )
        self.assertTrue(typecheck_claim(spec, WORLD).ok)


if __name__ == "__main__":
    unittest.main()
