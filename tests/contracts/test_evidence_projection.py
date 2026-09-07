"""Evidence cannot make a stronger claim than the registered handle permits."""

from __future__ import annotations

import unittest

from farfield.extras.claimspec import (
    ClaimSpec,
    ClaimType,
    HandleKind,
    HandleSpec,
    MeasurementSpec,
    MechanismSpec,
    MechanismStatus,
    ScopedVerdict,
    project_evidence_to_claim,
)
from farfield.extras.packet import render_result_to_claim


SPEC = ClaimSpec(
    claim_id="c1",
    target_object="program state",
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
    measurement=MeasurementSpec(dv="false_accept_fraction"),
    claim_type=ClaimType.ASSOCIATION,
    statement="retrieval index causes reduced false acceptance",
)


class EvidenceProjectionTests(unittest.TestCase):
    def test_significant_difference_is_association_only(self) -> None:
        projection = project_evidence_to_claim(
            SPEC,
            arithmetic_verdict="supports",
            observed_effect={"delta": -0.149},
            metric="false_accept_fraction",
        )
        self.assertEqual(projection.record.verdict, ScopedVerdict.SUPPORTS_ASSOCIATION)
        self.assertNotEqual(projection.record.verdict, ScopedVerdict.SUPPORTS_MECHANISM)
        text = projection.scientific_text().lower()
        self.assertIn("differed", text)
        self.assertIn("does not establish the proposed mechanism", text)
        self.assertNotIn("retrieval index supported", text)
        self.assertNotIn("reachability cover supported", text)

    def test_writeup_cannot_launder_the_mechanism(self) -> None:
        projection = project_evidence_to_claim(
            SPEC,
            arithmetic_verdict="supports",
            observed_effect={"delta": -0.149},
        )
        text = render_result_to_claim(
            {
                "claim": SPEC.statement,
                "probe_kind": "WORLD",
                "verdict": "supports",
                "world_id": "program-state-ds",
                "claim_spec": SPEC.to_dict(),
                "scoped_verdict": projection.record.verdict.value,
                "evidence_projection": projection.scientific_text(),
                "delta": -0.149,
            }
        )
        lowered = text.lower()
        self.assertIn("supports_association", lowered)
        self.assertNotIn("retrieval index supported", lowered)
        self.assertNotIn("reachability cover supported", lowered)
        self.assertIn("does not establish the proposed mechanism", lowered)

    def test_legacy_typecheck_wrapper_still_projects(self) -> None:
        from farfield.extras.claimspec import CompileDisposition, TypecheckResult

        wrapped = TypecheckResult(
            CompileDisposition.OBJECT_VALIDATED, SPEC
        ).to_dict()
        text = render_result_to_claim(
            {
                "claim": SPEC.statement,
                "probe_kind": "WORLD",
                "verdict": "supports",
                "world_id": "program-state-ds",
                "claim_spec": wrapped,
                "delta": -0.149,
            }
        )
        lowered = text.lower()
        self.assertIn("does not establish the proposed mechanism", lowered)
        self.assertIn("accepted-state retrieval index", lowered)
        self.assertNotIn("retrieval index supported", lowered)


if __name__ == "__main__":
    unittest.main()
