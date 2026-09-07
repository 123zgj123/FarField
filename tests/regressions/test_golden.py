"""Golden regression oracles compressed from live mission failures.

These fixtures are compiler specifications. They do not call an LLM.
"""

from __future__ import annotations

import json
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
    handle_kind_of,
    typecheck_claim,
)


ROOT = Path(__file__).resolve().parent


def _load(name: str) -> dict:
    return json.loads((ROOT / name / "expected.json").read_text(encoding="utf-8"))


class GoldenRegressionTests(unittest.TestCase):
    def test_ds_retrieval_index_cannot_compile_as_mechanism(self) -> None:
        fixture = _load("ds")
        world = WorldObjectRegistry.from_names(fixture["world_objects"])
        claim = fixture["claim"]
        spec = ClaimSpec(
            claim_id="ds",
            target_object="executable program state",
            mechanism=MechanismSpec(
                name=claim["mechanism"],
                status=MechanismStatus.ABSENT,
            ),
            handle=HandleSpec(
                id=claim["handle"],
                kind=handle_kind_of(claim["handle"]),
                world_refs=(claim["handle"],) if world.exists(claim["handle"]) else (),
            ),
            measurement=MeasurementSpec(dv="false_accept_fraction"),
            claim_type=ClaimType(claim["claim_type"]),
            statement=claim["statement"],
        )
        result = typecheck_claim(spec, world)
        self.assertEqual(result.disposition.value, fixture["expected"]["disposition"])
        self.assertEqual(spec.handle.kind.value, fixture["expected"]["handle_type"])
        self.assertNotEqual(result.disposition, CompileDisposition.OBJECT_VALIDATED)

    def test_fixtures_exist_for_each_live_failure(self) -> None:
        for name in ("v2", "v3", "v4", "ds"):
            self.assertTrue((ROOT / name / "expected.json").is_file(), name)


if __name__ == "__main__":
    unittest.main()
