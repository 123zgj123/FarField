"""Weak literature cannot become related work."""

from __future__ import annotations

import unittest

from farfield.extras.claimspec import ContractViolation, LiteratureSet, assert_related_work_subset
from farfield.extras.worldfields import papers_on_claim_object


STRONG = {
    "arxiv_id": "P1",
    "title": "Live-SWE-agent tool-call traces",
    "abstract": "We study created_tools on SWE-bench execution traces.",
}
WEAK_CAD = {
    "arxiv_id": "CAD1",
    "title": "Something about tool-call",
    "abstract": "We study tool-call trajectories in general.",
}
WEAK_ROOM = {
    "arxiv_id": "ROOM1",
    "title": "created tools",
    "abstract": "created_tools events on traces.",
}

TOPIC = "code world model of Live-SWE-agent tool-call trajectories"
CLAIM = (
    "On Live-SWE-agent trajectories, created_tools events forecast "
    "next-step nonzero returncode."
)


class LiteratureIdentityTests(unittest.TestCase):
    def test_strong_is_not_padded_with_weak(self) -> None:
        kept = papers_on_claim_object(
            [STRONG, WEAK_CAD, WEAK_ROOM],
            claim=CLAIM,
            topic=TOPIC,
            minimum="strong",
        )
        self.assertEqual([row["arxiv_id"] for row in kept], ["P1"])

    def test_empty_claim_is_a_contract_violation(self) -> None:
        with self.assertRaises(ContractViolation):
            papers_on_claim_object([STRONG], claim="", topic=TOPIC)

    def test_review_prompt_pool_is_strong_only(self) -> None:
        from types import SimpleNamespace

        from farfield.extras.review import _strong_review_works

        kept = _strong_review_works(
            SimpleNamespace(claim=CLAIM), TOPIC, [STRONG, WEAK_CAD, WEAK_ROOM]
        )
        self.assertEqual([row["arxiv_id"] for row in kept], ["P1"])

    def test_related_work_subset_rejects_weak_ids(self) -> None:
        literature = LiteratureSet(claim_object_strong=(STRONG,), claim_object_weak=(WEAK_CAD,))
        assert_related_work_subset(["P1"], literature)
        with self.assertRaises(ContractViolation):
            assert_related_work_subset(["P1", "CAD1"], literature)


if __name__ == "__main__":
    unittest.main()
