"""A failed mandatory review cannot transition into evidence."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from farfield.extras.claimspec import cards_eligible_for_evidence


def _card(card_id: str) -> SimpleNamespace:
    return SimpleNamespace(card_id=card_id)


class ReviewFailClosedTests(unittest.TestCase):
    def test_invalid_review_schema_emits_no_evidence(self) -> None:
        batch = [(_card("gen_a"), "far"), (_card("gen_b"), "far")]
        events = [
            {
                "stage": "idea_review_refused",
                "card_id": "gen_a",
                "missing_capability": "model_review_schema",
            },
            {
                "stage": "idea_review",
                "card_id": "gen_b",
            },
        ]
        eligible = cards_eligible_for_evidence(batch, events, review_required=True)
        self.assertEqual([item[0].card_id for item in eligible], ["gen_b"])

    def test_all_reviews_refused_emits_nothing(self) -> None:
        batch = [(_card("gen_a"), "far")]
        events = [{"stage": "idea_review_refused", "card_id": "gen_a"}]
        eligible = cards_eligible_for_evidence(batch, events, review_required=True)
        self.assertEqual(eligible, [])

    def test_missing_review_is_fail_closed_when_required(self) -> None:
        batch = [(_card("gen_a"), "far")]
        eligible = cards_eligible_for_evidence(batch, [], review_required=True)
        self.assertEqual(eligible, [])

    def test_review_not_required_keeps_the_one_shot_test_path(self) -> None:
        batch = [(_card("gen_a"), "far")]
        eligible = cards_eligible_for_evidence(batch, [], review_required=False)
        self.assertEqual([item[0].card_id for item in eligible], ["gen_a"])


if __name__ == "__main__":
    unittest.main()
