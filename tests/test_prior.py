"""False-novelty checks: apply-X-to-Y dies, restated claims are found."""

from __future__ import annotations

import unittest

from farfield.extras.livefeed import FreshWork
from farfield.extras.prior import (
    apply_x_to_y,
    claim_coverage,
    claim_support,
    strongest_prior,
)


class ApplyXYTests(unittest.TestCase):
    def test_juxtaposition_is_the_named_failure(self) -> None:
        self.assertTrue(
            apply_x_to_y(
                "apply transformers to protein folding",
                "we use transformers for the protein folding task",
                ("transformers", "protein folding"),
            )
        )

    def test_a_third_technical_idea_survives(self) -> None:
        self.assertFalse(
            apply_x_to_y(
                "a wavelet-tree encoding of simplex pivots yields O(log n) last-repeat queries",
                "the pivot log is stored as a wavelet tree so rank queries reuse cached boundaries",
                ("succinct data structure", "pivot rule"),
            )
        )


class ClaimCoverageTests(unittest.TestCase):
    def test_a_restated_claim_is_a_killing_prior(self) -> None:
        claim = (
            "wavelet tree encodings of simplex pivot logs give last-repeat "
            "queries in logarithmic time"
        )
        work = FreshWork(
            title="Wavelet trees for simplex pivot logs",
            published="2026-01-01",
            arxiv_id="2601.00001",
            abstract=(
                "We show that wavelet tree encodings of simplex pivot logs "
                "give last-repeat queries in logarithmic time on public traces."
            ),
        )
        hit = strongest_prior(claim, [work])
        self.assertIsNotNone(hit)
        self.assertGreaterEqual(hit.support, 0.7)
        self.assertTrue(hit.kills)
        self.assertIn("wavelet tree", hit.span.lower())

    def test_a_nearby_abstract_is_not_a_restatement(self) -> None:
        claim = (
            "wavelet tree encodings of simplex pivot logs give last-repeat "
            "queries in logarithmic time"
        )
        work = FreshWork(
            title="A very fresh preprint",
            published="2026-08-15",
            arxiv_id="2608.09999v1",
            abstract="We study nearby constructions in this field.",
        )
        self.assertIsNone(strongest_prior(claim, [work]))
        self.assertLess(claim_coverage(claim, work.abstract), 0.3)

    def test_scattered_field_words_are_not_a_supporting_span(self) -> None:
        claim = (
            "wavelet tree encodings of simplex pivot logs give last-repeat "
            "queries in logarithmic time"
        )
        soup = (
            "Time, queries, encodings, logarithmic methods, trees, and logs "
            "appear throughout the literature on public data and results."
        )
        coverage = claim_coverage(claim, soup)
        support, span = claim_support(claim, soup)
        self.assertGreaterEqual(coverage, 0.4)
        self.assertLess(support, 0.5)
        self.assertIsNone(
            strongest_prior(
                claim,
                [
                    FreshWork(
                        title="A survey of nearby constructions",
                        published="2026-01-01",
                        arxiv_id="2601.11111",
                        abstract=soup,
                    )
                ],
            )
        )
        self.assertFalse(span)


if __name__ == "__main__":
    unittest.main()
