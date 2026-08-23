"""Scholar and OpenAlex adapters: fixtures only, no socket."""

from __future__ import annotations

import json
import unittest

from farfield.extras.livefeed import FreshWork
from farfield.extras.sources import merge_works, search_openalex, search_scholar

S2 = json.dumps(
    {
        "data": [
            {
                "paperId": "abc123",
                "title": "Wavelet trees meet pivots",
                "abstract": "We compress pivot histories.",
                "year": 2026,
                "venue": "SODA",
                "url": "https://example.test/s2",
                "externalIds": {"ArXiv": "2601.11111", "DOI": "10.1234/soda"},
            }
        ]
    }
).encode()

OA = json.dumps(
    {
        "results": [
            {
                "id": "https://openalex.org/W999",
                "display_name": "Wavelet trees meet pivots",
                "publication_year": 2026,
                "doi": "https://doi.org/10.1234/soda",
                "abstract_inverted_index": {"We": [0], "compress": [1], "histories": [2]},
                "primary_location": {"source": {"display_name": "SODA"}},
            }
        ]
    }
).encode()


class ScholarTests(unittest.TestCase):
    def test_arxiv_id_is_preferred_as_the_cite_id(self) -> None:
        works = search_scholar("wavelet pivot", fetcher=lambda url: S2)
        self.assertEqual(len(works), 1)
        self.assertEqual(works[0].cite_id(), "2601.11111")
        self.assertEqual(works[0].source, "s2")
        self.assertIn("compress", works[0].abstract)


class OpenAlexTests(unittest.TestCase):
    def test_inverted_index_becomes_an_abstract(self) -> None:
        works = search_openalex("wavelet pivot", fetcher=lambda url: OA)
        self.assertEqual(works[0].abstract, "We compress histories")
        self.assertTrue(works[0].cite_id().startswith("doi:"))


class MergeTests(unittest.TestCase):
    def test_the_same_title_collapses_to_the_richer_abstract(self) -> None:
        thin = FreshWork(title="Wavelet trees meet pivots", published="2026", work_id="s2:x")
        rich = FreshWork(
            title="Wavelet trees meet pivots",
            published="2026",
            arxiv_id="2601.11111",
            abstract="We compress pivot histories.",
            source="s2",
        )
        merged = merge_works([thin], [rich])
        self.assertEqual(len(merged), 1)
        self.assertIn("compress", merged[0].abstract)


if __name__ == "__main__":
    unittest.main()
