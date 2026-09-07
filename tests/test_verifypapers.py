"""verify_papers: arXiv / CrossRef / Scholar admission. No sockets in tests."""

from __future__ import annotations

import json
import unittest
import urllib.error

from farfield.extras.livefeed import FreshWork
from farfield.extras.verifypapers import (
    ERROR,
    PENDING,
    UNVERIFIED,
    VERIFIED,
    admitted,
    verify_paper,
    verify_works,
)

ARXIV_HIT = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2608.01234v1</id>
    <title>Succinct wavelet trees meet pivot rules</title>
  </entry>
</feed>
"""
ARXIV_EMPTY = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""
CROSSREF_HIT = json.dumps(
    {"message": {"title": ["Succinct wavelet trees meet pivot rules"]}}
).encode()
S2_HIT = json.dumps(
    {
        "data": [
            {
                "title": "Succinct wavelet trees meet pivot rules",
                "externalIds": {"ArXiv": "2608.01234"},
            }
        ]
    }
).encode()
S2_MISS = json.dumps(
    {"data": [{"title": "Completely unrelated drought index"}]}
).encode()


def _http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "blocked", hdrs={}, fp=None)


def route(mapping: dict[str, bytes | int]):
    def fetch(url: str) -> bytes:
        for needle, body in mapping.items():
            if needle in url:
                if isinstance(body, int):
                    raise _http_error(url, body)
                return body
        raise _http_error(url, 404)

    return fetch


class VerifyPaperTests(unittest.TestCase):
    def test_an_arxiv_id_that_opens_is_verified(self) -> None:
        work = FreshWork(
            title="Succinct wavelet trees meet pivot rules",
            published="2026-08-12",
            arxiv_id="2608.01234v1",
        )
        report = verify_paper(work, fetcher=route({"arxiv.org": ARXIV_HIT}))
        self.assertEqual(report.status, VERIFIED)
        self.assertEqual(report.layer, "arxiv")

    def test_a_doi_opens_on_crossref(self) -> None:
        work = FreshWork(
            title="Succinct wavelet trees meet pivot rules",
            published="2026",
            work_id="doi:10.1145/example",
        )
        report = verify_paper(work, fetcher=route({"crossref.org": CROSSREF_HIT}))
        self.assertEqual(report.status, VERIFIED)
        self.assertEqual(report.layer, "crossref")

    def test_a_title_match_on_scholar_is_verified(self) -> None:
        work = FreshWork(
            title="Succinct wavelet trees meet pivot rules",
            published="2026",
            work_id="s2:abc",
        )
        report = verify_paper(work, fetcher=route({"semanticscholar": S2_HIT}))
        self.assertEqual(report.status, VERIFIED)
        self.assertEqual(report.layer, "scholar")

    def test_a_missing_index_row_is_unverified_not_pending(self) -> None:
        work = FreshWork(
            title="A paper that was never written",
            published="2026",
            arxiv_id="9999.99999",
        )
        report = verify_paper(
            work, fetcher=route({"arxiv.org": ARXIV_EMPTY, "semanticscholar": S2_MISS})
        )
        self.assertEqual(report.status, UNVERIFIED)
        self.assertNotIn(report.work, admitted([report]))

    def test_http_429_is_pending_not_a_hallucination(self) -> None:
        work = FreshWork(
            title="Succinct wavelet trees meet pivot rules",
            published="2026",
            arxiv_id="2608.01234",
        )
        report = verify_paper(work, fetcher=route({"arxiv.org": 429}))
        self.assertEqual(report.status, PENDING)
        self.assertEqual(admitted([report]), [])

    def test_http_500_is_pending(self) -> None:
        work = FreshWork(
            title="Succinct wavelet trees meet pivot rules",
            published="2026",
            work_id="doi:10.1145/example",
        )
        report = verify_paper(work, fetcher=route({"crossref.org": 503}))
        self.assertEqual(report.status, PENDING)

    def test_a_title_mismatch_on_a_named_id_falls_through_then_can_fail(self) -> None:
        work = FreshWork(
            title="A forged title about agent safety",
            published="2026",
            arxiv_id="2608.01234",
        )
        report = verify_paper(
            work,
            fetcher=route({"arxiv.org": ARXIV_HIT, "semanticscholar": S2_MISS}),
        )
        self.assertEqual(report.status, UNVERIFIED)

    def test_malformed_atom_is_error_then_scholar_may_save_it(self) -> None:
        work = FreshWork(
            title="Succinct wavelet trees meet pivot rules",
            published="2026",
            arxiv_id="2608.01234",
        )
        report = verify_paper(
            work,
            fetcher=route({"arxiv.org": b"<html>nope", "semanticscholar": S2_HIT}),
        )
        self.assertEqual(report.status, VERIFIED)
        self.assertEqual(report.layer, "scholar")

    def test_verify_works_keeps_only_admitted_rows(self) -> None:
        rows = [
            FreshWork(
                title="Succinct wavelet trees meet pivot rules",
                published="2026",
                arxiv_id="2608.01234",
            ),
            FreshWork(title="Never published", published="2026", work_id="s2:x"),
        ]
        reports = verify_works(
            rows,
            fetcher=route({"arxiv.org": ARXIV_HIT, "semanticscholar": S2_MISS}),
        )
        self.assertEqual([row.status for row in reports], [VERIFIED, UNVERIFIED])
        self.assertEqual(len(admitted(reports)), 1)
        self.assertNotEqual(ERROR, reports[0].status)


if __name__ == "__main__":
    unittest.main()
