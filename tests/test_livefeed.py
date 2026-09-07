"""The live arXiv feed: advisory answers, honest blocks, no network in tests."""

from __future__ import annotations

import io
import unittest
import urllib.error
from unittest import mock

from farfield.extras.livefeed import ArxivFeed, FeedBlocked, survey_queries

ATOM_PAGE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2608.01234v1</id>
    <published>2026-08-12T17:59:02Z</published>
    <title>Succinct wavelet trees
      meet pivot rules</title>
    <summary>We compress pivot histories with a wavelet tree.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2608.00777v2</id>
    <published>2026-08-10T09:00:00Z</published>
    <title>A second fresh paper</title>
  </entry>
</feed>
"""


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def serving(body: bytes):
    return mock.patch(
        "farfield.extras.livefeed.urllib.request.urlopen",
        return_value=FakeResponse(body),
    )


def unreachable():
    return mock.patch(
        "farfield.extras.livefeed.urllib.request.urlopen",
        side_effect=urllib.error.URLError("no route to host"),
    )


class RecentInFieldTests(unittest.TestCase):
    def test_entries_come_back_as_dated_titles_newest_first(self) -> None:
        with serving(ATOM_PAGE):
            works = ArxivFeed().recent_in_field(("succinct data structure",))
        self.assertEqual(len(works), 2)
        self.assertEqual(
            works[0].line(), "Succinct wavelet trees meet pivot rules (2026-08-12)"
        )
        self.assertEqual(works[0].arxiv_id, "2608.01234v1")
        self.assertIn("wavelet tree", works[0].abstract)

    def test_a_dead_network_returns_a_block_that_names_the_attempt(self) -> None:
        with unreachable():
            result = ArxivFeed().recent_in_field(("succinct data structure",))
        self.assertIsInstance(result, FeedBlocked)
        self.assertIn("succinct data structure", result.attempted)

    def test_a_malformed_feed_is_a_block_not_a_crash(self) -> None:
        with serving(b"<html>rate limited</wrong>"):
            result = ArxivFeed().recent_in_field(("anything",))
        self.assertIsInstance(result, FeedBlocked)
        self.assertIn("malformed", result.reason)

    def test_http_429_names_the_code_as_retryable(self) -> None:
        err = urllib.error.HTTPError(
            "https://export.arxiv.org/api/query",
            429,
            "Too Many Requests",
            hdrs={},
            fp=None,
        )
        with mock.patch(
            "farfield.extras.livefeed.urllib.request.urlopen",
            side_effect=err,
        ):
            result = ArxivFeed().recent_in_field(("succinct data structure",))
        self.assertIsInstance(result, FeedBlocked)
        self.assertIn("429", result.reason)
        self.assertIn("retryable", result.reason)


class PairProbeTests(unittest.TestCase):
    def test_a_hit_reports_combined_with_evidence(self) -> None:
        with serving(ATOM_PAGE):
            probe = ArxivFeed().pair_recently_combined(
                "succinct data structure", "pivot rule"
            )
        self.assertTrue(probe["combined"])
        self.assertEqual(len(probe["evidence"]), 2)
        self.assertEqual(probe["evidence"][0]["published"], "2026-08-12")

    def test_an_empty_feed_means_the_pair_is_still_open(self) -> None:
        empty = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        with serving(empty):
            probe = ArxivFeed().pair_recently_combined("a", "b")
        self.assertEqual(probe, {"combined": False, "evidence": []})


class SurveyQueryTests(unittest.TestCase):
    def test_a_topic_does_not_query_the_far_concept_alone(self) -> None:
        topic = "code world models of executable program state"
        queries = survey_queries(
            "recursive algorithm",
            "stochastic reward",
            topic=topic,
        )
        joined = " ".join(queries)
        self.assertTrue(queries)
        self.assertFalse(
            any(
                "stochastic reward" in item and "code world" not in item.lower()
                and "program state" not in item.lower()
                and "executable" not in item.lower()
                for item in queries
            )
        )
        self.assertIn("stochastic reward", joined)

    def test_no_topic_keeps_the_pair_queries(self) -> None:
        queries = survey_queries("succinct data structure", "pivot rule")
        self.assertEqual(len(queries), 3)
        self.assertTrue(any("succinct data structure" in item and "pivot rule" not in item for item in queries))


if __name__ == "__main__":
    unittest.main()
