"""Host harvest: retrieve, verify, bank. Fake feeds stay trusted."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.knowledge import admit_works, feed_verifies, harvest_recent, harvest_survey
from farfield.extras.livefeed import CompositeFeed, FeedBlocked, FreshWork
from farfield.extras.snapshot import RecordingFeed, ReplayFeed
from farfield.extras.state import load, record_wiki, wiki_for
from farfield.extras.verifypapers import PENDING, UNVERIFIED, VERIFIED
from tests.test_mission import FakeFeed
from tests.test_verifypapers import ARXIV_EMPTY, ARXIV_HIT, S2_HIT, S2_MISS, route


class DeadArxiv:
    def recent_in_field(self, concepts, *, max_results=6):
        return FeedBlocked(attempted="arxiv", reason="HTTP 429 (retryable)")

    def survey_around(self, concept_a, concept_b, **kwargs):
        return FeedBlocked(attempted="arxiv", reason="HTTP 429 (retryable)")

    def pair_recently_combined(self, concept_a, concept_b, *, max_results=5):
        return FeedBlocked(attempted="arxiv", reason="HTTP 429 (retryable)")


S2_WORK = FreshWork(
    title="Scholar paper on succinct indexes",
    published="2026",
    work_id="s2:abc",
    source="s2",
    abstract="We compress genomic sequence collections.",
)
OA_WORK = FreshWork(
    title="OpenAlex paper on wavelet trees",
    published="2026",
    work_id="doi:10.1145/example",
    source="openalex",
    abstract="Wavelet trees for pivot logs.",
)


class CompositeFreshTests(unittest.TestCase):
    def test_arxiv_429_still_merges_scholar_and_openalex(self) -> None:
        feed = CompositeFeed(
            arxiv=DeadArxiv(),
            scholar=lambda query, max_results=5: [S2_WORK],
            openalex=lambda query, max_results=5: [OA_WORK],
        )
        works = feed.recent_in_field(("succinct data structure",))
        self.assertNotIsInstance(works, FeedBlocked)
        titles = {work.title for work in works}
        self.assertIn(S2_WORK.title, titles)
        self.assertIn(OA_WORK.title, titles)
        self.assertTrue(
            any("429" in row.get("reason", "") for row in feed.last_sources_blocked)
        )


class HarvestTests(unittest.TestCase):
    def test_fake_feed_is_trusted_and_does_not_hit_an_index(self) -> None:
        feed = FakeFeed()
        self.assertFalse(feed_verifies(feed))
        harvest = harvest_recent(feed, ("succinct data structure",))
        self.assertEqual(len(harvest.works), 3)
        self.assertIn("A very fresh preprint", {work.title for work in harvest.works})
        self.assertEqual(harvest.reports[0].layer, "trusted_feed")
        self.assertEqual(harvest.reports[0].status, VERIFIED)

    def test_unverified_rows_are_not_admitted(self) -> None:
        work = FreshWork(
            title="A paper that was never written",
            published="2026",
            arxiv_id="9999.99999",
        )
        harvest = admit_works(
            [work],
            verify=True,
            fetcher=route({"arxiv.org": ARXIV_EMPTY, "semanticscholar": S2_MISS}),
        )
        self.assertEqual(harvest.works, [])
        self.assertEqual(harvest.reports[0].status, UNVERIFIED)

    def test_pending_429_is_reported_and_not_banked(self) -> None:
        work = FreshWork(
            title="Succinct wavelet trees meet pivot rules",
            published="2026",
            arxiv_id="2608.01234",
        )
        harvest = admit_works(
            [work], verify=True, fetcher=route({"arxiv.org": 429})
        )
        self.assertEqual(harvest.works, [])
        self.assertEqual(harvest.reports[0].status, PENDING)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            added = record_wiki(
                path, "test-corpus", "succinct data structure", harvest.works
            )
            self.assertEqual(added, 0)
            self.assertEqual(
                wiki_for(load(path), "test-corpus", "succinct data structure"),
                [],
            )

    def test_verified_rows_are_banked(self) -> None:
        work = FreshWork(
            title="Succinct wavelet trees meet pivot rules",
            published="2026-08-12",
            arxiv_id="2608.01234",
        )
        harvest = admit_works(
            [work], verify=True, fetcher=route({"arxiv.org": ARXIV_HIT})
        )
        self.assertEqual(len(harvest.works), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            added = record_wiki(
                path, "test-corpus", "succinct data structure", harvest.works
            )
            self.assertEqual(added, 1)
            rows = wiki_for(load(path), "test-corpus", "succinct data structure")
            self.assertTrue(rows[0].get("verified"))

    def test_explicit_unverified_dict_is_not_banked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            added = record_wiki(
                path,
                "test-corpus",
                "succinct data structure",
                [
                    {
                        "title": "Forged",
                        "cite_id": "9999.99999",
                        "verified": False,
                    }
                ],
            )
            self.assertEqual(added, 0)

    def test_legacy_wiki_rows_without_verified_stay_citable(self) -> None:
        from farfield.extras.wiki import as_works, prompt_lines

        row = {
            "cite_id": "2001.00001",
            "title": "Old attested paper",
            "published": "2020-01-01",
            "abstract": "We encode pivot logs.",
        }
        self.assertEqual(len(as_works([row])), 1)
        self.assertTrue(prompt_lines([row]))

    def test_survey_harvest_uses_the_same_admission_gate(self) -> None:
        class Live:
            verify_literature = True

            def survey_around(self, a, b, **kwargs):
                return [
                    FreshWork(
                        title="Succinct wavelet trees meet pivot rules",
                        published="2026",
                        arxiv_id="2608.01234",
                    )
                ]

        harvest = harvest_survey(
            Live(),
            "succinct data structure",
            "pivot rule",
            fetcher=route({"arxiv.org": ARXIV_HIT, "semanticscholar": S2_HIT}),
        )
        self.assertEqual(len(harvest.works), 1)

    def test_unhashable_typeerror_is_not_a_signature_mismatch(self) -> None:
        class Boom:
            def survey_around(self, a, b, **kwargs):
                raise TypeError("unhashable type: 'dict'")

        with self.assertRaises(TypeError) as ctx:
            harvest_survey(Boom(), "a", "b")
        self.assertIn("unhashable", str(ctx.exception))

    def test_recording_feed_proxies_verify_literature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recording = RecordingFeed(FakeFeed(), Path(tmp))
            self.assertFalse(feed_verifies(recording))
            replay = ReplayFeed(Path(tmp))
            self.assertFalse(feed_verifies(replay))


class ResearchLitPluginTests(unittest.TestCase):
    def test_execute_admits_canned_works_without_a_socket(self) -> None:
        from farfield.extras.plugins import PluginHost, repo_root

        host = PluginHost.load(repo_root(), include_dsh=False)
        plugin = host.by_name("research-lit")
        self.assertIsNotNone(plugin)
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            result = json.loads(
                plugin.call(
                    "execute",
                    args={
                        "works": [
                            {
                                "title": "Canned attested preprint",
                                "arxiv_id": "2608.09999",
                                "abstract": "However, we do not evaluate query time.",
                            }
                        ],
                        "skip_verify": True,
                        "state_store": str(state),
                        "anchor": "succinct data structure",
                        "corpus": "test-corpus",
                        "seen_at": "2026-08-31",
                    },
                )
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["admitted"], 1)


if __name__ == "__main__":
    unittest.main()
