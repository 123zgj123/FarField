"""The real-corpus builder must not hand the walker the future it is predicting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.corpus import CorpusError, CorpusSpec, build_corpus, verify_corpus
from farfield.graph import load_graph
from farfield.timeslice import PINNED_T, load_heldout, load_walker_graph


def work(n: int, year: int, cited: int, references: list[int]) -> dict:
    return {
        "id": f"https://openalex.org/W{n}",
        "title": f"Work {n}",
        "publication_year": year,
        "cited_by_count": cited,
        "referenced_works": [f"https://openalex.org/W{r}" for r in references],
    }


class FakeOpenAlex:
    """Roots W1..W4 (pre-T), each cited by a pre-T and a post-T follower."""

    def __init__(self) -> None:
        self.roots = [work(n, 2010 + n, 500, []) for n in range(1, 5)]
        self.followers = {}
        for n in range(1, 5):
            self.followers[n] = [
                # A pre-T follower, so G_<=T has edges of its own to walk.
                work(100 + n, 2016, 300, [n]),
                # The post-T follower's citation count varies with n, so the
                # confirmed quantile has a spread to cut on rather than a tie.
                work(200 + n, 2020, 100 * n, [n, 100 + n]),
            ]
        self.urls = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        if "search=" in url:
            return json.dumps({"results": self.roots}).encode("utf-8")
        cited = int(url.split("cites:")[1].split("&")[0].lstrip("W"))
        return json.dumps({"results": self.followers[cited]}).encode("utf-8")


class CorpusBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self._tmp.name) / "corpus"
        self.addCleanup(self._tmp.cleanup)
        self.api = FakeOpenAlex()
        self.manifest = build_corpus(
            CorpusSpec(topic="a real topic", corpus_id="test-corpus", roots=4),
            self.dest,
            transport=self.api,
        )

    def test_the_walker_graph_carries_no_citation_counts(self) -> None:
        # Today's cited_by_count includes post-T citations, so a walker that
        # could read it would be reading the answer it is asked to predict.
        payload = json.loads((self.dest / "G_le_T.json").read_text(encoding="utf-8"))
        for node in payload["nodes"]:
            self.assertNotIn("cited_by_count", node)
            self.assertLessEqual(node["year"], PINNED_T)
        full = json.loads((self.dest / "G_full.json").read_text(encoding="utf-8"))
        self.assertTrue(any("cited_by_count" in node for node in full["nodes"]))

    def test_edges_point_forward_in_time(self) -> None:
        # OpenAlex gives referenced_works newer -> older, so every edge is
        # reversed on the way in; walking outgoing has to move toward later work.
        full = load_graph(self.dest / "G_full.json")
        years = {node_id: node["year"] for node_id, node in full.nodes.items()}
        for src, dst in full.edges:
            self.assertLessEqual(years[src], years[dst], f"{src} -> {dst}")

    def test_confirmed_nodes_are_post_t_and_absent_from_the_walker_graph(self) -> None:
        heldout = load_heldout(self.dest / "heldout_post_T.json")
        walker = load_walker_graph(self.dest / "G_le_T.json")
        full = load_graph(self.dest / "G_full.json")
        self.assertTrue(heldout.confirmed)
        for node_id in heldout.confirmed:
            self.assertNotIn(node_id, walker.nodes)
            self.assertGreater(full.nodes[node_id]["year"], PINNED_T)

    def test_confirmation_is_a_quantile_so_it_stays_selective(self) -> None:
        # A fixed citation count would confirm every post-T node in a busy topic
        # and none in a quiet one; the quantile keeps confirmation relative.
        full = load_graph(self.dest / "G_full.json")
        post_t = [
            node_id
            for node_id, node in full.nodes.items()
            if node["year"] > PINNED_T
        ]
        confirmed = load_heldout(self.dest / "heldout_post_T.json").confirmed
        self.assertLess(len(confirmed), len(post_t))
        self.assertIn("top", self.manifest["spec"]["confirmed_rule"])

    def test_a_confirmed_successor_is_reachable_from_a_walker_endpoint(self) -> None:
        # Without one, successor_hits_confirmed could never fire on this corpus.
        full = load_graph(self.dest / "G_full.json")
        walker = load_walker_graph(self.dest / "G_le_T.json")
        confirmed = load_heldout(self.dest / "heldout_post_T.json").confirmed
        reachable = [
            node_id
            for node_id in walker.nodes
            if any(child in confirmed for child in full.outgoing.get(node_id, ()))
        ]
        self.assertTrue(reachable)

    def test_every_fetched_page_is_recorded_with_its_digest(self) -> None:
        pages = self.manifest["pages"]
        self.assertEqual(len(pages), len(self.api.urls))
        for page in pages:
            self.assertTrue((self.dest / "raw" / f"{page['digest'][:16]}.json").is_file())

    def test_the_corpus_id_names_the_content_not_just_the_topic(self) -> None:
        # OpenAlex answers the same query differently months later, so a result
        # citing the topic id alone would not say which fetch it was measured on.
        uid = self.manifest["corpus_uid"]
        self.assertTrue(uid.startswith("test-corpus-"))
        self.assertNotEqual(uid, "test-corpus")
        walker = json.loads((self.dest / "G_le_T.json").read_text(encoding="utf-8"))
        self.assertIn(self.manifest["content_digest"], walker["snapshot_id"])

    def test_an_edited_corpus_refuses_to_verify(self) -> None:
        self.assertEqual(verify_corpus(self.dest)["corpus_uid"], self.manifest["corpus_uid"])
        heldout = self.dest / "heldout_post_T.json"
        payload = json.loads(heldout.read_text(encoding="utf-8"))
        payload["confirmed"] = payload["confirmed"][:-1]
        heldout.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(CorpusError) as ctx:
            verify_corpus(self.dest)
        self.assertEqual(ctx.exception.record.missing_capability, "corpus_integrity")

    def test_rebuilding_a_changed_corpus_in_place_is_refused(self) -> None:
        moved = FakeOpenAlex()
        moved.roots = moved.roots[:3]
        with self.assertRaises(CorpusError) as ctx:
            build_corpus(
                CorpusSpec(topic="a real topic", corpus_id="test-corpus", roots=4),
                self.dest,
                transport=moved,
            )
        self.assertEqual(ctx.exception.record.missing_capability, "corpus_immutability")

    def test_the_manifest_pins_the_thresholds_that_decide_the_corpus(self) -> None:
        spec = self.manifest["spec"]
        for key in (
            "T",
            "confirmed_quantile",
            "confirmed_min_citations",
            "counterexample_min_in_degree",
            "counterexample_quantile",
            "counterexample_min_age",
        ):
            self.assertIn(key, spec)
        self.assertIn("known_limitation", self.manifest)
        self.assertEqual(set(self.manifest["digests"]), {"G_full", "G_le_T", "heldout"})

    def test_a_counterexample_is_rare_enough_for_the_probe_to_mean_something(
        self,
    ) -> None:
        # A falsifier card passes only if its endpoint is a declared
        # counterexample, so a corpus where most nodes qualify makes it free.
        walker = load_graph(self.dest / "G_le_T.json")
        share = len(walker.counterexample_nodes) / len(walker.nodes)
        self.assertLess(share, 0.15)

    def test_a_young_dead_end_is_not_a_counterexample(self) -> None:
        # A node published just before T has had no time to be built on, so its
        # missing successor is its age rather than a dead end.
        walker = load_graph(self.dest / "G_le_T.json")
        cutoff = PINNED_T - self.manifest["spec"]["counterexample_min_age"]
        for node_id in walker.counterexample_nodes:
            self.assertLessEqual(walker.nodes[node_id]["year"], cutoff)

    def test_a_replayed_rebuild_reproduces_the_same_corpus(self) -> None:
        from farfield.extras.corpus import replay_transport

        twin = self.dest.parent / "twin"
        manifest = build_corpus(
            CorpusSpec(topic="a real topic", corpus_id="test-corpus", roots=4),
            twin,
            transport=replay_transport(self.dest),
        )
        self.assertEqual(manifest["content_digest"], self.manifest["content_digest"])


if __name__ == "__main__":
    unittest.main()
