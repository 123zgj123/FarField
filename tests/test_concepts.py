"""The concept corpus must not let the walker read a combination it has to predict."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from farfield.extras.concepts import (
    MIN_COUNTEREXAMPLES,
    SEED_BAND,
    ConceptSpec,
    Work,
    build_concept_corpus,
    realization_years,
    verify_concepts,
)
from farfield.extras.corpus import CorpusError
from farfield.graph import PINNED_K, load_graph
from farfield.timeslice import PINNED_T, load_heldout, load_walker_graph

EARLY_YEARS = (2008, 2009, 2010, 2011, 2012, 2013, 2014)
RECENT_YEARS = (2015, 2016, 2017)
POST_YEARS = (2018, 2019, 2020)

GROUPS = 20
WORKS_PER_GROUP = 10
STEADY_PER_GROUP = 6
RECENT_WORKS = 100
VERBOSE_WORKS = 40
FILLERS = 24
FILLERS_PER_WORK = 18
POST_WORKS = 200
POST_ONLY = 60
POST_ONLY_SPAN = 12

# Coprime with the group size, so a post-T work's concepts come from three
# different groups and their pairing is new.
CROSS_GROUP_STRIDE = 37


def synthetic_works() -> tuple[list[Work], list[Work]]:
    """A corpus shaped to exercise the builder's own floors.

    The awkward parts are deliberate. Every stale concept is given exactly the
    same co-occurrence partners so that all of them share one degree: the
    counterexample cut is a quantile over the declining concepts' degrees, so a
    fixture whose degrees spread out would lose three quarters of them to that
    cut and starve the pool for reasons that have nothing to do with staleness.
    """
    pre: list[Work] = []

    # Early works. Each group of ten repeats one pair of soon-to-decline
    # concepts alongside six steady ones, and the pair appears nowhere else.
    for group in range(GROUPS):
        stale = (f"stale concept {2 * group:02d}", f"stale concept {2 * group + 1:02d}")
        steady = tuple(
            f"steady concept {group:02d} {index}" for index in range(STEADY_PER_GROUP)
        )
        for index in range(WORKS_PER_GROUP):
            pre.append(
                Work(
                    id=f"early:{group:02d}:{index}",
                    year=EARLY_YEARS[index % len(EARLY_YEARS)],
                    labels=frozenset(stale + steady),
                )
            )

    # Recent works carry the steady concepts only, so the stale ones decline to
    # nothing across the window rather than merely thinning out.
    steady_all = [
        f"steady concept {group:02d} {index}"
        for group in range(GROUPS)
        for index in range(STEADY_PER_GROUP)
    ]
    per_recent = len(steady_all) * 5 // RECENT_WORKS
    for index in range(RECENT_WORKS):
        start = (index * per_recent) % len(steady_all)
        labels = [steady_all[(start + step) % len(steady_all)] for step in range(per_recent)]
        pre.append(
            Work(
                id=f"recent:{index:03d}",
                year=RECENT_YEARS[index % len(RECENT_YEARS)],
                labels=frozenset(labels),
            )
        )

    # Verbose works exist to make the per-work label cap bind. `hub` is the most
    # frequent label any of them carries, and it is carried by nothing else.
    fillers = [f"filler concept {index:02d}" for index in range(FILLERS)]
    for index in range(VERBOSE_WORKS):
        start = (index * FILLERS_PER_WORK) % FILLERS
        labels = {fillers[(start + step) % FILLERS] for step in range(FILLERS_PER_WORK)}
        labels.add("hub concept")
        pre.append(
            Work(
                id=f"verbose:{index:02d}",
                year=RECENT_YEARS[index % len(RECENT_YEARS)],
                labels=frozenset(labels),
            )
        )

    post: list[Work] = []
    new_concepts = [f"post only concept {index:02d}" for index in range(POST_ONLY)]
    for index in range(POST_WORKS):
        # Deliberately across groups. Pre-T co-occurrence never leaves a group,
        # so these pairs are combinations the corpus makes only after T -- which
        # is what a walker graph built from post-T edges would leak, and what
        # this fixture would be unable to detect if the pairs stayed inside one
        # group.
        labels = {
            steady_all[(index + step * CROSS_GROUP_STRIDE) % len(steady_all)]
            for step in range(3)
        }
        for offset, name in enumerate(new_concepts):
            if index in {
                (offset * 3 + step) % POST_WORKS for step in range(POST_ONLY_SPAN)
            }:
                labels.add(name)
        post.append(
            Work(
                id=f"post:{index:03d}",
                year=POST_YEARS[index % len(POST_YEARS)],
                labels=frozenset(labels),
            )
        )
    return pre, post


class FakeSource:
    """A catalogue that hands over both sides of T without a network."""

    def __init__(self, pre: list[Work], post: list[Work]) -> None:
        self.pre = pre
        self.post = post
        self.harvested_with: list[int] = []

    def describe(self) -> dict[str, Any]:
        return {"catalogue": "synthetic", "selection_rule": "a hand-built fixture"}

    def harvest(self, fetcher: Any, T: int) -> tuple[list[Work], list[Work]]:
        self.harvested_with.append(T)
        return list(self.pre), list(self.post)


def spec(**overrides: Any) -> ConceptSpec:
    base = {
        "topic": "a synthetic subject",
        "corpus_id": "test-concepts",
        "min_df": 5,
        "confirmed_min_df": 5,
    }
    return ConceptSpec(**(base | overrides))


class ConceptCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.pre, cls.post = synthetic_works()
        cls.source = FakeSource(cls.pre, cls.post)
        cls.dest = cls.root / "corpus"
        cls.manifest = build_concept_corpus(spec(), cls.dest, source=cls.source)
        cls.walker = load_walker_graph(cls.dest / "G_le_T.json")
        cls.full = load_graph(cls.dest / "G_full.json")
        cls.heldout = load_heldout(cls.dest / "heldout_post_T.json")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_split_reads_the_year_the_source_reported(self) -> None:
        self.assertEqual(self.source.harvested_with, [PINNED_T])
        self.assertEqual(self.manifest["spec"]["T"], PINNED_T)

    def test_a_confirmed_concept_is_absent_from_the_walker_graph(self) -> None:
        # Confirmed is the target: a concept the field did not have at T. If one
        # were also a walker node the walker could propose it by name.
        self.assertTrue(self.heldout.confirmed)
        for node_id in self.heldout.confirmed:
            self.assertNotIn(node_id, self.walker.nodes)
            self.assertIn(node_id, self.full.nodes)
            self.assertGreater(self.full.nodes[node_id]["year"], PINNED_T)

    def test_every_walker_edge_was_made_by_a_pre_t_work(self) -> None:
        # A co-occurrence edge is dated by the work that asserts it, not by its
        # endpoints, so filtering nodes by year cannot produce this graph: two
        # old concepts first combined after T must not be joined here.
        titles = {
            node_id: self.full.nodes[node_id]["title"] for node_id in self.walker.nodes
        }
        vocabulary = set(titles.values())
        combined_before_t = set()
        for work in self.pre:
            names = sorted(work.labels & vocabulary)
            for left in names:
                for right in names:
                    if left != right:
                        combined_before_t.add((left, right))
        for src, dst in self.walker.edges:
            self.assertIn((titles[src], titles[dst]), combined_before_t)

    def test_a_post_t_only_combination_is_in_the_evaluator_graph_alone(self) -> None:
        # Without this the two graphs would be the same object and there would be
        # nothing for the evaluator to know that the walker does not.
        self.assertGreater(len(self.full.edges), len(self.walker.edges))

    def test_the_fixture_contains_a_combination_made_only_after_t(self) -> None:
        # A guard on the fixture rather than on the builder. If every post-T pair
        # of walker concepts had already been combined before T, then a walker
        # graph built from post-T edges would be identical to the correct one and
        # the leak test above would pass no matter what the builder did.
        walker_pairs = set(self.walker.edges)
        crossed = {
            (left, right)
            for left, right in self.full.edges
            if left in self.walker.nodes and right in self.walker.nodes
        }
        self.assertTrue(crossed - walker_pairs)

    def test_the_walker_graph_carries_no_post_t_frequency(self) -> None:
        payload = json.loads((self.dest / "G_le_T.json").read_text(encoding="utf-8"))
        for node in payload["nodes"]:
            self.assertNotIn("post_t_df", node)
            self.assertIn("pre_t_df", node)
        full = json.loads((self.dest / "G_full.json").read_text(encoding="utf-8"))
        self.assertTrue(any("post_t_df" in node for node in full["nodes"]))

    def test_counterexamples_are_readable_from_pre_t_data(self) -> None:
        # The falsifier target has to be something the walker could have seen;
        # a counterexample it cannot inspect is an unfalsifiable check.
        self.assertGreaterEqual(
            len(self.walker.counterexample_nodes), MIN_COUNTEREXAMPLES
        )
        for node_id in self.walker.counterexample_nodes:
            self.assertIn(node_id, self.walker.nodes)

    def test_a_counterexample_is_a_concept_whose_use_collapsed_not_a_rare_one(
        self,
    ) -> None:
        # Rarity before T is not evidence the field moved off a concept. The
        # rule demands real use in the early window and then a collapse, so a
        # concept nobody used much cannot qualify however flat its later rate is.
        recent_from = PINNED_T - self.manifest["spec"]["stale_years"] + 1
        early_use: dict[str, int] = {}
        recent_use: dict[str, int] = {}
        for work in self.pre:
            counter = early_use if work.year < recent_from else recent_use
            for name in work.labels:
                counter[name] = counter.get(name, 0) + 1
        titles = {
            node_id: self.full.nodes[node_id]["title"] for node_id in self.walker.nodes
        }
        for node_id in self.walker.counterexample_nodes:
            name = titles[node_id]
            self.assertGreaterEqual(
                early_use.get(name, 0), self.manifest["spec"]["min_df"]
            )
            self.assertLess(recent_use.get(name, 0), early_use.get(name, 0))

    def test_the_most_frequent_label_of_a_verbose_work_survives_the_cap(self) -> None:
        # `carried` keeps a work's most frequent labels, matching how the
        # vocabulary itself was chosen. Keeping the rarest instead would drop
        # `hub` from every work that carries it, leaving a concept the corpus
        # selected for being used with no edge showing it was ever used.
        hub = "concept:hub-concept"
        self.assertIn(hub, self.walker.nodes)
        self.assertTrue(self.walker.outgoing.get(hub))

    def test_the_per_work_label_cap_bounds_what_one_work_can_assert(self) -> None:
        # One verbose abstract carrying hundreds of concepts would assert tens of
        # thousands of pairs and dominate the edge set. A cap of 12 here touches
        # only the verbose works -- every other work carries fewer labels than
        # that -- so the pair count has to fall and nothing else may move.
        tighter = build_concept_corpus(
            spec(max_labels_per_work=12),
            self.root / "capped-per-work",
            source=FakeSource(self.pre, self.post),
        )
        self.assertLess(
            tighter["counts"]["pre_t_pairs"], self.manifest["counts"]["pre_t_pairs"]
        )
        self.assertEqual(
            tighter["counts"]["counterexample_nodes"],
            self.manifest["counts"]["counterexample_nodes"],
        )

    def test_seeds_come_from_the_middle_of_the_frequency_band(self) -> None:
        seeds = self.manifest["seed_nodes"]
        self.assertTrue(seeds)
        counts = sorted(self.full.nodes[n]["pre_t_df"] for n in self.walker.nodes)
        low = counts[int(SEED_BAND[0] * (len(counts) - 1))]
        high = counts[int(SEED_BAND[1] * (len(counts) - 1))]
        for node_id in seeds:
            self.assertIn(node_id, self.walker.nodes)
            self.assertGreaterEqual(self.full.nodes[node_id]["pre_t_df"], low)
            self.assertLessEqual(self.full.nodes[node_id]["pre_t_df"], high)

    def test_the_manifest_records_the_catalogue_it_asked(self) -> None:
        # Two corpora built from the same topic against different catalogues are
        # different corpora, so the source belongs in the record.
        self.assertEqual(self.manifest["source"]["catalogue"], "synthetic")
        self.assertEqual(set(self.manifest["digests"]), {"G_full", "G_le_T", "heldout"})
        self.assertIn("known_limitation", self.manifest)

    def test_the_corpus_id_names_its_content(self) -> None:
        uid = self.manifest["corpus_uid"]
        self.assertTrue(uid.startswith("test-concepts-"))
        self.assertIn(self.manifest["content_digest"][:12], uid)
        self.assertIn(self.manifest["content_digest"][:12], self.walker.snapshot_id)

    def test_an_edited_corpus_refuses_to_verify(self) -> None:
        twin = self.root / "edited"
        build_concept_corpus(
            spec(corpus_id="test-concepts"), twin, source=FakeSource(self.pre, self.post)
        )
        self.assertEqual(
            verify_concepts(twin)["content_digest"], self.manifest["content_digest"]
        )
        heldout = twin / "heldout_post_T.json"
        payload = json.loads(heldout.read_text(encoding="utf-8"))
        payload["confirmed"] = payload["confirmed"][:-1]
        heldout.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(CorpusError) as ctx:
            verify_concepts(twin)
        self.assertEqual(ctx.exception.record.missing_capability, "corpus_integrity")

    def test_rebuilding_a_different_corpus_in_place_is_refused(self) -> None:
        # The id embeds the content digest so that results can cite a corpus;
        # overwriting the directory would break that promise silently.
        fewer = FakeSource(self.pre, self.post[:-40])
        with self.assertRaises(CorpusError) as ctx:
            build_concept_corpus(spec(), self.dest, source=fewer)
        self.assertEqual(ctx.exception.record.missing_capability, "corpus_immutability")

    def test_a_rebuild_of_the_same_corpus_is_allowed(self) -> None:
        again = build_concept_corpus(spec(), self.dest, source=FakeSource(self.pre, self.post))
        self.assertEqual(again["content_digest"], self.manifest["content_digest"])


class ConceptCorpusRefusalTest(unittest.TestCase):
    """Each floor has to report what would unlock it, not raise a bare error."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.pre, self.post = synthetic_works()

    def build(self, dest: str, *, pre=None, post=None, **overrides: Any) -> dict:
        return build_concept_corpus(
            spec(**overrides),
            self.root / dest,
            source=FakeSource(
                self.pre if pre is None else pre, self.post if post is None else post
            ),
        )

    def refusal(self, dest: str, **kwargs: Any) -> str:
        with self.assertRaises(CorpusError) as ctx:
            self.build(dest, **kwargs)
        record = ctx.exception.record
        self.assertTrue(record.unlock_condition.strip())
        return record.missing_capability

    def test_too_few_pre_t_works_is_refused(self) -> None:
        self.assertEqual(
            self.refusal("thin", pre=self.pre[:100]), "pre_t_works"
        )

    def test_a_vocabulary_cap_that_starves_the_counterexample_pool_is_refused(
        self,
    ) -> None:
        # Capping the vocabulary keeps the most frequent labels, which are the
        # declining ones here; the pool they are cut from shrinks with them.
        self.assertEqual(
            self.refusal("capped", max_vocabulary=60), "counterexample_concepts"
        )

    def test_a_corpus_with_no_new_concepts_after_t_is_refused(self) -> None:
        # Nothing to predict: every post-T work reuses labels the walker has.
        recycled = [
            Work(id=work.id, year=work.year, labels=self.pre[0].labels)
            for work in self.post
        ]
        self.assertEqual(
            self.refusal("nothing-new", post=recycled), "confirmed_concepts"
        )

    def test_two_labels_that_share_a_node_id_are_refused(self) -> None:
        # `_slug` folds punctuation, so `brier score` and `brier-score` would
        # become one node and silently merge two concepts.
        # Both spellings have to land inside the frequency band, or they are cut
        # before the check and the corpus builds with neither of them.
        clashing = list(self.pre)
        for index, work in enumerate(clashing[:200]):
            if index % 20 > 1:
                continue
            extra = "brier score" if index % 20 else "brier-score"
            clashing[index] = Work(
                id=work.id, year=work.year, labels=work.labels | {extra}
            )
        self.assertEqual(
            self.refusal("clash", pre=clashing), "distinct_concept_ids"
        )


class RealizationYearTests(unittest.TestCase):
    """The lag reading re-derives the pair space; it must match the build's."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.pre, cls.post = synthetic_works()
        cls.dest = cls.root / "corpus"
        build_concept_corpus(spec(), cls.dest, source=FakeSource(cls.pre, cls.post))
        cls.years = realization_years(
            cls.dest, spec(), source=FakeSource(cls.pre, cls.post)
        )
        cls.walker = load_walker_graph(cls.dest / "G_le_T.json")
        cls.full = load_graph(cls.dest / "G_full.json")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_every_dated_pair_is_a_post_t_new_combination(self) -> None:
        walker_edges = {
            (src, dst) for src, targets in self.walker.outgoing.items() for dst in targets
        }
        full_edges = {
            (src, dst) for src, targets in self.full.outgoing.items() for dst in targets
        }
        self.assertTrue(self.years)
        for left, right in self.years:
            self.assertNotIn((left, right), walker_edges)
            self.assertIn((left, right), full_edges)

    def test_a_pair_made_pre_t_has_no_realization_year(self) -> None:
        walker_pair = next(
            (src, dst)
            for src, targets in self.walker.outgoing.items()
            for dst in targets
            if src < dst
        )
        self.assertNotIn(tuple(sorted(walker_pair)), self.years)

    def test_the_year_is_the_earliest_work_that_made_the_pair(self) -> None:
        for year in self.years.values():
            self.assertGreater(year, PINNED_T)
            self.assertLessEqual(year, max(POST_YEARS))
        self.assertIn(min(POST_YEARS), set(self.years.values()))

    def test_a_manifest_that_disagrees_on_pair_counts_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            twin = Path(tmp) / "corpus"
            build_concept_corpus(spec(), twin, source=FakeSource(self.pre, self.post))
            manifest_path = twin / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["counts"]["post_t_new_pairs"] += 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(CorpusError) as caught:
                realization_years(
                    twin, spec(), source=FakeSource(self.pre, self.post)
                )
            self.assertIn("drifted", str(caught.exception))


class PerCorpusCutTests(unittest.TestCase):
    """A slice may declare its own T; the file it loads still cannot.

    The cl slice harvests from mid-2015 and cuts at 2020, so the loaders
    take the declared cut as an argument. The protection must be unchanged:
    the expected year comes from the caller's declaration, and a graph or
    held-out file claiming a different year is refused exactly as before.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write_walker(self, year: int) -> Path:
        path = self.root / "G_le_T.json"
        path.write_text(
            json.dumps(
                {
                    "snapshot_id": "cut-test",
                    "k": PINNED_K,
                    "nodes": [{"id": "concept:a", "year": year, "title": "a"}],
                    "edges": [],
                }
            )
        )
        return path

    def write_heldout(self, t: int) -> Path:
        path = self.root / "heldout_post_T.json"
        path.write_text(json.dumps({"T": t, "confirmed": ["concept:z"]}))
        return path

    def test_the_default_cut_is_still_the_kernel_pin(self) -> None:
        path = self.write_walker(PINNED_T + 3)
        with self.assertRaises(ValueError):
            load_walker_graph(path)

    def test_a_declared_later_cut_admits_years_up_to_it(self) -> None:
        path = self.write_walker(PINNED_T + 3)
        graph = load_walker_graph(path, t=PINNED_T + 3)
        self.assertIn("concept:a", graph.nodes)

    def test_a_declared_cut_still_refuses_years_beyond_it(self) -> None:
        path = self.write_walker(PINNED_T + 4)
        with self.assertRaises(ValueError):
            load_walker_graph(path, t=PINNED_T + 3)

    def test_a_heldout_file_cannot_smuggle_its_own_year(self) -> None:
        # The declaration comes from the caller; a file whose T disagrees is
        # refused whichever side is larger.
        path = self.write_heldout(PINNED_T + 3)
        with self.assertRaises(ValueError):
            load_heldout(path)
        held = load_heldout(path, t=PINNED_T + 3)
        self.assertEqual(held.T, PINNED_T + 3)
        with self.assertRaises(ValueError):
            load_heldout(self.write_heldout(PINNED_T), t=PINNED_T + 3)


if __name__ == "__main__":
    unittest.main()
