"""Neighbourhood literature wiki: retrieved papers persist across missions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from farfield.extras.generate import generate_card
from farfield.extras.livefeed import FreshWork
from farfield.extras.prior import strongest_prior
from farfield.extras.state import load, record_wiki, wiki_for, wiki_works
from farfield.extras.wiki import (
    block,
    gap_lines,
    limitation_sentences,
    limitation_phrases,
    mechanism_phrases,
    merge_works,
    title_phrases,
    prompt_lines,
)
from tests.test_generate import FakeClient, LABELS, answer
from tests.test_mission import BabblingClient, FakeFeed, SchemingClient, TOPIC, run_mission


class WikiStoreTests(unittest.TestCase):
    def test_a_survey_is_banked_and_merged_by_cite_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            first = FreshWork(
                title="Wavelet trees for simplex pivot logs",
                published="2026-01-01",
                arxiv_id="2601.00001",
                abstract="We show that wavelet tree encodings work.",
            )
            added = record_wiki(
                path,
                "test-corpus",
                "succinct data structure",
                [first],
                pair=["succinct data structure", "pivot rule"],
                seen_at="2026-01-01",
            )
            self.assertEqual(added, 1)
            again = record_wiki(
                path,
                "test-corpus",
                "succinct data structure",
                [
                    FreshWork(
                        title="Wavelet trees for simplex pivot logs",
                        published="2026-01-01",
                        arxiv_id="2601.00001",
                        abstract="We show that wavelet tree encodings of simplex pivot logs give last-repeat queries in logarithmic time.",
                    )
                ],
                pair=["succinct data structure", "hash table"],
                seen_at="2026-08-18",
            )
            self.assertEqual(again, 0)
            rows = wiki_for(load(path), "test-corpus", "succinct data structure")
            self.assertEqual(len(rows), 1)
            self.assertGreater(len(rows[0]["abstract"]), 40)
            self.assertEqual(len(rows[0]["pairs"]), 2)
            works = wiki_works(load(path), "test-corpus", "succinct data structure")
            self.assertEqual(works[0].cite_id(), "2601.00001")

    def test_limitation_sentences_become_gaps_not_claims(self) -> None:
        hits = limitation_sentences(
            "We encode the log. However, we do not evaluate query time. "
            "Future work will scale the sketch."
        )
        self.assertGreaterEqual(len(hits), 2)
        self.assertTrue(any("However" in item for item in hits))
        lines = prompt_lines(
            [
                {
                    "cite_id": "2601.00001",
                    "title": "Wavelet trees for simplex pivot logs",
                    "published": "2026-01-01",
                    "abstract": "We encode pivot logs.",
                    "limitations": hits,
                }
            ]
        )
        self.assertTrue(any("Literature gap" in line for line in lines))
        self.assertTrue(gap_lines([{"cite_id": "2601.00001", "limitations": hits}]))

    def test_verified_titles_become_far_mechanisms_not_the_topic_object(self) -> None:
        phrases = mechanism_phrases(
            [
                {
                    "title": "Shadow replay gates for self-modifying validators",
                    "abstract": (
                        "We add shadow replay gates that intercept write-read "
                        "cycles of executable program state."
                    ),
                    "verified": True,
                },
                {
                    "title": "Code world models of executable program state",
                    "abstract": "A world model over program cells.",
                    "verified": True,
                },
            ],
            "code world models of executable program state as the world of an agent harness",
        )
        self.assertTrue(any("shadow replay" in item for item in phrases))
        self.assertFalse(any("code world" in item for item in phrases))
        self.assertFalse(any("program state" in item for item in phrases))

    def test_off_claim_titles_do_not_donate_mechanisms(self) -> None:
        phrases = mechanism_phrases(
            [
                {
                    "title": "Drought indexes for arid basins",
                    "abstract": "We map obesity prevalence with a drought index.",
                    "verified": True,
                }
            ],
            "code world models of executable program state as the world of an agent harness",
        )
        self.assertEqual(phrases, ())

    def test_limitation_phrases_are_a_separate_lens_from_titles(self) -> None:
        wiki = [
            {
                "title": "Shadow replay gates for self-modifying validators",
                "abstract": (
                    "We intercept write-read cycles of executable program "
                    "state."
                ),
                "limitations": [
                    "However, we do not evaluate frozen validators."
                ],
                "verified": True,
            }
        ]
        topic = (
            "code world models of executable program state as the world "
            "of an agent harness"
        )
        titles = title_phrases(wiki, topic)
        gaps = limitation_phrases(wiki, topic)
        self.assertTrue(any("shadow replay" in item for item in titles))
        self.assertFalse(any("shadow replay" in item for item in gaps))
        self.assertTrue(any("frozen" in item for item in gaps))

    def test_a_system_name_in_the_title_is_not_a_mechanism(self) -> None:
        # cwm-iclr2027 offered `darwin godel`, `arc agi`, `live swe` as far
        # mechanisms. Those are names the abstract only ever capitalises.
        from farfield.extras.wiki import described_methods, lowercase_attested

        topic = (
            "code world models of executable program state as the world "
            "of an agent harness"
        )
        title = "Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents"
        abstract = (
            "We introduce the Darwin Gödel Machine (DGM), a self-improving "
            "system that iteratively modifies its own code and validates each "
            "change on coding benchmarks over executable program state. "
            "The archive keeps open-ended evolution of agents alive. "
            "However, we do not evaluate on ARC-AGI or with Live-SWE-agent."
        )
        methods = described_methods(title, abstract, topic)
        joined = " | ".join(methods)
        self.assertNotIn("darwin", joined)
        self.assertNotIn("gödel", joined)
        self.assertNotIn("godel", joined)
        self.assertNotIn("arc agi", joined)
        self.assertNotIn("live swe", joined)
        self.assertTrue(
            any("self-improving" in item or "self improving" in item or "own code" in item
                or "open-ended" in item or "open ended" in item for item in methods),
            methods,
        )
        self.assertFalse(lowercase_attested("darwin gödel", abstract))
        self.assertTrue(lowercase_attested("self improving system", abstract))
        self.assertTrue(lowercase_attested("open ended evolution", abstract))
        # A metadata-only row with a Title Case title cannot tell a name from
        # a method and donates nothing rather than a name.
        self.assertEqual(described_methods(title, "", topic), ())
        # A lowercase-styled on-object title without an abstract still donates.
        lone = described_methods(
            "shadow replay gates over executable program state", "", topic
        )
        self.assertTrue(any("shadow replay" in item for item in lone), lone)

    def test_a_bibliographic_title_is_not_a_mechanism(self) -> None:
        phrases = mechanism_phrases(
            [
                {
                    "title": "A very fresh preprint",
                    "abstract": "We compress genomic sequence collections.",
                    "verified": True,
                }
            ],
            TOPIC,
        )
        self.assertEqual(phrases, ())

    def test_aimed_gap_is_not_a_solved_claim(self) -> None:
        from farfield.extras.wiki import aimed_gap

        gap = "However, we do not evaluate query time of last-repeat lookups."
        hit = aimed_gap(
            "query time of last-repeat lookups stays logarithmic",
            "the wavelet dictionary stores pivot repeats",
            [f"[2601.00001] {gap}"],
        )
        self.assertIn("query time", hit)
        miss = aimed_gap(
            "attention heads share a rotary basis",
            "the cache is quantized",
            [f"[2601.00001] {gap}"],
        )
        self.assertEqual(miss, "")

    def test_an_empty_wiki_leaves_the_generation_prompt_byte_identical(self) -> None:
        client = FakeClient(answer())
        generate_card(
            client,
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="analogy",
            alienness=0.5,
            label_to_node=LABELS,
        )
        self.assertNotIn("Literature wiki", client.prompts[0])

    def test_wiki_lines_reach_the_generation_prompt(self) -> None:
        client = FakeClient(answer())
        generate_card(
            client,
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="analogy",
            alienness=0.5,
            label_to_node=LABELS,
            wiki=prompt_lines(
                [
                    {
                        "cite_id": "2601.00001",
                        "title": "Wavelet trees for simplex pivot logs",
                        "published": "2026-01-01",
                        "abstract": "We encode pivot logs.",
                    }
                ]
            ),
        )
        self.assertIn("Literature wiki", client.prompts[0])
        self.assertIn("2601.00001", client.prompts[0])
        self.assertTrue(block(("a line",)).startswith("Literature wiki"))


class WikiMissionTests(unittest.TestCase):
    def test_fresh_verified_papers_are_in_the_same_mission_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = SchemingClient()
            events = list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=client,
                    feed=FakeFeed(),
                    state_store=Path(tmp) / "state.json",
                    policy_log=Path(tmp) / "log.json",
                    policy_file=Path(tmp) / "policy.json",
                    polish_rounds=0,
                    explore="fixed",
                    experiment_rounds=0,
                )
            )
            fresh = next(e for e in events if e["stage"] == "fresh")
            self.assertGreaterEqual(int(fresh.get("wiki_added") or 0), 1)
            self.assertTrue(
                any("Literature wiki" in p for p in client.prompts),
                client.prompts[0][:240] if client.prompts else "no prompts",
            )

    def test_retrieved_papers_reach_the_next_mission_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shared = {
                "state_store": Path(tmp) / "state.json",
                "policy_log": Path(tmp) / "log.json",
                "policy_file": Path(tmp) / "policy.json",
                "polish_rounds": 0,
                "explore": "fixed",
                "experiment_rounds": 0,
            }
            list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=BabblingClient(),
                    feed=FakeFeed(),
                    **shared,
                )
            )
            client = SchemingClient()
            second = list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=client,
                    feed=FakeFeed(dead=True),
                    **shared,
                )
            )
            wiki_state = next(e for e in second if e["stage"] == "state")
            self.assertTrue(wiki_state.get("wiki"))
            self.assertTrue(
                any("Literature wiki" in p for p in client.prompts),
                client.prompts[0][:240] if client.prompts else "no prompts",
            )

    def test_a_literature_gap_becomes_an_agenda_target(self) -> None:
        class GapFeed(FakeFeed):
            def survey_around(self, concept_a, concept_b, **kwargs):
                return [
                    FreshWork(
                        title="Wavelet trees for genomic sequence collections",
                        published="2026-01-01",
                        arxiv_id="2601.00001",
                        abstract=(
                            "We encode genomic sequence collections. However, "
                            "we do not evaluate query time on public traces."
                        ),
                    )
                ]

        with tempfile.TemporaryDirectory() as tmp:
            shared = {
                "state_store": Path(tmp) / "state.json",
                "policy_log": Path(tmp) / "log.json",
                "policy_file": Path(tmp) / "policy.json",
                "polish_rounds": 0,
                "explore": "fixed",
                "experiment_rounds": 0,
            }
            list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=SchemingClient(),
                    feed=GapFeed(),
                    **shared,
                )
            )
            second = list(
                run_mission(
                    TOPIC,
                    jumps=1,
                    candidates=1,
                    client=SchemingClient(),
                    feed=FakeFeed(dead=True),
                    **shared,
                )
            )
            agenda = next(e for e in second if e["stage"] == "agenda")
            self.assertTrue(
                any("literature gap" in str(t).lower() for t in agenda["targets"]),
                agenda["targets"],
            )


class WikiPriorTests(unittest.TestCase):
    def test_a_wiki_span_can_close_a_claim_the_new_survey_did_not_pull(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            claim = (
                "wavelet tree encodings of simplex pivot logs give last-repeat "
                "queries in logarithmic time"
            )
            record_wiki(
                path,
                "test-corpus",
                "succinct data structure",
                [
                    FreshWork(
                        title="Wavelet trees for simplex pivot logs",
                        published="2026-01-01",
                        arxiv_id="2601.00001",
                        abstract=claim + " on public traces.",
                    )
                ],
                seen_at="2026-01-01",
            )
            pool = merge_works([], wiki_works(load(path), "test-corpus", "succinct data structure"))
            hit = strongest_prior(claim, pool)
            self.assertIsNotNone(hit)
            self.assertTrue(hit.kills)


class MergeWorksTests(unittest.TestCase):
    def test_live_survey_wins_on_the_same_id(self) -> None:
        wiki = [
            FreshWork(
                title="Old title",
                published="2020-01-01",
                arxiv_id="2001.00001",
                abstract="short",
            )
        ]
        live = [
            FreshWork(
                title="New title",
                published="2020-01-01",
                arxiv_id="2001.00001",
                abstract="longer abstract that restates the claim",
            )
        ]
        merged = merge_works(live, wiki)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].title, "New title")

    def test_merge_works_accepts_wiki_dicts(self) -> None:
        live = [
            FreshWork(
                title="Live paper",
                published="2026-01-02",
                arxiv_id="2601.00002",
                abstract="live abstract",
            )
        ]
        wiki = [
            {
                "title": "Wiki paper",
                "published": "2026-01-01",
                "arxiv_id": "2601.00001",
                "abstract": "wiki abstract",
            }
        ]
        merged = merge_works(live, wiki)
        titles = {work.title for work in merged}
        self.assertEqual(titles, {"Live paper", "Wiki paper"})


if __name__ == "__main__":
    unittest.main()
