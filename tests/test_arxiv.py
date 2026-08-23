"""arXiv labels must trace back to a recorded abstract, and the split to a submission."""

from __future__ import annotations

import tempfile
import unittest
import urllib.parse
from pathlib import Path

from farfield.extras.arxiv import (
    MAX_GRAM,
    MIN_GRAM,
    NOUN_DISCIPLINE,
    STOPWORDS,
    ArxivError,
    ArxivSource,
    DISCIPLINE_BREAKERS,
    phrases,
)
from farfield.extras.corpus import Fetcher

HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <responseDate>2026-01-01T00:00:00Z</responseDate>
"""
FOOTER = "</OAI-PMH>\n"


def record(
    identifier: str,
    title: str,
    dates: tuple[str, ...],
    abstract: str = "",
    *,
    deleted: bool = False,
) -> str:
    """One OAI record. `dates` is ordered as arXiv serves it, newest first."""
    if deleted:
        return (
            f'<record><header status="deleted">'
            f"<identifier>{identifier}</identifier></header></record>"
        )
    dated = "".join(f"<dc:date>{value}</dc:date>" for value in dates)
    described = f"<dc:description>{abstract}</dc:description>" if abstract else ""
    return (
        "<record><header>"
        f"<identifier>{identifier}</identifier><datestamp>2026-01-01</datestamp>"
        "</header><metadata><dc>"
        f"<dc:title>{title}</dc:title>{dated}{described}"
        "</dc></metadata></record>"
    )


def page(records: str, token: str = "") -> bytes:
    resumption = f"<resumptionToken>{token}</resumptionToken>" if token else ""
    return (
        f"{HEADER}  <ListRecords>{records}{resumption}</ListRecords>\n{FOOTER}"
    ).encode("utf-8")


def error_page(code: str, text: str = "nothing here") -> bytes:
    return f'{HEADER}  <error code="{code}">{text}</error>\n{FOOTER}'.encode("utf-8")


class FakeOAI:
    """Serves a fixed list of pages in order and records what was asked."""

    def __init__(self, pages: list[bytes]) -> None:
        self.pages = pages
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        index = min(len(self.urls) - 1, len(self.pages) - 1)
        return self.pages[index]


class PhraseTest(unittest.TestCase):
    def test_a_single_word_is_not_a_concept(self) -> None:
        # A one-token label is a word, and a network of words is a network of the
        # language rather than of the field.
        for label in phrases("cache oblivious algorithm layout"):
            self.assertGreaterEqual(len(label.split()), MIN_GRAM)
            self.assertLessEqual(len(label.split()), MAX_GRAM)

    def test_a_phrase_may_not_cross_a_stopword(self) -> None:
        # "attention for sparse" is adjacency, not something an author wrote.
        found = phrases("attention for sparse matrix")
        self.assertIn("sparse matrix", found)
        self.assertNotIn("attention sparse", found)
        self.assertNotIn("attention for", found)

    def test_a_phrase_may_not_cross_punctuation_or_a_number(self) -> None:
        self.assertNotIn("attention sparse", phrases("dense attention. Sparse routing"))
        self.assertNotIn("layer layer", phrases("layer 12 layer"))

    def test_a_hyphen_reads_as_a_word_boundary(self) -> None:
        # Otherwise "cache-oblivious" and "cache oblivious" are two concepts, and
        # worse, they slug to the same node id and collide.
        self.assertEqual(
            phrases("cache-oblivious algorithms"), phrases("cache oblivious algorithms")
        )
        self.assertIn("stencil layout", phrases("stencil-layout"))

    def test_trailing_plurals_fold_together(self) -> None:
        self.assertEqual(
            phrases("sparse algorithms"), phrases("sparse algorithm")
        )

    def test_the_plural_fold_is_a_key_rather_than_english(self) -> None:
        # "oblivious" is not a plural, and it folds anyway. That is deliberate:
        # both spellings of anything fold the same way, so the key still groups
        # what it should, and a real stemmer would merge concepts on rules no
        # reader of the corpus could audit.
        self.assertIn("cache obliviou", phrases("cache oblivious algorithms"))

    def test_inline_math_is_dropped_rather_than_tokenised(self) -> None:
        self.assertNotIn("bound n", phrases("a bound of $O(n \\log n)$ on depth"))

    def test_every_label_is_a_phrase_of_the_text_it_came_from(self) -> None:
        # The whole point of a phrase counter over a language model: a reader can
        # check the labelling by searching the abstract.
        text = "Cache-oblivious stencil computations on modern memory hierarchies"
        folded = text.lower().replace("-", " ")
        for label in phrases(text):
            head = label.split()[0]
            self.assertIn(head[: len(head) - 1] if head.endswith("s") else head, folded)

    def test_the_gram_range_is_a_parameter_not_a_constant(self) -> None:
        wide = phrases("sparse low rank matrix approximation", min_gram=2, max_gram=4)
        self.assertIn("sparse low rank matrix", wide)
        self.assertNotIn(
            "sparse low rank matrix", phrases("sparse low rank matrix approximation")
        )


class DisciplineTest(unittest.TestCase):
    """The noun-phrase discipline: junk classes out, terms of art untouched.

    Every rejected example here is a label the undisciplined ds or cl corpus
    actually carried, and every kept one is a term the discipline must not
    cost. The lists in `arxiv.py` were curated against those vocabularies, so
    these are the receipts.
    """

    def test_verb_phrases_break_apart(self) -> None:
        found = phrases(
            "an algorithm computes the maximum matching", discipline=True
        )
        self.assertNotIn("algorithm compute", found)
        self.assertIn("maximum matching", found)

    def test_quantifier_fragments_break_apart(self) -> None:
        found = phrases("uses at least one of two attention mechanisms", discipline=True)
        self.assertNotIn("least one", found)
        self.assertNotIn("two attention", found)
        self.assertIn("attention mechanism", found)

    def test_generic_adjectives_and_adverbs_break_apart(self) -> None:
        for junk, text in (
            ("considerable attention", "received considerable attention lately"),
            ("adaptively chosen", "under adaptively chosen queries"),
            ("different dataset", "across different datasets"),
            ("best known", "improves the best known bound"),
            ("jointly optimize", "we jointly optimize both objectives"),
        ):
            self.assertNotIn(junk, phrases(text, discipline=True), junk)

    def test_terms_of_art_survive_the_discipline(self) -> None:
        for term, text in (
            ("worst case", "worst case complexity of the structure"),
            ("fully dynamic", "a fully dynamic connectivity structure"),
            ("expected running time", "expected running time analysis"),
            ("shared task", "results on the shared task benchmark"),
            ("adversarial training", "adversarial training of encoders"),
            ("byte pair encoding", "byte pair encoding vocabulary"),
            ("language understanding", "grounded language understanding models"),
            ("least square", "least squares regression estimate"),
        ):
            self.assertIn(term, phrases(text, discipline=True), term)

    def test_the_discipline_is_off_by_default(self) -> None:
        # The pre-discipline corpora rebuild under their old uids only if the
        # default path never sees the breaker list.
        self.assertIn("algorithm compute", phrases("an algorithm computes it"))

    def test_the_breaker_list_never_shadows_a_stopword(self) -> None:
        # An entry both lists carry would make removing it from one of them a
        # silent no-op; the discipline must stay auditable on its own.
        self.assertFalse(DISCIPLINE_BREAKERS & STOPWORDS)

    def test_a_disciplined_source_says_so_and_an_undisciplined_one_stays_silent(
        self,
    ) -> None:
        plain = ArxivSource(set_spec="cs:cs:DS", since="2012-01-01").describe()
        self.assertNotIn("label_discipline", plain)
        disciplined = ArxivSource(
            set_spec="cs:cs:DS", since="2012-01-01", discipline=NOUN_DISCIPLINE
        ).describe()
        self.assertEqual(disciplined["label_discipline"], NOUN_DISCIPLINE)
        self.assertIn(NOUN_DISCIPLINE, disciplined["label_rule"])
        del plain["label_rule"], disciplined["label_rule"], disciplined["label_discipline"]
        self.assertEqual(plain, disciplined)

    def test_an_unknown_discipline_name_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ArxivSource(set_spec="cs:cs:DS", since="2012-01-01", discipline="v9")


class HarvestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def harvest(
        self, pages: list[bytes], source: ArxivSource | None = None, T: int = 2017
    ):
        api = FakeOAI(pages)
        fetcher = Fetcher(self.root / "raw", transport=api)
        source = source or ArxivSource(set_spec="cs:cs:XX", since="2012-01-01")
        return source.harvest(fetcher, T), api, fetcher

    def test_the_split_reads_the_earliest_date_not_the_revision(self) -> None:
        # arXiv serves the revision dates too. A 2016 paper revised in 2020 is a
        # pre-T paper; reading the latest date would move it across the split and
        # hand the walker a work from the future.
        (pre, post), _, _ = self.harvest(
            [
                page(
                    record("oai:arXiv.org:1", "Early work revised late", ("2020-05-01", "2016-03-01"))
                    + record("oai:arXiv.org:2", "Genuinely later work", ("2019-01-01",))
                )
            ]
        )
        self.assertEqual([work.id for work in pre], ["oai:arXiv.org:1"])
        self.assertEqual([work.id for work in post], ["oai:arXiv.org:2"])
        self.assertEqual(pre[0].year, 2016)

    def test_a_work_older_than_the_window_is_dropped(self) -> None:
        # It is in a from=2012 harvest only because somebody revised it after
        # 2012, so it is a sample of unusually long-lived old papers rather than
        # of old papers. Keeping it would put a biased sliver of pre-history next
        # to a complete recent window and any early-versus-late comparison would
        # read the sampling as a trend.
        (pre, post), _, _ = self.harvest(
            [
                page(
                    record("oai:arXiv.org:old", "A 2003 paper revised in 2020", ("2020-01-01", "2003-01-01"))
                    + record("oai:arXiv.org:in", "Inside the window", ("2013-01-01",))
                )
            ]
        )
        self.assertEqual([work.id for work in pre], ["oai:arXiv.org:in"])
        self.assertEqual(post, [])

    def test_a_closing_year_filters_submissions_and_still_pages_past_them(self) -> None:
        # Closing the datestamp window instead would be cheaper and would drop
        # exactly the works revised after it, which is a sample of revised papers.
        source = ArxivSource(set_spec="cs:cs:XX", since="2012-01-01", until_year=2018)
        (pre, post), api, _ = self.harvest(
            [
                page(record("oai:arXiv.org:a", "Kept", ("2013-01-01",)), token="more"),
                page(record("oai:arXiv.org:b", "Too late", ("2019-01-01",))),
            ],
            source=source,
        )
        self.assertEqual([work.id for work in pre], ["oai:arXiv.org:a"])
        self.assertEqual(post, [])
        self.assertEqual(len(api.urls), 2)

    def test_paging_follows_the_resumption_token(self) -> None:
        (pre, _), api, _ = self.harvest(
            [
                page(record("oai:arXiv.org:a", "First page", ("2013-01-01",)), token="tok en/1"),
                page(record("oai:arXiv.org:b", "Second page", ("2014-01-01",))),
            ]
        )
        self.assertEqual({work.id for work in pre}, {"oai:arXiv.org:a", "oai:arXiv.org:b"})
        self.assertIn("resumptionToken=", api.urls[1])
        # The token is opaque and arrives with characters that would otherwise
        # end the query string or split a parameter.
        self.assertIn(urllib.parse.quote("tok en/1", safe=""), api.urls[1])

    def test_an_empty_first_page_is_refused_rather_than_returned_empty(self) -> None:
        # A set that matches nothing is a wrong set spec, not an empty subject.
        with self.assertRaises(ArxivError) as ctx:
            self.harvest([error_page("noRecordsMatch")])
        self.assertEqual(ctx.exception.record.missing_capability, "arxiv_oai")
        self.assertIn("noRecordsMatch", ctx.exception.record.unlock_condition)

    def test_an_exhausted_window_on_a_later_page_just_ends_the_harvest(self) -> None:
        (pre, _), api, _ = self.harvest(
            [
                page(record("oai:arXiv.org:a", "Only page", ("2013-01-01",)), token="more"),
                error_page("noRecordsMatch"),
            ]
        )
        self.assertEqual([work.id for work in pre], ["oai:arXiv.org:a"])
        self.assertEqual(len(api.urls), 2)

    def test_a_set_larger_than_the_page_cap_is_refused(self) -> None:
        # Silently returning the first N pages would be a partial slice presented
        # as a complete one, which is the one thing this catalogue was chosen for.
        source = ArxivSource(set_spec="cs:cs:XX", since="2012-01-01", max_pages=3)
        with self.assertRaises(ArxivError) as ctx:
            self.harvest(
                [page(record("oai:arXiv.org:a", "Endless", ("2013-01-01",)), token="more")],
                source=source,
            )
        self.assertEqual(ctx.exception.record.missing_capability, "arxiv_oai_pages")

    def test_a_deleted_or_untitled_record_is_skipped(self) -> None:
        (pre, _), _, _ = self.harvest(
            [
                page(
                    record("oai:arXiv.org:gone", "", (), deleted=True)
                    + record("oai:arXiv.org:undated", "No date at all", ())
                    + record("oai:arXiv.org:ok", "A real one", ("2013-01-01",))
                )
            ]
        )
        self.assertEqual([work.id for work in pre], ["oai:arXiv.org:ok"])

    def test_the_abstract_is_labelled_alongside_the_title(self) -> None:
        (pre, _), _, _ = self.harvest(
            [
                page(
                    record(
                        "oai:arXiv.org:a",
                        "Stencil computations",
                        ("2013-01-01",),
                        abstract="We give a cache oblivious layout.",
                    )
                )
            ]
        )
        self.assertIn("cache obliviou layout", pre[0].labels)
        self.assertIn("stencil computation", pre[0].labels)

    def test_the_title_alone_can_be_the_label_source(self) -> None:
        source = ArxivSource(
            set_spec="cs:cs:XX", since="2012-01-01", use_abstract=False
        )
        (pre, _), _, _ = self.harvest(
            [
                page(
                    record(
                        "oai:arXiv.org:a",
                        "Stencil computations",
                        ("2013-01-01",),
                        abstract="We give a cache oblivious layout.",
                    )
                )
            ],
            source=source,
        )
        self.assertNotIn("cache obliviou layout", pre[0].labels)
        self.assertIn("stencil computation", pre[0].labels)

    def test_a_title_never_crosses_into_the_abstract(self) -> None:
        # They are joined for extraction, so the join has to be a hard boundary
        # or the last title word pairs with the first abstract word.
        (pre, _), _, _ = self.harvest(
            [
                page(
                    record(
                        "oai:arXiv.org:a",
                        "Sparse routing",
                        ("2013-01-01",),
                        abstract="Dense layers cost more.",
                    )
                )
            ]
        )
        self.assertNotIn("routing dense", pre[0].labels)

    def test_every_page_is_recorded_as_xml_under_its_digest(self) -> None:
        # The recorded pages are what makes a rebuild a replay rather than a
        # second harvest of a catalogue that has moved.
        _, api, fetcher = self.harvest(
            [
                page(record("oai:arXiv.org:a", "First", ("2013-01-01",)), token="more"),
                page(record("oai:arXiv.org:b", "Second", ("2014-01-01",))),
            ]
        )
        self.assertEqual(len(fetcher.pages), len(api.urls))
        for entry in fetcher.pages:
            self.assertTrue((self.root / "raw" / entry["file"]).is_file())
            self.assertTrue(entry["file"].endswith(".xml"))

    def test_the_description_records_what_the_slice_is(self) -> None:
        described = ArxivSource(set_spec="cs:cs:LG", since="2012-01-01").describe()
        self.assertEqual(described["catalogue"], "arxiv-oai-pmh")
        self.assertEqual(described["set_spec"], "cs:cs:LG")
        for key in (
            "selection_rule",
            "window_rule",
            "completeness_rule",
            "label_rule",
        ):
            self.assertTrue(described[key].strip())
        self.assertIn("2012-01-01", described["completeness_rule"])


if __name__ == "__main__":
    unittest.main()
