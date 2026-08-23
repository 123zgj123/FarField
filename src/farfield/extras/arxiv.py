"""Concept corpora harvested from arXiv's OAI-PMH feed.

`concepts.py` says why a concept corpus measures novelty better than a citation
corpus. This module supplies one from a second catalogue, and the reason is not
only that OpenAlex answers a day's worth of queries and then stops.

**A slice with no expansion cone.** Every corpus built so far starts at a handful
of root works and expands outward, so membership is decided by reachability from
the roots. Whatever sits outside that cone cannot be observed at all, which is
why the bridge pairs could score zero without anyone being able to say whether
the proposals were wrong or the corpus simply could not see them. OAI-PMH does
no relevance selection: it serves *every* record in a set inside a datestamp
window. A harvest of `cs:cs:LG` is the whole subject, so a combination made
anywhere in that field is visible, and the research topic enters through the
seed node instead of through corpus membership.

**Why a `from` date is a completeness bound.** The OAI datestamp is when the
record last changed, not when the paper was submitted: a 2016 paper revised in
2020 is stamped 2020. That makes the window useless as a T split, and at the
same time makes it safe as a lower bound, because a datestamp is never earlier
than the submission it belongs to. Harvesting `from=X` therefore returns every
paper submitted on or after X, plus some older ones that were revised since. The
split is done on the submission date each record carries, never on the window.

Those older ones are then dropped rather than kept as a bonus. A paper from 2003
is in a `from=2012` harvest only because somebody revised it after 2012, and
papers that get revised eight years later are not a sample of 2003 -- they are a
sample of unusually long-lived 2003 papers. Keeping them would put a biased
sliver of pre-history next to a complete recent window, and any measurement that
compares early usage against late usage would read that sampling artefact as a
trend. So the slice is exactly the submissions in `[since, now]`, all of them.

**Where concepts come from.** arXiv publishes no controlled vocabulary finer
than its category list, so the labels are phrases lifted from the title and
abstract -- the same signal Science4Cast built its 64,719-concept network from.
The extractor is a phrase counter with a frequency band, not a language model:
it has to run in the kernel's dependency-free environment, and more importantly
every concept it emits has to trace back to a literal substring of a recorded
abstract, so a reader can check the labelling instead of trusting it.
"""

from __future__ import annotations

import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from ..models import BlockedRecord
from .concepts import Work
from .corpus import CorpusError, Fetcher

OAI = "http://export.arxiv.org/oai2"
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

OAI_PAUSE = 3.0

MAX_PAGES = 2000

MIN_TOKEN = 2
MAX_GRAM = 3

MIN_GRAM = 2

STOPWORDS = frozenset(
    """
    a about above across after against all almost along already also although always
    among an and another any are around as at
    be because been before behind being below beside besides between beyond both but by
    can cannot could
    did do does doing done due during
    each either else enough especially even every except
    far few for from further furthermore
    had has have having he hence her here hers him his how however
    i if in indeed inside instead into is it its itself
    just
    let like
    may me might more moreover most much must my
    near nearly neither no nor not now
    of off on once only onto or other others otherwise ought our ours out outside over own
    per perhaps
    rather
    same shall she should since so some still such
    than that the their theirs them then there thereby therefore these they this those
    though through throughout thus to together too toward towards
    under underneath until up upon us use used uses using usually
    very via
    was we well were what when whereas where wherein whether which while who whom whose
    why will with within without would
    yet you your yours
    al et etc ie eg
    achieve achieved achieves address addressed
    allow allows approach approaches
    base based
    consider considered
    demonstrate demonstrates demonstrated
    develop developed
    empirical empirically evaluate evaluated
    exist existing experiment experiments experimental
    find finding findings first
    give given
    however
    introduce introduces
    lead leads
    make makes many
    new novel
    obtain obtained
    paper papers present presents presented previous prior problem problems propose
    proposed proposes provide provides
    recent recently
    require requires respectively result resulting results
    second several show showed shows shown significant significantly study studies
    suggest suggests
    take taken third
    various
    work works
    """.split()
)


NOUN_DISCIPLINE = "noun-phrases-v1"

# The vocabulary discipline. The stopword list above already breaks runs at
# function words and a few reporting verbs, but the corpora built without more
# than that carry labels like `algorithm compute`, `considerable attention` and
# `least one`: verb phrases and quantifier fragments that no author would name
# as a concept. A phrase like that pollutes both ends of the measurement -- it
# widens the vocabulary the novelty oracle draws pairs from, and it hands the
# generator "concepts" that are not concepts.
#
# The discipline is a second, versioned set of run breakers, applied only when a
# source asks for it, because a rule change is a new corpus: the corpora already
# cited by results must rebuild byte-identically, so the flag is off by default
# and absent from their recorded source descriptions.
#
# Every entry is a decision against this corpus family's actual labels, not a
# grammar: `matching`, `training`, `encoding`, `computing`, `understanding` stay
# because they head real terms (maximum matching, adversarial training, byte
# pair encoding), while `matches`, `trained`, `compute` go because they only
# ever appear as verbs. Number words go even though that costs `one sided
# error`, because `least one`, `two attention mechanism` and their kin are far
# more numerous. `worst`, `expected`, `shared`, `fully` stay for worst case,
# expected time, shared task and fully dynamic.
_DISCIPLINE_VERBS = frozenset(
    """
    achieving aim aimed aiming aims allowed allowing analyze analyzed analyzes
    analyzing applied applies apply applying arise arises arising arose asked
    asking assume assumed assumes assuming attain attained attaining attains
    avoid avoided avoiding avoids became become becomes becoming believe
    believed believes believing bring bringing brings brought build builds built
    call called calling calls came capture captured captures capturing choose
    chooses choosing chose chosen combine combined combines combining come comes
    coming compare compared compares comparing compute computed computes conduct
    conducted conducting conducts consist consisted consisting consists
    construct constructed constructing constructs contain contained containing
    contains converge converged converges converging create created creates
    creating denote denoted denotes denoting depend depended depending depends
    derive derived derives deriving describe described describes describing
    designed designing determine determined determines determining discuss
    discussed discusses discussing employ employed employing employs enable
    enabled enables enabling encode encoded encodes enjoy enjoyed enjoys ensure
    ensured ensures ensuring establish established establishes establishing
    evaluates evaluating examine examined examines examining exceed exceeded
    exceeding exceeds exclude excluded excludes excluding exhibit exhibited
    exhibiting exhibits expect expecting expects explain explained explaining
    explains exploit exploited exploiting exploits explore explored explores
    exploring extend extended extending extends finds focused focuses focusing
    found gained gaining gets getting giving got grew grow growing grown grows
    handle handled handles handling helped helping helps hold holding holds held
    identified identifies identify identifying illustrate illustrated
    illustrates illustrating implied implies imply implying improve improved
    improves improving included includes including incorporate incorporated
    incorporates incorporating indicate indicated indicates indicating
    introduced introducing investigate investigated investigates investigating
    involve involved involves involving knew know knowing known knows leading
    led leveraged leverages leveraging lie lied lies lying made making
    maintain maintained maintaining maintains matched matches meant means
    measured measuring minimize minimized minimizes minimizing maximize
    maximized maximizes maximizing needed needing needs note noted notes noting
    observe observed observes observing obtaining obtains occur occurred
    occurring occurs offer offered offering offers optimize optimized optimizes
    optimizing outperform outperformed outperforming outperforms overcome
    overcomes overcoming perform performed performing performs produce produced
    produces producing prove proved proven proves proving provided providing
    reach reached reaches reaching receive received receives receiving reduce
    reduced reduces reducing relied relies rely relying remain remained
    remaining remains remove removed removes removing represent represented
    representing represents required requiring return returned returning
    returns reveal revealed revealing reveals satisfied satisfies satisfy
    satisfying saw see seeing seek seeking seeks seem seemed seems seen serve
    served serves serving share shares sharing showing sought solve solved
    solves studied studying suffer suffered suffering suffers suggested
    suggesting tackle tackled tackles tackling takes taking targeted targeting
    tend tended tending tends took train trained trains treat treated treating
    treats tried tries try trying turn turned turning turns understand
    understands understood utilize utilized utilizes utilizing want wanted
    wanting wants yield yielded yielding yields
    """.split()
)

_DISCIPLINE_QUANTIFIERS = frozenset(
    """
    one two three four five six seven eight nine ten
    little lot lots
    """.split()
)

_DISCIPLINE_GENERICS = frozenset(
    """
    additional art basic best better certain clear close considerable current
    different easier extensive faster harder important key main notable
    numerous past possible potential promising simpler slower stronger
    substantial technical tighter weaker worse
    """.split()
)

_DISCIPLINE_ADVERBS = frozenset(
    """
    adaptively approximately arbitrarily commonly conceptually considerably
    currently directly easily effectively efficiently essentially extremely
    highly jointly merely namely particularly previously purely relatively
    roughly simply substantially typically widely
    """.split()
)

DISCIPLINE_BREAKERS = (
    _DISCIPLINE_VERBS
    | _DISCIPLINE_QUANTIFIERS
    | _DISCIPLINE_GENERICS
    | _DISCIPLINE_ADVERBS
)


class ArxivError(CorpusError):
    pass


_MATH = re.compile(r"\$[^$]*\$")

_TOKEN = re.compile(r"[a-z][a-z0-9']*|[0-9][a-z0-9']*|[^\sa-z0-9]")

_JOINERS = str.maketrans("-/", "  ")


def _singular(word: str) -> str:
    """The same plural fold the topic router uses.

    It is a folding key, not English: `bias` folds to `bia`. That is harmless
    because both spellings fold the same way, and it is deliberately this crude
    because a real stemmer would merge concepts on rules no reader can audit.
    """
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _runs(
    text: str, *, breakers: frozenset[str] = frozenset()
) -> list[list[str]]:
    """Token runs an n-gram may not cross.

    A run ends at a stopword, a number, or any punctuation, because a phrase
    spanning `... attention. Sparse ...` is an artefact of adjacency rather than
    something an author wrote down. `breakers` adds a second, versioned set of
    cut points -- the noun-phrase discipline -- checked against the raw token,
    before the plural fold, because its entries are the inflections as written.
    """
    runs: list[list[str]] = []
    current: list[str] = []
    cleaned = _MATH.sub(" ", text.lower()).translate(_JOINERS)
    for token in _TOKEN.findall(cleaned):
        word = token.replace("'", "")
        if (
            not word
            or not word[0].isalpha()
            or len(word) < MIN_TOKEN
            or word in STOPWORDS
            or word in breakers
        ):
            if current:
                runs.append(current)
                current = []
            continue
        current.append(_singular(word))
    if current:
        runs.append(current)
    return runs


def phrases(
    text: str,
    *,
    min_gram: int = MIN_GRAM,
    max_gram: int = MAX_GRAM,
    discipline: bool = False,
) -> set[str]:
    """Every contiguous min_gram..max_gram token sequence inside a run.

    Interned, because a whole-subject harvest holds every record in memory at
    once and the same few million phrases recur across hundreds of thousands of
    abstracts. Without interning each recurrence is its own object and the
    harvest costs gigabytes for no reason.
    """
    out: set[str] = set()
    breakers = DISCIPLINE_BREAKERS if discipline else frozenset()
    for run in _runs(text, breakers=breakers):
        for size in range(min_gram, max_gram + 1):
            for start in range(len(run) - size + 1):
                out.add(sys.intern(" ".join(run[start : start + size])))
    return out


def _text(node: ET.Element, path: str) -> list[str]:
    return [
        (child.text or "").strip()
        for child in node.findall(path, NS)
        if (child.text or "").strip()
    ]


def _work(
    record: ET.Element,
    *,
    min_gram: int,
    max_gram: int,
    use_abstract: bool,
    discipline: bool = False,
) -> Work | None:
    header = record.find("oai:header", NS)
    if header is None or header.get("status") == "deleted":
        return None
    identifier = header.findtext("oai:identifier", default="", namespaces=NS).strip()
    metadata = record.find("oai:metadata", NS)
    if not identifier or metadata is None:
        return None
    titles = _text(metadata, ".//dc:title")
    if not titles:
        return None

    years = sorted(
        int(value[:4]) for value in _text(metadata, ".//dc:date") if value[:4].isdigit()
    )
    if not years:
        return None
    body = titles[0]
    if use_abstract:
        descriptions = _text(metadata, ".//dc:description")

        if descriptions:
            body = f"{titles[0]}. {max(descriptions, key=len)}"
    return Work(
        id=identifier,
        year=years[0],
        labels=frozenset(
            phrases(
                body, min_gram=min_gram, max_gram=max_gram, discipline=discipline
            )
        ),
    )


@dataclass(frozen=True)
class ArxivSource:
    """A complete arXiv subject slice, split on submission year.

    `since` is a completeness bound rather than a filter: see the module
    docstring. `set_spec` is an OAI set such as `cs:cs:LG`; `ListSets` on the
    endpoint enumerates them.
    """

    set_spec: str
    since: str
    until_year: int | None = None
    min_gram: int = MIN_GRAM
    max_gram: int = MAX_GRAM
    use_abstract: bool = True
    max_pages: int = MAX_PAGES

    # A discipline name or None. Named rather than boolean so that revising the
    # breaker list is forced to mint a new name, and with it new corpus uids;
    # None keeps the recorded descriptions of the pre-discipline corpora
    # byte-identical, which is what lets them rebuild under their old uids.
    discipline: str | None = None

    def __post_init__(self) -> None:
        if self.discipline not in (None, NOUN_DISCIPLINE):
            raise ValueError(
                f"unknown label discipline {self.discipline!r};"
                f" the only defined one is {NOUN_DISCIPLINE!r}"
            )

    def describe(self) -> dict[str, Any]:
        described = self._describe()
        if self.discipline:
            described["label_discipline"] = self.discipline
            described["label_rule"] += (
                f"; under the {self.discipline} discipline a run also breaks at"
                " a versioned list of verb inflections, number words, generic"
                " adjectives and discourse adverbs, so every label is"
                " noun-phrase shaped"
            )
        return described

    def _describe(self) -> dict[str, Any]:
        return {
            "catalogue": "arxiv-oai-pmh",
            "endpoint": OAI,
            "set_spec": self.set_spec,
            "since": self.since,
            "until_year": self.until_year,
            "metadata_prefix": "oai_dc",
            "min_gram": self.min_gram,
            "max_gram": self.max_gram,
            "use_abstract": self.use_abstract,
            "selection_rule": (
                f"every work in OAI set {self.set_spec}"
                f" submitted on or after {self.since}"
                + (f" and no later than {self.until_year}" if self.until_year else "")
                + "; no relevance ranking and no expansion from roots, so the slice"
                " is the whole subject over that period and a combination made"
                " anywhere in it is observable"
            ),
            "window_rule": (
                "the closing year is applied to the submission date, not to the"
                " datestamp, and the harvest still pages through every later"
                " record and discards it. Closing the datestamp window instead"
                " would be far cheaper and would drop exactly the works revised"
                " after it, which is a sample of revised papers rather than of"
                " papers"
                if self.until_year
                else "open-ended: every submission since the opening date"
            ),
            "completeness_rule": (
                f"a datestamp is never earlier than the submission it belongs to,"
                f" so a datestamp harvest from "
                f"{self.since} contains every work submitted since; works submitted"
                f" earlier also come back when they were revised since, and are"
                f" dropped, because they are a sample of long-lived old papers"
                f" rather than of old papers"
            ),
            "label_rule": (
                f"lowercased {self.min_gram}..{self.max_gram}-grams of the title"
                + (" and abstract" if self.use_abstract else "")
                + ", cut at stopwords, numbers and punctuation, with hyphens read"
                " as spaces and trailing plurals folded; every label is a phrase"
                " some recorded abstract contains"
            ),
        }

    def _fetcher(self, base: Fetcher) -> Fetcher:
        return Fetcher(
            raw_dir=base.raw_dir,
            transport=base.transport,
            pages=base.pages,
            api="arxiv_oai",
            host="export.arxiv.org",
            suffix="xml",
            pause=OAI_PAUSE if base.transport is None else 0.0,
        )

    def harvest(self, fetcher: Fetcher, T: int) -> tuple[list[Work], list[Work]]:
        pulled = self._fetcher(fetcher)
        url = f"{OAI}?" + urllib.parse.urlencode(
            {
                "verb": "ListRecords",
                "set": self.set_spec,
                "metadataPrefix": "oai_dc",
                "from": self.since,
            }
        )
        floor = int(self.since[:4])
        pre: list[Work] = []
        post: list[Work] = []
        for page in range(self.max_pages):
            root = ET.fromstring(pulled.get_bytes(url))
            error = root.find("oai:error", NS)
            if error is not None:
                if error.get("code") == "noRecordsMatch" and page:
                    break
                raise ArxivError(
                    BlockedRecord(
                        missing_capability="arxiv_oai",
                        attempted=f"ListRecords for {self.set_spec} since {self.since}",
                        unlock_condition=f"the endpoint answered {error.get('code')}: "
                        f"{(error.text or '').strip()}",
                    )
                )

            for record in root.findall(".//oai:record", NS):
                work = _work(
                    record,
                    min_gram=self.min_gram,
                    max_gram=self.max_gram,
                    use_abstract=self.use_abstract,
                    discipline=bool(self.discipline),
                )
                if work is None:
                    continue
                if work.year < floor or (
                    self.until_year and work.year > self.until_year
                ):
                    continue
                (pre if work.year <= T else post).append(work)
            token = root.find(".//oai:resumptionToken", NS)
            text = (token.text or "").strip() if token is not None else ""
            if not text:
                break
            url = f"{OAI}?verb=ListRecords&resumptionToken=" + urllib.parse.quote(
                text, safe=""
            )
        else:
            raise ArxivError(
                BlockedRecord(
                    missing_capability="arxiv_oai_pages",
                    attempted=f"harvest {self.set_spec} in {self.max_pages} pages",
                    unlock_condition="the set is larger than the page cap; raise"
                    " max_pages or harvest a narrower set",
                )
            )

        pre.sort(key=lambda work: work.id)
        post.sort(key=lambda work: work.id)
        return pre, post
