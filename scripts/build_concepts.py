"""Build concept co-occurrence corpora from complete arXiv subject slices.

The citation corpora under `corpora/` are grown outward from a handful of roots,
which makes them blind in a way that matters for this measurement: a combination
made outside the expansion cone is not merely unobserved, it is unobservable, and
a bridge card is exactly the kind of card whose target lies out there. An OAI set
has no cone. Every submission to the subject over the period comes back, so a
pair being absent from the walker graph means the field had not made it.

Nodes are concepts rather than works, and an edge is co-occurrence of two labels
on one work, so novelty is local: whether a pair was already combined is read off
the pre-T side alone, without asking who cited whom.

Recovered file. Its bytecode was never written -- a script run as `__main__`
leaves no `__pycache__` entry -- so it is checked by replaying the recorded pages
of the `ds` slice and reproducing that corpus's content digest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from farfield.extras.arxiv import NOUN_DISCIPLINE, ArxivSource
from farfield.extras.concepts import ConceptSpec, build_concept_corpus
from farfield.extras.corpus import CorpusError, replay_transport

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "concepts"


@dataclass(frozen=True)
class Slice:
    spec: ConceptSpec
    source: ArxivSource

    # Which recorded harvest a `--replay` rebuild reads. Defaults to the
    # corpus's own directory; the v2 slices point at the v1 directories,
    # because a vocabulary rule is applied to pages, not fetched with them --
    # the same raw XML, read under a different discipline, is a new corpus.
    replay_source: str | None = None


# `since` is part of what the corpus is, not a knob: harvesting the same set from
# a different date yields a different pre-T vocabulary, so it belongs in the id
# via the recorded source description. The two dates differ because the slices
# were opened at different times, and moving `ds` to match `cl` would silently
# redefine a corpus that results already cite.
SLICES = (
    Slice(
        spec=ConceptSpec(
            topic="data structures and algorithms",
            corpus_id="ds-arxiv-concepts",
            min_df=10,
            confirmed_min_df=10,
        ),
        source=ArxivSource(set_spec="cs:cs:DS", since="2012-01-01"),
    ),
    # `T` differs from the ds slice because the harvest starts in mid-2015:
    # counterexample selection needs a concept's rise and fall visible inside
    # the pre-T years, and 2015..2017 is too short a window for either. The
    # first build attempt blocked on exactly that, which is why this corpus
    # existed only as recorded raw pages for a while. T=2020 gives five pre-T
    # years and five post-T years; realization rates are therefore not
    # comparable across the two corpora, only each arm against its own
    # matched chance within one corpus.
    Slice(
        spec=ConceptSpec(
            topic="computation and language",
            corpus_id="cl-arxiv-concepts",
            T=2020,
            min_df=10,
            confirmed_min_df=10,
        ),
        source=ArxivSource(set_spec="cs:cs:CL", since="2015-06-24"),
    ),
    # The v2 slices: the same recorded harvests, re-read under the noun-phrase
    # discipline. The v1 corpora carried labels like `algorithm compute` and
    # `least one` -- three of the six cl seeds were that kind of fragment --
    # which polluted both the menus the generator draws from and the pair
    # space the novelty oracle measures in. Same policy numbers otherwise, so
    # the only difference between v1 and v2 is the vocabulary rule, and the
    # comparison between them is a reading on that rule alone.
    Slice(
        spec=ConceptSpec(
            topic="data structures and algorithms",
            corpus_id="ds-arxiv-concepts-v2",
            min_df=10,
            confirmed_min_df=10,
        ),
        source=ArxivSource(
            set_spec="cs:cs:DS", since="2012-01-01", discipline=NOUN_DISCIPLINE
        ),
        replay_source="ds-arxiv-concepts",
    ),
    Slice(
        spec=ConceptSpec(
            topic="computation and language",
            corpus_id="cl-arxiv-concepts-v2",
            T=2020,
            min_df=10,
            confirmed_min_df=10,
        ),
        source=ArxivSource(
            set_spec="cs:cs:CL", since="2015-06-24", discipline=NOUN_DISCIPLINE
        ),
        replay_source="cl-arxiv-concepts",
    ),
    # The current-knowledge slice. The recorded ds harvest runs to 2026-08-14,
    # so the 2017 cut in the measurement corpora is a choice, not a limit: it
    # buys nine years of held-out future to score realization against. This
    # slice moves the cut to T=2025 for the *production* path — topic in, idea
    # out — where the question is not "would history have realized this" but
    # "is this actually open today". Vocabulary and walker edges therefore
    # cover 2012..2025; the full graph carries everything to the harvest end,
    # and the oracle for live idea generation reads that full graph, because
    # a combination the field made in 2026 is not open. Same discipline, same
    # recorded pages: the rebuild is offline replay, and refreshing it later
    # means re-harvesting the same set with a later end date.
    # `confirmed_min_df` is lower than the measurement slices': post-T here is
    # only the 2026 harvest months, so a df-10 floor leaves too few confirmed
    # pairs and the build blocks on exactly that. The confirmed set only
    # anchors the value scale; the production oracle reads the full graph.
    Slice(
        spec=ConceptSpec(
            topic="data structures and algorithms",
            corpus_id="ds-arxiv-concepts-2026",
            T=2025,
            min_df=10,
            confirmed_min_df=3,
        ),
        source=ArxivSource(
            set_spec="cs:cs:DS", since="2012-01-01", discipline=NOUN_DISCIPLINE
        ),
        replay_source="ds-arxiv-concepts",
    ),
)

BY_ID = {item.spec.corpus_id: item for item in SLICES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus_id",
        nargs="*",
        default=sorted(BY_ID),
        help=f"which slices to build (default: all of {', '.join(sorted(BY_ID))})",
    )
    parser.add_argument(
        "--replay",
        metavar="DIR",
        help="rebuild offline from the pages recorded under DIR/<corpus_id> by an earlier build",
    )
    parser.add_argument(
        "--dest",
        default=str(DEST),
        help="where the corpus directories are written (default: concepts/)",
    )
    args = parser.parse_args(argv)

    unknown = [name for name in args.corpus_id if name not in BY_ID]
    if unknown:
        parser.error(f"unknown slice(s): {', '.join(unknown)}")

    dest_root = Path(args.dest)
    failures = 0
    for name in args.corpus_id:
        item = BY_ID[name]
        transport = None
        if args.replay:
            source_dir = Path(args.replay) / (item.replay_source or name)
            if not (source_dir / "manifest.json").is_file():
                print(f"SKIP  {name}: no recorded build at {source_dir}")
                failures += 1
                continue
            transport = replay_transport(source_dir)

        try:
            manifest = build_concept_corpus(
                item.spec,
                dest_root / name,
                source=item.source,
                transport=transport,
            )
        except CorpusError as error:
            # A harvest that ran out of pages or came back short is a report
            # naming what would unlock it, not a traceback.
            print(f"BLOCK {name}: {error}")
            failures += 1
            continue

        counts = manifest["counts"]
        print(
            f"OK    {manifest['corpus_uid']}  "
            f"works {counts['pre_works']} pre / {counts['post_works']} post  "
            f"vocabulary {counts['vocabulary']}  "
            f"pairs {counts['pre_t_pairs']} pre / {counts['post_t_new_pairs']} new post  "
            f"confirmed {counts['confirmed']} (df floor {counts['confirmed_df_floor']})  "
            f"counterexamples {counts['counterexample_nodes']}"
        )
        print(f"      seeds {' '.join(manifest['seed_nodes'])}")
        print(f"      digests {json.dumps(manifest['digests'])}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
