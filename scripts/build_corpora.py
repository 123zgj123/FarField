"""Build the three real citation corpora the audit and the anchors run on.

Each corpus is one OpenAlex topic search for pre-T roots, the works citing each
root, `referenced_works` reversed into forward "older -> newer" edges, and a
split at T=2017 into a walker graph and an evaluator graph.

`--replay` is the point of the `-p2` rebuild. The counterexample rule changed;
OpenAlex also changes month to month. Re-fetching would deliver both changes at
once and neither could be attributed, so the rebuild replays the pages the `-p1`
build recorded and the only thing that moves is the rule.

Recovered file. Its bytecode was never written -- a script run as `__main__`
leaves no `__pycache__` entry -- so it is checked by replaying the committed
corpora and reproducing their content digests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from farfield.extras.corpus import (
    CorpusError,
    CorpusSpec,
    build_corpus,
    replay_transport,
)

ROOT = Path(__file__).resolve().parents[1]

# Three unrelated fields on purpose. A value anchor measured on one topic is a
# measurement of that topic; the audit needs snapshots that do not share a
# literature, so a win cannot come from one corpus's peculiar shape.
SPECS = (
    CorpusSpec(
        topic="attention mechanism neural sequence model",
        corpus_id="attn-2017-p2",
    ),
    CorpusSpec(
        topic="cache oblivious algorithms memory hierarchy",
        corpus_id="cacheob-2017-p2",
    ),
    CorpusSpec(
        topic="sparse matrix factorization low rank approximation",
        corpus_id="sparse-2017-p2",
    ),
)


def replay_source(root: Path, corpus_id: str) -> Path:
    """The `-p1` directory whose recorded pages answer the `-p2` rebuild."""
    return root / corpus_id.removesuffix("-p2")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        metavar="DIR",
        help="rebuild offline from the pages recorded under DIR by an earlier build",
    )
    parser.add_argument(
        "--dest",
        default=str(ROOT / "corpora"),
        help="where the corpus directories are written (default: corpora/)",
    )
    args = parser.parse_args(argv)

    dest_root = Path(args.dest)
    replay_root = Path(args.replay) if args.replay else None
    failures = 0

    for spec in SPECS:
        transport = None
        if replay_root is not None:
            source = replay_source(replay_root, spec.corpus_id)
            if not (source / "manifest.json").is_file():
                print(f"SKIP  {spec.corpus_id}: no recorded build at {source}")
                failures += 1
                continue
            transport = replay_transport(source)

        try:
            manifest = build_corpus(
                spec, dest_root / spec.corpus_id, transport=transport
            )
        except CorpusError as error:
            # A blocked build is a report, not a crash: it names the capability
            # that was missing so the next run knows what to unlock.
            print(f"BLOCK {spec.corpus_id}: {error}")
            failures += 1
            continue

        counts = manifest["counts"]
        print(
            f"OK    {manifest['corpus_uid']}  "
            f"walker {counts['nodes_walker']}n/{counts['edges_walker']}e  "
            f"full {counts['nodes_full']}n/{counts['edges_full']}e  "
            f"confirmed {counts['confirmed']} (floor {counts['confirmed_citation_floor']})  "
            f"counterexamples {counts['counterexample_nodes']}"
        )
        print(f"      digests {json.dumps(manifest['digests'])}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
