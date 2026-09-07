"""Deterministic embeddings and seed-field anchor selection.

Re-embedding the same input must reproduce the same vector digest, and
the anchors chosen for a topic must be the concepts the topic itself
names, not cosine neighbours wearing its clothes. The vector jump
sampler that used to be tested here (operators, near control, KS stake)
was removed with the graph-walk product path.
"""

from __future__ import annotations

import math
import unittest

from farfield.extras.embed import (
    EmbedError,
    cosine_distance,
    embed_nodes,
    features,
)
from farfield.extras.farspace import select_near_anchors

# Two lexical clusters far apart on purpose: attention papers and protein
# papers share no content words, so any sampler that cannot separate them
# has no business on a real corpus. Three anchor words are shared by every
# title in a cluster and one word is unique per node, so within-cluster
# similarity dominates the random-projection noise instead of hiding in it.
def two_cluster_nodes() -> dict[str, dict[str, object]]:
    nodes: dict[str, dict[str, object]] = {}
    for i in range(15):
        nodes[f"attn:{i}"] = {
            "id": f"attn:{i}",
            "title": f"sparse attention transformer variant{i}",
        }
    for i in range(15):
        nodes[f"prot:{i}"] = {
            "id": f"prot:{i}",
            "title": f"protein folding energy mutant{i}",
        }
    return nodes


class EmbeddingTests(unittest.TestCase):
    def test_the_same_corpus_reproduces_the_same_digest(self) -> None:
        """P0 criterion one, stated on the digest rather than on trust."""
        first = embed_nodes(two_cluster_nodes())
        second = embed_nodes(two_cluster_nodes())
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.vectors, second.vectors)

    def test_a_different_corpus_gets_a_different_digest(self) -> None:
        nodes = two_cluster_nodes()
        baseline = embed_nodes(nodes).digest
        nodes["extra"] = {"id": "extra", "title": "quantum error correction"}
        self.assertNotEqual(embed_nodes(nodes).digest, baseline)

    def test_vectors_are_unit_length(self) -> None:
        space = embed_nodes(two_cluster_nodes())
        for vector in space.vectors.values():
            self.assertAlmostEqual(
                math.sqrt(sum(v * v for v in vector)), 1.0, places=9
            )

    def test_shared_words_mean_nearer_vectors(self) -> None:
        """The space is lexical; the test asks exactly that much of it."""
        space = embed_nodes(two_cluster_nodes())
        within = cosine_distance(space.vectors["attn:0"], space.vectors["attn:1"])
        across = cosine_distance(space.vectors["attn:0"], space.vectors["prot:0"])
        self.assertLess(within, across)

    def test_an_unreadable_title_is_excluded_on_the_record(self) -> None:
        """The real corpora contain one non-English title; a lexical space
        has no location for it, but the exclusion must be a named fact that
        moves the digest, not a silent shrink of the far pool."""
        nodes = two_cluster_nodes()
        baseline = embed_nodes(nodes).digest
        nodes["bad"] = {"id": "bad", "title": "of the and"}
        space = embed_nodes(nodes)
        self.assertEqual(space.unreadable, ("bad",))
        self.assertNotIn("bad", space.vectors)
        self.assertNotEqual(space.digest, baseline)

    def test_a_corpus_with_nothing_readable_is_refused(self) -> None:
        with self.assertRaises(EmbedError):
            embed_nodes({"bad": {"id": "bad", "title": "of the and"}})

    def test_out_of_corpus_text_embeds_into_the_same_space(self) -> None:
        space = embed_nodes(two_cluster_nodes())
        vector = space.embed("sparse attention transformer")
        self.assertAlmostEqual(
            math.sqrt(sum(v * v for v in vector)), 1.0, places=9
        )
        near_attn = cosine_distance(vector, space.vectors["attn:0"])
        near_prot = cosine_distance(vector, space.vectors["prot:0"])
        self.assertLess(near_attn, near_prot)

    def test_unknown_words_push_a_seed_away_from_everything(self) -> None:
        """Out-of-vocabulary mass reads as distance, not as a silent drop.

        A seed half about molecules embedded against an attention corpus
        should sit far from both clusters; dropping the unknown words would
        instead fake a confident in-cluster location.
        """
        space = embed_nodes(two_cluster_nodes())
        known = space.embed("sparse attention transformer")
        mixed = space.embed("sparse attention for unseen molecules")
        to_cluster_known = cosine_distance(known, space.vectors["attn:0"])
        to_cluster_mixed = cosine_distance(mixed, space.vectors["attn:0"])
        self.assertLess(to_cluster_known, to_cluster_mixed)

    def test_features_reuse_the_concept_tokenizer(self) -> None:
        """A phrase that would be a concept node is also an embedding feature."""
        self.assertIn("sparse attention", features("Sparse attention at scale"))


class AnchorSelectionTests(unittest.TestCase):
    def test_phrase_overlap_beats_a_generic_lexical_neighbour(self) -> None:
        nodes = {
            "generic:0": {
                "id": "generic:0",
                "title": "golden ratio compression ratio bound",
            },
            "generic:1": {
                "id": "generic:1",
                "title": "golden ratio polynomial time bound",
            },
            "object:0": {
                "id": "object:0",
                "title": "formal verification of concurrent protocols",
            },
            "object:1": {
                "id": "object:1",
                "title": "model checking safety properties",
            },
        }
        for i in range(8):
            nodes[f"pad:{i}"] = {
                "id": f"pad:{i}",
                "title": f"unrelated protein folding mutant{i}",
            }
        space = embed_nodes(nodes)
        labels = {nid: str(row["title"]) for nid, row in nodes.items()}
        chosen = select_near_anchors(
            space,
            "formal verification of concurrent systems",
            labels,
            k=3,
        )
        picked = [labels[nid] for _, nid in chosen]
        self.assertTrue(
            any("verification" in label or "checking" in label for label in picked)
        )
        self.assertFalse(
            any(label.startswith("golden ratio") for label in picked[:2])
        )

    def test_covering_labels_are_not_padded_with_cosine_neighbours(self) -> None:
        nodes = {
            "object:0": {
                "id": "object:0",
                "title": "coding agent evaluation traces",
            },
            "metric:0": {"id": "metric:0", "title": "finite metric z-score distance"},
            "tree:0": {"id": "tree:0", "title": "balanced tree range count"},
        }
        for i in range(8):
            nodes[f"pad:{i}"] = {
                "id": f"pad:{i}",
                "title": f"unrelated protein folding mutant{i}",
            }
        space = embed_nodes(nodes)
        labels = {nid: str(row["title"]) for nid, row in nodes.items()}
        chosen = select_near_anchors(
            space,
            "coding-agent evaluation on attested tool-call traces",
            labels,
            k=6,
        )
        picked = [labels[nid] for _, nid in chosen]
        self.assertTrue(any("coding agent" in label for label in picked))
        self.assertFalse(any("finite metric" in label for label in picked))
        self.assertFalse(any("balanced tree" in label for label in picked))


if __name__ == "__main__":
    unittest.main()
