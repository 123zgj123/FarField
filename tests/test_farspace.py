"""P0 of the v2 design: deterministic embeddings and the far-field sampler.

The exit criteria being enforced are the two the design states verbatim:
re-embedding the same input must reproduce the same vector digest, and the
alienness distribution of far jumps must be KS-separable from a near-field
control. Everything else in this file exists to make those two claims mean
something (unit norms, replayable jumps, honest neighborhoods).
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
from farfield.extras.farspace import (
    JUMP_OPERATORS,
    FarSpaceError,
    crossover_jump,
    ks_statistic,
    near_control,
    neighborhood,
    sample_jumps,
)

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


class NeighborhoodTests(unittest.TestCase):
    def test_the_near_set_is_dominated_by_the_seeds_cluster(self) -> None:
        """Dominance, not purity: random projection at 64 dimensions has
        real noise, and a test that demands a spotless neighborhood would be
        testing the fixture's luck rather than the geometry."""
        space = embed_nodes(two_cluster_nodes())
        hood = neighborhood(space, "attn:0", near_quantile=0.4)
        same = sum(1 for n in hood.near if n.startswith("attn:"))
        self.assertGreaterEqual(same / len(hood.near), 0.8)
        self.assertTrue(any(n.startswith("prot:") for n in hood.far))

    def test_a_text_seed_goes_through_the_same_door(self) -> None:
        space = embed_nodes(two_cluster_nodes())
        hood = neighborhood(
            space, "sparse attention transformer", near_quantile=0.4
        )
        same = sum(1 for n in hood.near if n.startswith("attn:"))
        self.assertGreaterEqual(same / len(hood.near), 0.8)


class SamplerTests(unittest.TestCase):
    def test_the_same_inputs_replay_the_same_jumps(self) -> None:
        space = embed_nodes(two_cluster_nodes())
        first = sample_jumps(space, "attn:0", rng_seed="s1", count=8)
        second = sample_jumps(space, "attn:0", rng_seed="s1", count=8)
        self.assertEqual(
            [j.to_dict() for j in first], [j.to_dict() for j in second]
        )

    def test_omitting_domain_quantile_keeps_retrieve_identical(self) -> None:
        space = embed_nodes(two_cluster_nodes())
        open_pool = sample_jumps(space, "attn:0", rng_seed="s1", count=4)
        also_open = sample_jumps(
            space, "attn:0", rng_seed="s1", count=4, domain_quantile=None
        )
        self.assertEqual(
            [j.to_dict() for j in open_pool], [j.to_dict() for j in also_open]
        )

    def test_a_domain_band_changes_retrieve_not_the_landing(self) -> None:
        space = embed_nodes(two_cluster_nodes())
        open_pool = sample_jumps(space, "attn:0", rng_seed="s1", count=4)
        banded = sample_jumps(
            space, "attn:0", rng_seed="s1", count=4, domain_quantile=0.45
        )
        self.assertEqual(
            [j.landing for j in open_pool], [j.landing for j in banded]
        )
        self.assertTrue(any(j.params.get("domain_quantile") == 0.45 for j in banded))

    def test_a_different_rng_seed_is_a_different_campaign(self) -> None:
        space = embed_nodes(two_cluster_nodes())
        first = sample_jumps(space, "attn:0", rng_seed="s1", count=4)
        second = sample_jumps(space, "attn:0", rng_seed="s2", count=4)
        self.assertNotEqual(
            [j.to_dict() for j in first], [j.to_dict() for j in second]
        )

    def test_the_operator_menu_is_cycled_not_sampled(self) -> None:
        space = embed_nodes(two_cluster_nodes())
        jumps = sample_jumps(space, "attn:0", rng_seed="s1", count=8)
        self.assertEqual(
            tuple(j.operator for j in jumps), JUMP_OPERATORS + JUMP_OPERATORS
        )

    def test_every_jump_retrieves_real_nodes_nearest_first(self) -> None:
        space = embed_nodes(two_cluster_nodes())
        for jump in sample_jumps(space, "attn:0", rng_seed="s1", count=4):
            self.assertEqual(len(jump.retrieved), 5)
            distances = [d for _, d in jump.retrieved]
            self.assertEqual(distances, sorted(distances))
            for node_id, _ in jump.retrieved:
                self.assertIn(node_id, space.vectors)

    def test_far_jumps_are_ks_separable_from_the_near_control(self) -> None:
        """P0 criterion two. If this fails, P0 has not been reached."""
        space = embed_nodes(two_cluster_nodes())
        far = [
            j.alienness
            for j in sample_jumps(space, "attn:0", rng_seed="s1", count=40)
        ]
        near = near_control(space, "attn:0", rng_seed="s1", count=40)
        self.assertGreaterEqual(ks_statistic(far, near), 0.5)

    def test_alienness_is_distance_to_the_neighborhood_not_the_seed(self) -> None:
        """A landing next to a near node is not alien, however far the seed."""
        space = embed_nodes(two_cluster_nodes())
        hood = neighborhood(space, "attn:0")
        self.assertGreater(len(hood.near), 0)
        some_near = space.vectors[hood.near[0]]
        self.assertAlmostEqual(hood.alienness(some_near, space), 0.0, places=9)


class CrossoverTests(unittest.TestCase):
    """Recombination must be a pure function of its parents, or lineage lies."""

    def setUp(self) -> None:
        self.space = embed_nodes(two_cluster_nodes())
        self.hood = neighborhood(self.space, "attn:0")

    def test_the_same_parents_recombine_identically(self) -> None:
        first = crossover_jump(self.space, self.hood, "prot:3", "prot:7")
        second = crossover_jump(self.space, self.hood, "prot:3", "prot:7")
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.operator, "crossover")

    def test_the_child_lands_between_its_parents(self) -> None:
        jump = crossover_jump(self.space, self.hood, "prot:3", "attn:5")
        to_a = cosine_distance(jump.landing, self.space.vectors["prot:3"])
        to_b = cosine_distance(jump.landing, self.space.vectors["attn:5"])
        apart = cosine_distance(
            self.space.vectors["prot:3"], self.space.vectors["attn:5"]
        )
        self.assertLess(to_a, apart)
        self.assertLess(to_b, apart)

    def test_lineage_is_in_the_params_and_alienness_is_accounted(self) -> None:
        jump = crossover_jump(self.space, self.hood, "prot:3", "prot:7")
        self.assertEqual(jump.params["parent_a"], "prot:3")
        self.assertEqual(jump.params["parent_b"], "prot:7")
        self.assertGreaterEqual(jump.alienness, 0.0)
        self.assertTrue(jump.retrieved)

    def test_identical_or_unknown_parents_are_refused(self) -> None:
        with self.assertRaises(FarSpaceError):
            crossover_jump(self.space, self.hood, "prot:3", "prot:3")
        with self.assertRaises(FarSpaceError):
            crossover_jump(self.space, self.hood, "prot:3", "not:a-node")


class KSTests(unittest.TestCase):
    def test_identical_samples_have_zero_distance(self) -> None:
        self.assertEqual(ks_statistic([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0)

    def test_disjoint_samples_have_distance_one(self) -> None:
        self.assertEqual(ks_statistic([1.0, 2.0], [5.0, 6.0]), 1.0)

    def test_an_empty_sample_is_refused(self) -> None:
        with self.assertRaises(FarSpaceError):
            ks_statistic([], [1.0])


if __name__ == "__main__":
    unittest.main()
