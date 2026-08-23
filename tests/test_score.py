"""The score is a claim about how to spend selection slots; pin its behavior.

The β sweep's conclusions are only as good as the scorer's determinism and
its handling of ties and duplicates, so those are what these tests pin —
not the wisdom of any particular β, which is the experiment's job.
"""

import unittest

from farfield.extras.score import percentile_ranks, score_pool, select


def row(card_id, pair, plaus, alien):
    return {
        "card_id": card_id,
        "pair_nodes": list(pair),
        "plausibility_signal": plaus,
        "alienness": alien,
    }


class PercentileTests(unittest.TestCase):
    def test_empty_pool_ranks_nothing(self):
        self.assertEqual(percentile_ranks([]), [])

    def test_distinct_values_rank_in_order_and_inside_the_open_interval(self):
        ranks = percentile_ranks([3.0, 1.0, 2.0])
        self.assertEqual(sorted(range(3), key=lambda i: ranks[i]), [1, 2, 0])
        self.assertTrue(all(0.0 < r < 1.0 for r in ranks))

    def test_a_pool_of_identical_values_scores_half_everywhere(self):
        self.assertEqual(percentile_ranks([2.0, 2.0, 2.0]), [0.5, 0.5, 0.5])

    def test_ties_get_the_midrank_not_the_input_order(self):
        ranks = percentile_ranks([1.0, 2.0, 2.0, 3.0])
        self.assertEqual(ranks[1], ranks[2])
        self.assertLess(ranks[0], ranks[1])
        self.assertLess(ranks[2], ranks[3])


class ScorePoolTests(unittest.TestCase):
    def test_input_order_does_not_change_the_scores(self):
        rows = [
            row("c", ("n:1", "n:2"), 0.3, 0.9),
            row("a", ("n:3", "n:4"), 0.1, 0.5),
            row("b", ("n:5", "n:6"), 0.2, 0.7),
        ]
        forward = score_pool(rows)
        backward = score_pool(list(reversed(rows)))
        self.assertEqual(forward, backward)

    def test_beta_zero_is_plausibility_alone(self):
        scored = score_pool(
            [
                row("plausible", ("n:1", "n:2"), 0.9, 0.1),
                row("alien", ("n:3", "n:4"), 0.1, 0.9),
            ]
        )
        top = select(scored, beta=0.0, k=1)
        self.assertEqual(top[0].card_id, "plausible")

    def test_large_beta_hands_the_slot_to_alienness(self):
        scored = score_pool(
            [
                row("plausible", ("n:1", "n:2"), 0.9, 0.1),
                row("alien", ("n:3", "n:4"), 0.1, 0.9),
            ]
        )
        top = select(scored, beta=4.0, k=1)
        self.assertEqual(top[0].card_id, "alien")

    def test_negative_beta_penalizes_alienness(self):
        scored = score_pool(
            [
                row("near", ("n:1", "n:2"), 0.5, 0.1),
                row("far", ("n:3", "n:4"), 0.5, 0.9),
            ]
        )
        top = select(scored, beta=-1.0, k=1)
        self.assertEqual(top[0].card_id, "near")


class NLLComponentTests(unittest.TestCase):
    def test_a_full_nll_pool_blends_both_plausibility_components(self) -> None:
        # Graph signal says A; NLL surprise says B; the blend is a tie, so
        # neither component silently owns the axis.
        rows = [
            row("graphy", ("n:1", "n:2"), 0.9, 0.5) | {"mean_nll": 5.0},
            row("naturall", ("n:3", "n:4"), 0.1, 0.5) | {"mean_nll": 1.0},
        ]
        scored = {c.card_id: c for c in score_pool(rows)}
        self.assertEqual(
            scored["graphy"].t_plausibility, scored["naturall"].t_plausibility
        )

    def test_low_nll_ranks_as_more_plausible(self) -> None:
        rows = [
            row("surprising", ("n:1", "n:2"), 0.5, 0.5) | {"mean_nll": 5.0},
            row("natural", ("n:3", "n:4"), 0.5, 0.5) | {"mean_nll": 1.0},
        ]
        scored = {c.card_id: c for c in score_pool(rows)}
        self.assertGreater(
            scored["natural"].t_plausibility, scored["surprising"].t_plausibility
        )

    def test_one_missing_nll_reading_disables_the_component_for_the_pool(self) -> None:
        with_nll = [
            row("a", ("n:1", "n:2"), 0.9, 0.5) | {"mean_nll": 5.0},
            row("b", ("n:3", "n:4"), 0.1, 0.5) | {"mean_nll": 1.0},
        ]
        mixed = [dict(with_nll[0]), dict(with_nll[1]) | {"mean_nll": None}]
        graph_only = [row("a", ("n:1", "n:2"), 0.9, 0.5), row("b", ("n:3", "n:4"), 0.1, 0.5)]
        self.assertEqual(score_pool(mixed), score_pool(graph_only))
        self.assertNotEqual(score_pool(with_nll), score_pool(graph_only))


class SelectTests(unittest.TestCase):
    def test_score_ties_break_on_card_id_never_on_input_order(self):
        rows = [
            row("zeta", ("n:1", "n:2"), 0.5, 0.5),
            row("alpha", ("n:3", "n:4"), 0.5, 0.5),
        ]
        top = select(score_pool(rows), beta=1.0, k=1)
        self.assertEqual(top[0].card_id, "alpha")

    def test_a_repeated_pair_occupies_one_slot_not_two(self):
        rows = [
            row("first", ("n:1", "n:2"), 0.9, 0.9),
            row("repeat", ("n:2", "n:1"), 0.8, 0.8),
            row("other", ("n:3", "n:4"), 0.1, 0.1),
        ]
        top = select(score_pool(rows), beta=1.0, k=2)
        self.assertEqual([c.card_id for c in top], ["first", "other"])

    def test_asking_for_more_than_the_pool_returns_the_pool(self):
        rows = [row("only", ("n:1", "n:2"), 0.5, 0.5)]
        self.assertEqual(len(select(score_pool(rows), beta=1.0, k=10)), 1)


if __name__ == "__main__":
    unittest.main()
