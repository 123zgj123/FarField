"""The evolution loop's guarantees live here, not in the experiment script.

What P3 claims rests on four mechanisms: killed cards cannot re-enter, the
archive cannot fill with rephrasings of one pair, the judge's verdict is
parsed strictly and its presentation order is a seeded coin, and the trend
test is a real permutation test. Each is pinned against the cheapest way
it could rot.
"""

from __future__ import annotations

import json
import unittest

from farfield.extras.archive import (
    DebateRefused,
    EloTable,
    QDArchive,
    debate,
    failure_capsule,
    trend_test,
)
from farfield.extras.embed import embed_nodes


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


def row(card_id, far, alienness, *, near="attn:0", killed=False):
    return {
        "card_id": card_id,
        "pair_nodes": [near, far],
        "alienness": alienness,
        "killed": killed,
        "claim": f"claim of {card_id}",
        "mechanism": f"mechanism of {card_id}",
        "prediction": f"prediction of {card_id}",
    }


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.space = embed_nodes(two_cluster_nodes())
        cls.seeds = ("attn:0", "prot:0")

    def archive(self) -> QDArchive:
        return QDArchive(self.space, self.seeds)

    def test_a_killed_card_cannot_enter_no_matter_its_score(self) -> None:
        with self.assertRaises(ValueError):
            self.archive().admit(row("dead", "prot:3", 0.9, killed=True), 99.0)

    def test_a_duplicate_pair_is_refused_before_score_is_consulted(self) -> None:
        archive = self.archive()
        archive.admit(row("first", "prot:3", 0.9), 0.5)
        decision = archive.admit(row("rephrase", "prot:3", 0.9), 99.0)
        self.assertEqual(decision.outcome, "duplicate_pair")
        self.assertEqual(archive.occupancy()["pairs"], 1)

    def test_a_cell_incumbent_falls_only_to_a_strictly_higher_score(self) -> None:
        archive = self.archive()
        # Pick three far nodes the archive itself puts in one cell, so the
        # test exercises the incumbent rule rather than projection noise.
        target = archive.cell_of("prot:3", 0.9)
        same_cell = [
            f"prot:{i}"
            for i in range(3, 15)
            if archive.cell_of(f"prot:{i}", 0.9) == target
        ][:3]
        self.assertEqual(len(same_cell), 3)
        archive.admit(row("incumbent", same_cell[0], 0.9), 0.5)
        tie = archive.admit(row("tie", same_cell[1], 0.9), 0.5)
        self.assertEqual(tie.outcome, "kept_incumbent")
        better = archive.admit(row("better", same_cell[2], 0.9), 0.6)
        self.assertEqual(better.outcome, "displaced_incumbent")
        elites = archive.elites()
        self.assertEqual(elites[0]["card_id"], "better")
        self.assertNotIn("incumbent", [e["card_id"] for e in elites])

    def test_different_cells_hold_their_own_elites(self) -> None:
        archive = self.archive()
        near_home = archive.admit(row("near", "attn:5", 0.05), 0.4)
        far_out = archive.admit(row("far", "prot:5", 0.95), 0.3)
        self.assertEqual(near_home.outcome, "new_cell")
        self.assertEqual(far_out.outcome, "new_cell")
        self.assertNotEqual(near_home.cell, far_out.cell)
        self.assertEqual(archive.occupancy()["cells"], 2)

    def test_elites_rank_by_score_with_card_id_breaking_ties(self) -> None:
        archive = self.archive()
        archive.admit(row("zeta", "prot:3", 0.95), 0.5)
        archive.admit(row("alpha", "attn:5", 0.05), 0.5)
        self.assertEqual(
            [e["card_id"] for e in archive.elites()], ["alpha", "zeta"]
        )


class FakeJudge:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, purpose: str, system: str | None = None):
        self.prompts.append(prompt)

        class _C:
            text = self.texts[min(len(self.prompts) - 1, len(self.texts) - 1)]
            model = "fake"
            digest = f"digest{len(self.prompts)}"
            mode = "replay"
            finish_reason = "stop"

            def assert_usable(self) -> None:
                return None

        return _C()


class DebateTests(unittest.TestCase):
    def cards(self):
        return row("card_x", "prot:3", 0.9), row("card_y", "prot:4", 0.9)

    def test_the_winner_is_mapped_back_through_the_seeded_coin(self) -> None:
        a, b = self.cards()
        judge = FakeJudge(json.dumps({"winner": "A", "reason": "sharper"}))
        verdict = debate(judge, a, b, rng_seed="t")
        first_claim = f"claim of {verdict.presented_first}"
        self.assertIn(f"claim: {first_claim}", judge.prompts[0].split("Hypothesis B")[0])
        self.assertEqual(verdict.winner_card, verdict.presented_first)
        self.assertEqual(
            {verdict.winner_card, verdict.loser_card}, {"card_x", "card_y"}
        )

    def test_presentation_order_is_the_coin_not_the_argument_order(self) -> None:
        a, b = self.cards()
        judge = FakeJudge(json.dumps({"winner": "A", "reason": "r"}))
        forward = debate(judge, a, b, rng_seed="t")
        judge2 = FakeJudge(json.dumps({"winner": "A", "reason": "r"}))
        backward = debate(judge2, b, a, rng_seed="t")
        self.assertEqual(forward.presented_first, backward.presented_first)

    def test_an_off_schema_verdict_is_refused_not_guessed(self) -> None:
        a, b = self.cards()
        for bad in ("no json here", json.dumps({"winner": "C", "reason": "?"})):
            with self.assertRaises(DebateRefused):
                debate(FakeJudge(bad), a, b, rng_seed="t")

    def test_the_judge_never_sees_card_ids_or_scores(self) -> None:
        a = row("card_x", "prot:3", 0.9) | {
            "claim": "sparse retrieval",
            "mechanism": "shared bottleneck",
            "prediction": "a joint benchmark",
            "archive_score": 1.7,
        }
        b = row("card_y", "prot:4", 0.9) | {
            "claim": "protein prompts",
            "mechanism": "shared alphabet",
            "prediction": "a transfer result",
            "archive_score": 0.2,
        }
        judge = FakeJudge(json.dumps({"winner": "B", "reason": "r"}))
        debate(judge, a, b, rng_seed="t")
        self.assertNotIn("card_x", judge.prompts[0])
        self.assertNotIn("card_y", judge.prompts[0])
        self.assertNotIn("1.7", judge.prompts[0])


class EloTests(unittest.TestCase):
    def test_a_win_moves_ratings_symmetrically(self) -> None:
        table = EloTable()
        table.record("w", "l")
        self.assertAlmostEqual(table.rating("w") - 1000.0, 1000.0 - table.rating("l"))
        self.assertGreater(table.rating("w"), table.rating("l"))

    def test_an_upset_moves_more_than_an_expected_win(self) -> None:
        table = EloTable()
        table.ratings = {"favorite": 1200.0, "underdog": 800.0}
        table.record("underdog", "favorite")
        upset_gain = table.rating("underdog") - 800.0
        table2 = EloTable()
        table2.ratings = {"favorite": 1200.0, "underdog": 800.0}
        table2.record("favorite", "underdog")
        expected_gain = table2.rating("favorite") - 1200.0
        self.assertGreater(upset_gain, expected_gain)


class CapsuleTests(unittest.TestCase):
    def test_a_capsule_names_the_pair_and_the_killer(self) -> None:
        capsule = failure_capsule(
            {"pair": ["alpha", "beta"], "killed_by": ["endpoint_is_not_a_concept_hub"]}
        )
        self.assertIn("alpha", capsule["context"])
        self.assertIn("beta", capsule["context"])
        self.assertIn("endpoint_is_not_a_concept_hub", capsule["mechanism"])


class TrendTests(unittest.TestCase):
    def test_fewer_than_two_generations_is_not_a_trend(self) -> None:
        self.assertFalse(trend_test([(10, 5)], rng_seed="t")["measurable"])

    def test_a_flat_series_is_not_called_a_trend(self) -> None:
        result = trend_test([(20, 10), (20, 10), (20, 10)], rng_seed="t")
        self.assertGreater(result["p_value_two_sided"], 0.5)

    def test_a_steep_rise_is_detected(self) -> None:
        result = trend_test([(20, 1), (20, 10), (20, 19)], rng_seed="t")
        self.assertLess(result["p_value_two_sided"], 0.01)

    def test_a_steep_fall_is_a_finding_too(self) -> None:
        result = trend_test([(20, 19), (20, 10), (20, 1)], rng_seed="t")
        self.assertLess(result["p_value_two_sided"], 0.01)

    def test_the_same_seed_reproduces_the_same_p_value(self) -> None:
        first = trend_test([(12, 3), (12, 5), (12, 8)], rng_seed="t")
        second = trend_test([(12, 3), (12, 5), (12, 8)], rng_seed="t")
        self.assertEqual(first, second)

    def test_impossible_counts_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            trend_test([(5, 9), (5, 1)], rng_seed="t")


if __name__ == "__main__":
    unittest.main()
