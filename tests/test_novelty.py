"""Novelty scoring must be able to say drift is no better than chance."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.drift import generate
from farfield.graph import PINNED_K, load_graph
from farfield.novelty import (
    MIN_NOVEL_PAIRS,
    beats_chance,
    card_pairs,
    matched_chance,
    novel_pre_t,
    realized_post_t,
    score_arm_novelty,
)
from farfield.timeslice import PINNED_T, load_walker_graph

REPO = Path(__file__).resolve().parents[1]
G_LE_T = REPO / "tests" / "fixtures" / "G_le_T.json"
G_FULL = REPO / "tests" / "fixtures" / "G_full.json"
SEED = "beat a frozen mean baseline on synthetic linear holdout MSE"
SEED_NODE = "arxiv:1901.00001"


def graph_file(path: Path, nodes: list[dict], edges: list[list[str]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "snapshot_id": path.stem,
                "k": PINNED_K,
                "nodes": nodes,
                "edges": edges,
                "counterexample_nodes": [],
            }
        ),
        encoding="utf-8",
    )
    return path


class NoveltyDefinitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)

        nodes = [
            {"id": name, "title": name, "year": 2010}
            for name in ("a", "b", "c", "e")
        ] + [{"id": "d", "title": "d", "year": 2012}]
        self.walker = load_walker_graph(
            graph_file(
                root / "w.json",
                nodes,
                [["a", "b"], ["a", "d"], ["c", "d"]],
            )
        )
        self.full = load_graph(
            graph_file(
                root / "f.json",
                nodes + [{"id": "post", "title": "post", "year": 2020}],
                [
                    ["a", "b"],
                    ["a", "d"],
                    ["c", "d"],
                    ["a", "post"],
                    ["e", "post"],
                ],
            )
        )

    def test_a_direct_citation_is_not_a_new_combination(self) -> None:
        self.assertFalse(novel_pre_t(self.walker, "a", "b"))
        self.assertFalse(novel_pre_t(self.walker, "b", "a"))

    def test_a_pre_t_co_citation_is_not_a_new_combination(self) -> None:
        self.assertFalse(novel_pre_t(self.walker, "a", "c"))

    def test_a_never_combined_pair_is_novel(self) -> None:
        self.assertTrue(novel_pre_t(self.walker, "a", "e"))

    def test_realization_only_counts_post_t_citers(self) -> None:
        self.assertEqual(realized_post_t(self.full, "a", "e", t=PINNED_T), ("post",))
        # d cites both a and c, but it is pre-T, so it is not a realization.
        self.assertEqual(realized_post_t(self.full, "a", "c", t=PINNED_T), ())


class NoveltyArmTest(unittest.TestCase):
    def setUp(self) -> None:
        self.walker = load_walker_graph(G_LE_T)
        self.full = load_graph(G_FULL)

    def test_the_dump_arm_proposes_no_new_combination_by_construction(self) -> None:
        # The dump arm hands back the seed's own bibliography, so every pair it
        # proposes is joined by a direct citation and none of them is a new
        # combination. With no novel pair there is no realization rate to report.
        near = generate(SEED, self.walker, mode="dump_seed_refs", seed_node_id=SEED_NODE)
        scored = score_arm_novelty(
            "near", near.cards, SEED_NODE, self.walker, self.full
        )
        self.assertGreater(scored.n_pairs, 0)
        self.assertEqual(scored.n_novel, 0)
        self.assertIsNone(scored.realization_rate)

    def test_drift_pairs_include_the_two_endpoints_of_a_bridge_card(self) -> None:
        far = generate(SEED, self.walker, mode="drift", seed_node_id=SEED_NODE)
        bridges = [card for card in far.cards if card.mode == "bridge"]
        if not bridges:
            self.skipTest("this fixture seed produced no bridge card")
        pairs = card_pairs(bridges, SEED_NODE)
        self.assertTrue(any(SEED_NODE not in pair for pair in pairs))

    def test_chance_is_a_control_that_can_beat_the_arm(self) -> None:
        far = generate(SEED, self.walker, mode="drift", seed_node_id=SEED_NODE)
        scored = score_arm_novelty(
            "drift", far.cards, SEED_NODE, self.walker, self.full
        )
        if not scored.n_novel:
            self.skipTest("this fixture seed produced no novel pair")
        chance = matched_chance(
            scored.novel_pairs, self.walker, self.full, rng_seed="test", resamples=50
        )
        self.assertTrue(chance["measurable"])
        self.assertEqual(chance["resamples"], 50)
        # chance_rate is hits divided by resamples times pairs, so it is a
        # proportion: a value outside [0, 1] is a mis-count, not a result.
        self.assertGreaterEqual(chance["chance_rate"], 0.0)
        self.assertLessEqual(chance["chance_rate"], 1.0)

    def test_a_handful_of_pairs_yields_no_verdict(self) -> None:
        far = generate(SEED, self.walker, mode="drift", seed_node_id=SEED_NODE)
        scored = score_arm_novelty(
            "drift", far.cards, SEED_NODE, self.walker, self.full
        )
        chance = matched_chance(
            scored.novel_pairs, self.walker, self.full, rng_seed="test", resamples=50
        ) if scored.n_novel else {"measurable": False, "reason": "no novel pair"}
        verdict = beats_chance(scored, chance)
        self.assertFalse(verdict["decided"])
        self.assertLess(scored.n_novel, MIN_NOVEL_PAIRS)

    def test_an_arm_with_no_novel_pair_has_no_rate_rather_than_zero(self) -> None:
        scored = score_arm_novelty("empty", (), SEED_NODE, self.walker, self.full)
        self.assertEqual(scored.n_pairs, 0)
        self.assertIsNone(scored.realization_rate)
        self.assertFalse(
            beats_chance(scored, {"measurable": False, "reason": "nothing"})["decided"]
        )


if __name__ == "__main__":
    unittest.main()
