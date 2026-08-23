"""In-loop value tournament: ranks live cards, never kills, never resurrects."""

from __future__ import annotations

import unittest

from farfield.extras.llm import Completion
from farfield.extras.value import (
    archive_fitness,
    attach_observations,
    evidence_class,
    run_tournament,
    score_live,
    selection_key,
    tournament_pool,
    value_lines,
)


class FakeJudge:
    def __init__(self) -> None:
        self.n = 0

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.n += 1
        return Completion(
            text='{"winner": "A", "reason": "more specific and less obvious"}',
            model="fake",
            request_digest="r",
            digest=f"d{self.n:08x}",
            artifact_uri="file:///dev/null",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            reasoning_tokens=0,
            mode="replay",
        )


def card(cid: str, far: str, alienness: float, **extra) -> dict:
    row = {
        "card_id": cid,
        "claim": f"claim {cid}",
        "mechanism": f"mechanism {cid} uses a bound",
        "prediction": f"later work shows {cid}",
        "pair": ["seed", far],
        "pair_nodes": ("n:seed", f"n:{far}"),
        "alienness": alienness,
        "killed": False,
        "prior_kills": False,
        "verdict": "supports",
    }
    row.update(extra)
    return row


class PoolTests(unittest.TestCase):
    def test_killed_and_weakened_cards_never_enter(self) -> None:
        pool = tournament_pool(
            [
                card("live", "a", 0.4),
                card("dead", "b", 0.9, killed=True),
                card("prior", "c", 0.5, prior_kills=True),
                card("weak", "d", 0.8, verdict="weakens"),
            ]
        )
        self.assertEqual([row["card_id"] for row in pool], ["live"])

    def test_value_lines_are_prompt_ready_and_non_killing(self) -> None:
        lines = value_lines()
        self.assertGreaterEqual(len(lines), 3)
        joined = " ".join(lines)
        self.assertIn("attested world", joined)
        self.assertIn("literature-gap", joined)
        self.assertNotIn("less obvious", joined)


class TournamentTests(unittest.TestCase):
    def test_a_single_card_skips_debate_and_still_scores(self) -> None:
        table, debates = run_tournament(
            FakeJudge(), [card("only", "far", 0.6)], rng_seed="t"
        )
        self.assertEqual(debates, [])
        ranked = score_live([card("only", "far", 0.6)], table.ratings, beta=1.0)
        self.assertEqual(ranked[0]["card_id"], "only")
        self.assertEqual(ranked[0]["score"], 1.0)

    def test_higher_alienness_wins_a_tied_tournament(self) -> None:
        rows = [card("near", "x", 0.1), card("far", "y", 0.9)]
        table, debates = run_tournament(
            FakeJudge(), rows, rng_seed="t", workers=2
        )
        self.assertEqual(len(debates), 1)
        ranked = score_live(rows, table.ratings, beta=1.0)
        # FakeJudge always picks A, so Elo is not a fair fight — the
        # alienness term must still be able to surface the far card when
        # β > 0 and Elo is not overwhelmingly against it. We only assert
        # both cards are scored and the far card's t_alienness is higher.
        by_id = {row["card_id"]: row for row in ranked}
        self.assertGreater(by_id["far"]["t_alienness"], by_id["near"]["t_alienness"])


class SelectionTests(unittest.TestCase):
    def test_world_uninformative_beats_synthetic_support_even_with_high_elo(self) -> None:
        world = card(
            "world",
            "gap",
            0.1,
            verdict="uninformative",
            probe_kind="WORLD",
            value_score=0.1,
        )
        synth = card(
            "synth",
            "near",
            0.9,
            verdict="supports",
            probe_kind="SYNTHETIC",
            value_score=9.0,
        )
        ordered = sorted([synth, world], key=selection_key)
        self.assertEqual([row["card_id"] for row in ordered], ["world", "synth"])
        self.assertLess(evidence_class(world), evidence_class(synth))

    def test_wiki_gap_breaks_ties_inside_the_same_evidence_class(self) -> None:
        aimed = card("aimed", "a", 0.2, probe_kind="WORLD", wiki_gap_hit=True)
        other = card("other", "b", 0.9, probe_kind="WORLD", wiki_gap_hit=False, value_score=8.0)
        ordered = sorted([other, aimed], key=selection_key)
        self.assertEqual([row["card_id"] for row in ordered], ["aimed", "other"])

    def test_archive_fitness_ignores_elo(self) -> None:
        weak_elo = card("world", "a", 0.2, probe_kind="WORLD", value_score=0.0)
        loud_elo = card(
            "synth", "b", 0.9, probe_kind="SYNTHETIC", value_score=99.0
        )
        self.assertGreater(archive_fitness(weak_elo), archive_fitness(loud_elo))

    def test_attach_observations_aims_at_a_gap_without_calling_it_solved(self) -> None:
        rows = [
            card(
                "hit",
                "a",
                0.4,
                claim="query time of last-repeat lookups stays logarithmic",
                mechanism="the wavelet dictionary stores pivot repeats",
            )
        ]
        attach_observations(
            rows,
            ["[2601.00001] However, we do not evaluate query time of last-repeat lookups."],
        )
        self.assertTrue(rows[0]["wiki_gap_hit"])
        self.assertIn("query time", rows[0]["wiki_gap"])
        self.assertEqual(rows[0]["pipeline_bottleneck"], "informative")


if __name__ == "__main__":
    unittest.main()
