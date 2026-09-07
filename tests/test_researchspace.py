"""Trajectory-guided far-field search, not a concept-graph walk."""

from __future__ import annotations

import unittest

from farfield.extras.livefeed import FreshWork
from farfield.extras.researchspace import (
    JUMP_MOVES,
    SearchLedger,
    apply_policy,
    band_for_index,
    evaluate_landing,
    initial_state,
    observed_patterns,
    propose_jump,
    recover_trajectories,
    _usable_phrase,
)


def _papers() -> list[FreshWork]:
    return [
        FreshWork(
            title="Wavelet tree indexes for genomic collections",
            published="2022-03-01",
            arxiv_id="2203.11111v1",
            abstract=(
                "We index genomic sequence collections with wavelet trees. "
                "However, we do not support dynamic inserts."
            ),
        ),
        FreshWork(
            title="Dynamic succinct indexes for mutating genomes",
            published="2024-06-01",
            arxiv_id="2406.22222v1",
            abstract=(
                "We drop the static-collection assumption and allow inserts. "
                "However, we do not test compressed query logs."
            ),
            references=("2203.11111v1",),
        ),
    ]


TOPIC = "compress genomic sequence collections with succinct data structures"


class ResearchSpaceTests(unittest.TestCase):
    def test_user_input_defines_the_starting_position_not_a_graph_node(self) -> None:
        state = initial_state(
            "self-improving agents without frozen replay of program state"
        )
        self.assertTrue(state.object)
        self.assertTrue(any("program" in item or "agent" in item for item in state.object))
        self.assertTrue(state.tensions or state.summary)
        self.assertIn("object:", state.summary)

    def test_papers_become_a_development_trajectory(self) -> None:
        state = initial_state(TOPIC)
        trajectories = recover_trajectories(_papers(), TOPIC, state)
        self.assertTrue(trajectories)
        steps = trajectories[0].steps
        self.assertGreaterEqual(len(steps), 2)
        self.assertIn("wavelet", steps[0].mechanism)
        self.assertTrue(steps[0].new_problem)
        self.assertTrue(steps[1].inherited or steps[1].limitation_addressed)
        self.assertTrue(trajectories[0].patterns)

    def test_first_jump_applies_an_observed_move_instead_of_linear_extension(self) -> None:
        state = initial_state(TOPIC)
        trajectories = recover_trajectories(_papers(), TOPIC, state)
        proposal = propose_jump(
            index=0, state=state, trajectories=trajectories
        )
        self.assertEqual(proposal.band, "early")
        self.assertIn(proposal.move, JUMP_MOVES)
        self.assertTrue(proposal.far_labels)
        latest = trajectories[0].latest_mechanism
        self.assertFalse(
            set(proposal.far_labels) <= {latest},
            proposal.far_labels,
        )
        joined = " ".join(proposal.far_labels)
        self.assertTrue(
            "wavelet" in joined or "dynamic" in joined or "query" in joined or "insert" in joined,
            proposal.far_labels,
        )
        chain = " ".join(proposal.reasoning)
        self.assertIn("Existing knowledge", chain)
        self.assertIn("Jump operation", chain)
        self.assertEqual(proposal.support, "from_trajectory")
        self.assertNotEqual(proposal.novelty, "linear_extension")

    def test_empty_wiki_is_an_empty_menu_not_a_graph_fallback(self) -> None:
        topic = (
            "code world models of executable program state as the world "
            "of an agent harness"
        )
        state = initial_state(topic)
        proposal = propose_jump(index=0, state=state, trajectories=())
        self.assertEqual(proposal.far_labels, ())
        self.assertEqual(proposal.support, "none")

    def test_a_verified_method_title_lands_as_a_trajectory_mechanism(self) -> None:
        topic = (
            "code world models of executable program state as the world "
            "of an agent harness"
        )
        papers = [
            FreshWork(
                title="Shadow replay gates for self-modifying validators",
                published="2026-08-15",
                arxiv_id="2608.11111v1",
                abstract=(
                    "We add shadow replay gates that intercept write-read "
                    "cycles of executable program state. However, we do "
                    "not test replay on frozen validators."
                ),
            )
        ]
        state = initial_state(topic)
        trajectories = recover_trajectories(papers, topic, state)
        proposal = propose_jump(
            index=0, state=state, trajectories=trajectories
        )
        joined = " ".join(proposal.far_labels)
        self.assertIn("shadow replay", joined)
        self.assertNotIn("edge set", joined)
        self.assertNotIn("min-cut", joined)

    def test_a_named_system_is_not_a_mechanism_and_a_singleton_is_not_a_move(self) -> None:
        """Regression for cwm-iclr2027's menu.

        Every 'trajectory' was one paper whose mechanism was a title
        bigram (`darwin godel`, `arc agi`, `live swe`) and whose move was
        booked as `migrate_setting`. Names have no lever; a lone paper has
        no observed move.
        """
        topic = (
            "code world models of executable program state as the world "
            "of an agent harness"
        )
        papers = [
            FreshWork(
                title="Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents",
                published="2025-05-29",
                arxiv_id="2505.22954v3",
                abstract=(
                    "We introduce the Darwin Gödel Machine (DGM), a self-improving "
                    "system that iteratively modifies its own code over "
                    "executable program state and validates each change. "
                    "However, we do not evaluate on ARC-AGI."
                ),
            ),
            FreshWork(
                title="Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?",
                published="2025-11-17",
                arxiv_id="2511.13646v3",
                abstract=(
                    "Live-SWE-agent evolves its executable program state "
                    "while solving tasks."
                ),
            ),
        ]
        state = initial_state(topic)
        trajectories = recover_trajectories(papers, topic, state)
        mechanisms = [
            step.mechanism for line in trajectories for step in line.steps
        ]
        joined = " | ".join(mechanisms)
        for name in ("darwin", "godel", "gödel", "arc agi", "live swe", "swe agent"):
            self.assertNotIn(name, joined, mechanisms)
        for line in trajectories:
            self.assertNotEqual(line.lineage.lower(), "live swe")
            for step in line.steps:
                self.assertNotEqual(step.move, "migrate_setting", step)
        self.assertNotIn("migrate_setting", observed_patterns(trajectories))
        # The second paper only names itself: no mechanism, no limitation,
        # so it is not a trajectory at all.
        cited = {step.cite_id for line in trajectories for step in line.steps}
        self.assertNotIn("2511.13646v3", cited)

    def test_later_jumps_read_feedback_instead_of_drawing_independently(self) -> None:
        state = initial_state(TOPIC)
        trajectories = recover_trajectories(_papers(), TOPIC, state)
        first = propose_jump(index=0, state=state, trajectories=trajectories)
        ledger = SearchLedger()
        too_close = evaluate_landing(
            first,
            far=trajectories[0].latest_mechanism,
            category="entered",
            trajectories=trajectories,
            state=state,
        )
        self.assertEqual(too_close.distance, "too_close")
        self.assertEqual(too_close.policy, "increase_distance")
        ledger.absorb(too_close, far=trajectories[0].latest_mechanism)
        second = propose_jump(
            index=1, state=state, trajectories=trajectories, ledger=ledger
        )
        self.assertNotEqual(second.band, "early")
        self.assertTrue(second.shaped_by)
        self.assertNotIn(trajectories[0].latest_mechanism, second.far_labels)

    def test_an_unmoored_landing_tightens_the_next_trajectory_constraint(self) -> None:
        state = initial_state(TOPIC)
        trajectories = recover_trajectories(_papers(), TOPIC, state)
        proposal = propose_jump(
            index=0, state=state, trajectories=trajectories
        )
        feedback = evaluate_landing(
            proposal,
            far="feedback edge set",
            category="generation_refused",
            trajectories=trajectories,
            state=state,
        )
        self.assertEqual(feedback.novelty, "unmoored")
        self.assertEqual(feedback.policy, "tighten_trajectory")
        self.assertEqual(apply_policy("middle", "tighten_trajectory"), "early")

    def test_distance_grows_with_the_search_not_as_a_random_draw(self) -> None:
        self.assertEqual(band_for_index(0), "early")
        self.assertEqual(band_for_index(1), "middle")
        self.assertEqual(band_for_index(3), "far")
        self.assertEqual(apply_policy("early", "increase_distance"), "middle")
        self.assertEqual(apply_policy("far", "increase_distance"), "far")

    def test_observed_patterns_are_the_first_jump_prior(self) -> None:
        state = initial_state(TOPIC)
        trajectories = recover_trajectories(_papers(), TOPIC, state)
        patterns = observed_patterns(trajectories)
        self.assertTrue(patterns)
        self.assertTrue(set(patterns) <= set(JUMP_MOVES))

    def test_off_claim_papers_do_not_donate_mechanisms(self) -> None:
        topic = (
            "process supervision of chain-of-thought reasoning traces"
        )
        state = initial_state(topic)
        trajectories = recover_trajectories(_papers(), topic, state)
        self.assertEqual(trajectories, ())
        proposal = propose_jump(index=0, state=state, trajectories=trajectories)
        self.assertEqual(proposal.far_labels, ())

    def test_glue_tokens_do_not_forge_one_history(self) -> None:
        topic = "process supervision of chain-of-thought reasoning traces"
        papers = [
            FreshWork(
                title="Process reward models for chain-of-thought traces",
                published="2023-01-01",
                arxiv_id="2301.00001v1",
                abstract=(
                    "We score chain-of-thought reasoning traces with a "
                    "process reward. However, we do not verify search."
                ),
            ),
            FreshWork(
                title="Policy-gradient training of language agents",
                published="2024-01-01",
                arxiv_id="2401.00001v1",
                abstract=(
                    "We train agents with policy gradients on chain-of-thought "
                    "reasoning traces. However, we do not freeze replay."
                ),
            ),
        ]
        trajectories = recover_trajectories(papers, topic, initial_state(topic))
        self.assertEqual(len(trajectories), 2, [row.lineage for row in trajectories])
        self.assertTrue(all(len(row.steps) == 1 for row in trajectories))

    def test_a_citation_joins_a_development_line(self) -> None:
        topic = "process supervision of chain-of-thought reasoning traces"
        papers = [
            FreshWork(
                title="Process reward models for chain-of-thought traces",
                published="2023-01-01",
                arxiv_id="2301.00001v1",
                abstract=(
                    "We score chain-of-thought reasoning traces with a "
                    "process reward. However, we do not verify search."
                ),
            ),
            FreshWork(
                title="Policy-gradient training of language agents",
                published="2024-01-01",
                arxiv_id="2401.00001v1",
                abstract=(
                    "We train agents with policy gradients on chain-of-thought "
                    "reasoning traces. However, we do not freeze replay."
                ),
                references=("2301.00001v1",),
            ),
        ]
        trajectories = recover_trajectories(papers, topic, initial_state(topic))
        self.assertEqual(len(trajectories), 1, [row.lineage for row in trajectories])
        self.assertEqual(len(trajectories[0].steps), 2)

    def test_a_weak_limitation_clause_is_not_a_mechanism(self) -> None:
        self.assertFalse(_usable_phrase("support dynamic"))
        self.assertTrue(_usable_phrase("dynamic inserts"))
        self.assertTrue(_usable_phrase("wavelet tree"))
        self.assertTrue(_usable_phrase("shadow replay"))


if __name__ == "__main__":
    unittest.main()
