"""Idea kinds: compile policy, boards, and packet honesty."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from farfield.extras.farcompile import WorldMenu, assert_compile
from farfield.extras.generate import GenerationRefused, generate_card
from farfield.extras.ideakind import (
    ACQUIRE,
    PROBE,
    QUESTION,
    THEORY,
    idea_kind_of,
    parse_idea_kind,
    render_archive,
    render_question_board,
    runs_probe,
    skip_writeup_on_needs_world,
)
from farfield.extras.packet import packet_worthy, render_idea_human_packet
from farfield.extras.workspace import write_archive, write_question_board
from tests.test_generate import FakeClient, LABELS, answer


class ParseTests(unittest.TestCase):
    def test_empty_defaults_to_probe(self) -> None:
        self.assertEqual(parse_idea_kind(""), PROBE)
        self.assertEqual(parse_idea_kind(None), PROBE)
        self.assertEqual(idea_kind_of({}), PROBE)

    def test_aliases(self) -> None:
        self.assertEqual(parse_idea_kind("harvest"), ACQUIRE)
        self.assertEqual(parse_idea_kind("derivation"), THEORY)
        self.assertFalse(runs_probe(QUESTION))
        self.assertFalse(runs_probe(ACQUIRE))
        self.assertFalse(skip_writeup_on_needs_world(ACQUIRE))
        self.assertTrue(skip_writeup_on_needs_world(PROBE))


class CompileKindTests(unittest.TestCase):
    def test_a_question_refuses_dropout_while_a_probe_may_weld_it(self) -> None:
        menu = WorldMenu(
            levers=("dropout", "contrast:validator_write"),
            observables=("false_accept_fraction",),
            inert=frozenset(),
            attested_fields=("false_accept_fraction",),
        )

        def refuse(attempted, unlock, capability="", failed_far="", used_handle=False):
            raise RuntimeError(unlock)

        mapping = (
            "simulation agent becomes dropout by masking committed "
            "harness updates on attested false_accept_fraction"
        )
        with self.assertRaises(RuntimeError):
            assert_compile(
                far_label="simulation agent",
                world_lever="dropout",
                world_observable="false_accept_fraction",
                far_maps_to_lever=mapping,
                prediction="false_accept_fraction drops by 0.08 under dropout",
                menu=menu,
                refuse=refuse,
                idea_kind=QUESTION,
            )
        lever, observable, _ = assert_compile(
            far_label="simulation agent",
            world_lever="dropout",
            world_observable="false_accept_fraction",
            far_maps_to_lever=mapping,
            prediction="false_accept_fraction drops by 0.08 under dropout",
            menu=menu,
            refuse=refuse,
            idea_kind=PROBE,
        )
        self.assertEqual((lever, observable), ("dropout", "false_accept_fraction"))


class GenerateKindTests(unittest.TestCase):
    def test_missing_idea_kind_defaults_to_probe(self) -> None:
        card = generate_card(
            FakeClient(answer()),
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="directional",
            alienness=0.61,
            label_to_node=LABELS,
        )
        self.assertEqual(card.idea_kind, PROBE)
        self.assertEqual(card.to_dict()["idea_kind"], PROBE)

    def test_an_acquire_card_may_weld_freeze_validators(self) -> None:
        menu = WorldMenu(
            levers=("dropout", "freeze_validators", "contrast:validator_write"),
            observables=("false_accept_fraction",),
            inert=frozenset(),
            attested_fields=("false_accept_fraction",),
        )
        card = generate_card(
            FakeClient(
                answer(
                    idea_kind="acquire",
                    world_lever="freeze_validators",
                    world_observable="false_accept_fraction",
                    far_maps_to_lever=(
                        "protein folding becomes freeze_validators by "
                        "harvesting a world whose validators stay mutable"
                    ),
                    prediction="harvest a world with validators_mutable true",
                )
            ),
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="directional",
            alienness=0.61,
            label_to_node=LABELS,
            world_menu=menu,
        )
        self.assertEqual(card.idea_kind, ACQUIRE)
        self.assertEqual(card.world_lever, "freeze_validators")

    def test_theory_without_derivation_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused) as caught:
            generate_card(
                FakeClient(answer(idea_kind="theory", derivation="too short")),
                seed_label="sparse attention",
                near_labels=("kv cache",),
                far_labels=("protein folding",),
                operator="directional",
                alienness=0.61,
                label_to_node=LABELS,
            )
        self.assertEqual(
            caught.exception.record.missing_capability, "model_card_schema"
        )


class BoardTests(unittest.TestCase):
    def test_boards_write_and_name_kinds(self) -> None:
        board = render_question_board(
            scorable=("contrast:validator_write/false_accept_fraction — writes; measure FA",),
            records=[{"idea_kind": "question", "claim": "is FA higher on validator-write?", "world_lever": "contrast:validator_write"}],
            interventional=("freeze_validators/false_accept_fraction",),
        )
        self.assertIn("Scorable on this freeze", board)
        self.assertIn("is FA higher", board)
        self.assertIn("freeze_validators", board)
        archive = render_archive(
            [{"idea_kind": "acquire", "pair": ["code world", "shadow replay"], "claim": "need mutable validators"}]
        )
        self.assertIn("`acquire`", archive)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            write_question_board(dest, board)
            write_archive(dest, archive)
            self.assertTrue((dest / "QUESTION_BOARD.md").is_file())
            self.assertTrue((dest / "ARCHIVE.md").is_file())


class PacketKindTests(unittest.TestCase):
    def test_acquire_occupies_ideas_without_a_two_arm(self) -> None:
        rec = {
            "idea_kind": ACQUIRE,
            "claim": "need a mutable-validator world",
            "pair": ["code world model", "shadow replay"],
            "world_lever": "freeze_validators",
            "prediction": "validators_mutable must be true",
        }
        self.assertTrue(packet_worthy(rec))
        text = render_idea_human_packet("code world model", rec, lang="en")
        self.assertIn("harvest / import / derive", text.lower())
        self.assertIn("do not run a two-arm", text.lower())


if __name__ == "__main__":
    unittest.main()
