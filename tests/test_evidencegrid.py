"""Contrast handles, the margin gate's ceiling, typed constraints, the grid."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.derive import program_state_from_traces
from farfield.extras.dynworld import (
    PROGRAM_CONTRASTS,
    contrast_meaning,
    is_contrast,
    levers_of,
    response_card,
)
from farfield.extras.evidencegrid import compile_grid, outcome_type, render_grid
from farfield.extras.explore import ExplorationPrior
from farfield.extras.farcompile import WorldMenu, assert_compile, menu_from_card
from farfield.extras.freeze import derive_world, freeze_world


def _trace(instance, label, tools, *, run=True):
    steps = [{"t": 0, "action": "ls", "returncode": 0}]
    for name, sha in tools:
        steps.append({"t": len(steps), "action": f"cat <<'EOF' > {name}\nx\nEOF", "returncode": 0})
        if run:
            steps.append({"t": len(steps), "action": f"python3 {name}", "returncode": 0})
    return {
        "id": f"m/{instance}", "label": label, "instance_id": instance,
        "template_id": instance.split("__")[0], "steps": steps,
        "created_tools": [{"name": n, "chars": 3, "sha256": s} for n, s in tools],
    }


def _payload():
    return {
        "schema": "labeled_traces", "id": "t", "object_type": "trace", "n": 6,
        "traces": [
            _trace("a__1", "resolved", [("edit.py", "1"), ("review_fix.py", "2")]),
            _trace("b__2", "unresolved", [("edit.py", "3")], run=False),
            _trace("c__3", "resolved", [("verify.py", "4"), ("scan.py", "5"), ("edit.py", "1"), ("x.py", "9")]),
            _trace("d__4", "unresolved", [("scan.py", "6"), ("edit.py", "7"), ("y.py", "8"), ("z.py", "10")]),
            _trace("e__5", "resolved", [("edit.py", "1")]),
            _trace("f__6", "unresolved", [("check.py", "11")]),
        ],
    }


class ContrastTests(unittest.TestCase):
    def test_a_replayed_history_advertises_contrasts_with_oracle_false_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "worlds"
            catalog.mkdir()
            src = root / "t.json"
            src.write_text(json.dumps(_payload()), encoding="utf-8")
            freeze_world(world_id="tr", schema="labeled_traces", slice_rule="all", catalog=catalog, source_file=src)
            child = derive_world(world_id="ps", parent="tr", schema="program_state", catalog=catalog)
            levers = levers_of(child)
            self.assertIn("contrast:validator_write", levers)
            self.assertIn("contrast:heavy_episode", levers)
            self.assertTrue(all(is_contrast(l) for l in levers if l.startswith("contrast:")))
            self.assertTrue(contrast_meaning("contrast:validator_write"))
            card = response_card(child)
            # false_accept_fraction is now oracle-relative: the validator-write
            # stratum moves it (the fixture's review/verify episodes include an
            # unresolved one).
            self.assertIn("contrast:validator_write", card["levers"])
            menu = menu_from_card(card, child)
            block = menu.prompt_block(())
            self.assertIn("Observational handles", block)
            self.assertIn("contrast:validator_write", block)
            # A used handle produces a hard constraint listing what is left.
            constrained = menu.prompt_block(("dropout/false_accept_fraction",))
            self.assertIn("HARD CONSTRAINT", constrained)
            self.assertNotIn("dropout/false_accept_fraction", constrained.split("MUST be one of")[1].split(".")[0])

    def test_unused_handles_exclude_used_and_inert(self) -> None:
        menu = WorldMenu(
            levers=("dropout", "contrast:validator_write"),
            observables=("a", "b"),
            inert=frozenset({"dropout/b"}),
            attested_fields=(),
            load_hint="",
        )
        self.assertEqual(
            menu.unused_handles(("dropout/a",)),
            ("contrast:validator_write/a", "contrast:validator_write/b"),
        )
        # A bare used lever blocks every handle on that lever.
        self.assertEqual(menu.unused_handles(("dropout",)), ("contrast:validator_write/a", "contrast:validator_write/b"))

    def test_unused_observational_and_blocked_handles(self) -> None:
        menu = WorldMenu(
            levers=("dropout", "contrast:validator_write"),
            observables=("false_accept_fraction",),
            inert=frozenset(),
            attested_fields=("false_accept_fraction",),
        ).with_blocked(("dropout/false_accept_fraction",))
        self.assertEqual(
            menu.unused_observational(()),
            ("contrast:validator_write/false_accept_fraction",),
        )
        self.assertNotIn("dropout/false_accept_fraction", menu.unused_handles(()))
        questions = menu.scorable_questions()
        self.assertTrue(questions)
        self.assertIn("contrast:validator_write/false_accept_fraction", questions[0])
        block = menu.prompt_block(())
        self.assertIn("Scorable questions", block)
        self.assertIn("Blocked attractor", block)

    def test_replay_history_refuses_intervention_while_contrasts_remain(self) -> None:
        menu = WorldMenu(
            levers=("dropout", "contrast:validator_write"),
            observables=("false_accept_fraction",),
            inert=frozenset(),
            attested_fields=("false_accept_fraction", "task_return"),
        )

        def refuse(attempted, unlock, capability="", failed_far="", used_handle=False):
            raise RuntimeError(unlock)

        with self.assertRaises(RuntimeError) as caught:
            assert_compile(
                far_label="simulation agent",
                world_lever="dropout",
                world_observable="false_accept_fraction",
                far_maps_to_lever=(
                    "simulation agent becomes dropout by masking committed "
                    "harness updates on attested false_accept_fraction"
                ),
                prediction="false_accept_fraction drops by 0.08 under dropout",
                menu=menu,
                refuse=refuse,
                idea_kind="question",
            )
        self.assertIn("causal overclaim", str(caught.exception).lower())
        lever, obs, _ = assert_compile(
            far_label="simulation agent",
            world_lever="contrast:validator_write",
            world_observable="false_accept_fraction",
            far_maps_to_lever=(
                "simulation agent becomes contrast:validator_write by "
                "splitting validator-writing updates on attested fields"
            ),
            prediction="false_accept_fraction is higher on validator_write records",
            menu=menu,
            refuse=refuse,
        )
        self.assertEqual(lever, "contrast:validator_write")
        self.assertEqual(obs, "false_accept_fraction")

    def test_acquire_may_name_an_interventional_lever_while_contrasts_remain(self) -> None:
        menu = WorldMenu(
            levers=("dropout", "freeze_validators", "contrast:validator_write"),
            observables=("false_accept_fraction",),
            inert=frozenset(),
            attested_fields=("false_accept_fraction", "task_return"),
        )

        def refuse(attempted, unlock, capability="", failed_far="", used_handle=False):
            raise RuntimeError(unlock)

        lever, obs, _ = assert_compile(
            far_label="simulation agent",
            world_lever="freeze_validators",
            world_observable="false_accept_fraction",
            far_maps_to_lever=(
                "simulation agent becomes freeze_validators by harvesting "
                "a world whose validators stay mutable on attested fields"
            ),
            prediction="harvest a world with validators_mutable true",
            menu=menu,
            refuse=refuse,
            idea_kind="acquire",
        )
        self.assertEqual(lever, "freeze_validators")
        self.assertEqual(obs, "false_accept_fraction")
        with self.assertRaises(RuntimeError):
            assert_compile(
                far_label="simulation agent",
                world_lever="freeze_validators",
                world_observable="false_accept_fraction",
                far_maps_to_lever=(
                    "simulation agent becomes freeze_validators by masking "
                    "committed harness updates on attested false_accept_fraction"
                ),
                prediction="false_accept_fraction drops by 0.08",
                menu=menu,
                refuse=refuse,
                idea_kind="question",
            )


class ConstraintTests(unittest.TestCase):
    def test_typed_negative_results_become_binding_lines(self) -> None:
        prior = ExplorationPrior()
        prior.learn({"stage": "evidence", "verdict": "uninformative", "dv_blind": True, "reason": "dv_blind: ..."})
        prior.learn({"stage": "evidence", "verdict": "uninformative", "reason": "placebo control: |sep_real - sep_placebo| ... the experiment did not consume the bound world"})
        prior.learn({"stage": "evidence", "verdict": "uninformative", "separation": 0.031, "margin": 0.25})
        prior.learn({"stage": "margin_refused", "world_lever": "dropout", "world_observable": "false_accept_fraction", "ceiling": 0.097})
        prior.learn(
            {
                "stage": "world_scout",
                "levers": ["dropout", "freeze_validators", "contrast:validator_write"],
                "observables": ["a", "b"],
                "inert": ["freeze_validators/a", "freeze_validators/b"],
            }
        )
        prior.world_facts.append("oracle failure given accepted 0.303 vs rejected 0.305")
        text = "\n".join(prior.prompt_lines())
        for needle in ("dv_blind happened", "placebo_fail happened", "under_margin happened", "effect ceiling", "inert levers on this world: freeze_validators", "observational handles", "Measured on the bound world", "0.303"):
            self.assertIn(needle, text)
        # Deduplicated.
        prior.learn({"stage": "evidence", "verdict": "uninformative", "dv_blind": True})
        self.assertEqual(sum(1 for c in prior.constraints if c.startswith("dv_blind")), 1)
        self.assertIn("constraints", prior.snapshot())

    def test_two_misses_on_one_handle_become_a_blocked_attractor(self) -> None:
        prior = ExplorationPrior()
        miss = {
            "stage": "evidence",
            "verdict": "uninformative",
            "dv_blind": True,
            "world_lever": "dropout",
            "world_observable": "false_accept_fraction",
        }
        prior.learn(miss)
        self.assertEqual(prior.blocked_handles, [])
        prior.learn(miss)
        self.assertEqual(len(prior.blocked_handles), 1)
        self.assertIn("dropout/false_accept_fraction", prior.blocked_handles[0])
        self.assertIn("dv_blind", prior.blocked_handles[0])
        self.assertTrue(any(c.startswith("attractor:") for c in prior.constraints))
        self.assertIn("blocked_handles", prior.snapshot())
        self.assertIn("Blocked attractor handles", "\n".join(prior.prompt_lines()))


class GridTests(unittest.TestCase):
    def test_outcome_types_are_read_off_rows(self) -> None:
        self.assertEqual(outcome_type({"verdict": "uninformative", "dv_blind": True}), "blind_measure")
        self.assertEqual(outcome_type({"verdict": "uninformative", "reason": "placebo control: ... did not consume the bound world"}), "placebo_fail")
        self.assertEqual(outcome_type({"verdict": "uninformative", "object_absent": True}), "object_absent")
        self.assertEqual(outcome_type({"verdict": "uninformative"}), "under_margin")
        self.assertEqual(outcome_type({"verdict": "supports"}), "supports")
        self.assertEqual(outcome_type({"status": "crashed"}), "crashed")

    def test_grid_names_fillable_cells_and_worlds_to_acquire(self) -> None:
        card = {
            "levers": {
                "dropout": {"false_accept_fraction": -0.097},
                "freeze_validators": {"false_accept_fraction": 0.0},
                "contrast:validator_write": {"false_accept_fraction": 0.267},
            },
            "inert": ["freeze_validators/false_accept_fraction"],
        }
        grid = compile_grid(
            claim="self-written validators raise false acceptance",
            world={"id": "swe", "schema": "program_state", "object_properties": {"validators_mutable": True, "has_oracle": True}},
            dynamics_card=card,
            diagnosis={"world_lever": "dropout", "world_observable": "false_accept_fraction", "alternative": "difficulty", "control_arm": "uniform"},
            attempts=[
                {"attempt": 0, "verdict": "uninformative", "dv_blind": True, "evidence_id": "e0"},
                {"attempt": 1, "verdict": "uninformative", "evidence_id": "e1", "treatment": 0.58, "control": 0.55},
            ],
            catalog_worlds=[{"id": "oe", "schema": "program_state", "object_properties": {"validators_mutable": False}}],
            required_properties={"validators_mutable": True, "has_oracle": True},
        )
        ids = {c["id"]: c for c in grid["cells"]}
        self.assertEqual(ids["C1.main"]["status"], "filled:under_margin")
        self.assertEqual([f["type"] for f in ids["C1.main"]["fills"]], ["blind_measure", "under_margin"])
        self.assertEqual(ids["C1.contrast.validator_write"]["status"], "fillable")
        self.assertEqual(ids["C1.intervene.validators"]["status"], "needs_world")
        self.assertEqual(ids["C1.intervene.validators"]["acquire"]["how"], "harvest")
        # The OE world contradicts validators_mutable, so replication needs a world.
        self.assertEqual(ids["C1.replicate"]["status"], "needs_world")
        self.assertEqual(grid["coverage"]["supported"], 0)
        text = render_grid(grid)
        self.assertIn("needs a world", text)
        self.assertIn("blind_measure", text)


class ProgramStatePlaceboTests(unittest.TestCase):
    """The structure placebo must destroy program_state structure, not copy it."""

    def test_placebo_permutes_structural_fields_and_keeps_marginals(self) -> None:
        from farfield.extras.dynworld import materialize_placebo

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "worlds"
            catalog.mkdir()
            src = root / "t.json"
            src.write_text(json.dumps(_payload()), encoding="utf-8")
            freeze_world(world_id="tr", schema="labeled_traces", slice_rule="all", catalog=catalog, source_file=src)
            child = derive_world(world_id="ps", parent="tr", schema="program_state", catalog=catalog)
            real = json.loads((child.root / "world.json").read_text(encoding="utf-8"))["updates"]
            placebo = materialize_placebo(child, root / "placebo", seed=3)
            fake = json.loads((placebo.root / "world.json").read_text(encoding="utf-8"))["updates"]
            self.assertEqual(len(real), len(fake))
            # Marginals kept: the multiset of every structural field is identical.
            for field in ("divergent", "accepted", "episode", "validator_writes"):
                self.assertEqual(
                    sorted(json.dumps(u.get(field), sort_keys=True) for u in real),
                    sorted(json.dumps(u.get(field), sort_keys=True) for u in fake),
                )
            # Structure destroyed: the file is not a byte-identical copy.
            self.assertNotEqual(
                (child.root / "world.json").read_bytes(), (placebo.root / "world.json").read_bytes()
            )
            # The oracle stays where it was: this placebo is about structure.
            self.assertEqual([u["task_return"] for u in real], [u["task_return"] for u in fake])
