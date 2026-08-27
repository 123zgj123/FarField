"""The awakened world: runtime-owned dynamics over frozen bytes.

A fixture used to be a byte blob a model-written script read however it
liked. The dynamics bridge gives every schema family levers (the only
ways a mechanism can act on this world), observables (verifiers the
model did not write), and a forward simulation any auditor can
recompute. These tests pin the contract: deterministic, stepwise on the
evolving state, vocabulary-closed — and the diagnosis may choose a
handle from that vocabulary, not invent one.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from farfield.extras.diagnose import DiagnosisRefused, diagnosis_from_payload
from farfield.extras.dynworld import (
    DynamicsUnavailable,
    forward_simulate,
    has_dynamics,
    levers_of,
    materialize_placebo,
    response_card,
    scout_summary,
    world_consumed,
)
from farfield.extras.schemas import SCHEMAS
from farfield.extras.world import load_catalog

ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT)

# One attested fixture per schema family, straight from the repo catalog.
FIXTURE_BY_SCHEMA = {
    "undirected_graph": "zachary-karate",
    "text_stream": "gutenberg-alice",
    "fasta": "phix174",
    "numeric_table": "fisher-iris",
    "symbolic_trace": "tcp-linux-server",
}


class BridgeCoverageTests(unittest.TestCase):
    def test_every_registered_schema_family_has_a_dynamics_bridge(self) -> None:
        for schema, world_id in FIXTURE_BY_SCHEMA.items():
            self.assertIn(schema, SCHEMAS)
            fixture = CATALOG[world_id]
            self.assertTrue(has_dynamics(fixture), schema)
            self.assertTrue(levers_of(fixture), schema)

    def test_a_schema_without_a_bridge_has_no_dynamics(self) -> None:
        self.assertFalse(has_dynamics(CATALOG["path-trace"]))
        self.assertEqual(levers_of(CATALOG["path-trace"]), ())

    def test_named_graphs_share_the_undirected_bridge(self) -> None:
        self.assertTrue(has_dynamics(CATALOG["florentine-families"]))
        self.assertIn("dropout", levers_of(CATALOG["les-miserables"]))

    def test_every_bridge_declares_the_common_dropout_lever(self) -> None:
        for world_id in FIXTURE_BY_SCHEMA.values():
            self.assertIn("dropout", levers_of(CATALOG[world_id]))


class ForwardSimulationTests(unittest.TestCase):
    def test_the_trajectory_is_deterministic_and_digested(self) -> None:
        first = forward_simulate(CATALOG["zachary-karate"], "dropout", seed=7)
        second = forward_simulate(CATALOG["zachary-karate"], "dropout", seed=7)
        self.assertEqual(first, second)
        self.assertTrue(first["report_digest"])
        other_seed = forward_simulate(CATALOG["zachary-karate"], "dropout", seed=8)
        self.assertNotEqual(first["report_digest"], other_seed["report_digest"])

    def test_step_zero_is_the_untouched_world(self) -> None:
        fixture = CATALOG["zachary-karate"]
        baselines = {
            lever: forward_simulate(fixture, lever, seed=3)["steps"][0]
            for lever in levers_of(fixture)
        }
        rows = list(baselines.values())
        for row in rows[1:]:
            self.assertEqual(row["observables"], rows[0]["observables"])
        self.assertEqual(rows[0]["intensity"], 0)

    def test_an_invented_lever_is_refused(self) -> None:
        with self.assertRaises(DynamicsUnavailable):
            forward_simulate(CATALOG["zachary-karate"], "tool_use_distillation")

    def test_the_response_is_a_dose_reading_not_an_assertion(self) -> None:
        trajectory = forward_simulate(
            CATALOG["zachary-karate"], "hub_removal", seed=0
        )
        # Removing hubs must shrink the giant component — a world fact the
        # runtime measured, available to any diagnosis before it registers.
        self.assertLess(trajectory["response"]["largest_component_fraction"], -0.03)
        self.assertTrue(trajectory["analysis"]["summary"])
        self.assertIn(trajectory["stop_reason"], {"absorbing", "plateau", "horizon"})

    def test_walk_executes_the_trace_until_a_runtime_stop(self) -> None:
        trajectory = forward_simulate(CATALOG["tcp-linux-server"], "walk", seed=0)
        self.assertIn("walk", levers_of(CATALOG["tcp-linux-server"]))
        self.assertIn(trajectory["stop_reason"], {"absorbing", "plateau", "horizon"})
        self.assertGreaterEqual(len(trajectory["steps"]), 1)
        self.assertEqual(trajectory["analysis"]["stop_reason"], trajectory["stop_reason"])
        self.assertTrue(trajectory["analysis"]["summary"])

    def test_later_steps_apply_the_lever_to_the_previous_state(self) -> None:
        trajectory = forward_simulate(
            CATALOG["zachary-karate"], "dropout", seed=0, horizon=4
        )
        degrees = [
            step["observables"]["mean_degree"] for step in trajectory["steps"]
        ]
        self.assertGreaterEqual(len(degrees), 2)
        # Dropout on the evolving graph cannot restore edges the previous
        # step already removed.
        self.assertLessEqual(degrees[-1], degrees[1])


class ResponseCardTests(unittest.TestCase):
    def test_the_card_covers_every_lever_and_names_inert_pairs(self) -> None:
        card = response_card(CATALOG["tcp-linux-server"])
        self.assertEqual(
            sorted(card["levers"]), sorted(levers_of(CATALOG["tcp-linux-server"]))
        )
        # Rewiring a dense automaton to random states barely moves the
        # observables: the card must say so instead of leaving a diagnosis
        # to discover it with a burned probe.
        self.assertTrue(any(pair.startswith("rewire/") for pair in card["inert"]))
        self.assertTrue(card["report_digest"])

    def test_placebo_consumption_uses_margin_not_a_z_test(self) -> None:
        # Scenery: almost the same separation on a ruined copy.
        self.assertFalse(world_consumed(-0.948, -0.950, 0.05))
        # Real consumption: displacement clears the registered margin.
        self.assertTrue(world_consumed(-0.60, -0.10, 0.05))

    def test_a_placebo_copy_keeps_schema_and_ruins_structure(self) -> None:
        import tempfile

        fixture = CATALOG["zachary-karate"]
        with tempfile.TemporaryDirectory() as tmp:
            placebo = materialize_placebo(fixture, Path(tmp) / "p", seed=1)
            self.assertEqual(placebo.schema, fixture.schema)
            self.assertEqual(placebo.role, "placebo")
            self.assertNotEqual(placebo.digest, fixture.digest)

    def test_the_scout_summary_flags_a_registered_inert_lever(self) -> None:
        card = response_card(CATALOG["tcp-linux-server"])
        summary = scout_summary(card, "rewire")
        self.assertIn("inert", summary)
        self.assertIn("rewire", summary)


def _payload(**extra) -> dict:
    base = {
        "alternative": "caching explains the speedup, not the encoding",
        "experiment": "same query log, cache off in both arms",
        "treatment_arm": "encoding on",
        "control_arm": "encoding off",
        "expected_direction": "treatment_lower",
        "alternative_direction": "treatment_higher",
        "expected_if_alternative": "the arms tie once the cache is off",
        "margin": 0.05,
        "margin_reason": "toy counter noise stays under five percent",
    }
    base.update(extra)
    return base


class LeverVocabularyGateTests(unittest.TestCase):
    """The model may choose a handle; it may not invent one."""

    VOCAB = {
        "_world_levers": ("dropout", "hub_removal"),
        "_world_observables": ("largest_component_fraction", "mean_degree"),
    }

    def test_a_declared_lever_and_observable_bind(self) -> None:
        diagnosis = diagnosis_from_payload(
            "c1",
            _payload(
                world_lever="dropout",
                world_observable="mean_degree",
                **self.VOCAB,
            ),
        )
        self.assertEqual(diagnosis.world_lever, "dropout")
        self.assertEqual(diagnosis.world_observable, "mean_degree")

    def test_an_invented_lever_is_refused(self) -> None:
        with self.assertRaises(DiagnosisRefused):
            diagnosis_from_payload(
                "c1",
                _payload(
                    world_lever="distill_tools",
                    world_observable="mean_degree",
                    **self.VOCAB,
                ),
            )

    def test_an_invented_observable_is_refused(self) -> None:
        with self.assertRaises(DiagnosisRefused):
            diagnosis_from_payload(
                "c1",
                _payload(
                    world_lever="dropout",
                    world_observable="student_accuracy",
                    **self.VOCAB,
                ),
            )

    def test_an_honest_none_passes_and_clears_the_observable(self) -> None:
        diagnosis = diagnosis_from_payload(
            "c1",
            _payload(
                world_lever="none",
                world_observable="mean_degree",
                **self.VOCAB,
            ),
        )
        self.assertEqual(diagnosis.world_lever, "none")
        self.assertEqual(diagnosis.world_observable, "")

    def test_a_missing_lever_defaults_to_none_not_to_a_binding(self) -> None:
        diagnosis = diagnosis_from_payload("c1", _payload(**self.VOCAB))
        self.assertEqual(diagnosis.world_lever, "none")

    def test_without_a_vocabulary_the_fields_stay_unvalidated(self) -> None:
        diagnosis = diagnosis_from_payload(
            "c1", _payload(world_lever="", world_observable="")
        )
        self.assertEqual(diagnosis.world_lever, "")


if __name__ == "__main__":
    unittest.main()
