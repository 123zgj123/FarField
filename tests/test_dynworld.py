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
    compose_simulation,
    forward_simulate,
    has_dynamics,
    levers_of,
    literature_phrases,
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
if "live-swe-agent-verified-v1" in CATALOG:
    FIXTURE_BY_SCHEMA["labeled_traces"] = "live-swe-agent-verified-v1"


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


class LabeledTraceDynamicsTests(unittest.TestCase):
    @unittest.skipUnless("live-swe-agent-verified-v1" in CATALOG, "optional Live-SWE trajectories are not shipped")
    def test_live_swe_exposes_object_native_levers_not_token_dropout(self) -> None:
        fixture = CATALOG["live-swe-agent-verified-v1"]
        self.assertTrue(has_dynamics(fixture))
        levers = levers_of(fixture)
        self.assertIn("dropout", levers)
        self.assertIn("mask_tools", levers)
        self.assertIn("drop_majority", levers)
        self.assertNotIn("window_shuffle", levers)
        plan = compose_simulation(fixture)
        self.assertTrue(plan["available"])
        self.assertEqual(plan["origin"], "bound")
        self.assertEqual(plan["schema"], "labeled_traces")

    @unittest.skipUnless("live-swe-agent-verified-v1" in CATALOG, "optional Live-SWE trajectories are not shipped")
    def test_mask_tools_moves_the_tooled_fraction(self) -> None:
        fixture = CATALOG["live-swe-agent-verified-v1"]
        trajectory = forward_simulate(fixture, "mask_tools", seed=0, horizon=3)
        self.assertLess(trajectory["response"]["tooled_fraction"], -0.03)
        self.assertIn("mean_toolset_jaccard", trajectory["steps"][0]["observables"])
        self.assertIn("mean_action_chars", trajectory["steps"][0]["observables"])
        self.assertGreater(
            abs(trajectory["response"]["mean_toolset_jaccard"]),
            0.03,
        )

    def test_a_constructed_traces_payload_gets_the_traces_bridge(self) -> None:
        import json
        import tempfile

        from farfield.extras.world import WorldFixture

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "traces.json").write_text(
                json.dumps(
                    {
                        "traces": [
                            {
                                "id": "a",
                                "label": "resolved",
                                "steps": [{"t": 0, "returncode": 0}],
                            },
                            {
                                "id": "b",
                                "label": "unresolved",
                                "steps": [{"t": 0, "returncode": 1}],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            fixture = WorldFixture(
                id="constructed-traces",
                title="same-family stub",
                source="generated",
                retrieved_at="2026-08-30",
                digest="0" * 64,
                files=("traces.json",),
                root=root,
                domains=("swe",),
                role="generated",
                schema="labeled_traces",
            )
            plan = compose_simulation(fixture)
            self.assertTrue(plan["available"])
            self.assertEqual(plan["origin"], "constructed")
            self.assertIn("mask_failures", levers_of(fixture))
            self.assertIn("drop_majority", levers_of(fixture))

    def test_a_novel_is_not_a_simulation_playground_for_traces(self) -> None:
        from farfield.extras.world import pick_world

        catalog = CATALOG
        chosen = pick_world(
            "code world models for recursive self-improvement of "
            "software engineering agents on Live-SWE execution trajectories",
            catalog,
        )
        if "live-swe-agent-verified-v1" in catalog:
            self.assertIsNotNone(chosen)
            self.assertEqual(chosen.id, "live-swe-agent-verified-v1")
        else:
            self.assertIsNone(chosen, "missing trajectories must not bind unrelated bytes")
        pride_plan = compose_simulation(catalog["gutenberg-pride"])
        self.assertEqual(pride_plan["schema"], "text_stream")
        self.assertNotIn("mask_tools", pride_plan["levers"])


class GeneratedTableDynamicsTests(unittest.TestCase):
    def test_list_rows_do_not_crash_the_numeric_table_bridge(self) -> None:
        import json
        import tempfile

        from farfield.extras.world import WorldFixture

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "table.json").write_text(
                json.dumps(
                    {
                        "columns": ["x", "y"],
                        "rows": [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
                    }
                ),
                encoding="utf-8",
            )
            fixture = WorldFixture(
                id="generated-table-lists",
                title="scaffold numeric_table",
                source="generated",
                retrieved_at="2026-09-01",
                digest="0" * 64,
                files=("table.json",),
                root=root,
                domains=("representation",),
                role="generated",
                schema="numeric_table",
            )
            plan = compose_simulation(fixture)
            self.assertTrue(plan["available"])
            self.assertEqual(plan["origin"], "constructed")
            self.assertIn("dropout", levers_of(fixture))


class GeneratedGraphDynamicsTests(unittest.TestCase):
    def test_dict_nodes_do_not_crash_the_graph_bridge(self) -> None:
        import json
        import tempfile

        from farfield.extras.world import WorldFixture

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "graph.json").write_text(
                json.dumps(
                    {
                        "nodes": [
                            {"id": "n0", "label": "write"},
                            {"id": "n1", "label": "read"},
                            {"id": "n2", "label": "validate"},
                        ],
                        "edges": [
                            {"src": "n0", "dst": "n1"},
                            ["n1", "n2"],
                            {"from": "n2", "to": "n0"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            fixture = WorldFixture(
                id="generated-graph-dicts",
                title="scaffold undirected_graph",
                source="generated",
                retrieved_at="2026-09-03",
                digest="0" * 64,
                files=("graph.json",),
                root=root,
                domains=("graph",),
                role="generated",
                schema="undirected_graph",
            )
            plan = compose_simulation(fixture)
            self.assertTrue(plan["available"], plan)
            self.assertEqual(plan["origin"], "constructed")
            card = response_card(fixture, seed=0, horizon=2)
            self.assertIn("dropout", card["levers"])
            self.assertGreaterEqual(len(card["observables"]), 1)


class BoundProgramStateLayoutTests(unittest.TestCase):
    def test_data_world_json_exposes_replay_gate(self) -> None:
        import json
        import tempfile

        from farfield.extras.world import WorldFixture

        payload = {
            "object_type": "executable",
            "schema": "program_state",
            "cells": ["cell_0", "cell_1", "cell_2"],
            "validators": [{"id": "v_guard", "reads": ["cell_1", "cell_2"]}],
            "updates": [
                {
                    "id": "u1",
                    "epoch": 1,
                    "writes": ["cell_1"],
                    "reads": ["cell_2"],
                    "validator_writes": ["v_guard"],
                    "accepted": True,
                    "divergent": True,
                    "task_return": 0.8,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "world.json").write_text(
                json.dumps({"id": "card", "schema": "program_state", "role": "generated"}),
                encoding="utf-8",
            )
            (root / "data" / "world.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            fixture = WorldFixture(
                id="generated-program_state",
                title="executable stub",
                source="generated",
                retrieved_at="2026-09-05",
                digest="0" * 64,
                files=("world.json",),
                root=root,
                domains=("program",),
                role="generated",
                schema="program_state",
            )
            levers = levers_of(fixture)
            self.assertIn("replay_gate", levers)
            self.assertIn("freeze_validators", levers)
            plan = compose_simulation(fixture)
            self.assertTrue(plan["available"])


class LiteraturePhraseTests(unittest.TestCase):
    def test_truncate_on_traces_includes_code_world_models(self) -> None:
        phrases = literature_phrases(
            levers=("truncate",), schema="labeled_traces"
        )
        self.assertIn("truncate", phrases)
        self.assertIn("code world model", phrases)
        self.assertNotIn("rewire", phrases)

    def test_a_graph_rewire_does_not_borrow_trace_literature(self) -> None:
        phrases = literature_phrases(
            levers=("rewire",), schema="undirected_graph"
        )
        self.assertIn("rewire", phrases)
        self.assertNotIn("code world model", phrases)
        self.assertNotIn("query budget", phrases)


if __name__ == "__main__":
    unittest.main()
