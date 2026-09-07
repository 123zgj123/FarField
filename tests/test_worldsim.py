"""World rehearsal: three-tier idea simulation, not a freeze and not evidence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.cli import build_parser
from farfield.extras.diagnose import diagnosis_from_payload, judge_probe
from farfield.extras.livefeed import FreshWork
from farfield.extras.packet import render_analyze_results, render_result_to_claim
from farfield.extras.worldsim import (
    KIND_WORLD_SIM,
    WORLD_SIM_FOLDER,
    WhatIfExperiment,
    WorldSimError,
    maybe_world_sim_after_diagnosis,
    papers_for_rehearsal,
    run_world_sim,
    validate_tiers,
    world_sim_dir,
)


CLAIM = (
    "full-trace supervision of code world models improves next-event prediction "
    "on truncated trajectories"
)
MECHANISM = (
    "prefix-to-next process reward on an observation-action rollout, "
    "not outcome reward after the final patch"
)


def _brief() -> str:
    return "\n".join(
        [
            "# Research brief",
            "",
            "**主张。**",
            "",
            CLAIM,
            "",
            "**机制。**",
            "",
            MECHANISM,
            "",
        ]
    )


def _plan() -> str:
    return "\n".join(
        [
            "# Experiment plan",
            "",
            "### 实验块 0",
            "",
            "Same two-arm measure. Treatment truncates to the next event.",
            "",
        ]
    )


def _idea(root: Path) -> Path:
    folder = root / "ideas" / "full-trace-supervision"
    (folder / "idea-stage").mkdir(parents=True)
    (folder / "refine-logs").mkdir(parents=True)
    (folder / "idea-stage" / "RESEARCH_BRIEF.md").write_text(_brief(), encoding="utf-8")
    (folder / "refine-logs" / "EXPERIMENT_PLAN.md").write_text(_plan(), encoding="utf-8")
    return folder


def _tiers() -> dict:
    return {
        "scientific_invariants": [
            "same bound freeze",
            "same two-arm measure",
        ],
        "shared_research_constraints": ["no invented wall-clock fields"],
        "tier_varying_factors": ["effect size"],
        "tiers": {
            "best": {
                "meaning": "prefix-to-next catches the failure early",
                "key_assumptions": ["the signal lives in the prefix"],
                "number_ranges": {"delta": [0.20, 0.40]},
                "narrative_direction": "supports",
            },
            "median": {
                "meaning": "small but real lift",
                "key_assumptions": ["partial prefix signal"],
                "number_ranges": {"delta": [0.05, 0.15]},
                "narrative_direction": "uninformative",
            },
            "worst": {
                "meaning": "outcome reward already saturates",
                "key_assumptions": ["no prefix signal"],
                "number_ranges": {"delta": [-0.05, 0.04]},
                "narrative_direction": "weakens",
            },
        },
    }


def _elab(delta: float) -> list[dict]:
    return [
        {
            "name": "registered two-arm",
            "plan_id": "block-0",
            "description": "same measure, mechanism flag only",
            "expected_results": {"delta": delta},
            "result_rationale": (
                "the prefix-to-next lever should move this metric "
                "under the registered two-arm protocol"
            ),
        }
    ]


def _run(idea: Path, **kwargs):
    elaborations = kwargs.pop(
        "canned_elaborations",
        {"best": _elab(0.30), "median": _elab(0.10), "worst": _elab(0.00)},
    )
    return run_world_sim(
        idea,
        canned_tiers=_tiers(),
        canned_elaborations=elaborations,
        claim=CLAIM,
        mechanism=MECHANISM,
        world_lever=kwargs.pop("world_lever", "truncate"),
        schema=kwargs.pop("schema", "labeled_traces"),
        scout=kwargs.pop(
            "scout",
            {
                "levers": {"truncate": {}, "mask_tools": {}},
                "schema": "labeled_traces",
            },
        ),
        sim_id=kwargs.pop("sim_id", "sim1"),
        **kwargs,
    )


class IsolationTests(unittest.TestCase):
    def test_reports_cannot_land_under_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "candidates" / "gen_x" / "probe"
            probe.mkdir(parents=True)
            with self.assertRaises(WorldSimError) as caught:
                world_sim_dir(probe, "sim1")
            self.assertIn("probe", str(caught.exception))

    def test_reports_cannot_land_on_a_fixture_bind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "world"
            (fixture / "data").mkdir(parents=True)
            (fixture / "world.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(WorldSimError) as caught:
                world_sim_dir(fixture, "sim1")
            self.assertIn("fixture bind", str(caught.exception))


class HonestyTests(unittest.TestCase):
    def test_imagined_numbers_do_not_enter_judge_or_results(self) -> None:
        diag = diagnosis_from_payload(
            "gen_x",
            {
                "alternative": "outcome reward already saturates",
                "experiment": "same freeze, truncate flag only",
                "treatment_arm": "truncate_to_next_event on",
                "control_arm": "full outcome reward",
                "expected_direction": "treatment_lower",
                "alternative_direction": "treatment_higher",
                "expected_if_alternative": "arms tie",
                "margin": 0.05,
                "margin_reason": "intern noise on this freeze stays under five percent",
            },
        )
        with self.assertRaises(WorldSimError):
            judge_probe(diag, treatment=0.169, control=0.171, kind=KIND_WORLD_SIM)
        rec = {
            "claim": CLAIM,
            "probe_kind": KIND_WORLD_SIM,
            "results_status": "imagined",
            "treatment": 0.169,
            "control": 0.171,
            "verdict": "supports",
        }
        result = render_result_to_claim(rec)
        analysis = render_analyze_results(rec)
        self.assertNotIn("0.169", result)
        self.assertNotIn("0.171", result)
        self.assertNotIn("0.169", analysis)
        self.assertIn("WORLD_SIM", result)
        self.assertIn("withheld", analysis)

    def test_run_does_not_rewrite_expected_direction_or_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            idea = _idea(Path(tmp))
            protocol = {
                "expected_direction": "treatment_lower",
                "compute_tier": "host",
            }
            protocol_path = idea / "protocol.json"
            protocol_path.write_text(
                json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
            )
            results_path = idea / "refine-logs" / "EXPERIMENT_RESULTS.md"
            original = "# Experiment results\n\nattested intern: 0.169 vs 0.171\n"
            results_path.write_text(original, encoding="utf-8")
            payload = _run(idea)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["kind"], KIND_WORLD_SIM)
            self.assertTrue(payload["cannot_corroborate"])
            self.assertTrue(payload["cannot_rewrite_expected_direction"])
            self.assertTrue(payload["cannot_block_execute"])
            dest = Path(payload["dest"])
            self.assertEqual(dest.parent.name, WORLD_SIM_FOLDER)
            self.assertTrue((dest / "simulated.json").is_file())
            self.assertNotIn("/probe/", str(dest).replace("\\", "/"))
            meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
            self.assertIn("truncate", meta.get("levers") or [])
            self.assertEqual(meta.get("schema"), "labeled_traces")
            self.assertNotIn("operators", meta)
            written = json.loads(protocol_path.read_text(encoding="utf-8"))
            self.assertEqual(written["expected_direction"], "treatment_lower")
            self.assertEqual(results_path.read_text(encoding="utf-8"), original)
            dump = json.loads((dest / "simulated.json").read_text(encoding="utf-8"))
            self.assertEqual(dump["kind"], KIND_WORLD_SIM)
            self.assertEqual(dump["results_status"], "imagined")

    def test_a_failed_rehearsal_cannot_block_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "ideas" / "missing"
            empty.mkdir(parents=True)
            payload = maybe_world_sim_after_diagnosis(empty, enabled=True)
            self.assertIsNotNone(payload)
            self.assertFalse(payload["ok"])
            self.assertTrue(payload["cannot_block_execute"])
            self.assertEqual(payload["kind"], KIND_WORLD_SIM)


class KnowledgeAdmitTests(unittest.TestCase):
    def test_code_world_models_enter_and_drought_index_does_not(self) -> None:
        cwm = FreshWork(
            title="Code world models for observation-action rollouts",
            published="2026-01-01",
            arxiv_id="2601.00001",
            abstract="A code world model simulates the next observation from actions.",
        )
        drought = FreshWork(
            title="A drought index for obesity surveillance",
            published="2024-01-01",
            arxiv_id="2401.00002",
            abstract="We correlate a drought index with obesity prevalence.",
        )
        brand = FreshWork(
            title="Live-SWE-agent: an open-source coding agent",
            published="2025-01-01",
            arxiv_id="2501.00003",
            abstract="We release Live-SWE-agent trajectories on SWE-bench Verified.",
        )
        kept = papers_for_rehearsal(
            [cwm, drought, brand], claim=CLAIM, mechanism=MECHANISM
        )
        ids = {work.cite_id() for work in kept}
        self.assertIn("2601.00001", ids)
        self.assertNotIn("2401.00002", ids)
        self.assertNotIn("2501.00003", ids)

    def test_a_registered_graph_lever_does_not_admit_trace_literature(self) -> None:
        cwm = FreshWork(
            title="Code world models for observation-action rollouts",
            published="2026-01-01",
            arxiv_id="2601.00001",
            abstract="A code world model simulates the next observation from actions.",
        )
        kept = papers_for_rehearsal(
            [cwm],
            claim=CLAIM,
            mechanism=MECHANISM,
            world_lever="rewire",
            schema="undirected_graph",
        )
        self.assertEqual(kept, [])

    def test_truncate_on_traces_admits_code_world_models(self) -> None:
        cwm = FreshWork(
            title="Code world models for observation-action rollouts",
            published="2026-01-01",
            arxiv_id="2601.00001",
            abstract="A code world model simulates the next observation from actions.",
        )
        kept = papers_for_rehearsal(
            [cwm],
            claim=CLAIM,
            mechanism=MECHANISM,
            world_lever="truncate",
            schema="labeled_traces",
        )
        self.assertEqual([work.cite_id() for work in kept], ["2601.00001"])


class TierTests(unittest.TestCase):
    def test_best_must_dominate_median_and_worst(self) -> None:
        payload = _tiers()
        payload["tiers"]["worst"]["number_ranges"] = {"delta": [0.50, 0.90]}
        with self.assertRaises(WorldSimError) as caught:
            validate_tiers(payload)
        self.assertIn("monotonic", str(caught.exception))

    def test_named_invariants_as_an_object_are_facts_not_an_empty_list(self) -> None:
        # Every Phase 1 answer in the cwm-iclr2027 mission came back with
        # scientific_invariants as an object of named facts and
        # key_assumptions as an object; the list-only reader threw the
        # whole rehearsal away four times.
        payload = _tiers()
        payload["scientific_invariants"] = {
            "paired_design": "both arms see the same ordered patches",
            "experimental_arms": {"treatment": {"replay_gate": True}},
            "schema": "program_state",
        }
        payload["tiers"]["best"]["key_assumptions"] = {
            "shadow_fidelity": "true for nearly all transitions",
        }
        out = validate_tiers(payload)
        self.assertEqual(len(out["scientific_invariants"]), 3)
        self.assertTrue(any(fact.startswith("paired_design:") for fact in out["scientific_invariants"]))
        self.assertTrue(any("replay_gate" in fact for fact in out["scientific_invariants"]))
        self.assertEqual(
            out["tiers"]["best"]["key_assumptions"],
            ["shadow_fidelity: true for nearly all transitions"],
        )

    def test_number_ranges_accept_min_max_objects_and_skip_flags(self) -> None:
        payload = _tiers()
        for name, (lo, hi) in {
            "best": (0.20, 0.40),
            "median": (0.05, 0.15),
            "worst": (-0.05, 0.04),
        }.items():
            payload["tiers"][name]["number_ranges"] = {
                "delta": {"min": lo, "max": hi},
                "claim_thresholds_met": name != "worst",
                "interpretation": "hypothetical ranges, not observations",
            }
        out = validate_tiers(payload)
        self.assertEqual(out["tiers"]["best"]["number_ranges"], {"delta": [0.20, 0.40]})

    def test_a_lower_is_better_metric_may_ascend_across_tiers(self) -> None:
        payload = _tiers()
        payload["tiers"]["best"]["number_ranges"]["gap_points"] = [0, 3]
        payload["tiers"]["median"]["number_ranges"]["gap_points"] = [2, 5]
        payload["tiers"]["worst"]["number_ranges"]["gap_points"] = [5, 12]
        out = validate_tiers(payload)
        self.assertEqual(out["tiers"]["worst"]["number_ranges"]["gap_points"], [5.0, 12.0])

    def test_a_metric_that_zigzags_is_still_refused(self) -> None:
        payload = _tiers()
        payload["tiers"]["best"]["number_ranges"]["rate"] = [0.0, 0.1]
        payload["tiers"]["median"]["number_ranges"]["rate"] = [0.3, 0.5]
        payload["tiers"]["worst"]["number_ranges"]["rate"] = [0.1, 0.2]
        with self.assertRaises(WorldSimError) as caught:
            validate_tiers(payload)
        self.assertIn("rate", str(caught.exception))

    def test_only_flags_and_prose_in_number_ranges_is_no_ranges(self) -> None:
        payload = _tiers()
        payload["tiers"]["best"]["number_ranges"] = {"claim_thresholds_met": True}
        with self.assertRaises(WorldSimError) as caught:
            validate_tiers(payload)
        self.assertIn("number_ranges", str(caught.exception))


class SkipAndWhatIfTests(unittest.TestCase):
    def test_smart_skip_reuses_unchanged_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            idea = _idea(Path(tmp))
            first = _run(idea, smart_skip=True)
            self.assertFalse(first["skipped"])
            second = _run(idea, smart_skip=True)
            self.assertTrue(second["skipped"])
            self.assertEqual(second["kind"], KIND_WORLD_SIM)

    def test_what_if_appends_without_rewriting_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            idea = _idea(Path(tmp))
            plan_path = idea / "refine-logs" / "EXPERIMENT_PLAN.md"
            before = plan_path.read_text(encoding="utf-8")
            extra = WhatIfExperiment(
                name="prefix monitor budget",
                claim="a fail-fast prefix monitor saves queries",
                description="add a query-budget what-if on the same freeze",
                estimated_gpu_hours=1.0,
            )
            elaborations = {
                "best": _elab(0.30)
                + [
                    {
                        "name": "what-if prefix monitor",
                        "plan_id": "what_if",
                        "description": extra.description,
                        "expected_results": {"delta": 0.22},
                        "result_rationale": (
                            "a fail-fast prefix monitor is an extra experiment "
                            "not a rewrite of the registered plan"
                        ),
                    }
                ],
                "median": _elab(0.10),
                "worst": _elab(0.00),
            }
            payload = _run(idea, what_if=[extra], canned_elaborations=elaborations)
            self.assertTrue(payload["ok"])
            self.assertEqual(plan_path.read_text(encoding="utf-8"), before)
            dest = Path(payload["dest"])
            imagined = json.loads(
                (dest / "best" / "imagined_experiments.json").read_text(encoding="utf-8")
            )
            ids = {row["plan_id"] for row in imagined["experiments"]}
            self.assertIn("what_if", ids)
            self.assertIn("block-0", ids)


class CliTests(unittest.TestCase):
    def test_world_sim_command_and_mission_flag(self) -> None:
        parser = build_parser()
        ns = parser.parse_args(["world-sim", "/tmp/idea", "--smart-skip"])
        self.assertEqual(ns.command, "world-sim")
        self.assertTrue(ns.smart_skip)
        research = parser.parse_args(
            ["research", "/tmp/proj", "--topic", "code world models", "--world-sim"]
        )
        self.assertTrue(research.world_sim)
        self.assertEqual(research.world, "auto")
        defaulted = parser.parse_args(
            ["research", "/tmp/proj", "--topic", "code world models"]
        )
        self.assertTrue(defaulted.world_sim)
        skipped = parser.parse_args(
            ["research", "/tmp/proj", "--topic", "code world models", "--no-world-sim"]
        )
        self.assertFalse(skipped.world_sim)


if __name__ == "__main__":
    unittest.main()
