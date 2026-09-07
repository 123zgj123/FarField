"""Diagnosis schema teeth and the arithmetic verdict."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from farfield.extras.diagnose import (
    Diagnosis,
    DiagnosisRefused,
    diagnosis_from_payload,
    judge_probe,
    judge_replicated,
    write_diagnosis,
)


def payload(**overrides):
    base = {
        "alternative": "the speedup comes from caching, not the encoding",
        "experiment": "same query log, cache disabled in both arms",
        "treatment_arm": "succinct encoding on, cache off",
        "control_arm": "plain array, cache off",
        "expected_direction": "treatment_lower",
        "alternative_direction": "treatment_higher",
        "expected_if_alternative": "the two arms tie once the cache is off",
        "margin": 0.05,
        "margin_reason": "constant-factor noise in the toy runner stays under five percent",
    }
    base.update(overrides)
    return base


def diagnosis(**overrides) -> Diagnosis:
    return diagnosis_from_payload("gen_x", payload(**overrides))


class SchemaTests(unittest.TestCase):
    def test_a_full_diagnosis_is_accepted(self) -> None:
        diag = diagnosis()
        self.assertEqual(diag.expected_direction, "treatment_lower")
        self.assertEqual(diag.margin, 0.05)

    def test_identical_arms_are_refused(self) -> None:
        with self.assertRaises(DiagnosisRefused) as caught:
            diagnosis(control_arm="succinct encoding on, cache off")
        self.assertIn("identical", caught.exception.record.unlock_condition)

    def test_an_empty_alternative_is_refused(self) -> None:
        with self.assertRaises(DiagnosisRefused):
            diagnosis(alternative="")

    def test_a_direction_outside_the_register_is_refused(self) -> None:
        with self.assertRaises(DiagnosisRefused):
            diagnosis(expected_direction="whatever looks good")

    def test_a_degenerate_margin_is_refused(self) -> None:
        with self.assertRaises(DiagnosisRefused):
            diagnosis(margin=0.0)
        with self.assertRaises(DiagnosisRefused):
            diagnosis(margin=0.9)

    def test_a_non_discriminating_design_is_refused(self) -> None:
        # The core of discriminative preregistration: if mechanism and
        # alternative predict the same direction, the experiment can only
        # flatter the mechanism — a live batch produced 14/21 supports from
        # exactly such "having it beats not having it" designs.
        with self.assertRaises(DiagnosisRefused) as ctx:
            diagnosis(alternative_direction="treatment_lower")
        self.assertIn("SAME direction", ctx.exception.record.unlock_condition)

    def test_a_missing_alternative_direction_is_refused(self) -> None:
        with self.assertRaises(DiagnosisRefused):
            diagnosis(alternative_direction="")

    def test_a_missing_margin_reason_is_refused(self) -> None:
        # 23/23 live diagnoses copied margin=0.05 from the template; the
        # number must now come with a sentence of thought behind it.
        with self.assertRaises(DiagnosisRefused) as ctx:
            diagnosis(margin_reason="")
        self.assertIn("margin", ctx.exception.record.unlock_condition)


class VerdictTests(unittest.TestCase):
    def test_the_preregistered_direction_supports(self) -> None:
        verdict = judge_probe(diagnosis(), treatment=80.0, control=120.0)
        self.assertEqual(verdict["verdict"], "supports")

    def test_the_opposite_direction_weakens(self) -> None:
        verdict = judge_probe(diagnosis(), treatment=150.0, control=120.0)
        self.assertEqual(verdict["verdict"], "weakens")

    def test_inside_the_margin_is_uninformative(self) -> None:
        verdict = judge_probe(diagnosis(), treatment=119.0, control=120.0)
        self.assertEqual(verdict["verdict"], "uninformative")

    def test_the_model_cannot_flip_a_verdict_after_the_fact(self) -> None:
        # The verdict is a pure function of the pre-registered fields and
        # the two numbers; the same run under the opposite registration
        # flips, which is exactly why registration happens before the run.
        supports = judge_probe(diagnosis(), treatment=80.0, control=120.0)
        weakens = judge_probe(
            diagnosis(
                expected_direction="treatment_higher",
                alternative_direction="treatment_lower",
            ),
            treatment=80.0,
            control=120.0,
        )
        self.assertEqual(supports["verdict"], "supports")
        self.assertEqual(weakens["verdict"], "weakens")

    def test_world_sim_numbers_cannot_become_a_verdict(self) -> None:
        from farfield.extras.worldsim import WorldSimError

        with self.assertRaises(WorldSimError) as caught:
            judge_probe(diagnosis(), treatment=0.9, control=0.1, kind="WORLD_SIM")
        self.assertIn("WORLD_SIM", str(caught.exception))
        self.assertIn("judge_probe", str(caught.exception))


class ReplicatedVerdictTests(unittest.TestCase):
    """The confirmation tier: arithmetic over executor-owned reruns.

    The pairs come from the trusted executor rerunning the registered
    script — the model never reports its own replicates. The whole 95%
    interval must clear the margin.
    """

    def test_tight_replicates_in_the_preregistered_direction_support(self) -> None:
        verdict = judge_replicated(
            diagnosis(),
            [(80.0, 120.0), (81.0, 119.0), (79.0, 121.0),
             (80.5, 120.5), (79.5, 119.5)],
        )
        self.assertEqual(verdict["verdict"], "supports")
        self.assertTrue(verdict["replicated"])
        self.assertEqual(verdict["n"], 5)
        self.assertLess(verdict["ci_high"], -0.05)

    def test_a_mean_that_clears_while_the_interval_straddles_cannot_confirm(self) -> None:
        # The reason this judge exists: seed-averaged arms of (80, 118, 90)
        # vs 120 give a scalar separation of -0.2 — a clean scalar
        # "supports" — while the per-seed spread leaves the 95% interval
        # straddling the margin. The scalar filter may pass; the
        # confirmation tier must not.
        pairs = [(80.0, 120.0), (118.0, 120.0), (90.0, 120.0)]
        scalar = judge_probe(diagnosis(), treatment=96.0, control=120.0)
        self.assertEqual(scalar["verdict"], "supports")
        verdict = judge_replicated(diagnosis(), pairs)
        self.assertEqual(verdict["verdict"], "uninformative")
        self.assertIn("does not clear", verdict["reason"])

    def test_replicates_against_the_direction_weaken(self) -> None:
        verdict = judge_replicated(
            diagnosis(),
            [(150.0, 120.0), (151.0, 119.0), (149.0, 121.0), (150.0, 120.5)],
        )
        self.assertEqual(verdict["verdict"], "weakens")

    def test_too_few_replicates_are_uninformative_by_name(self) -> None:
        verdict = judge_replicated(diagnosis(), [(80.0, 120.0), (81.0, 119.0)])
        self.assertEqual(verdict["verdict"], "uninformative")
        self.assertFalse(verdict["replicated"])
        self.assertIn("at least 3", verdict["reason"])

    def test_a_non_finite_arm_is_a_refusal_not_a_guess(self) -> None:
        verdict = judge_replicated(
            diagnosis(),
            [(80.0, 120.0), (float("inf"), 120.0), (79.0, 121.0)],
        )
        self.assertEqual(verdict["verdict"], "uninformative")
        self.assertFalse(verdict["replicated"])

    def test_identical_runs_are_invariance_not_stability(self) -> None:
        # Every seed produced the same numbers: either the effect is
        # deterministic (nothing was varied) or the script ignored its
        # seed — the two are indistinguishable from the outside, so the
        # confirmation tier refuses to tell them apart in the claim's
        # favour.
        verdict = judge_replicated(diagnosis(), [(80.0, 120.0)] * 5)
        self.assertEqual(verdict["verdict"], "uninformative")
        self.assertTrue(verdict["invariant"])
        self.assertIn("invariant", verdict["reason"])


class EchoClient:
    """Returns a fixed valid diagnosis and records the prompt it saw."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.prompts.append(prompt)
        return SimpleNamespace(
            text=json.dumps(payload()),
            assert_usable=lambda: None,
        )


CARD = SimpleNamespace(
    card_id="gen_x",
    claim="a claim",
    mechanism="a mechanism",
    prediction="a prediction",
)


class ScriptedClient:
    def __init__(self, answers: list[dict]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.prompts.append(prompt)
        return SimpleNamespace(
            text=json.dumps(self.answers.pop(0)),
            assert_usable=lambda: None,
        )


class RetryTests(unittest.TestCase):
    def test_a_non_discriminating_design_earns_one_quoted_retry(self) -> None:
        # A rejected registration is a schema error, not a scientific
        # verdict: the writer sees exactly why and tries again once.
        client = ScriptedClient([
            payload(alternative_direction="treatment_lower"),
            payload(),
        ])
        diag = write_diagnosis(client, CARD, "a topic")
        self.assertEqual(diag.alternative_direction, "treatment_higher")
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("was rejected", client.prompts[1])
        self.assertIn("SAME direction", client.prompts[1])

    def test_the_final_refusal_still_propagates(self) -> None:
        client = ScriptedClient([
            payload(alternative_direction="treatment_lower"),
            payload(alternative_direction="treatment_lower"),
        ])
        with self.assertRaises(DiagnosisRefused):
            write_diagnosis(client, CARD, "a topic")
        self.assertEqual(len(client.prompts), 2)


class FollowupTests(unittest.TestCase):
    def test_an_uninformative_earlier_probe_is_quoted_back(self) -> None:
        # The spiral for stalled speculative lines: the next diagnosis of
        # the same hypothesis sees the design that failed to separate and
        # must sharpen it, not repeat it.
        client = EchoClient()
        write_diagnosis(
            client,
            CARD,
            "a topic",
            prior_probe={
                "experiment": "count comparisons on ten random instances",
                "verdict": "uninformative",
            },
        )
        self.assertIn("count comparisons on ten random instances", client.prompts[0])
        self.assertIn("Do not repeat that design", client.prompts[0])
        self.assertIn("chase a win", client.prompts[0])

    def test_an_unnamed_competing_lever_is_quoted_back(self) -> None:
        client = EchoClient()
        write_diagnosis(
            client,
            CARD,
            "a topic",
            prior_probe={
                "experiment": "count comparisons on ten random instances",
                "verdict": "uninformative",
                "alternative": "the speedup comes from caching",
                "world_lever": "dropout",
                "world_levers": ["dropout", "hub_removal"],
            },
        )
        prompt = client.prompts[0]
        self.assertIn("another declared lever of this world", prompt)
        self.assertIn("hub_removal", prompt)
        self.assertIn("the speedup comes from caching", prompt)
        named = EchoClient()
        write_diagnosis(
            named,
            CARD,
            "a topic",
            prior_probe={
                "experiment": "count comparisons on ten random instances",
                "verdict": "uninformative",
                "alternative": "hub_removal would also shrink the giant component",
                "world_lever": "dropout",
                "world_levers": ["dropout", "hub_removal"],
            },
        )
        self.assertNotIn("another declared lever of this world", named.prompts[0])

    def test_failed_experiments_are_quoted_without_arm_numbers(self) -> None:
        client = EchoClient()
        write_diagnosis(
            client,
            CARD,
            "a topic",
            prior_probe={
                "experiment": "count comparisons on ten random instances",
                "verdict": "uninformative",
            },
            failed_experiments=["time both arms on a cache-hot log"],
        )
        prompt = client.prompts[0]
        self.assertIn("time both arms on a cache-hot log", prompt)
        self.assertNotIn("treatment: 80", prompt)
        self.assertNotIn("control: 120", prompt)
        self.assertIn("none are provided", prompt)

    def test_a_settled_earlier_probe_adds_nothing(self) -> None:
        client = EchoClient()
        write_diagnosis(
            client,
            CARD,
            "a topic",
            prior_probe={"experiment": "an old design", "verdict": "supports"},
        )
        self.assertNotIn("an old design", client.prompts[0])
        write_diagnosis(client, CARD, "a topic", prior_probe=None)
        self.assertEqual(client.prompts[0], client.prompts[1])

    def test_a_bound_world_reaches_the_diagnosis_prompt(self) -> None:
        from farfield.extras.world import load_catalog
        from pathlib import Path

        world = load_catalog(Path(__file__).resolve().parents[1])["path-trace"]
        client = EchoClient()
        write_diagnosis(client, CARD, "a topic")
        write_diagnosis(client, CARD, "a topic", world=None)
        self.assertEqual(client.prompts[0], client.prompts[1])
        write_diagnosis(client, CARD, "a topic", world=world)
        self.assertIn("frozen experimental world", client.prompts[2])
        self.assertIn("path-trace", client.prompts[2])
        self.assertIn("data/path.json", client.prompts[2])
        self.assertNotIn("frozen experimental world", client.prompts[0])

    def test_a_requirement_without_a_freeze_does_not_invent_a_world(self) -> None:
        client = EchoClient()
        write_diagnosis(
            client,
            CARD,
            "a topic",
            requirement={"object_type": "formula"},
        )
        self.assertIn("Do not construct a GENERATED stand-in", client.prompts[0])
        self.assertIn("formula", client.prompts[0])
        self.assertIn("Do not invent a dataset", client.prompts[0])
        self.assertNotIn("CONSTRUCT a GENERATED world", client.prompts[0])

    def test_a_committed_program_reaches_the_diagnosis_prompt(self) -> None:
        client = EchoClient()
        write_diagnosis(client, CARD, "a topic")
        write_diagnosis(client, CARD, "a topic", program="")
        self.assertEqual(client.prompts[0], client.prompts[1])
        write_diagnosis(
            client,
            CARD,
            "a topic",
            program=("Continue this line: keep the cache",),
        )
        self.assertIn("committed research program", client.prompts[2].lower())
        self.assertIn("keep the cache", client.prompts[2])
        self.assertNotIn("committed research program", client.prompts[0].lower())

    def test_retrieved_papers_become_runnable_baselines_not_related_work(self) -> None:
        from farfield.extras.livefeed import FreshWork

        paper = FreshWork(
            title="A wavelet baseline",
            published="2026-01-01",
            arxiv_id="2601.00001v1",
            abstract="We propose a wavelet counter as the control.",
        )
        client = EchoClient()
        write_diagnosis(client, CARD, "a topic")
        write_diagnosis(client, CARD, "a topic", works=[paper])
        self.assertNotIn("Papers retrieved this mission", client.prompts[0])
        self.assertIn("Papers retrieved this mission", client.prompts[1])
        self.assertIn("2601.00001v1", client.prompts[1])
        self.assertIn("not related-work", client.prompts[1])
        with self.assertRaises(DiagnosisRefused):
            diagnosis(baseline_cite_id="invented-id", _allowed_baseline_ids={"2601.00001v1"})


if __name__ == "__main__":
    unittest.main()
