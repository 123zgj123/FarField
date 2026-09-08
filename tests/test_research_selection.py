"""Research experiment ranking is conditional on explicit live alternatives."""
import copy
import math
import unittest

from farfield.extras.openworld import scheduler
from farfield.extras.openworld.actions import ActionInstance, PROBE, SPECS, SURVEY
from farfield.extras.openworld.state import ScientificState


def proposal(left="H1", right="H2"):
    return {
        "hypotheses_compared": [left, right],
        "expected_outcomes": {
            left: {"observable": "retention", "direction": "increase"},
            right: {"observable": "retention", "direction": "unchanged"},
        },
        "nuisance_factors": ["different exposure time"],
        "minimal_discriminating_intervention": "Vary feedback with exposure time held fixed",
        "interpretation_if_positive": "Feedback mechanism remains plausible",
        "interpretation_if_negative": "Feedback mechanism is weakened",
        "interpretation_if_inconclusive": "Improve precision before updating the theory",
        "expected_scientific_gain": 1.0, "transfer_value": 1.0,
        "frontier_unlock": 1.0, "cost": 0.0, "risk": 0.0,
    }


def research_state():
    state = ScientificState(goal="retention")
    state.theories = [{"id": hid} for hid in ("H1", "H2", "H3")]
    state.belief_states = {hid: {"posterior_confidence": .5, "competing_theories": []}
                           for hid in ("H1", "H2", "H3")}
    return state


def experiment(p):
    return ActionInstance(SPECS[PROBE], target="retention", extra={
        "causal_handle": "feedback", "measure": "retention", "freeze": "W1",
        "experiment_proposal": p,
    })


class ResearchSelectionTests(unittest.TestCase):
    def priority(self, p, state):
        self.assertTrue(hasattr(scheduler, "hypothesis_resolution_priority"),
                        "hypothesis-aware selector is missing")
        try:
            return scheduler.hypothesis_resolution_priority(p, state)
        except OverflowError:
            self.fail("invalid numeric input must fail closed, not crash the research loop")

    def test_fixed_action_scores_are_honestly_labeled_heuristics(self):
        result = scheduler.score_action(ActionInstance(SPECS[SURVEY], target="retention"), research_state())
        self.assertNotIn("expected_information_gain", result.scores)
        self.assertIn("heuristic_information_value", result.scores)

    def test_changing_competing_theories_flips_experiment_selection(self):
        state = research_state()
        state.belief_states["H1"]["competing_theories"] = ["H2"]
        a, b = experiment(proposal()), experiment(proposal("H1", "H3"))
        self.assertIs(scheduler.schedule(state, [b, a]).action, a)
        state.belief_states["H1"]["competing_theories"] = ["H3"]
        self.assertIs(scheduler.schedule(state, [a, b]).action, b)

    def test_pair_resolution_uses_replayable_posterior_weights(self):
        state = research_state()
        state.theories = state.theories[:2]
        state.belief_states.pop("H3")
        row = self.priority(proposal(), state)
        self.assertTrue(row["valid"])
        self.assertEqual(row["expected_hypothesis_resolution"], .5)
        self.assertEqual(row["priority"], .5)
        self.assertEqual(row["score_basis"], "proposal_estimate_not_evidence")
        state.belief_states["H2"]["posterior_confidence"] = 0.0
        self.assertEqual(self.priority(proposal(), state)["expected_hypothesis_resolution"], 0.0)

    def test_identical_predictions_do_not_claim_ambiguity_resolution(self):
        p = proposal()
        p["expected_outcomes"]["H2"]["direction"] = "increase"
        self.assertEqual(self.priority(p, research_state())["expected_hypothesis_resolution"], 0.0)

    def test_priority_is_multiplicative_gain_minus_cost_and_risk(self):
        state = research_state()
        state.theories = state.theories[:2]
        p = proposal()
        p.update(expected_scientific_gain=.8, transfer_value=.5, frontier_unlock=.5, cost=.02, risk=.01)
        self.assertAlmostEqual(self.priority(p, state)["priority"], .07)

    def test_malformed_proposals_fail_closed_and_are_not_scheduled(self):
        mutations = [
            ("hypotheses_compared", ["H1", "made-up"]),
            ("hypotheses_compared", ["H1", "H1"]),
            ("expected_outcomes", {"H1": "increase", "H2": "unchanged"}),
            ("expected_outcomes", {"H1": {"observable": "retention", "direction": "inconclusive"},
                                   "H2": {"observable": "retention", "direction": "unchanged"}}),
            ("nuisance_factors", "none"),
            ("minimal_discriminating_intervention", ""),
            ("interpretation_if_inconclusive", ""),
            ("expected_scientific_gain", math.nan),
            ("transfer_value", math.inf),
            ("frontier_unlock", "1"),
            ("risk", -1),
            ("cost", True),
            ("cost", 10 ** 1000),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                p = proposal()
                p[key] = value
                result = self.priority(p, research_state())
                self.assertFalse(result["valid"])
                self.assertTrue(result["reason"])
                self.assertIsNone(scheduler.schedule(research_state(), [experiment(p)]))

    def test_finite_inputs_cannot_overflow_to_infinite_priority(self):
        p = proposal()
        p.update(cost=1e308, risk=1e308)
        self.assertFalse(self.priority(p, research_state())["valid"])

    def test_structured_interpretations_are_supported_without_relaxing_contract(self):
        p = proposal()
        p["interpretations"] = {outcome: p.pop("interpretation_if_" + outcome)
                                for outcome in ("positive", "negative", "inconclusive")}
        self.assertTrue(self.priority(p, research_state())["valid"])
        del p["interpretations"]["negative"]
        self.assertFalse(self.priority(p, research_state())["valid"])

    def test_unlike_observables_cannot_be_treated_as_discrimination(self):
        p = proposal()
        p["expected_outcomes"]["H2"]["observable"] = "latency"
        self.assertFalse(self.priority(p, research_state())["valid"])

    def test_bad_belief_is_not_replaced_by_generated_confidence(self):
        state = research_state()
        state.belief_states["H2"]["posterior_confidence"] = float("nan")
        self.assertFalse(self.priority(proposal(), state)["valid"])

    def test_selector_does_not_mutate_proposal_or_beliefs(self):
        state, p = research_state(), proposal()
        old_p, old_state = copy.deepcopy(p), copy.deepcopy(state.to_dict())
        self.priority(p, state)
        self.assertEqual(p, old_p)
        self.assertEqual(state.to_dict(), old_state)


if __name__ == "__main__":
    unittest.main()
