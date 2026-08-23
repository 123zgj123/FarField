"""Model-written judges must earn the right to kill, and cannot escape the box."""

from __future__ import annotations

import unittest

from farfield.extras.predicates import (
    KERNEL_PREDICATES,
    Predicate,
    PredicateError,
    PredicateRefused,
    compile_predicate,
    criteria_lines,
    gate_report,
    paraphrase,
    run,
    select_marginal,
)


def card(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "claim": "combining a with b reduces query time to O(log n)",
        "mechanism": "a's index structure amortizes b's updates",
        "prediction": "update time drops below 5 ms at n = 10^6",
        "falsifier": "a benchmark where the combined structure is slower",
        "concept_a": "succinct data structure",
        "concept_b": "dynamic connectivity",
        "alienness": 0.7,
        "plausibility_signal": 0.1,
    }
    return base | overrides


def compiled(source: str, name: str = "candidate") -> Predicate:
    return compile_predicate(name, source, provenance="model:test")


class CompilerTests(unittest.TestCase):
    def test_a_simple_text_predicate_compiles_and_runs(self) -> None:
        predicate = compiled(
            "def check(card):\n"
            "    return 'log' in str(card['claim']).lower()\n"
        )
        self.assertTrue(run(predicate, card()))
        self.assertFalse(run(predicate, card(claim="a helps b")))

    def test_an_import_is_refused(self) -> None:
        for source in (
            "import os\ndef check(card):\n    return True\n",
            "def check(card):\n    import os\n    return True\n",
            "from os import path\ndef check(card):\n    return True\n",
        ):
            with self.assertRaises(PredicateRefused):
                compiled(source)

    def test_a_while_loop_is_refused(self) -> None:
        with self.assertRaises(PredicateRefused):
            compiled("def check(card):\n    while True:\n        pass\n")

    def test_underscore_attributes_and_names_are_refused(self) -> None:
        # The classic escapes all route through dunders; the wall is one rule.
        for source in (
            "def check(card):\n    return card.__class__ is dict\n",
            "def check(card):\n    x = __import__\n    return True\n",
            "def check(card):\n    return ().__class__.__bases__ is not None\n",
        ):
            with self.assertRaises(PredicateRefused):
                compiled(source)

    def test_the_function_must_be_check_of_card_alone(self) -> None:
        with self.assertRaises(PredicateRefused):
            compiled("def judge(card):\n    return True\n")
        with self.assertRaises(PredicateRefused):
            compiled("def check(card, graph):\n    return True\n")
        with self.assertRaises(PredicateRefused):
            compiled("def check(card):\n    return True\n\nx = 1\n")

    def test_forbidden_builtins_are_simply_absent(self) -> None:
        predicate = compiled("def check(card):\n    return open('x') is None\n")
        with self.assertRaises(PredicateError):
            run(predicate, card())

    def test_re_and_math_are_available(self) -> None:
        predicate = compiled(
            "def check(card):\n"
            "    return bool(re.search(r'\\d', str(card['prediction'])))"
            " and math.floor(1.5) == 1\n"
        )
        self.assertTrue(run(predicate, card()))


class RuntimeTests(unittest.TestCase):
    def test_a_runaway_loop_dies_by_budget_not_by_hanging(self) -> None:
        predicate = compiled(
            "def check(card):\n"
            "    total = 0\n"
            "    for i in range(10**9):\n"
            "        total = total + 1\n"
            "    return True\n"
        )
        with self.assertRaises(PredicateError) as caught:
            run(predicate, card())
        self.assertIn("budget", str(caught.exception))

    def test_a_crash_is_the_predicates_failure_not_a_kill(self) -> None:
        predicate = compiled("def check(card):\n    return 1 // 0 == 0\n")
        with self.assertRaises(PredicateError):
            run(predicate, card())

    def test_a_non_boolean_verdict_is_refused(self) -> None:
        predicate = compiled("def check(card):\n    return 1\n")
        with self.assertRaises(PredicateError):
            run(predicate, card())

    def test_the_card_view_is_fields_only(self) -> None:
        # Whatever extra keys a caller's card row carries, the predicate sees
        # the declared view; a judge cannot key on bookkeeping like `killed`.
        predicate = compiled("def check(card):\n    return 'killed' in card\n")
        self.assertFalse(run(predicate, card(killed=True)))


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bench = [
            card(claim="x strictly dominates y on regret"),
            card(claim="a may relate to b somehow", prediction="future work"),
            card(claim="c cannot beat d's lower bound"),
        ]
        self.bad = [card(claim="a is connected to b", prediction="tbd")]

    def test_a_predicate_that_passes_everything_fails_the_fires_gate(self) -> None:
        report = gate_report(
            compiled("def check(card):\n    return True\n"), self.bench, self.bad
        )
        self.assertFalse(report["gates"]["fires_on_history"])
        self.assertFalse(report["installable"])

    def test_a_predicate_that_kills_everything_fails_the_spares_gate(self) -> None:
        report = gate_report(
            compiled("def check(card):\n    return False\n"), self.bench, self.bad
        )
        self.assertTrue(report["gates"]["fires_on_history"])
        self.assertFalse(report["gates"]["spares_some_history"])
        self.assertFalse(report["installable"])

    def test_missing_a_known_bad_card_closes_the_gate(self) -> None:
        lenient = compiled(
            "def check(card):\n"
            "    return 'tbd' not in str(card['claim'])\n"
        )
        report = gate_report(lenient, self.bench, self.bad)
        self.assertFalse(report["gates"]["catches_known_bad"])

    def test_a_predicate_that_crashes_anywhere_is_not_installable(self) -> None:
        brittle = compiled(
            "def check(card):\n"
            "    return len(card['claim']) // len(str(card['prediction'])"
            ".replace('future work', '')) >= 0\n"
        )
        report = gate_report(brittle, self.bench, self.bad)
        self.assertFalse(report["gates"]["never_errors"])
        self.assertFalse(report["installable"])

    def test_a_discriminating_predicate_passes_all_gates(self) -> None:
        report = gate_report(
            KERNEL_PREDICATES[1], self.bench, self.bad
        )
        self.assertTrue(report["installable"], report)


class ParaphraseGateTests(unittest.TestCase):
    """The fourth gate: a verdict must survive a synonym rewording.

    The first installed generation of model judges died of exactly this —
    'demonstrates' passed, 'presents' was killed — so the failure mode is
    empirical, not hypothetical.
    """

    def test_paraphrase_swaps_verbs_and_nothing_else(self) -> None:
        original = card(
            prediction="later work demonstrates a faster structure",
            mechanism="the index enables batched updates",
        )
        rewritten = paraphrase(original)
        self.assertIn("presents", rewritten["prediction"])
        self.assertNotIn("demonstrates", rewritten["prediction"])
        self.assertIn("lets", rewritten["mechanism"])
        for untouched in ("falsifier", "concept_a", "concept_b", "alienness"):
            self.assertEqual(rewritten[untouched], original[untouched])

    def test_word_boundaries_protect_terms_of_art(self) -> None:
        original = card(claim="the showstopper is the reporter lower bound")
        self.assertEqual(paraphrase(original)["claim"], original["claim"])

    def test_a_future_marker_phrase_list_fails_the_paraphrase_gate(self) -> None:
        """The second generation's blind spot: 'later work' spared, 'a later
        paper' killed. Markers of futurity are wording, and the gate must
        treat them as such."""
        marker_list = compiled(
            "def check(card):\n"
            "    return 'later work' in str(card['prediction'] or '').lower()\n"
        )
        bench = [
            card(prediction="later work will report a 20% drop in query time"),
            card(prediction="the oracle answers in O(1)"),
        ]
        bad = [card(prediction="this will become an active research area")]
        report = gate_report(marker_list, bench, bad)
        self.assertFalse(report["gates"]["verdict_survives_paraphrase"])
        self.assertFalse(report["installable"])

    def test_a_verb_whitelist_judge_fails_the_paraphrase_gate(self) -> None:
        whitelist = compiled(
            "def check(card):\n"
            "    return 'demonstrate' in str(card['prediction'] or '').lower()\n"
        )
        bench = [
            card(prediction="later work demonstrates the oracle"),
            card(prediction="the oracle answers in O(1)"),
        ]
        bad = [card(prediction="this will become an active research area")]
        report = gate_report(whitelist, bench, bad)
        self.assertTrue(report["gates"]["fires_on_history"])
        self.assertTrue(report["gates"]["spares_some_history"])
        self.assertTrue(report["gates"]["catches_known_bad"])
        self.assertFalse(report["gates"]["verdict_survives_paraphrase"])
        self.assertFalse(report["installable"])
        self.assertEqual(report["paraphrase_flips"], 1)

    def test_kernel_predicates_survive_the_paraphrase_gate(self) -> None:
        bench = [
            card(
                claim="a strictly dominates b on regret",
                prediction="later work demonstrates a regret bound of O(log T)",
            ),
            card(
                claim="a may relate to b somehow",
                prediction="future work will explore this",
            ),
        ]
        bad = [card(claim="a is connected to b", prediction="tbd")]
        for kernel in KERNEL_PREDICATES:
            report = gate_report(kernel, bench, bad)
            self.assertTrue(
                report["gates"]["verdict_survives_paraphrase"],
                (kernel.name, report),
            )

    def test_criteria_lines_carry_name_and_rationale(self) -> None:
        lines = criteria_lines(KERNEL_PREDICATES)
        self.assertEqual(len(lines), len(KERNEL_PREDICATES))
        for line, kernel in zip(lines, KERNEL_PREDICATES):
            self.assertIn(kernel.name, line)
            self.assertIn(kernel.rationale, line)


class MarginalGainTests(unittest.TestCase):
    def report(self, name: str, kills: list[int], installable: bool = True) -> dict:
        return {
            "predicate": name,
            "bench_kills": kills,
            "installable": installable,
        }

    def test_a_redundant_judge_is_refused_even_when_sound(self) -> None:
        installed, log = select_marginal(
            [
                self.report("wide", [0, 1, 2]),
                self.report("subset", [1, 2]),
            ]
        )
        self.assertEqual(installed, ["wide"])
        refused = [entry for entry in log if not entry["installed"]]
        self.assertEqual(refused[0]["predicate"], "subset")
        self.assertIn("marginal", refused[0]["reason"])

    def test_complementary_judges_are_both_installed(self) -> None:
        installed, _ = select_marginal(
            [self.report("a", [0, 1]), self.report("b", [2])]
        )
        self.assertEqual(sorted(installed), ["a", "b"])

    def test_ties_go_to_the_sharper_judge(self) -> None:
        # After `wide` is installed, `blunt` and `sharp` both add exactly card
        # 4; the one whose total kill set is smaller wins, because a broader
        # judge bringing the same new coverage is blunter.
        installed, _ = select_marginal(
            [
                self.report("wide", [0, 1, 2, 3]),
                self.report("blunt", [0, 1, 4]),
                self.report("sharp", [4]),
            ]
        )
        self.assertEqual(installed, ["wide", "sharp"])

    def test_an_ungated_candidate_never_enters_selection(self) -> None:
        installed, log = select_marginal(
            [self.report("bad", [0], installable=False)]
        )
        self.assertEqual(installed, [])
        self.assertEqual(log, [])


class KernelPredicateTests(unittest.TestCase):
    """The P4 brief, checked against the kind of text the pools actually hold."""

    def test_a_prediction_with_no_quantity_is_killed(self) -> None:
        vague = card(prediction="this combination will inspire further study")
        self.assertFalse(run(KERNEL_PREDICATES[0], vague))

    def test_a_prediction_naming_a_quantity_passes(self) -> None:
        for text in (
            "query time drops to O(log n) per update",
            "accuracy improves by 3 points on the benchmark",
            "the approximation ratio falls below 1.5",
        ):
            self.assertTrue(
                run(KERNEL_PREDICATES[0], card(prediction=text)), text
            )

    def test_a_hedged_claim_with_no_commitment_is_killed(self) -> None:
        for text in (
            "a may be related to b in interesting ways",
            "combining these could potentially help downstream tasks",
        ):
            self.assertFalse(run(KERNEL_PREDICATES[1], card(claim=text)), text)

    def test_a_committed_claim_passes(self) -> None:
        for text in (
            "the combined structure needs at most O(n log n) space",
            "no deterministic algorithm can beat this lower bound",
            "the reduction implies a 2-approximation is tight",
        ):
            self.assertTrue(run(KERNEL_PREDICATES[1], card(claim=text)), text)

    def test_the_kernel_predicates_compile_through_the_public_gate(self) -> None:
        for predicate in KERNEL_PREDICATES:
            twin = compile_predicate(
                predicate.name, predicate.source, provenance="kernel"
            )
            self.assertTrue(run(twin, card()))


if __name__ == "__main__":
    unittest.main()
