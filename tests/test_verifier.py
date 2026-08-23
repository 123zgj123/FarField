"""The Verifier meta loop: shadow judges earn lethality on held-out missions.

τ_judge in executable form. A shadow judge watches, its flags are logged,
and promotion happens only when both the nomination half and the held-out
half show it anticipated WORLD-weakens without ever flagging a
WORLD-supported card. Elo and SYNTHETIC verdicts never enter.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from farfield.extras.routing import _digest, append_mission
from farfield.extras.verifier import (
    SHADOW_CAP,
    VerifierError,
    add_shadow,
    compile_bench,
    judge_stats,
    load_bench,
    maybe_promote,
    propose_shadow,
    shadow_flags,
    tau_judge_report,
)


FLAGS_MAGIC = '''
def check(card):
    return "magic" not in str(card["claim"]).lower()
'''

ALWAYS_CRASHES = '''
def check(card):
    return 1 / (len(str(card["claim"])) - len(str(card["claim"]))) > 0
'''


def mission(cards: list[dict], *, shadow_errors: list[str] | None = None) -> dict:
    return {
        "topic_digest": "t",
        "corpus": "c",
        "anchor": "a",
        "cards": cards,
        "shadow_errors": shadow_errors or [],
    }


def card(
    *,
    shadow: list[str] | None = None,
    verdict: str | None = None,
    kind: str = "SYNTHETIC",
    killed_by: list[str] | None = None,
) -> dict:
    return {
        "operator": "op",
        "track": "farfield",
        "killed": bool(killed_by),
        "killed_by": killed_by or [],
        "shadow_kills": shadow or [],
        "verdict": verdict,
        "probe_kind": kind,
    }


class BenchTests(unittest.TestCase):
    def test_a_shadow_judge_is_compiled_through_the_whitelist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judges.json"
            report = add_shadow(
                path,
                name="no_magic",
                source=FLAGS_MAGIC,
                provenance="test",
                rationale="magic is not a mechanism",
            )
            self.assertEqual(report["bench"], "shadow")
            bench = load_bench(path)
            self.assertEqual(len(bench["shadow"]), 1)
            self.assertEqual(bench["lethal"], [])

    def test_a_duplicate_name_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judges.json"
            add_shadow(path, name="no_magic", source=FLAGS_MAGIC, provenance="test")
            with self.assertRaises(VerifierError):
                add_shadow(
                    path, name="no_magic", source=FLAGS_MAGIC, provenance="test"
                )

    def test_an_unwhitelisted_source_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judges.json"
            with self.assertRaises(VerifierError):
                add_shadow(
                    path,
                    name="evil",
                    source="import os\ndef check(card):\n    return True\n",
                    provenance="test",
                )
            self.assertEqual(load_bench(path)["shadow"], [])

    def test_an_edited_bench_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judges.json"
            add_shadow(path, name="no_magic", source=FLAGS_MAGIC, provenance="test")
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("no_magic", "yes_magic"), encoding="utf-8")
            with self.assertRaises(VerifierError):
                load_bench(path)


class ShadowFlagTests(unittest.TestCase):
    def test_a_flag_is_recorded_and_never_kills(self) -> None:
        judges = compile_bench(
            [{"name": "no_magic", "source": FLAGS_MAGIC, "provenance": "test"}]
        )
        flags, errors = shadow_flags(
            judges, {"claim": "a magic speedup", "mechanism": ""}
        )
        self.assertEqual(flags, ["no_magic"])
        self.assertEqual(errors, [])

    def test_a_crashing_shadow_judge_is_its_own_problem(self) -> None:
        judges = compile_bench(
            [{"name": "crash", "source": ALWAYS_CRASHES, "provenance": "test"}]
        )
        flags, errors = shadow_flags(judges, {"claim": "anything"})
        self.assertEqual(flags, [])
        self.assertEqual(errors, ["crash"])


class SettlementTests(unittest.TestCase):
    def _log(self, path: Path, missions: list[dict]) -> None:
        for record in missions:
            append_mission(path, record)

    def _bench(self, path: Path) -> None:
        add_shadow(path, name="no_magic", source=FLAGS_MAGIC, provenance="test")

    def test_heldout_world_weakens_catches_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            bench = Path(tmp) / "judges.json"
            self._bench(bench)
            catch = card(shadow=["no_magic"], verdict="weakens", kind="WORLD")
            self._log(log, [mission([catch]) for _ in range(6)])
            decision = maybe_promote(log, bench)
            self.assertEqual(decision["promoted"], ["no_magic"])
            after = load_bench(bench)
            self.assertEqual(after["shadow"], [])
            self.assertEqual(after["lethal"][0]["name"], "no_magic")
            self.assertIn("promoted_by", after["lethal"][0])

    def test_a_flag_on_a_world_support_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            bench = Path(tmp) / "judges.json"
            self._bench(bench)
            catch = card(shadow=["no_magic"], verdict="weakens", kind="WORLD")
            false_kill = card(shadow=["no_magic"], verdict="supports", kind="WORLD")
            self._log(
                log,
                [mission([catch]) for _ in range(5)] + [mission([false_kill])],
            )
            decision = maybe_promote(log, bench)
            self.assertEqual(decision["promoted"], [])
            self.assertIn("WORLD-supported", decision["rejected"][0]["reason"])
            self.assertEqual(len(load_bench(bench)["shadow"]), 1)

    def test_synthetic_verdicts_settle_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            bench = Path(tmp) / "judges.json"
            self._bench(bench)
            synthetic = card(
                shadow=["no_magic"], verdict="weakens", kind="SYNTHETIC"
            )
            self._log(log, [mission([synthetic]) for _ in range(6)])
            decision = maybe_promote(log, bench)
            self.assertEqual(decision["promoted"], [])
            self.assertIn("nomination half", decision["rejected"][0]["reason"])

    def test_a_judge_that_errored_is_not_promotable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            bench = Path(tmp) / "judges.json"
            self._bench(bench)
            catch = card(shadow=["no_magic"], verdict="weakens", kind="WORLD")
            rows = [mission([catch]) for _ in range(6)]
            rows[3] = mission([catch], shadow_errors=["no_magic"])
            self._log(log, rows)
            decision = maybe_promote(log, bench)
            self.assertEqual(decision["promoted"], [])
            self.assertIn("errored", decision["rejected"][0]["reason"])

    def test_a_short_log_settles_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            bench = Path(tmp) / "judges.json"
            self._bench(bench)
            catch = card(shadow=["no_magic"], verdict="weakens", kind="WORLD")
            self._log(log, [mission([catch]) for _ in range(5)])
            self.assertIsNone(maybe_promote(log, bench))

    def test_an_empty_bench_settles_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            bench = Path(tmp) / "judges.json"
            self._log(log, [mission([card()]) for _ in range(6)])
            self.assertIsNone(maybe_promote(log, bench))

    def test_legacy_missions_do_not_settle_judges(self) -> None:
        # Pre-schema records cannot vote here either: their verdicts
        # cannot prove an attested world, so they can neither promote
        # nor burn a shadow judge.
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            bench = Path(tmp) / "judges.json"
            self._bench(bench)
            catch = card(shadow=["no_magic"], verdict="weakens", kind="WORLD")
            rows = [mission([catch]) for _ in range(6)]
            log.write_text(
                json.dumps({"missions": rows, "digest": _digest(rows)}),
                encoding="utf-8",
            )
            self.assertIsNone(maybe_promote(log, bench))


class Proposer:
    """A fake LLM client that answers the shadow-judge prompt verbatim."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.prompts.append(prompt)
        return SimpleNamespace(text=self.text)


GOOD_PROPOSAL = json.dumps(
    {
        "name": "no_magic_claims",
        "source": FLAGS_MAGIC,
        "rationale": "a magic word is not a mechanism",
    }
)

FAILED_VIEW = {"claim": "a magic speedup appears", "mechanism": "magic"}
PASSING_VIEW = {"claim": "a plain logarithmic bound", "mechanism": "partitioning"}


class ProposalTests(unittest.TestCase):
    def test_a_good_proposal_lands_on_the_shadow_bench_powerless(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "judges.json"
            report = propose_shadow(
                Proposer(GOOD_PROPOSAL),
                failures=[FAILED_VIEW, dict(FAILED_VIEW)],
                passes=[PASSING_VIEW],
                bench_path=bench,
            )
            self.assertTrue(report["admitted"])
            self.assertEqual(report["flagged_failures"], 2)
            self.assertIn("cannot kill", report["note"])
            after = load_bench(bench)
            self.assertEqual(after["shadow"][0]["name"], "no_magic_claims")
            self.assertEqual(after["lethal"], [])

    def test_prose_instead_of_json_is_a_named_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "judges.json"
            report = propose_shadow(
                Proposer("a fascinating idea for a predicate!"),
                failures=[FAILED_VIEW],
                passes=[],
                bench_path=bench,
            )
            self.assertFalse(report["admitted"])
            self.assertEqual(load_bench(bench)["shadow"], [])

    def test_flagging_a_supported_card_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "judges.json"
            report = propose_shadow(
                Proposer(GOOD_PROPOSAL),
                failures=[FAILED_VIEW],
                passes=[{"claim": "magic here too", "mechanism": ""}],
                bench_path=bench,
            )
            self.assertFalse(report["admitted"])
            self.assertIn("supported card", report["reason"])

    def test_flagging_none_of_the_failures_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "judges.json"
            report = propose_shadow(
                Proposer(GOOD_PROPOSAL),
                failures=[{"claim": "no trigger word at all", "mechanism": ""}],
                passes=[],
                bench_path=bench,
            )
            self.assertFalse(report["admitted"])
            self.assertIn("flags none", report["reason"])

    def test_a_full_bench_refuses_before_spending_a_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "judges.json"
            for index in range(SHADOW_CAP):
                add_shadow(
                    bench,
                    name=f"judge_{index}",
                    source=FLAGS_MAGIC,
                    provenance="test",
                )
            proposer = Proposer(GOOD_PROPOSAL)
            report = propose_shadow(
                proposer,
                failures=[FAILED_VIEW],
                passes=[],
                bench_path=bench,
            )
            self.assertFalse(report["admitted"])
            self.assertIn("full", report["reason"])
            self.assertEqual(proposer.prompts, [])


class ReadoutTests(unittest.TestCase):
    def test_judge_stats_split_catch_from_false_kill(self) -> None:
        rows = [
            mission(
                [
                    card(shadow=["j"], verdict="weakens", kind="WORLD"),
                    card(shadow=["j"], verdict="supports", kind="WORLD"),
                    card(shadow=["j"], verdict="supports", kind="SYNTHETIC"),
                ]
            )
        ]
        stats = judge_stats(rows, ["j"])
        self.assertEqual(stats["j"]["flagged"], 3)
        self.assertEqual(stats["j"]["caught_weakens"], 1)
        self.assertEqual(stats["j"]["flagged_supports"], 1)

    def test_tau_judge_reads_whether_kills_fall_over_time(self) -> None:
        early = [mission([card(killed_by=["strict"])]) for _ in range(3)]
        late = [mission([card()]) for _ in range(3)]
        report = tau_judge_report(early + late)
        self.assertEqual(report["strict"]["early_kills"], 3)
        self.assertEqual(report["strict"]["late_kills"], 0)
        self.assertTrue(report["strict"]["learning"])


if __name__ == "__main__":
    unittest.main()
