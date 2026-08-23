"""The meta loop: trajectories in, routing proposal out, held-out gate decides."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import json

from farfield.extras.routing import (
    LOG_SCHEMA,
    _digest,
    append_mission,
    commit_if_better,
    load_log,
    load_policy,
    maybe_update,
    operator_report,
    preferred_operators,
    propose_policy,
    score_policy,
    settlement_missions,
)


def mission(operator: str, promoted: bool) -> dict:
    return {
        "topic_digest": "t",
        "corpus": "c",
        "anchor": "a",
        "cards": [
            {
                "operator": operator,
                "track": "farfield",
                "killed": not promoted,
                "briefed": promoted,
                "verdict": "supports" if promoted else None,
                "promoted": "corroborated" if promoted else None,
            }
        ],
    }


def legacy_log(path: Path, missions: list[dict]) -> None:
    """Write records exactly as the pre-`probe_kind` era did: no schema."""
    path.write_text(
        json.dumps({"missions": missions, "digest": _digest(missions)}),
        encoding="utf-8",
    )


class LogTests(unittest.TestCase):
    def test_trajectories_accumulate_and_report_per_operator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            append_mission(log, mission("directional", True))
            append_mission(log, mission("analogy", False))
            report = operator_report(load_log(log))
            self.assertEqual(report["directional"]["promoted"], 1)
            self.assertEqual(report["analogy"]["promoted"], 0)

    def test_every_banked_mission_carries_the_current_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            append_mission(log, mission("directional", True))
            banked = load_log(log)[0]
            self.assertEqual(banked["schema"], LOG_SCHEMA)

    def test_legacy_records_are_history_not_evidence(self) -> None:
        rows = [mission("directional", True) for _ in range(3)]
        self.assertEqual(settlement_missions(rows), [])
        stamped = [{**row, "schema": LOG_SCHEMA} for row in rows]
        self.assertEqual(len(settlement_missions(stamped + rows)), 3)


class GateTests(unittest.TestCase):
    def test_a_candidate_that_helps_on_heldout_is_committed(self) -> None:
        fitting = [mission("directional", True), mission("analogy", False)]
        heldout = [mission("directional", True), mission("analogy", False)]
        candidate = propose_policy(fitting)
        self.assertGreater(
            candidate["weights"]["directional"], candidate["weights"]["analogy"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            decision = commit_if_better(policy_path, candidate, heldout)
            self.assertTrue(decision["committed"])
            self.assertGreater(
                decision["candidate_score"], decision["incumbent_score"]
            )
            installed = load_policy(policy_path)
            self.assertEqual(installed["weights"], candidate["weights"])

    def test_a_candidate_that_fails_on_heldout_is_rejected(self) -> None:
        # Fitted on missions where analogy won, judged on missions where
        # it loses: the incumbent (uniform) must stand.
        fitting = [mission("analogy", True), mission("directional", False)]
        heldout = [mission("directional", True), mission("analogy", False)]
        candidate = propose_policy(fitting)
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            decision = commit_if_better(policy_path, candidate, heldout)
            self.assertFalse(decision["committed"])
            self.assertIsNone(load_policy(policy_path))
            self.assertIn("incumbent stands", decision["reason"])

    def test_an_arithmetically_equal_candidate_does_not_commit(self) -> None:
        # A live commit slipped through when the candidate's uniform
        # weights (all 0.2) scored identically to the uniform incumbent
        # except for the last float ulp. Equal is not better.
        fitting = [
            mission("directional", True),
            mission("interpolate", True),
            mission("assumption_removal", True),
        ]
        candidate = propose_policy(fitting)
        self.assertEqual(len(set(candidate["weights"].values())), 1)
        heldout = [
            mission("directional", True),
            mission("interpolate", False),
            mission("assumption_removal", False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            decision = commit_if_better(policy_path, candidate, heldout)
            self.assertFalse(decision["committed"])
            self.assertIsNone(load_policy(policy_path))

    def test_no_heldout_means_no_commit(self) -> None:
        candidate = propose_policy([mission("directional", True)])
        with tempfile.TemporaryDirectory() as tmp:
            decision = commit_if_better(Path(tmp) / "p.json", candidate, [])
            self.assertFalse(decision["committed"])

    def test_uniform_scoring_is_the_plain_promotion_rate(self) -> None:
        records = [mission("directional", True), mission("analogy", False)]
        self.assertAlmostEqual(score_policy(None, records), 0.5)

    def test_a_support_without_probe_kind_earns_no_routing_credit(self) -> None:
        # F1: 21 live-log supports predate probe_kind. "Not provably
        # SYNTHETIC" is not attested; a missing kind must earn zero.
        legacy = {
            "cards": [
                {
                    "operator": "analogy",
                    "killed": False,
                    "verdict": "supports",
                    "promoted": None,
                }
            ]
        }
        self.assertEqual(score_policy(None, [legacy]), 0.0)
        self.assertEqual(operator_report([legacy])["analogy"]["world_supported"], 0)

    def test_synthetic_value_does_not_steer_routing(self) -> None:
        synthetic = {
            "cards": [
                {
                    "operator": "analogy",
                    "killed": False,
                    "verdict": "supports",
                    "promoted": None,
                    "value_score": 8.0,
                    "probe_kind": "SYNTHETIC",
                }
            ]
        }
        world = {
            "cards": [
                {
                    "operator": "directional",
                    "killed": False,
                    "verdict": "supports",
                    "promoted": "corroborated",
                    "value_score": 1.0,
                    "probe_kind": "WORLD",
                }
            ]
        }
        self.assertEqual(score_policy(None, [synthetic]), 0.0)
        self.assertGreater(score_policy(None, [world]), score_policy(None, [synthetic]))
        report = operator_report([synthetic, world])
        self.assertEqual(report["analogy"]["world_supported"], 0)
        self.assertEqual(report["directional"]["world_supported"], 1)
        self.assertNotIn("value_mass", report["analogy"])


    def test_high_elo_without_world_support_does_not_beat_a_promotion(self) -> None:
        loud = {
            "cards": [
                {
                    "operator": "analogy",
                    "killed": False,
                    "verdict": "supports",
                    "promoted": None,
                    "value_score": 50.0,
                    "probe_kind": "WORLD",
                }
            ]
        }
        climbed = {
            "cards": [
                {
                    "operator": "directional",
                    "killed": False,
                    "verdict": "supports",
                    "promoted": "corroborated",
                    "value_score": 0.0,
                    "probe_kind": "WORLD",
                }
            ]
        }
        self.assertGreater(score_policy(None, [climbed]), score_policy(None, [loud]))


class RoutingTests(unittest.TestCase):
    def test_without_an_installed_policy_the_declared_order_stands(self) -> None:
        order = ("assumption_removal", "boundary_search", "contradiction_search")
        self.assertEqual(preferred_operators(None, order), order)

    def test_an_installed_policy_reorders_by_weight(self) -> None:
        policy = {"weights": {"contradiction_search": 0.9, "assumption_removal": 0.1}}
        order = preferred_operators(
            policy, ("assumption_removal", "boundary_search", "contradiction_search")
        )
        self.assertEqual(order[0], "contradiction_search")

    def test_the_far_field_cycle_is_routable_too(self) -> None:
        policy = {"weights": {"analogy": 0.8, "directional": 0.2}}
        order = preferred_operators(
            policy, ("directional", "interpolate", "low_density", "analogy")
        )
        self.assertEqual(order[0], "analogy")


class AutoUpdateTests(unittest.TestCase):
    def test_a_short_log_declines_to_decide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            for _ in range(3):
                append_mission(log, mission("directional", True))
            self.assertIsNone(maybe_update(log, Path(tmp) / "p.json"))

    def test_a_long_consistent_log_commits_through_the_heldout_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            for _ in range(6):
                mixed = mission("directional", True)
                mixed["cards"] += mission("analogy", False)["cards"]
                append_mission(log, mixed)
            policy_path = Path(tmp) / "p.json"
            decision = maybe_update(log, policy_path)
            self.assertIsNotNone(decision)
            self.assertTrue(decision["committed"])
            installed = load_policy(policy_path)
            self.assertGreater(
                installed["weights"]["directional"],
                installed["weights"]["analogy"],
            )

    def test_a_log_full_of_legacy_missions_settles_nothing(self) -> None:
        # F4: sixteen pre-schema missions must not vote. A log that is
        # long but inadmissible declines to decide, same as a short one.
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            legacy_log(log, [mission("directional", True) for _ in range(8)])
            self.assertIsNone(maybe_update(log, Path(tmp) / "p.json"))
            # New-schema missions appended on top start the count fresh.
            for _ in range(5):
                append_mission(log, mission("directional", True))
            self.assertIsNone(maybe_update(log, Path(tmp) / "p.json"))
            mixed = mission("directional", True)
            mixed["cards"] += mission("analogy", False)["cards"]
            append_mission(log, mixed)
            self.assertIsNotNone(maybe_update(log, Path(tmp) / "p.json"))


class ConcurrentLogTests(unittest.TestCase):
    def test_parallel_appends_do_not_drop_missions(self) -> None:
        import multiprocessing

        def _write(path: str, operator: str) -> None:
            append_mission(Path(path), mission(operator, True))

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            workers = [
                multiprocessing.Process(target=_write, args=(str(log), f"op{i}"))
                for i in range(8)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=10)
                self.assertEqual(worker.exitcode, 0)
            banked = load_log(log)
            self.assertEqual(len(banked), 8)
            self.assertEqual(len({row["cards"][0]["operator"] for row in banked}), 8)


if __name__ == "__main__":
    unittest.main()
