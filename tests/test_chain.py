"""The epistemic event chain: append-only, ordered, tamper-evident."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.chain import (
    EXECUTE_PROTOCOL,
    FREEZE_WORLD,
    GENESIS,
    PROMOTE,
    REGISTER_PROTOCOL,
    append_event,
    audit_promotion_gaps,
    read_events,
    verify_chain,
)


class ChainTests(unittest.TestCase):
    def test_events_chain_from_genesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            first = append_event(path, FREEZE_WORLD, {"world_id": "w1"})
            second = append_event(path, REGISTER_PROTOCOL, {"card_id": "c1"})
            self.assertEqual(first["seq"], 0)
            self.assertEqual(first["prev"], GENESIS)
            self.assertEqual(second["prev"], first["digest"])
            report = verify_chain(path)
            self.assertTrue(report["ok"])
            self.assertEqual(report["length"], 2)

    def test_the_order_of_acts_is_provable(self) -> None:
        # Freezing bytes proves what was frozen; the chain proves the
        # freeze came before the registration and the execution.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            append_event(path, FREEZE_WORLD, {"world_id": "w1"})
            append_event(path, REGISTER_PROTOCOL, {"card_id": "c1"})
            append_event(path, EXECUTE_PROTOCOL, {"card_id": "c1"})
            append_event(path, PROMOTE, {"card_id": "c1"})
            kinds = [event["kind"] for event in read_events(path)]
            self.assertEqual(
                kinds, [FREEZE_WORLD, REGISTER_PROTOCOL, EXECUTE_PROTOCOL, PROMOTE]
            )
            self.assertTrue(verify_chain(path)["ok"])

    def test_an_edited_event_breaks_every_later_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            append_event(path, FREEZE_WORLD, {"world_id": "w1"})
            append_event(path, PROMOTE, {"status": "speculative"})
            lines = path.read_text(encoding="utf-8").splitlines()
            row = json.loads(lines[1])
            row["payload"]["status"] = "verified"
            lines[1] = json.dumps(row, ensure_ascii=False, sort_keys=True)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = verify_chain(path)
            self.assertFalse(report["ok"])
            self.assertIn("digest", report["error"])

    def test_a_dropped_event_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            append_event(path, FREEZE_WORLD, {"world_id": "w1"})
            append_event(path, REGISTER_PROTOCOL, {"card_id": "c1"})
            append_event(path, PROMOTE, {"card_id": "c1"})
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
            self.assertFalse(verify_chain(path)["ok"])

    def test_a_reordered_chain_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            append_event(path, FREEZE_WORLD, {"world_id": "w1"})
            append_event(path, REGISTER_PROTOCOL, {"card_id": "c1"})
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
            self.assertFalse(verify_chain(path)["ok"])

    def test_an_empty_or_missing_log_verifies_trivially(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = verify_chain(Path(tmp) / "missing.jsonl")
            self.assertTrue(report["ok"])
            self.assertEqual(report["length"], 0)


class PromotionAuditTests(unittest.TestCase):
    """The chain as a gate: promotion consumes the audit, not just the diary."""

    def test_register_before_execute_licenses_a_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            append_event(path, REGISTER_PROTOCOL, {"experiment_digest": "a" * 64})
            append_event(path, EXECUTE_PROTOCOL, {"evidence_id": "e" * 64})
            gaps = audit_promotion_gaps(
                path, experiment_digest="a" * 64, evidence_id="e" * 64
            )
            self.assertEqual(gaps, ())

    def test_a_missing_chain_is_a_missing_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gaps = audit_promotion_gaps(Path(tmp) / "missing.jsonl")
            self.assertEqual(len(gaps), 1)
            self.assertIn("no event chain recorded", gaps[0])

    def test_an_unregistered_protocol_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            append_event(path, EXECUTE_PROTOCOL, {"evidence_id": "e" * 64})
            gaps = audit_promotion_gaps(path, evidence_id="e" * 64)
            self.assertTrue(any("never registered" in gap for gap in gaps))

    def test_a_registration_after_the_run_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            append_event(path, EXECUTE_PROTOCOL, {"evidence_id": "e" * 64})
            append_event(path, REGISTER_PROTOCOL, {"experiment_digest": "a" * 64})
            gaps = audit_promotion_gaps(
                path, experiment_digest="a" * 64, evidence_id="e" * 64
            )
            self.assertTrue(
                any("registered after the execution" in gap for gap in gaps)
            )

    def test_a_registration_for_another_experiment_does_not_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            append_event(path, REGISTER_PROTOCOL, {"experiment_digest": "f" * 64})
            append_event(path, EXECUTE_PROTOCOL, {"evidence_id": "e" * 64})
            gaps = audit_promotion_gaps(
                path, experiment_digest="a" * 64, evidence_id="e" * 64
            )
            self.assertTrue(any("never registered" in gap for gap in gaps))

    def test_a_broken_chain_refuses_regardless_of_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            append_event(path, REGISTER_PROTOCOL, {"experiment_digest": "a" * 64})
            append_event(path, EXECUTE_PROTOCOL, {"evidence_id": "e" * 64})
            lines = path.read_text(encoding="utf-8").splitlines()
            row = json.loads(lines[0])
            row["payload"]["experiment_digest"] = "d" * 64
            lines[0] = json.dumps(row, ensure_ascii=False, sort_keys=True)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            gaps = audit_promotion_gaps(
                path, experiment_digest="d" * 64, evidence_id="e" * 64
            )
            self.assertEqual(len(gaps), 1)
            self.assertIn("event chain broken", gaps[0])


if __name__ == "__main__":
    unittest.main()
