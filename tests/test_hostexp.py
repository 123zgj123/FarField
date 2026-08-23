"""Host execution of protocol.json: parent digest, fair arms, not a discovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.cli import main
from farfield.extras.freeze import freeze_world
from farfield.extras.hostexp import (
    ExecuteError,
    PROTOCOL_EXECUTED,
    attest_external_run,
    execute_protocol,
)


SCRIPT = """import json
from pathlib import Path
table = json.loads(Path('data/table.json').read_text(encoding='utf-8'))
n = float(table['n'])
def measure(use_mech):
    return n if use_mech else n + 1.0
Path('metrics.json').write_text(
    json.dumps({'treatment': float(measure(True)), 'control': float(measure(False))}),
    encoding='utf-8',
)
"""


class HostExecuteTests(unittest.TestCase):
    def test_host_run_on_slice_is_protocol_executed_not_a_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            source = repo / "t.csv"
            source.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n", encoding="utf-8")
            fixture = freeze_world(
                world_id="tiny-table",
                schema="numeric_table",
                slice_rule="first_rows:2",
                catalog=repo / "worlds",
                url="https://example.invalid/t.csv",
                source_file=source,
            )
            folder = repo / "card"
            folder.mkdir()
            (folder / "experiment.py").write_text(SCRIPT, encoding="utf-8")
            (folder / "protocol.json").write_text(
                json.dumps(
                    {
                        "card_id": "gen_x",
                        "world": {"id": fixture.id, "source_digest": fixture.source_digest},
                        "cheap_probe": {"measure": "row count"},
                        "diagnosis": {
                            "alternative": "the drop is noise",
                            "experiment": "count rows with and without the flag",
                            "treatment_arm": "mechanism on",
                            "control_arm": "mechanism off",
                            "expected_direction": "treatment_lower",
                            "alternative_direction": "treatment_higher",
                            "margin": 0.05,
                            "margin_reason": "one extra row is above five percent",
                        },
                    }
                ),
                encoding="utf-8",
            )
            record = execute_protocol(
                folder, catalog_root=repo, prefer_parent=False, timeout_seconds=5
            )
            self.assertTrue(record["ok"])
            self.assertEqual(record["bound"], "slice")
            self.assertEqual(record["kind"], "HOST")
            self.assertEqual(record["verdict"], "supports")
            self.assertIn("not a scientific discovery", record["ladder"])
            self.assertEqual(record["honesty"], PROTOCOL_EXECUTED)
            self.assertTrue((folder / "host_run.json").is_file())
            # Same bytes as the probe slice: an E1 same-instance reproduction.
            self.assertEqual(record["replication"], "E1_same_instance")
            self.assertEqual(record["data_digest"], fixture.digest)

            parent = execute_protocol(
                folder, catalog_root=repo, prefer_parent=True, timeout_seconds=5
            )
            self.assertTrue(parent["ok"])
            self.assertEqual(parent["bound"], "parent")
            self.assertEqual(parent["treatment"], 4.0)
            # Different bytes: an E2 scale replication under its own
            # EvidenceID — it generalizes the probe, it cannot confirm it.
            self.assertEqual(parent["replication"], "E2_scale_replication")
            self.assertEqual(parent["data_digest"], fixture.source_digest)
            self.assertNotEqual(parent["evidence_id"], record["evidence_id"])
            self.assertEqual(
                parent["replication_group_id"], record["replication_group_id"]
            )
            self.assertIn("scale replication", parent["ladder"])

            from farfield.extras.chain import verify_chain

            report = verify_chain(folder / "chain.jsonl")
            self.assertTrue(report["ok"])
            self.assertEqual(report["length"], 2)

    def test_a_preregistered_heavy_tier_sets_budget_and_whitelist(self) -> None:
        heavy_script = SCRIPT.replace(
            "import json\n", "import json\nimport statistics\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            source = repo / "t.csv"
            source.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            fixture = freeze_world(
                world_id="tiny-table",
                schema="numeric_table",
                slice_rule="all",
                catalog=repo / "worlds",
                source_file=source,
            )
            folder = repo / "card"
            folder.mkdir()
            (folder / "experiment.py").write_text(heavy_script, encoding="utf-8")
            (folder / "protocol.json").write_text(
                json.dumps(
                    {
                        "card_id": "gen_h",
                        "world": {"id": fixture.id},
                        "cheap_probe": {"measure": "row count"},
                        "diagnosis": {
                            "alternative": "noise",
                            "experiment": "count rows",
                            "treatment_arm": "on",
                            "control_arm": "off",
                            "expected_direction": "treatment_lower",
                            "alternative_direction": "treatment_higher",
                            "margin": 0.05,
                            "margin_reason": "one row is above margin",
                            "compute_tier": "host-heavy",
                        },
                    }
                ),
                encoding="utf-8",
            )
            record = execute_protocol(folder, catalog_root=repo, prefer_parent=False)
            self.assertTrue(record["ok"])
            self.assertEqual(record["compute_tier"], "host-heavy")
            self.assertEqual(record["timeout_seconds"], 3600.0)

    def test_missing_world_or_script_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "card"
            folder.mkdir()
            (folder / "protocol.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ExecuteError):
                execute_protocol(folder)
            (folder / "experiment.py").write_text(SCRIPT, encoding="utf-8")
            with self.assertRaises(ExecuteError):
                execute_protocol(folder)

    def test_missing_parent_cache_is_blocked_not_silently_sliced(self) -> None:
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            source = repo / "t.csv"
            source.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n", encoding="utf-8")
            fixture = freeze_world(
                world_id="tiny-table",
                schema="numeric_table",
                slice_rule="first_rows:2",
                catalog=repo / "worlds",
                url="https://example.invalid/t.csv",
                source_file=source,
            )
            folder = repo / "card"
            folder.mkdir()
            (folder / "experiment.py").write_text(SCRIPT, encoding="utf-8")
            (folder / "protocol.json").write_text(
                json.dumps(
                    {
                        "world": {
                            "id": fixture.id,
                            "source_digest": fixture.source_digest,
                        },
                        "cheap_probe": {"measure": "row count"},
                    }
                ),
                encoding="utf-8",
            )
            shutil.rmtree(repo / "worlds" / ".cache")
            with self.assertRaises(ExecuteError) as caught:
                execute_protocol(folder, catalog_root=repo, prefer_parent=True)
            self.assertIn("parent cache missing", str(caught.exception))

    def test_a_changed_parent_digest_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            source = repo / "t.csv"
            source.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            fixture = freeze_world(
                world_id="tiny-table",
                schema="numeric_table",
                slice_rule="first_rows:2",
                catalog=repo / "worlds",
                source_file=source,
            )
            folder = repo / "card"
            folder.mkdir()
            (folder / "experiment.py").write_text(SCRIPT, encoding="utf-8")
            (folder / "protocol.json").write_text(
                json.dumps(
                    {
                        "world": {
                            "id": fixture.id,
                            "source_digest": "0" * 64,
                        },
                        "cheap_probe": {"measure": "row count"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ExecuteError) as caught:
                execute_protocol(
                    folder, catalog_root=repo, prefer_parent=False, timeout_seconds=5
                )
            self.assertIn("does not match", str(caught.exception))

    def test_a_changed_experiment_digest_cannot_confirm_the_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            source = repo / "t.csv"
            source.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n", encoding="utf-8")
            fixture = freeze_world(
                world_id="tiny-table",
                schema="numeric_table",
                slice_rule="first_rows:2",
                catalog=repo / "worlds",
                source_file=source,
            )
            folder = repo / "card"
            folder.mkdir()
            (folder / "experiment.py").write_text(SCRIPT, encoding="utf-8")
            (folder / "protocol.json").write_text(
                json.dumps(
                    {
                        "card_id": "gen_x",
                        "experiment_digest": "0" * 64,
                        "world": {
                            "id": fixture.id,
                            "source_digest": fixture.source_digest,
                        },
                        "cheap_probe": {"measure": "row count"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ExecuteError) as caught:
                execute_protocol(
                    folder, catalog_root=repo, prefer_parent=False, timeout_seconds=5
                )
            self.assertIn("digest", str(caught.exception).lower())

    def test_external_run_is_judged_by_registered_arithmetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "card"
            folder.mkdir()
            (folder / "protocol.json").write_text(
                json.dumps(
                    {
                        "card_id": "gen_e",
                        "diagnosis": {
                            "alternative": "throughput, not the mechanism",
                            "experiment": "train both arms to the same budget",
                            "treatment_arm": "mechanism on",
                            "control_arm": "mechanism off",
                            "expected_direction": "treatment_lower",
                            "alternative_direction": "treatment_higher",
                            "margin": 0.05,
                            "margin_reason": "run-to-run noise is under 5%",
                        },
                    }
                ),
                encoding="utf-8",
            )
            metrics = Path(tmp) / "metrics.json"
            metrics.write_text(
                json.dumps({"treatment": 1.0, "control": 2.0}), encoding="utf-8"
            )
            record = attest_external_run(
                folder,
                metrics_path=metrics,
                environment="8xA100, torch 2.6",
                runner="lab",
            )
            self.assertEqual(record["verdict"], "supports")
            self.assertEqual(record["status"], "externally_replicated")
            self.assertIn("not corroborated", record["ladder"])
            self.assertTrue((folder / "external_run.json").is_file())

            # a submitted verdict is refused — only the numbers count
            metrics.write_text(
                json.dumps(
                    {"treatment": 1.0, "control": 2.0, "verdict": "supports"}
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ExecuteError):
                attest_external_run(folder, metrics_path=metrics)

            # arms inside the margin are uninformative, not a win
            metrics.write_text(
                json.dumps({"treatment": 1.0, "control": 1.01}), encoding="utf-8"
            )
            judged = attest_external_run(folder, metrics_path=metrics)
            self.assertEqual(judged["verdict"], "uninformative")

    def test_external_run_needs_a_registered_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "card"
            folder.mkdir()
            (folder / "protocol.json").write_text(
                json.dumps({"card_id": "gen_e"}), encoding="utf-8"
            )
            metrics = Path(tmp) / "metrics.json"
            metrics.write_text(
                json.dumps({"treatment": 1.0, "control": 2.0}), encoding="utf-8"
            )
            with self.assertRaises(ExecuteError):
                attest_external_run(folder, metrics_path=metrics)

    def test_cli_attest_run_records_the_replication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "card"
            folder.mkdir()
            (folder / "protocol.json").write_text(
                json.dumps(
                    {
                        "card_id": "gen_c",
                        "diagnosis": {
                            "alternative": "noise",
                            "experiment": "two-arm",
                            "treatment_arm": "on",
                            "control_arm": "off",
                            "expected_direction": "treatment_higher",
                            "alternative_direction": "treatment_lower",
                            "margin": 0.1,
                            "margin_reason": "10% run noise",
                        },
                    }
                ),
                encoding="utf-8",
            )
            metrics = Path(tmp) / "m.json"
            metrics.write_text(
                json.dumps({"treatment": 3.0, "control": 2.0}), encoding="utf-8"
            )
            code = main(
                [
                    "attest-run",
                    str(folder),
                    "--metrics",
                    str(metrics),
                    "--environment",
                    "external cluster",
                ]
            )
            self.assertEqual(code, 0)
            record = json.loads(
                (folder / "external_run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["verdict"], "supports")
            self.assertEqual(record["environment"], "external cluster")

    def test_cli_execute_needs_a_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["execute", str(Path(tmp) / "missing")])
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
