"""Extension surfaces: runners, campaigns, submission skeletons."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from farfield.cli import main
from farfield.extras.artifact import compile_artifact
from farfield.extras.campaign import (
    CampaignError,
    append_mission,
    init_campaign,
    load_campaign,
    rollback_stage,
)
from farfield.extras.hostexp import ExecuteError
from farfield.extras.plugins import PluginHost
from farfield.extras.runners import dispatch_run, resolve_runner


class RunnerTests(unittest.TestCase):
    def test_unknown_runner_does_not_invent_gpu(self) -> None:
        with self.assertRaises(ExecuteError):
            resolve_runner("gpu-cluster")
        self.assertEqual(resolve_runner("", compute_tier="host-heavy").name, "host-heavy")
        self.assertFalse(resolve_runner("external").can_corroborate)

    def test_external_without_plugin_writes_a_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "card"
            folder.mkdir()
            (folder / "protocol.json").write_text(
                json.dumps(
                    {
                        "card_id": "gen_e",
                        "host": {"runner": "external"},
                        "diagnosis": {
                            "alternative": "noise",
                            "experiment": "two-arm",
                            "treatment_arm": "on",
                            "control_arm": "off",
                            "expected_direction": "treatment_lower",
                            "alternative_direction": "treatment_higher",
                            "margin": 0.05,
                            "margin_reason": "five percent",
                        },
                    }
                ),
                encoding="utf-8",
            )
            record = dispatch_run(folder)
            self.assertFalse(record["ok"])
            self.assertEqual(record["status"], "awaiting_metrics")
            self.assertFalse(record["can_corroborate"])
            self.assertTrue((folder / "runner_receipt.json").is_file())

    def test_plugin_numbers_still_go_through_attest_arithmetic(self) -> None:
        plugin = SimpleNamespace(
            name="lab-gpu",
            has=lambda hook: hook == "run_external",
            call=lambda hook, **kwargs: {
                "treatment": 1.0,
                "control": 2.0,
                "environment": "8xA100",
            },
            to_dict=lambda: {},
        )
        host = PluginHost(plugins=(plugin,))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "card"
            folder.mkdir()
            (folder / "protocol.json").write_text(
                json.dumps(
                    {
                        "card_id": "gen_e",
                        "diagnosis": {
                            "alternative": "noise",
                            "experiment": "two-arm",
                            "treatment_arm": "on",
                            "control_arm": "off",
                            "expected_direction": "treatment_lower",
                            "alternative_direction": "treatment_higher",
                            "margin": 0.05,
                            "margin_reason": "five percent",
                        },
                    }
                ),
                encoding="utf-8",
            )
            record = dispatch_run(folder, runner="external", plugin_host=host)
            self.assertTrue(record["ok"])
            self.assertEqual(record["verdict"], "supports")
            self.assertEqual(record["status"], "externally_replicated")
            self.assertFalse(record["can_corroborate"])
            self.assertIn("not corroborated", record["ladder"])


class CampaignTests(unittest.TestCase):
    def test_append_and_rollback_keep_mission_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mission = root / "var" / "missions" / "m1"
            mission.mkdir(parents=True)
            (mission / "summary.json").write_text(
                json.dumps({"corroborated": 1}), encoding="utf-8"
            )
            init_campaign(root, "line-a", intent="keep the standing intent stable")
            append_mission(root, "line-a", mission, authority="operator")
            rolled = rollback_stage(
                root, "line-a", reason="method search is a dead end", authority="manager"
            )
            self.assertEqual(rolled["stages"][-1]["kind"], "rollback")
            self.assertEqual(rolled["missions"], [str(mission)])
            self.assertTrue((mission / "summary.json").is_file())
            payload = json.loads((mission / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["corroborated"], 1)
            with self.assertRaises(CampaignError):
                rollback_stage(root, "line-a", reason="short")

    def test_hand_edit_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_campaign(root, "line-b", intent="frozen intent for this line")
            path = root / "var" / "campaigns" / "line-b" / "campaign.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["intent"] = "quietly changed"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(CampaignError):
                load_campaign(root, "line-b")


class ArtifactTests(unittest.TestCase):
    def test_skeleton_results_stay_empty_without_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "card"
            folder.mkdir()
            (folder / "protocol.json").write_text(
                json.dumps(
                    {
                        "card_id": "gen_a",
                        "title": "a claim about masking",
                        "claim": "masking zero-advantage tokens helps",
                        "mechanism": "less gradient noise",
                        "diagnosis": {
                            "treatment_arm": "advantage mask",
                            "control_arm": "gradient-norm mask",
                            "expected_direction": "treatment_higher",
                            "margin": 0.05,
                            "experiment": "two-arm next-token",
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = compile_artifact(folder)
            self.assertFalse(manifest["has_results"])
            self.assertTrue(manifest["not_a_discovery"])
            self.assertEqual(manifest["kind"], "attested_interchange")
            attested = json.loads(
                (folder / "artifact" / "attested.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(attested["results"])
            writing = (folder / "artifact" / "WRITING.md").read_text(encoding="utf-8")
            self.assertIn("jin-s13/ai-research-writing-skill", writing)
            self.assertNotIn("Declared venue", writing)
            md = (folder / "artifact" / "paper.md").read_text(encoding="utf-8")
            self.assertIn("empty until", md)
            self.assertIn("Not a paper", md.lower() + md)

    def test_compile_artifact_venue_is_a_writing_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "card"
            folder.mkdir()
            (folder / "protocol.json").write_text(
                json.dumps({"card_id": "gen_v", "claim": "x"}), encoding="utf-8"
            )
            compile_artifact(folder, venue="cav")
            writing = (folder / "artifact" / "WRITING.md").read_text(encoding="utf-8")
            self.assertIn("Declared venue: cav", writing)
            self.assertIn("jin-s13/ai-research-writing-skill", writing)
            self.assertIn("official template", writing)
            self.assertEqual(main(["compile-artifact", str(folder), "--venue", "stoc"]), 0)
            writing = (folder / "artifact" / "WRITING.md").read_text(encoding="utf-8")
            self.assertIn("Declared venue: stoc", writing)

    def test_ingest_papers_banks_the_neighbourhood_wiki(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers.json"
            papers.write_text(
                json.dumps(
                    [
                        {
                            "title": "We do not handle dynamic cuts",
                            "arxiv_id": "2601.00002",
                            "abstract": "However, we do not scale past ten nodes.",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            state = root / "state.json"
            self.assertEqual(
                main(
                    [
                        "ingest-papers",
                        "--state-store",
                        str(state),
                        "--anchor",
                        "succinct data structure",
                        "--file",
                        str(papers),
                        "--corpus",
                        "test-corpus",
                        "--seen-at",
                        "2026-08-30",
                        "--skip-verify",
                    ]
                ),
                0,
            )
            from farfield.extras.state import load, wiki_for

            rows = wiki_for(load(state), "test-corpus", "succinct data structure")
            self.assertEqual(len(rows), 1)
            self.assertIn("2601.00002", str(rows[0].get("cite_id") or rows[0].get("arxiv_id")))

    def test_cli_compile_and_campaign_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "card"
            folder.mkdir()
            (folder / "protocol.json").write_text(
                json.dumps({"card_id": "gen_c", "claim": "x"}), encoding="utf-8"
            )
            self.assertEqual(main(["compile-artifact", str(folder)]), 0)
            self.assertTrue((folder / "artifact" / "attested.json").is_file())
            self.assertTrue((folder / "artifact" / "paper.tex").is_file())
            self.assertEqual(
                main(
                    [
                        "campaign",
                        "init",
                        "--id",
                        "demo",
                        "--root",
                        str(root),
                        "--intent",
                        "stable intent for the demo line",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "campaign",
                        "append",
                        "--id",
                        "demo",
                        "--root",
                        str(root),
                        "--mission",
                        str(folder),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(["run", str(folder), "--runner", "external"]),
                0,
            )
            receipt = json.loads((folder / "runner_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "awaiting_metrics")
            sky = (folder / "sky_task.yaml").read_text(encoding="utf-8")
            self.assertIn("python experiment.py", sky)
            self.assertIn("SkyPilot", sky)


if __name__ == "__main__":
    unittest.main()
