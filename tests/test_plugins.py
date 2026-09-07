"""PluginHost: skills with hooks that actually refuse, not prompt paste."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from farfield.cli import build_parser
from farfield.extras.benchloop import LocalWorkspace, action_from_payload, run_intern
from farfield.extras.brief import brief_from_payload
from farfield.extras.harness import DeepSeekHarnessPlugin, dsh_status, persist_skill, run_dsh
from farfield.extras.plugins import PluginHost, bind_host, repo_root, reset_default_host
from farfield.extras.probeexp import ProbeRefused, spec_from_payload
from farfield.extras.skills import parse_skill_md


FAIR_SOURCE = (
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "def measure(use_shortcut):\n"
    "    hops = 0\n"
    "    pos = 0\n"
    "    end = 8\n"
    "    while pos < end:\n"
    "        hops += 1\n"
    "        if use_shortcut and pos == 0:\n"
    "            pos = end\n"
    "        else:\n"
    "            pos += 1\n"
    "    return hops\n"
    "\n"
    "Path('metrics.json').write_text("
    "json.dumps({'treatment': float(measure(True)),"
    " 'control': float(measure(False))}), encoding='utf-8')\n"
)

MARKER_PLUGIN = """
def validate_probe(tree=None, source='', **_):
    if 'FORBIDDEN_MARKER' in (source or ''):
        return 'plugin hook killed this source'
    return None
"""

SKILL_MD = """---
name: marker-kill
description: Refuse a probe whose source contains a test marker.
stage: probe
admitted: true
audience: research
entry: plugin.py
---

A procedure. The hook is what runs.
"""


def _write_skill(root: Path, name: str, body: str, plugin: str | None) -> Path:
    dest = root / ".agents" / "skills" / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SKILL.md").write_text(body, encoding="utf-8")
    if plugin is not None:
        (dest / "plugin.py").write_text(plugin, encoding="utf-8")
    return dest / "SKILL.md"


class CatalogIsolationTests(unittest.TestCase):
    def test_seed_plugins_are_executable_and_cursor_skills_stay_out(self) -> None:
        host = PluginHost.load(repo_root())
        names = {skill.name for skill in host.catalog}
        self.assertIn("fair-two-arm-probe", names)
        self.assertIn("claim-domain-lock", names)
        self.assertIn("external-gpu-run", names)
        self.assertIn("attested-writeup", names)
        self.assertIn("campaign-horizon", names)
        self.assertIn("research-lit", names)
        self.assertIn("experiment-plan", names)
        self.assertIn("result-to-claim", names)
        self.assertIn("ablation-planner", names)
        self.assertIn("experiment-audit", names)
        self.assertIn("citation-audit", names)
        self.assertNotIn("farfield-research", names)
        executable = {row["name"] for row in host.summary() if row.get("executable")}
        self.assertIn("fair-two-arm-probe", executable)
        self.assertIn("claim-domain-lock", executable)
        self.assertIn("external-gpu-run", executable)
        self.assertIn("research-lit", executable)
        self.assertIn("experiment-audit", executable)
        self.assertIn("citation-audit", executable)
        self.assertIn("deepseek-harness", executable)
        self.assertIn("builtin", executable)
        fair = host.by_name("fair-two-arm-probe")
        self.assertIsNotNone(fair)
        assert fair is not None
        self.assertTrue(fair.has("validate_probe"))
        self.assertTrue(fair.has("execute"))

    def test_a_skill_md_without_plugin_py_is_prompt_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skill(root, "prompt-only", SKILL_MD.replace("marker-kill", "prompt-only"), None)
            host = PluginHost.load(root, include_dsh=False)
            rows = {row["name"]: row for row in host.summary()}
            self.assertIn("prompt-only", rows)
            self.assertFalse(rows["prompt-only"]["executable"])
            self.assertEqual(rows["prompt-only"]["hooks"], [])


class HookCausalityTests(unittest.TestCase):
    def test_a_plugin_hook_refuses_a_fair_script_that_the_builtin_would_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skill(root, "marker-kill", SKILL_MD, MARKER_PLUGIN)
            host = PluginHost.load(root, include_builtin=False, include_dsh=False)
            with bind_host(host):
                spec_from_payload({"measure": "hops", "source": FAIR_SOURCE})
                with self.assertRaises(ProbeRefused) as caught:
                    spec_from_payload(
                        {
                            "measure": "hops",
                            "source": FAIR_SOURCE + "# FORBIDDEN_MARKER\n",
                        }
                    )
            self.assertIn("plugin hook killed", caught.exception.record.unlock_condition)

    def test_workspace_plugin_py_is_not_executed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            workspace = Path(tmp) / "mission"
            _write_skill(
                repo,
                "safe",
                SKILL_MD.replace("marker-kill", "safe"),
                "def validate_probe(tree=None, source='', **_):\n    return None\n",
            )
            evil = workspace / ".agents" / "skills" / "evil"
            evil.mkdir(parents=True)
            (evil / "SKILL.md").write_text(
                SKILL_MD.replace("marker-kill", "evil"), encoding="utf-8"
            )
            (evil / "plugin.py").write_text(
                "def validate_probe(tree=None, source='', **_):\n"
                "    return 'untrusted workspace plugin ran'\n",
                encoding="utf-8",
            )
            host = PluginHost.load(repo, workspace, include_builtin=False, include_dsh=False)
            names = [plugin.name for plugin in host.plugins]
            self.assertNotIn("evil", names)
            with bind_host(host):
                spec_from_payload(
                    {"measure": "hops", "source": FAIR_SOURCE + "# FORBIDDEN_MARKER\n"}
                )

    def test_lock_writeup_hook_refuses_a_field_change(self) -> None:
        from farfield.extras.generate import GenerationRefused

        with self.assertRaises(GenerationRefused) as caught:
            brief_from_payload(
                "card",
                {
                    "title": "Certified Replay for LLM Tool-Use Safety",
                    "plain_title": "回放证书能保证工具调用安全吗",
                    "one_liner": "给工具调用的执行轨迹加一个可验证的回放证书，能否证明安全性质？",
                    "why_it_matters": "如果能，部署方就可以只审计证书而不用重放全部轨迹。",
                    "gap": "the retrieved paper leaves the bound open",
                    "idea": "Verify tool-calling agents with a replay certificate",
                    "approach": "instrument traces",
                    "first_steps": ["read the paper", "write a checker"],
                    "baseline": "unverified replay",
                    "risks": "the bound does not transfer",
                    "read_first": [],
                },
                [],
                topic="formal verification of safety properties for LLM agents and tool-using autonomous agents",
                home="tight hardness of fine-grained reductions for graph alignment",
                seed_label="graph alignment",
            )
        self.assertIn("field", caught.exception.record.unlock_condition)


class DshPluginTests(unittest.TestCase):
    def test_missing_dsh_is_an_observation_not_a_fake_success(self) -> None:
        status = dsh_status()
        self.assertIn("available", status)
        self.assertEqual(status["role"], "plugin")
        result = run_dsh("ping")
        if not result.available:
            self.assertIn("not on PATH", result.output)
            self.assertEqual(result.command, ())
        plugin = DeepSeekHarnessPlugin()
        self.assertTrue(plugin.has("execute"))
        observation = plugin.execute(args={"instruction": "ping"})
        self.assertTrue(observation)

    def test_default_host_still_validates_without_dsh(self) -> None:
        reset_default_host()
        with self.assertRaises(ProbeRefused):
            spec_from_payload(
                {
                    "measure": "literals",
                    "source": (
                        "import json\n"
                        "from pathlib import Path\n"
                        "Path('metrics.json').write_text("
                        "json.dumps({'treatment': 80, 'control': 120}))\n"
                    ),
                }
            )


class InternSkillTests(unittest.TestCase):
    def test_skill_action_runs_execute_not_the_prompt(self) -> None:
        import asyncio

        class ScriptedClient:
            def __init__(self, answers):
                self.answers = list(answers)

            def complete(self, prompt, *, purpose, system=None, logprobs=False):
                return SimpleNamespace(
                    text=json.dumps(self.answers.pop(0)),
                    assert_usable=lambda: None,
                )

        host = PluginHost.load(repo_root())
        client = ScriptedClient(
            [
                {
                    "action": "diagnose",
                    "criterion": "report.md exists",
                    "alternative": "claim done without writing",
                    "plan": "validate then write report.md",
                    "deliverable": "report.md",
                },
                {
                    "action": "skill",
                    "name": "fair-two-arm-probe",
                    "args": {"source": FAIR_SOURCE},
                },
                {
                    "action": "write",
                    "path": "report.md",
                    "content": "ok",
                },
                {"action": "done", "summary": "wrote report.md after the plugin ran"},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            report = asyncio.run(
                run_intern(
                    client,
                    LocalWorkspace(Path(tmp)),
                    "Write report.md",
                    host=host,
                )
            )
        self.assertTrue(report.done)
        skill_steps = [row for row in report.steps if row.get("action") == "skill"]
        self.assertEqual(len(skill_steps), 1)
        payload = json.loads(skill_steps[0]["observation"])
        self.assertTrue(payload["ok"])

    def test_prompt_only_skill_cannot_be_invoked_as_a_tool(self) -> None:
        action = action_from_payload(
            {"action": "skill", "name": "prompt-only", "args": {}}
        )
        self.assertEqual(action["name"], "prompt-only")
        host = PluginHost(plugins=(), catalog=())
        observation = host.execute("prompt-only", {})
        self.assertIn("unknown plugin", observation)


class DistillDoesNotWritePluginTests(unittest.TestCase):
    def test_persist_skill_writes_skill_md_only(self) -> None:
        skill = parse_skill_md(SKILL_MD, source="distill")
        with tempfile.TemporaryDirectory() as tmp:
            written = persist_skill(
                skill, workspace=Path(tmp) / "mission", catalog=Path(tmp) / "cat"
            )
            path = Path(written["workspace"])
            self.assertEqual(path.name, "SKILL.md")
            self.assertFalse((path.parent / "plugin.py").exists())


class CliProductPathTests(unittest.TestCase):
    def test_control_arm_commands_are_gone(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["run", "/tmp/x", "--seed", "s"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["install", "/tmp/x", "--graph", "g", "--seed-node", "n"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["dossier", "/tmp/x"])
        args = parser.parse_args(["freeze", "--id", "snap-ca-grqc", "--file", "/tmp/x"])
        self.assertEqual(args.command, "freeze")
        self.assertEqual(args.world_id, "snap-ca-grqc")
        args = parser.parse_args(["execute", "/tmp/card"])
        self.assertEqual(args.command, "execute")
        args = parser.parse_args(["research", "/tmp/x", "--topic", "t"])
        self.assertTrue(args.host_execute)
        self.assertEqual(args.venue, "")
        args = parser.parse_args(
            ["research", "/tmp/x", "--topic", "t", "--venue", "neurips"]
        )
        self.assertEqual(args.venue, "neurips")
        args = parser.parse_args(
            [
                "ingest-papers",
                "--state-store",
                "/tmp/state.json",
                "--anchor",
                "graph",
                "--file",
                "/tmp/papers.json",
            ]
        )
        self.assertEqual(args.command, "ingest-papers")
        args = parser.parse_args(
            ["freeze", "--id", "phage-lambda", "--schema", "fasta", "--slice", "first_bases:50000", "--file", "/tmp/x"]
        )
        self.assertEqual(args.schema, "fasta")
        self.assertEqual(args.slice_rule, "first_bases:50000")


if __name__ == "__main__":
    unittest.main()
