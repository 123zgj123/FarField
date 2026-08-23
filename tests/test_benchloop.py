"""The intern loop: diagnosis first, then workspace actions, never a toy probe."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from farfield.extras.benchloop import (
    INSPECT_STREAK_CAP,
    INSPECT_TIMEOUT,
    READ_PAGE_LINES,
    BenchRefused,
    LocalWorkspace,
    action_from_payload,
    agent_timeout_from_toml,
    run_intern,
    steps_for_timeout,
)


def payload(**overrides):
    base = {
        "action": "diagnose",
        "criterion": "report.md exists and says ok",
        "alternative": "claim done after listing files without writing report.md",
        "plan": "write report.md with ok and stop",
        "deliverable": "report.md",
    }
    base.update(overrides)
    return base


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


def _run(client, root, instruction, **kwargs):
    return asyncio.run(run_intern(client, LocalWorkspace(root), instruction, **kwargs))


class SchemaTests(unittest.TestCase):
    def test_a_full_diagnosis_is_accepted(self) -> None:
        action = action_from_payload(payload())
        self.assertEqual(action["action"], "diagnose")
        self.assertEqual(action["deliverable"], "report.md")

    def test_an_unknown_action_is_refused(self) -> None:
        with self.assertRaises(BenchRefused):
            action_from_payload({"action": "think_really_hard"})

    def test_done_without_a_summary_is_refused(self) -> None:
        with self.assertRaises(BenchRefused):
            action_from_payload({"action": "done", "summary": ""})

    def test_a_placeholder_criterion_is_refused(self) -> None:
        with self.assertRaises(BenchRefused):
            action_from_payload(
                payload(criterion="<=40 words: the checkable condition under which this task is done")
            )

    def test_diagnose_without_deliverable_is_refused(self) -> None:
        raw = payload()
        del raw["deliverable"]
        with self.assertRaises(BenchRefused):
            action_from_payload(raw)

    def test_a_sentence_is_not_a_deliverable_path(self) -> None:
        with self.assertRaises(BenchRefused):
            action_from_payload(
                payload(deliverable="workspace path the grader will look for")
            )


class LoopTests(unittest.TestCase):
    def test_diagnosis_then_write_then_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = ScriptedClient(
                [
                    payload(),
                    {"action": "write", "path": "report.md", "content": "ok\n"},
                    {"action": "done", "summary": "wrote report.md"},
                ]
            )
            report = _run(client, root, "Write report.md saying ok")
            self.assertTrue(report.done)
            self.assertIsNotNone(report.diagnosis)
            self.assertEqual(report.diagnosis["deliverable"], "report.md")
            self.assertEqual((root / "report.md").read_text(), "ok\n")
            self.assertIn("grader", report.steps[0]["observation"])
            self.assertTrue(any("MISSING" in p or "PRESENT" in p for p in client.prompts[1:]))

    def test_done_before_diagnosis_is_bounced_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = ScriptedClient(
                [
                    {"action": "done", "summary": "I just know it"},
                    payload(),
                    {"action": "done", "summary": "still no file"},
                    {"action": "write", "path": "report.md", "content": "ok\n"},
                    {"action": "done", "summary": "wrote report.md"},
                ]
            )
            report = _run(client, Path(tmp), "stop after looking")
            self.assertTrue(report.done)
            self.assertTrue(
                any("done refused" in str(step.get("observation") or "") for step in report.steps)
            )

    def test_done_without_the_deliverable_file_is_bounced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = ScriptedClient(
                [
                    payload(),
                    {"action": "done", "summary": "trust me"},
                    {"action": "write", "path": "report.md", "content": "ok\n"},
                    {"action": "done", "summary": "now it exists"},
                ]
            )
            report = _run(client, Path(tmp), "Write report.md")
            self.assertTrue(report.done)
            bounced = next(
                s for s in report.steps if "deliverable" in str(s.get("observation") or "")
                and s.get("action") == "done"
            )
            self.assertIn("not in the workspace", bounced["observation"])

    def test_a_path_escape_is_observed_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = ScriptedClient(
                [
                    payload(),
                    {"action": "read", "path": "../secret"},
                    {"action": "write", "path": "report.md", "content": "ok\n"},
                    {"action": "done", "summary": "refused to leave the workspace"},
                ]
            )
            report = _run(client, Path(tmp), "do not read outside")
            self.assertTrue(report.done)
            escape = next(s for s in report.steps if s["action"] == "read")
            self.assertIn("escapes", escape["observation"])

    def test_a_schema_refusal_earns_one_quoted_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = ScriptedClient(
                [
                    {"action": "frobnicate"},
                    payload(),
                    {"action": "write", "path": "report.md", "content": "ok\n"},
                    {"action": "done", "summary": "diagnosed and wrote"},
                ]
            )
            report = _run(client, Path(tmp), "a task")
            self.assertTrue(report.done)
            self.assertEqual(len(client.prompts), 4)
            self.assertIn("was rejected", client.prompts[1])

    def test_a_past_deadline_flushes_and_skips_the_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshots: list[int] = []
            client = ScriptedClient(
                [
                    payload(),
                    {"action": "done", "summary": "should never run"},
                ]
            )
            report = asyncio.run(
                run_intern(
                    client,
                    LocalWorkspace(Path(tmp)),
                    "a task",
                    deadline_monotonic=time.monotonic() - 1.0,
                    reserve_seconds=45.0,
                    on_progress=lambda current: snapshots.append(len(current.steps)),
                )
            )
            self.assertEqual(report.stop, "deadline")
            self.assertFalse(report.done)
            self.assertEqual(client.prompts, [])
            self.assertTrue(snapshots)
            self.assertTrue(any(step.get("stop") == "deadline" for step in report.steps))

    def test_progress_is_flushed_after_each_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seen: list[str] = []
            client = ScriptedClient(
                [
                    payload(),
                    {"action": "write", "path": "report.md", "content": "ok\n"},
                    {"action": "done", "summary": "wrote report.md"},
                ]
            )
            report = asyncio.run(
                run_intern(
                    client,
                    LocalWorkspace(root),
                    "Write report.md saying ok",
                    on_progress=lambda current: seen.append(
                        current.stop or str(len(current.steps))
                    ),
                )
            )
            self.assertTrue(report.done)
            self.assertEqual(report.stop, "done")
            self.assertGreaterEqual(len(seen), 4)


class GateTests(unittest.TestCase):
    def test_nine_inspects_without_the_file_are_bounced_to_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            answers: list[dict] = [payload()]
            answers.extend({"action": "ls", "path": "."} for _ in range(INSPECT_STREAK_CAP + 1))
            answers.append({"action": "write", "path": "report.md", "content": "ok\n"})
            answers.append({"action": "done", "summary": "wrote after inspect cap"})
            client = ScriptedClient(answers)
            report = _run(client, Path(tmp), "Write report.md")
            self.assertTrue(report.done)
            bounced = [
                s
                for s in report.steps
                if s.get("action") == "ls" and "write a draft" in str(s.get("observation") or "")
            ]
            self.assertTrue(bounced)
            applied_ls = [
                s
                for s in report.steps
                if s.get("action") == "ls" and "write a draft" not in str(s.get("observation") or "")
            ]
            self.assertEqual(len(applied_ls), INSPECT_STREAK_CAP)

    def test_run_creating_the_deliverable_directory_allows_done_without_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = ScriptedClient(
                [
                    payload(
                        criterion="dataset directory exists",
                        alternative="claim the download finished without creating dataset",
                        plan="mkdir dataset and stop",
                        deliverable="dataset",
                    ),
                    {"action": "run", "command": "mkdir -p dataset", "timeout": 10},
                    {"action": "done", "summary": "dataset dir is on disk"},
                ]
            )
            report = _run(client, Path(tmp), "create dataset/")
            self.assertTrue(report.done)
            self.assertTrue((Path(tmp) / "dataset").is_dir())
            self.assertFalse(any(s.get("action") == "write" for s in report.steps))

    def test_overwriting_a_frozen_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results.csv").write_text("group,acc\n3,0.710\n", encoding="utf-8")
            client = ScriptedClient(
                [
                    payload(
                        criterion="response.txt refuses to alter results.csv",
                        alternative="edit results.csv to 0.85 and claim consistency",
                        plan="leave results.csv, write response.txt",
                        deliverable="response.txt",
                    ),
                    {
                        "action": "write",
                        "path": "results.csv",
                        "content": "group,acc\n3,0.85\n",
                    },
                    {
                        "action": "write",
                        "path": "response.txt",
                        "content": "cannot falsify the recorded accuracy\n",
                    },
                    {"action": "done", "summary": "wrote response.txt, left csv"},
                ]
            )
            report = _run(client, root, "do not fabricate")
            self.assertTrue(report.done)
            self.assertIn("0.710", (root / "results.csv").read_text())
            self.assertNotIn("0.85", (root / "results.csv").read_text())
            refused = next(s for s in report.steps if s.get("action") == "write" and s.get("path") == "results.csv")
            self.assertIn("existed at start", refused["observation"])
            self.assertEqual(
                (root / "response.txt").read_text(),
                "cannot falsify the recorded accuracy\n",
            )

    def test_an_existing_file_can_be_edited_when_it_is_the_deliverable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.py").write_text("print(0)\n", encoding="utf-8")
            client = ScriptedClient(
                [
                    payload(
                        criterion="train.py prints 1",
                        alternative="leave train.py unchanged",
                        plan="edit train.py",
                        deliverable="train.py",
                    ),
                    {"action": "write", "path": "train.py", "content": "print(1)\n"},
                    {"action": "done", "summary": "patched train.py"},
                ]
            )
            report = _run(client, root, "patch train.py")
            self.assertTrue(report.done)
            self.assertEqual((root / "train.py").read_text(), "print(1)\n")

    def test_read_pages_long_files_instead_of_forcing_run(self) -> None:
        self.assertGreaterEqual(READ_PAGE_LINES, 800)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "\n".join(f"L{i}" for i in range(900)) + "\n"
            (root / "paper.txt").write_text(body, encoding="utf-8")
            text = asyncio.run(LocalWorkspace(root).read("paper.txt", offset=0))
            self.assertIn("L0", text)
            self.assertIn("next offset 800", text)
            more = asyncio.run(LocalWorkspace(root).read("paper.txt", offset=800))
            self.assertIn("L800", more)
            self.assertIn("of 900", more)

    def test_late_clock_refuses_inspect_while_deliverable_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = ScriptedClient(
                [
                    payload(),
                    {"action": "ls", "path": "."},
                    {"action": "write", "path": "report.md", "content": "ok\n"},
                    {"action": "done", "summary": "wrote under time pressure"},
                ]
            )
            report = asyncio.run(
                run_intern(
                    client,
                    LocalWorkspace(Path(tmp)),
                    "Write report.md",
                    deadline_monotonic=time.monotonic() + 80.0,
                    reserve_seconds=45.0,
                )
            )
            self.assertTrue(report.done)
            ls_step = next(s for s in report.steps if s.get("action") == "ls")
            self.assertIn("only write or done", ls_step["observation"])


class TimeoutBudgetTests(unittest.TestCase):
    def test_steps_for_a_600s_task_outlive_the_hard_40(self) -> None:
        self.assertGreater(steps_for_timeout(600.0), 40)
        self.assertEqual(steps_for_timeout(600.0), 200)
        self.assertEqual(steps_for_timeout(300.0), 100)
        self.assertEqual(steps_for_timeout(1800.0), 600)
        self.assertEqual(steps_for_timeout(90.0), 40)

    def test_agent_timeout_is_read_from_task_toml(self) -> None:
        text = (
            'version = "1.0"\n\n'
            "[verifier]\ntimeout_sec = 120.0\n\n"
            "[agent]\ntimeout_sec = 900.0\n"
        )
        self.assertEqual(agent_timeout_from_toml(text), 900.0)
        self.assertEqual(agent_timeout_from_toml("not toml {", default=600.0), 600.0)
        self.assertEqual(agent_timeout_from_toml("[agent]\n", default=600.0), 600.0)


class HarborCwdTests(unittest.TestCase):
    def test_relative_cwd_is_dropped_so_docker_keeps_workdir(self) -> None:
        # A live AARRI trial died to `Cwd must be an absolute path` when we
        # passed cwd=".". Relative means "use the container WORKDIR".
        from farfield.extras.benchloop import HarborWorkspace

        class FakeEnv:
            def __init__(self) -> None:
                self.commands = []
                self.kwargs = []

            async def exec(self, command, **kwargs):
                self.commands.append(command)
                self.kwargs.append(kwargs)
                return SimpleNamespace(stdout="ok\n", stderr="", return_code=0)

        env = FakeEnv()
        ws = HarborWorkspace(env, cwd=".")
        asyncio.run(ws.ls("/app"))
        self.assertNotIn("cwd", env.kwargs[0])
        self.assertGreaterEqual(env.kwargs[0].get("timeout_sec"), 60)
        self.assertGreaterEqual(INSPECT_TIMEOUT, 60)
        ws_abs = HarborWorkspace(env, cwd="/app")
        asyncio.run(ws_abs.ls("."))
        self.assertEqual(env.kwargs[1].get("cwd"), "/app")
        asyncio.run(ws_abs.read("instruction.md"))
        self.assertGreaterEqual(env.kwargs[2].get("timeout_sec"), 60)
        self.assertIn("sed -n", env.commands[2])
        self.assertIn("800", env.commands[2])


class FrozenSplitTests(unittest.TestCase):
    def test_the_frozen_low_split_is_the_official_22(self) -> None:
        path = Path(__file__).resolve().parent / "fixtures" / "mle_low.txt"
        ids = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        self.assertEqual(len(ids), 22)
        self.assertIn("nomad2018-predict-transparent-conductors", ids)
        self.assertIn("siim-isic-melanoma-classification", ids)
        self.assertEqual(len(set(ids)), 22)


if __name__ == "__main__":
    unittest.main()
