"""The two-arm probe: whitelist compile, isolated run, both arms or nothing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from farfield.extras.probeexp import (
    COMPUTE_TIERS,
    ProbeRefused,
    replication_seeds,
    resolve_tier,
    run_probe,
    spec_from_payload,
    validate_source,
    write_probe,
)

GOOD = {
    "measure": "hops from origin to end on a constructed path",
    "source": (
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
    ),
}


class ComputeTierTests(unittest.TestCase):
    def test_unknown_tier_never_invents_a_budget(self) -> None:
        self.assertEqual(resolve_tier("").name, "host")
        self.assertEqual(resolve_tier("gpu-cluster").name, "host")
        self.assertEqual(resolve_tier("HOST-HEAVY").name, "host-heavy")

    def test_numpy_needs_the_heavy_tier(self) -> None:
        source = "import numpy\n"
        with self.assertRaises(ProbeRefused):
            validate_source(source)
        with self.assertRaises(ProbeRefused):
            validate_source(source, tier=resolve_tier("host"))
        validate_source(source, tier=resolve_tier("host-heavy"))

    def test_heavy_tier_keeps_every_other_sandbox_rule(self) -> None:
        with self.assertRaises(ProbeRefused):
            validate_source("import os\n", tier=resolve_tier("host-heavy"))
        with self.assertRaises(ProbeRefused):
            validate_source(
                "x = ().__class__\n", tier=resolve_tier("host-heavy")
            )

    def test_tiers_escalate_budgets_in_order(self) -> None:
        self.assertLess(
            COMPUTE_TIERS["sandbox"].timeout_seconds,
            COMPUTE_TIERS["host"].timeout_seconds,
        )
        self.assertLess(
            COMPUTE_TIERS["host"].timeout_seconds,
            COMPUTE_TIERS["host-heavy"].timeout_seconds,
        )
        self.assertIsNotNone(COMPUTE_TIERS["host-heavy"].memory_bytes)


class CompileTests(unittest.TestCase):
    def test_a_stdlib_script_is_accepted(self) -> None:
        spec = spec_from_payload(GOOD)
        self.assertIn("hops", spec.measure)
        validate_source(spec.source)

    def test_with_open_and_lambda_are_allowed(self) -> None:
        # The first live mission's probe was refused for `with open(...)`
        # — the idiom every model reaches for. Under the import whitelist
        # these nodes add no capability, so the sandbox admits them.
        validate_source(
            "import json\n"
            "rows = sorted([3, 1, 2], key=lambda x: -x)\n"
            "with open('metrics.json', 'w') as sink:\n"
            "    json.dump({'treatment': float(rows[0]), 'control': 2.0}, sink)\n"
        )

    def test_a_plain_class_is_allowed(self) -> None:
        # The second live mission wrote its toy structure as a class.
        validate_source(
            "import json\n"
            "class Sketch:\n"
            "    def add(self, x):\n"
            "        self.total = x\n"
            "s = Sketch()\n"
            "s.add(3.0)\n"
            "json.dump({'treatment': s.total, 'control': 2.0}, open('metrics.json', 'w'))\n"
        )

    def test_bit_ops_and_nonlocal_are_allowed(self) -> None:
        # Two live probes died to `>>` and `nonlocal` — the mother tongue
        # of succinct-structure code. Under the import whitelist these add
        # no capability.
        validate_source(
            "import json\n"
            "def make_counter():\n"
            "    total = 0\n"
            "    def bump(x):\n"
            "        nonlocal total\n"
            "        total = total + (x >> 2) + (x << 1) + (x & 7) + (x | 1) + (x ^ 3) + (~x)\n"
            "        return total\n"
            "    return bump\n"
            "bump = make_counter()\n"
            "json.dump({'treatment': float(bump(9)), 'control': 2.0}, open('metrics.json', 'w'))\n"
        )

    def test_an_os_import_is_refused(self) -> None:
        with self.assertRaises(ProbeRefused):
            spec_from_payload(
                {
                    **GOOD,
                    "source": "import os\nprint(os.getcwd())\n",
                }
            )

    def test_eval_is_refused(self) -> None:
        with self.assertRaises(ProbeRefused):
            spec_from_payload({**GOOD, "source": "eval('1+1')\n"})

    def test_a_single_underscore_helper_is_allowed(self) -> None:
        validate_source(
            "import json\n"
            "def _load_fixture():\n"
            "    return 1.0\n"
            "json.dump({'treatment': float(_load_fixture()), "
            "'control': 2.0}, open('metrics.json', 'w'))\n"
        )

    def test_dunder_names_are_still_refused(self) -> None:
        with self.assertRaises(ProbeRefused):
            validate_source("x = __import__('os')\n")
        with self.assertRaises(ProbeRefused):
            validate_source("v = ().__class__\n")


class RunTests(unittest.TestCase):
    def test_a_valid_probe_reports_both_arms(self) -> None:
        spec = spec_from_payload(GOOD)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_probe(spec, Path(tmp))
        self.assertEqual(result.status, "ran")
        self.assertEqual(result.treatment, 1.0)
        self.assertEqual(result.control, 8.0)

    def test_a_probe_missing_the_control_arm_is_a_failure(self) -> None:
        spec = spec_from_payload(
            {
                **GOOD,
                "source": (
                    "import json\n"
                    "from pathlib import Path\n"
                    "Path('metrics.json').write_text("
                    "json.dumps({'treatment': 80.0}), encoding='utf-8')\n"
                ),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run_probe(spec, Path(tmp))
        self.assertEqual(result.status, "bad_metric")
        self.assertIn("control", result.error)


BAD_SOURCE = "import time\ntime.sleep(1)\n"

CARD = SimpleNamespace(
    card_id="gen_retry",
    claim="a claim",
    mechanism="a mechanism",
)
DIAGNOSIS = SimpleNamespace(
    alternative="a stronger story",
    experiment="count both arms",
    treatment_arm="with the mechanism",
    control_arm="without it",
)


class ScriptedClient:
    """Returns queued probe payloads and records every prompt."""

    def __init__(self, sources: list[str]) -> None:
        self.sources = list(sources)
        self.prompts: list[str] = []

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.prompts.append(prompt)
        payload = {"measure": "operations counted", "source": self.sources.pop(0)}
        return SimpleNamespace(
            text=json.dumps(payload),
            assert_usable=lambda: None,
        )


class RetryTests(unittest.TestCase):
    def test_a_sandbox_refusal_earns_one_quoted_retry(self) -> None:
        # Live missions showed a long tail of one-off refusals (`import
        # time`, `__slots__`); a compiler error is not a scientific verdict,
        # so the writer sees the exact rejection once and tries again.
        client = ScriptedClient([BAD_SOURCE, GOOD["source"]])
        spec = write_probe(client, CARD, DIAGNOSIS, "a topic")
        self.assertIn("treatment", spec.source)
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("rejected by the sandbox", client.prompts[1])
        self.assertIn("'time'", client.prompts[1])

    def test_the_final_refusal_still_propagates(self) -> None:
        client = ScriptedClient([BAD_SOURCE, BAD_SOURCE])
        with self.assertRaises(ProbeRefused):
            write_probe(client, CARD, DIAGNOSIS, "a topic")
        self.assertEqual(len(client.prompts), 2)

    def test_an_implementation_failure_is_quoted_without_arm_numbers(self) -> None:
        client = ScriptedClient([GOOD["source"]])
        write_probe(
            client,
            CARD,
            DIAGNOSIS,
            "a topic",
            prior_failure={"status": "crashed", "error": "constructed boom"},
        )
        prompt = client.prompts[0]
        self.assertIn("constructed boom", prompt)
        self.assertIn("implementation bug", prompt)
        self.assertNotIn("treatment: 80", prompt)
        self.assertNotIn("control: 120", prompt)


class WorldBindingTests(unittest.TestCase):
    def test_a_bound_world_refuses_an_invented_dataset(self) -> None:
        from farfield.extras.world import load_catalog

        world = load_catalog(Path(__file__).resolve().parents[1])["path-trace"]
        client = ScriptedClient([GOOD["source"], GOOD["source"]])
        with self.assertRaises(ProbeRefused) as caught:
            write_probe(client, CARD, DIAGNOSIS, "a topic", world=world)
        self.assertIn("SYNTHETIC", caught.exception.record.unlock_condition)
        self.assertIn("Do NOT invent", client.prompts[0])
        self.assertIn("Schema:", client.prompts[0])
        self.assertIn("path.json", client.prompts[0])

    def test_a_script_that_reads_data_is_admitted(self) -> None:
        from farfield.extras.world import load_catalog

        world = load_catalog(Path(__file__).resolve().parents[1])["path-trace"]
        source = (
            "import json\n"
            "from pathlib import Path\n"
            "world = json.loads(Path('data/path.json').read_text(encoding='utf-8'))\n"
            "end = len(world['nodes']) - 1\n"
            "def measure(use_shortcut):\n"
            "    hops = 0\n"
            "    pos = 0\n"
            "    while pos < end:\n"
            "        hops += 1\n"
            "        if use_shortcut and pos == 0:\n"
            "            pos = end\n"
            "        else:\n"
            "            pos += 1\n"
            "    return hops\n"
            "Path('metrics.json').write_text("
            "json.dumps({'treatment': float(measure(True)),"
            " 'control': float(measure(False))}), encoding='utf-8')\n"
        )
        client = ScriptedClient([source])
        spec = write_probe(client, CARD, DIAGNOSIS, "a topic", world=world)
        self.assertIn("data/path.json", spec.source)

    def test_naming_data_seed_is_not_reading_the_bound_automaton(self) -> None:
        from farfield.extras.world import load_catalog

        world = load_catalog(Path(__file__).resolve().parents[1])["a2a-task-lifecycle"]
        source = (
            "import json\n"
            "from pathlib import Path\n"
            "seed = json.loads((Path('data') / 'seed.json').read_text()) "
            "if False else 0\n"
            "def measure(flag):\n"
            "    n = 0\n"
            "    for i in range(4):\n"
            "        n += i if flag else 1\n"
            "    return float(n)\n"
            "Path('metrics.json').write_text("
            "json.dumps({'treatment': float(measure(True)),"
            " 'control': float(measure(False))}), encoding='utf-8')\n"
        )
        client = ScriptedClient([source, source])
        with self.assertRaises(ProbeRefused):
            write_probe(client, CARD, DIAGNOSIS, "a topic", world=world)


class ReplicationSeedTests(unittest.TestCase):
    """Executor-owned confirmation: the model is never asked for a heavy
    script, and it cannot know the seeds its script will be replayed under."""

    def test_the_model_is_never_asked_for_a_confirmation_script(self) -> None:
        # The old design prompted for a "heavy" rewrite — the audited
        # object writing its own audit evidence. The prompt must not
        # mention it, and write_probe must not accept it.
        client = ScriptedClient([GOOD["source"]])
        write_probe(client, CARD, DIAGNOSIS, "a topic")
        self.assertNotIn("CONFIRMATION RUN", client.prompts[0])
        with self.assertRaises(TypeError):
            write_probe(client, CARD, DIAGNOSIS, "a topic", heavy=True)

    def test_the_prompt_carries_the_seed_contract(self) -> None:
        client = ScriptedClient([GOOD["source"]])
        write_probe(client, CARD, DIAGNOSIS, "a topic")
        self.assertIn("data/seed.json", client.prompts[0])

    def test_seeds_are_derived_from_the_evidence_identity(self) -> None:
        # seed_i = H(experiment_digest ‖ world_digest ‖ i): recomputable
        # by any auditor, unknowable to the script at writing time
        # (the digest covers the script's own bytes), and outside the
        # executor's discretion too.
        first = replication_seeds("a" * 64, "b" * 64, 5)
        self.assertEqual(first, replication_seeds("a" * 64, "b" * 64, 5))
        self.assertEqual(len(set(first)), 5)
        self.assertNotEqual(first, replication_seeds("c" * 64, "b" * 64, 5))
        self.assertNotEqual(first, replication_seeds("a" * 64, "d" * 64, 5))

    def test_the_executor_injects_the_seed_before_the_run(self) -> None:
        spec = spec_from_payload(
            dict(
                GOOD,
                source=(
                    "import json\n"
                    "from pathlib import Path\n"
                    "seed = json.loads(Path('data/seed.json').read_text())['seed']\n"
                    "treatment = float(seed % 97)\n"
                    "control = float(seed % 89) + 1.0\n"
                    "Path('metrics.json').write_text(json.dumps("
                    "{'treatment': treatment, 'control': control}))\n"
                ),
            ),
            tier=resolve_tier("toy"),
            claim=CARD.claim,
            mechanism=CARD.mechanism,
            topic="a topic",
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run_probe(spec, Path(tmp), seed=12345)
            self.assertEqual(result.status, "ran")
            self.assertEqual(result.treatment, float(12345 % 97))
            written = json.loads((Path(tmp) / "data" / "seed.json").read_text())
            self.assertEqual(written["seed"], 12345)


class FairnessTests(unittest.TestCase):
    def test_literal_metrics_are_refused(self) -> None:
        with self.assertRaises(ProbeRefused) as caught:
            spec_from_payload(
                {
                    "measure": "hardcoded",
                    "source": (
                        "import json\n"
                        "from pathlib import Path\n"
                        "Path('metrics.json').write_text("
                        "json.dumps({'treatment': 80.0, 'control': 120.0}),"
                        " encoding='utf-8')\n"
                    ),
                }
            )
        self.assertIn("literals", caught.exception.record.unlock_condition)

    def test_pass_versus_count_is_refused(self) -> None:
        with self.assertRaises(ProbeRefused):
            spec_from_payload(
                {
                    "measure": "bugs",
                    "source": (
                        "import json\n"
                        "from pathlib import Path\n"
                        "treatment = True\n"
                        "detected = 0\n"
                        "if treatment:\n"
                        "    pass\n"
                        "else:\n"
                        "    detected += 1\n"
                        "Path('metrics.json').write_text("
                        "json.dumps({'treatment': float(detected),"
                        " 'control': 1.0}), encoding='utf-8')\n"
                    ),
                }
            )

    def test_unconditional_treatment_increment_is_refused(self) -> None:
        with self.assertRaises(ProbeRefused):
            spec_from_payload(
                {
                    "measure": "detections",
                    "source": (
                        "import json\n"
                        "from pathlib import Path\n"
                        "flag = True\n"
                        "detected = 0\n"
                        "if flag:\n"
                        "    detected += 1\n"
                        "else:\n"
                        "    if False:\n"
                        "        detected += 1\n"
                        "Path('metrics.json').write_text("
                        "json.dumps({'treatment': float(detected),"
                        " 'control': 0.0}), encoding='utf-8')\n"
                    ),
                }
            )


if __name__ == "__main__":
    unittest.main()
