"""Derived, imported, and harvested program_state worlds.

A `program_state` claim used to have exactly one way to an attested
WORLD: an operator fetching a published archive before any claim. These
tests pin the three host-side paths that replace that, and the honesty
checks on each.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from farfield.extras import chain
from farfield.extras.derive import program_state_from_traces
from farfield.extras.freeze import (
    FreezeError,
    derive_world,
    freeze_history,
    freeze_world,
    resolve_pending_worlds,
)
from farfield.extras.genworld import execute_program_state
from farfield.extras.harvest import (
    HarvestError,
    parse_recipe,
    registered,
    run_harvest,
    sibling_seeds,
)
from farfield.extras.histories import HistoryError, import_history
from farfield.extras.world import load_catalog, load_fixture, pick_acquired_world


def _trace(instance: str, label: str, tools: list[tuple[str, str]], *, run: bool = True) -> dict:
    steps = [{"t": 0, "action": "ls /testbed", "returncode": 0}]
    for index, (name, sha) in enumerate(tools):
        steps.append(
            {
                "t": len(steps),
                "action": f"cat <<'EOF' > /testbed/{name}\nprint('x')\nEOF",
                "returncode": 0,
            }
        )
        if run:
            steps.append({"t": len(steps), "action": f"cd /testbed && python3 {name} a b", "returncode": 0})
    steps.append({"t": len(steps), "action": "git diff", "returncode": 0})
    return {
        "id": f"m/{instance}",
        "label": label,
        "instance_id": instance,
        "template_id": instance.split("__")[0],
        "steps": steps,
        "created_tools": [{"name": n, "chars": 10, "sha256": s} for n, s in tools],
    }


def _traces_payload() -> dict:
    return {
        "schema": "labeled_traces",
        "id": "t",
        "object_type": "trace",
        "traces": [
            _trace("astropy__astropy-1", "resolved", [("edit_tool.py", "aaa"), ("review_fix.py", "bbb")]),
            _trace("django__django-2", "unresolved", [("edit_tool.py", "ccc")], run=False),
            _trace("sympy__sympy-3", "resolved", [("edit_tool.py", "aaa"), ("verify.py", "ddd")]),
            _trace("flask__flask-4", "resolved", []),
            _trace("numpy__numpy-5", "unresolved", [("scan.py", "eee")]),
        ],
        "n": 5,
    }


class DeriveTests(unittest.TestCase):
    def test_tool_creation_becomes_an_ordered_update_history(self) -> None:
        derived = program_state_from_traces(_traces_payload())
        self.assertEqual(execute_program_state(derived), [])
        updates = derived["updates"]
        self.assertEqual(len(updates), 6)
        self.assertEqual([u["epoch"] for u in updates], list(range(6)))
        first = updates[0]
        self.assertEqual(first["writes"], ["edit_tool.py"])
        self.assertTrue(first["accepted"])
        self.assertEqual(first["task_return"], 1.0)
        self.assertFalse(first["divergent"])
        # The self-validator the agent wrote is a validator it mutated.
        review = updates[1]
        self.assertEqual(review["validator_writes"], ["self:review_fix.py"])
        self.assertIn("edit_tool.py", review["reads"])
        # Tool written but never run: the agent did not keep it.
        django = updates[2]
        self.assertFalse(django["accepted"])
        self.assertEqual(django["task_return"], 0.0)
        # Same harness cell rewritten with other content diverges; the same
        # content does not.
        self.assertTrue(django["divergent"])
        sympy_edit = updates[3]
        self.assertEqual(sympy_edit["writes"], ["edit_tool.py"])
        self.assertTrue(sympy_edit["divergent"])
        self.assertIn("tests:astropy__astropy-1", {v["id"] for v in derived["validators"]})
        self.assertEqual(derived["derivation"]["episodes"], 4)

    def test_no_created_tools_is_a_refusal_not_an_empty_world(self) -> None:
        payload = _traces_payload()
        for trace in payload["traces"]:
            trace["created_tools"] = []
        with self.assertRaises(ValueError):
            program_state_from_traces(payload)


class DeriveWorldTests(unittest.TestCase):
    def _catalog_with_traces(self, root: Path) -> Path:
        catalog = root / "worlds"
        catalog.mkdir()
        src = root / "traces.json"
        src.write_text(json.dumps(_traces_payload()), encoding="utf-8")
        freeze_world(
            world_id="swe-traces",
            schema="labeled_traces",
            slice_rule="all",
            catalog=catalog,
            source_file=src,
            domains=("swe", "agent", "harness", "self-improvement"),
        )
        return catalog

    def test_derivation_records_parent_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = self._catalog_with_traces(root)
            child = derive_world(
                world_id="swe-selfmod",
                parent="swe-traces",
                schema="program_state",
                catalog=catalog,
            )
            self.assertEqual(child.schema, "program_state")
            self.assertEqual(child.provenance, "derived")
            parent = load_fixture(catalog / "swe-traces")
            origin = json.loads((child.root / "origin.json").read_text(encoding="utf-8"))
            self.assertEqual(origin["derived_from"]["world_id"], "swe-traces")
            self.assertEqual(origin["derived_from"]["digest"], parent.digest)
            self.assertIn("created_tools", origin["derived_from"]["rule"])
            manifest = json.loads((child.root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["provenance"], "derived")
            self.assertEqual(manifest["derived_from"], "swe-traces")
            payload = json.loads((child.root / "world.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "program_state")
            self.assertEqual(execute_program_state(payload), [])
            events = chain.read_events(catalog / "chain.jsonl")
            self.assertEqual(events[-1]["payload"].get("provenance"), "derived")

    def test_unregistered_pair_and_missing_parent_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = self._catalog_with_traces(root)
            with self.assertRaises(FreezeError):
                derive_world(world_id="x", parent="swe-traces", schema="fasta", catalog=catalog)
            with self.assertRaises(FreezeError):
                derive_world(world_id="x", parent="nope", schema="program_state", catalog=catalog)

    def test_a_program_state_wish_without_a_url_is_derived_from_the_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = self._catalog_with_traces(root)
            var = root / "var"
            var.mkdir()
            (var / "world_wishlist.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "key": "k",
                                "object_type": "executable",
                                "freeze_schema": "program_state",
                                "freeze_url": "",
                                "named_instance": "runtime self-modification",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            acquired = resolve_pending_worlds(root)
            self.assertEqual(len(acquired), 1)
            self.assertEqual(acquired[0]["schema"], "program_state")
            fixture = load_fixture(catalog / acquired[0]["world_id"])
            self.assertEqual(fixture.provenance, "derived")
            # A second wish for the same pair reuses the child, it does not
            # derive a duplicate.
            payload = json.loads((var / "world_wishlist.json").read_text(encoding="utf-8"))
            payload["entries"].append(
                {
                    "key": "k2",
                    "object_type": "executable",
                    "freeze_schema": "program_state",
                    "named_instance": "another wording",
                }
            )
            (var / "world_wishlist.json").write_text(json.dumps(payload), encoding="utf-8")
            resolve_pending_worlds(root)
            payload = json.loads((var / "world_wishlist.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [e["acquired_id"] for e in payload["entries"]],
                [acquired[0]["world_id"]] * 2,
            )
            derived = [
                p for p in catalog.iterdir()
                if (p / "manifest.json").is_file()
                and json.loads((p / "manifest.json").read_text())["schema"] == "program_state"
            ]
            self.assertEqual(len(derived), 1)

    def test_auto_binds_only_acquired_worlds_of_the_task_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = self._catalog_with_traces(root)
            topic = (
                "code world models of executable program state as the world "
                "for an agent harness in recursive self-improvement"
            )
            # Only a shipped-style (no provenance) labeled_traces world: no bind.
            self.assertIsNone(pick_acquired_world(topic, load_catalog(root)))
            derive_world(
                world_id="swe-selfmod",
                parent="swe-traces",
                schema="program_state",
                catalog=catalog,
                domains=("program", "executable", "agent", "harness", "self-improvement"),
            )
            picked = pick_acquired_world(topic, load_catalog(root))
            self.assertIsNotNone(picked)
            self.assertEqual(picked.id, "swe-selfmod")
            # A different object family does not bind the same acquired world.
            self.assertIsNone(
                pick_acquired_world(
                    "compress genomic sequence collections with succinct data structures",
                    load_catalog(root),
                )
            )


def _write_openevolve_checkpoint(root: Path, *, n: int = 6, seed: int = 0) -> None:
    programs = root / "programs"
    programs.mkdir(parents=True)
    ids = []
    for index in range(n):
        ident = f"p{seed}-{index}"
        ids.append(ident)
        row = {
            "id": ident,
            "code": f"def f():\n    return {index + seed}\n",
            "language": "python",
            "parent_id": ids[index - 1] if index else None,
            "generation": index,
            "iteration_found": index,
            "metrics": {"combined_score": 0.5 + 0.05 * index} if index != 3 else {},
            "metadata": {"island": index % 2},
        }
        (programs / f"{ident}.json").write_text(json.dumps(row), encoding="utf-8")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "islands": [[ids[0], ids[2]], [ids[5]]],
                "archive": [ids[4], ids[5]],
                "best_program_id": ids[5],
                "island_best_programs": [ids[2], ids[5]],
            }
        ),
        encoding="utf-8",
    )


def _write_dgm_run(root: Path) -> None:
    root.mkdir(parents=True)
    generations = [
        {
            "generation": 0,
            "selfimprove_entries": [["initial", "solve_stochasticity"], ["initial", "django__django-1"]],
            "children": ["c0", "c1"],
            "children_compiled": ["c0"],
            "archive": ["initial", "c0"],
        },
        {
            "generation": 1,
            "selfimprove_entries": [["c0", "solve_empty_patches"], ["c0", "astropy__astropy-2"]],
            "children": ["c2", "c3"],
            "children_compiled": ["c2", "c3"],
            "archive": ["initial", "c0", "c2", "c3"],
        },
    ]
    (root / "dgm_metadata.jsonl").write_text(
        "".join(json.dumps(g, indent=2) + "\n" for g in generations), encoding="utf-8"
    )
    meta = {
        "c0": ("initial", 0.25, True, "tools/edit.py"),
        "c1": ("initial", 0.0, False, ""),
        "c2": ("c0", 0.35, True, "tools/patch_review.py"),
        "c3": ("c0", 0.3, True, "coding_agent.py"),
    }
    for child, (parent, score, compiled, touched) in meta.items():
        (root / child).mkdir()
        (root / child / "metadata.json").write_text(
            json.dumps(
                {
                    "run_id": child,
                    "parent_commit": parent,
                    "entry": "e",
                    "is_compiled": compiled,
                    "overall_performance": {"accuracy_score": score},
                }
            ),
            encoding="utf-8",
        )
        if touched:
            (root / child / "model_patch.diff").write_text(
                f"diff --git a/{touched} b/{touched}\n--- a/{touched}\n+++ b/{touched}\n@@ -1 +1 @@\n-x\n+y\n",
                encoding="utf-8",
            )


class HistoryImportTests(unittest.TestCase):
    def test_openevolve_checkpoint_is_a_fixed_evaluator_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint_10"
            _write_openevolve_checkpoint(root)
            payload, manifest = import_history("openevolve", root)
            self.assertEqual(manifest["format"], "openevolve")
            self.assertIn("metadata.json", manifest["files"])
            self.assertEqual(len(manifest["source_digest"]), 64)
            updates = payload["updates"]
            self.assertEqual(len(updates), 6)
            self.assertTrue(all(u["validator_writes"] == [] for u in updates))
            self.assertTrue(updates[4]["accepted"] and updates[5]["accepted"])
            self.assertFalse(updates[1]["accepted"])
            # Island membership is population, not acceptance: p0-2 sits in
            # an island but not in the archive.
            self.assertFalse(updates[2]["accepted"])
            self.assertTrue(updates[2]["in_population"])
            self.assertTrue(updates[3]["divergent"])  # no metrics
            self.assertEqual(updates[1]["reads"], ["p0-0"])
            self.assertEqual(execute_program_state(payload), [])

    def test_dgm_run_maps_archive_membership_and_self_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_dgm_run(root)
            payload, manifest = import_history("dgm", root)
            self.assertEqual(manifest["format"], "dgm")
            updates = {u["id"]: u for u in payload["updates"]}
            self.assertEqual(list(updates), ["c0", "c1", "c2", "c3"])
            self.assertTrue(updates["c0"]["accepted"])
            self.assertFalse(updates["c1"]["accepted"])
            self.assertTrue(updates["c1"]["divergent"])
            self.assertEqual(updates["c2"]["validator_writes"], ["self:tools/patch_review.py"])
            self.assertEqual(updates["c2"]["reads"], ["c0"])
            self.assertEqual(updates["c2"]["task_return"], 0.35)
            self.assertEqual(execute_program_state(payload), [])

    def test_unknown_format_and_empty_dirs_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(HistoryError):
                import_history("sica", Path(tmp))
            with self.assertRaises(HistoryError):
                import_history("openevolve", Path(tmp))
            with self.assertRaises(HistoryError):
                import_history("dgm", Path(tmp))

    def test_freeze_history_digests_what_it_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint_10"
            _write_openevolve_checkpoint(root)
            catalog = Path(tmp) / "worlds"
            fixture = freeze_history(
                world_id="oe-run", run_dir=root, fmt="openevolve", catalog=catalog
            )
            _, manifest = import_history("openevolve", root)
            self.assertEqual(fixture.source_digest, manifest["source_digest"])
            self.assertEqual(fixture.provenance, "")
            origin = json.loads((fixture.root / "origin.json").read_text(encoding="utf-8"))
            self.assertEqual(origin["imported"]["format"], "openevolve")
            with self.assertRaises(FreezeError):
                freeze_history(world_id="oe-run", run_dir=root, fmt="openevolve", catalog=catalog)


class HarvestTests(unittest.TestCase):
    """A harvest runs a harness under a recipe registered first."""

    def _recipe(self, tmp: Path, *, seed: int = 0, world_id: str = "oe-harvest") -> dict:
        script = tmp / "fake_harness.py"
        script.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "seed = int(sys.argv[1]); n = int(sys.argv[2])\n"
            "root = Path('out') / ('checkpoint_%d' % n)\n"
            "(root / 'programs').mkdir(parents=True, exist_ok=True)\n"
            "ids = []\n"
            "for i in range(n):\n"
            "    ident = 'p%d-%d' % (seed, i); ids.append(ident)\n"
            "    row = {'id': ident, 'code': 'x', 'parent_id': ids[i-1] if i else None,\n"
            "           'generation': i, 'iteration_found': i,\n"
            "           'metrics': {'combined_score': 0.1 * i + seed}, 'metadata': {'island': 0}}\n"
            "    (root / 'programs' / (ident + '.json')).write_text(json.dumps(row))\n"
            "(root / 'metadata.json').write_text(json.dumps({'islands': [ids[-2:]], 'archive': ids[-1:],\n"
            "    'best_program_id': ids[-1], 'island_best_programs': ids[-1:]}))\n",
            encoding="utf-8",
        )
        return {
            "world_id": world_id,
            "harness": "fake-openevolve",
            "command": [sys_executable(), str(script), "{seed}", "{iterations}"],
            "format": "openevolve",
            "output": "out/checkpoint_6",
            "workdir": str(tmp),
            "seed": seed,
            "iterations": 6,
            "timeout": 60,
        }

    def test_recipe_is_registered_before_the_run_and_the_world_says_harvested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "worlds"
            recipe = parse_recipe(self._recipe(root))
            self.assertFalse(registered(recipe, catalog=catalog))
            result = run_harvest(recipe, catalog=catalog)
            self.assertTrue(result["ok"])
            self.assertEqual(result["provenance"], "harvested")
            events = chain.read_events(catalog / "chain.jsonl")
            kinds = [e["kind"] for e in events]
            self.assertEqual(kinds, [chain.REGISTER_HARVEST, chain.EXECUTE_HARVEST, chain.FREEZE_WORLD])
            self.assertEqual(events[0]["payload"]["recipe_digest"], recipe.digest())
            fixture = load_fixture(catalog / "oe-harvest")
            self.assertEqual(fixture.provenance, "harvested")
            origin = json.loads((fixture.root / "origin.json").read_text(encoding="utf-8"))
            run = origin["harness_run"]
            self.assertEqual(run["exit_code"], 0)
            self.assertEqual(run["seed"], 0)
            self.assertEqual(run["recipe_digest"], recipe.digest())
            manifest = json.loads((fixture.root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["recipe_body"], recipe.body_digest())
            self.assertEqual(sibling_seeds("oe-harvest", catalog=catalog), [])

    def test_a_second_seed_is_the_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "worlds"
            run_harvest(parse_recipe(self._recipe(root, seed=0)), catalog=catalog)
            run_harvest(
                parse_recipe(self._recipe(root, seed=7, world_id="oe-harvest-s7")),
                catalog=catalog,
            )
            siblings = sibling_seeds("oe-harvest", catalog=catalog)
            self.assertEqual([s["world_id"] for s in siblings], ["oe-harvest-s7"])
            self.assertEqual(siblings[0]["seed"], 7)

    def test_reuse_run_freezes_from_the_recorded_execution_without_rerunning(self) -> None:
        # An importer fix must not cost another paid harness run: the chain
        # already says this exact recipe ran with exit 0 and the output is
        # still there, so the freeze cites that run.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "worlds"
            recipe = parse_recipe(self._recipe(root))
            with self.assertRaises(HarvestError):
                run_harvest(recipe, catalog=catalog, reuse_run=True)
            first = run_harvest(recipe, catalog=catalog)
            calls: list[list[str]] = []

            def must_not_run(argv, cwd, env, timeout):
                calls.append(list(argv))
                raise AssertionError("harness re-ran under reuse_run")

            again = run_harvest(recipe, catalog=catalog, force=True, reuse_run=True, runner=must_not_run)
            self.assertEqual(calls, [])
            self.assertEqual(again["harness_run"]["exit_code"], 0)
            self.assertEqual(again["harness_run"]["recipe_digest"], recipe.digest())
            self.assertEqual(again["harness_run"]["chain_seq"], 1)
            self.assertEqual(again["source_digest"], first["source_digest"])
            kinds = [e["kind"] for e in chain.read_events(catalog / "chain.jsonl")]
            self.assertEqual(kinds.count(chain.EXECUTE_HARVEST), 1)
            self.assertEqual(kinds.count(chain.FREEZE_WORLD), 2)

    def test_a_failing_harness_freezes_nothing_but_is_on_the_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "worlds"
            body = self._recipe(root)
            body["command"] = [sys_executable(), "-c", "import sys; sys.exit(3)"]
            recipe = parse_recipe(body)
            with self.assertRaises(HarvestError):
                run_harvest(recipe, catalog=catalog)
            self.assertFalse((catalog / "oe-harvest").exists())
            kinds = [e["kind"] for e in chain.read_events(catalog / "chain.jsonl")]
            self.assertEqual(kinds, [chain.REGISTER_HARVEST, chain.EXECUTE_HARVEST])

    def test_recipes_are_argv_lists_with_importable_formats(self) -> None:
        with self.assertRaises(HarvestError):
            parse_recipe({"world_id": "x", "harness": "h", "command": [], "format": "openevolve", "output": "o"})
        with self.assertRaises(HarvestError):
            parse_recipe({"world_id": "x", "harness": "h", "command": ["a"], "format": "sica", "output": "o"})
        recipe = parse_recipe(
            {"world_id": "x", "harness": "h", "command": "python run.py --seed {seed}", "format": "dgm", "output": "o"}
        )
        self.assertEqual(recipe.command, ("python", "run.py", "--seed", "{seed}"))
        self.assertNotEqual(recipe.digest(), recipe.body_digest())


def sys_executable() -> str:
    import sys

    return sys.executable


class WishlistUrlTests(unittest.TestCase):
    """P8: a landing page is not a data source, decided before any fetch."""

    def test_landing_pages_and_documents_are_refused_without_a_fetch(self) -> None:
        from farfield.extras.freeze import url_refusal

        self.assertTrue(url_refusal("https://arxiv.org/abs/2605.05138", "symbolic_trace"))
        self.assertTrue(url_refusal("https://doi.org/10.1000/xyz", "numeric_table"))
        self.assertTrue(url_refusal("https://github.com/org/repo", "labeled_traces"))
        self.assertTrue(url_refusal("https://example.org/paper.pdf", "text_stream"))
        self.assertTrue(url_refusal("ftp://example.org/data.txt", "text_stream"))
        self.assertEqual(
            url_refusal("https://github.com/org/repo/releases/download/v1/data.zip", "labeled_traces"), ""
        )
        self.assertEqual(url_refusal("https://snap.stanford.edu/data/ca-GrQc.txt.gz", "undirected_graph"), "")

    def test_a_refused_wish_is_recorded_and_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "worlds").mkdir()
            var = root / "var"
            var.mkdir()
            (var / "world_wishlist.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "key": "formula|symbolic_trace",
                                "object_type": "formula",
                                "freeze_schema": "symbolic_trace",
                                "freeze_url": "https://arxiv.org/abs/2605.05138",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(resolve_pending_worlds(root), [])
            payload = json.loads((var / "world_wishlist.json").read_text(encoding="utf-8"))
            failed = payload["entries"][0]["acquire_failed"]
            self.assertEqual(failed["url"], "https://arxiv.org/abs/2605.05138")
            self.assertIn("landing page", failed["reason"])
            # Second start: same url, no new attempt (and no network).
            self.assertEqual(resolve_pending_worlds(root), [])


class ResumeTests(unittest.TestCase):
    """P6: a spent balance is not retried, and the block says how to resume."""

    def test_quota_exhaustion_is_recognised_and_the_hint_names_replay(self) -> None:
        from farfield.extras.llm import RESUME_HINT, quota_exhausted

        self.assertTrue(quota_exhausted('{"error": {"code": "credit_balance_exhausted"}}'))
        self.assertTrue(quota_exhausted("You have no credits remaining."))
        self.assertTrue(quota_exhausted('"type": "insufficient_quota"'))
        self.assertFalse(quota_exhausted("Rate limit reached for requests"))
        self.assertIn("rerun the same command", RESUME_HINT)
        self.assertIn("replay", RESUME_HINT)


class ObjectPropertyTests(unittest.TestCase):
    """P4: what the bytes are, matched to what the claim says its object is."""

    def test_properties_are_read_off_the_bytes_and_written_on_the_manifest(self) -> None:
        from farfield.extras.objectprops import object_properties, required_properties

        derived = program_state_from_traces(_traces_payload())
        props = object_properties("program_state", derived)
        self.assertEqual(
            props,
            {
                "has_oracle": True,
                "validators_mutable": True,
                "gate_endogenous": True,
                "acceptance_varies": True,
                "self_modifying": True,
            },
        )
        # An evaluator's own score is not an oracle.
        fixed = {
            "updates": [
                {"accepted": True, "validator_writes": [], "writes": ["program"], "task_return": 1.0},
                {"accepted": False, "validator_writes": [], "writes": ["program"], "task_return": 0.5},
            ],
            "oracle": {"field": "task_return", "independent_of_gate": False},
        }
        self.assertFalse(object_properties("program_state", fixed)["has_oracle"])
        self.assertFalse(object_properties("program_state", fixed)["validators_mutable"])
        self.assertEqual(
            required_properties("evolutionary search where a fixed evaluator ranks candidate programs"),
            {"validators_mutable": False},
        )
        self.assertEqual(
            required_properties("agents that rewrite their own validators and the false acceptance this causes"),
            {"validators_mutable": True, "has_oracle": True},
        )
        # Contradictory statements cancel.
        self.assertEqual(
            required_properties("a fixed evaluator; agents rewrite their own validators")
            .get("validators_mutable", "cancelled"),
            "cancelled",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "worlds"
            catalog.mkdir()
            src = root / "traces.json"
            src.write_text(json.dumps(_traces_payload()), encoding="utf-8")
            freeze_world(world_id="swe-traces", schema="labeled_traces", slice_rule="all", catalog=catalog, source_file=src)
            child = derive_world(world_id="swe-selfmod", parent="swe-traces", schema="program_state", catalog=catalog,
                                 domains=("program", "executable", "agent", "harness", "self-improvement"))
            manifest = json.loads((child.root / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["object_properties"]["validators_mutable"])
            # A claim about a fixed evaluator does not bind this world even
            # though schema and domain words match.
            self.assertIsNone(
                pick_acquired_world(
                    "executable program state under a fixed evaluator in an agent harness for self-improvement",
                    load_catalog(root),
                )
            )
            self.assertIsNotNone(
                pick_acquired_world(
                    "executable program state in an agent harness whose agents rewrite their own validators",
                    load_catalog(root),
                )
            )


class DVSanityTests(unittest.TestCase):
    """P2: a measure of an oracle-relative quantity must respond to the oracle."""

    def test_oracle_placebo_permutes_only_the_oracle_field(self) -> None:
        from farfield.extras.dvsanity import dv_blind, materialize_oracle_placebo, oracle_dependent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "worlds"
            catalog.mkdir()
            src = root / "traces.json"
            src.write_text(json.dumps(_traces_payload()), encoding="utf-8")
            freeze_world(world_id="swe-traces", schema="labeled_traces", slice_rule="all", catalog=catalog, source_file=src)
            child = derive_world(world_id="swe-selfmod", parent="swe-traces", schema="program_state", catalog=catalog)
            placebo = materialize_oracle_placebo(child, root / "placebo")
            self.assertIsNotNone(placebo)
            real = json.loads((child.root / "world.json").read_text(encoding="utf-8"))["updates"]
            fake = json.loads((placebo.root / "world.json").read_text(encoding="utf-8"))["updates"]
            self.assertEqual([u["accepted"] for u in real], [u["accepted"] for u in fake])
            self.assertEqual([u["writes"] for u in real], [u["writes"] for u in fake])
            self.assertEqual(sorted(u["task_return"] for u in real), sorted(u["task_return"] for u in fake))
            self.assertNotEqual([u["task_return"] for u in real], [u["task_return"] for u in fake])
        self.assertTrue(dv_blind((0.4279, 0.4279), (0.4279, 0.4279)))
        self.assertFalse(dv_blind((0.30, 0.31), (0.30, 0.25)))
        self.assertTrue(oracle_dependent("reduce false_accept_fraction at matched accepted_fraction"))
        self.assertFalse(oracle_dependent("raise accepted_fraction versus uniform dropout"))
