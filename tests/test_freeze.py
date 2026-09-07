"""Host-side freeze: retrieve, hash, slice, write a world. No probe network."""

from __future__ import annotations

import gzip
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from farfield.cli import main
from farfield.extras.freeze import (
    FreezeError,
    freeze_world,
    parse_dot_machine,
    parse_slice_rule,
    resolve_pending_worlds,
    unique_undirected,
)
from farfield.extras.world import (
    infer_requirement,
    load_catalog,
    load_fixture,
    match_world,
    pick_world,
)

ROOT = Path(__file__).resolve().parents[1]

SNAP_CA_GRQC = (
    "# Directed graph (each unordered pair of nodes is saved once)\n"
    "# Nodes: 6 Edges: 8\n"
    "1\t2\n"
    "2\t1\n"
    "1\t3\n"
    "3\t1\n"
    "2\t3\n"
    "3\t2\n"
    "3\t4\n"
    "4\t3\n"
    "4\t5\n"
    "5\t4\n"
)

SNAP_URL = "https://snap.stanford.edu/data/ca-GrQc.txt.gz"


def _write_source(folder: Path, text: str, *, gzipped: bool = False) -> Path:
    payload = text.encode("utf-8")
    if gzipped:
        path = folder / "graph.txt.gz"
        path.write_bytes(gzip.compress(payload))
        return path
    path = folder / "graph.txt"
    path.write_bytes(payload)
    return path


class SliceRuleTests(unittest.TestCase):
    def test_first_edges_and_all_are_the_shipped_rules(self) -> None:
        self.assertEqual(parse_slice_rule("all"), {"kind": "all"})
        self.assertEqual(parse_slice_rule("first_edges:10000"), {"kind": "first_edges", "n": 10000})
        self.assertEqual(parse_slice_rule("first_bases:50000"), {"kind": "first_bases", "n": 50000})
        self.assertEqual(parse_slice_rule("first_tokens:50000"), {"kind": "first_tokens", "n": 50000})
        self.assertEqual(parse_slice_rule("first_rows:10000"), {"kind": "first_rows", "n": 10000})
        self.assertEqual(parse_slice_rule("first_states:16"), {"kind": "first_states", "n": 16})
        self.assertEqual(parse_slice_rule("first_traces:50"), {"kind": "first_traces", "n": 50})
        with self.assertRaises(FreezeError):
            parse_slice_rule("biggest_clique")
        with self.assertRaises(FreezeError):
            parse_slice_rule("first_edges:0")


class FreezeIngestTests(unittest.TestCase):
    def test_a_local_edgelist_freezes_an_official_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_source(root, SNAP_CA_GRQC, gzipped=True)
            fixture = freeze_world(
                world_id="tiny-snap",
                schema="undirected_graph",
                slice_rule="first_edges:3",
                catalog=root / "worlds",
                url=SNAP_URL,
                source_file=source,
                title="tiny SNAP prefix",
                source="synthetic listing for freeze tests",
                domains=("graph", "snap"),
                retrieved_at="2026-08-18",
            )
            graph = json.loads((fixture.root / "graph.json").read_text(encoding="utf-8"))
            origin = json.loads((fixture.root / "origin.json").read_text(encoding="utf-8"))
            self.assertEqual(graph["m"], 3)
            self.assertEqual(graph["edges"], [[1, 2], [1, 3], [2, 3]])
            self.assertEqual(origin["url"], SNAP_URL)
            self.assertEqual(origin["slice_rule"], {"kind": "first_edges", "n": 3})
            self.assertEqual(origin["parent"]["m"], 5)
            self.assertEqual(origin["source_digest"], fixture.source_digest)
            reloaded = load_fixture(fixture.root)
            self.assertEqual(reloaded.digest, fixture.digest)
            self.assertEqual(reloaded.slice_rule, {"kind": "first_edges", "n": 3})
            listed, _ = unique_undirected(
                [
                    (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2),
                    (3, 4), (4, 3), (4, 5), (5, 4),
                ]
            )
            self.assertEqual(listed[:3], graph["edges"])

    def test_missing_source_is_blocked_and_overwrite_needs_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "worlds"
            with self.assertRaises(FreezeError):
                freeze_world(
                    world_id="ghost",
                    schema="undirected_graph",
                    slice_rule="all",
                    catalog=root,
                )
            source = _write_source(Path(tmp), SNAP_CA_GRQC)
            freeze_world(
                world_id="once",
                schema="undirected_graph",
                slice_rule="all",
                catalog=root,
                source_file=source,
            )
            with self.assertRaises(FreezeError):
                freeze_world(
                    world_id="once",
                    schema="undirected_graph",
                    slice_rule="all",
                    catalog=root,
                    source_file=source,
                )
            again = freeze_world(
                world_id="once",
                schema="undirected_graph",
                slice_rule="first_edges:2",
                catalog=root,
                source_file=source,
                force=True,
            )
            graph = json.loads((again.root / "graph.json").read_text(encoding="utf-8"))
            self.assertEqual(graph["m"], 2)

    def test_cli_freeze_writes_a_catalog_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = _write_source(Path(tmp), SNAP_CA_GRQC)
            catalog = Path(tmp) / "worlds"
            code = main(
                [
                    "freeze",
                    "--id",
                    "cli-snap",
                    "--file",
                    str(source),
                    "--url",
                    SNAP_URL,
                    "--slice",
                    "first_edges:2",
                    "--catalog",
                    str(catalog),
                    "--title",
                    "cli prefix",
                    "--domain",
                    "graph",
                ]
            )
            self.assertEqual(code, 0)
            fixture = load_fixture(catalog / "cli-snap")
            self.assertEqual(fixture.schema, "undirected_graph")
            graph = json.loads((fixture.root / "graph.json").read_text(encoding="utf-8"))
            self.assertEqual(graph["edges"], [[1, 2], [1, 3]])

    def test_cli_missing_source_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(
                [
                    "freeze",
                    "--id",
                    "ghost",
                    "--catalog",
                    str(Path(tmp) / "worlds"),
                ]
            )
            self.assertEqual(code, 2)


class ShippedSnapTests(unittest.TestCase):
    def test_ca_grqc_prefix_is_attested_and_preferred_for_connectivity(self) -> None:
        catalog = load_catalog(ROOT)
        self.assertIn("snap-ca-grqc", catalog)
        fixture = catalog["snap-ca-grqc"]
        self.assertEqual(fixture.schema, "undirected_graph")
        self.assertEqual(fixture.slice_rule, {"kind": "first_edges", "n": 10000})
        self.assertEqual(
            fixture.source_digest,
            "a254442cdf5d684712578b630c2e0d7543518ab154ef2341cabb607572ce7230",
        )
        self.assertTrue(fixture.source_url.endswith("ca-GrQc.txt.gz"))
        graph = json.loads((fixture.root / "graph.json").read_text(encoding="utf-8"))
        origin = json.loads((fixture.root / "origin.json").read_text(encoding="utf-8"))
        self.assertEqual(graph["m"], 10000)
        self.assertEqual(graph["edges"][0], [937, 3466])
        self.assertEqual(origin["parent"]["m"], 14484)
        self.assertEqual(origin["parent"]["listed_m"], 28980)
        self.assertEqual(
            pick_world(
                "dynamic graph connectivity under adversarial edge updates",
                catalog,
            ).id,
            "snap-ca-grqc",
        )


class ShippedMidbandTests(unittest.TestCase):
    def test_sequence_stream_and_table_are_mid_band(self) -> None:
        catalog = load_catalog(ROOT)
        phage = catalog["phage-lambda"]
        self.assertEqual(phage.schema, "fasta")
        self.assertEqual(phage.slice_rule, {"kind": "first_bases", "n": 50000})
        meta = json.loads((phage.root / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["bp"], 48502)
        pride = catalog["gutenberg-pride"]
        self.assertEqual(pride.schema, "text_stream")
        stream = json.loads((pride.root / "stream.json").read_text(encoding="utf-8"))
        self.assertEqual(stream["n"], 50000)
        letters = catalog["uci-letters"]
        self.assertEqual(letters.schema, "numeric_table")
        table = json.loads((letters.root / "table.json").read_text(encoding="utf-8"))
        self.assertEqual(table["n"], 10000)
        self.assertEqual(table["d"], 16)
        self.assertEqual(pick_world("compress genomic sequence collections", catalog).id, "phage-lambda")
        self.assertIsNone(
            pick_world("learned index structures for high-dimensional similarity search", catalog)
        )
        self.assertIsNone(
            pick_world("space-efficient sketches for heavy hitters in adversarial streams", catalog)
        )


class OtherSchemaFreezeTests(unittest.TestCase):
    def test_fasta_and_table_freeze_from_local_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / "s.fasta"
            fasta.write_text(">x\nACGTACGTACGT\n", encoding="utf-8")
            seq = freeze_world(
                world_id="tiny-seq",
                schema="fasta",
                slice_rule="first_bases:8",
                catalog=root / "worlds",
                url="https://example.invalid/s.fasta",
                source_file=fasta,
            )
            meta = json.loads((seq.root / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["bp"], 8)
            csv_path = root / "t.csv"
            csv_path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
            table = freeze_world(
                world_id="tiny-table",
                schema="numeric_table",
                slice_rule="first_rows:2",
                catalog=root / "worlds",
                source_file=csv_path,
            )
            payload = json.loads((table.root / "table.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["n"], 2)
            self.assertEqual(payload["columns"], ["a", "b"])


def _live_swe_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for model, instance, resolved, tool in (
            ("model-a", "repo__repo-1", True, True),
            ("model-a", "repo__repo-2", False, False),
            ("model-b", "other__other-3", True, False),
        ):
            folder = f"swebench_verified/{model}/{instance}"
            trajectory = {
                "instance_id": instance,
                "trajectory_format": "mini-swe-agent-1",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "issue"},
                    {
                        "role": "assistant",
                        "content": "THOUGHT: inspect\n```bash\npytest -q\n```",
                        "extra": {"response": {"model": f"{model}@date"}},
                    },
                    {
                        "role": "user",
                        "content": "<returncode>0</returncode>\n<output>ok</output>",
                    },
                ],
                "info": {
                    "mini_version": "1.14.2",
                    "submission": "diff --git a/x b/x",
                    "model_stats": {"instance_cost": 1.5, "api_calls": 2},
                },
            }
            archive.writestr(
                f"{folder}/{instance}.traj.json", json.dumps(trajectory)
            )
            archive.writestr(
                f"{folder}/created_tools.json",
                json.dumps({"helper.py": "print('x')"}) if tool else "{}",
            )
        archive.writestr(
            "swebench_verified/model-a/eval_result.json",
            json.dumps(
                {
                    "resolved_ids": ["repo__repo-1"],
                    "unresolved_ids": ["repo__repo-2"],
                    "error_ids": [],
                    "empty_patch_ids": [],
                }
            ),
        )
        archive.writestr(
            "swebench_verified/model-b/eval_result.json",
            json.dumps(
                {
                    "resolved_ids": ["other__other-3"],
                    "unresolved_ids": [],
                    "error_ids": [],
                    "empty_patch_ids": [],
                }
            ),
        )
    return buffer.getvalue()


def _program_history(n: int = 8) -> dict:
    cells = ["cell_0", "cell_1", "cell_2"]
    validators = [
        {"id": "v_guard", "reads": ["cell_1"]},
        {"id": "v_frozen", "reads": ["cell_0"]},
    ]
    updates = []
    for i in range(n):
        on_cycle = i % 3 == 1
        updates.append(
            {
                "id": f"u{i}",
                "epoch": i,
                "writes": ["cell_1" if on_cycle else f"cell_{i % 3}"],
                "reads": [f"cell_{(i + 1) % 3}"],
                "validator_writes": ["v_guard"] if on_cycle else [],
                "accepted": bool(on_cycle or i % 2 == 0),
                "divergent": i % 3 != 0,
                "task_return": 0.5,
            }
        )
    return {
        "cells": cells,
        "validators": validators,
        "invariant": "an update may not approve itself",
        "updates": updates,
    }


class ProgramStateFreezeTests(unittest.TestCase):
    """program_state is freezable: a host exports a self-modification history."""

    def test_program_state_is_on_the_freezable_list(self) -> None:
        from farfield.extras.schemas import FREEZABLE_SCHEMAS

        self.assertIn("program_state", FREEZABLE_SCHEMAS)
        self.assertEqual(parse_slice_rule("first_updates:6"), {"kind": "first_updates", "n": 6})

    def test_json_history_freezes_and_slices_first_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dgm-archive.json"
            source.write_text(json.dumps(_program_history(10)), encoding="utf-8")
            fixture = freeze_world(
                world_id="dgm-archive-test",
                schema="program_state",
                slice_rule="first_updates:6",
                catalog=root / "worlds",
                source_file=source,
                url="https://example.invalid/dgm-archive.json",
            )
            world = json.loads((fixture.root / "world.json").read_text(encoding="utf-8"))
            origin = json.loads((fixture.root / "origin.json").read_text(encoding="utf-8"))
            self.assertEqual(fixture.schema, "program_state")
            self.assertEqual(world["n"], 6)
            self.assertEqual([row["id"] for row in world["updates"]], [f"u{i}" for i in range(6)])
            self.assertEqual(world["cells"], ["cell_0", "cell_1", "cell_2"])
            self.assertEqual(origin["parent"]["updates"], 10)
            self.assertEqual(origin["parent"]["format"], "json_program_state")
            self.assertEqual(load_fixture(fixture.root).digest, fixture.digest)
            self.assertNotEqual(fixture.role, "generated")

    def test_jsonl_history_with_header_freezes(self) -> None:
        history = _program_history(6)
        header = {
            "cells": history["cells"],
            "validators": history["validators"],
            "invariant": history["invariant"],
        }
        lines = [json.dumps(header)] + [json.dumps(row) for row in history["updates"]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "history.jsonl"
            source.write_text("\n".join(lines) + "\n", encoding="utf-8")
            fixture = freeze_world(
                world_id="dgm-jsonl-test",
                schema="program_state",
                slice_rule="all",
                catalog=root / "worlds",
                source_file=source,
                url="https://example.invalid/history.jsonl",
            )
            origin = json.loads((fixture.root / "origin.json").read_text(encoding="utf-8"))
            self.assertEqual(origin["parent"]["format"], "jsonl_program_state")
            self.assertEqual(origin["slice"]["updates"], 6)

    def test_a_history_that_cannot_replay_is_refused(self) -> None:
        broken = _program_history(6)
        broken["updates"][2]["writes"] = ["cell_99"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "broken.json"
            source.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(FreezeError) as caught:
                freeze_world(
                    world_id="dgm-broken-test",
                    schema="program_state",
                    slice_rule="all",
                    catalog=root / "worlds",
                    source_file=source,
                    url="https://example.invalid/broken.json",
                )
            self.assertIn("cell_99", str(caught.exception))

    def test_a_missing_update_field_is_refused(self) -> None:
        broken = _program_history(6)
        del broken["updates"][0]["divergent"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "broken.json"
            source.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(FreezeError) as caught:
                freeze_world(
                    world_id="dgm-missing-test",
                    schema="program_state",
                    slice_rule="all",
                    catalog=root / "worlds",
                    source_file=source,
                    url="https://example.invalid/broken.json",
                )
            self.assertIn("divergent", str(caught.exception))


class LabeledTraceFreezeTests(unittest.TestCase):
    def test_live_swe_zip_freezes_and_slices_in_stable_member_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "live-swe.zip"
            source.write_bytes(_live_swe_archive())
            fixture = freeze_world(
                world_id="live-swe-test",
                schema="labeled_traces",
                slice_rule="first_traces:2",
                catalog=root / "worlds",
                source_file=source,
                url="https://example.invalid/live-swe.zip",
                domains=("swe", "agent", "execution"),
            )
            world = json.loads(
                (fixture.root / "world.json").read_text(encoding="utf-8")
            )
            origin = json.loads(
                (fixture.root / "origin.json").read_text(encoding="utf-8")
            )
            self.assertEqual(world["n"], 2)
            self.assertEqual(world["steps"], 2)
            self.assertEqual(
                [trace["id"] for trace in world["traces"]],
                ["model-a/repo__repo-1", "model-a/repo__repo-2"],
            )
            self.assertEqual(world["traces"][0]["label"], "resolved")
            self.assertEqual(world["traces"][1]["label"], "unresolved")
            self.assertEqual(world["traces"][0]["created_tools"][0]["name"], "helper.py")
            self.assertFalse(
                world["traces"][0]["derived"]["has_structured_predicted_state"]
            )
            self.assertEqual(origin["parent"]["n"], 3)
            self.assertEqual(origin["parent"]["format"], "live_swe_agent_release_zip")
            self.assertEqual(load_fixture(fixture.root).digest, fixture.digest)

    def test_trace_json_rejects_unlabeled_or_single_outcome_worlds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "traces.json"
            source.write_text(
                json.dumps(
                    {
                        "traces": [
                            {"id": "a", "label": "same", "steps": [{"t": 0}]},
                            {"id": "b", "label": "same", "steps": [{"t": 0}]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FreezeError):
                freeze_world(
                    world_id="bad-traces",
                    schema="labeled_traces",
                    slice_rule="all",
                    catalog=root / "worlds",
                    source_file=source,
                )

    def test_html_cannot_be_frozen_as_labeled_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "paper.html"
            source.write_text(
                "<html><body>trajectory results</body></html>", encoding="utf-8"
            )
            with self.assertRaises(FreezeError):
                freeze_world(
                    world_id="not-a-trace",
                    schema="labeled_traces",
                    slice_rule="all",
                    catalog=root / "worlds",
                    source_file=source,
                )


TINY_MEALY_DOT = (
    "digraph G {\n"
    'label=""\n'
    's0 [color="red"]\n'
    "s1\n"
    "s2\n"
    "__start0 [shape=none]\n"
    "__start0 -> s0;\n"
    's0 [label="s0"];\n'
    's0 -> s1[label="SYN / SYN+ACK"]\n'
    's0 -> s0[label="RST / TIMEOUT"]\n'
    's1 -> s2[label="ACK / TIMEOUT"]\n'
    's2 -> s0[label="RST / TIMEOUT"]\n'
    "}\n"
)


class SymbolicTraceFreezeTests(unittest.TestCase):
    def test_dot_machine_parses_states_transitions_and_start(self) -> None:
        states, transitions, start = parse_dot_machine(TINY_MEALY_DOT)
        self.assertEqual([s["id"] for s in states], ["s0", "s1", "s2"])
        self.assertEqual(start, "s0")
        self.assertEqual(len(transitions), 4)
        self.assertEqual(
            transitions[0],
            {"src": "s0", "dst": "s1", "action": "SYN", "output": "SYN+ACK"},
        )

    def test_trace_freeze_slices_by_first_states_and_binds_formula_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "machine.dot"
            source.write_text(TINY_MEALY_DOT, encoding="utf-8")
            fixture = freeze_world(
                world_id="tiny-trace",
                schema="symbolic_trace",
                slice_rule="first_states:2",
                catalog=root / "worlds",
                url="https://example.invalid/machine.dot",
                source_file=source,
                title="tiny protocol machine",
            )
            self.assertEqual(fixture.schema, "symbolic_trace")
            world = json.loads((fixture.root / "world.json").read_text(encoding="utf-8"))
            self.assertEqual([s["id"] for s in world["states"]], ["s0", "s1"])
            # only transitions among the kept states survive the slice
            self.assertEqual(world["m"], 2)
            self.assertEqual(world["initial"], "s0")
            self.assertIn("deterministic", world["invariant"])
            origin = json.loads((fixture.root / "origin.json").read_text(encoding="utf-8"))
            self.assertEqual(origin["parent"], {
                "n": 3, "m": 4, "initial": "s0", "format": "dot_labeled_digraph",
            })
            req = infer_requirement("formal verification of protocol state machines")
            self.assertIsNotNone(req)
            self.assertEqual(req.schema, "symbolic_trace")
            bound = match_world(req, {fixture.id: fixture})
            self.assertIsNotNone(bound)
            self.assertEqual(bound.id, "tiny-trace")

    def test_wrong_slice_kind_for_trace_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "machine.dot"
            source.write_text(TINY_MEALY_DOT, encoding="utf-8")
            with self.assertRaises(FreezeError):
                freeze_world(
                    world_id="bad-slice",
                    schema="symbolic_trace",
                    slice_rule="first_edges:2",
                    catalog=root / "worlds",
                    source_file=source,
                )

    def test_pending_resolve_skips_recipes_without_a_freezable_source(self) -> None:
        from farfield.extras.world import record_world_wishlist

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record_world_wishlist(
                root,
                [
                    {
                        "object_type": "io",
                        "schema": "",
                        "named_instance": "a buffer-pool B-tree",
                    },
                    {
                        "object_type": "formula",
                        "schema": "symbolic_trace",
                        "freeze_schema": "symbolic_trace",
                    },
                ],
            )
            self.assertEqual(resolve_pending_worlds(root), [])


if __name__ == "__main__":
    unittest.main()
