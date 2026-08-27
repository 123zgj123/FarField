"""Attested experimental worlds: frozen files, not invented datasets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.world import (
    KIND_SYNTHETIC,
    bind_world,
    infer_requirement,
    in_mission_constructable,
    lineage_conflicts,
    load_catalog,
    load_fixture,
    match_world,
    pick_world,
    reads_world_data,
    record_world_wishlist,
    resolve_world,
    wishlist_path,
    world_attested,
)

ROOT = Path(__file__).resolve().parents[1]


class KindTests(unittest.TestCase):
    def test_missing_kind_is_synthetic(self) -> None:
        self.assertFalse(world_attested({}))
        self.assertFalse(world_attested({"probe_kind": KIND_SYNTHETIC}))
        self.assertTrue(world_attested({"probe_kind": "WORLD"}))

    def test_a_script_must_name_data_to_count_as_reading_the_world(self) -> None:
        self.assertTrue(reads_world_data("Path('data/path.json').read_text()"))
        self.assertFalse(reads_world_data("nodes = list(range(8))"))
        fixture = load_catalog(ROOT)["a2a-task-lifecycle"]
        self.assertTrue(reads_world_data("Path('data/world.json').read_text()", fixture))
        self.assertFalse(
            reads_world_data(
                "json.loads((Path('data') / 'seed.json').read_text())", fixture
            )
        )


class CatalogTests(unittest.TestCase):
    def test_the_shipped_path_trace_loads_and_binds(self) -> None:
        catalog = load_catalog(ROOT)
        self.assertIn("path-trace", catalog)
        fixture = catalog["path-trace"]
        self.assertIn("path.json", fixture.files)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            data = bind_world(dest, fixture)
            payload = json.loads((data / "path.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["nodes"][-1], 7)
            meta = json.loads((dest / "world.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["id"], "path-trace")
            self.assertEqual(meta["digest"], fixture.digest)

    def test_a_tampered_digest_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "broken"
            folder.mkdir()
            (folder / "x.json").write_text("{}", encoding="utf-8")
            (folder / "manifest.json").write_text(
                json.dumps(
                    {
                        "id": "broken",
                        "files": ["x.json"],
                        "digest": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_fixture(folder)

    def test_unknown_world_names_are_errors_and_none_is_synthetic(self) -> None:
        catalog = load_catalog(ROOT)
        self.assertIsNone(resolve_world("none", catalog))
        self.assertIsNone(resolve_world("", catalog))
        with self.assertRaises(KeyError):
            resolve_world("no-such-world", catalog)

    def test_auto_picks_a_real_world_and_skips_the_test_path(self) -> None:
        catalog = load_catalog(ROOT)
        self.assertIn("zachary-karate", catalog)
        self.assertIn("phix174", catalog)
        self.assertIn("gutenberg-alice", catalog)
        self.assertIn("fisher-iris", catalog)
        self.assertIn("lusseau-dolphins", catalog)
        self.assertIn("snap-ca-grqc", catalog)
        self.assertIn("phage-lambda", catalog)
        self.assertIn("gutenberg-pride", catalog)
        self.assertIn("uci-letters", catalog)
        self.assertEqual(catalog["path-trace"].role, "test")
        self.assertEqual(catalog["phix174"].schema, "fasta")
        self.assertIn("phix174.fasta", catalog["phix174"].load_hint)
        genomic = pick_world(
            "compress genomic sequence collections with succinct data structures",
            catalog,
        )
        self.assertIsNotNone(genomic)
        self.assertEqual(genomic.id, "phage-lambda")
        graph = pick_world("dynamic graph connectivity under adversarial edge updates", catalog)
        self.assertIsNotNone(graph)
        self.assertEqual(graph.id, "snap-ca-grqc")
        self.assertEqual(
            pick_world(
                "learned index structures for high-dimensional similarity search",
                catalog,
            ).id,
            "uci-letters",
        )
        self.assertEqual(
            pick_world(
                "space-efficient sketches for heavy hitters in adversarial streams",
                catalog,
            ).id,
            "gutenberg-pride",
        )
        auto = resolve_world(
            "auto", catalog, topic="cache text search on a public prose dictionary"
        )
        self.assertIsNotNone(auto)
        self.assertEqual(auto.id, "gutenberg-pride")
        self.assertNotEqual(auto.id, "path-trace")
        # Formula claims bind the frozen learned automaton — the schema
        # gate still holds: only a symbolic_trace fixture is eligible.
        formula = pick_world("formal verification of smt solver encodings", catalog)
        self.assertIsNotNone(formula)
        self.assertEqual(formula.schema, "symbolic_trace")
        self.assertEqual(formula.id, "tcp-linux-server")
        # A bare ML topic must not bind the automaton via title prose.
        self.assertIsNone(pick_world("language model training dynamics", catalog))
        self.assertIsNone(pick_world("external memory algorithms for graphs", catalog))
        self.assertIsNone(pick_world("a generic data structure idea", catalog))
        # Experiment verbs (hash, sketch, cache) are not instance identity.
        # An agent-tool claim must not bind a novel token stream.
        self.assertIsNone(
            pick_world("caching hash sketches of agent tool traces", catalog)
        )
        from farfield.extras.world import WorldRequirement, requirement_compatible

        self.assertFalse(
            requirement_compatible(
                WorldRequirement(
                    object_type="graph",
                    schema="undirected_graph",
                    operations=("directed",),
                ),
                catalog["zachary-karate"],
            )
        )
        self.assertTrue(
            requirement_compatible(
                WorldRequirement(
                    object_type="graph",
                    schema="undirected_graph",
                    operations=("edges",),
                ),
                catalog["zachary-karate"],
            )
        )
        # Schema match is not scientific match: a token-level distillation
        # claim must not bind Pride, and an A2A card claim must not bind TCP.
        self.assertIsNone(
            pick_world(
                "agent trajectory distillation data pipeline for tool-use SFT",
                catalog,
            )
        )
        distill = infer_requirement(
            "agent trajectory distillation data pipeline for tool-use",
            "token-level loss masks on teacher tool-call trajectories",
        )
        self.assertIsNone(match_world(distill, catalog))
        a2a_topic = (
            "A2A Agent2Agent protocol security: Agent Card authentication "
            "and JSON-RPC AUTH_REQUIRED"
        )
        a2a = pick_world(a2a_topic, catalog)
        self.assertIsNotNone(a2a)
        self.assertEqual(a2a.id, "a2a-task-lifecycle")
        card_req = infer_requirement(
            a2a_topic,
            "an initial Agent Card that omits AUTH_REQUIRED never inserts the state",
        )
        bound = match_world(card_req, catalog)
        self.assertIsNotNone(bound)
        self.assertEqual(bound.id, "a2a-task-lifecycle")
        # A formula claim that also mentions a genomic red herring still
        # binds the attested automaton — foreign-object refuse is
        # same-schema only; fasta tokens must not unbind symbolic_trace.
        mixed = infer_requirement(
            "formal verification of smt solver encodings",
            "an SMT solver that encodes Ackermann will never finish; "
            "combining a succinct index will compress a genomic sequence",
        )
        self.assertEqual(mixed.schema, "symbolic_trace")
        self.assertEqual(match_world(mixed, catalog).id, "tcp-linux-server")
        pride = catalog["gutenberg-pride"]
        self.assertEqual(
            set(pride.domains), {"prose", "english", "narrative"}
        )
        self.assertFalse({"sketch", "hash", "cache"} & set(pride.domains))
        self.assertTrue(
            lineage_conflicts(
                pride,
                named_instance="agent tool traces from the retrieved papers",
                lineage_schema="text_stream",
            )
        )
        self.assertFalse(
            lineage_conflicts(
                pride,
                named_instance="a public prose token stream",
                lineage_schema="text_stream",
            )
        )
        a2a = catalog["a2a-task-lifecycle"]
        self.assertFalse(
            lineage_conflicts(
                a2a,
                named_instance="attested tool-call traces with authorization preconditions",
                lineage_schema="labeled_traces",
            )
        )
        self.assertTrue(
            in_mission_constructable(
                WorldRequirement(object_type="formula", schema="symbolic_trace")
            )
        )
        self.assertFalse(
            in_mission_constructable(WorldRequirement(object_type="io"))
        )
        self.assertTrue(
            in_mission_constructable(
                WorldRequirement(object_type="graph", schema="undirected_graph")
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            bind_world(dest, catalog["zachary-karate"])
            payload = json.loads((dest / "data" / "graph.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["m"], 78)
            self.assertEqual(len(payload["edges"]), 78)
            meta = json.loads((dest / "world.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["schema"], "undirected_graph")
            iris = bind_world(Path(tmp) / "iris", catalog["fisher-iris"])
            table = json.loads((iris / "iris.json").read_text(encoding="utf-8"))
            self.assertEqual(table["n"], 150)


class WishlistTests(unittest.TestCase):
    def test_unmatched_requirements_merge_into_the_wishlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = record_world_wishlist(
                root,
                [{"object_type": "io", "schema": "", "operations": ["disk-based"]}],
                topic="external memory b-trees",
                seen_at="2026-08-21",
            )
            self.assertEqual(first, wishlist_path(root))
            again = record_world_wishlist(
                root,
                [
                    {"object_type": "io", "schema": "", "operations": ["buffer-pool"]},
                    {"object_type": "table", "schema": "numeric_table"},
                ],
                topic="cache-oblivious search",
                seen_at="2026-08-22",
            )
            payload = json.loads(again.read_text(encoding="utf-8"))
            entries = {row["key"]: row for row in payload["entries"]}
            io_row = entries["io|"]
            self.assertEqual(io_row["misses"], 2)
            self.assertEqual(io_row["operations"], ["buffer-pool", "disk-based"])
            self.assertEqual(io_row["last_seen"], "2026-08-22")
            self.assertEqual(entries["table|numeric_table"]["misses"], 1)
            # empty batches never touch the file
            self.assertIsNone(record_world_wishlist(root, []))

    def test_wishlist_ignores_rows_without_a_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                record_world_wishlist(Path(tmp), [{"object_type": "", "schema": ""}])
            )


if __name__ == "__main__":
    unittest.main()
