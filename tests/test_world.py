"""Attested experimental worlds: frozen files, not invented datasets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.world import (
    KIND_SYNTHETIC,
    bind_world,
    hint_hits,
    infer_requirement,
    in_mission_constructable,
    instance_tokens,
    lineage_conflicts,
    load_catalog,
    load_fixture,
    match_world,
    pick_world,
    reads_world_data,
    record_world_wishlist,
    resolve_world,
    wishlist_path,
    wishlist_requirement,
    wishlist_requirements,
    world_attested,
)
from farfield.extras.freeze import freeze_world

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
            ),
            None,
        )
        self.assertIsNone(
            pick_world(
                "space-efficient sketches for heavy hitters in adversarial streams",
                catalog,
            )
        )
        self.assertIsNone(
            pick_world(
                "preference post-training with token-level loss on resolved labels",
                catalog,
            )
        )
        self.assertIsNone(
            pick_world(
                "multi-agent bargaining games with seat-scoped legal actions",
                catalog,
            )
        )
        auto = resolve_world(
            "auto", catalog, topic="cache text search on a public prose dictionary"
        )
        self.assertIsNone(auto)
        picked = pick_world(
            "cache text search on a public prose dictionary", catalog
        )
        self.assertIsNotNone(picked)
        self.assertEqual(picked.id, "gutenberg-pride")
        self.assertNotEqual(picked.id, "path-trace")
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
        # An agent-tool claim must not bind a prose token stream; when an
        # attested labeled-trace fixture exists, it is the correct family.
        agent_trace = pick_world("caching hash sketches of agent tool traces", catalog)
        self.assertIsNotNone(agent_trace)
        self.assertEqual(agent_trace.schema, "labeled_traces")
        self.assertEqual(agent_trace.id, "live-swe-agent-verified-v1")
        daojo = pick_world(
            "Daojo Lab matrixgames decision traces prisoners dilemma "
            "seat-scoped observations payoffs Stag Hunt",
            catalog,
        )
        self.assertIsNotNone(daojo)
        self.assertEqual(daojo.id, "daojo-matrix-pd-v1")
        self.assertIsNone(
            pick_world(
                "Language-model data efficiency and decoding efficiency "
                "speculative decoding KV-cache early-exit",
                catalog,
            )
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
        # A trajectory topic and claim bind the attested trace family rather
        # than a prose stream. An A2A card claim likewise must not bind TCP.
        distill_topic_world = pick_world(
            "agent trajectory distillation data pipeline for tool-use SFT",
            catalog,
        )
        self.assertIsNotNone(distill_topic_world)
        self.assertEqual(distill_topic_world.schema, "labeled_traces")
        distill = infer_requirement(
            "agent trajectory distillation data pipeline for tool-use",
            "token-level loss masks on teacher tool-call trajectories",
        )
        self.assertEqual(
            match_world(distill, catalog).id, "live-swe-agent-verified-v1"
        )
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
        self.assertFalse(
            lineage_conflicts(
                a2a,
                named_instance=(
                    "The medication-review A2A lifecycle automaton in "
                    "Coupled Graph--Policy Distillation (arXiv:2608.09443v1)"
                ),
                lineage_schema="symbolic_trace",
            ),
            "a paper-title brand must not unbind a freeze that already shares A2A tags",
        )
        swe = catalog["live-swe-agent-verified-v1"]
        self.assertFalse(
            instance_tokens(swe) & {"agent", "tool", "software", "swe", "execution"}
        )
        self.assertTrue(
            lineage_conflicts(
                swe,
                named_instance="Pride and Prejudice agent-tool evaluation traces",
                lineage_schema="labeled_traces",
            ),
            "a category word like 'agent' must not exempt a foreign named instance",
        )
        self.assertTrue(
            lineage_conflicts(
                pride,
                named_instance="an agent-authored software tool log",
                lineage_schema="text_stream",
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

    def test_attested_swe_traces_bind_software_execution_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "traces.json"
            source.write_text(
                json.dumps(
                    {
                        "traces": [
                            {
                                "id": "m/task-1",
                                "label": "resolved",
                                "steps": [{"t": 0, "action": "pytest"}],
                            },
                            {
                                "id": "m/task-2",
                                "label": "unresolved",
                                "steps": [{"t": 0, "action": "pytest"}],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            fixture = freeze_world(
                world_id="swe-traces",
                schema="labeled_traces",
                slice_rule="all",
                catalog=root / "worlds",
                source_file=source,
                domains=("swe", "software", "agent", "execution", "trajectory"),
            )
            req = infer_requirement(
                "software engineering agents with execution trajectories",
                "select CWM patch trajectories by runtime outcome",
            )
            self.assertIsNotNone(req)
            self.assertEqual(req.schema, "labeled_traces")
            self.assertEqual(match_world(req, {fixture.id: fixture}).id, fixture.id)


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

    def test_a_generated_scientific_verdict_is_a_freeze_recipe(self) -> None:
        generated = wishlist_requirement(
            {
                "probe_kind": "GENERATED",
                "verdict": "supports",
                "world_schema": "text_stream",
                "world_requirement": {
                    "object_type": "text",
                    "schema": "text_stream",
                },
                "world_path": {
                    "freeze_url": "https://example.test/stream.txt",
                    "freeze_source": "public dump",
                },
            },
            topic="space-efficient sketches for heavy hitters",
        )
        self.assertIsNotNone(generated)
        self.assertTrue(generated["generated"])
        self.assertEqual(generated["schema"], "text_stream")
        self.assertEqual(generated["freeze_url"], "https://example.test/stream.txt")
        crash = wishlist_requirement(
            {
                "generated_world": True,
                "probe_kind": "GENERATED",
                "verdict": None,
                "world_schema": "text_stream",
            },
            topic="space-efficient sketches for heavy hitters",
        )
        self.assertIsNotNone(crash)
        self.assertTrue(crash["generated"])
        self.assertEqual(crash["schema"], "text_stream")
        synthetic = wishlist_requirement(
            {
                "probe_kind": "SYNTHETIC",
                "verdict": "supports",
                "world_schema": "text_stream",
            },
            topic="space-efficient sketches for heavy hitters",
        )
        self.assertIsNone(synthetic)
        incompatible = wishlist_requirement(
            {
                "world_incompatible": True,
                "world_requirement": {"object_type": "io", "schema": ""},
            }
        )
        self.assertIsNotNone(incompatible)
        self.assertEqual(incompatible["object_type"], "io")
        self.assertNotIn("generated", incompatible)
        merged = wishlist_requirements(
            [
                {
                    "world_incompatible": True,
                    "world_requirement": {
                        "object_type": "text",
                        "schema": "text_stream",
                    },
                },
                {
                    "probe_kind": "GENERATED",
                    "verdict": "uninformative",
                    "world_requirement": {
                        "object_type": "text",
                        "schema": "text_stream",
                        "freeze_url": "https://example.test/stream.txt",
                    },
                },
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0]["generated"])
        self.assertEqual(merged[0]["freeze_url"], "https://example.test/stream.txt")


class ProgramStateInferenceTests(unittest.TestCase):
    RSI_TOPIC = (
        "code world models of executable program state as the world of "
        "an agent harness, with recursive self-improvement as a lever "
        "on that same program state"
    )

    def test_hyphenated_hints_count_when_every_part_is_present(self) -> None:
        from farfield.extras.prior import content_tokens

        wanted = content_tokens("recursive self-improvement of an executable program")
        self.assertGreater(
            hint_hits(wanted, frozenset({"self-improvement", "executable"})),
            0,
        )
        self.assertEqual(
            hint_hits(content_tokens("self assembly of proteins"), frozenset({"self-improvement"})),
            0,
        )

    def test_an_rsi_topic_infers_program_state_not_a_graph(self) -> None:
        req = infer_requirement(self.RSI_TOPIC)
        self.assertIsNotNone(req)
        self.assertEqual(req.schema, "program_state")
        catalog = load_catalog(ROOT)
        # No shipped fixture (graph, prose, automaton, A2A) may bind an
        # RSI topic. An acquired program_state freeze — derived from the
        # Live-SWE-agent traces or harvested — is the one thing that may.
        from farfield.extras.world import ACQUIRED_PROVENANCE

        picked = pick_world(self.RSI_TOPIC, catalog)
        if picked is not None:
            self.assertEqual(picked.schema, "program_state")
            self.assertIn(picked.provenance, ACQUIRED_PROVENANCE)
        shipped = {k: v for k, v in catalog.items() if not v.provenance}
        self.assertIsNone(pick_world(self.RSI_TOPIC, shipped))

    def test_a_far_graph_noun_does_not_vote_the_schema(self) -> None:
        from farfield.extras.domain import without_far_terms

        claim = (
            "replay a self-approving update against frozen program state "
            "instead of deleting a feedback edge set"
        )
        stripped = without_far_terms(claim, "feedback edge set", self.RSI_TOPIC)
        req = infer_requirement(self.RSI_TOPIC, stripped)
        self.assertEqual(req.schema, "program_state")
        leaked = infer_requirement(self.RSI_TOPIC, claim)
        # Even with the far noun still in the claim, executable / program
        # outrank a single "edge" vote. The stripper is the insulation.
        self.assertEqual(leaked.schema, "program_state")


if __name__ == "__main__":
    unittest.main()
