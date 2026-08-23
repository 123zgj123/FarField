"""EvidenceID, dual ledgers, host confirmation teeth."""

from __future__ import annotations

import unittest

from farfield.extras.evidence import (
    LEDGER_DIAGNOSTIC,
    LEDGER_SCIENTIFIC,
    SCIENTIFIC_INVALID,
    SCIENTIFIC_SUPPORTS,
    TIER_SAME_INSTANCE,
    TIER_SCALE,
    TIER_SYNTHETIC,
    campaign_funnel,
    classify,
    evidence_id,
    heavy_confirmation_gaps,
    heavy_confirms,
    host_confirmation_gaps,
    host_confirms,
    hypothesis_revision_id,
    replication_group_id,
    replication_tier,
)
from farfield.extras.world import infer_requirement, match_world


def confirmable(**overrides):
    """A fully-provenanced host confirmation row: every field present, matching."""
    eid = "e" * 64
    base = {
        "verdict": "supports",
        "probe_kind": "WORLD",
        "host_ok": True,
        "protocol_complete": True,
        "evidence_id": eid,
        "host_evidence_id": eid,
        "experiment_digest": "a" * 64,
        "host_experiment_digest": "a" * 64,
        "world_digest": "b" * 64,
        "host_world_digest": "b" * 64,
        "data_digest": "b" * 64,
        "host_data_digest": "b" * 64,
        "host_replication": TIER_SAME_INSTANCE,
    }
    base.update(overrides)
    return base


class EvidenceIdTests(unittest.TestCase):
    def test_the_same_bytes_hash_the_same(self) -> None:
        left = evidence_id(
            claim_id="c1",
            experiment_digest="a" * 64,
            world_digest="b" * 64,
            data_digest="b" * 64,
        )
        right = evidence_id(
            claim_id="c1",
            experiment_digest="a" * 64,
            world_digest="b" * 64,
            data_digest="b" * 64,
        )
        self.assertEqual(left, right)

    def test_a_changed_experiment_is_a_new_object(self) -> None:
        left = evidence_id(claim_id="c1", experiment_digest="a")
        right = evidence_id(claim_id="c1", experiment_digest="b")
        self.assertNotEqual(left, right)


class LedgerTests(unittest.TestCase):
    def test_full_provenance_host_support_confirms(self) -> None:
        row = confirmable()
        labels = classify(row)
        self.assertEqual(labels["ledger"], LEDGER_SCIENTIFIC)
        self.assertEqual(labels["scientific"], SCIENTIFIC_SUPPORTS)
        self.assertTrue(host_confirms(row))
        self.assertEqual(host_confirmation_gaps(row), ())

    def test_missing_provenance_is_a_refusal_not_a_warning(self) -> None:
        # The old fail-open contract: host_ok + supports with no EvidenceID
        # and no digests still corroborated. That must never confirm again.
        row = {
            "verdict": "supports",
            "probe_kind": "WORLD",
            "host_ok": True,
            "protocol_complete": True,
        }
        self.assertFalse(host_confirms(row))
        gaps = host_confirmation_gaps(row)
        self.assertTrue(any("evidence_id missing" in gap for gap in gaps))
        self.assertTrue(any("digest missing" in gap for gap in gaps))

    def test_each_missing_field_blocks_confirmation(self) -> None:
        for key in (
            "evidence_id",
            "host_evidence_id",
            "experiment_digest",
            "host_experiment_digest",
            "world_digest",
            "host_world_digest",
            "data_digest",
            "host_data_digest",
        ):
            row = confirmable(**{key: ""})
            self.assertFalse(host_confirms(row), f"{key} missing must refuse")

    def test_protocol_complete_must_be_affirmatively_true(self) -> None:
        self.assertFalse(host_confirms(confirmable(protocol_complete=None)))
        self.assertFalse(host_confirms(confirmable(protocol_complete=False)))

    def test_a_mismatched_evidence_object_cannot_confirm(self) -> None:
        row = confirmable(host_data_digest="f" * 64, host_evidence_id="f" * 64)
        self.assertFalse(host_confirms(row))
        gaps = host_confirmation_gaps(row)
        self.assertTrue(any("mismatch" in gap for gap in gaps))

    def test_a_scale_replication_cannot_confirm_the_probe(self) -> None:
        row = confirmable(host_replication=TIER_SCALE)
        self.assertFalse(host_confirms(row))
        self.assertTrue(
            any("same-instance" in gap for gap in host_confirmation_gaps(row))
        )

    def test_missing_experiment_is_diagnostic_invalid(self) -> None:
        row = {
            "verdict": "supports",
            "probe_kind": "WORLD",
            "host_ok": False,
            "error": "missing experiment.py",
        }
        labels = classify(row)
        self.assertEqual(labels["ledger"], LEDGER_DIAGNOSTIC)
        self.assertEqual(labels["scientific"], SCIENTIFIC_INVALID)
        self.assertFalse(host_confirms(row))


class ReplicationTaxonomyTests(unittest.TestCase):
    def test_same_bytes_is_e1_and_parent_bytes_is_e2(self) -> None:
        self.assertEqual(replication_tier(confirmable()), TIER_SAME_INSTANCE)
        scale = confirmable(
            host_replication="", host_data_digest="f" * 64
        )
        self.assertEqual(replication_tier(scale), TIER_SCALE)

    def test_synthetic_probes_are_e0(self) -> None:
        self.assertEqual(
            replication_tier({"verdict": "supports"}), TIER_SYNTHETIC
        )

    def test_different_data_is_a_different_evidence_object(self) -> None:
        slice_run = evidence_id(
            claim_id="c1", experiment_digest="a", data_digest="slice"
        )
        parent_run = evidence_id(
            claim_id="c1", experiment_digest="a", data_digest="parent"
        )
        self.assertNotEqual(slice_run, parent_run)
        # ... but they share one replication group: same claim family.
        self.assertEqual(
            replication_group_id(
                claim_id="c1", experiment_digest="a", source_digest="s"
            ),
            replication_group_id(
                claim_id="c1", experiment_digest="a", source_digest="s"
            ),
        )


class RevisionIdentityTests(unittest.TestCase):
    def test_rewording_is_not_a_new_revision(self) -> None:
        self.assertEqual(
            hypothesis_revision_id("line1", "X improves Y via mechanism A."),
            hypothesis_revision_id("line1", "x improves y,  via mechanism a"),
        )

    def test_a_changed_mechanism_is_a_new_revision(self) -> None:
        self.assertNotEqual(
            hypothesis_revision_id("line1", "X improves Y via mechanism A"),
            hypothesis_revision_id("line1", "X improves Y via mechanism B"),
        )

    def test_revisions_are_scoped_to_their_research_line(self) -> None:
        self.assertNotEqual(
            hypothesis_revision_id("line1", "same claim"),
            hypothesis_revision_id("line2", "same claim"),
        )


class RequirementTests(unittest.TestCase):
    def test_smt_requires_a_symbolic_trace_fixture(self) -> None:
        req = infer_requirement("formal verification of smt solver encodings")
        self.assertIsNotNone(req)
        self.assertEqual(req.object_type, "formula")
        self.assertEqual(req.schema, "symbolic_trace")
        # No frozen trace in the catalog is still WORLD_INCOMPATIBLE —
        # a formula claim must never bind a substitute schema.
        self.assertIsNone(match_world(req, {}))

    def test_external_memory_stays_incompatible(self) -> None:
        req = infer_requirement("external memory b-tree with i/o complexity bounds")
        self.assertIsNotNone(req)
        self.assertEqual(req.object_type, "io")
        self.assertIsNone(match_world(req, {}))

    def test_graph_connectivity_is_a_graph_schema(self) -> None:
        req = infer_requirement("dynamic graph connectivity under edge updates")
        self.assertEqual(req.object_type, "graph")
        self.assertEqual(req.schema, "undirected_graph")

    def test_ablation_is_another_declared_lever_not_a_word_list(self) -> None:
        from farfield.extras.evidence import ablation_required

        levers = ("dropout", "hub_removal")
        # Prose alternative names no handle on this object.
        self.assertTrue(
            ablation_required(
                "the speedup comes from caching",
                "count k-mers on the attested genome",
                levers=levers,
                world_lever="dropout",
            )
        )
        # Named competing handle must appear in the experiment.
        self.assertTrue(
            ablation_required(
                "hub_removal would also shrink the giant component",
                "dropout edges only",
                levers=levers,
                world_lever="dropout",
            )
        )
        self.assertFalse(
            ablation_required(
                "hub_removal would also shrink the giant component",
                "dropout vs hub_removal on the same graph",
                levers=levers,
                world_lever="dropout",
            )
        )
        # No dynamics: nothing to ablate.
        self.assertFalse(
            ablation_required("caching", "any experiment", levers=(), world_lever="")
        )


def heavy_row(**overrides):
    """A complete heavy confirmation record: executor-owned replication of
    the registered script (same experiment digest as the corroborating
    evidence), interval clear, own EvidenceID, same world."""
    base = {
        "verdict": "supports",
        "replicated": True,
        "runner": "executor",
        "seeds": [11, 22, 33, 44, 55],
        "n": 5,
        "mean_separation": -0.4,
        "ci_low": -0.5,
        "ci_high": -0.3,
        "margin": 0.05,
        "expected_direction": "treatment_lower",
        "evidence_id": "h" * 64,
        "experiment_digest": "a" * 64,
        "world_digest": "b" * 64,
        "replication": TIER_SAME_INSTANCE,
    }
    base.update(overrides)
    return base


class HeavyConfirmationTests(unittest.TestCase):
    """The verified gate: strictly harder than the corroboration gate."""

    def test_a_complete_heavy_run_confirms(self) -> None:
        outcome = confirmable(heavy=heavy_row())
        self.assertEqual(heavy_confirmation_gaps(outcome), ())
        self.assertTrue(heavy_confirms(outcome))

    def test_no_heavy_run_is_the_first_gap(self) -> None:
        gaps = heavy_confirmation_gaps(confirmable())
        self.assertEqual(gaps, ("no heavy confirmation run recorded",))

    def test_a_bare_supports_flag_is_not_a_confirmation(self) -> None:
        # The old gate: `heavy_confirmed = (verdict == supports)`. A heavy
        # dict with the flag but no statistics must name every hole.
        gaps = heavy_confirmation_gaps(
            confirmable(heavy={"verdict": "supports"})
        )
        self.assertTrue(any("per-seed replicates" in gap for gap in gaps))
        self.assertTrue(any("EvidenceID" in gap for gap in gaps))
        self.assertTrue(any("world digest" in gap for gap in gaps))

    def test_the_interval_is_rechecked_not_trusted(self) -> None:
        # A recorded `supports` whose own numbers straddle the margin is
        # refused here: the flag is an input, the arithmetic is the proof.
        gaps = heavy_confirmation_gaps(
            confirmable(heavy=heavy_row(ci_high=-0.01))
        )
        self.assertTrue(any("does not clear" in gap for gap in gaps))

    def test_too_few_replicates_cannot_confirm(self) -> None:
        gaps = heavy_confirmation_gaps(confirmable(heavy=heavy_row(n=2)))
        self.assertTrue(any("at least 3" in gap for gap in gaps))

    def test_a_heavy_run_on_another_world_cannot_confirm(self) -> None:
        gaps = heavy_confirmation_gaps(
            confirmable(heavy=heavy_row(world_digest="f" * 64))
        )
        self.assertTrue(any("different world" in gap for gap in gaps))

    def test_a_heavy_weakens_or_uninformative_verdict_cannot_confirm(self) -> None:
        gaps = heavy_confirmation_gaps(
            confirmable(heavy=heavy_row(verdict="uninformative"))
        )
        self.assertTrue(any("not supports" in gap for gap in gaps))

    def test_self_reported_replication_is_not_replication(self) -> None:
        # The trust boundary: the model may generate computation, never
        # the evidence that audits it. Replicates without the executor's
        # signature are refused whatever their statistics say.
        gaps = heavy_confirmation_gaps(confirmable(heavy=heavy_row(runner="")))
        self.assertTrue(any("trusted executor" in gap for gap in gaps))

    def test_a_heavy_replay_of_a_different_script_cannot_confirm(self) -> None:
        # The confirmation must rerun the *registered* bytes: a different
        # experiment digest is a different protocol, not a confirmation.
        gaps = heavy_confirmation_gaps(
            confirmable(heavy=heavy_row(experiment_digest="c" * 64))
        )
        self.assertTrue(any("different experiment" in gap for gap in gaps))

    def test_an_invariant_experiment_cannot_confirm(self) -> None:
        # All seeds produced identical numbers: the script did not consume
        # the registered variation, so stability was not demonstrated.
        gaps = heavy_confirmation_gaps(
            confirmable(heavy=heavy_row(invariant=True))
        )
        self.assertTrue(any("invariant" in gap for gap in gaps))

    def test_repeated_seeds_cannot_confirm(self) -> None:
        gaps = heavy_confirmation_gaps(
            confirmable(heavy=heavy_row(seeds=[7, 7, 7, 7, 7]))
        )
        self.assertTrue(any("distinct registered seeds" in gap for gap in gaps))


class FunnelTests(unittest.TestCase):
    def test_funnel_separates_gate_kills_from_host_confirmation(self) -> None:
        report = campaign_funnel(
            [
                {"card_id": "a", "killed": True},
                {"card_id": "b", "killed": False},
                {"card_id": "c", "killed": False},
            ],
            [
                {
                    "card_id": "b",
                    "probe_ran": True,
                    "verdict": "supports",
                    "host_ok": True,
                },
                {
                    "card_id": "c",
                    "world_incompatible": True,
                },
            ],
            [{"card_id": "b", "status": "corroborated"}],
        )
        self.assertEqual(report["generated"], 3)
        self.assertEqual(report["text_pass"], 2)
        self.assertEqual(report["world_valid"], 1)
        self.assertEqual(report["probe_supports"], 1)
        self.assertEqual(report["corroborated"], 1)
        self.assertIsNotNone(report["rates"]["P(text_pass|generated)"])
