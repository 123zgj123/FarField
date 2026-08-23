"""The research state: executable promotion, explicit unknowns, no contamination."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras import chain as chain_module
from farfield.extras.evidence import hypothesis_revision_id
from farfield.extras.state import (
    StateError,
    apply_outcomes,
    hypothesis_id,
    line_status,
    load,
    migrate_identities,
    next_status,
    probe_history,
    prompt_lines,
    record_rejections,
    rejections_for,
    summary,
)

CORPUS = "test-corpus"
ANCHOR = "succinct data structure"


def outcome(**overrides):
    base = {
        "card_id": "gen_x",
        "pair": ["succinct data structure", "pivot rule"],
        "claim": "a succinct structure stores pivot history in linear space",
        "prior_kills": False,
        "verdict": None,
        "alternative": "",
    }
    base.update(overrides)
    return base


def register_hypothesis(directory, pair, claim):
    """The act the promotion gate demands: this (line, revision) entered
    the research ledger before its evidence."""
    line_id = hypothesis_id(ANCHOR, pair)
    chain_module.append_event(
        Path(directory) / "state.chain.jsonl",
        chain_module.REGISTER_HYPOTHESIS,
        {
            "line_id": line_id,
            "revision_id": hypothesis_revision_id(line_id, claim),
        },
    )


def world_support(chain_dir=None, **overrides):
    # Full provenance: host confirmation is fail-closed, so a confirming
    # outcome must carry a matching EvidenceID and all three digest pairs.
    # Promotion additionally audits the event chain, so a `chain_dir`
    # writes the REGISTER→EXECUTE history this evidence claims to have,
    # registers the hypothesis in the research ledger, and — when a heavy
    # dict rides along — records the seed executions and the aggregation
    # the verified gate reverse-checks.
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
        "host_replication": "E1_same_instance",
    }
    base.update(overrides)
    merged = outcome(**base)
    if chain_dir is not None:
        register_hypothesis(chain_dir, merged["pair"], merged["claim"])
        heavy = merged.get("heavy") if isinstance(merged.get("heavy"), dict) else None
        merged["chain_path"] = str(
            write_candidate_chain(
                chain_dir,
                experiment_digest=merged["experiment_digest"],
                evidence_id=merged["host_evidence_id"],
                heavy=heavy,
            )
        )
    return merged


def write_candidate_chain(
    directory, *, experiment_digest, evidence_id, reverse=False, heavy=None
):
    """A candidate chain proving REGISTER before EXECUTE (or the tampered
    order, with `reverse=True`). When `heavy` is given, the chain also
    carries the executor's seed executions and the aggregation that names
    them — the acts the verified gate audits."""
    name = "candidate-chain-heavy.jsonl" if heavy else "candidate-chain.jsonl"
    path = Path(directory) / name
    events = [
        (chain_module.REGISTER_PROTOCOL, {"experiment_digest": experiment_digest}),
        (chain_module.EXECUTE_PROTOCOL, {"evidence_id": evidence_id}),
    ]
    if reverse:
        events.reverse()
    if not path.is_file():
        for kind, payload in events:
            chain_module.append_event(path, kind, payload)
        if heavy:
            parents = []
            for seed in heavy.get("seeds") or []:
                record = chain_module.append_event(
                    path,
                    chain_module.EXECUTE_HEAVY_SEED,
                    {
                        "seed": seed,
                        "experiment_digest": heavy.get(
                            "experiment_digest", experiment_digest
                        ),
                        "world_digest": heavy.get("world_digest", ""),
                    },
                )
                parents.append(record["digest"])
            chain_module.append_event(
                path,
                chain_module.AGGREGATE_HEAVY,
                {
                    "evidence_id": heavy.get("evidence_id", ""),
                    "experiment_digest": heavy.get(
                        "experiment_digest", experiment_digest
                    ),
                    "seeds": list(heavy.get("seeds") or []),
                    "parents": parents,
                },
            )
    return path


def heavy_confirmation(**overrides):
    # The verified gate's evidence object: executor-owned replication of
    # the registered script (same experiment digest as the corroborating
    # evidence) whose whole 95% interval clears the margin, on the same
    # world, under runtime-derived seeds.
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
        "replication": "E1_same_instance",
    }
    base.update(overrides)
    return base


class LadderTests(unittest.TestCase):
    def test_a_synthetic_support_does_not_corroborate(self) -> None:
        status, reason = next_status(None, outcome(verdict="supports"))
        self.assertEqual(status, "speculative")
        self.assertIn("SYNTHETIC", reason)

    def test_a_world_support_without_host_stays_speculative(self) -> None:
        status, reason = next_status(
            None, outcome(verdict="supports", probe_kind="WORLD")
        )
        self.assertEqual(status, "speculative")
        self.assertIn("pending host", reason)

    def test_a_host_confirmed_world_support_corroborates(self) -> None:
        status, reason = next_status(None, world_support())
        self.assertEqual(status, "corroborated")
        self.assertIn("host", reason)

    def test_verified_needs_two_world_supports_and_the_heavy_confirmation(self) -> None:
        status, reason = next_status(
            "corroborated",
            world_support(heavy_confirmed=True, heavy=heavy_confirmation()),
        )
        self.assertEqual(status, "verified")
        self.assertIn("confidence interval", reason)

    def test_a_heavy_flag_without_the_statistics_cannot_verify(self) -> None:
        # The rung above corroborated is strictly harder: a bare
        # `supports` boolean with no replicates, no interval, and no
        # EvidenceID is a refusal, not a promotion.
        status, reason = next_status(
            "corroborated",
            world_support(heavy_confirmed=True),
        )
        self.assertEqual(status, "corroborated")
        self.assertIn("fail-closed", reason)

    def test_a_heavy_interval_that_straddles_the_margin_cannot_verify(self) -> None:
        status, reason = next_status(
            "corroborated",
            world_support(
                heavy_confirmed=True,
                heavy=heavy_confirmation(ci_high=-0.01),
            ),
        )
        self.assertEqual(status, "corroborated")
        self.assertIn("fail-closed", reason)

    def test_a_heavy_run_on_a_different_world_cannot_verify(self) -> None:
        status, reason = next_status(
            "corroborated",
            world_support(
                heavy_confirmed=True,
                heavy=heavy_confirmation(world_digest="f" * 64),
            ),
        )
        self.assertEqual(status, "corroborated")
        self.assertIn("fail-closed", reason)

    def test_without_the_heavy_run_a_second_world_support_stays_corroborated(self) -> None:
        status, reason = next_status("corroborated", world_support())
        self.assertEqual(status, "corroborated")
        self.assertIn("awaits the heavier", reason)

    def test_a_synthetic_support_keeps_corroboration_only_with_a_proof(self) -> None:
        # The grounding proof is resolved from the research ledger by
        # apply_outcomes: the PROMOTE event that granted the previous
        # rank, carrying a host EvidenceID and a world digest.
        status, reason = next_status(
            "corroborated",
            outcome(
                verdict="supports",
                prev_grounding_proof={
                    "host_evidence_id": "e" * 64,
                    "world_digest": "b" * 64,
                },
            ),
        )
        self.assertEqual(status, "corroborated")
        self.assertIn("SYNTHETIC", reason)

    def test_a_synthetic_support_without_a_proof_drops_the_rank(self) -> None:
        # No ledger proof that the earlier corroboration was host-ground:
        # fail-closed, the rank is not kept. (An earlier revision of this
        # rule returned True unconditionally.)
        status, reason = next_status("corroborated", outcome(verdict="supports"))
        self.assertEqual(status, "speculative")
        self.assertIn("not host-confirmed", reason)

    def test_no_evidence_stays_speculative(self) -> None:
        status, reason = next_status(None, outcome())
        self.assertEqual(status, "speculative")
        self.assertIn("no discriminating evidence", reason)

    def test_an_uninformative_probe_does_not_erase_host_corroboration(self) -> None:
        status, _ = next_status("corroborated", outcome(verdict="uninformative"))
        self.assertEqual(status, "corroborated")

    def test_host_failure_invalidates_earlier_corroboration(self) -> None:
        status, reason = next_status(
            "corroborated",
            world_support(host_ok=False, host_error="missing experiment.py"),
        )
        self.assertEqual(status, "speculative")
        self.assertIn("invalidated", reason)

    def test_a_prior_hit_closes_the_claim_not_the_pair(self) -> None:
        status, reason = next_status(None, outcome(prior_kills=True))
        self.assertEqual(status, "closed_by_prior")
        self.assertIn("supporting span", reason)

    def test_a_host_run_without_provenance_cannot_corroborate(self) -> None:
        # The old fail-open path: host_ok + WORLD supports, but no
        # EvidenceID and no digests. Fail-closed means speculative.
        status, reason = next_status(
            None,
            outcome(
                verdict="supports",
                probe_kind="WORLD",
                host_ok=True,
                protocol_complete=True,
            ),
        )
        self.assertEqual(status, "speculative")
        self.assertIn("fail-closed", reason)

    def test_a_scale_replication_cannot_corroborate(self) -> None:
        # Probe on the slice, host on the parent: different bytes are a
        # different evidence object, not a same-instance reproduction.
        status, reason = next_status(
            None,
            world_support(
                host_data_digest="f" * 64,
                host_evidence_id="f" * 64,
                host_replication="E2_scale_replication",
            ),
        )
        self.assertEqual(status, "speculative")
        self.assertIn("fail-closed", reason)


class StoreTests(unittest.TestCase):
    def test_only_corroborated_claims_become_known_ground(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            apply_outcomes(path, CORPUS, ANCHOR, [
                world_support(chain_dir=tmp),
                outcome(
                    card_id="gen_y",
                    pair=["succinct data structure", "auction theory"],
                    claim="auctions price cache lines",
                    alternative="it is just load balancing",
                ),
            ])
            report = summary(load(path), CORPUS, ANCHOR)
            self.assertEqual(len(report["known"]), 1)
            self.assertIn("pivot history", report["known"][0]["claim"])
            self.assertEqual(len(report["unknowns"]), 1)
            self.assertIn("load balancing", report["unknowns"][0])
            lines = prompt_lines(report)
            self.assertTrue(any("Established" in line for line in lines))
            self.assertTrue(any("Open question" in line for line in lines))
            self.assertFalse(
                any("auctions price cache lines" in line and "Established" in line
                    for line in lines),
                "a speculative claim never rides as established ground",
            )

    def test_weakening_a_corroborated_claim_records_a_contradiction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            apply_outcomes(path, CORPUS, ANCHOR, [world_support(chain_dir=tmp)])
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome(verdict="weakens")])
            self.assertEqual(events[0]["previous"], "corroborated")
            self.assertEqual(events[0]["status"], "weakened")
            report = summary(load(path), CORPUS, ANCHOR)
            self.assertEqual(len(report["contradictions"]), 1)
            self.assertIn("weakened by a later probe", report["contradictions"][0])

    def test_identity_is_the_anchor_domain_and_the_far_concept(self) -> None:
        # A live batch watched "succinct data" vs "succinct data structure"
        # split one research line in two and break the promotion ladder.
        # The near side is phrasing; the far concept is the bet.
        self.assertEqual(
            hypothesis_id(ANCHOR, ["succinct data", "Pivot Rule"]),
            hypothesis_id(ANCHOR, ["succinct data structure", "pivot rule"]),
        )
        self.assertNotEqual(
            hypothesis_id(ANCHOR, ["succinct data", "pivot rule"]),
            hypothesis_id("another domain", ["succinct data", "pivot rule"]),
        )

    def test_near_side_synonyms_climb_the_same_ladder(self) -> None:
        # supports under one near-phrasing, then supports under another:
        # same line, so the second mission verifies.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            apply_outcomes(path, CORPUS, ANCHOR, [world_support(chain_dir=tmp)])
            events = apply_outcomes(path, CORPUS, ANCHOR, [
                world_support(
                    chain_dir=tmp,
                    pair=["succinct data", "pivot rule"],
                    heavy_confirmed=True,
                    heavy=heavy_confirmation(),
                )
            ])
            self.assertEqual(events[0]["previous"], "corroborated")
            self.assertEqual(events[0]["status"], "verified")
            self.assertEqual(
                line_status(load(path), CORPUS, ANCHOR,
                            ["any phrasing at all", "pivot rule"]),
                "verified",
            )

    def test_migration_merges_lines_split_by_old_identity(self) -> None:
        # Rebuild the live accident: two entries for the same (anchor, far)
        # line under different old keys; migration must fold them into one,
        # keeping the higher-ranked survivor and the union of evidence.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            apply_outcomes(path, CORPUS, ANCHOR, [world_support(chain_dir=tmp)])
            topics = load(path)
            bucket = topics[f"{CORPUS}::{ANCHOR}"]
            survivor = next(iter(bucket["hypotheses"].values()))
            stray = dict(survivor)
            stray["pair"] = ["succinct data", "pivot rule"]
            stray["status"] = "speculative"
            stray["missions"] = 2
            stray["evidence"] = [{"kind": "probe", "direction": "0", "note": "old"}]
            bucket["hypotheses"]["deadbeef00000000"] = stray
            import farfield.extras.state as state_module
            state_module._save(path, topics)
            merged = migrate_identities(path)
            self.assertEqual(merged, 1)
            merged_topics = load(path)
            merged_bucket = merged_topics[f"{CORPUS}::{ANCHOR}"]
            self.assertEqual(len(merged_bucket["hypotheses"]), 1)
            entry = next(iter(merged_bucket["hypotheses"].values()))
            self.assertEqual(entry["status"], "corroborated")
            self.assertEqual(entry["missions"], 3)
            self.assertEqual(len(entry["evidence"]), 2)
            self.assertEqual(migrate_identities(path), 0, "idempotent")

    def test_the_last_probe_design_is_kept_for_the_next_diagnosis(self) -> None:
        # An uninformative design must be retrievable by the hypothesis's
        # next mission, so the follow-up experiment can sharpen it.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            apply_outcomes(path, CORPUS, ANCHOR, [
                outcome(
                    verdict="uninformative",
                    experiment="count comparisons on ten random instances",
                )
            ])
            history = probe_history(
                load(path), CORPUS, ANCHOR,
                ["succinct data structure", "pivot rule"],
            )
            self.assertEqual(history["verdict"], "uninformative")
            self.assertIn("ten random instances", history["experiment"])
            apply_outcomes(path, CORPUS, ANCHOR, [
                outcome(
                    verdict="uninformative",
                    experiment="count comparisons on a larger instance",
                )
            ])
            history = probe_history(
                load(path), CORPUS, ANCHOR,
                ["succinct data structure", "pivot rule"],
            )
            self.assertTrue(history["must_switch_mechanism"])
            self.assertIsNone(
                probe_history(load(path), CORPUS, ANCHOR, ["never", "seen"])
            )

    def test_a_supported_but_low_value_claim_banks_an_open_question(self) -> None:
        # The probe held but the reviewer scored every dimension low: the
        # tension is recorded as a named target for the next rerun, not
        # left as a silent "all supports, zero leftovers" success.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            apply_outcomes(path, CORPUS, ANCHOR, [
                outcome(
                    verdict="supports",
                    value_vector={
                        "novelty": 2, "importance": 1,
                        "information_gain": 2, "transfer_potential": 1,
                    },
                )
            ])
            report = summary(load(path), CORPUS, ANCHOR)
            self.assertTrue(
                any("advantage over known methods is unproven" in u
                    for u in report["unknowns"])
            )

    def test_a_supported_high_value_claim_leaves_no_such_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            apply_outcomes(path, CORPUS, ANCHOR, [
                outcome(
                    verdict="supports",
                    value_vector={
                        "novelty": 4, "importance": 3,
                        "information_gain": 4, "transfer_potential": 2,
                    },
                )
            ])
            report = summary(load(path), CORPUS, ANCHOR)
            self.assertFalse(report["unknowns"])

    def test_no_outcomes_leave_the_store_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            self.assertEqual(apply_outcomes(path, CORPUS, ANCHOR, []), [])
            self.assertFalse(path.exists())

    def test_gate_kills_become_rejection_capsules_not_hypotheses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            added = record_rejections(path, CORPUS, ANCHOR, [
                {"pair": ["alpha", "beta"], "killed_by": ["already_combined"]},
                {"pair": ["alpha", "beta"], "killed_by": ["already_combined"]},
            ])
            self.assertEqual(added, 1, "the same dead pair is one lesson")
            capsules = rejections_for(load(path), CORPUS, ANCHOR)
            self.assertEqual(capsules[0]["context"], "alpha x beta")
            self.assertIn("already_combined", capsules[0]["mechanism"])
            report = summary(load(path), CORPUS, ANCHOR)
            self.assertEqual(report["hypotheses"], 0,
                             "a rejected card never entered research")

    def test_rejections_are_capped_at_the_freshest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            record_rejections(path, CORPUS, ANCHOR, [
                {"pair": [f"c{i}", "x"], "killed_by": ["hub"]} for i in range(20)
            ])
            capsules = rejections_for(load(path), CORPUS, ANCHOR)
            self.assertEqual(len(capsules), 12)
            self.assertEqual(capsules[-1]["context"], "c19 x x")

    def test_a_claim_revision_does_not_inherit_truth_status(self) -> None:
        # Claim v1 is host-corroborated. Claim v2 keeps the pair but names
        # a different mechanism: same research line, new revision — the
        # ladder restarts and the old evidence is superseded context.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            apply_outcomes(path, CORPUS, ANCHOR, [world_support(chain_dir=tmp)])
            events = apply_outcomes(path, CORPUS, ANCHOR, [
                outcome(
                    claim="pivot history compresses because of cache blocking",
                    verdict="supports",
                )
            ])
            self.assertTrue(events[0]["revision_reset"])
            self.assertIsNone(events[0]["previous"])
            self.assertEqual(events[0]["status"], "speculative")
            self.assertIn("not inherited truth status", events[0]["reason"])
            report = summary(load(path), CORPUS, ANCHOR)
            self.assertEqual(
                report["known"], [],
                "the superseded corroboration must not ride as ground",
            )
            bucket = load(path)[f"{CORPUS}::{ANCHOR}"]
            entry = next(iter(bucket["hypotheses"].values()))
            self.assertTrue(
                all(item["validity"] == "superseded"
                    for item in entry["evidence"][:-1]),
            )

    def test_the_same_claim_keeps_climbing_its_own_ladder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            apply_outcomes(path, CORPUS, ANCHOR, [world_support(chain_dir=tmp)])
            events = apply_outcomes(
                path, CORPUS, ANCHOR, [world_support(chain_dir=tmp)]
            )
            self.assertFalse(events[0]["revision_reset"])
            self.assertEqual(events[0]["previous"], "corroborated")
            self.assertEqual(events[0]["status"], "corroborated")

    def test_promotions_land_in_a_digest_chained_log(self) -> None:
        from farfield.extras.chain import verify_chain

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            apply_outcomes(path, CORPUS, ANCHOR, [world_support(chain_dir=tmp)])
            apply_outcomes(path, CORPUS, ANCHOR, [outcome(verdict="weakens")])
            chain_path = Path(tmp) / "state.chain.jsonl"
            self.assertTrue(chain_path.is_file())
            report = verify_chain(chain_path)
            self.assertTrue(report["ok"])
            # One hypothesis registration and two promotions.
            self.assertEqual(report["length"], 3)
            # Tampering with a recorded promotion breaks the chain.
            lines = chain_path.read_text().splitlines()
            target = next(
                index for index, line in enumerate(lines) if "corroborated" in line
            )
            lines[target] = lines[target].replace("corroborated", "verified")
            chain_path.write_text("\n".join(lines) + "\n")
            self.assertFalse(verify_chain(chain_path)["ok"])

    def test_a_confirmation_without_a_chain_cannot_promote(self) -> None:
        # The host run may be real, but unprovable order is not
        # provenance: no recorded chain, no corroboration.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            events = apply_outcomes(path, CORPUS, ANCHOR, [world_support()])
            self.assertEqual(events[0]["status"], "speculative")
            self.assertIn("event chain refused", events[0]["reason"])
            self.assertIn("no event chain recorded", events[0]["reason"])

    def test_a_protocol_registered_after_the_run_cannot_promote(self) -> None:
        # EXECUTE before REGISTER is the tell of a protocol rewritten to
        # fit its own result; the order proof must refuse it.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            support = world_support()
            support["chain_path"] = str(
                write_candidate_chain(
                    tmp,
                    experiment_digest=support["experiment_digest"],
                    evidence_id=support["host_evidence_id"],
                    reverse=True,
                )
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [support])
            self.assertEqual(events[0]["status"], "speculative")
            self.assertIn("registered after the execution", events[0]["reason"])

    def test_a_broken_candidate_chain_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            support = world_support(chain_dir=tmp)
            chain_file = Path(support["chain_path"])
            lines = chain_file.read_text().splitlines()
            lines[0] = lines[0].replace(
                support["experiment_digest"], "d" * 64
            )
            chain_file.write_text("\n".join(lines) + "\n")
            events = apply_outcomes(path, CORPUS, ANCHOR, [support])
            self.assertEqual(events[0]["status"], "speculative")
            self.assertIn("event chain broken", events[0]["reason"])

    def test_a_tampered_promotion_log_refuses_the_whole_write(self) -> None:
        # The store's own chain is audited before anything is written:
        # editing a recorded promotion does not just break the diary, it
        # freezes the store until the tamper is explained.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            apply_outcomes(path, CORPUS, ANCHOR, [world_support(chain_dir=tmp)])
            chain_path = Path(tmp) / "state.chain.jsonl"
            lines = chain_path.read_text().splitlines()
            target = next(
                index for index, line in enumerate(lines) if "corroborated" in line
            )
            lines[target] = lines[target].replace("corroborated", "verified")
            chain_path.write_text("\n".join(lines) + "\n")
            with self.assertRaises(StateError):
                apply_outcomes(
                    path, CORPUS, ANCHOR, [outcome(verdict="weakens")]
                )

    def test_a_hand_edited_store_refuses_to_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            apply_outcomes(path, CORPUS, ANCHOR, [outcome(verdict="supports")])
            payload = json.loads(path.read_text())
            payload["topics"][f"{CORPUS}::{ANCHOR}"]["hypotheses"] = {}
            path.write_text(json.dumps(payload))
            with self.assertRaises(StateError):
                load(path)


if __name__ == "__main__":
    unittest.main()
