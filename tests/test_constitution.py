"""The epistemic constitution, as an executable test matrix.

One principle governs every gate: the model may generate computation —
hypotheses, protocols, programs — but every fact that can change the
epistemic state (execution identity, raw measurements, replicates,
digests, ordering, promotion proof) must be produced by the trusted
runtime outside the model.

Each test here is a clause. The happy paths prove a legitimate discovery
can actually climb the ladder; every negative path proves one named way
of cheating (or one named accident) is refused with the reason spelled
out. This file is the canonical regression matrix: when a refactor
breaks a clause, the refactor is wrong until it argues otherwise.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras import chain
from farfield.extras.diagnose import diagnosis_from_payload
from farfield.extras.evidence import hypothesis_revision_id
from farfield.extras.mission import _heavy_confirmation
from farfield.extras.probeexp import replication_seeds, resolve_tier, spec_from_payload
from farfield.extras.state import (
    StateError,
    apply_outcomes,
    hypothesis_id,
    load,
    migrate_identities,
    summary,
)

CORPUS = "constitution"
ANCHOR = "succinct data structure"
PAIR = ["succinct data structure", "pivot rule"]
CLAIM = "a succinct structure stores pivot history in linear space"
EXPERIMENT_DIGEST = "a" * 64
WORLD_DIGEST = "b" * 64
EVIDENCE_ID = "e" * 64
HEAVY_EVIDENCE_ID = "h" * 64
HEAVY_SEEDS = [11, 22, 33, 44, 55]


# --- trusted-runtime fixtures ------------------------------------------------


def register_hypothesis(tmp: str, claim: str = CLAIM, pair=None) -> None:
    line_id = hypothesis_id(ANCHOR, pair or PAIR)
    chain.append_event(
        Path(tmp) / "state.chain.jsonl",
        chain.REGISTER_HYPOTHESIS,
        {"line_id": line_id, "revision_id": hypothesis_revision_id(line_id, claim)},
    )


def candidate_chain(
    tmp: str,
    *,
    reverse: bool = False,
    heavy: bool = False,
    name: str = "candidate.jsonl",
) -> Path:
    """The runtime-recorded history a promotion must be able to show."""
    path = Path(tmp) / name
    events = [
        (chain.REGISTER_PROTOCOL, {"experiment_digest": EXPERIMENT_DIGEST}),
        (chain.EXECUTE_PROTOCOL, {"evidence_id": EVIDENCE_ID}),
    ]
    if reverse:
        events.reverse()
    for kind, payload in events:
        chain.append_event(path, kind, payload)
    if heavy:
        parents = []
        for seed in HEAVY_SEEDS:
            record = chain.append_event(
                path,
                chain.EXECUTE_HEAVY_SEED,
                {
                    "seed": seed,
                    "experiment_digest": EXPERIMENT_DIGEST,
                    "world_digest": WORLD_DIGEST,
                },
            )
            parents.append(record["digest"])
        chain.append_event(
            path,
            chain.AGGREGATE_HEAVY,
            {
                "evidence_id": HEAVY_EVIDENCE_ID,
                "experiment_digest": EXPERIMENT_DIGEST,
                "seeds": HEAVY_SEEDS,
                "parents": parents,
            },
        )
    return path


def confirmed_outcome(**overrides):
    """A fully provenanced E1 host confirmation, as the runtime records it."""
    base = {
        "card_id": "gen_x",
        "pair": list(PAIR),
        "claim": CLAIM,
        "prior_kills": False,
        "alternative": "",
        "verdict": "supports",
        "probe_kind": "WORLD",
        "host_ok": True,
        "protocol_complete": True,
        "evidence_id": EVIDENCE_ID,
        "host_evidence_id": EVIDENCE_ID,
        "experiment_digest": EXPERIMENT_DIGEST,
        "host_experiment_digest": EXPERIMENT_DIGEST,
        "world_digest": WORLD_DIGEST,
        "host_world_digest": WORLD_DIGEST,
        "data_digest": WORLD_DIGEST,
        "host_data_digest": WORLD_DIGEST,
        "host_replication": "E1_same_instance",
    }
    base.update(overrides)
    return base


def heavy_dict(**overrides):
    """The executor's aggregate, as `_heavy_confirmation` emits it."""
    base = {
        "verdict": "supports",
        "replicated": True,
        "runner": "executor",
        "seeds": list(HEAVY_SEEDS),
        "n": 5,
        "mean_separation": -0.4,
        "ci_low": -0.5,
        "ci_high": -0.3,
        "margin": 0.05,
        "expected_direction": "treatment_lower",
        "evidence_id": HEAVY_EVIDENCE_ID,
        "experiment_digest": EXPERIMENT_DIGEST,
        "world_digest": WORLD_DIGEST,
        "replication": "E1_same_instance",
    }
    base.update(overrides)
    return base


def corroborate(tmp: str) -> Path:
    """Mission one: a clean corroboration, returning the state path."""
    path = Path(tmp) / "state.json"
    register_hypothesis(tmp)
    outcome = confirmed_outcome(chain_path=str(candidate_chain(tmp)))
    events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
    assert events[0]["status"] == "corroborated", events[0]
    return path


# --- happy paths: the ladder must actually be climbable ----------------------


class HappyPathTests(unittest.TestCase):
    def test_a_clean_world_confirmation_corroborates(self) -> None:
        # WORLD probe supports → protocol registered → E1 exact host
        # reproduction → identities match → chain audits pass → corroborated.
        with tempfile.TemporaryDirectory() as tmp:
            path = corroborate(tmp)
            report = summary(load(path), CORPUS, ANCHOR)
            self.assertEqual(len(report["known"]), 1)

    def test_a_ledgered_executor_confirmation_verifies(self) -> None:
        # corroborated → executor-owned heavy (distinct derived seeds, CI
        # entirely beyond the margin, same world, same registered script)
        # → seed executions and the aggregate in the ledger → verified.
        with tempfile.TemporaryDirectory() as tmp:
            path = corroborate(tmp)
            outcome = confirmed_outcome(
                heavy_confirmed=True,
                heavy=heavy_dict(),
                chain_path=str(candidate_chain(tmp, heavy=True, name="c2.jsonl")),
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
            self.assertEqual(events[0]["status"], "verified")


# --- the trust boundary: evidence metadata must be runtime-made --------------


class TrustBoundaryTests(unittest.TestCase):
    def test_self_reported_replicates_cannot_verify(self) -> None:
        # The audited object may not supply its own audit evidence: a heavy
        # dict without the executor's signature stays corroborated.
        with tempfile.TemporaryDirectory() as tmp:
            path = corroborate(tmp)
            outcome = confirmed_outcome(
                heavy_confirmed=True,
                heavy=heavy_dict(runner=""),
                chain_path=str(candidate_chain(tmp, heavy=True, name="c2.jsonl")),
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
            self.assertEqual(events[0]["status"], "corroborated")
            self.assertIn("fail-closed", events[0]["reason"])

    def test_a_heavy_that_replayed_different_bytes_cannot_verify(self) -> None:
        # Protocol drift: the confirmation must rerun the registered script.
        with tempfile.TemporaryDirectory() as tmp:
            path = corroborate(tmp)
            outcome = confirmed_outcome(
                heavy_confirmed=True,
                heavy=heavy_dict(experiment_digest="c" * 64),
                chain_path=str(candidate_chain(tmp, heavy=True, name="c2.jsonl")),
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
            self.assertEqual(events[0]["status"], "corroborated")

    def test_an_unledgered_heavy_cannot_verify(self) -> None:
        # The dict says a heavy run happened; the ledger has no seed
        # executions and no aggregate. "Provably occurred" is the gate.
        with tempfile.TemporaryDirectory() as tmp:
            path = corroborate(tmp)
            outcome = confirmed_outcome(
                heavy_confirmed=True,
                heavy=heavy_dict(),
                chain_path=str(candidate_chain(tmp, name="c2.jsonl")),
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
            self.assertEqual(events[0]["status"], "corroborated")
            self.assertIn("ledger-gated", events[0]["reason"])

    def test_the_model_cannot_invent_a_world_lever(self) -> None:
        # Wrapping, not authoring: when the bound world declares runtime
        # dynamics, the diagnosis may choose a lever from the measured
        # vocabulary or say "none" — a lever the bridge does not declare
        # is a transition rule written after seeing the claim.
        from farfield.extras.diagnose import DiagnosisRefused

        base = {
            "alternative": "the gap is cache noise, not the encoding",
            "experiment": "count comparisons under both encodings",
            "treatment_arm": "succinct encoding",
            "control_arm": "plain array",
            "expected_direction": "treatment_lower",
            "alternative_direction": "treatment_higher",
            "expected_if_alternative": "the arms tie",
            "margin": 0.05,
            "margin_reason": "toy-runner noise stays under five percent",
            "_world_levers": ("dropout",),
            "_world_observables": ("mean_degree",),
        }
        with self.assertRaises(DiagnosisRefused):
            diagnosis_from_payload(
                "c1",
                {**base, "world_lever": "distill_tools", "world_observable": "mean_degree"},
            )
        with self.assertRaises(DiagnosisRefused):
            diagnosis_from_payload(
                "c1",
                {**base, "world_lever": "dropout", "world_observable": "student_accuracy"},
            )
        honest = diagnosis_from_payload("c1", {**base, "world_lever": "none"})
        self.assertEqual(honest.world_lever, "none")
        self.assertEqual(honest.world_observable, "")

    def test_an_unconsumed_world_cannot_corroborate(self) -> None:
        from farfield.extras.evidence import host_confirmation_gaps

        outcome = {
            "probe_kind": "WORLD",
            "verdict": "supports",
            "host_ok": True,
            "protocol_complete": True,
            "evidence_id": "e" * 64,
            "host_evidence_id": "e" * 64,
            "experiment_digest": "a" * 64,
            "host_experiment_digest": "a" * 64,
            "world_digest": "b" * 64,
            "host_world_digest": "b" * 64,
            "data_digest": "b" * 64,
            "host_data_digest": "b" * 64,
            "host_replication": "E1_same_instance",
            "world_has_dynamics": True,
            "world_consumed": False,
        }
        gaps = host_confirmation_gaps(outcome)
        self.assertTrue(any("placebo" in gap or "consume" in gap for gap in gaps))

    def test_a_consumed_dynamic_world_can_still_corroborate(self) -> None:
        # The placebo gate must not freeze the ladder: a WORLD support
        # that actually displaced vs the ruined copy still climbs.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            register_hypothesis(tmp)
            outcome = confirmed_outcome(
                world_has_dynamics=True,
                world_consumed=True,
                chain_path=str(candidate_chain(tmp)),
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
            self.assertEqual(events[0]["status"], "corroborated")

    def test_the_heavy_interval_is_rechecked_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = corroborate(tmp)
            outcome = confirmed_outcome(
                heavy_confirmed=True,
                heavy=heavy_dict(ci_high=-0.01),
                chain_path=str(candidate_chain(tmp, heavy=True, name="c2.jsonl")),
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
            self.assertEqual(events[0]["status"], "corroborated")

    def test_a_heavy_from_another_world_cannot_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = corroborate(tmp)
            outcome = confirmed_outcome(
                heavy_confirmed=True,
                heavy=heavy_dict(world_digest="f" * 64),
                chain_path=str(candidate_chain(tmp, heavy=True, name="c2.jsonl")),
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
            self.assertEqual(events[0]["status"], "corroborated")

    def test_an_invariant_experiment_cannot_verify(self) -> None:
        # Identical numbers under every registered seed: determinism and a
        # seed-ignoring script are indistinguishable, so neither confirms.
        with tempfile.TemporaryDirectory() as tmp:
            path = corroborate(tmp)
            outcome = confirmed_outcome(
                heavy_confirmed=True,
                heavy=heavy_dict(invariant=True, verdict="uninformative"),
                chain_path=str(candidate_chain(tmp, heavy=True, name="c2.jsonl")),
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
            self.assertEqual(events[0]["status"], "corroborated")


# --- provenance and ordering: unprovable order is not provenance -------------


class ProvenanceOrderTests(unittest.TestCase):
    def test_execute_before_register_cannot_corroborate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            register_hypothesis(tmp)
            outcome = confirmed_outcome(
                chain_path=str(candidate_chain(tmp, reverse=True))
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
            self.assertEqual(events[0]["status"], "speculative")
            self.assertIn("registered after the execution", events[0]["reason"])

    def test_a_missing_candidate_chain_cannot_corroborate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            register_hypothesis(tmp)
            events = apply_outcomes(path, CORPUS, ANCHOR, [confirmed_outcome()])
            self.assertEqual(events[0]["status"], "speculative")
            self.assertIn("no event chain recorded", events[0]["reason"])

    def test_a_tampered_candidate_chain_cannot_corroborate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            register_hypothesis(tmp)
            chain_file = candidate_chain(tmp)
            lines = chain_file.read_text().splitlines()
            lines[0] = lines[0].replace(EXPERIMENT_DIGEST, "d" * 64)
            chain_file.write_text("\n".join(lines) + "\n")
            events = apply_outcomes(
                path, CORPUS, ANCHOR, [confirmed_outcome(chain_path=str(chain_file))]
            )
            self.assertEqual(events[0]["status"], "speculative")
            self.assertIn("event chain broken", events[0]["reason"])

    def test_an_unregistered_hypothesis_cannot_corroborate(self) -> None:
        # The claim must be in the research ledger before its evidence:
        # a hypothesis that appears only at promotion time could have been
        # written after — and to fit — the result.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            outcome = confirmed_outcome(chain_path=str(candidate_chain(tmp)))
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
            self.assertEqual(events[0]["status"], "speculative")
            self.assertIn("never registered", events[0]["reason"])

    def test_a_tampered_promotion_log_freezes_the_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = corroborate(tmp)
            ledger = Path(tmp) / "state.chain.jsonl"
            lines = ledger.read_text().splitlines()
            target = next(
                index for index, line in enumerate(lines) if "corroborated" in line
            )
            lines[target] = lines[target].replace("corroborated", "verified")
            ledger.write_text("\n".join(lines) + "\n")
            with self.assertRaises(StateError):
                apply_outcomes(
                    path, CORPUS, ANCHOR, [confirmed_outcome(verdict="weakens")]
                )


# --- identity: evidence belongs to one revision of one claim -----------------


class IdentityTests(unittest.TestCase):
    def test_missing_evidence_identity_cannot_corroborate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            register_hypothesis(tmp)
            outcome = confirmed_outcome(
                host_evidence_id="", chain_path=str(candidate_chain(tmp))
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
            self.assertEqual(events[0]["status"], "speculative")

    def test_a_host_run_on_different_bytes_cannot_corroborate(self) -> None:
        # E2 scale replication is generalization evidence, not an E1
        # reproduction; mismatched data digests must not confirm.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            register_hypothesis(tmp)
            outcome = confirmed_outcome(
                host_data_digest="f" * 64,
                host_evidence_id="f" * 64,
                host_replication="E2_scale_replication",
                chain_path=str(candidate_chain(tmp)),
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [outcome])
            self.assertEqual(events[0]["status"], "speculative")

    def test_a_claim_revision_after_the_experiment_resets_the_ladder(self) -> None:
        # Result-aware rewriting: the pair survives, the wording changes —
        # the new revision starts from scratch and the old evidence is
        # superseded context, never inherited truth.
        with tempfile.TemporaryDirectory() as tmp:
            path = corroborate(tmp)
            revised = confirmed_outcome(
                claim="pivot history compresses because of cache blocking",
                chain_path=str(candidate_chain(tmp, name="c2.jsonl")),
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [revised])
            self.assertTrue(events[0]["revision_reset"])
            self.assertIsNone(events[0]["previous"])
            self.assertNotEqual(events[0]["status"], "verified")

    def test_a_synthetic_support_cannot_keep_an_unproven_rank(self) -> None:
        # Without a grounding proof in the ledger, a SYNTHETIC follow-up
        # drops the rank; with the proof (written by corroborate) it holds.
        with tempfile.TemporaryDirectory() as tmp:
            path = corroborate(tmp)
            synthetic = confirmed_outcome(
                probe_kind="SYNTHETIC",
                host_ok=None,
                protocol_complete=None,
                host_evidence_id="",
                host_experiment_digest="",
                host_world_digest="",
                host_data_digest="",
                world_digest="",
                data_digest="",
                host_replication="",
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [synthetic])
            self.assertEqual(events[0]["status"], "corroborated")
        with tempfile.TemporaryDirectory() as tmp:
            # Same store surgery, but the ledger's PROMOTE lacks host
            # provenance: the rank was an assertion and does not survive.
            path = Path(tmp) / "state.json"
            register_hypothesis(tmp)
            apply_outcomes(
                path, CORPUS, ANCHOR, [confirmed_outcome()]
            )  # speculative: no chain
            topics = load(path)
            entry = next(
                iter(topics[f"{CORPUS}::{ANCHOR}"]["hypotheses"].values())
            )
            entry["status"] = "corroborated"  # forged rank, no ledger proof
            import farfield.extras.state as state_module

            state_module._save(path, topics)
            synthetic = confirmed_outcome(
                probe_kind="SYNTHETIC",
                host_ok=None,
                protocol_complete=None,
                host_evidence_id="",
                host_experiment_digest="",
                host_world_digest="",
                host_data_digest="",
                world_digest="",
                data_digest="",
                host_replication="",
            )
            events = apply_outcomes(path, CORPUS, ANCHOR, [synthetic])
            self.assertEqual(events[0]["status"], "speculative")

    def test_legacy_evidence_never_migrates_across_revisions(self) -> None:
        # Lines may merge; revisions may not. Evidence from a different
        # claim wording is quarantined as legacy_unresolved, not inherited.
        with tempfile.TemporaryDirectory() as tmp:
            path = corroborate(tmp)
            topics = load(path)
            bucket = topics[f"{CORPUS}::{ANCHOR}"]
            survivor = next(iter(bucket["hypotheses"].values()))
            stray = dict(survivor)
            stray["pair"] = ["succinct data", "pivot rule"]
            stray["claim"] = "an entirely different mechanism"
            stray["revision_id"] = ""
            stray["status"] = "speculative"
            stray["evidence"] = [
                {"kind": "probe", "direction": "+", "note": "old", "validity": "active"}
            ]
            bucket["hypotheses"]["deadbeef00000000"] = stray
            import farfield.extras.state as state_module

            state_module._save(path, topics)
            self.assertEqual(migrate_identities(path), 1)
            entry = next(
                iter(load(path)[f"{CORPUS}::{ANCHOR}"]["hypotheses"].values())
            )
            self.assertEqual(entry["status"], "corroborated")
            inherited = [
                item for item in entry["evidence"] if item.get("note") == "old"
            ]
            self.assertTrue(inherited)
            self.assertTrue(
                all(item["validity"] == "legacy_unresolved" for item in inherited)
            )


# --- the executor end to end: real reruns, real ledger ------------------------


SEEDED_SOURCE = """import json
from pathlib import Path
seed_file = Path('data/seed.json')
seed = json.loads(seed_file.read_text())['seed'] if seed_file.is_file() else 0
treatment = 80.0 + (seed % 7) * 0.1
control = 120.0 + (seed % 5) * 0.1
Path('metrics.json').write_text(json.dumps(
    {'treatment': treatment, 'control': control}))
"""

CONSTANT_SOURCE = """import json
from pathlib import Path
work = [1] * 80
treatment = float(len(work))
control = float(len(work) + 40)
Path('metrics.json').write_text(json.dumps(
    {'treatment': treatment, 'control': control}))
"""


class _Card:
    card_id = "gen_const"
    claim = CLAIM
    mechanism = "pivot history is compressible"
    pair = tuple(PAIR)
    operator = "farfield"


def _diagnosis():
    return diagnosis_from_payload(
        "gen_const",
        {
            "alternative": "the gap is cache noise, not the encoding",
            "experiment": "count comparisons under both encodings",
            "treatment_arm": "succinct encoding",
            "control_arm": "plain array",
            "expected_direction": "treatment_lower",
            "alternative_direction": "treatment_higher",
            "expected_if_alternative": "the arms tie",
            "margin": 0.05,
            "margin_reason": "toy-runner noise stays under five percent",
        },
    )


def _spec(source: str):
    return spec_from_payload(
        {
            "measure": "comparisons",
            "treatment_arm": "succinct encoding",
            "control_arm": "plain array",
            "source": source,
        },
        tier=resolve_tier("toy"),
        claim=CLAIM,
        mechanism="pivot history is compressible",
        topic=ANCHOR,
    )


class ExecutorOwnedHeavyTests(unittest.TestCase):
    def test_the_executor_reruns_the_registered_script_and_ledgers_it(self) -> None:
        # End to end: derived seeds, N real subprocess reruns of the
        # registered bytes, executor statistics, and a ledger that can
        # prove all of it to the promotion gate afterwards.
        with tempfile.TemporaryDirectory() as tmp:
            events = list(
                _heavy_confirmation(_Card(), _diagnosis(), _spec(SEEDED_SOURCE), Path(tmp))
            )
            aggregate = events[-1]
            self.assertEqual(aggregate["stage"], "heavy_evidence")
            self.assertEqual(aggregate["runner"], "executor")
            self.assertEqual(aggregate["verdict"], "supports")
            self.assertEqual(
                aggregate["seeds"],
                list(replication_seeds(aggregate["experiment_digest"], "", 5)),
            )
            chain_path = Path(tmp) / "candidates" / "gen_const" / "chain.jsonl"
            self.assertTrue(chain_path.is_file())
            gaps = chain.audit_heavy_gaps(
                chain_path,
                evidence_id=aggregate["evidence_id"],
                experiment_digest=aggregate["experiment_digest"],
            )
            self.assertEqual(gaps, ())

    def test_a_seed_ignoring_script_is_named_invariant(self) -> None:
        # The constant script produces identical numbers under every seed:
        # the executor detects it and the verdict is uninformative — a
        # script cannot opt out of variation and still get confirmed.
        with tempfile.TemporaryDirectory() as tmp:
            events = list(
                _heavy_confirmation(
                    _Card(), _diagnosis(), _spec(CONSTANT_SOURCE), Path(tmp)
                )
            )
            aggregate = events[-1]
            self.assertEqual(aggregate["verdict"], "uninformative")
            self.assertTrue(aggregate["invariant"])


if __name__ == "__main__":
    unittest.main()
