"""Deterministic fixtures for typed idea_kind across the research lifecycle."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.diagnose import diagnosis_from_payload
from farfield.extras.evidence import host_confirmation_gaps
from farfield.extras.evidencegrid import compile_grid
from farfield.extras.explore import ExplorationPrior, decide_idea_refine
from farfield.extras.farcompile import WorldMenu, assert_compile
from farfield.extras.generate import GeneratedCard, GenerationRefused, generate_card, refine_card
from farfield.extras.ideakind import ACQUIRE, PROBE, QUESTION, THEORY, runs_probe
from farfield.extras.lifecycle import (
    CAUSAL_IDENTIFICATION_READY,
    CrossWorldEvidenceError,
    HARVEST_SUCCESS,
    LOCAL_ORIGIN_LABEL,
    MISSING_WORLD,
    ORIGIN_LOCAL,
    admit_kind_transition,
    assert_evidence_world,
    attractor_key,
    child_probe_freeze,
    design_signature,
    feasibility_requires_two_arm,
    feasibility_rubric,
    handle_is_blacklisted,
    harvest_acquired_world,
    lineage_id_for,
    opportunity_from_menu,
    parse_kind_fields,
    propose_acquire,
    refuse_kind_transition,
    two_arm_feasibility_leak,
    validate_acquire_recipe,
    version_from_fixture,
)
from farfield.extras.mission import execute_acquire_harvest
from farfield.extras.paperplan import AUTHOR_TEMPLATE
from farfield.extras.review import KIND_REVIEW_BRIEFS
from farfield.extras.state import programs_for_pair, record_program
from farfield.extras.world import WorldFixture, digest_files
from tests.test_generate import FakeClient, LABELS, answer


def _menu(*, room: bool = False) -> WorldMenu:
    levers_map = {
        "contrast:validator_write": {"false_accept_fraction": 0.12 if not room else 0.12},
    }
    if room:
        levers_map["dropout"] = {"false_accept_fraction": 0.08}
    return WorldMenu(
        levers=tuple(levers_map),
        observables=("false_accept_fraction",),
        inert=frozenset({"dropout/false_accept_fraction"} if not room else ()),
        attested_fields=("false_accept_fraction",),
        room_to_move=room,
        world_id="W0",
    )


def _card(**overrides) -> GeneratedCard:
    payload = dict(
        card_id="gen_lifecycle01",
        operator="directional",
        claim="is false acceptance higher on validator-write episodes?",
        mechanism="the freeze already recorded the contrast",
        prediction="compare false_accept_fraction across the contrast",
        falsifier="pair_not_already_combined",
        pair=("code world model", "local problem structure"),
        pair_nodes=("concept:a", "local:problem-structure"),
        alienness=0.2,
        model="fake",
        artifact_digest="d",
        artifact_uri="file:///dev/null",
        replay_mode="replay",
        idea_kind=QUESTION,
        origin=ORIGIN_LOCAL,
        lineage_id=lineage_id_for(
            ("code world model", "local problem structure"), origin=ORIGIN_LOCAL
        ),
        research_target="is false acceptance higher on validator-write episodes?",
        world_lever="contrast:validator_write",
        world_observable="false_accept_fraction",
        kind_fields={
            "research_question": "is false acceptance higher on validator-write episodes?",
            "contrast": "contrast:validator_write",
            "why_answer_matters": "it decides whether to harvest a mutable-validator world",
            "possible_outcomes": "higher, lower, or indistinguishable",
            "decision_change": "if higher, acquire a harvestable validator world",
            "required_world_state": "replayed history with contrast strata",
        },
    )
    payload.update(overrides)
    return GeneratedCard(**payload)


def _world(tmp: Path, *, world_id: str = "W0") -> WorldFixture:
    root = tmp / world_id
    root.mkdir(parents=True)
    (root / "world.json").write_text("{\"n\": 2}\n", encoding="utf-8")
    digest = digest_files(root, ["world.json"])
    return WorldFixture(
        id=world_id,
        title="parent freeze",
        source="test",
        retrieved_at="2026-09-07",
        digest=digest,
        files=("world.json",),
        root=root,
        schema="program_state",
        provenance="published",
        version="W0",
    )


class OpportunityTests(unittest.TestCase):
    def test_case_a_no_intervention_still_opens_question(self) -> None:
        menu = WorldMenu(
            levers=("contrast:validator_write", "dropout"),
            observables=("false_accept_fraction",),
            inert=frozenset({"dropout/false_accept_fraction"}),
            room_to_move=False,
            world_id="W0",
        )
        opp = opportunity_from_menu(menu)
        self.assertFalse(opp.probe_room)
        self.assertNotIn(PROBE, opp.eligible_kinds())
        self.assertIn(QUESTION, opp.eligible_kinds())
        self.assertIn(ACQUIRE, opp.eligible_kinds())
        self.assertTrue(opp.research_open())
        self.assertFalse(runs_probe(QUESTION))

    def test_missing_world_raises_acquire_priority(self) -> None:
        opp = opportunity_from_menu(
            None, missing_capabilities=("validators_mutable snapshot",)
        )
        self.assertGreater(opp.acquire_priority(), 0)
        self.assertIn(ACQUIRE, opp.eligible_kinds(local_problem=True))


class SchemaTests(unittest.TestCase):
    def test_question_does_not_need_struggle_or_two_arm(self) -> None:
        card = generate_card(
            FakeClient(
                answer(
                    idea_kind="question",
                    origin="local_problem_structure",
                    pair=["sparse attention", LOCAL_ORIGIN_LABEL],
                    claim="does sparse attention change cache occupancy on this freeze?",
                    mechanism="the freeze already records occupancy by layer",
                    prediction="compare occupancy across recorded layers",
                    dead_end="",
                    why_failed="",
                    reframe="",
                    objection="",
                    research_question="does occupancy differ by layer?",
                    contrast="recorded layer strata",
                    why_answer_matters="it decides whether a cache probe is worth registering",
                    possible_outcomes="differs or does not",
                    decision_change="if it differs, register an observational measure",
                )
            ),
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=(),
            operator="directional",
            alienness=0.1,
            label_to_node=LABELS,
        )
        self.assertEqual(card.idea_kind, QUESTION)
        self.assertEqual(card.origin, ORIGIN_LOCAL)
        self.assertEqual(card.pair[1], LOCAL_ORIGIN_LABEL)
        self.assertFalse(card.dead_end)

    def test_case_h_empty_far_does_not_invent_a_mechanism(self) -> None:
        with self.assertRaises(GenerationRefused):
            generate_card(
                FakeClient(
                    answer(
                        idea_kind="question",
                        pair=["sparse attention", "protein folding"],
                        claim="apply protein folding to sparse attention",
                    )
                ),
                seed_label="sparse attention",
                near_labels=("kv cache",),
                far_labels=(),
                operator="directional",
                alienness=0.1,
                label_to_node=LABELS,
            )


class TransitionTests(unittest.TestCase):
    def test_case_b_question_may_become_acquire_on_missing_world(self) -> None:
        self.assertTrue(admit_kind_transition(QUESTION, ACQUIRE, MISSING_WORLD))
        self.assertTrue(refuse_kind_transition(PROBE, THEORY, "weak_result"))
        parent = _card()
        child = generate_card(
            FakeClient(
                answer(
                    idea_kind="acquire",
                    origin="local_problem_structure",
                    pair=["code world model", LOCAL_ORIGIN_LABEL],
                    claim="harvest a world whose validators stay mutable",
                    mechanism="this freeze cannot snapshot validator rewrites",
                    prediction="validators_mutable must be true and attested",
                    recipe="copy the harness and enable an evolvable evaluator cell",
                    validators="object_properties.validators_mutable is true",
                    readiness_criteria="world.json attests validators_mutable",
                    capability_gap="no validator snapshot capability",
                    transition_reason=MISSING_WORLD,
                    far_maps_to_lever="",
                    world_lever="",
                    dead_end="",
                    why_failed="",
                    reframe="",
                    objection="",
                )
            ),
            seed_label="code world model",
            near_labels=(),
            far_labels=(LOCAL_ORIGIN_LABEL,),
            operator="directional",
            alienness=0.1,
            label_to_node={"code world model": "concept:cwm"},
        )
        self.assertEqual(child.idea_kind, ACQUIRE)
        fields = parse_kind_fields(ACQUIRE, child.kind_fields or child.to_dict())
        self.assertTrue(fields.get("recipe"))
        record = validate_acquire_recipe(propose_acquire(child))
        self.assertIn(record.state, {"PROPOSED", "RECIPE_VALIDATED"})
        self.assertFalse(feasibility_requires_two_arm(ACQUIRE))
        self.assertFalse(two_arm_feasibility_leak(feasibility_rubric(ACQUIRE), ACQUIRE))


class IdentityTests(unittest.TestCase):
    def test_case_d_same_pair_two_kinds_are_two_ideas(self) -> None:
        q = _card(card_id="gen_q1", idea_kind=QUESTION)
        a = _card(
            card_id="gen_a1",
            idea_kind=ACQUIRE,
            claim="harvest a mutable-validator world",
            research_target="harvest a mutable-validator world",
            lineage_id=q.lineage_id,
        )
        self.assertNotEqual(q.card_id, a.card_id)
        self.assertEqual(q.lineage_id, a.lineage_id)
        self.assertEqual(q.pair, a.pair)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            record_program(
                path,
                "c1",
                "anchor",
                {
                    "commit": "question H",
                    "pair": list(q.pair),
                    "idea_id": q.card_id,
                    "lineage_id": q.lineage_id,
                    "idea_kind": QUESTION,
                    "card_id": q.card_id,
                },
            )
            record_program(
                path,
                "c1",
                "anchor",
                {
                    "commit": "acquire H",
                    "pair": list(a.pair),
                    "idea_id": a.card_id,
                    "lineage_id": a.lineage_id,
                    "idea_kind": ACQUIRE,
                    "card_id": a.card_id,
                },
            )
            from farfield.extras.state import load

            rows = programs_for_pair(load(path), "c1", "anchor", q.pair)
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["idea_id"] for row in rows}, {q.card_id, a.card_id})


class WorldVersionTests(unittest.TestCase):
    def test_case_c_harvest_creates_w1_without_rewriting_w0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = _world(root / "parent", world_id="W0")
            w0_digest = parent.digest
            w0_text = (parent.root / "world.json").read_text(encoding="utf-8")
            card = _card(
                card_id="gen_acq1",
                idea_kind=ACQUIRE,
                claim="harvest a snapshot-capable world",
                kind_fields={
                    "capability_gap": "no validator snapshot",
                    "recipe": "derive a child world with snapshot validators",
                    "validators": "manifest.world_evidence_id is new",
                    "readiness_criteria": "child digest differs from parent",
                    "expected_affordances": "child probe can freeze to W1",
                },
            )
            child, bound = execute_acquire_harvest(
                root / "mission",
                card,
                parent,
                new_id="W1",
            )
            self.assertIsNotNone(child)
            self.assertEqual(bound.world_id, "W1")
            self.assertNotEqual(bound.evidence_id, "")
            self.assertEqual(parent.digest, w0_digest)
            self.assertEqual((parent.root / "world.json").read_text(encoding="utf-8"), w0_text)
            self.assertNotEqual(child.digest, parent.digest)
            self.assertEqual(child.parent_id, "W0")
            self.assertEqual(child.version, "W1")
            probe = _card(
                card_id="gen_p1",
                idea_kind=PROBE,
                world_version="W1",
                lineage_id=card.lineage_id,
            )
            freeze = child_probe_freeze(
                probe,
                [version_from_fixture(child, path=str(child.root)).to_dict()],
            )
            self.assertEqual(freeze["world_id"], "W1")
            self.assertEqual(freeze["parent_world"], "W0")

    def test_case_i_w0_evidence_cannot_promote_w1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = _world(Path(tmp), world_id="W0")
            child, _ = harvest_acquired_world(
                parent,
                Path(tmp) / "W1",
                validate_acquire_recipe(
                    propose_acquire(
                        _card(
                            card_id="gen_acq2",
                            idea_kind=ACQUIRE,
                            kind_fields={
                                "recipe": "derive",
                                "validators": "digest",
                                "readiness_criteria": "new id",
                            },
                        )
                    )
                ),
                new_id="W1",
            )
            with self.assertRaises(CrossWorldEvidenceError):
                assert_evidence_world(
                    {
                        "evidence_id": "e0",
                        "world_id": "W0",
                        "world_digest": parent.digest,
                    },
                    child,
                )
            gaps = host_confirmation_gaps(
                {
                    "probe_kind": "WORLD",
                    "verdict": "supports",
                    "host_ok": True,
                    "protocol_complete": True,
                    "evidence_id": "e0",
                    "host_evidence_id": "e0",
                    "world_id": "W0",
                    "claim_world_id": "W1",
                    "world_digest": parent.digest,
                    "claim_world_digest": child.digest,
                    "experiment_digest": "x",
                    "host_experiment_digest": "x",
                    "data_digest": "d",
                    "host_data_digest": "d",
                }
            )
            self.assertTrue(any("cross-world" in gap or "digest" in gap for gap in gaps))


class HandleScopeTests(unittest.TestCase):
    def test_case_e_blacklist_is_design_and_world_scoped(self) -> None:
        handle = "contrast:validator_write/false_accept_fraction"
        sig_a = design_signature(
            lever="contrast:validator_write",
            observable="false_accept_fraction",
            treatment="stratum write",
            control="rest",
        )
        sig_b = design_signature(
            lever="contrast:validator_write",
            observable="false_accept_fraction",
            treatment="stratum write matched on template",
            control="rest matched on updates",
        )
        self.assertNotEqual(sig_a, sig_b)
        key_a = attractor_key(
            world_id="W0", handle=handle, failure_mode="poor_matching", design_sig=sig_a
        )
        self.assertTrue(
            handle_is_blacklisted([key_a], handle, world_id="W0", failure_mode="poor_matching", design_sig=sig_a)
        )
        self.assertFalse(
            handle_is_blacklisted([key_a], handle, world_id="W0", failure_mode="poor_matching", design_sig=sig_b)
        )
        menu = WorldMenu(
            levers=("contrast:validator_write",),
            observables=("false_accept_fraction",),
            inert=frozenset(),
            blocked_handles=frozenset(),
        )
        unused = menu.unused_handles([handle], reuse_scope="spray")
        self.assertNotIn(handle, unused)
        reused = menu.unused_handles(
            [handle], reuse_scope="lineage", lineage_handles=[handle]
        )
        self.assertIn(handle, reused)
        prior = ExplorationPrior()
        prior.absorb_landing(
            pair=("code world model", "local problem structure"),
            world_lever="contrast:validator_write",
            world_observable="false_accept_fraction",
            category="entered",
        )
        self.assertIn(handle, prior.used_levers)


class ReviewerTests(unittest.TestCase):
    def test_case_f_acquire_rubric_must_not_demand_two_arm(self) -> None:
        self.assertFalse(feasibility_requires_two_arm(ACQUIRE))
        self.assertFalse(two_arm_feasibility_leak(feasibility_rubric(ACQUIRE), ACQUIRE))
        self.assertNotIn("two-arm test is obvious", feasibility_rubric(ACQUIRE))
        self.assertNotIn("two-arm test is obvious", KIND_REVIEW_BRIEFS[ACQUIRE])
        probe_only = "5 = a two-arm test is obvious"
        self.assertTrue(two_arm_feasibility_leak(probe_only, ACQUIRE))

    def test_case_g_question_cannot_be_causalized(self) -> None:
        menu = WorldMenu(
            levers=("contrast:validator_write", "dropout"),
            observables=("false_accept_fraction",),
            inert=frozenset(),
            attested_fields=("false_accept_fraction",),
        )

        with self.assertRaises(RuntimeError):
            assert_compile(
                far_label="local problem structure",
                world_lever="dropout",
                world_observable="false_accept_fraction",
                far_maps_to_lever=(
                    "local problem structure becomes dropout by masking updates"
                ),
                prediction="false_accept_fraction drops by a margin of 0.08",
                menu=menu,
                refuse=lambda attempted, unlock, capability="", failed_far="", used_handle=False: RuntimeError(unlock),
                idea_kind=QUESTION,
            )
        with self.assertRaises(GenerationRefused) as caught:
            generate_card(
                FakeClient(
                    answer(
                        idea_kind="question",
                        pair=["sparse attention", LOCAL_ORIGIN_LABEL],
                        claim="dropout causes false acceptance to drop",
                        mechanism="independent masking is the intervention",
                        prediction="false_accept_fraction drops by a margin of 0.08",
                        world_lever="dropout",
                        world_observable="false_accept_fraction",
                        dead_end="",
                        why_failed="",
                        reframe="",
                        objection="",
                    )
                ),
                seed_label="sparse attention",
                near_labels=("kv cache",),
                far_labels=(),
                operator="directional",
                alienness=0.1,
                label_to_node=LABELS,
                world_menu=menu,
            )
        self.assertEqual(
            caught.exception.record.missing_capability, "model_causal_overclaim"
        )

    def test_refine_controller_uses_failure_type(self) -> None:
        decision = decide_idea_refine(
            attempt=0,
            cap=2,
            alive=True,
            failure_types=("missing_world",),
            idea_kind=QUESTION,
        )
        self.assertTrue(decision["continue"])
        self.assertEqual(decision["action"], "transition_kind")


class DiagnosePaperTests(unittest.TestCase):
    def test_question_diagnosis_does_not_require_margin(self) -> None:
        diag = diagnosis_from_payload(
            "gen_q",
            {"experiment": "compare false_accept_fraction on validator-write vs rest"},
            idea_kind=QUESTION,
        )
        self.assertEqual(diag.margin, 0.0)
        self.assertEqual(diag.expected_direction, "")

    def test_paper_template_does_not_force_two_arm_h1(self) -> None:
        self.assertIn("Do not rewrite it as a two-arm H1", AUTHOR_TEMPLATE)
        grid = compile_grid(
            claim="need a mutable-validator world",
            world={"id": None},
            dynamics_card=None,
            diagnosis={},
            attempts=[],
            idea_kind=ACQUIRE,
        )
        self.assertEqual(grid["evidence_role"], "AcquisitionEvidence")
        self.assertNotEqual(grid["cells"][0]["id"], "C1.main")


class SprayStopTests(unittest.TestCase):
    def test_case_a_no_global_spray_closed_from_probe_room(self) -> None:
        import inspect
        from farfield.extras import mission as mission_mod

        source = inspect.getsource(mission_mod)
        self.assertNotIn('spray_closed_no_room = True', source)
        self.assertIn("probe_room_closed", source)


if __name__ == "__main__":
    unittest.main()
