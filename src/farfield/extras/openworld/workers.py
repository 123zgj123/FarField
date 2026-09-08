"""Single-action adapters over the mature research workers.

The controller owns scheduling and redesign. These functions never enter a
mission, synthesize arm numbers, or retry a scientific result.
"""
from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Mapping

from .. import chain
from ..brief import compile_brief
from ..diagnose import diagnosis_from_payload, judge_probe, write_diagnosis
from ..evidence import ablation_required, evidence_id, sha256_text
from ..generate import GeneratedCard, GenerationRefused, generate_card
from ..hostexp import ExecuteError, execute_protocol
from ..knowledge import harvest_survey
from ..livefeed import CompositeFeed, FreshWork
from ..llm import LLMUnavailable
from ..packet import render_protocol
from ..prior import strongest_prior
from ..probeexp import floor_tier, probe_timeout, resolve_tier, run_probe, write_probe
from ..reframe import generate_reframe
from ..world import bind_world, digest_files, load_fixture, reads_world_data
from ..workspace import write_idea
from .actions import PROBE, SURVEY, SYNTHESIZE, THEORIZE, VERIFY
from .events import make_event
from .kernel import CandidateResult, readonly_view
from .worldio import is_attested


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _card(raw: Any) -> GeneratedCard:
    if isinstance(raw, GeneratedCard):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("no generated card is attached to this theory")
    allowed = {item.name for item in fields(GeneratedCard)}
    values = {key: value for key, value in raw.items() if key in allowed}
    for name in ("pair", "pair_nodes"):
        if name in values:
            values[name] = tuple(values[name])
    result = GeneratedCard(**values)
    if not all((result.card_id, result.claim, result.mechanism, result.prediction)):
        raise ValueError("the generated card is incomplete")
    if Path(result.card_id).name != result.card_id or result.card_id in {".", ".."}:
        raise ValueError("invalid card id")
    return result


class LegacyResearchWorkers:
    """Inject a replay/live client and feed; no network work at construction.

    Executors accept ``(action, state, **kwargs)``. ``world`` may be an
    explicit fixture; otherwise it is loaded from workspace/worlds/<id>.
    ProbeEvidence carries ``card``, ``diagnosis``, ``protocol_dir``, and
    immutable registration identities for subsequent VERIFY/SYNTHESIZE.
    """

    def __init__(self, client=None, feed=None, workspace=None):
        self.client = client
        self.feed = feed
        self.workspace = Path(workspace) if workspace is not None else None

    def executors(self):
        return {SURVEY: self.survey, THEORIZE: self.theorize, PROBE: self.probe,
                VERIFY: self.verify, SYNTHESIZE: self.synthesize}

    def _workspace(self, kwargs):
        value = kwargs.get("workspace") or self.workspace
        if value is None:
            raise ValueError("a workspace is required to preserve research artifacts")
        return Path(value)

    def _failure(self, action, reason, *, status="blocked", failed=False, **extra):
        payload = {"reason": str(reason), "theory_id": action.target,
                   "question_id": action.frontier_target_id, **extra}
        return CandidateResult(status=status, events=[make_event(
            "ActionFailed" if failed else "ActionBlocked", payload,
            action_type=action.action_type, frontier_target_id=action.frontier_target_id or action.target)],
            extras=payload)

    def _theory(self, action, state):
        row = next((item for item in state.get("theories", [])
                    if item.get("id") == action.target), None)
        if row is None and action.extra.get("theory"):
            row = action.extra["theory"]
        return dict(row or {})

    def _works(self, state, question_id=""):
        result = {}
        for artifact in state.get("artifacts", []):
            if artifact.get("kind") != "literature_survey":
                continue
            if question_id and artifact.get("question_id") not in {"", question_id}:
                continue
            for row in artifact.get("works", []):
                work = FreshWork.from_dict(row)
                if work.cite_id():
                    result[work.cite_id()] = work
        return list(result.values())

    def _world(self, state, kwargs):
        supplied = kwargs.get("world")
        world_id = str(state.get("world_id") or kwargs.get("world_id") or "")
        if supplied is not None:
            world = load_fixture(Path(supplied.root))
            if world.digest != supplied.digest or world.id != supplied.id:
                raise ValueError("world identity differs from its manifest")
        else:
            if not world_id or Path(world_id).name != world_id or world_id in {".", ".."}:
                raise ValueError("no attested world is bound")
            world = load_fixture(self._workspace(kwargs) / "worlds" / world_id)
        if world_id and world.id != world_id:
            raise ValueError("world is not the active controller world")
        if not is_attested(world):
            raise ValueError("world is generated or its freeze cannot be attested")
        return world

    def survey(self, action, state, **kwargs):
        state = readonly_view(state)
        topic = str(state.get("goal") or action.target)
        question = next((row for row in state.get("open_questions", [])
                         if row.get("id") == action.target), {})
        try:
            harvest = harvest_survey(self.feed if self.feed is not None else CompositeFeed(),
                                     topic, str(question.get("text") or topic), topic=topic,
                                     claim=str(action.extra.get("claim") or ""),
                                     fetcher=kwargs.get("fetcher"))
            if not harvest.works:
                return self._failure(action, "no_verified_literature", citations=[], harvest=harvest.to_dict())
            works = list(harvest.works)
            prior = strongest_prior(str(action.extra.get("claim") or question.get("text") or topic), works)
            spans = [{"cite_id": work.cite_id(), "field": "abstract", "start": 0,
                      "end": len(work.abstract), "text": work.abstract, "url": work.href()}
                     for work in works if work.abstract]
            identity = sha256_text(json.dumps(harvest.to_dict(), sort_keys=True))[:20]
            path = self._workspace(kwargs) / "literature" / f"{identity}.json"
            artifact = {"id": f"survey-{identity}", "kind": "literature_survey",
                        "theory_id": action.extra.get("theory_id", ""),
                        "question_id": action.target, "citations": [work.cite_id() for work in works],
                        "read_first": [work.cite_id() for work in works], "spans": spans,
                        "prior": prior.to_dict() if prior else None, "path": str(path),
                        **harvest.to_dict()}
            if action.extra.get("extract_trajectories"):
                from ..explore import extract_paper_state, extract_transition
                papers, transitions, gaps = [], [], []
                for work in works[:6]:
                    try:
                        papers.append(extract_paper_state(work, client=self.client))
                    except (LLMUnavailable, ValueError, KeyError, TypeError) as exc:
                        gaps.append({"cite_id": work.cite_id(), "reason": str(exc)})
                        papers.append(extract_paper_state(work))
                for source in papers:
                    for target in papers:
                        from ..explore import _research_citation
                        if source.cite_id == target.cite_id or _research_citation(source.cite_id) not in {_research_citation(r) for r in target.references}:
                            continue
                        try:
                            transition = extract_transition(source, target, client=self.client)
                        except (LLMUnavailable, ValueError, KeyError, TypeError) as exc:
                            gaps.append({"source": source.cite_id, "target": target.cite_id, "reason": str(exc)})
                            transition = extract_transition(source, target)
                        transitions.append(transition.to_dict())
                anchor = {"problem": topic, "research_question": str(question.get("text") or topic),
                          "epistemic": "GENERATED", "anchor_basis": [p.cite_id for p in papers],
                          "unresolved_questions": [r.get("text") for r in state.get("open_questions", [])],
                          "actual_evidence": list(state.get("evidence_records", []))}
                for name in ("bottleneck_gap", "assumptions", "representation", "evidence"):
                    anchor[name] = list(dict.fromkeys(getattr(p, name) for p in papers if getattr(p, name) != "unknown"))
                artifact.update(paper_states=[p.to_dict() for p in papers], research_transitions=transitions,
                                research_state=anchor, extraction_gaps=gaps)
            _json(path, artifact)
            return CandidateResult(status="surveyed", events=[make_event(
                "ArtifactAdded", artifact, action_type=SURVEY,
                frontier_target_id=action.frontier_target_id or action.target)], extras=artifact)
        except (OSError, ValueError, LLMUnavailable) as exc:
            return self._failure(action, exc, citations=[])

    def theorize(self, action, state, **kwargs):
        state = readonly_view(state)
        if action.extra.get("research_lane"):
            return self._research_branch(action, state, **kwargs)
        if self.client is None:
            return self._failure(action, "research_model_not_connected")
        topic = str(state.get("goal") or action.target)
        works = self._works(state, action.target)
        question = next((row for row in state.get("open_questions", []) if row.get("id") == action.target), {})
        knowledge = tuple(str(row.get("text") or row.get("reason") or "")
                          for row in state.get("failed_designs", []))
        context = tuple(f"{work.cite_id()}: {work.title}. {work.abstract}" for work in works)
        common = dict(seed_label=topic, near_labels=(topic,), topic=topic, context=context,
                      knowledge=knowledge, program=(str(question.get("text") or action.target),))
        try:
            if action.extra.get("far_labels"):
                card = generate_card(self.client, **common,
                    far_labels=tuple(action.extra["far_labels"]),
                    operator=str(action.extra.get("operator") or "directional"),
                    alienness=float(action.extra.get("alienness") or 0.0),
                    label_to_node=dict(action.extra.get("label_to_node") or {}),
                    world_menu=action.extra.get("world_menu"))
            else:
                card = generate_reframe(self.client, **common,
                                        operator=str(action.extra.get("operator") or "assumption_removal"))
            competing = list(dict.fromkeys([card.objection] + list(state.get("competing_explanations", []))))
            competing = [text for text in competing if text]
            theory = {"id": card.card_id, "question_id": action.target, "target": action.target,
                      "text": card.claim, "mechanism": card.mechanism,
                      "assumptions": [card.pair[1]], "predictions": [card.prediction],
                      "boundary_conditions": [], "competing": competing,
                      "discriminating_predictions": [{"prediction": card.prediction, "maps_to": PROBE,
                          "competitor": competing[0] if competing else "", "competitor_differs": False}],
                      "card": card.to_dict(), "epistemic": "GENERATED",
                      "evidence_considered": [row.get("evidence_id") for row in state.get("evidence_records", [])],
                      "read_first": [work.cite_id() for work in works]}
            return CandidateResult(status="theorized", events=[make_event(
                "TheoryCreated", theory, action_type=THEORIZE,
                frontier_target_id=action.frontier_target_id or action.target)], extras={"theory": theory})
        except (GenerationRefused, LLMUnavailable, ValueError, KeyError) as exc:
            return self._failure(action, exc)

    def _research_branch(self, action, state, **kwargs):
        """Apply a historical operator or explore a different lane, once per action."""
        from ..explore import copy_check, retrieve_transitions
        if self.client is None:
            return self._failure(action, "research_model_not_connected")
        works = self._works(state, action.target)
        if not works:
            return self._failure(action, "verified_literature_required_before_theory")
        lane = action.extra["research_lane"]
        parent_id = action.extra.get("parent_theory_id", "")
        parent = next((t for t in state.get("theories", []) if t.get("id") == parent_id), {})
        hop = int(parent.get("hop") or 1) + 1 if parent_id else 1
        requested_hop = action.extra.get("hop", hop)
        if (type(requested_hop) is not int or requested_hop != hop
                or parent_id and (not parent or parent.get("question_id") != action.target)):
            return self._failure(action, "parent_lineage_or_hop_mismatch")
        # Re-read canonical state, never trust caller-supplied evidence_context.
        evidence = [r for r in state.get("evidence_records", []) if r.get("theory_id") == parent_id
                    and r.get("epistemic") == "WORLD" and r.get("attested")
                    and r.get("execution_status") == "ran" and r.get("world_digest") and r.get("experiment_digest")]
        failures = [r for r in state.get("failed_designs", []) if r.get("theory_id") == parent_id]
        if hop > 1 and not (evidence or failures):
            return self._failure(action, "next_hop_requires_actual_parent_evidence_or_explicit_failure")
        surveys = [r for r in state.get("artifacts", []) if r.get("kind") == "literature_survey"]
        transitions = [t for r in surveys for t in r.get("research_transitions", [])]
        anchor = dict(next((r["research_state"] for r in reversed(surveys) if r.get("research_state")), {}))
        anchor.update(problem=state.get("goal"), actual_evidence=evidence,
                      contradiction=state.get("contradictions", []), failed_designs=failures)
        if parent.get("assumptions"):
            anchor["assumptions"] = parent["assumptions"]
        selected = retrieve_transitions(anchor, transitions)
        policy = next((r.get("policy") for r in reversed(state.get("artifacts", []))
                       if r.get("kind") == "research_policy_snapshot"), None)
        if policy:
            from ..routing import preferred_operators
            order = preferred_operators(policy, tuple(dict.fromkeys(t.operator for t in selected)))
            selected.sort(key=lambda t: order.index(t.operator))
        if evidence and any(r.get("outcome") in {"weakens", "negative"} for r in evidence):
            selected = [t for t in selected if t.operator != parent.get("operator_reused")]
        if lane == "trajectory_supported" and not selected:
            if not parent_id:
                return self._failure(action, "no_applicable_grounded_trajectory", research_lane=lane)
            # An evidence-driven pivot may leave a depleted historical library.
            lane = "assumption_reframe"
        chosen = selected[:2] if lane == "trajectory_supported" else []
        operator = chosen[0].operator if chosen else (
            "representation_change" if evidence else "assumption_removal")
        if lane == "unconstrained":
            operator = "unconstrained_exploration"
        schema = {
            "assumptions": ["explicit unresolved assumption"],
            "competing_explanations": [{"id": "alternative", "claim": "substantive alternative mechanism",
                                        "expected_outcome": {"observable": "metric", "direction": "unchanged"}}],
            "closest_prior_work": ["COPY verified citation id"],
            "inherited_components": ["component inherited from closest prior"],
            "genuinely_changed_components": ["new mechanism or boundary, not only new topic"],
            "claim_delta": "falsifiable claim beyond prior", "mechanism_delta": "specific mechanism change",
            "why_not_reconstruction": "why this is not historical target reconstruction",
            "experiment_proposal": {"hypotheses_compared": ["self", "alternative"],
                "expected_outcomes": {"self": {"observable": "metric", "direction": "decrease"},
                                      "alternative": {"observable": "metric", "direction": "unchanged"}},
                "nuisance_factors": ["measurement confound"], "minimal_discriminating_intervention": "matched intervention",
                "interpretation_if_positive": "which explanation weakens", "interpretation_if_negative": "which claim weakens",
                "interpretation_if_inconclusive": "what remains unknown", "expected_scientific_gain": 0.5,
                "transfer_value": 0.5, "frontier_unlock": 0.5, "cost": 0.1, "risk": 0.1},
        }
        instructions = ("In addition to the worker schema include a concrete falsifier and research_metadata following "
            "this shape (examples are schema only, NOT supplied scientific facts): " + json.dumps(schema) +
            "\nNo arbitrary confidence, copying a target paper or apply-X-to-Y-only claim. "
            "Describe competing predictions over the SAME observable and a minimal distinguishing intervention. "
            "REFERENCE INTEGRITY: the focal hypothesis ID is exactly 'self'. Each competing_explanations.id "
            "must be unique and must not be 'self'. hypotheses_compared must contain 'self' and the exact "
            "competitor IDs; expected_outcomes keys must exactly equal hypotheses_compared. Never replace "
            "an ID with a descriptive name or leave the example 'alternative' after renaming its competitor. "
            "Directions must be increase/decrease/unchanged. Priority factors are proposal estimates, not evidence.")
        transform = [{"source_state": t.source_state.to_dict(), "operator": t.operator,
                      "changed_component": t.changed_component, "applicability_conditions": t.applicability_conditions,
                      "known_failure_conditions": t.known_failure_conditions} for t in chosen]
        knowledge = (json.dumps({"anchor": anchor, "parent_hypothesis": parent,
            "actual_parent_evidence": evidence, "execution_failures_not_evidence": failures,
            "transferable_operators": transform}, ensure_ascii=False),)
        topic = str(state.get("goal") or "")
        common = dict(seed_label=topic, near_labels=(topic,), topic=topic,
                      context=tuple(f"{w.cite_id()}: {w.title}. {w.abstract}" for w in works),
                      knowledge=knowledge, program=(instructions, f"Research transformation: {operator}"),
                      research_mode=True)
        try:
            world = None
            if kwargs.get("world") is not None or kwargs.get("workspace") is not None or self.workspace is not None:
                try:
                    world = self._world(state, kwargs)
                except FileNotFoundError:
                    if kwargs.get("world") is not None:
                        raise
                    # An intent without a materialized freeze remains speculative.
            if world is not None:
                from ..farcompile import menu_from_card
                from ..dynworld import levers_of, observables_of
                menu = menu_from_card(None, world)
                # These are existing runtime registrations, not invented scout
                # responses, movement estimates, or evidence of effectiveness.
                common["world_menu"] = replace(menu, levers=levers_of(world), observables=observables_of(world))
            if lane == "assumption_reframe":
                card = generate_reframe(self.client, **common, operator=operator)
            else:
                labels = (operator.replace("_", " "),) if chosen else ("unconstrained mechanism exploration",)
                card = generate_card(self.client, **common, far_labels=labels, operator=operator,
                                     alienness=0.0, label_to_node={})
            if world is not None and card.claim_spec:
                card = replace(card, claim_spec={**card.claim_spec, "world_digest": world.digest})
            metadata = dict(card.research_metadata or {})
            required = ("assumptions", "competing_explanations", "closest_prior_work", "inherited_components",
                        "genuinely_changed_components", "claim_delta", "mechanism_delta", "why_not_reconstruction",
                        "experiment_proposal")
            if any(not metadata.get(k) for k in required) or card.falsifier == "claim_contains_a_falsifiable_assertion":
                raise ValueError("structured_hypothesis_contract_missing")
            if not set(metadata["closest_prior_work"]).issubset({w.cite_id() for w in works}):
                raise ValueError("closest_prior_work_must_be_verified_literature")
            paper_targets = [{"target_state": p} for r in surveys for p in r.get("paper_states", [])]
            paper_targets += [{"target_state": w.to_dict()} for w in works]
            rejected = copy_check(card.to_dict(), transitions + paper_targets)
            if rejected:
                return CandidateResult(status="hypothesis_rejected", events=[make_event("EvidenceRejected", {
                    "theory_id": card.card_id, "epistemic": "GENERATED", "reasons": list(rejected),
                    "research_lane": lane, "question_id": action.target}, action_type=THEORIZE)],
                    extras={"reason": "; ".join(rejected)})
            alternatives = metadata["competing_explanations"]
            if not isinstance(alternatives, list) or any(not isinstance(a, dict) or not a.get("id") or not a.get("claim") for a in alternatives):
                raise ValueError("competing_explanations_require_structured_ids_and_claims")
            identifiers = [a["id"] for a in alternatives]
            proposal = dict(metadata["experiment_proposal"])
            compared = proposal.get("hypotheses_compared", [])
            outcomes = proposal.get("expected_outcomes", {})
            if (any(not isinstance(i, str) or i == "self" for i in identifiers)
                    or len(set(identifiers)) != len(identifiers)
                    or not isinstance(compared, list) or not isinstance(outcomes, dict)
                    or any(not isinstance(i, str) for i in compared)
                    or len(set(compared)) != len(compared)
                    or set(compared) != {"self", *identifiers} or set(outcomes) != set(compared)):
                raise ValueError("hypothesis_reference_contract: use unique competitor IDs and 'self'; "
                                 "hypotheses_compared and expected_outcomes must name exactly those IDs")
            id_map = {"self": card.card_id, **{a["id"]: f"{card.card_id}:alternative:{i}" for i, a in enumerate(alternatives)}}
            proposal["hypotheses_compared"] = [id_map[h] for h in proposal["hypotheses_compared"]]
            proposal["expected_outcomes"] = {id_map[h]: p for h, p in proposal["expected_outcomes"].items()}
            theory = {**metadata, "id": card.card_id, "question_id": action.target,
                "claim": card.claim, "text": card.claim, "mechanism": card.mechanism,
                "falsifier": card.falsifier, "discriminating_prediction": card.prediction,
                "predictions": [card.prediction], "card": card.to_dict(), "epistemic": "GENERATED",
                "research_lane": lane, "requested_lane": action.extra["research_lane"],
                "operator_reused": operator, "source_trajectories_used": [
                    {"source": t.source_cite_id, "target": t.target_cite_id, "operator": t.operator} for t in chosen],
                "hop": hop, "parent_theory_id": parent_id,
                "transition_kind": action.extra.get("transition_kind", "first_hop"),
                "failure_pivots": int(action.extra.get("failure_pivots") or 0),
                "evidence_considered": [r["evidence_id"] for r in evidence],
                "evidence_context_digest": sha256_text(json.dumps(evidence, sort_keys=True)),
                "competing": [id_map[a["id"]] for a in alternatives], "experiment_proposal": proposal,
                "read_first": [w.cite_id() for w in works]}
            events = [make_event("TheoryCreated", theory, action_type=THEORIZE)]
            events += [make_event("TheoryCreated", {"id": id_map[a["id"]], "question_id": action.target,
                "claim": a["claim"], "text": a["claim"], "role": "competing_explanation",
                "competing": [card.card_id], "epistemic": "GENERATED"}, action_type=THEORIZE) for a in alternatives]
            from .reducer import apply_events
            from .state import ScientificState
            from .scheduler import hypothesis_resolution_priority
            preview = apply_events(ScientificState.from_dict(state), events)
            validity = hypothesis_resolution_priority(proposal, preview)
            if not validity["valid"]:
                raise ValueError(validity["reason"])
            return CandidateResult(status="theorized", events=events, extras={"theory": theory})
        except (GenerationRefused, LLMUnavailable, ValueError, KeyError, TypeError, OSError) as exc:
            return self._failure(action, exc, research_lane=lane)

    def probe(self, action, state, **kwargs):
        state = readonly_view(state)
        try:
            workspace = self._workspace(kwargs)
            world = self._world(state, kwargs)
            theory = self._theory(action, state)
            card = _card(action.extra.get("card") or theory.get("card"))
            if theory.get("card"):
                canonical = _card(theory["card"])
                if card.to_dict() != canonical.to_dict() or card.card_id != theory.get("id"):
                    raise ValueError("probe_card_does_not_match_canonical_theory")
            if theory.get("research_lane") and not any(t.get("id") == theory.get("id") for t in state.get("theories", [])):
                raise ValueError("probe_requires_canonical_theory")
            if str(card.idea_kind or "probe").lower() != "probe":
                raise ValueError("requires_probe_compilation_or_world_acquisition: "
                                 "a question/acquire/theory card is not an executable intervention")
            if self.client is None:
                raise ValueError("research_model_not_connected")
        except (OSError, ValueError, TypeError) as exc:
            return self._failure(action, exc, status="probe_execution_blocked")
        topic = str(state.get("goal") or card.claim)
        works = self._works(state, str(theory.get("question_id") or ""))
        if theory.get("research_lane") and not any(a.get("kind") == "literature_survey" and
                a.get("theory_id") == theory.get("id") and a.get("works") for a in state.get("artifacts", [])):
            return self._failure(action, "hypothesis_specific_literature_check_required", status="probe_execution_blocked")
        prior = strongest_prior(card.claim, works) if works else None
        if prior and prior.kills:
            return self._failure(action, "claim_already_stated_in_verified_literature",
                                 status="probe_execution_blocked", prior=prior.to_dict())
        try:
            from ..claimspec import in_world_testable, scientific_for
            if not in_world_testable(card.claim_spec):
                return self._failure(action, "claim_not_testable_on_bound_world",
                                     status="probe_execution_blocked")
            claim_checked = False
            if card.claim_spec:
                if ((card.claim_spec.get("world_id") and card.claim_spec["world_id"] != world.id)
                        or (card.claim_spec.get("world_digest") and card.claim_spec["world_digest"] != world.digest)):
                    raise ValueError("claim_world_identity_mismatch")
                from ..claimspec import WorldObjectRegistry, claim_spec_from_payload, typecheck_claim
                from ..dynworld import levers_of, observables_of
                registered = claim_spec_from_payload(card.claim_spec)
                checked = typecheck_claim(registered, WorldObjectRegistry.from_names(
                    (*levers_of(world), *observables_of(world))))
                claim_checked = bool(checked.ok and registered.claim_id == card.card_id
                                     and registered.statement == card.claim
                                     and registered.handle.id and registered.measurement.dv)
                if not claim_checked:
                    return self._failure(action, "claim_object_validation_failed",
                        status="probe_execution_blocked", claim_reasons=list(checked.reasons))
            scientific = scientific_for(card=card, world=world, workspace=workspace)
            diagnosis = write_diagnosis(self.client, card, topic, world=world, works=works,
                                       program=json.dumps({"experiment_proposal": theory.get("experiment_proposal"),
                                                           "competing_theories": theory.get("competing"),
                                                           "evidence_considered": theory.get("evidence_considered")}),
                                       scientific=scientific,
                                       prior_probe=action.extra.get("prior_probe"),
                                       failed_experiments=[str(row.get("experiment") or "")
                                                           for row in state.get("failed_designs", [])])
            diagnosis = replace(diagnosis, compute_tier=floor_tier(diagnosis.compute_tier, world.schema))
            if claim_checked and (diagnosis.handle_held or diagnosis.world_lever != registered.handle.id
                                  or diagnosis.world_observable != registered.measurement.dv):
                raise ValueError("diagnosis_differs_from_registered_claim")
            if diagnosis.world_lever == "none":
                return self._failure(action, "no_handle", status="probe_execution_blocked")
            spec = write_probe(self.client, card, diagnosis, topic, world=world,
                               scientific=scientific,
                               registered_measure=registered.measurement.dv if claim_checked else "",
                               prior_failure=action.extra.get("prior_failure"))
            if claim_checked and spec.measure.strip() != registered.measurement.dv:
                raise ValueError("probe_measure_differs_from_registered_claim")
            if not reads_world_data(spec.source, world):
                return self._failure(action, "probe_does_not_read_world", status="probe_policy_refused")
        except GenerationRefused as exc:
            return self._failure(action, exc.record.unlock_condition, status="probe_policy_refused",
                                 missing_capability=exc.record.missing_capability)
        except (LLMUnavailable, OSError, ValueError) as exc:
            return self._failure(action, exc, status="probe_execution_blocked")
        return self._execute_registered(action, state, card, diagnosis, spec, theory, world,
                                        works, topic, workspace, claim_checked=claim_checked)

    def _execute_registered(self, action, state, card, diagnosis, spec, theory, world,
                            works, topic, workspace, *, claim_checked):
        """One immutable registered execution, shared by first and heldout probes."""
        experiment_digest = sha256_text(spec.source)
        eid = evidence_id(claim_id=card.card_id, experiment_digest=experiment_digest,
                          world_digest=world.digest, data_digest=world.digest)
        registration_workspace = workspace / "research_workers" / eid
        folder = registration_workspace / "candidates" / card.card_id
        if folder.exists():
            return self._failure(action, "registration_already_exists_use_verify",
                                 status="probe_execution_blocked", protocol_dir=str(folder))
        registration = {"kind": "WORLD", "source": spec.source, "measure": spec.measure,
                        "experiment_digest": experiment_digest, "world_digest": world.digest,
                        "data_digest": world.digest, "evidence_id": eid}
        brief = compile_brief(card, works, diagnosis=diagnosis, topic=topic)
        protocol = render_protocol(topic, card, brief, works, probe=registration, diagnosis=diagnosis, world=world)
        proposal = json.loads(json.dumps(theory.get("experiment_proposal") or {}))
        if not isinstance(proposal, dict):
            proposal = {}
        # These scientific decisions are committed before either arm runs.
        protocol.payload["experiment_proposal"] = proposal
        protocol.payload["diagnosis"] = diagnosis.to_dict()
        protocol.payload["card"] = card.to_dict()
        compared = proposal.get("hypotheses_compared") or []
        predictions = proposal.get("expected_outcomes") or {}
        if not isinstance(compared, list) or not all(isinstance(hid, str) for hid in compared):
            compared = []
        if not isinstance(predictions, Mapping):
            predictions = {}
        live_ids = {row.get("id") for row in state.get("theories", [])}
        competitors = [hid for hid in compared if hid != card.card_id]
        directions = {"increase": "treatment_higher", "decrease": "treatment_lower",
                      "treatment_higher": "treatment_higher", "treatment_lower": "treatment_lower"}
        own = predictions.get(card.card_id) or {}
        other = predictions.get(competitors[0]) or {} if len(competitors) == 1 else {}
        discriminator = bool(len(compared) == 2 and len(set(compared)) == 2
            and card.card_id in compared and set(compared) <= live_ids
            and isinstance(own, Mapping) and isinstance(other, Mapping)
            and own.get("observable") == other.get("observable") == diagnosis.world_observable
            and bool(diagnosis.world_observable)
            and spec.measure.strip() == diagnosis.world_observable
            and directions.get(own.get("direction")) == diagnosis.expected_direction
            and directions.get(other.get("direction")) == diagnosis.alternative_direction
            and diagnosis.expected_direction != diagnosis.alternative_direction
            and bool(proposal.get("minimal_discriminating_intervention"))
            and bool(proposal.get("nuisance_factors")))
        protocol.payload["discriminator_registered"] = discriminator
        # The registration is immutable: measured values live in probe.json.
        for key in ("treatment", "control", "verdict", "status"):
            (protocol.payload.get("cheap_probe") or {}).pop(key, None)
        try:
            folder = write_idea(registration_workspace, card_id=card.card_id, brief=brief.to_dict(),
                                works=[work.to_dict() for work in works], probe=registration,
                                protocol=protocol.payload, protocol_md=protocol.markdown, readme_md=protocol.readme)
            _json(folder / "card.json", card.to_dict())
            protocol_digest = sha256_text((folder / "protocol.json").read_text())
            tier = resolve_tier(diagnosis.compute_tier)
            dest = folder / "probe" / "execution"
            bind_world(dest, world)
            result = run_probe(spec, dest, timeout_seconds=probe_timeout(tier), tier=tier)
            if (digest_files(dest / "data", world.files) != world.digest
                    or sha256_text((folder / "experiment.py").read_text()) != experiment_digest
                    or sha256_text((folder / "protocol.json").read_text()) != protocol_digest):
                raise ValueError("registered source, protocol, or consumed world digest changed during execution")
        except (OSError, ValueError) as exc:
            return self._failure(action, exc, status="probe_execution_blocked", failed=True,
                                 protocol_dir=str(folder))
        if result.status != "ran":
            _json(folder / "execution_failure.json", result.to_dict())
            return self._failure(action, result.error or result.status, status="probe_execution_blocked",
                                 failed=True, executor_status=result.status, diagnosis=diagnosis.to_dict(),
                                 protocol_dir=str(folder), experiment_digest=experiment_digest)
        measured = judge_probe(diagnosis, result.treatment, result.control)
        gate = self._gates(card, diagnosis, spec, result, world, folder, tier)
        if gate.get("validation_gaps") or gate.get("dv_blind") or gate.get("world_consumed") is False:
            measured["verdict"] = "uninformative"
        evidence = {**result.to_dict(), **measured, **gate, **registration,
                    "executor_status": result.status, "role": "ProbeEvidence", "epistemic": "WORLD",
                    "execution_status": result.status, "claim_consistent": False,
                    "literature_checked": bool(works), "mechanism_validated": False,
                    "causal_scope": "unestablished", "replication_required": True,
                    "probe_kind": "WORLD", "attested": True, "card_id": card.card_id,
                    "theory_id": theory.get("id") or card.card_id,
                    "question_id": theory.get("question_id") or action.frontier_target_id,
                    "world_id": world.id, "protocol_dir": str(folder), "protocol_digest": protocol_digest,
                    "experiment_proposal": proposal, "discriminator_registered": discriminator,
                    "diagnosis": diagnosis.to_dict(), "card": card.to_dict(),
                    "claim": card.claim, "harness_version": str(state.get("harness_version") or "")}
        if card.claim_spec:
            try:
                from ..claimspec import claim_spec_from_payload, project_evidence_to_claim
                projection = project_evidence_to_claim(claim_spec_from_payload(card.claim_spec),
                    arithmetic_verdict=measured["verdict"], observed_effect={key: measured.get(key)
                    for key in ("treatment", "control", "separation")},
                    metric=diagnosis.world_observable or card.world_observable)
                evidence["scoped_verdict"] = projection.record.verdict.value
                evidence["supports_scope"] = list(projection.record.supports_scope)
                evidence["prohibited_inferences"] = list(projection.record.prohibited_inferences)
                evidence["claim_consistent"] = bool(
                    claim_checked
                    and not gate.get("validation_gaps") and not gate.get("dv_blind")
                    and gate.get("world_consumed") is not False)
            except ValueError as exc:
                evidence.update(verdict="uninformative", cannot_corroborate=True, claim_gate_error=str(exc))
        else:
            evidence["cannot_corroborate"] = True
            evidence["scope_gap"] = "claim_spec_not_registered"
        if (discriminator and evidence.get("claim_consistent") and evidence.get("verdict") == "supports"
                and not evidence.get("ablation_required") and not evidence.get("cannot_corroborate")):
            evidence["mechanism_validated"] = "registered_discriminator_within_tested_conditions"
            evidence["mechanism_identified"] = True
            evidence["causal_scope"] = "within_tested_conditions"
        evidence["outcome"] = evidence["verdict"]
        _json(folder / "probe.json", evidence)
        _json(folder / "metrics.json", result.metrics)
        status = {"supports": "probe_positive", "weakens": "probe_scientifically_negative",
                  "uninformative": "probe_inconclusive"}[evidence["verdict"]]
        return CandidateResult(status=status, evidence=evidence, evidence_id=eid, world_id=world.id,
                               extras={"executor_status": result.status, "protocol_dir": str(folder)})

    def _gates(self, card, diagnosis, spec, result, world, folder, tier):
        """Fixed validation work, never a redesign loop or sign-seeking retry."""
        from ..dvsanity import dv_blind, materialize_oracle_placebo, oracle_dependent, oracle_field
        from ..dynworld import arm_separation, has_dynamics, levers_of, materialize_placebo, world_consumed
        gate = {"validation_gaps": [], "ablation_required": ablation_required(
            diagnosis.alternative, diagnosis.experiment, levers=levers_of(world), world_lever=diagnosis.world_lever)}
        if gate["ablation_required"]:
            gate.update(ablation_required=True, mechanism_identified=False, cannot_corroborate=True)
            gate["missing_capability"] = "registered_competing_mechanism_ablation"
        try:
            if oracle_field(world.schema) and oracle_dependent(card.claim, diagnosis.experiment,
                                                              diagnosis.treatment_arm, diagnosis.control_arm):
                fixture = materialize_oracle_placebo(world, folder / "checks" / "oracle")
                if fixture is None:
                    gate["validation_gaps"].append("oracle_sanity_not_identifiable")
                else:
                    dest = folder / "checks" / "oracle-run"
                    bind_world(dest, fixture)
                    check = run_probe(spec, dest, timeout_seconds=probe_timeout(tier), tier=tier)
                    if check.status != "ran":
                        gate["validation_gaps"].append("oracle_sanity_execution_failed")
                    else:
                        gate["dv_blind"] = dv_blind((result.treatment, result.control),
                                                   (check.treatment, check.control))
            if judge_probe(diagnosis, result.treatment, result.control)["verdict"] == "supports" and has_dynamics(world):
                separations = []
                for seed in range(5):
                    fixture = materialize_placebo(world, folder / "checks" / f"placebo-{seed}", seed=seed)
                    dest = folder / "checks" / f"placebo-run-{seed}"
                    bind_world(dest, fixture)
                    check = run_probe(spec, dest, timeout_seconds=probe_timeout(tier), tier=tier)
                    if check.status != "ran":
                        raise ValueError(f"world consumption check {check.status}")
                    separations.append(arm_separation(check.treatment, check.control))
                gate["world_consumed"] = world_consumed(arm_separation(result.treatment, result.control),
                                                         sum(separations) / len(separations), diagnosis.margin)
                gate["placebo_separations"] = separations
                chain.append_event(folder / "chain.jsonl", chain.EXECUTE_PLACEBO,
                    {"card_id": card.card_id, "consumed": gate["world_consumed"],
                     "experiment_digest": sha256_text(spec.source), "world_digest": world.digest})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            gate["validation_gaps"].append(str(exc))
        if gate["validation_gaps"] or gate.get("dv_blind") or gate.get("world_consumed") is False:
            gate["cannot_corroborate"] = True
        if result.treatment == 0 and result.control == 0:
            gate.update(object_absent=True, cannot_corroborate=True)
        return gate

    def verify(self, action, state, **kwargs):
        state = readonly_view(state)
        target = next((row for row in state.get("evidence_records", [])
                       if row.get("evidence_id") == action.target), None)
        if target is None:
            return self._failure(action, "no_target_probe_evidence")
        try:
            workspace = self._workspace(kwargs)
            original_world_id = str(target.get("world_id") or "")
            if not original_world_id or Path(original_world_id).name != original_world_id or original_world_id in {".", ".."}:
                raise ValueError("invalid original evidence world id")
            world = load_fixture(workspace / "worlds" / original_world_id)
            if not is_attested(world):
                raise ValueError("original evidence world cannot be attested")
            if target.get("world_id") != world.id or target.get("world_digest") != world.digest:
                raise ValueError("target evidence belongs to another world; transfer is unverified")
            folder = Path(target["protocol_dir"])
            if not folder.resolve().is_relative_to(self._workspace(kwargs).resolve()):
                raise ValueError("protocol is outside the research workspace")
            protocol_text = (folder / "protocol.json").read_text()
            if sha256_text(protocol_text) != target.get("protocol_digest"):
                raise ValueError("registered protocol was changed")
            protocol = json.loads(protocol_text)
            for key in ("experiment_digest", "world_digest", "data_digest", "evidence_id"):
                if not target.get(key) or protocol.get(key) != target[key]:
                    raise ValueError(f"registered {key} differs from the target evidence")
            source = (folder / "experiment.py").read_text()
            if sha256_text(source) != target["experiment_digest"]:
                raise ValueError("registered experiment source was changed")
            replication_id = str(action.extra.get("replication_world_id") or "")
            if "replication_world_id" not in action.extra and target.get("reproduction_ok"):
                replication_id = next((str(row.get("id") or row.get("world_id") or "")
                    for row in state.get("world_versions", []) if row.get("role") == "heldout"
                    and not row.get("deprecated") and (row.get("id") or row.get("world_id")) != world.id), "")
            if replication_id:
                if Path(replication_id).name != replication_id or replication_id in {".", ".."}:
                    raise ValueError("invalid heldout world id")
                heldout = load_fixture(workspace / "worlds" / replication_id)
                if not is_attested(heldout):
                    raise ValueError("heldout world must already be attested")
                if (heldout.id == world.id or heldout.digest == world.digest
                        or (heldout.source_digest and heldout.source_digest == world.source_digest
                            and heldout.slice_rule == world.slice_rule)):
                    raise ValueError("heldout replication requires different attested scientific data")
                if heldout.schema != world.schema:
                    raise ValueError("heldout world has a different scientific object schema")
                if world.domains and heldout.domains and not set(world.domains).intersection(heldout.domains):
                    raise ValueError("heldout world belongs to a different scientific domain")
                if protocol.get("card") != target.get("card") or protocol.get("diagnosis") != target.get("diagnosis"):
                    raise ValueError("complete claim and diagnosis must match the immutable registration")
                card = _card(protocol["card"])
                diagnosis = diagnosis_from_payload(card.card_id, protocol["diagnosis"])
                from ..claimspec import WorldObjectRegistry, claim_spec_from_payload, typecheck_claim
                from ..dynworld import levers_of, observables_of
                registered = claim_spec_from_payload(card.claim_spec)
                checked = typecheck_claim(registered, WorldObjectRegistry.from_names(
                    (*levers_of(heldout), *observables_of(heldout))))
                if (not checked.ok or registered.claim_id != card.card_id or registered.statement != card.claim
                        or registered.handle.id != diagnosis.world_lever
                        or registered.measurement.dv != diagnosis.world_observable):
                    raise ValueError("registered claim is not admissible on the heldout scientific object")
                if floor_tier(diagnosis.compute_tier, heldout.schema) != diagnosis.compute_tier:
                    raise ValueError("heldout world exceeds the pre-registered compute tier")
                from ..probeexp import spec_from_payload
                spec = spec_from_payload({"measure": str((protocol.get("cheap_probe") or {}).get("measure") or ""),
                    "source": source}, tier=resolve_tier(diagnosis.compute_tier), claim=card.claim,
                    mechanism=card.mechanism, topic=str(state.get("goal") or ""), schema=heldout.schema)
                if not reads_world_data(source, heldout):
                    raise ValueError("registered script does not read heldout world payload")
                theory = {"id": target.get("theory_id") or card.card_id,
                          "question_id": target.get("question_id"),
                          "experiment_proposal": protocol.get("experiment_proposal") or {}}
                works = [FreshWork.from_dict(row) for row in json.loads((folder / "papers.json").read_text())]
                execution = self._execute_registered(action, state, card, diagnosis, spec, theory, heldout,
                    works, str(protocol.get("topic") or state.get("goal") or ""), workspace, claim_checked=True)
                if execution.evidence is None:
                    return self._failure(action, execution.extras.get("reason") or execution.status,
                        transfer_ok=False, replication_world_id=heldout.id)
                replicated = dict(execution.evidence)
                new_folder = Path(replicated["protocol_dir"])
                host = execute_protocol(new_folder, catalog_root=kwargs.get("catalog_root") or workspace,
                                        prefer_parent=False)
                identities_match = all(host.get(key) == replicated.get(key) for key in
                    ("experiment_digest", "world_digest", "data_digest", "evidence_id"))
                reproduced = bool(host.get("ok") and identities_match and
                    host.get("treatment") == replicated.get("treatment") and
                    host.get("control") == replicated.get("control"))
                supports_transfer = bool(reproduced and target.get("outcome") == "supports"
                    and replicated.get("outcome") == "supports" and replicated.get("claim_consistent")
                    and not replicated.get("cannot_corroborate"))
                independent = bool(world.source_digest and heldout.source_digest
                                   and world.source_digest != heldout.source_digest)
                replicated.update(replication_of=target["evidence_id"], verify_count=1,
                    verify_kind="VERIFY_REPLICATION", verified_by="hostexp.execute_protocol",
                    reproduction_ok=reproduced, replay_ok=reproduced, identity_ok=identities_match,
                    executable_ok=bool(host.get("ok")), transfer_ok=supports_transfer,
                    cross_world_transfer=supports_transfer, independent_replication=independent and reproduced,
                    replication="E3_independent" if independent else "E2_scale_replication",
                    world_compatible=True, host_record=host)
                _json(new_folder / "probe.json", replicated)
                return CandidateResult(status="replicated" if supports_transfer else "replication_inconclusive",
                    events=[make_event("EvidenceAdded", replicated, action_type=VERIFY,
                                       frontier_target_id=action.frontier_target_id)],
                    evidence_id=replicated["evidence_id"], world_id=heldout.id, unlocked=supports_transfer,
                    extras={"replication_of": target["evidence_id"], "transfer_ok": supports_transfer,
                            "reproduction_ok": reproduced, "protocol_dir": str(new_folder)})
            record = execute_protocol(folder, catalog_root=kwargs.get("catalog_root") or self._workspace(kwargs),
                                      prefer_parent=False)
            same = all(record.get(key) == target.get(key) for key in
                       ("experiment_digest", "world_digest", "data_digest", "evidence_id"))
            replay_ok = bool(record.get("ok") and same and
                             record.get("treatment") == target.get("treatment") and
                             record.get("control") == target.get("control"))
            updated = {**target, "verify_count": int(target.get("verify_count") or 0) + 1,
                       "verify_kind": str(action.extra.get("verify_kind") or "VERIFY_REPLICATION"),
                       "verified_by": "hostexp.execute_protocol", "world_compatible": True,
                       "replay_ok": replay_ok, "replication": record.get("replication"),
                       "cross_world_transfer": False, "independent_replication": False,
                       "reproduction_ok": replay_ok, "identity_ok": same,
                       "executable_ok": bool(record.get("ok")),
                       "claim_consistent": bool(target.get("claim_consistent")),
                       "transfer_ok": False, "replication_required": True,
                       "host_record": record}
            return CandidateResult(status="verified" if replay_ok else "verify_failed",
                events=[make_event("EvidenceAdded", updated, action_type=VERIFY,
                                   frontier_target_id=action.frontier_target_id)],
                evidence_id=str(target["evidence_id"]), world_id=world.id, unlocked=replay_ok,
                extras={"replay_ok": replay_ok, "same_instance": same, "independent_replication": False,
                        "cross_world_transfer": False, "host_record": record})
        except (ExecuteError, GenerationRefused, OSError, ValueError, KeyError, TypeError) as exc:
            return self._failure(action, exc, same_instance=False, cross_world_transfer=False)

    def synthesize(self, action, state, **kwargs):
        state = readonly_view(state)
        rows = [row for row in state.get("evidence_records", [])
                if row.get("role") == "ProbeEvidence" and row.get("executor_status") == "ran"
                and row.get("attested") and row.get("protocol_dir")]
        if not rows:
            return self._failure(action, "no_executed_research_evidence")
        selected = next((row for row in rows if action.target in
                         {row.get("theory_id"), row.get("evidence_id"), row.get("question_id")}), rows[-1])
        try:
            folder = Path(selected["protocol_dir"])
            if not folder.resolve().is_relative_to(self._workspace(kwargs).resolve()):
                raise ValueError("protocol is outside the research workspace")
            protocol_text = (folder / "protocol.json").read_text()
            if sha256_text(protocol_text) != selected.get("protocol_digest"):
                raise ValueError("registered protocol was changed")
            protocol = json.loads(protocol_text)
            registered_probe = json.loads((folder / "probe.json").read_text())
            for key in ("evidence_id", "experiment_digest", "world_digest", "data_digest", "treatment", "control"):
                if registered_probe.get(key) != selected.get(key):
                    raise ValueError(f"synthesis evidence differs from registered probe: {key}")
            if protocol.get("card") and protocol["card"] != selected.get("card"):
                raise ValueError("synthesis card differs from immutable registration")
            card = _card(protocol.get("card") or selected["card"])
            diagnosis = diagnosis_from_payload(card.card_id, protocol.get("diagnosis") or selected["diagnosis"])
            works = [FreshWork.from_dict(row) for row in json.loads((folder / "papers.json").read_text())]
            brief = compile_brief(card, works, diagnosis=diagnosis, topic=str(state.get("goal") or ""))
            from ..paperplan import gather_facts
            facts = gather_facts(folder.parent.parent, folder, card.card_id,
                                 catalog_root=kwargs.get("catalog_root") or self._workspace(kwargs))
            scope = facts["scientific_scope"]
            from ..writeup import render_note
            # The legacy note's probe paragraph assumes constructed data. Supply
            # its measured paragraph from the same registered facts as the draft.
            note = render_note(str(state.get("goal") or ""), card, brief, works, probe=None)
            measurement = (
                f"Registered comparison on world `{facts['world'].get('id')}` "
                f"(digest `{scope['world_digest']}`), EvidenceID `{scope['evidence_id']}`: "
                f"treatment = {facts['probe'].get('treatment')}, control = {facts['probe'].get('control')}; "
                f"arithmetic verdict = `{scope['arithmetic_verdict']}`. "
                f"Claim strength: `{scope['claim_strength']}`. An arithmetic result does not "
                "establish a mechanism or generalization beyond the registered scope."
            )
            markdown = note.markdown.replace("No probe was run.", measurement, 1)
            markdown = markdown.replace("## Claim\n", "## Claim (GENERATED hypothesis)\n", 1)
            path = folder / "research_note.md"
            path.write_text(markdown + "\n\nScientific scope: " +
                            json.dumps(scope, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            _json(folder / "PAPER_FACTS.json", facts)
            artifact = {"id": f"synthesis-{selected['evidence_id']}", "kind": "research_synthesis",
                        "path": str(path), "evidence_id": selected["evidence_id"],
                        "question_id": selected.get("question_id"), "verdict": selected.get("verdict"),
                        "cannot_corroborate": scope["cannot_corroborate"],
                        "read_first": list(brief.read_first), "protocol_dir": str(folder)}
            artifact["facts_path"] = str(folder / "PAPER_FACTS.json")
            if action.extra.get("paper_draft"):
                if self.client is None:
                    return self._failure(action, "paper_draft_requires_research_model")
                from ..paperplan import compile_paper
                artifact["paper"] = compile_paper(folder.parent.parent, folder, card.card_id,
                    client=self.client, rounds=0, lang=str(action.extra.get("lang") or "en"),
                    catalog_root=kwargs.get("catalog_root") or self._workspace(kwargs))
            return CandidateResult(status="synthesized", events=[make_event("ArtifactAdded", artifact,
                action_type=SYNTHESIZE, frontier_target_id=action.frontier_target_id)], extras=artifact)
        except (OSError, ValueError, KeyError, TypeError, LLMUnavailable) as exc:
            return self._failure(action, exc)
